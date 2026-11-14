#!/usr/bin/env python
"""Phase 2, step 2.3b - the network-only detector arm (Arm N), calibrated on the benign capture.

Roles are assigned per connection from behaviour, not from addresses (on this testbed
every endpoint is 127.0.0.1, because Docker Desktop's port proxy relays OpenPLC):
    plc        Modbus client that writes coils (FC15)                 ~9.67 cycles/s
    pub        Modbus client that reads coils (FC1) but never writes  1 Hz
    mqtt_pub   MQTT client that publishes;   mqtt_sub   MQTT client that subscribes

Zero-cost checks (silent on a genuine stack during steady state, so no budget):
    SYN        a new TCP connection to the twin's Modbus port (5020/502). New or closed MQTT
               connections on 1883 are not checked: operators attach read-only subscribers freely
    END        a FIN or RST on one of the long-lived Modbus connections
    WL         a Modbus request outside its role's whitelist (function code, start, qty,
               unit id), learned from the calibration half
    EXC        a Modbus exception response
    UNM        a Modbus response with no matching request, or a request never answered
Statistics, per 60 s block, for roles plc and pub; four are budgeted (BUDGETED below):
    lat        median FC4 response latency            (a different responder)
    gap        mean FC4 inter-request interval         (a different or delayed client)
    jit        spread of FC4 intervals, p90 - p10      (a proxy or relay in the path)
Each is scored two-sided as |z| = |x - median| / (1.4826 MAD) on the calibration half.

Unlike the physics arms there is no first-principles null for traffic, so:
  * calibration = first half of the steady state, held-out = second half (chronological)
  * the threshold for each component is extrapolated from an exponential tail fitted to
    the top 10 % of its calibration |z| (peaks over threshold), at a per-block
    exceedance probability of share * far_per_day / 1440
  * the held-out half validates the whole arm against the budget.
Budget: --far-per-day episodes per 24 h, split equally over the 4 budgeted components
(plc.lat, plc.gap, plc.jit, pub.lat). Episodes are runs of consecutive 60 s blocks with an
alarm, as for the physics arms.

Cross-session finding (attribution control, 14 Nov 2026; scripts/phase2_net_arm_blocks.py):
applied unchanged to later sessions, the response-latency components (plc.lat, pub.lat) stayed
inside their thresholds on benign captures and alarmed on every block with the relay in the
path, while the request-clock components (plc.gap, plc.jit) drifted and alarmed on benign
captures too. A saved calibration is therefore valid within its session; cross-session use of
plc.gap / plc.jit needs recalibration.

    python scripts/phase2_net_arm.py data/raw/pcap/benign_20260902T005748Z.pcap --calibrate
    python scripts/phase2_net_arm.py data/raw/pcap/<attack run>.pcap --score     # step 2.4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_pcap_inventory import link_payload, mqtt_packets, parse_ip, read_pcap  # noqa: E402

BLOCK = 60
MODBUS_PORTS = {502, 5020}
MQTT_PORTS = {1883}
STATS = ("lat", "gap", "jit")                 # computed for both roles; only BUDGETED are scored
ROLES = ("plc", "pub")
# The publisher polls at 1 Hz, so its block-mean interval drifts with host-clock discipline
# (the Phase 1 data card: the publisher sleeps in monotonic time, stamps in wall time, and the
# clock was slewed forward by NTP). That slew is fractional, so on a 1 s gap it swamps a MAD-tight
# band, while on the PLC's 103 ms gap it is invisible. Latency is a difference of two packets ms
# apart, so slew cancels. Budgeted components are therefore: all three PLC statistics, publisher
# latency only. gap and jit come from the fast poller, where the clock is not the signal.
BUDGETED = (("plc", "lat"), ("plc", "gap"), ("plc", "jit"), ("pub", "lat"))
ZERO_COST = ("SYN", "END", "WL", "EXC", "UNM")


# ---------------------------------------------------------------- extraction
def extract(path: Path) -> dict:
    """One pass over a capture: per-connection Modbus requests/latencies, MQTT activity,
    and TCP lifecycle events."""
    gen = read_pcap(path)
    next(gen)
    t0 = None
    conns = defaultdict(lambda: {"fc": Counter(), "req_t": defaultdict(list), "lat": defaultdict(list),
                                 "lat_t": defaultdict(list), "reqs": Counter(), "key_t": {}, "pending": {},
                                 "unmatched_t": [], "exc_t": [], "first": None, "last": None})
    mqtt = defaultdict(Counter)
    tcp_events = []                                   # (ts, kind, conn name, server port)
    n = 0
    for ts, lt, fr in gen:
        n += 1
        if n % 1_000_000 == 0:
            print(f"    {n / 1e6:.0f} M packets ...", flush=True)
        if t0 is None:
            t0 = ts
        lp = link_payload(lt, fr)
        if lp is None:
            continue
        ip = parse_ip(lp[0], lp[1])
        if ip is None or ip[2] != 6 or len(ip[3]) < 20:
            continue
        src, dst, _, l4, _ = ip
        sport, dport, _seq, _ack, offr, flags = struct.unpack("!HHIIBB", l4[:14])
        payload = l4[(offr >> 4) * 4:]
        server_side = dport if (dport in MODBUS_PORTS or dport in MQTT_PORTS) else sport
        client = f"{src}:{sport}" if server_side == dport else f"{dst}:{dport}"
        cname = f"{client}->{server_side}"
        if flags & 0x02 and not flags & 0x10 and dport in (MODBUS_PORTS | MQTT_PORTS):
            tcp_events.append((ts, "SYN", cname, dport))
        if flags & 0x05 and server_side in (MODBUS_PORTS | MQTT_PORTS):
            tcp_events.append((ts, "RST" if flags & 0x04 else "FIN", cname, server_side))
        if not payload:
            continue
        if server_side in MODBUS_PORTS:
            c = conns[cname]
            c["first"] = ts if c["first"] is None else c["first"]
            c["last"] = ts
            is_req = dport in MODBUS_PORTS
            off = 0
            while off + 8 <= len(payload):
                tid, pid, length, unit = struct.unpack("!HHHB", payload[off:off + 7])
                if pid != 0 or length < 2 or off + 6 + length > len(payload):
                    break
                fc = payload[off + 7]
                if is_req:
                    start = qty = None
                    if fc in (1, 2, 3, 4, 15, 16) and off + 12 <= len(payload):
                        start, qty = struct.unpack("!HH", payload[off + 8:off + 12])
                    elif fc in (5, 6) and off + 10 <= len(payload):
                        start, qty = struct.unpack("!H", payload[off + 8:off + 10])[0], 1
                    key = (fc, start, qty, unit)
                    c["reqs"][key] += 1
                    c["key_t"].setdefault(key, ts)              # first time this request shape was seen
                    c["req_t"][fc].append(ts)
                    c["pending"][tid] = (ts, fc)
                    if len(c["pending"]) > 50_000:
                        c["pending"] = {k: v for k, v in c["pending"].items() if ts - v[0] < 10}
                else:
                    if fc >= 0x80:
                        c["exc_t"].append(ts)
                    req = c["pending"].pop(tid, None)
                    if req is None:
                        c["unmatched_t"].append(ts)
                    else:
                        c["lat"][req[1]].append(ts - req[0])
                        c["lat_t"][req[1]].append(ts)
                c["fc"][fc] += 1
                off += 6 + length
        elif server_side in MQTT_PORTS:
            for mtype in mqtt_packets(payload):
                mqtt[cname][(("c2s" if dport in MQTT_PORTS else "s2c"), mtype)] += 1

    # roles from behaviour
    roles = {}
    for name, c in conns.items():
        fcs = {k[0] for k in c["reqs"]}
        if 15 in fcs:
            roles.setdefault("plc", []).append(name)
        elif 1 in fcs:
            roles.setdefault("pub", []).append(name)
        else:
            roles.setdefault("other_modbus", []).append(name)
    for name, m in mqtt.items():
        if m[("c2s", 3)]:
            roles.setdefault("mqtt_pub", []).append(name)
        elif m[("c2s", 8)]:
            roles.setdefault("mqtt_sub", []).append(name)
    # requests never answered (still pending at the end, older than 10 s)
    for c in conns.values():
        if c["last"] is not None:
            c["unmatched_t"] += [t for (t, _f) in c["pending"].values() if c["last"] - t > 10]
    return {"t0": t0, "packets": n, "conns": conns, "mqtt": mqtt, "roles": roles, "tcp_events": tcp_events}


def main_conn(ex: dict, role: str):
    """The long-lived connection for a role (most Modbus requests, or most MQTT packets)."""
    names = ex["roles"].get(role, [])
    if not names:
        return None
    if role.startswith("mqtt"):
        return max(names, key=lambda nm: sum(ex["mqtt"][nm].values()))
    return max(names, key=lambda nm: sum(ex["conns"][nm]["reqs"].values()))


def steady_window(ex: dict) -> tuple[float, float]:
    """From 60 s after both Modbus roles are up to 60 s before the first of them stops."""
    starts, ends = [], []
    for role in ROLES:
        nm = main_conn(ex, role)
        if nm is None:
            raise SystemExit(f"no '{role}' connection found in the capture")
        c = ex["conns"][nm]
        starts.append(c["first"])
        ends.append(c["last"])
    return max(starts) + 60.0, min(ends) - 60.0


# ---------------------------------------------------------------- block statistics
def block_stats(ex: dict, t_start: float, t_end: float) -> dict:
    """Per-role arrays over 60 s blocks in [t_start, t_end): lat (ms), gap (ms), jit (ms)."""
    nb = int((t_end - t_start) // BLOCK)
    out = {"t_start": t_start, "n_blocks": nb}
    for role in ROLES:
        c = ex["conns"][main_conn(ex, role)]
        rt = np.asarray(c["req_t"][4])
        lt_ = np.asarray(c["lat_t"][4])
        lv = np.asarray(c["lat"][4]) * 1000.0
        gaps = np.diff(rt) * 1000.0
        gt = rt[1:]
        lat = np.full(nb, np.nan)
        gap = np.full(nb, np.nan)
        jit = np.full(nb, np.nan)
        bl = ((lt_ - t_start) // BLOCK).astype(int)
        bg = ((gt - t_start) // BLOCK).astype(int)
        expected = len(rt) / max(rt[-1] - rt[0], 1.0) * BLOCK if len(rt) > 1 else 0
        # group latencies and intervals by block
        order_l = np.argsort(bl, kind="stable")
        bl_s, lv_s = bl[order_l], lv[order_l]
        cuts_l = np.searchsorted(bl_s, np.arange(nb + 1))
        order_g = np.argsort(bg, kind="stable")
        bg_s, gv_s = bg[order_g], gaps[order_g]
        cuts_g = np.searchsorted(bg_s, np.arange(nb + 1))
        for b in range(nb):
            x = lv_s[cuts_l[b]:cuts_l[b + 1]]
            g = gv_s[cuts_g[b]:cuts_g[b + 1]]
            if len(g) >= 0.5 * expected and len(x):
                lat[b] = np.median(x)
                gap[b] = g.mean()
                jit[b] = np.quantile(g, 0.9) - np.quantile(g, 0.1)
        out[role] = {"lat": lat, "gap": gap, "jit": jit, "expected_per_block": expected}
    return out


def zero_cost_events(ex: dict, t_start: float, t_end: float, whitelist: dict | None) -> list:
    """(ts, kind, detail) for every zero-cost check that fires inside [t_start, t_end)."""
    ev = []
    # SYN / END apply to the Modbus path only. On this testbed operators attach and detach
    # read-only MQTT subscribers (mosquitto_sub) freely during normal operation, so a new or
    # closed :1883 connection is benign; a new client on the twin's control port is not.
    long_lived = {main_conn(ex, r) for r in ROLES}
    for ts, kind, cname, port in ex["tcp_events"]:
        if t_start <= ts < t_end and port in MODBUS_PORTS:
            if kind == "SYN":
                ev.append((ts, "SYN", f"{cname} (port {port})"))
            elif cname in long_lived:
                ev.append((ts, "END", f"{kind} on {cname}"))
    role_of = {main_conn(ex, r): r for r in ROLES}
    for name, c in ex["conns"].items():
        role = role_of.get(name, "other_modbus")
        for t in c["exc_t"]:
            if t_start <= t < t_end:
                ev.append((t, "EXC", name))
        for t in c["unmatched_t"]:
            if t_start <= t < t_end:
                ev.append((t, "UNM", name))
        if whitelist is not None:
            allowed = {tuple(x) for x in whitelist.get(role, [])}
            for key, t in c["key_t"].items():
                if key not in allowed and t_start <= t < t_end:
                    ev.append((t, "WL", f"{role} {name} fc{key[0]} start {key[1]} qty {key[2]} unit {key[3]}"))
    return sorted(ev)


def learn_whitelist(ex: dict, t_end: float) -> dict:
    """Request shapes each role used before t_end (the end of the calibration half)."""
    wl = {}
    for role in ROLES:
        c = ex["conns"][main_conn(ex, role)]
        wl[role] = sorted([list(k) for k, t in c["key_t"].items() if t < t_end], key=str)
    return wl


# ---------------------------------------------------------------- thresholds
def fit_component(x: np.ndarray, p_block: float) -> dict:
    x = x[np.isfinite(x)]
    med = float(np.median(x))
    mad = float(1.4826 * np.median(np.abs(x - med)))
    scale = mad if mad > 0 else float(np.std(x)) or 1e-9
    z = np.abs(x - med) / scale
    u = float(np.quantile(z, 0.9))
    y = z[z > u] - u
    sigma = float(y.mean()) if y.size else 1.0
    tau = u + sigma * math.log(0.1 / p_block) if p_block < 0.1 else float(np.quantile(z, 1 - p_block))
    return {"median": med, "scale": scale, "u90": u, "tail_sigma": sigma, "tau": tau,
            "cal_max_z": float(z.max()), "cal_blocks": int(z.size)}


def episodes(alarm_blocks: np.ndarray) -> int:
    a = np.asarray(alarm_blocks, bool)
    return int(a[0]) + int(np.count_nonzero(a[1:] & ~a[:-1])) if a.size else 0


def poisson_ci(n: int) -> tuple[float, float]:
    lo = 0.0 if n == 0 else n * (1 - 1 / (9 * n) - 1.96 / (3 * math.sqrt(n))) ** 3
    m = n + 1
    return lo, m * (1 - 1 / (9 * m) + 1.96 / (3 * math.sqrt(m))) ** 3


def score(bs: dict, fits: dict, zc: list, t_start: float, nb: int):
    """Per-block alarm per component, plus the arm's OR. Returns (dict of bool arrays, arm)."""
    al = {}
    for role, st in BUDGETED:
        f = fits[f"{role}.{st}"]
        x = bs[role][st]
        z = np.abs(x - f["median"]) / f["scale"]
        al[f"{role}.{st}"] = np.nan_to_num(z, nan=0.0) > f["tau"]
    for kind in ZERO_COST:
        a = np.zeros(nb, bool)
        for ts, k, _ in zc:
            if k == kind:
                b = int((ts - t_start) // BLOCK)
                if 0 <= b < nb:
                    a[b] = True
        al[kind] = a
    return al, np.logical_or.reduce(list(al.values()))


# ---------------------------------------------------------------- scoring a new capture
def score_capture(path: Path, cal_path: Path) -> None:
    """Apply a saved Arm N calibration to another capture (e.g. a live attack run, step 2.4)."""
    cal = json.loads(cal_path.read_text())
    iso = lambda t: datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")
    print(f"scoring {path} with {cal_path} (calibrated on {cal['pcap']}) ...", flush=True)
    ex = extract(path)
    print("roles: " + "; ".join(f"{r}: {', '.join(v)}" for r, v in ex["roles"].items()))
    ss0, ss1 = steady_window(ex)
    bs = block_stats(ex, ss0, ss1)
    zc = zero_cost_events(ex, ss0, ss1, cal["whitelist"])
    al, arm = score(bs, cal["fits"], zc, ss0, bs["n_blocks"])
    print(f"steady state {iso(ss0)} -> {iso(ss1)} ({bs['n_blocks']} blocks)")
    print(f"Arm N: {episodes(arm)} episodes; first alarm "
          f"{iso(ss0 + BLOCK * int(np.argmax(arm))) if arm.any() else '-'}")
    for k, v in al.items():
        if v.any():
            first = ss0 + BLOCK * int(np.argmax(v))
            print(f"    {k:<9} {episodes(v):>3} episodes, first block starting {iso(first)}")
    for e in zc[:20]:
        print(f"    zero-cost: {iso(e[0])} {e[1]} {e[2]}")
    out = Path("reports") / f"phase2_net_arm_score_{path.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "pcap": str(path), "calibration": str(cal_path), "steady_state": [iso(ss0), iso(ss1)],
        "episodes": episodes(arm),
        "alarm_blocks": [iso(ss0 + BLOCK * i) for i in np.flatnonzero(arm)],
        "by_component": {k: {"episodes": episodes(v),
                             "first": iso(ss0 + BLOCK * int(np.argmax(v))) if v.any() else None}
                         for k, v in al.items()},
        "zero_cost_events": [(iso(t), k, d) for t, k, d in zc]}, indent=2))
    print(f"wrote {out}")


# ---------------------------------------------------------------- main
def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pcap")
    ap.add_argument("--calibrate", action="store_true", help="fit Arm N on this (benign) capture")
    ap.add_argument("--score", action="store_true", help="score this capture with a saved calibration")
    ap.add_argument("--calibration", default="reports/phase2_net_arm_calibration.json")
    ap.add_argument("--far-per-day", type=float, default=1.0)
    ap.add_argument("--out", default="reports/phase2_net_arm_calibration.json")
    a = ap.parse_args()
    if a.calibrate == a.score:
        ap.error("pass exactly one of --calibrate or --score")
    path = Path(a.pcap)
    if a.score:
        return score_capture(path, Path(a.calibration))
    t_run = datetime.now()
    print(f"reading {path} ({path.stat().st_size / 1e6:.0f} MB) ...", flush=True)
    ex = extract(path)
    print(f"    {ex['packets']:,} packets in {(datetime.now() - t_run).total_seconds():.0f} s")
    print("roles: " + "; ".join(f"{r}: {', '.join(v)}" for r, v in ex["roles"].items()))

    ss0, ss1 = steady_window(ex)
    mid = ss0 + ((ss1 - ss0) / 2 // BLOCK) * BLOCK
    iso = lambda t: datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")
    print(f"steady state {iso(ss0)} -> {iso(ss1)} ({(ss1 - ss0) / 3600:.2f} h); "
          f"calibration up to {iso(mid)}, held out after")

    wl = learn_whitelist(ex, mid)
    print(f"whitelist learned on the calibration half (fc, start, qty, unit): {wl}")
    bs_cal = block_stats(ex, ss0, mid)
    bs_hold = block_stats(ex, mid, ss1)
    share = 1.0 / len(BUDGETED)
    p_block = share * a.far_per_day / 1440.0
    fits = {}
    print(f"\nbudgeted components (per-block exceedance target {p_block:.2e}, two-sided |z|)")
    print(f"  {'component':<10}{'median':>10}{'scale':>10}{'tau |z|':>9}{'threshold band':>26}{'cal max |z|':>13}")
    for role, st in BUDGETED:
        f = fits[f"{role}.{st}"] = fit_component(bs_cal[role][st], p_block)
        lo, hi = f["median"] - f["tau"] * f["scale"], f["median"] + f["tau"] * f["scale"]
        print(f"  {role + '.' + st:<10}{f['median']:>10.4f}{f['scale']:>10.4f}{f['tau']:>9.2f}"
              f"{f'{lo:.4f} .. {hi:.4f} ms':>26}{f['cal_max_z']:>13.2f}")

    results = {}
    for name, (t_a, t_b, bs) in {"calibration": (ss0, mid, bs_cal), "held_out": (mid, ss1, bs_hold)}.items():
        zc = zero_cost_events(ex, t_a, t_b, wl)
        al, arm = score(bs, fits, zc, t_a, bs["n_blocks"])
        hours = bs["n_blocks"] * BLOCK / 3600
        n_ep = episodes(arm)
        expected = a.far_per_day * hours / 24
        lo, hi = poisson_ci(n_ep)
        results[name] = {"hours": round(hours, 3), "episodes": n_ep, "expected": round(expected, 3),
                         "ci95": [round(lo, 2), round(hi, 2)],
                         "verdict": "consistent" if lo <= expected <= hi else "NOT consistent",
                         "by_component": {k: episodes(v) for k, v in al.items()},
                         "zero_cost_events": [(iso(t), k, d) for t, k, d in zc][:50]}
    print("\nfalse alarms (episodes of 60 s blocks)")
    for name, r in results.items():
        print(f"  {name:<12} {r['hours']:.2f} h: {r['episodes']} episodes, expected {r['expected']:.2f} "
              f"(95 % CI {r['ci95'][0]:.2f}-{r['ci95'][1]:.2f}) -> {r['verdict']}")
        print("      " + "  ".join(f"{k} {v}" for k, v in r["by_component"].items()))
        for e in r["zero_cost_events"][:10]:
            print(f"      zero-cost: {e}")

    try:
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        head = "unknown"
    rep = {
        "step": "phase2 2.3b network-only arm (Arm N)",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": head, "pcap": str(path), "pcap_md5": md5(path),
        "roles": ex["roles"], "steady_state": [iso(ss0), iso(ss1)], "split": iso(mid),
        "params": {"far_per_day": a.far_per_day, "block_s": BLOCK, "share": share, "p_block": p_block,
                   "tail": "exponential fitted to top 10 % of calibration |z|"},
        "whitelist": wl, "fits": fits, "results": results,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
