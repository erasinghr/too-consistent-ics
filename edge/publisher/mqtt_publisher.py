#!/usr/bin/env python3
"""Hybrid AI-Quantum Cognitive Security Framework Phase 1 - edge MQTT publisher.

Sits beside the plant on the edge tier, polls the digital twin over Modbus/TCP
(exactly as a data-acquisition gateway would poll a real skid) and republishes
each sample as a JSON telemetry frame on MQTT for the fog tier.

Keeping this a *separate process* from the twin is deliberate: it produces real
Modbus and MQTT traffic on the wire, which Phase 2 onwards uses to derive the
network-signal features S_net of the PN-Fusion detector.

Telemetry schema (topic cps/edge/plant1/telemetry, one JSON object per sample):
    seq        int    monotonically increasing sample counter
    ts         float  epoch seconds, publisher clock
    ts_iso     str    ISO-8601 UTC timestamp
    h1_m       float  tank 1 level, metres
    h2_m       float  tank 2 level, metres
    q_in_lpm   float  pump inflow,  litres/min
    q12_lpm    float  V12 flow,     litres/min
    q_out_lpm  float  outlet flow,  litres/min
    pump       int    PLC pump coil, 0/1
    valve      int    PLC valve coil, 0/1
    alarm      int    PLC alarm coil, 0/1
    src        str    originating node id

Usage:
    python edge/publisher/mqtt_publisher.py --config configs/twin.yaml
    python edge/publisher/mqtt_publisher.py --config configs/twin.yaml --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import time
from datetime import datetime, timezone

import yaml
from pymodbus.client import ModbusTcpClient

LOG = logging.getLogger("publisher")
_RUN = True


def _stop(*_):
    global _RUN
    _RUN = False


def build_frame(seq: int, regs: list[int], coils: list[bool], src: str) -> dict:
    now = time.time()
    return {
        "seq": seq,
        "ts": round(now, 3),
        "ts_iso": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "h1_m": regs[0] / 1000.0,
        "h2_m": regs[1] / 1000.0,
        "q_in_lpm": regs[2] / 100.0,
        "q12_lpm": regs[3] / 100.0,
        "q_out_lpm": regs[4] / 100.0,
        "pump": int(bool(coils[0])),
        "valve": int(bool(coils[1])),
        "alarm": int(bool(coils[2])),
        "src": src,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/twin.yaml")
    ap.add_argument("--modbus-host", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print frames instead of publishing")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = forever")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")

    cfg = yaml.safe_load(open(a.config))
    mb, mq = cfg["modbus"], cfg["mqtt"]
    host = a.modbus_host or ("127.0.0.1" if mb["host"] in ("0.0.0.0", "") else mb["host"])
    period = 1.0 / float(mq["publish_hz"])
    unit = int(mb["unit_id"])

    client = ModbusTcpClient(host, port=int(mb["port"]))
    if not client.connect():
        raise ConnectionError(f"cannot reach twin at {host}:{mb['port']}")

    mqtt_client = None
    if not a.dry_run:
        import paho.mqtt.client as mqtt
        try:                      # paho-mqtt >= 2.0
            mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                      client_id=mq["client_id"])
        except AttributeError:    # paho-mqtt 1.x
            mqtt_client = mqtt.Client(client_id=mq["client_id"])
        mqtt_client.connect(mq["host"], int(mq["port"]), keepalive=30)
        mqtt_client.loop_start()
        LOG.info("connected to MQTT %s:%s, topic %s", mq["host"], mq["port"], mq["topic"])

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    seq, sent, errors = 0, 0, 0
    t_end = time.perf_counter() + a.duration if a.duration else float("inf")
    LOG.info("publishing at %.2f Hz from modbus %s:%s", mq["publish_hz"], host, mb["port"])
    try:
        while _RUN and time.perf_counter() < t_end:
            cycle = time.perf_counter()
            rr = client.read_input_registers(0, 5, slave=unit)
            rc = client.read_coils(0, 3, slave=unit)
            if rr.isError() or rc.isError():
                errors += 1
                LOG.warning("modbus read error (%d total)", errors)
            else:
                frame = build_frame(seq, rr.registers, rc.bits[:3], mq["client_id"])
                payload = json.dumps(frame, separators=(",", ":"))
                if a.dry_run:
                    print(payload, flush=True)
                else:
                    mqtt_client.publish(mq["topic"], payload, qos=int(mq.get("qos", 0)))
                seq += 1
                sent += 1
                if sent % 60 == 0:
                    LOG.info("%d frames published (h1=%.3f m, pump=%d)",
                             sent, frame["h1_m"], frame["pump"])
            time.sleep(max(0.0, period - (time.perf_counter() - cycle)))
    finally:
        client.close()
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        LOG.info("stopped: %d frames published, %d modbus errors", sent, errors)


if __name__ == "__main__":
    main()
