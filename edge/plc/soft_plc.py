#!/usr/bin/env python3
"""Soft-PLC: a Modbus-master test double for OpenPLC.

It executes the *same* control law as edge/plc/two_tank.st on a fixed scan
cycle, polling the twin's input registers and writing its coils. Two uses:

  1. Bring-up / CI - verify the twin and the control law without OpenPLC.
  2. Fault isolation - if the loop closes with the soft PLC but not with
     OpenPLC, the problem is in the OpenPLC slave-device configuration.

The experiments use OpenPLC; this module exists so that the plant and
the control law can be validated independently of it.

    python edge/plc/soft_plc.py --config configs/twin.yaml --duration 600
"""
from __future__ import annotations

import argparse
import logging
import time

import yaml
from pymodbus.client import ModbusTcpClient

LOG = logging.getLogger("soft-plc")


def run(cfg: dict, duration: float, host: str | None = None) -> dict:
    mb, ctl = cfg["modbus"], cfg["control"]
    host = host or ("127.0.0.1" if mb["host"] in ("0.0.0.0", "") else mb["host"])
    scan = ctl["scan_ms"] / 1000.0
    client = ModbusTcpClient(host, port=int(mb["port"]))
    if not client.connect():
        raise ConnectionError(f"cannot reach twin at {host}:{mb['port']}")

    pump = False
    hh_trip = False
    scans = pump_scans = 0
    t_end = time.perf_counter() + duration if duration else float("inf")
    LOG.info("soft PLC scanning every %d ms (SP %d/%d mm, HH %d mm)",
             ctl["scan_ms"], ctl["sp_low_mm"], ctl["sp_high_mm"], ctl["hh_limit_mm"])
    try:
        while time.perf_counter() < t_end:
            cycle = time.perf_counter()
            rr = client.read_input_registers(0, 5, slave=int(mb["unit_id"]))
            if rr.isError():
                LOG.warning("input-register read failed: %s", rr)
                time.sleep(scan)
                continue
            h1, h2 = rr.registers[0], rr.registers[1]

            if h1 >= ctl["hh_limit_mm"] or h2 >= ctl["hh_limit_mm"]:
                hh_trip = True
            elif h1 < ctl["sp_high_mm"] and h2 < ctl["sp_high_mm"]:
                hh_trip = False

            if hh_trip:
                pump = False
            elif h1 <= ctl["sp_low_mm"]:
                pump = True
            elif h1 >= ctl["sp_high_mm"]:
                pump = False

            client.write_coils(0, [pump, True, hh_trip], slave=int(mb["unit_id"]))
            scans += 1
            pump_scans += int(pump)
            time.sleep(max(0.0, scan - (time.perf_counter() - cycle)))
    finally:
        client.close()
    duty = pump_scans / scans if scans else 0.0
    LOG.info("scans=%d  pump duty cycle=%.1f%%", scans, 100 * duty)
    return {"scans": scans, "pump_duty": duty}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/twin.yaml")
    ap.add_argument("--host", default=None)
    ap.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = forever")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    run(yaml.safe_load(open(a.config)), a.duration, a.host)


if __name__ == "__main__":
    main()
