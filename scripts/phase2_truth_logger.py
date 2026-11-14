#!/usr/bin/env python
"""Phase 2, step 2.4 - ground-truth logger for a live attack run.

In a source-spoof run the recorded telemetry is the fabricated feed, so it no longer
reflects the real plant. This polls the TRUE twin directly (on :5030, beside the proxy,
not through it) at 1 Hz and writes the real levels and coils to a CSV, so the physical
consequence of a fabrication - did the real tank overflow or run dry, and when - can be
measured against the injector's onset.

It is read-only (FC4 input registers, FC1 coils) and adds one more Modbus client to the
twin; it does not touch the attack path. Uses the same pymodbus API as edge/plc/soft_plc.py.

    python scripts/phase2_truth_logger.py --port 5030 --out logs/truth_<run>.csv
"""
from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pymodbus.client import ModbusTcpClient

RUN = True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5030, help="the TRUE twin's port (not the proxy's 5020)")
    ap.add_argument("--unit", type=int, default=1)
    ap.add_argument("--hz", type=float, default=1.0)
    ap.add_argument("--out", default=None, help="CSV path; default logs/truth_<utc>.csv")
    a = ap.parse_args()

    out = Path(a.out or f"logs/truth_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    client = ModbusTcpClient(a.host, port=a.port)
    if not client.connect():
        sys.exit(f"ERROR: cannot reach the true twin at {a.host}:{a.port} - is it up on {a.port}?")

    def stop(*_):
        global RUN
        RUN = False
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    period = 1.0 / a.hz
    n = bad = 0
    print(f"ground-truth logger: {a.host}:{a.port} unit {a.unit} at {a.hz} Hz -> {out}", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_unix", "ts_iso", "h1_true_mm", "h2_true_mm", "q_in_clpm", "q12_clpm",
                    "q_out_clpm", "pump", "valve", "alarm"])
        while RUN:
            t = time.perf_counter()
            now = time.time()
            rr = client.read_input_registers(0, 5, slave=a.unit)
            rc = client.read_coils(0, 3, slave=a.unit)
            if rr.isError() or rc.isError():
                bad += 1
            else:
                reg, co = rr.registers, rc.bits
                w.writerow([f"{now:.3f}", datetime.fromtimestamp(now, timezone.utc).isoformat(),
                            reg[0], reg[1], reg[2], reg[3], reg[4],
                            int(co[0]), int(co[1]), int(co[2])])
                n += 1
                if n % 60 == 0:
                    f.flush()
                    print(f"  {n} rows; last true h1={reg[0]} mm h2={reg[1]} mm pump={int(co[0])} "
                          f"alarm={int(co[2])}", flush=True)
            time.sleep(max(0.0, period - (time.perf_counter() - t)))
    client.close()
    print(f"\nstopped: {n} rows, {bad} failed reads -> {out}", flush=True)


if __name__ == "__main__":
    main()
