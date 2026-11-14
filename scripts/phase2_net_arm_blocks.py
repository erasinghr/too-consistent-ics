#!/usr/bin/env python
"""Phase 2.5 - per-block view of Arm N on one or more captures (attribution control).

For every budgeted component (plc.lat, plc.gap, plc.jit, pub.lat) it prints the block
values and the signed z = (x - cal median) / cal scale against the saved calibration,
so a benign control and an attack capture can be compared component by component:
which components separate them, by how much, and in which direction a drift goes.
Read-only: it reuses phase2_net_arm.py and changes nothing.

    python scripts/phase2_net_arm_blocks.py data/raw/pcap_benign_ctrl/*.pcap data/raw/pcap_attack/*.pcap
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_net_arm import BLOCK, BUDGETED, block_stats, extract, steady_window  # noqa: E402


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pcaps", nargs="+", type=Path)
    ap.add_argument("--calibration", default="reports/phase2_net_arm_calibration.json")
    ap.add_argument("--out", default="reports/phase2_net_arm_blocks.json")
    a = ap.parse_args()

    cal = json.loads(Path(a.calibration).read_text())
    fits = cal["fits"]
    comps = [f"{r}.{s}" for r, s in BUDGETED]
    print(f"calibration: {a.calibration} (day: {cal['pcap']})")
    print(f"  {'component':<9} {'cal median':>11} {'scale':>7} {'tau':>7}   alarm band (ms)")
    for k in comps:
        f = fits[k]
        lo, hi = f["median"] - f["tau"] * f["scale"], f["median"] + f["tau"] * f["scale"]
        print(f"  {k:<9} {f['median']:>11.3f} {f['scale']:>7.3f} {f['tau']:>7.2f}   {lo:.3f} - {hi:.3f}")

    rep = {"calibration": a.calibration, "calibration_day": cal["pcap"], "captures": {}}
    for p in a.pcaps:
        print(f"\n== {p.name}")
        try:
            ex = extract(p)
            ss0, ss1 = steady_window(ex)
            bs = block_stats(ex, ss0, ss1)
        except (SystemExit, Exception) as e:          # an aborted or proxy-only capture
            print(f"   skipped: {e}")
            continue
        nb = bs["n_blocks"]
        if nb < 1:
            print(f"   skipped: steady state {iso(ss0)} -> {iso(ss1)} holds no full {BLOCK} s block")
            continue
        print(f"   steady state {iso(ss0)} -> {iso(ss1)}  ({nb} blocks)")
        print(f"   {'component':<9} {'median':>9} {'shift':>8} {'shift z':>8} {'min|z|':>7} {'max|z|':>7} "
              f"{'tau':>6} {'alarmed':>8}")
        cap = {"steady_state": [iso(ss0), iso(ss1)], "n_blocks": nb, "components": {}}
        rows = []
        for (role, st), k in zip(BUDGETED, comps):
            f = fits[k]
            x = np.asarray(bs[role][st], float)
            ok = np.isfinite(x)
            z = (x - f["median"]) / f["scale"]
            az = np.abs(z[ok])
            alarm = np.nan_to_num(np.abs(z), nan=0.0) > f["tau"]
            med = float(np.median(x[ok])) if ok.any() else float("nan")
            shift = med - f["median"]
            print(f"   {k:<9} {med:>9.3f} {shift:>+8.3f} {shift / f['scale']:>+8.1f} "
                  f"{(az.min() if az.size else float('nan')):>7.1f} {(az.max() if az.size else float('nan')):>7.1f} "
                  f"{f['tau']:>6.2f} {int(alarm.sum()):>4}/{nb:<3}")
            rows.append((k, z, alarm))
            cap["components"][k] = {
                "block_values_ms": [None if not np.isfinite(v) else round(float(v), 4) for v in x],
                "signed_z": [None if not np.isfinite(v) else round(float(v), 2) for v in z],
                "median_ms": med, "shift_ms": shift, "tau": f["tau"],
                "alarmed_blocks": int(alarm.sum()),
            }
        print("   per-block signed z (* = alarm):")
        for k, z, alarm in rows:
            cells = ["   nan" if not np.isfinite(v) else f"{v:>6.1f}" + ("*" if al else " ")
                     for v, al in zip(z, alarm)]
            print(f"     {k:<9}" + "".join(c if c.endswith(("*", " ")) else c + " " for c in cells))
        rep["captures"][str(p)] = cap

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
