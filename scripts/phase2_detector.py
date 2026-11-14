#!/usr/bin/env python
"""Phase 2, step 2.2b - the physics detector arms, calibrated on the simulator null.

Statistics, per tank (z = r / sigma0 is the raw rate residual of add_derived;
a = h - h(q) is the algebraic residual: measured level minus the level that
Torricelli says produced the measured flow, in mm):

  upper side - "the residual is too LARGE" - both arms
    C  CUSUM on z, both bias signs, skipping samples around pump switches
    M  |mean z| over --mean-window s (all rows: the central differences telescope)
    T  |mean a| over --alg-window s (a standing offset between level and flow)
    X  q_in disagrees with Qpump * pump away from a switch          (zero cost)
  lower side - "the residual is too SMALL / too regular" - Arm B only
    V  mean z^2 over --window s implausibly below 1
    P  lag-2 autocorrelation of z far from the -0.5 a genuine sensor gives
    L  a level off the 1 mm lattice                                  (zero cost)
    F  a flow off the 1 cL/min lattice                               (zero cost)
Arm A = C, M, T, X.  Arm B = Arm A + V, P, L, F.  Both arms carry the same upper
side, so the ablation isolates the lower side.

Null: benign days from scripts/phase2_sim.py (validated against the released
day on 29 statistics, 13 Oct 2026). Every threshold is set so its component
produces its share of the false-alarm budget on the simulated days; the
released day is HELD OUT and only validates.

Budget (--far-per-day episodes per arm, split per tank):
    Arm A: C, M, T 1/6 each.   Arm B: C, M, T, V, P 1/10 each.
Zero-cost components never fire on a genuine stack, so they take no budget.

Episodes are counted on 60 s blocks: an episode is a run of consecutive blocks
that each contain at least one alarm. The same rule is used for calibration,
validation and the attack runs.

    python scripts/phase2_detector.py --calibrate            # ~8-10 min for 200 days
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_sim import level_from_flow, simulate_iter, switch_mask  # noqa: E402

SIGMA_SENSOR_MM = 1.5
SIGMA0 = math.sqrt(SIGMA_SENSOR_MM**2 + 1 / 12) * math.sqrt(2) / 2     # 1.080 mm/s at dt = 1 s
LPM_TO_M3S = 1.0 / 60000.0
BLOCK = 60
TANKS = ("1", "2")
BUDGETED = ("C", "M", "T", "V", "P")
DIRECTION = {"C": "above", "M": "above", "T": "above", "V": "below", "P": "above"}
SHARE = {"A": {"C": 1 / 6, "M": 1 / 6, "T": 1 / 6},
         "B": {"C": 1 / 10, "M": 1 / 10, "T": 1 / 10, "V": 1 / 10, "P": 1 / 10}}
ZERO_COST = {"A": ("X",), "B": ("X", "L", "F")}
UNITS = {"C": "sigma0", "M": "sigma0", "T": "mm", "V": "ratio", "P": "|acf2+0.5|"}


# ---------------------------------------------------------------- statistics
def cusum2(z: np.ndarray, valid: np.ndarray, k: float) -> np.ndarray:
    """max(S+, S-), Lindley recursion in closed form; masked samples hold the statistic."""
    out = None
    for sign in (1.0, -1.0):
        x = np.where(valid, sign * z - k, 0.0)
        c = np.cumsum(x)
        s = c - np.minimum(0.0, np.minimum.accumulate(c))
        out = s if out is None else np.maximum(out, s)
    return out


def window_qp(z: np.ndarray, valid: np.ndarray, w: int):
    zs = pd.Series(np.where(valid, z, np.nan))
    q = zs.pow(2).rolling(w, min_periods=w // 2).mean().to_numpy()
    z2 = zs.shift(2)
    num = (zs * z2).rolling(w, min_periods=w // 2).mean()
    den = (0.5 * (zs.pow(2) + z2.pow(2))).where(zs.notna() & z2.notna()).rolling(w, min_periods=w // 2).mean()
    return q, np.abs((num / den).to_numpy() + 0.5)


def window_mean_abs(x: np.ndarray, w: int) -> np.ndarray:
    return np.abs(pd.Series(x).rolling(w, min_periods=w).mean().to_numpy())


def statistics(df: pd.DataFrame, plant: dict, p: dict) -> dict:
    """Per-sample statistics of one telemetry frame (schema of baseline_v1.parquet)."""
    g = float(plant["g"])
    pump = df["pump"].to_numpy().astype(int)
    mask = switch_mask(pump, p["mask_pad"])
    valve_open = df["valve"].to_numpy().astype(bool) if "valve" in df else np.ones(len(df), bool)
    out = {}
    for t in TANKS:
        r = df[f"res{t}"].to_numpy() * 1000.0
        v = ~mask & np.isfinite(r)
        z = np.where(v, r / SIGMA0, 0.0)
        q, pp = window_qp(z, v, p["window"])
        h_mm = df[f"h{t}_m"].to_numpy() * 1000.0
        if t == "1":
            a = h_mm - level_from_flow(df["q12_lpm"].to_numpy(), float(plant["Cv12"]), g)
            a = np.where(valve_open, a, np.nan)
        else:
            a = h_mm - level_from_flow(df["q_out_lpm"].to_numpy(), float(plant["Cv2o"]), g)
        off = np.abs(h_mm - np.rint(h_mm)) > 1e-6
        out[t] = {
            "C": cusum2(z, v, p["k"]),
            "M": window_mean_abs(r / SIGMA0, p["mean_window"]),
            "T": window_mean_abs(a, p["alg_window"]),
            "V": q, "P": pp,
            "L": pd.Series(off.astype(float)).rolling(p["window"], min_periods=1).sum().to_numpy() > 0,
        }
    qin_expected = float(plant["Qpump"]) / LPM_TO_M3S * pump
    out["X"] = (np.abs(df["q_in_lpm"].to_numpy() - qin_expected) > 0.005) & ~mask
    flows = df[["q_in_lpm", "q12_lpm", "q_out_lpm"]].to_numpy() * 100.0
    out["F"] = np.any(np.abs(flows - np.rint(flows)) > 1e-6, axis=1)
    return out


def component_alarms(st: dict, thr: dict, arm: str) -> dict:
    """Boolean alarm series per component (keys like 'C1', 'T2', 'X', 'F')."""
    al = {}
    for t in TANKS:
        for c, th in thr[arm].items():
            x = st[t][c]
            al[c + t] = (np.nan_to_num(x, nan=np.inf) < th[t]) if DIRECTION[c] == "below" \
                else (np.nan_to_num(x, nan=-np.inf) > th[t])
        if arm == "B":
            al["L" + t] = st[t]["L"]
    for z in ("X", "F"):
        if z in ZERO_COST[arm]:
            al[z] = st[z]
    return al


def arm_alarm(st: dict, thr: dict, arm: str) -> np.ndarray:
    return np.logical_or.reduce(list(component_alarms(st, thr, arm).values()))


# ---------------------------------------------------------------- blocks and episodes
def blocks(x: np.ndarray, how: str) -> np.ndarray:
    n = math.ceil(len(x) / BLOCK) * BLOCK
    y = np.full(n, np.nan)
    y[: len(x)] = x
    y = y.reshape(-1, BLOCK)
    with np.errstate(all="ignore"):
        if np.all(np.isnan(y), axis=1).any():
            fill = -np.inf if how == "max" else np.inf
            y = np.where(np.all(np.isnan(y), axis=1, keepdims=True), fill, y)
        r = np.nanmax(y, axis=1) if how == "max" else np.nanmin(y, axis=1)
    return r.astype(np.float32)


def episodes(alarm_blocks: np.ndarray) -> int:
    a = np.asarray(alarm_blocks, bool)
    return int(a[0]) + int(np.count_nonzero(a[1:] & ~a[:-1])) if a.size else 0


def bool_blocks(x: np.ndarray) -> np.ndarray:
    return blocks(x.astype(float), "max") > 0.5


def solve(bl: np.ndarray, target: float, below: bool) -> float:
    finite = bl[np.isfinite(bl)]
    lo, hi = float(finite.min()), float(finite.max())
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        ev = episodes(bl < mid) if below else episodes(bl > mid)
        if (ev > target) != below:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- helpers
def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def poisson_ci(n: int) -> tuple[float, float]:
    lo = 0.0 if n == 0 else n * (1 - 1 / (9 * n) - 1.96 / (3 * math.sqrt(n))) ** 3
    m = n + 1
    return lo, m * (1 - 1 / (9 * m) + 1.96 / (3 * math.sqrt(m))) ** 3


def load_detector(path: str = "reports/phase2_detector_calibration.json") -> tuple[dict, dict]:
    """(thresholds, params) for use by the attack harness."""
    rep = json.loads(Path(path).read_text())
    return rep["thresholds"], rep["params"]


# ---------------------------------------------------------------- calibration
def calibrate(a) -> None:
    cfg = yaml.safe_load(open(a.config))
    plant = cfg["plant"]
    p = {"window": a.window, "mean_window": a.mean_window, "alg_window": a.alg_window, "k": a.k,
         "mask_pad": a.mask_pad}

    # ---- 1. the simulator null
    store = {t: {c: [] for c in BUDGETED} for t in TANKS}
    zero_cost_hits = {"X": 0, "F": 0, "L1": 0, "L2": 0}
    days_done, batch_i = 0, 0
    t0 = datetime.now()
    while days_done < a.days:
        runs = min(a.batch, a.days - days_done)
        print(f"null batch {batch_i + 1}: simulating {runs} benign days (seed {a.seed + batch_i}) ...", flush=True)
        for df in simulate_iter(cfg, runs, 86400, a.seed + batch_i):
            st = statistics(df, plant, p)
            for t in TANKS:
                for c in BUDGETED:
                    how = "min" if DIRECTION[c] == "below" else "max"
                    sep = np.array([np.inf if how == "min" else -np.inf], np.float32)   # no episode spans two runs
                    store[t][c].extend([blocks(st[t][c], how), sep])
                zero_cost_hits["L" + t] += int(st[t]["L"].sum())
            zero_cost_hits["X"] += int(st["X"].sum())
            zero_cost_hits["F"] += int(st["F"].sum())
        days_done += runs
        batch_i += 1
        print(f"    {days_done}/{a.days} days, {(datetime.now() - t0).total_seconds() / 60:.1f} min elapsed", flush=True)
    null = {t: {c: np.concatenate(v) for c, v in d.items()} for t, d in store.items()}

    thr = {arm: {c: {} for c in sh} for arm, sh in SHARE.items()}
    print(f"\nthresholds from {a.days} simulated days (budget {a.far_per_day:g} episode per 24 h per arm)")
    for arm, sh in SHARE.items():
        for c, share in sh.items():
            for t in TANKS:
                target = a.far_per_day * share * a.days
                thr[arm][c][t] = solve(null[t][c], target, DIRECTION[c] == "below")
    for c in BUDGETED:
        cells = []
        for arm in ("A", "B"):
            if c in thr[arm]:
                cells.append(f"Arm {arm}: " + "  ".join(
                    f"tank {t} {'<' if DIRECTION[c] == 'below' else '>'} {thr[arm][c][t]:.4f}" for t in TANKS))
        extra = ""
        if c == "M":
            extra = f"   (Arm A ~ {thr['A']['M']['1'] * SIGMA0:.4f} / {thr['A']['M']['2'] * SIGMA0:.4f} mm/s)"
        print(f"    {c} [{UNITS[c]}]  " + "   |   ".join(cells) + extra)
    print(f"    zero-cost components on the null (must all be 0): {zero_cost_hits}")

    # ---- 2. the released day, held out
    pq = Path(a.parquet)
    real = pd.read_parquet(pq)
    real = real[real["seq"] >= 74].reset_index(drop=True)
    hours = len(real) / 3600
    st = statistics(real, plant, p)

    print("\nnull fidelity - quantiles of 60 s block maxima (minima for V), simulated null vs released day")
    qs_hi, qs_lo = (0.5, 0.9, 0.99, 0.999), (0.001, 0.01, 0.1, 0.5)
    fidelity = {}
    for c in BUDGETED:
        qs = qs_lo if DIRECTION[c] == "below" else qs_hi
        how = "min" if DIRECTION[c] == "below" else "max"
        print(f"    {c:<3}{'q':>12}" + "".join(f"{x:>9}" for x in qs))
        for t in TANKS:
            nb = null[t][c][np.isfinite(null[t][c])]
            rb = blocks(st[t][c], how)
            rb = rb[np.isfinite(rb)]
            nq, rq = np.quantile(nb, qs), np.quantile(rb, qs)
            fidelity[c + t] = {"null": nq.round(5).tolist(), "real": rq.round(5).tolist(), "q": list(qs)}
            print(f"       tank {t} null " + "".join(f"{x:>9.4f}" for x in nq))
            print(f"       tank {t} real " + "".join(f"{x:>9.4f}" for x in rq))

    expected = a.far_per_day * hours / 24
    fa = {}
    print(f"\nfalse alarms on the held-out released day ({hours:.2f} h, {expected:.2f} expected per arm)")
    for arm in ("A", "B"):
        comps = component_alarms(st, thr, arm)
        per = {k: episodes(bool_blocks(v)) for k, v in comps.items()}
        n = episodes(bool_blocks(np.logical_or.reduce(list(comps.values()))))
        lo, hi = poisson_ci(n)
        verdict = "consistent" if lo <= expected <= hi else "NOT consistent"
        fa[arm] = {"episodes": n, "ci95": [round(lo, 2), round(hi, 2)], "expected": round(expected, 3),
                   "verdict": verdict, "by_component": per}
        print(f"    Arm {arm}: {n} episodes (95 % CI {lo:.2f}-{hi:.2f}) -> {verdict}")
        print("        " + "  ".join(f"{k} {v}" for k, v in per.items()))

    report = {
        "step": "phase2 2.2b physics detector, simulator-null calibrated, released-day validated",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": git_head(),
        "config": a.config, "config_md5": md5(Path(a.config)),
        "parquet": str(pq), "parquet_md5": md5(pq),
        "null": {"days": a.days, "seed": a.seed, "batch": a.batch, "zero_cost_hits": zero_cost_hits},
        "params": {**p, "far_per_day": a.far_per_day, "block_s": BLOCK, "sigma0_mm_s": SIGMA0,
                   "share": SHARE, "zero_cost": ZERO_COST, "direction": DIRECTION},
        "thresholds": thr,
        "fidelity": fidelity,
        "false_alarms_released_day": fa,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--config", default="configs/twin.yaml")
    ap.add_argument("--parquet", default="data/processed/baseline_v1.parquet")
    ap.add_argument("--days", type=int, default=200, help="simulated benign days for the null")
    ap.add_argument("--batch", type=int, default=50, help="days simulated in parallel (~0.4 GB at 50)")
    ap.add_argument("--far-per-day", type=float, default=1.0)
    ap.add_argument("--window", type=int, default=60, help="V and P window, s")
    ap.add_argument("--mean-window", type=int, default=300, help="M window, s")
    ap.add_argument("--alg-window", type=int, default=120, help="T window, s")
    ap.add_argument("--k", type=float, default=0.5, help="CUSUM reference value, sigma0 units")
    ap.add_argument("--mask-pad", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--out", default="reports/phase2_detector_calibration.json")
    a = ap.parse_args()
    if not a.calibrate:
        ap.error("nothing to do - pass --calibrate")
    calibrate(a)


if __name__ == "__main__":
    main()
