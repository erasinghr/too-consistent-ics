#!/usr/bin/env python
"""Phase 2, step 2.1 - set and validate the thresholds of the two physics detector arms.

Arm A  "upper side only" - the conventional physics detector (Table 3.2 baseline).
       Fires when the residual is too LARGE, on two time scales, per tank:
         C  CUSUM on the normalised raw residual z = r / sigma0, both bias signs
            (fast attacks; blind to sustained rate biases below k * sigma0)
         M  |mean of z| over a long window (slow ramps; the central-difference
            residual telescopes, so its window mean is a level balance)

Arm B  "two-sided residual test" - the contribution.
       Arm A's C and M (upper side)  +  lower-side fabrication tells, per tank:
         V  windowed variance ratio Q = mean(z^2) implausibly BELOW 1
         P  windowed lag-2 autocorrelation far from the -0.5 that white measurement
            noise through a central difference must produce
         L  any level off the 1 mm quantisation lattice
Both arms carry the same upper-side tests, so the ablation isolates the lower side.

Thresholds come from the first-principles null model, NOT from the benign data:
    level error e = round(u + 1.5 mm * N(0,1)) - u,  u ~ U(0,1) mm   (sensor + 1 mm quantiser)
    r_t = (e_{t+1} - e_{t-1}) / 2 dt,  dt = 1 s  ->  sigma0 = 1.080 mm/s, rho(2) = -0.5
  Tank 1 only (for M): the pump switches at an unknown instant inside the 1 s
  sample interval, which the 1 Hz pump flag cannot place. A switch at fraction f
  before sample i adds  s*f/2  to r[i-1]  and  s*(f-1)/2  to r[i],  f ~ U(0,1),
  s = +/- the pump step. The null replays the benign day's own switch times.
  C, V and P skip the samples around switches (masked), so this term only enters M.
The null is simulated for --mc-days days and each threshold is set so that its
component produces its share of the false-alarm budget. The whole benign day is
then HELD OUT: it validates the thresholds, it never sets them.

Matched false-alarm rate: both arms get the same budget (--far-per-day episodes).
    Arm A, per tank: 1/4 to C, 1/4 to M.
    Arm B, per tank: 1/8 each to C, M, V and P.  L costs nothing - a genuine
    quantised sensor is never off the lattice.
So Arm B pays for its lower side with higher upper thresholds - the fair ablation.

Built on the RAW residual (res1, res2): its null is derived from first principles.
The smoothed floor is tank-specific for reasons not yet established.

Writes reports/phase2_physics_arms_calibration.json. Uses no attack data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SIGMA_SENSOR_MM = 1.5
PRED_SIGMA0 = math.sqrt(SIGMA_SENSOR_MM**2 + 1 / 12) * math.sqrt(2) / 2   # 1.080 mm/s at dt = 1 s
TANKS = (("h1_m", "res1"), ("h2_m", "res2"))
SHARE = {"A_cusum": 1 / 4, "A_mean": 1 / 4,                                     # of the per-arm budget, per tank
         "B_cusum": 1 / 8, "B_mean": 1 / 8, "B_var": 1 / 8, "B_acf": 1 / 8}
QUANTILES = (0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999)


# ---------------------------------------------------------------- statistics (shared by null and data)
def cusum2(z: np.ndarray, valid: np.ndarray, k: float) -> np.ndarray:
    """max(S+, S-) of a CUSUM on both bias signs. Lindley recursion in closed form:
    S_t = C_t - min(0, min_{j<=t} C_j), C = cumsum(x). A zero increment holds S,
    so masked samples leave the statistic unchanged."""
    out = None
    for sign in (1.0, -1.0):
        x = np.where(valid, sign * z - k, 0.0)   # masked -> hold
        c = np.cumsum(x)
        s = c - np.minimum(0.0, np.minimum.accumulate(c))
        out = s if out is None else np.maximum(out, s)
    return out


def window_stats(z: np.ndarray, valid: np.ndarray, w: int):
    """Q: mean z^2 over the trailing window.  P: |lag-2 autocorrelation + 0.5|, with
    numerator and denominator taken over the SAME valid pairs so masking cannot bias it."""
    zs = pd.Series(np.where(valid, z, np.nan))
    q = zs.pow(2).rolling(w, min_periods=w // 2).mean().to_numpy()
    z2 = zs.shift(2)
    num = (zs * z2).rolling(w, min_periods=w // 2).mean()
    den = (0.5 * (zs.pow(2) + z2.pow(2))).where(zs.notna() & z2.notna()).rolling(w, min_periods=w // 2).mean()
    p = np.abs((num / den).to_numpy() + 0.5)
    return q, p


def window_mean_abs(z: np.ndarray, w: int) -> np.ndarray:
    """|trailing mean of z| over exactly w samples, NO masking (keeps the telescoping)."""
    c = np.concatenate(([0.0], np.cumsum(np.nan_to_num(z))))
    out = np.full(len(z), np.nan)
    out[w - 1:] = np.abs(c[w:] - c[:-w]) / w
    return out


def events(alarm: np.ndarray, gap: int) -> int:
    idx = np.flatnonzero(alarm)
    return 0 if idx.size == 0 else int(1 + np.count_nonzero(np.diff(idx) > gap))


# ---------------------------------------------------------------- null model
def simulate_null(days: float, w: int, wm: int, k: float, seed: int, valid_pattern: np.ndarray,
                  sw_idx: np.ndarray, sw_step: np.ndarray):
    """Return float32 arrays of the null: C, Q, P (shared by both tanks, masked samples
    skipped) and M for tank 1 (with the pump-switch timing term) and tank 2 (without).
    Each chunk replays the benign day's own mask pattern and switch times."""
    rng = np.random.default_rng(seed)
    n = len(valid_pattern)
    out = {key: [] for key in ("C", "Q", "P", "M1", "M2")}
    for _ in range(math.ceil(days * 86400 / n)):
        u = rng.random(n + 2)
        e = np.round(u + SIGMA_SENSOR_MM * rng.standard_normal(n + 2)) - u
        r = (e[2:] - e[:-2]) / 2.0                                   # mm/s, dt = 1 s
        z = r / PRED_SIGMA0
        q, p = window_stats(z, valid_pattern, w)
        out["C"].append(cusum2(z, valid_pattern, k).astype(np.float32))
        out["Q"].append(q.astype(np.float32))
        out["P"].append(p.astype(np.float32))
        out["M2"].append(window_mean_abs(z, wm).astype(np.float32))
        r1 = r.copy()
        f = rng.random(len(sw_idx))
        r1[sw_idx - 1] += sw_step * f / 2
        r1[sw_idx] += sw_step * (f - 1) / 2
        out["M1"].append(window_mean_abs(r1 / PRED_SIGMA0, wm).astype(np.float32))
    return {key: np.concatenate(v) for key, v in out.items()}


def solve_threshold(stat: np.ndarray, target_events: float, below: bool, gap: int) -> float:
    """Bisection on the threshold so that the null produces `target_events` alarm episodes."""
    finite = stat[np.isfinite(stat)]
    lo, hi = float(finite.min()), float(finite.max())
    ok = np.isfinite(stat)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        al = (stat < mid) if below else (stat > mid)
        ev = events(al & ok, gap)
        # 'below': larger mid -> more events.  'above': larger mid -> fewer events.
        if (ev > target_events) != below:
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


def transition_mask(df: pd.DataFrame, pad: int) -> np.ndarray:
    """True where a central-difference residual straddles a pump/valve switch.
    r[t] uses h[t+1] - h[t-1]; a switch between samples i-1 and i touches r[i-1], r[i]."""
    n = len(df)
    m = np.zeros(n, dtype=bool)
    for col in ("pump", "valve"):
        if col in df:
            s = df[col].to_numpy()
            for i in np.flatnonzero(s[1:] != s[:-1]) + 1:
                m[max(0, i - 1 - pad): min(n, i + 1 + pad)] = True
    m[0] = m[-1] = True
    return m


def pump_switches(df: pd.DataFrame, fallback_step: float):
    """Switch indices i (first sample after the switch), signed step of the tank-1 model
    in mm/s, and how the step was obtained. The step is read from the model term
    itself (model1 = dh1_dt - res1), so it carries the plant's own q_in / A."""
    p = df["pump"].to_numpy()
    idx = np.flatnonzero(p[1:] != p[:-1]) + 1
    idx = idx[(idx >= 1) & (idx <= len(df) - 2)]
    sign = np.where(p[idx] > p[idx - 1], 1.0, -1.0)
    if "dh1_dt" in df:
        model = (df["dh1_dt"].to_numpy() - df["res1"].to_numpy()) * 1000.0
        jump = np.abs(model[idx] - model[idx - 1])
        step, how = float(np.nanmedian(jump)), "median jump of the tank-1 model term at pump switches"
    else:
        step, how = fallback_step, "--pump-step (dh1_dt column not found)"
    return idx, sign * step, step, how


def poisson_ci(n: int) -> tuple[float, float]:
    """Approximate 95 % interval for a Poisson count (Wilson-Hilferty / Byar)."""
    lo = 0.0 if n == 0 else n * (1 - 1 / (9 * n) - 1.96 / (3 * math.sqrt(n))) ** 3
    m = n + 1
    hi = m * (1 - 1 / (9 * m) + 1.96 / (3 * math.sqrt(m))) ** 3
    return lo, hi


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", default="data/processed/baseline_v1.parquet")
    ap.add_argument("--far-per-day", type=float, default=1.0, help="false-alarm episodes per 24 h, per arm")
    ap.add_argument("--window", type=int, default=60, help="lower-side window (V, P), samples at 1 Hz")
    ap.add_argument("--mean-window", type=int, default=300, help="slow-ramp window (M), samples at 1 Hz")
    ap.add_argument("--k", type=float, default=0.5, help="CUSUM reference value, sigma0 units")
    ap.add_argument("--gap", type=int, default=60, help="alarms closer than this many samples are one episode")
    ap.add_argument("--mask-pad", type=int, default=1, help="extra samples masked each side of a switch")
    ap.add_argument("--pump-step", type=float, default=3.896, help="mm/s, used only if dh1_dt is absent")
    ap.add_argument("--mc-days", type=int, default=200, help="days of null simulated to set thresholds")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", default="reports/phase2_physics_arms_calibration.json")
    a = ap.parse_args()

    # ---- 0. the benign day, its switch mask and switch times (both replayed by the null)
    pq = Path(a.parquet)
    df = pd.read_parquet(pq)
    df = df[df["seq"] >= 74].reset_index(drop=True)
    hours = len(df) / 3600
    mask = transition_mask(df, a.mask_pad)
    sw_idx, sw_step, step, step_how = pump_switches(df, a.pump_step)
    valve_sw = int(np.count_nonzero(np.diff(df["valve"].to_numpy()))) if "valve" in df else 0
    print(f"benign day: {len(df)} closed-loop rows (~{hours:.2f} h), "
          f"{mask.sum()} masked around pump/valve switches ({100 * mask.mean():.2f} %)")
    print(f"pump switches: {len(sw_idx)}   pump step {step:.3f} mm/s ({step_how})   valve switches: {valve_sw}")
    if valve_sw:
        print("    WARNING: valve switches are masked for C/V/P but not modelled in the M null")

    # ---- 1. thresholds from the null
    print(f"simulating {a.mc_days} days of the null model (sigma0 = {PRED_SIGMA0:.3f} mm/s) ...", flush=True)
    null = simulate_null(a.mc_days, a.window, a.mean_window, a.k, a.seed, ~mask, sw_idx, sw_step)
    null_days = len(null["C"]) / 86400

    def solve(comp: str, stat: np.ndarray, below: bool) -> float:
        target = a.far_per_day * SHARE[comp] * null_days
        return solve_threshold(stat, target, below, a.gap)

    thr = {"A_cusum": solve("A_cusum", null["C"], False), "B_cusum": solve("B_cusum", null["C"], False),
           "B_var": solve("B_var", null["Q"], True), "B_acf": solve("B_acf", null["P"], False)}
    thr_mean = {"res1": {"A_mean": solve("A_mean", null["M1"], False), "B_mean": solve("B_mean", null["M1"], False)},
                "res2": {"A_mean": solve("A_mean", null["M2"], False), "B_mean": solve("B_mean", null["M2"], False)}}
    for comp, v in thr.items():
        print(f"    {comp:<8}        threshold {'<' if comp == 'B_var' else '>'} {v:.4f}")
    for rcol, d in thr_mean.items():
        for comp, v in d.items():
            print(f"    {comp:<8} {rcol}   threshold > {v:.5f}   (a sustained rate bias of ~{v * PRED_SIGMA0:.4f} mm/s "
                  f"over {a.mean_window} s)")
    null_q = {"cusum": np.quantile(null["C"], QUANTILES), "Q": np.nanquantile(null["Q"], QUANTILES),
              "P": np.nanquantile(null["P"], QUANTILES),
              "M res1": np.nanquantile(null["M1"], QUANTILES), "M res2": np.nanquantile(null["M2"], QUANTILES)}
    del null

    # ---- 2. the benign day, entirely held out
    per_tank, alarms = {}, {"A": np.zeros(len(df), bool), "B": np.zeros(len(df), bool)}
    for hcol, rcol in TANKS:
        r = df[rcol].to_numpy() * 1000.0                     # m/s -> mm/s
        v = ~mask & np.isfinite(r)
        z = np.where(v, r / PRED_SIGMA0, 0.0)
        c = cusum2(z, v, a.k)
        q, p = window_stats(z, v, a.window)
        m = window_mean_abs(r / PRED_SIGMA0, a.mean_window)  # all rows: telescoping intact
        hmm = df[hcol].to_numpy() * 1000.0
        off = np.abs(hmm - np.round(hmm)) > 1e-6
        lat = pd.Series(off.astype(float)).rolling(a.window, min_periods=1).sum().to_numpy() > 0
        tm = thr_mean[rcol]

        comp = {"A_cusum": c > thr["A_cusum"], "A_mean": np.nan_to_num(m) > tm["A_mean"],
                "B_cusum": c > thr["B_cusum"], "B_mean": np.nan_to_num(m) > tm["B_mean"],
                "B_var": np.nan_to_num(q, nan=np.inf) < thr["B_var"],
                "B_acf": np.nan_to_num(p, nan=0.0) > thr["B_acf"], "B_lattice": lat}
        alarms["A"] |= comp["A_cusum"] | comp["A_mean"]
        alarms["B"] |= comp["B_cusum"] | comp["B_mean"] | comp["B_var"] | comp["B_acf"] | comp["B_lattice"]

        per_tank[rcol] = {
            "sigma_measured_mm_s": round(float(np.std(r[v])), 4),
            "sigma_predicted_mm_s": round(PRED_SIGMA0, 4),
            "acf_lag2_measured": round(float(pd.Series(np.where(v, r, np.nan)).autocorr(2)), 4),
            "off_lattice_samples": int(off.sum()),
            "quantiles_benign": {"cusum": np.quantile(c[v], QUANTILES).round(4).tolist(),
                                 "Q": np.nanquantile(q, QUANTILES).round(4).tolist(),
                                 "P": np.nanquantile(p, QUANTILES).round(4).tolist(),
                                 "M": np.nanquantile(m, QUANTILES).round(5).tolist()},
            "thresholds_mean": {k_: round(v_, 6) for k_, v_ in tm.items()},
            "episodes": {k_: events(v_, a.gap) for k_, v_ in comp.items()},
        }

    # ---- 3. report
    print("\nnull fidelity - quantiles of each statistic, null model vs benign day")
    print(f"    {'q':>11}" + "".join(f"{x:>9}" for x in QUANTILES))
    rows = [("cusum", "cusum", ("res1", "res2")), ("Q", "Q", ("res1", "res2")), ("P", "P", ("res1", "res2")),
            ("M res1", "M", ("res1",)), ("M res2", "M", ("res2",))]
    for null_key, key, tanks in rows:
        print(f"    {null_key + ' null':<14}" + "".join(f"{x:>9.4f}" for x in null_q[null_key]))
        for rcol in tanks:
            print(f"    {key + ' ' + rcol:<14}" + "".join(f"{x:>9.4f}" for x in per_tank[rcol]["quantiles_benign"][key]))

    for _, rcol in TANKS:
        p_ = per_tank[rcol]
        print(f"\n[{rcol}] sigma {p_['sigma_measured_mm_s']:.3f} mm/s (predicted {p_['sigma_predicted_mm_s']:.3f})   "
              f"acf lag 2 {p_['acf_lag2_measured']:+.3f} (predicted -0.500)   off-lattice {p_['off_lattice_samples']}")
        print("    false-alarm episodes by component: " +
              "  ".join(f"{k_} {v_}" for k_, v_ in p_["episodes"].items()))

    expected = a.far_per_day * hours / 24
    fa = {}
    print(f"\nfalse alarms on the held-out benign day   (budget {a.far_per_day:g} per 24 h per arm "
          f"-> {expected:.2f} expected in {hours:.1f} h)")
    for arm in ("A", "B"):
        n = events(alarms[arm], a.gap)
        lo, hi = poisson_ci(n)
        verdict = "consistent" if lo <= expected <= hi else "NOT consistent"
        fa[arm] = {"episodes": n, "per_24h": round(n * 24 / hours, 3), "ci95_count": [round(lo, 2), round(hi, 2)],
                   "expected": round(expected, 3), "verdict": verdict}
        print(f"    Arm {arm}: {n} episodes  (95 % CI {lo:.2f}-{hi:.2f})  -> {verdict} with the budget")

    report = {
        "step": "phase2 2.1 physics-arm thresholds (null-model calibrated, benign-validated)",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": git_head(),
        "parquet": str(pq), "parquet_md5": md5(pq),
        "rows_closed_loop": int(len(df)), "masked_rows": int(mask.sum()),
        "pump_switches": int(len(sw_idx)), "pump_step_mm_s": round(step, 4), "pump_step_source": step_how,
        "null_model": {"sigma_sensor_mm": SIGMA_SENSOR_MM, "quantum_mm": 1.0, "dt_s": 1.0,
                       "sigma0_mm_s": round(PRED_SIGMA0, 4), "mc_days": a.mc_days, "seed": a.seed,
                       "switch_timing": "tank 1 only: f ~ U(0,1) inside the sample interval",
                       "quantiles": {k_: v_.round(5).tolist() for k_, v_ in null_q.items()}},
        "params": {"far_per_day": a.far_per_day, "window": a.window, "mean_window": a.mean_window, "k": a.k,
                   "gap": a.gap, "mask_pad": a.mask_pad, "budget_share_per_tank": SHARE},
        "thresholds": {k_: round(v_, 5) for k_, v_ in thr.items()},
        "per_tank": per_tank,
        "false_alarms_benign": fa,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
