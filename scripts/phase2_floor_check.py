#!/usr/bin/env python
"""Phase 2, step 2.1 - diagnostic only (not a calibration artefact).

Tests four predictions derived from the sensor model (1.5 mm Gaussian noise,
1 mm quantisation) against the released benign capture, before the two-sided
residual test is calibrated:

  1. every level value sits on the 1 mm lattice (so a lattice-membership
     test cannot fire on anything a Modbus register can carry);
  2. the mass-balance residual floor is 1.080 mm/s under a central difference
     and 2.160 mm/s under a first difference - the measured 1.077 / 1.108
     identify which one make_baseline_dataset.py used;
  3. tank 1's excess over tank 2 comes from the two samples bracketing each
     pump transition, and disappears when they are masked;
  4. the residual ACF is ~0 at lag 1 and ~-0.5 at lag 2, induced by the
     central difference itself.

  5. the rolling-median ("smoothed") residual floor is NOT a constant: a
     median filter stops removing noise once the level moves ~1-2 mm per
     sample, so its floor rises with slope (~0.44 flat, ~0.83 at 2 mm/s),
     which is why tank 1 (always ramping) reads 0.882 and tank 2 0.445.
     The raw central-difference floor stays at 1.080 at every slope.

Also lists any residual columns already stored in the parquet.

A1 = A2 = 0.0154 m^2 is hard-coded from Table 4.1 because this is a one-off
check; the calibration script proper will read configs/twin.yaml.
"""
import sys

import numpy as np
import pandas as pd

PARQUET = "data/processed/baseline_v1.parquet"
A = 0.0154            # m^2, Table 4.1 (diagnostic only)
LPM = 1 / 60000       # L/min -> m^3/s
SIGMA_S, Q_STEP = 1.5, 1.0   # mm, sensor model

df = pd.read_parquet(PARQUET)
print(f"rows in file: {len(df)}")
print("columns:", list(df.columns))

need = ["h1_m", "h2_m", "q_in_lpm", "q12_lpm", "q_out_lpm", "pump"]
missing = [c for c in need if c not in df.columns]
if missing:
    sys.exit(f"STOP: columns not found {missing} - paste the column list above")

if "seq" in df.columns:
    df = df[df["seq"] >= 74]          # closed loop begins at seq 74
df = df.reset_index(drop=True)
N = len(df)
print(f"closed-loop rows used: {N}")

h1 = df["h1_m"].to_numpy(float) * 1e3        # mm
h2 = df["h2_m"].to_numpy(float) * 1e3
qin = df["q_in_lpm"].to_numpy(float) * LPM   # m^3/s
q12 = df["q12_lpm"].to_numpy(float) * LPM
qout = df["q_out_lpm"].to_numpy(float) * LPM
pump = df["pump"].to_numpy().astype(int)

# ---- 1. lattice -------------------------------------------------------------
print("\n[1] fraction of levels on the 1 mm lattice (predicted 1.000000)")
for name, h in (("h1", h1), ("h2", h2)):
    print(f"    {name}: {np.mean(np.abs(h - np.round(h)) < 1e-6):.6f}")

# ---- 2/3. residual floor ----------------------------------------------------
sw = np.flatnonzero(np.diff(pump) != 0)      # pump changes between i and i+1
mask = np.ones(N, bool)                      # False near a transition
for k in (-1, 0, 1, 2):
    mask[np.clip(sw + k, 0, N - 1)] = False
mask[[0, -1]] = False                        # np.gradient edges are one-sided


def resid(h, q_a, q_b, how):
    dh = np.gradient(h) if how == "central" else np.diff(h, prepend=np.nan)
    return dh - (q_a - q_b) / A * 1e3        # mm/s at the nominal 1 s interval


var_n = SIGMA_S**2 + Q_STEP**2 / 12          # per-sample level noise variance
floor_c = np.sqrt(var_n / 2)
floor_f = np.sqrt(2 * var_n)
step = qin.max() / A * 1e3                   # jump in dh1/dt when the pump switches
t1_unmasked = np.sqrt(floor_c**2 + len(sw) * 2 * step**2 / 12 / N)

print(f"\n[2] pump transitions: {len(sw)}   pump step in dh1/dt: {step:.3f} mm/s")
print(f"    predicted central diff: {floor_c:.3f} (tank 2; tank 1 masked)"
      f"   {t1_unmasked:.3f} (tank 1 unmasked)")
print(f"    predicted first diff:   {floor_f:.3f}")
res = {}
for tank, args in (("r1", (h1, qin, q12)), ("r2", (h2, q12, qout))):
    for how in ("central", "first"):
        x = resid(*args, how)
        res[(tank, how)] = x
        inner = x[1:-1]
        print(f"    {tank} {how:7s}  sigma all {np.nanstd(inner):.3f}"
              f"   masked {np.nanstd(x[mask]):.3f}")

# ---- 4. autocorrelation -----------------------------------------------------


def acf(x, k):
    a, b = x[:-k], x[k:]
    ok = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[ok], b[ok])[0, 1]


print("\n[4] central-difference residual ACF, lags 1-4 (predicted +0.000 -0.500 +0.000 +0.000)")
for tank in ("r1", "r2"):
    x = res[(tank, "central")].copy()
    x[~mask] = np.nan
    print(f"    {tank}: " + " ".join(f"{acf(x, k):+.3f}" for k in range(1, 5)))

# ---- 5. smoothed floor versus slope ---------------------------------------
MED = int(sys.argv[1]) if len(sys.argv) > 1 else 5   # rolling-median width
print(f"\n[5] residual sigma by local slope, rolling median width {MED}"
      "  (predicted raw 1.08 in every bin; smoothed rising with slope)")
for tank, h, q_a, q_b in (("r1", h1, qin, q12), ("r2", h2, q12, qout)):
    hs = pd.Series(h).rolling(MED, center=True).median().to_numpy()
    r_s = np.gradient(hs) - (q_a - q_b) / A * 1e3
    r_c = res[(tank, "central")]
    slope = np.abs(np.roll(h, -5) - np.roll(h, 5)) / 10          # mm/s, 10 s span
    for lo, hi in ((0, 0.5), (0.5, 1.5), (1.5, 99)):
        b = mask & (slope >= lo) & (slope < hi)
        b[:MED + 5] = b[-(MED + 5):] = False
        if b.sum() < 500:
            continue
        print(f"    {tank} |slope| {lo:3.1f}-{hi if hi < 99 else 'max':>3} mm/s"
              f"  n={b.sum():6d}  raw {np.std(r_c[b]):.3f}  smoothed {np.nanstd(r_s[b]):.3f}")

# ---- stored residual columns ------------------------------------------------
stored = [c for c in df.columns
          if c.lower().startswith(("r1", "r2")) or "resid" in c.lower()]
print("\n[6] residual columns already in the parquet (units as stored)")
if not stored:
    print("    none found by name - paste the column list and I'll point at them")
for c in stored:
    x = df[c].to_numpy(float)
    print(f"    {c:28s} sigma {np.nanstd(x):.4g}   masked {np.nanstd(x[mask]):.4g}")
