#!/usr/bin/env python3
"""Hybrid AI-Quantum Cognitive Security Framework Phase 1 - build the versioned benign-baseline dataset.

Consolidates the rotated CSVs written by fog/ingest/baseline_recorder.py into a
single Parquet file, adds the derived physics quantities that Phase 3 needs,
runs quality checks, and emits a data card.

Derived columns
---------------
    dt            seconds since previous sample
    dh1_dt, dh2_dt   numerical level derivatives (central difference), m/s
    res1, res2    mass-balance residuals, m/s:
                      res1 = dh1/dt - (q_in - q12)/A1
                      res2 = dh2/dt - (q12 - q_out)/A2
                  For benign operation these are pure sensor noise; they are the
                  raw material of the physics score S_phys in PN-Fusion.
    cycle         fill/drain cycle index, incremented on each pump rising edge
    label         0 = benign (Phase 1 contains benign data only)

    python scripts/make_baseline_dataset.py --config configs/twin.yaml \
           --version v1 --split 0.7 0.15 0.15
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

LPM_TO_M3S = 1.0 / 60000.0


def load_raw(indir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(indir / "baseline_*.csv")))
    if not files:
        raise SystemExit(f"no baseline_*.csv found in {indir} - run the recorder first")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.sort_values("ts").drop_duplicates(subset=["seq", "src"]).reset_index(drop=True)
    print(f"loaded {len(df):,} rows from {len(files)} file(s)")
    return df


def add_derived(df: pd.DataFrame, plant: dict, smooth: int = 5) -> pd.DataFrame:
    """Add derivatives, mass-balance residuals, cycle index and label.

    The level transducer is quantised to 1 mm, so differentiating the raw signal
    amplifies quantisation noise by 1/dt. A centred rolling median of `smooth`
    samples is therefore applied before differentiation; both the raw and the
    smoothed residuals are retained so that Phase 3 can justify its choice.
    """
    A1, A2 = float(plant["A1"]), float(plant["A2"])
    t = df["ts"].to_numpy()
    df["dt"] = df["ts"].diff()

    df["h1_s"] = df["h1_m"].rolling(smooth, center=True, min_periods=1).median()
    df["h2_s"] = df["h2_m"].rolling(smooth, center=True, min_periods=1).median()
    df["dh1_dt"] = np.gradient(df["h1_m"].to_numpy(), t)
    df["dh2_dt"] = np.gradient(df["h2_m"].to_numpy(), t)
    df["dh1_dt_s"] = np.gradient(df["h1_s"].to_numpy(), t)
    df["dh2_dt_s"] = np.gradient(df["h2_s"].to_numpy(), t)

    q_in = df["q_in_lpm"] * LPM_TO_M3S
    q12 = df["q12_lpm"] * LPM_TO_M3S
    q_out = df["q_out_lpm"] * LPM_TO_M3S
    model1 = (q_in - q12) / A1
    model2 = (q12 - q_out) / A2
    df["res1"] = df["dh1_dt"] - model1          # raw residual (noise dominated)
    df["res2"] = df["dh2_dt"] - model2
    df["res1_s"] = df["dh1_dt_s"] - model1      # smoothed residual (use this)
    df["res2_s"] = df["dh2_dt_s"] - model2

    df["cycle"] = (df["pump"].diff() == 1).cumsum().fillna(0).astype(int)
    df["label"] = 0
    return df


def quality_report(df: pd.DataFrame, expected_hz: float) -> dict:
    dt = df["dt"].dropna()
    span = df["ts"].iloc[-1] - df["ts"].iloc[0]
    gaps = int((dt > 3.0 / expected_hz).sum())
    rep = {
        "rows": int(len(df)),
        "duration_s": round(float(span), 1),
        "duration_h": round(float(span) / 3600.0, 3),
        "effective_hz": round(len(df) / span, 4) if span > 0 else None,
        "expected_hz": expected_hz,
        "sample_gaps_gt_3T": gaps,
        "seq_monotonic": bool(df["seq"].is_monotonic_increasing),
        "nan_cells": int(df.isna().sum().sum() - df["dt"].isna().sum()),
        "h1_m": {"min": float(df.h1_m.min()), "mean": float(df.h1_m.mean()),
                 "max": float(df.h1_m.max())},
        "h2_m": {"min": float(df.h2_m.min()), "mean": float(df.h2_m.mean()),
                 "max": float(df.h2_m.max())},
        "pump_duty": round(float(df["pump"].mean()), 4),
        "alarm_frames": int(df["alarm"].sum()),
        "cycles": int(df["cycle"].max()),
        "res1_std_raw": float(df["res1"].std()),
        "res2_std_raw": float(df["res2"].std()),
        "res1_std_smoothed": float(df["res1_s"].std()),
        "res2_std_smoothed": float(df["res2_s"].std()),
    }
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/twin.yaml")
    ap.add_argument("--indir", default=None)
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing dataset of the same version")
    ap.add_argument("--split", nargs=3, type=float, default=[0.7, 0.15, 0.15],
                    help="chronological train/val/test fractions")
    ap.add_argument("--smooth", type=int, default=5,
                    help="rolling-median window applied before differentiation")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    indir = Path(a.indir or cfg["storage"]["baseline_dir"])
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = add_derived(load_raw(indir), cfg["plant"], smooth=a.smooth)
    rep = quality_report(df, float(cfg["mqtt"]["publish_hz"]))

    # chronological split - never shuffle time series across the split boundary
    n = len(df)
    i1, i2 = int(a.split[0] * n), int((a.split[0] + a.split[1]) * n)
    df["split"] = np.where(df.index < i1, "train", np.where(df.index < i2, "val", "test"))

    out = outdir / f"baseline_{a.version}.parquet"
    if out.exists() and not a.force:
        raise SystemExit(
            f"{out} already exists - refusing to overwrite a versioned dataset.\n"
            f"Pass --force to replace it, or use a different --version."
        )
    df.to_parquet(out, index=False)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()

    card = {
        "name": f"cps_baseline_{a.version}",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 1,
        "description": "Benign-operation telemetry from the two-tank digital twin "
                       "under closed-loop OpenPLC control. Contains no attacks.",
        "source": {"twin": "edge/twin/TwoTank.mo", "config": str(a.config),
                   "raw_dir": str(indir)},
        "plant_parameters": cfg["plant"],
        "control": cfg["control"],
        "sensors": cfg["sensors"],
        "schema": {c: str(t) for c, t in df.dtypes.items()},
        "smoothing_window": a.smooth,
        "split_fractions": {"train": a.split[0], "val": a.split[1], "test": a.split[2]},
        "quality": rep,
        "artefact": {"file": out.name, "sha256": sha,
                     "bytes": out.stat().st_size},
        "intended_use": ["RO1 generative training and C2ST reference distribution",
                         "PN-Fusion physics-signal normal model",
                         "false-alarm-rate measurement under benign conditions"],
        "limitations": ["single operating point and set-point pair",
                        "simulated plant with Gaussian sensor noise only",
                        "no process disturbances, drift or maintenance events"],
    }
    (outdir / f"baseline_{a.version}_datacard.json").write_text(json.dumps(card, indent=2))

    print(json.dumps(rep, indent=2))
    print(f"\nwrote {out}  ({out.stat().st_size/1e6:.2f} MB, sha256 {sha[:12]}...)")
    print(f"wrote {outdir / f'baseline_{a.version}_datacard.json'}")
    print("\nNext:  dvc add", out, "&&  git add", str(out) + ".dvc")


if __name__ == "__main__":
    main()
