#!/usr/bin/env python
"""Phase 3, step 3.1b - the Phase 3 corpus: every attack class and benign operation, as
whole simulated runs, split by run, with the Phase 2 physics arms' verdict on each run.

Why runs and not windows: the generators train on 60 s windows, but the leave-one-attack-
out evaluation has to replay a detector over each run in time order to measure detection
delay and whether detection came before harm. So the corpus stores the 1 Hz telemetry of
whole runs, exactly as the simulator produced it (integer registers, lossless), and later
steps cut windows from it. One code path (frame() + add_derived) turns stored rows back
into the schema of baseline_v1.parquet.

Contents
  attack runs   11 rungs = the 7 Phase 2 rungs + the 4 Phase 3 classes, --runs-per-phase
                runs per onset phase (pump ON / OFF at onset), arming 600-900 s, horizon
                --horizon s after onset (same protocol as phase2_attacks / 3.1a, new seeds)
  benign days   --benign-days simulated days with no attack (same simulator, own seeds)
Families (the leave-one-attack-out unit):
  step R0 | ramp R1-drift, R1-creep | fab_clean R2 (+drain) | fab_matched R3 (+drain)
  freeze | scale | replay (+drain).  fab_matched and replay are covert to the physics arms
  (Phase 2, 3.1a) and are the negative controls.
Split, fixed before any model sees the data: within each rung and onset phase, runs are
shuffled with a fixed seed and assigned 60 % train / 20 % val / 20 % test; benign days the
same, by day. A run's pre-onset (benign) part stays in that run's split.

Files (directory --out, tracked by DVC as one directory):
  telemetry.parquet   one row per second per run: run_id, t, active (1 after onset),
                      h1_mm, h2_mm, q_in_clpm, q12_clpm, q_out_clpm (the registers the
                      telemetry saw), pump, alarm, pump_applied, h1_true, h2_true (m)
  runs.parquet        one row per run: identity, family, split, onset, Arms A/B verdict
                      (delay, first component, pre-onset false alarm), harm times
  datacard.json       provenance, seeds, counts, schema, sha256 of both parquet files
plus reports/phase3_corpus_v1_summary.json (small, for Git).

The build ends with a round-trip check: one stored run per rung is read back through
frame(), rescored, and must give the stored Arm A and Arm B delays exactly.

    python scripts/phase3_corpus.py --build                      # ~10 min
    python scripts/phase3_corpus.py --verify                     # re-check an existing corpus

Later steps:  from phase3_corpus import load, frame
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
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_baseline_dataset import add_derived  # noqa: E402
from phase2_attacks import RUNGS, make_attack, score_run  # noqa: E402
from phase2_detector import arm_alarm, bool_blocks, episodes, load_detector, statistics  # noqa: E402
from phase2_sim import simulate_iter  # noqa: E402
from phase3_attack_classes import CLASSES, make as make_p3  # noqa: E402

FAMILY = {
    "R0-step": "step",
    "R1-drift": "ramp", "R1-creep": "ramp",
    "R2-fab-clean": "fab_clean", "R2-fab-clean-drain": "fab_clean",
    "R3-fab-matched": "fab_matched", "R3-fab-matched-drain": "fab_matched",
    "P3-freeze": "freeze",
    "P3-scale": "scale",
    "P3-replay": "replay", "P3-replay-drain": "replay",
}
COVERT = ("fab_matched", "replay")
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)
TEL_SCHEMA = pa.schema([
    ("run_id", pa.int32()), ("t", pa.int32()), ("active", pa.int8()),
    ("h1_mm", pa.int16()), ("h2_mm", pa.int16()),
    ("q_in_clpm", pa.int16()), ("q12_clpm", pa.int16()), ("q_out_clpm", pa.int16()),
    ("pump", pa.int8()), ("alarm", pa.int8()), ("pump_applied", pa.int8()),
    ("h1_true", pa.float32()), ("h2_true", pa.float32()),
])


# ---------------------------------------------------------------- reading back (used by later steps)
def load(corpus: str | Path = "data/processed/phase3_corpus_v1", columns=None, filters=None):
    """(runs, telemetry) DataFrames. `filters` is passed to pyarrow, e.g. [("run_id", "in", ids)]."""
    corpus = Path(corpus)
    runs = pd.read_parquet(corpus / "runs.parquet")
    tel = pd.read_parquet(corpus / "telemetry.parquet", columns=columns, filters=filters)
    return runs, tel


def frame(rows: pd.DataFrame, plant: dict) -> pd.DataFrame:
    """Rows of ONE run (from telemetry.parquet) -> the schema of baseline_v1.parquet, derived
    columns recomputed by the released pipeline (add_derived), as phase2_sim produces it."""
    rows = rows.sort_values("t")
    t = rows["t"].to_numpy()
    df = pd.DataFrame({
        "seq": t.astype(np.int64), "ts": t.astype(float),
        "h1_m": rows["h1_mm"].to_numpy(np.int32) / 1000.0, "h2_m": rows["h2_mm"].to_numpy(np.int32) / 1000.0,
        "q_in_lpm": rows["q_in_clpm"].to_numpy(np.int32) / 100.0,
        "q12_lpm": rows["q12_clpm"].to_numpy(np.int32) / 100.0,
        "q_out_lpm": rows["q_out_clpm"].to_numpy(np.int32) / 100.0,
        "pump": rows["pump"].to_numpy(np.int32), "valve": 1, "alarm": rows["alarm"].to_numpy(np.int32),
        "h1_true": rows["h1_true"].to_numpy(np.float32), "h2_true": rows["h2_true"].to_numpy(np.float32),
        "pump_applied": rows["pump_applied"].to_numpy(np.int32),
    })
    return add_derived(df, plant)


# ---------------------------------------------------------------- helpers
def to_table(df: pd.DataFrame, run_id: int, onset: float | None) -> pa.Table:
    n = len(df)
    t = np.arange(n, dtype=np.int32)
    active = np.zeros(n, np.int8) if onset is None else (t >= math.ceil(onset)).astype(np.int8)

    def reg(col, scale):
        v = np.rint(df[col].to_numpy() * scale)
        assert v.min() >= -32768 and v.max() <= 32767, f"{col} out of int16 range"
        return v.astype(np.int16)

    return pa.table({
        "run_id": np.full(n, run_id, np.int32), "t": t, "active": active,
        "h1_mm": reg("h1_m", 1000.0), "h2_mm": reg("h2_m", 1000.0),
        "q_in_clpm": reg("q_in_lpm", 100.0), "q12_clpm": reg("q12_lpm", 100.0),
        "q_out_clpm": reg("q_out_lpm", 100.0),
        "pump": df["pump"].to_numpy().astype(np.int8), "alarm": df["alarm"].to_numpy().astype(np.int8),
        "pump_applied": df["pump_applied"].to_numpy().astype(np.int8),
        "h1_true": df["h1_true"].to_numpy().astype(np.float32),
        "h2_true": df["h2_true"].to_numpy().astype(np.float32),
    }, schema=TEL_SCHEMA)


def assign_splits(n: int, seed: int) -> list[str]:
    order = np.random.default_rng(seed).permutation(n)
    n_tr = int(round(SPLIT_FRACTIONS[0] * n))
    n_va = int(round(SPLIT_FRACTIONS[1] * n))
    lab = np.empty(n, object)
    lab[order[:n_tr]] = "train"
    lab[order[n_tr:n_tr + n_va]] = "val"
    lab[order[n_tr + n_va:]] = "test"
    return lab.tolist()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def spec_of(rung: str) -> tuple[dict, str]:
    return (RUNGS[rung], "phase2") if rung in RUNGS else (CLASSES[rung], "phase3")


# ---------------------------------------------------------------- build
def build(a) -> None:
    cfg = yaml.safe_load(open(a.config))
    plant = cfg["plant"]
    thr, params = load_detector(a.detector)
    p = {k: params[k] for k in ("window", "mean_window", "alg_window", "k", "mask_pad")}
    hmax = float(plant["h_max"])
    out = Path(a.out)
    if out.exists() and any(out.iterdir()) and not a.force:
        raise SystemExit(f"{out} exists and is not empty - refusing to overwrite a versioned corpus "
                         f"(pass --force, or use another --out)")
    out.mkdir(parents=True, exist_ok=True)
    rungs = list(FAMILY) if a.rungs == "all" else [x.strip() for x in a.rungs.split(",")]
    seconds = 900 + 120 + a.horizon

    writer = pq.ParquetWriter(out / "telemetry.parquet", TEL_SCHEMA, compression="zstd")
    runs, run_id = [], 0
    t_start = datetime.now()
    try:
        # ---- attack runs
        for name in rungs:
            spec, origin = spec_of(name)
            ri = list(FAMILY).index(name)
            drain_force = spec.get("force") == 0.0
            for pi, phase in enumerate((1, 0)):
                seed = a.seed_attack + 1000 * ri + pi
                t0 = np.random.default_rng(seed + 7).uniform(600, 900, a.runs_per_phase)
                atk = (make_attack(spec, cfg, a.runs_per_phase, t0, phase) if origin == "phase2"
                       else make_p3(spec, cfg, a.runs_per_phase, t0, phase))
                splits = assign_splits(a.runs_per_phase, seed + 11)
                drain = drain_force or (spec["kind"] == "freeze" and phase == 0)
                for r, df in enumerate(simulate_iter(cfg, a.runs_per_phase, seconds, seed, attack=atk)):
                    onset = float(atk.onset[r])
                    if not math.isfinite(onset):
                        print(f"    {name} phase {phase} run {r}: onset never reached - skipped", flush=True)
                        continue
                    st = statistics(df, plant, p)
                    sc = score_run(df, onset, st, thr, hmax)
                    harm = sc["t_dry"] if drain else sc["t_HH"]
                    writer.write_table(to_table(df, run_id, onset))
                    runs.append({
                        "run_id": run_id, "kind": "attack", "rung": name, "family": FAMILY[name],
                        "covert_expected": FAMILY[name] in COVERT, "origin": origin,
                        "phase": "ON" if phase else "OFF", "seed": seed, "run": r, "split": splits[r],
                        "onset_s": onset, "harm_metric": "t_dry" if drain else "t_HH", "harm_s": harm,
                        "before_harm_A": bool(sc["delay_A"] < harm), "before_harm_B": bool(sc["delay_B"] < harm),
                        "n_rows": len(df), **sc,
                    })
                    run_id += 1
                print(f"  {name:<22} pump {'ON ' if phase else 'OFF'} {a.runs_per_phase} runs "
                      f"({(datetime.now() - t_start).total_seconds() / 60:.1f} min)", flush=True)

        # ---- benign days
        splits = assign_splits(a.benign_days, a.seed_benign + 11)
        fa = {"A": 0, "B": 0}
        for d, df in enumerate(simulate_iter(cfg, a.benign_days, 86400, a.seed_benign)):
            st = statistics(df, plant, p)
            ep = {arm: episodes(bool_blocks(arm_alarm(st, thr, arm))) for arm in ("A", "B")}
            fa["A"] += ep["A"]
            fa["B"] += ep["B"]
            writer.write_table(to_table(df, run_id, None))
            runs.append({
                "run_id": run_id, "kind": "benign", "rung": "benign", "family": "benign",
                "covert_expected": False, "origin": "phase3", "phase": None, "seed": a.seed_benign,
                "run": d, "split": splits[d], "onset_s": np.nan, "harm_metric": None, "harm_s": np.inf,
                "before_harm_A": False, "before_harm_B": False, "n_rows": len(df),
                "fa_episodes_A": ep["A"], "fa_episodes_B": ep["B"],
                "pump_duty": float(df["pump"].mean()),
                "res1_sigma_mm_s": float(np.nanstd(df["res1"].to_numpy()) * 1000.0),
            })
            run_id += 1
        print(f"  benign days            {a.benign_days} days "
              f"({(datetime.now() - t_start).total_seconds() / 60:.1f} min)", flush=True)
    finally:
        writer.close()

    rdf = pd.DataFrame(runs)
    rdf.to_parquet(out / "runs.parquet", index=False)

    # ---- round-trip check: stored rows -> frame() -> rescore == stored verdict
    rt = roundtrip(out, cfg, thr, p, rdf)

    # ---- summary
    att = rdf[rdf["kind"] == "attack"]
    summary = {}
    print(f"\n{'rung':<22}{'family':<12}{'runs':>5}{'tr/va/te':>11}{'Arm A det':>11}{'Arm B det':>11}"
          f"{'A<harm':>8}{'B<harm':>8}{'med delay B s':>15}")
    for name in rungs:
        g = att[att["rung"] == name]
        if g.empty:
            continue
        sp = g["split"].value_counts()
        dA, dB = g["delay_A"].to_numpy(float), g["delay_B"].to_numpy(float)
        finB = dB[np.isfinite(dB)]
        row = {"family": FAMILY[name], "runs": int(len(g)),
               "split": {k: int(sp.get(k, 0)) for k in ("train", "val", "test")},
               "detected_A": int(np.isfinite(dA).sum()), "detected_B": int(np.isfinite(dB).sum()),
               "before_harm_A": int(g["before_harm_A"].sum()), "before_harm_B": int(g["before_harm_B"].sum()),
               "median_delay_B_s": float(np.median(finB)) if finB.size else None}
        summary[name] = row
        med = "-" if row["median_delay_B_s"] is None else f"{row['median_delay_B_s']:.0f}"
        print(f"{name:<22}{FAMILY[name]:<12}{row['runs']:>5}"
              f"{row['split']['train']:>5}/{row['split']['val']}/{row['split']['test']:<3}"
              f"{row['detected_A']:>7}/{row['runs']:<3}{row['detected_B']:>7}/{row['runs']:<3}"
              f"{row['before_harm_A']:>8}{row['before_harm_B']:>8}{med:>15}")
    ben = rdf[rdf["kind"] == "benign"]
    chance = a.runs_per_phase * 2 * (1 - math.exp(-float(params.get("far_per_day", 1.0)) * a.horizon / 86400))
    print(f"(chance: about {chance:.1f} of {a.runs_per_phase * 2} runs per rung alarm within the horizon with no attack)")
    print(f"\nbenign days: {len(ben)} ({', '.join(f'{k} {v}' for k, v in ben['split'].value_counts().items())}); "
          f"false-alarm episodes Arm A {int(ben['fa_episodes_A'].sum())}, Arm B {int(ben['fa_episodes_B'].sum())} "
          f"({len(ben) * float(params.get('far_per_day', 1.0)):.0f} expected per arm); "
          f"pump duty {ben['pump_duty'].mean():.4f}; res1 sigma {ben['res1_sigma_mm_s'].mean():.3f} mm/s")
    print(f"round-trip check ({len(rt)} runs, stored vs rescored Arm A/B delay): "
          f"{'PASS' if all(x['match'] for x in rt) else 'FAIL'}")

    tel_path, runs_path = out / "telemetry.parquet", out / "runs.parquet"
    card = {
        "name": "phase3_corpus_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "phase": 3, "step": "3.1b", "git_head": git_head(),
        "config": a.config, "config_md5": md5(Path(a.config)),
        "detector": a.detector, "detector_md5": md5(Path(a.detector)),
        "protocol": {"runs_per_phase": a.runs_per_phase, "onset_phases": ["ON", "OFF"],
                     "arming_s": [600, 900], "horizon_s": a.horizon, "record_s": seconds,
                     "seed_attack": f"{a.seed_attack} + 1000 * rung index + phase index (0 ON, 1 OFF); "
                                    f"arming rng seed + 7; split rng seed + 11",
                     "benign_days": a.benign_days, "seed_benign": a.seed_benign},
        "rungs": {n: {"family": FAMILY[n], "spec": spec_of(n)[0], "origin": spec_of(n)[1]} for n in rungs},
        "covert_families": list(COVERT),
        "split": {"fractions": dict(zip(("train", "val", "test"), SPLIT_FRACTIONS)),
                  "unit": "run (attack runs within rung x onset phase; benign by day)"},
        "rows": {"telemetry": int(rdf["n_rows"].sum()), "runs": int(len(rdf)),
                 "attack_runs": int(len(att)), "benign_days": int(len(ben))},
        "schema": {"telemetry": {f.name: str(f.type) for f in TEL_SCHEMA},
                   "runs": {c: str(t) for c, t in rdf.dtypes.items()}},
        "files": {"telemetry.parquet": {"sha256": sha256(tel_path), "bytes": tel_path.stat().st_size},
                  "runs.parquet": {"sha256": sha256(runs_path), "bytes": runs_path.stat().st_size}},
        "roundtrip": rt,
        "intended_use": ["3.2-3.5 generator training and C2ST reference (train/val runs)",
                         "3.6 leave-one-attack-out evaluation (test runs), physics arms as comparator"],
        "limitations": ["all runs simulated (phase2_sim, validated against the released day on 29 statistics); "
                        "the three tank-2 features of the real stack the simulator lacks are absent",
                        "source spoofs only; one operating point and set-point pair",
                        "no network data: the corpus covers the telemetry channel only"],
    }
    (out / "datacard.json").write_text(json.dumps(card, indent=2, default=float))
    rep = Path("reports/phase3_corpus_v1_summary.json")
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps({"datacard": card, "per_rung": summary,
                               "benign": {"days": int(len(ben)),
                                          "fa_episodes_A": int(ben["fa_episodes_A"].sum()),
                                          "fa_episodes_B": int(ben["fa_episodes_B"].sum())}},
                              indent=2, default=float))
    print(f"\nwrote {tel_path} ({tel_path.stat().st_size / 1e6:.1f} MB), {runs_path}, {out / 'datacard.json'}")
    print(f"wrote {rep}")
    print(f"total {(datetime.now() - t_start).total_seconds() / 60:.1f} min")


def roundtrip(out: Path, cfg: dict, thr: dict, p: dict, rdf: pd.DataFrame) -> list[dict]:
    """Read one stored run per rung back through frame(), rescore, compare with the stored verdict."""
    plant, hmax = cfg["plant"], float(cfg["plant"]["h_max"])
    att = rdf[rdf["kind"] == "attack"]
    pick = att.groupby("rung", sort=False).head(1)
    ids = [int(x) for x in pick["run_id"]]
    _, tel = load(out, filters=[("run_id", "in", ids)])
    res = []
    for _, row in pick.iterrows():
        df = frame(tel[tel["run_id"] == row["run_id"]], plant)
        sc = score_run(df, float(row["onset_s"]), statistics(df, plant, p), thr, hmax)
        match = all((sc[k] == row[k]) or (math.isinf(sc[k]) and math.isinf(row[k])) for k in ("delay_A", "delay_B"))
        res.append({"rung": row["rung"], "run_id": int(row["run_id"]), "match": bool(match),
                    "delay_A": sc["delay_A"], "delay_B": sc["delay_B"]})
    return res


def verify(a) -> None:
    out = Path(a.out)
    card = json.loads((out / "datacard.json").read_text())
    ok = True
    for name, meta in card["files"].items():
        got = sha256(out / name)
        same = got == meta["sha256"]
        ok &= same
        print(f"{name:<20} sha256 {'OK' if same else 'MISMATCH'}  {got[:16]}...")
    cfg = yaml.safe_load(open(a.config))
    thr, params = load_detector(a.detector)
    p = {k: params[k] for k in ("window", "mean_window", "alg_window", "k", "mask_pad")}
    rdf = pd.read_parquet(out / "runs.parquet")
    rt = roundtrip(out, cfg, thr, p, rdf)
    ok &= all(x["match"] for x in rt)
    print(f"round-trip ({len(rt)} runs): {'PASS' if all(x['match'] for x in rt) else 'FAIL'}")
    print("VERIFY PASS" if ok else "VERIFY FAIL")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--config", default="configs/twin.yaml")
    ap.add_argument("--detector", default="reports/phase2_detector_calibration.json")
    ap.add_argument("--out", default="data/processed/phase3_corpus_v1")
    ap.add_argument("--rungs", default="all", help=f"comma list or 'all' ({', '.join(FAMILY)})")
    ap.add_argument("--runs-per-phase", type=int, default=40)
    ap.add_argument("--horizon", type=int, default=3600)
    ap.add_argument("--benign-days", type=int, default=20)
    ap.add_argument("--seed-attack", type=int, default=202611170)
    ap.add_argument("--seed-benign", type=int, default=909111700)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.build:
        build(a)
    elif a.verify:
        verify(a)
    else:
        ap.error("pass --build or --verify")


if __name__ == "__main__":
    main()
