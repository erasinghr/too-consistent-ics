#!/usr/bin/env python
"""Phase 3, step 3.1b diagnostic - where do the physics arms' alarms on replay come from?

Corpus v1 (3.1b) found the arms alarming on replay runs more often than with no attack
(Arm A 18/160, Arm B 15/160, against 7/200 in the 3.1a no-attack control). Replay serves
genuine recorded telemetry, so the only new material in the replayed stream is the
splices, where one recorded segment is joined to another. Hypothesis: the alarms come
from the splices, where a falsified stream must stay continuous across its joins.

Method. The replay runs of the corpus are regenerated exactly (same seeds and protocol,
read from the corpus), which also yields the splice times the corpus does not store.
Regeneration is checked against runs.parquet: every run's Arm A and Arm B delay must match.
For each run and arm the first alarm is placed relative to the most recent splice. Under
the null "alarms are unrelated to splices", an alarm falls within D s after a splice with
probability equal to the fraction of that run's post-onset time lying within D s after a
splice; the expected count and the exact Poisson-binomial tail P(X >= observed) are
reported for D = 10, 60, 120 and 300 s (the C, V/P, T and M time scales).

    python scripts/phase3_replay_seams.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_attacks import score_run  # noqa: E402
from phase2_detector import load_detector, statistics  # noqa: E402
from phase2_sim import simulate_iter  # noqa: E402
from phase3_attack_classes import CLASSES, make as make_p3  # noqa: E402

WINDOWS = (10, 60, 120, 300)


def poisson_binomial_tail(probs: list[float], k: int) -> float:
    """P(X >= k) for X a sum of independent Bernoulli(p_i)."""
    dist = np.zeros(len(probs) + 1)
    dist[0] = 1.0
    for p in probs:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= (1 - p)
    return float(dist[k:].sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/processed/phase3_corpus_v1")
    ap.add_argument("--config", default="configs/twin.yaml")
    ap.add_argument("--detector", default="reports/phase2_detector_calibration.json")
    ap.add_argument("--rungs", default="P3-replay,P3-replay-drain")
    ap.add_argument("--out", default="reports/phase3_replay_seams.json")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    plant, hmax = cfg["plant"], float(cfg["plant"]["h_max"])
    thr, params = load_detector(a.detector)
    p = {k: params[k] for k in ("window", "mean_window", "alg_window", "k", "mask_pad")}
    card = json.loads((Path(a.corpus) / "datacard.json").read_text())
    n = int(card["protocol"]["runs_per_phase"])
    seconds = int(card["protocol"]["record_s"])
    runs = pd.read_parquet(Path(a.corpus) / "runs.parquet")

    rows, mismatches = [], 0
    for name in [x.strip() for x in a.rungs.split(",")]:
        spec = CLASSES[name]
        for phase in (1, 0):
            ph = "ON" if phase else "OFF"
            grp = runs[(runs["rung"] == name) & (runs["phase"] == ph)].set_index("run")
            seed = int(grp["seed"].iloc[0])
            t0 = np.random.default_rng(seed + 7).uniform(600, 900, n)
            atk = make_p3(spec, cfg, n, t0, phase)
            for r, df in enumerate(simulate_iter(cfg, n, seconds, seed, attack=atk)):
                onset = float(atk.onset[r])
                if not math.isfinite(onset) or r not in grp.index:
                    continue
                sc = score_run(df, onset, statistics(df, plant, p), thr, hmax)
                st = grp.loc[r]
                same = all((sc[k] == st[k]) or (math.isinf(sc[k]) and math.isinf(st[k]))
                           for k in ("delay_A", "delay_B"))
                mismatches += 0 if same else 1
                i0 = int(math.ceil(onset))
                post = np.arange(i0, len(df), dtype=float)
                sp = np.array([s[0] for s in atk.splices[r]])
                # seconds since the most recent splice, for every post-onset second
                idx = np.searchsorted(sp, post, side="right") - 1
                since = np.where(idx >= 0, post - sp[np.maximum(idx, 0)], np.inf)
                frac = {D: float(np.mean(since < D)) for D in WINDOWS}
                row = {"rung": name, "phase": ph, "run": r, "seed": seed, "onset_s": round(onset, 1),
                       "splices": len(sp), "max_jump_h1_mm": int(max(abs(s[1]) for s in atk.splices[r])),
                       "max_jump_h2_mm": int(max(abs(s[2]) for s in atk.splices[r])),
                       "frac_within": frac, "match_corpus": bool(same)}
                for arm in ("A", "B"):
                    d = sc[f"delay_{arm}"]
                    if math.isfinite(d):
                        ta = i0 + d
                        j = np.searchsorted(sp, ta, side="right") - 1
                        row[f"alarm_{arm}"] = {
                            "delay_s": d, "first": sc[f"first_{arm}"],
                            "since_splice_s": round(float(ta - sp[j]), 1) if j >= 0 else None,
                            "splices_before": int(j + 1), "harm_s": sc["t_HH"] if spec["force"] else sc["t_dry"]}
                rows.append(row)
        print(f"  {name} regenerated", flush=True)

    print(f"\nregeneration matches corpus runs.parquet: {len(rows) - mismatches}/{len(rows)} runs")
    summary = {}
    for arm in ("A", "B"):
        det = [x for x in rows if f"alarm_{arm}" in x]
        comps = {}
        for x in det:
            comps[x[f"alarm_{arm}"]["first"]] = comps.get(x[f"alarm_{arm}"]["first"], 0) + 1
        s = {"detected": len(det), "runs": len(rows),
             "first_component": dict(sorted(comps.items(), key=lambda kv: -kv[1])),
             "median_delay_s": float(np.median([x[f"alarm_{arm}"]["delay_s"] for x in det])) if det else None,
             "median_since_splice_s": float(np.median([x[f"alarm_{arm}"]["since_splice_s"] for x in det])) if det else None,
             "near_splice": {}}
        for D in WINDOWS:
            obs = sum(1 for x in det if x[f"alarm_{arm}"]["since_splice_s"] is not None
                      and x[f"alarm_{arm}"]["since_splice_s"] < D)
            probs = [x["frac_within"][D] for x in det]
            s["near_splice"][D] = {"observed": obs, "expected_if_unrelated": round(float(sum(probs)), 2),
                                   "p_tail": round(poisson_binomial_tail(probs, obs), 4)}
        summary[arm] = s
        print(f"\nArm {arm}: {s['detected']}/{s['runs']} replay runs alarmed; median delay {s['median_delay_s']} s; "
              f"median time since last splice {s['median_since_splice_s']} s")
        print("    first component: " + ", ".join(f"{k} {v}" for k, v in s["first_component"].items()))
        print(f"    {'alarm within D s after a splice':<34}{'observed':>9}{'expected':>10}{'P(X>=obs)':>11}")
        for D, v in s["near_splice"].items():
            print(f"    {'D = ' + str(D) + ' s':<34}{v['observed']:>9}{v['expected_if_unrelated']:>10}{v['p_tail']:>11}")
        print(f"    {'rung':<17}{'ph':<4}{'run':>4}{'delay s':>9}{'since splice s':>16}{'splices before':>16}   first")
        for x in det:
            al = x[f"alarm_{arm}"]
            print(f"    {x['rung']:<17}{x['phase']:<4}{x['run']:>4}{al['delay_s']:>9.0f}"
                  f"{al['since_splice_s'] if al['since_splice_s'] is not None else '-':>16}{al['splices_before']:>16}   {al['first']}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "step": "phase3 3.1b diagnostic: replay alarms relative to splices",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus": a.corpus, "rungs": a.rungs, "regeneration_mismatches": mismatches,
        "summary": summary, "runs": rows}, indent=2, default=lambda v: None if isinstance(v, float) and not math.isfinite(v) else float(v)))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
