#!/usr/bin/env python
"""Phase 3, step 3.1a - three new attack classes on the closed-loop simulator, scored by
both physics arms exactly as the Phase 2 rungs were (scripts/phase2_attacks.py).

Why: the leave-one-attack-out test of RQ1 needs more attack mechanisms than the Phase 2
ladder has (step, ramp, clean fabrication, matched fabrication). Before a new class
enters the Phase 3 corpus it is characterised by the calibrated Phase 2 detector, so the
corpus records which classes the physics arms can and cannot see.

All are SOURCE spoofs (the PLC and the telemetry read the same falsified registers), as
in Phase 2:
  P3-freeze        the h1 register is frozen at its value at onset (stuck-at). The PLC
                   keeps the pump as it was: overflow if ON at onset, dry-out if OFF.
  P3-scale         the h1 register is multiplied by a factor falling linearly from 1 to
                   0.6 over 600 s. A reported band of 250-350 mm then means a true band
                   of about 417-583 mm, above the 550 mm interlock level.
  P3-replay        every input register is replayed from the attacker's own recording
                   of the run before onset, tick by tick (what a man-in-the-middle sees
                   of the PLC's polls). It is spliced in where the recorded pump command
                   equals the PLC's and the recorded levels are closest to the live
                   readings; when the recording runs out it is re-spliced the same way.
                   The real pump is forced ON. No plant model is used, only the recording.
  P3-replay-drain  the same with the real pump forced OFF.

  P3-null          no attack: onset is marked exactly as for the others, nothing is
                   changed. Run on its own (--classes P3-null) it measures how often the
                   arms alarm by chance under this very protocol, the empirical baseline
                   against which "chance level" is judged.

Prediction, stated before the run: P3-replay carries genuine sensor noise and physically
consistent values, so both physics arms should stay at the chance rate, as for R3,
although this attacker needs no plant model. P3-freeze and P3-scale should be caught.

Thresholds come from reports/phase2_detector_calibration.json; nothing is recalibrated.

    python scripts/phase3_attack_classes.py                     # 4 classes x 2 phases x 25 runs
    python scripts/phase3_attack_classes.py --classes P3-replay --runs 5
    python scripts/phase3_attack_classes.py --classes P3-null --runs 100 --out reports/phase3_null_control.json

Seeds depend on the class's position in CLASSES, not on the selection, so a class run on
its own reproduces its rows of the full run exactly.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_attacks import Attack, fmt_s, score_run, summarise  # noqa: E402
from phase2_detector import load_detector, statistics  # noqa: E402
from phase2_sim import simulate_iter  # noqa: E402

REGS = ("h1", "h2", "qin", "q12", "qout")
CLASSES = {
    "P3-freeze":       {"kind": "freeze"},
    "P3-scale":        {"kind": "scale", "k_final": 0.6, "ramp_s": 600.0},
    "P3-replay":       {"kind": "replay", "force": 1.0, "min_seg_s": 60.0},
    "P3-replay-drain": {"kind": "replay", "force": 0.0, "min_seg_s": 60.0},
    "P3-null":         {"kind": "null"},           # control only: not part of 'all'
}


# ---------------------------------------------------------------- attacks (simulator hooks)
class Null(Attack):
    """No attack: onset is marked as for the other classes, the registers pass unchanged."""

    def __call__(self, t, regs, st):
        self.active(t, st)
        return regs


class Freeze(Attack):
    """h1 register held at its value at onset."""

    def __init__(self, cfg, n, t0, phase):
        super().__init__(cfg, n, t0, phase)
        self.held = np.zeros(n, np.int32)
        self.init = np.zeros(n, bool)

    def __call__(self, t, regs, st):
        act = self.active(t, st)
        new = act & ~self.init
        self.held[new] = regs["h1"][new]
        self.init |= new
        out = dict(regs)
        out["h1"] = np.where(act, self.held, regs["h1"]).astype(np.int32)
        return out


class Scale(Attack):
    """h1 register multiplied by a factor ramping from 1 to k_final over ramp_s seconds."""

    def __init__(self, cfg, n, t0, phase, k_final, ramp_s):
        super().__init__(cfg, n, t0, phase)
        self.kf, self.ramp = float(k_final), float(ramp_s)

    def __call__(self, t, regs, st):
        act = self.active(t, st)
        tau = np.where(act, t - np.nan_to_num(self.onset), 0.0)
        k = np.where(act, 1.0 - (1.0 - self.kf) * np.minimum(tau / self.ramp, 1.0), 1.0)
        out = dict(regs)
        out["h1"] = np.rint(regs["h1"] * k).astype(np.int32)      # integer mm: stays on the lattice
        return out


class Replay(Attack):
    """Replay the attacker's own pre-onset recording of all five input registers."""

    def __init__(self, cfg, n, t0, phase, force, min_seg_s):
        super().__init__(cfg, n, t0, phase)
        dt = float(cfg["run"]["dt"])
        self.force = float(force)
        self.min_seg = int(round(min_seg_s / dt))
        # recording window: from t = 0 to onset (arming <= max t0, plus the wait for the onset phase)
        self.cap = int(math.ceil((float(np.max(t0)) + 400.0) / dt)) + 1
        self.buf = {k: np.zeros((n, self.cap), np.int32) for k in REGS}
        self.cmd = np.zeros((n, self.cap), np.int8)
        self.length = np.zeros(n, np.int64)      # ticks recorded, per run
        self.end = np.zeros(n, np.int64)         # recording length, frozen at onset
        self.ptr = np.full(n, -1, np.int64)      # replay position; -1 = not replaying yet
        self.splices = [[] for _ in range(n)]    # (t_rel s, jump h1 mm, jump h2 mm) per splice

    def _splice(self, r: int, h1: int, h2: int, cmd: int, t: float) -> int:
        hi = int(self.end[r]) - self.min_seg   # every segment lasts at least min_seg ticks
        if hi <= 0:
            raise RuntimeError(f"run {r}: recording shorter than one replay segment")
        cand = np.flatnonzero(self.cmd[r, :hi] == cmd)
        if cand.size == 0:
            cand = np.arange(hi)
        d = np.abs(self.buf["h1"][r, cand] - h1) + np.abs(self.buf["h2"][r, cand] - h2)
        j = int(cand[int(np.argmin(d))])
        self.splices[r].append((round(float(t), 1), int(self.buf["h1"][r, j]) - int(h1),
                                int(self.buf["h2"][r, j]) - int(h2)))
        return j

    def __call__(self, t, regs, st):
        act = self.active(t, st)
        cmd = st["pump"].astype(np.int8)
        # 1. record, for runs not yet under attack
        rec = np.flatnonzero(~act & (self.length < self.cap))
        if rec.size:
            for k in REGS:
                self.buf[k][rec, self.length[rec]] = regs[k][rec]
            self.cmd[rec, self.length[rec]] = cmd[rec]
            self.length[rec] += 1
        # 2. first splice at onset: match the live readings and the PLC's current command
        for r in np.flatnonzero(act & (self.ptr < 0)):
            self.end[r] = self.length[r]
            self.ptr[r] = self._splice(r, int(regs["h1"][r]), int(regs["h2"][r]), int(cmd[r]), t)
        # 3. recording exhausted: re-splice, matching the last replayed values
        for r in np.flatnonzero(act & (self.ptr >= self.end)):
            last = int(self.ptr[r]) - 1
            self.ptr[r] = self._splice(r, int(self.buf["h1"][r, last]), int(self.buf["h2"][r, last]),
                                       int(cmd[r]), t)
        # 4. replay
        out = dict(regs)
        rows = np.flatnonzero(act)
        if rows.size:
            for k in REGS:
                v = regs[k].copy()
                v[rows] = self.buf[k][rows, self.ptr[rows]]
                out[k] = v
            self.ptr[rows] += 1
        return out, np.where(act, self.force, np.nan)


def make(spec: dict, cfg: dict, n: int, t0: np.ndarray, phase):
    if spec["kind"] == "null":
        return Null(cfg, n, t0, phase)
    if spec["kind"] == "freeze":
        return Freeze(cfg, n, t0, phase)
    if spec["kind"] == "scale":
        return Scale(cfg, n, t0, phase, spec["k_final"], spec["ramp_s"])
    return Replay(cfg, n, t0, phase, spec["force"], spec["min_seg_s"])


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None      # None = not detected / not reached
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/twin.yaml")
    ap.add_argument("--detector", default="reports/phase2_detector_calibration.json")
    ap.add_argument("--classes", default="all",
                    help=f"comma list or 'all' = every class except P3-null ({', '.join(CLASSES)})")
    ap.add_argument("--runs", type=int, default=25, help="runs per class per onset phase")
    ap.add_argument("--horizon", type=int, default=3600, help="s after onset")
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--out", default="reports/phase3_attack_classes.json")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    thr, params = load_detector(a.detector)
    p = {k: params[k] for k in ("window", "mean_window", "alg_window", "k", "mask_pad")}
    hmax = float(cfg["plant"]["h_max"])
    far = float(params.get("far_per_day", 1.0))
    names = [n for n in CLASSES if CLASSES[n]["kind"] != "null"] if a.classes == "all" \
        else [x.strip() for x in a.classes.split(",")]
    seconds = 900 + 120 + a.horizon

    results, per_run = {}, []
    t_start = datetime.now()
    for name in names:
        spec = CLASSES[name]
        ci = list(CLASSES).index(name)          # seed from the class's fixed position
        for pi, phase in enumerate((1, 0)):
            seed = a.seed + 100 * ci + pi
            t0 = np.random.default_rng(seed + 7).uniform(600, 900, a.runs)
            atk = make(spec, cfg, a.runs, t0, phase)
            # harm: dry-out when the real pump is forced OFF, or a frozen reading holds it OFF
            drain = spec.get("force") == 0.0 or (spec["kind"] == "freeze" and phase == 0)
            rows = []
            for r, df in enumerate(simulate_iter(cfg, a.runs, seconds, seed, attack=atk)):
                onset = atk.onset[r]
                if not np.isfinite(onset):
                    continue
                st = statistics(df, cfg["plant"], p)
                row = {"class": name, "phase": "ON" if phase else "OFF", "seed": seed, "run": r,
                       **score_run(df, onset, st, thr, hmax)}
                if spec["kind"] == "replay":
                    sp = atk.splices[r]
                    row["splices"] = len(sp)
                    row["max_splice_jump_h1_mm"] = max(abs(s[1]) for s in sp)
                    row["max_splice_jump_h2_mm"] = max(abs(s[2]) for s in sp)
                    row["recording_s"] = round(float(atk.end[r]) * float(cfg["run"]["dt"]), 1)
                rows.append(row)
                per_run.append(row)
            key = f"{name} | onset pump {'ON' if phase else 'OFF'}"
            results[key] = summarise(rows, drain, far, a.horizon)
            if spec["kind"] == "replay" and rows:
                results[key]["splices_median"] = float(np.median([x["splices"] for x in rows]))
                results[key]["max_splice_jump_h1_mm"] = int(max(x["max_splice_jump_h1_mm"] for x in rows))
                results[key]["max_splice_jump_h2_mm"] = int(max(x["max_splice_jump_h2_mm"] for x in rows))
            print(f"  {name:<18} pump {'ON ' if phase else 'OFF'} done "
                  f"({(datetime.now() - t_start).total_seconds() / 60:.1f} min)", flush=True)

    chance = next(iter(results.values()))["chance_alarms_expected"] if results else 0
    print(f"\n(by chance alone, about {chance:.1f} of {a.runs} runs per row would alarm within the "
          f"{a.horizon} s horizon - the false-alarm budget; a 'detected' count near that is not detection)")
    print(f"\n{'class | onset phase':<34}{'arm':>4}{'detected':>10}{'median s':>10}{'p90 s':>8}"
          f"{'before harm':>13}{'pre-FA':>8}   first component(s)")
    for key, s in results.items():
        for arm in ("A", "B"):
            x = s[arm]
            fc = ", ".join(f"{k} {v}" for k, v in list(x["first_component"].items())[:3])
            print(f"{key if arm == 'A' else '':<34}{arm:>4}{x['detected']:>5}/{s['runs']:<4}"
                  f"{fmt_s(x['median_delay_s']):>10}{fmt_s(x['p90_delay_s']):>8}"
                  f"{x['before_harm']:>8}/{s['runs']:<4}{x['pre_onset_fa']:>8}   {fc}")
    print(f"\n{'class | onset phase':<34}{'harm':>6}{'reached':>10}{'median t s':>12}{'overflow':>10}{'out of band s':>15}")
    for key, s in results.items():
        print(f"{key:<34}{s['harm'].replace('t_', ''):>6}{s['harm_reached']:>5}/{s['runs']:<4}"
              f"{fmt_s(s['median_time_to_harm_s']):>12}{s['overflow_runs']:>6}/{s['runs']:<4}{s['median_out_band_s']:>14.0f}")
    rep = [(k, s) for k, s in results.items() if "splices_median" in s]
    if rep:
        print(f"\n{'replay | onset phase':<34}{'splices (median)':>18}{'max jump h1 mm':>16}{'max jump h2 mm':>16}")
        for key, s in rep:
            print(f"{key:<34}{s['splices_median']:>18.0f}{s['max_splice_jump_h1_mm']:>16}{s['max_splice_jump_h2_mm']:>16}")

    try:
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        head = "unknown"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(clean({
        "step": "phase3 3.1a new attack classes on the closed-loop simulator, Phase 2 physics arms",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": head, "detector": a.detector, "runs_per_phase": a.runs, "horizon_s": a.horizon,
        "seed": a.seed, "classes": {n: CLASSES[n] for n in names},
        "summary": results, "runs": per_run}), indent=2))
    print(f"\nwrote {out}  ({len(per_run)} runs)")


if __name__ == "__main__":
    main()
