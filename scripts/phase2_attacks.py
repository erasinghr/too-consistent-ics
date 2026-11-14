#!/usr/bin/env python
"""Phase 2, step 2.2c - attack rungs on the closed-loop simulator, scored by both physics arms.

Every rung is a SOURCE spoof: the falsified registers are what both the PLC and the
telemetry read (a compromised field device, or a rogue responder impersonating the
twin). PLC-path-only spoofs are out of scope here: message signing would stop a
man-in-the-middle, and the physics arms are blind to them by construction.

Rungs, in order of attacker capability:
  R0-step         h1 reading stepped -20 mm. Positive control: every arm should fire.
  R1-drift        h1 reading drifts down at 0.2 mm/s (below the CUSUM's ~0.54 mm/s reach).
                  The PLC keeps filling against a low reading, so the real tank rises.
  R1-creep        the same at 0.01 mm/s (below the 300 s mean's ~0.045 mm/s reach).
  R2-fab-clean    the whole measurement vector (h1, h2, q_in, q12, q_out) comes from a
                  shadow model of the plant driven by the PLC's own commands, quantised
                  but WITHOUT sensor noise, while the real pump is forced ON.
  R3-fab-matched  as R2, with sensor noise matched to the real transducer (1.5 mm,
                  1 mm quantiser, flows rounded to cL/min). The covert limit.
  R2-/R3-...-drain  the same fabrications with the real pump forced OFF (dry-out).
The shadow model starts from the TRUE plant state at onset and uses the exact plant
equations and parameters: the attacker with full model knowledge (the worst case
for the detector).

Each rung runs --runs simulated episodes for each onset phase (attack starts while the
PLC has the pump ON, or OFF), with a random arming time 600-900 s into the record.
Scored per run and arm:
  delay       first alarm at or after onset, in s (inf = not detected in the horizon);
              residual-based alarms count from the sample at which they become computable
  first       which component(s) fired first
  pre-FA      any alarm between 300 s and onset (a false alarm, not a detection)
Consequences, from the TRUE levels:
  t_HH        s from onset until either true level reaches 550 mm (where the HH
              interlock would have tripped on honest readings)
  t_over      s until the true level reaches h_max (overflow)
  t_dry       s until true h1 falls to 20 mm
  out_band    s with true h1 outside 240-360 mm
"Detected before harm" = delay < t_HH (or < t_dry for drain rungs).

Thresholds come from reports/phase2_detector_calibration.json (step 2.2b).

    python scripts/phase2_attacks.py                       # all rungs, 25 runs per phase, ~4 min
    python scripts/phase2_attacks.py --rungs R2-fab-clean,R3-fab-matched --runs 50
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
from phase2_detector import component_alarms, load_detector, statistics  # noqa: E402
from phase2_sim import CLPM_PER_M3S, simulate_iter  # noqa: E402

RUNGS = {
    "R0-step":              {"kind": "bias", "step_mm": 20.0, "rate": 0.0, "cap_mm": 20.0},
    "R1-drift":             {"kind": "bias", "step_mm": 0.0, "rate": 0.2, "cap_mm": 300.0},
    "R1-creep":             {"kind": "bias", "step_mm": 0.0, "rate": 0.01, "cap_mm": 300.0},
    "R2-fab-clean":         {"kind": "fabricate", "noise": False, "force": 1.0},
    "R3-fab-matched":       {"kind": "fabricate", "noise": True, "force": 1.0},
    "R2-fab-clean-drain":   {"kind": "fabricate", "noise": False, "force": 0.0},
    "R3-fab-matched-drain": {"kind": "fabricate", "noise": True, "force": 0.0},
}


# ---------------------------------------------------------------- attacks (simulator hooks)
class Attack:
    """Arms at t0 (per run); onset is the first tick after arming at which the PLC's pump
    command equals `phase` (1 = ON, 0 = OFF, None = immediately)."""

    def __init__(self, cfg: dict, n: int, t0: np.ndarray, phase):
        self.cfg, self.n, self.t0, self.phase = cfg, n, t0, phase
        self.onset = np.full(n, np.nan)

    def active(self, t: float, st: dict) -> np.ndarray:
        armed = (t >= self.t0) & np.isnan(self.onset)
        if self.phase is not None:
            armed &= (st["pump"].astype(int) == self.phase)
        self.onset[armed] = t
        return t >= np.nan_to_num(self.onset, nan=np.inf)


class Bias(Attack):
    """Subtract a growing bias from the h1 register: the reading runs low, the real tank high."""

    def __init__(self, cfg, n, t0, phase, step_mm, rate, cap_mm):
        super().__init__(cfg, n, t0, phase)
        self.step, self.rate, self.cap = step_mm, rate, cap_mm

    def __call__(self, t, regs, st):
        act = self.active(t, st)
        tau = np.where(act, t - np.nan_to_num(self.onset), 0.0)
        bias = np.where(act, np.minimum(self.step + self.rate * tau, self.cap), 0.0)
        out = dict(regs)
        out["h1"] = np.maximum(0, regs["h1"] - np.rint(bias)).astype(np.int32)   # integer mm: stays on the lattice
        return out


class Fabricate(Attack):
    """Report a shadow plant driven by the PLC's commands; force the real pump."""

    def __init__(self, cfg, n, t0, phase, noise, force):
        super().__init__(cfg, n, t0, phase)
        p, s = cfg["plant"], cfg["sensors"]
        self.A1, self.A2 = float(p["A1"]), float(p["A2"])
        self.k12 = float(p["Cv12"]) * math.sqrt(2 * float(p["g"]))
        self.k2o = float(p["Cv2o"]) * math.sqrt(2 * float(p["g"]))
        self.Qp, self.hmax = float(p["Qpump"]), float(p["h_max"])
        self.dt = float(cfg["run"]["dt"])
        self.lstd = float(s["level_noise_std"]) if noise else 0.0
        self.fstd = float(s["flow_noise_std"]) if noise else 0.0
        self.qmm = int(s["quantisation_mm"])
        self.force = force
        self.h1s = np.zeros(n)
        self.h2s = np.zeros(n)
        self.init = np.zeros(n, bool)

    def _deriv(self, a, b, qin):
        q12 = self.k12 * np.sqrt(np.maximum(a, 1e-9))
        qo = self.k2o * np.sqrt(np.maximum(b, 1e-9))
        return (qin - q12) / self.A1, (q12 - qo) / self.A2

    def __call__(self, t, regs, st):
        act = self.active(t, st)
        cmd = st["pump"].astype(float)
        qin = self.Qp * cmd
        # advance the shadow of runs already under attack (same RK4 step as the twin)
        h1, h2, dt = self.h1s, self.h2s, self.dt
        a1, b1 = self._deriv(h1, h2, qin)
        a2, b2 = self._deriv(h1 + 0.5 * dt * a1, h2 + 0.5 * dt * b1, qin)
        a3, b3 = self._deriv(h1 + 0.5 * dt * a2, h2 + 0.5 * dt * b2, qin)
        a4, b4 = self._deriv(h1 + dt * a3, h2 + dt * b3, qin)
        nh1 = np.clip(h1 + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4), 0.0, self.hmax)
        nh2 = np.clip(h2 + dt / 6 * (b1 + 2 * b2 + 2 * b3 + b4), 0.0, self.hmax)
        self.h1s = np.where(self.init, nh1, self.h1s)
        self.h2s = np.where(self.init, nh2, self.h2s)
        # runs whose attack starts now: shadow := true state
        new = act & ~self.init
        self.h1s[new], self.h2s[new] = st["h1"][new], st["h2"][new]
        self.init |= new

        rng, n = st["rng"], self.n
        q12 = self.k12 * np.sqrt(np.maximum(self.h1s, 1e-9))
        qo = self.k2o * np.sqrt(np.maximum(self.h2s, 1e-9))
        fake = {
            "h1": np.rint((self.h1s + self.lstd * rng.standard_normal(n)) * 1000.0 / self.qmm) * self.qmm,
            "h2": np.rint((self.h2s + self.lstd * rng.standard_normal(n)) * 1000.0 / self.qmm) * self.qmm,
            "qin": np.rint((qin + self.fstd * rng.standard_normal(n)) * CLPM_PER_M3S),
            "q12": np.rint((q12 + self.fstd * rng.standard_normal(n)) * CLPM_PER_M3S),
            "qout": np.rint((qo + self.fstd * rng.standard_normal(n)) * CLPM_PER_M3S),
        }
        out = {k: np.where(act, np.clip(fake[k], 0, 65535), regs[k]).astype(np.int32) for k in regs}
        return out, np.where(act, self.force, np.nan)


def make_attack(spec: dict, cfg: dict, n: int, t0: np.ndarray, phase):
    if spec["kind"] == "bias":
        return Bias(cfg, n, t0, phase, spec["step_mm"], spec["rate"], spec["cap_mm"])
    return Fabricate(cfg, n, t0, phase, spec["noise"], spec["force"])


# ---------------------------------------------------------------- scoring
def first_at_or_after(x: np.ndarray, i0: int) -> float:
    idx = np.flatnonzero(x[i0:])
    return float(idx[0]) if idx.size else math.inf


def score_run(df, onset: float, st: dict, thr: dict, hmax: float, warm: int = 300) -> dict:
    i0 = int(math.ceil(onset))
    rec = {"onset_s": round(onset, 1), "phase_at_onset": int(df["pump"].iloc[min(i0, len(df) - 1)])}
    for arm in ("A", "B"):
        comps = component_alarms(st, thr, arm)
        # The central difference r[t] uses h[t+1], so a residual-based alarm indexed t is only
        # available at t+1. Shift C, M, V and P by one sample; T, L, X, F are causal as indexed.
        comps = {k: (np.r_[False, v[:-1]] if k[0] in "CMVP" else v) for k, v in comps.items()}
        al = np.logical_or.reduce(list(comps.values()))
        d = first_at_or_after(al, i0)
        rec[f"delay_{arm}"] = d
        rec[f"pre_fa_{arm}"] = bool(al[warm:i0].any())
        rec[f"first_{arm}"] = "+".join(k for k, v in comps.items() if math.isfinite(d) and v[i0 + int(d)]) or "-"
    h1, h2 = df["h1_true"].to_numpy() * 1000.0, df["h2_true"].to_numpy() * 1000.0
    rec["t_HH"] = first_at_or_after(np.maximum(h1, h2) >= 550.0, i0)
    rec["t_over"] = first_at_or_after(np.maximum(h1, h2) >= hmax * 1000.0 - 0.5, i0)
    rec["t_dry"] = first_at_or_after(h1 <= 20.0, i0)
    rec["out_band_s"] = int(((h1[i0:] < 240.0) | (h1[i0:] > 360.0)).sum())
    rec["max_true_h_mm"] = round(float(np.maximum(h1, h2)[i0:].max()), 1)
    rec["min_true_h1_mm"] = round(float(h1[i0:].min()), 1)
    return rec


def summarise(rows: list[dict], drain: bool, far_per_day: float, horizon: int) -> dict:
    harm = "t_dry" if drain else "t_HH"
    # Runs expected to alarm by chance inside the horizon even with no attack at all (the
    # false-alarm budget); detections at about this count are not evidence of detection.
    chance = len(rows) * (1 - math.exp(-far_per_day * horizon / 86400))
    out = {"runs": len(rows), "harm": harm, "chance_alarms_expected": round(chance, 2)}
    for arm in ("A", "B"):
        d = np.array([r[f"delay_{arm}"] for r in rows])
        h = np.array([r[harm] for r in rows])
        fin = d[np.isfinite(d)]
        firsts = {}
        for r in rows:
            firsts[r[f"first_{arm}"]] = firsts.get(r[f"first_{arm}"], 0) + 1
        out[arm] = {
            "detected": int(np.isfinite(d).sum()),
            "median_delay_s": float(np.median(fin)) if fin.size else None,
            "p90_delay_s": float(np.quantile(fin, 0.9)) if fin.size else None,
            "before_harm": int((d < h).sum()),
            "pre_onset_fa": int(sum(r[f"pre_fa_{arm}"] for r in rows)),
            "first_component": dict(sorted(firsts.items(), key=lambda kv: -kv[1])),
        }
    h = np.array([r[harm] for r in rows])
    out["harm_reached"] = int(np.isfinite(h).sum())
    out["median_time_to_harm_s"] = float(np.median(h[np.isfinite(h)])) if np.isfinite(h).any() else None
    out["overflow_runs"] = int(np.isfinite([r["t_over"] for r in rows]).sum())
    out["median_out_band_s"] = float(np.median([r["out_band_s"] for r in rows]))
    return out


def fmt_s(x) -> str:
    return "-" if x is None else f"{x:.0f}"


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/twin.yaml")
    ap.add_argument("--detector", default="reports/phase2_detector_calibration.json")
    ap.add_argument("--rungs", default="all", help=f"comma list or 'all' ({', '.join(RUNGS)})")
    ap.add_argument("--runs", type=int, default=25, help="runs per rung per onset phase")
    ap.add_argument("--horizon", type=int, default=3600, help="s after onset")
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--out", default="reports/phase2_attack_runs.json")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    thr, params = load_detector(a.detector)
    p = {k: params[k] for k in ("window", "mean_window", "alg_window", "k", "mask_pad")}
    hmax = float(cfg["plant"]["h_max"])
    names = list(RUNGS) if a.rungs == "all" else [x.strip() for x in a.rungs.split(",")]
    seconds = 900 + 120 + a.horizon

    results, per_run = {}, []
    t_start = datetime.now()
    for ri, name in enumerate(names):
        spec = RUNGS[name]
        drain = spec.get("force") == 0.0
        for pi, phase in enumerate((1, 0)):
            seed = a.seed + 100 * ri + pi
            rng = np.random.default_rng(seed + 7)
            t0 = rng.uniform(600, 900, a.runs)
            atk = make_attack(spec, cfg, a.runs, t0, phase)
            rows = []
            for r, df in enumerate(simulate_iter(cfg, a.runs, seconds, seed, attack=atk)):
                onset = atk.onset[r]
                if not np.isfinite(onset):
                    continue
                st = statistics(df, cfg["plant"], p)
                row = {"rung": name, "phase": "ON" if phase else "OFF", "seed": seed, "run": r,
                       **score_run(df, onset, st, thr, hmax)}
                rows.append(row)
                per_run.append(row)
            results[f"{name} | onset pump {'ON' if phase else 'OFF'}"] = summarise(
                rows, drain, float(params.get("far_per_day", 1.0)), a.horizon)
            print(f"  {name:<22} pump {'ON ' if phase else 'OFF'} done "
                  f"({(datetime.now() - t_start).total_seconds() / 60:.1f} min)", flush=True)

    chance = next(iter(results.values()))["chance_alarms_expected"] if results else 0
    print(f"\n(by chance alone, about {chance:.1f} of {a.runs} runs per row would alarm within the "
          f"{a.horizon} s horizon - the false-alarm budget; a 'detected' count near that is not detection)")
    print(f"\n{'rung | onset phase':<36}{'arm':>4}{'detected':>10}{'median s':>10}{'p90 s':>8}"
          f"{'before harm':>13}{'pre-FA':>8}   first component(s)")
    for key, s in results.items():
        for arm in ("A", "B"):
            x = s[arm]
            fc = ", ".join(f"{k} {v}" for k, v in list(x["first_component"].items())[:3])
            print(f"{key if arm == 'A' else '':<36}{arm:>4}{x['detected']:>5}/{s['runs']:<4}"
                  f"{fmt_s(x['median_delay_s']):>10}{fmt_s(x['p90_delay_s']):>8}"
                  f"{x['before_harm']:>8}/{s['runs']:<4}{x['pre_onset_fa']:>8}   {fc}")
    print(f"\n{'rung | onset phase':<36}{'harm':>6}{'reached':>10}{'median t s':>12}{'overflow':>10}{'out of band s':>15}")
    for key, s in results.items():
        print(f"{key:<36}{s['harm'].replace('t_', ''):>6}{s['harm_reached']:>5}/{s['runs']:<4}"
              f"{fmt_s(s['median_time_to_harm_s']):>12}{s['overflow_runs']:>6}/{s['runs']:<4}{s['median_out_band_s']:>14.0f}")

    try:
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        head = "unknown"
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (float, np.floating)):
            return float(x) if math.isfinite(x) else None      # None = not detected / not reached
        if isinstance(x, np.bool_):
            return bool(x)
        return x

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(clean({
        "step": "phase2 2.2c attack rungs on the closed-loop simulator",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": head, "detector": a.detector, "runs_per_phase": a.runs, "horizon_s": a.horizon,
        "seed": a.seed, "rungs": {n: RUNGS[n] for n in names},
        "summary": results, "runs": per_run}), indent=2))
    print(f"\nwrote {out}  ({len(per_run)} runs)")


if __name__ == "__main__":
    main()
