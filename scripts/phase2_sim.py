#!/usr/bin/env python
"""Phase 2, step 2.2a - vectorised closed-loop simulator of the Phase 1 stack.

Many independent runs at once, each reproducing the live stack tick by tick:
    plant    RK4 on the equations of edge/twin/physics.py (dt from configs/twin.yaml)
    sensors  twin_server.py: level + N(0, level_noise_std), 1 mm quantiser;
             flows + N(0, flow_noise_std), rounded to cL/min, clipped to 0..65535
    PLC      two_tank.st on every tick: pump ON at <= 250 mm, OFF at >= 350 mm,
             high-high trip latched at >= 550 mm on either tank, cleared when both
             are below 350 mm; valve held open
    telemetry  registers and coils sampled once per second (1 Hz publisher)
Residuals are then computed by the released pipeline itself (add_derived from
scripts/make_baseline_dataset.py), so simulated and real data go through the
same code.

An optional attack hook rewrites the input registers between the twin and its
consumers: the PLC, the telemetry, or both (`inject_to`), and may override the
pump the plant actually receives. Step 2.2c (scripts/phase2_attacks.py) uses it.

This step (--fidelity): simulate benign days and compare them, statistic by
statistic, with the released benign day (FMU + OpenPLC). If the real day falls
inside the spread of the simulated days, the simulator can serve both as the
null generator and as the attack harness.

    python scripts/phase2_sim.py --fidelity --runs 8 --hours 24
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
from make_baseline_dataset import add_derived  # noqa: E402  (the released pipeline)

LPM_TO_M3S = 1.0 / 60000.0
CLPM_PER_M3S = 6_000_000.0


# ---------------------------------------------------------------- simulator
def simulate(cfg: dict, n_runs: int, seconds: int, seed: int, warmup_s: int = 600,
             attack=None, inject_to=("plc", "telemetry")) -> list[pd.DataFrame]:
    """Return one telemetry DataFrame per run, in the schema of baseline_v1.parquet
    (plus h1_true, h2_true and, when attacked, the registers the PLC saw)."""
    return list(simulate_iter(cfg, n_runs, seconds, seed, warmup_s, attack, inject_to))


def simulate_iter(cfg: dict, n_runs: int, seconds: int, seed: int, warmup_s: int = 600,
                  attack=None, inject_to=("plc", "telemetry")):
    """As simulate(), but yields the frames one at a time (one run in memory at once)."""
    p, s, c = cfg["plant"], cfg["sensors"], cfg["control"]
    A1, A2 = float(p["A1"]), float(p["A2"])
    k12 = float(p["Cv12"]) * math.sqrt(2 * float(p["g"]))
    k2o = float(p["Cv2o"]) * math.sqrt(2 * float(p["g"]))
    Qp, hmax = float(p["Qpump"]), float(p["h_max"])
    dt = float(cfg["run"]["dt"])
    tps = int(round(1.0 / dt))                      # ticks per published sample
    lstd, fstd, qmm = float(s["level_noise_std"]), float(s["flow_noise_std"]), int(s["quantisation_mm"])
    sp_lo, sp_hi, hh_lim = int(c["sp_low_mm"]), int(c["sp_high_mm"]), int(c["hh_limit_mm"])

    rng = np.random.default_rng(seed)
    n = n_runs
    h1 = np.full(n, float(p["h1_0"]))
    h2 = np.full(n, float(p["h2_0"]))
    pump = np.zeros(n, bool)
    hh = np.zeros(n, bool)
    valve = np.ones(n, bool)

    def deriv(a, b, qin):
        q12 = valve * k12 * np.sqrt(np.maximum(a, 1e-9))
        qo = k2o * np.sqrt(np.maximum(b, 1e-9))
        return (qin - q12) / A1, (q12 - qo) / A2

    rec = {k: np.zeros((n, seconds), np.int32)
           for k in ("h1", "h2", "qin", "q12", "qout", "pump", "alarm", "pump_applied")}
    force = np.full(n, np.nan)                      # actuator override from the attack hook; NaN = none
    rec_plc = {k: np.zeros((n, seconds), np.int32) for k in ("h1", "h2")} if attack else None
    true1 = np.zeros((n, seconds), np.float32)
    true2 = np.zeros((n, seconds), np.float32)

    for sec in range(warmup_s + seconds):
        noise = rng.standard_normal((tps, 5, n))
        for k in range(tps):
            # 1. twin: coils -> plant, one RK4 step, physical clamps (physics.py)
            applied = np.where(np.isnan(force), pump, force > 0.5)
            qin = Qp * applied
            a1, b1 = deriv(h1, h2, qin)
            a2, b2 = deriv(h1 + 0.5 * dt * a1, h2 + 0.5 * dt * b1, qin)
            a3, b3 = deriv(h1 + 0.5 * dt * a2, h2 + 0.5 * dt * b2, qin)
            a4, b4 = deriv(h1 + dt * a3, h2 + dt * b3, qin)
            h1 = np.clip(h1 + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4), 0.0, hmax)
            h2 = np.clip(h2 + dt / 6 * (b1 + 2 * b2 + 2 * b3 + b4), 0.0, hmax)
            # 2. sensors -> input registers (twin_server.py)
            q12 = valve * k12 * np.sqrt(np.maximum(h1, 1e-9))
            qo = k2o * np.sqrt(np.maximum(h2, 1e-9))
            regs = {
                "h1": (np.rint((h1 + lstd * noise[k, 0]) * 1000.0 / qmm) * qmm).astype(np.int32),
                "h2": (np.rint((h2 + lstd * noise[k, 1]) * 1000.0 / qmm) * qmm).astype(np.int32),
                "qin": np.clip(np.rint((qin + fstd * noise[k, 2]) * CLPM_PER_M3S), 0, 65535).astype(np.int32),
                "q12": np.clip(np.rint((q12 + fstd * noise[k, 3]) * CLPM_PER_M3S), 0, 65535).astype(np.int32),
                "qout": np.clip(np.rint((qo + fstd * noise[k, 4]) * CLPM_PER_M3S), 0, 65535).astype(np.int32),
            }
            # 3. attack hook: what the PLC and the telemetry see
            t_rel = (sec - warmup_s) + k * dt
            regs_plc = regs_tel = regs
            if attack is not None and t_rel >= 0:
                # The hook returns spoofed registers, optionally with an actuator override
                # (float array, NaN = leave the PLC's command alone) applied from the next tick.
                out = attack(t_rel, regs, {"h1": h1, "h2": h2, "pump": pump, "applied": applied, "rng": rng})
                spoof, force = out if isinstance(out, tuple) else (out, np.full(n, np.nan))
                regs_plc = spoof if "plc" in inject_to else regs
                regs_tel = spoof if "telemetry" in inject_to else regs
            # 4. PLC scan (two_tank.st)
            pump_before = pump
            H1, H2 = regs_plc["h1"], regs_plc["h2"]
            hh = np.where((H1 >= hh_lim) | (H2 >= hh_lim), True,
                          np.where((H1 < sp_hi) & (H2 < sp_hi), False, hh))
            pump = np.where(hh, False, np.where(H1 <= sp_lo, True, np.where(H1 >= sp_hi, False, pump)))
            # 5. publisher: one sample per second
            if k == 0 and sec >= warmup_s:
                i = sec - warmup_s
                for key in ("h1", "h2", "qin", "q12", "qout"):
                    rec[key][:, i] = regs_tel[key]
                # The publisher's Modbus read lands at a random point in the 0.1 s tick, so it
                # catches a coil the PLC has just written before the twin applies it about half
                # the time (released day: 96 of 1743 switches; always-after would give ~10 %).
                rec["pump"][:, i] = np.where(rng.random(n) < 0.5, pump, pump_before)
                rec["alarm"][:, i] = hh
                rec["pump_applied"][:, i] = applied
                true1[:, i], true2[:, i] = h1, h2
                if rec_plc is not None:
                    rec_plc["h1"][:, i], rec_plc["h2"][:, i] = regs_plc["h1"], regs_plc["h2"]

    for r in range(n):
        df = pd.DataFrame({
            "seq": np.arange(seconds), "ts": np.arange(seconds, dtype=float),
            "h1_m": rec["h1"][r] / 1000.0, "h2_m": rec["h2"][r] / 1000.0,
            "q_in_lpm": rec["qin"][r] / 100.0, "q12_lpm": rec["q12"][r] / 100.0, "q_out_lpm": rec["qout"][r] / 100.0,
            "pump": rec["pump"][r], "valve": 1, "alarm": rec["alarm"][r],
            "h1_true": true1[r], "h2_true": true2[r], "pump_applied": rec["pump_applied"][r],
        })
        if rec_plc is not None:
            df["h1_plc_m"] = rec_plc["h1"][r] / 1000.0
            df["h2_plc_m"] = rec_plc["h2"][r] / 1000.0
        yield add_derived(df, p)


# ---------------------------------------------------------------- statistics compared
def level_from_flow(q_lpm: np.ndarray, cv: float, g: float) -> np.ndarray:
    """Torricelli inverted: the level (mm) that would produce the measured flow."""
    return ((q_lpm * LPM_TO_M3S) / (cv * math.sqrt(2 * g))) ** 2 * 1000.0


def switch_mask(pump: np.ndarray, pad: int = 1) -> np.ndarray:
    n = len(pump)
    m = np.zeros(n, bool)
    for i in np.flatnonzero(pump[1:] != pump[:-1]) + 1:
        m[max(0, i - 1 - pad): min(n, i + 1 + pad)] = True
    m[0] = m[-1] = True
    return m


def fidelity_stats(df: pd.DataFrame, plant: dict) -> dict:
    g = float(plant["g"])
    pump = df["pump"].to_numpy().astype(int)
    hours = len(df) / 3600.0
    m = switch_mask(pump)
    st = {
        "pump switches per 24 h": np.count_nonzero(np.diff(pump)) * 24 / hours,
        "pump duty (%)": 100 * pump.mean(),
        "mean h1 (mm)": 1000 * df["h1_m"].mean(),
        "mean h2 (mm)": 1000 * df["h2_m"].mean(),
    }
    for tank in ("1", "2"):
        r = df[f"res{tank}"].to_numpy() * 1000.0
        rv = np.where(m, np.nan, r)
        st[f"res{tank} sigma, masked (mm/s)"] = float(np.nanstd(rv))
        st[f"res{tank} lag-2 ACF"] = float(pd.Series(rv).autocorr(2))
        for W in (60, 300, 900, 3600):
            st[f"res{tank} {W:>4} s mean, sigma (mm/s)"] = float(pd.Series(r).rolling(W).mean().std())
    a1 = 1000 * df["h1_m"].to_numpy() - level_from_flow(df["q12_lpm"].to_numpy(), float(plant["Cv12"]), g)
    a2 = 1000 * df["h2_m"].to_numpy() - level_from_flow(df["q_out_lpm"].to_numpy(), float(plant["Cv2o"]), g)
    for name, a in (("a1 = h1 - h(q12)", a1), ("a2 = h2 - h(q_out)", a2)):
        st[f"{name}: mean (mm)"] = float(np.mean(a))
        st[f"{name}: sigma (mm)"] = float(np.std(a))
        for W in (60, 300, 900):
            st[f"{name}: {W:>4} s mean, sigma (mm)"] = float(pd.Series(a).rolling(W).mean().std())
    qin_expected = float(plant["Qpump"]) / LPM_TO_M3S * pump
    bad = np.abs(df["q_in_lpm"].to_numpy() - qin_expected) > 0.005
    st["q_in != Qpump*pump: rows"] = int(bad.sum())
    st["q_in != Qpump*pump: rows not next to a switch"] = int((bad & ~m).sum())
    flows = df[["q_in_lpm", "q12_lpm", "q_out_lpm"]].to_numpy() * 100.0
    st["flows off the 1 cL/min lattice: rows"] = int(np.any(np.abs(flows - np.rint(flows)) > 1e-6, axis=1).sum())
    return st


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/twin.yaml")
    ap.add_argument("--parquet", default="data/processed/baseline_v1.parquet")
    ap.add_argument("--fidelity", action="store_true", help="compare simulated benign days with the real one")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--out", default="reports/phase2_sim_fidelity.json")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if not a.fidelity:
        ap.error("nothing to do - pass --fidelity")

    real = pd.read_parquet(a.parquet)
    real = real[real["seq"] >= 74].reset_index(drop=True)
    seconds = int(a.hours * 3600)
    print(f"simulating {a.runs} benign runs x {a.hours:g} h (+600 s warm-up), seed {a.seed} ...", flush=True)
    t0 = datetime.now()
    sims = simulate(cfg, a.runs, seconds, a.seed)
    print(f"    done in {(datetime.now() - t0).total_seconds():.0f} s")

    s_real = fidelity_stats(real, cfg["plant"])
    s_sim = [fidelity_stats(d, cfg["plant"]) for d in sims]
    rows, verdicts = [], {}
    print(f"\n{'statistic':<46}{'real':>10}{'sim mean':>10}{'sim sd':>9}{'z':>7}{'diff %':>8}   verdict")
    for key, rv in s_real.items():
        vals = np.array([s[key] for s in s_sim], dtype=float)
        mu, sd = float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        z = (rv - mu) / sd if sd > 0 else (0.0 if rv == mu else math.inf)
        rel = 100 * (rv - mu) / abs(mu) if mu != 0 else math.nan
        ok = abs(z) <= 3 or (not math.isnan(rel) and abs(rel) <= 2)
        verdicts[key] = "ok" if ok else "CHECK"
        fmt = "{:>10.0f}" if max(abs(rv), abs(mu)) >= 100 else "{:>10.4f}"
        print(f"{key:<46}" + fmt.format(rv) + fmt.format(mu) + f"{sd:>9.4f}{z:>7.1f}"
              + (f"{rel:>8.1f}" if not math.isnan(rel) else f"{'-':>8}") + f"   {verdicts[key]}")
        rows.append({"statistic": key, "real": rv, "sim_mean": mu, "sim_sd": sd, "z": z,
                     "diff_pct": rel, "verdict": verdicts[key]})

    n_chk = sum(v == "CHECK" for v in verdicts.values())
    print(f"\n{len(verdicts)} statistics: {len(verdicts) - n_chk} ok (|z| <= 3 or within 2 %), {n_chk} to check")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "step": "phase2 2.2a simulator fidelity vs released benign day",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": a.config, "runs": a.runs, "hours": a.hours, "seed": a.seed,
        "rows": rows}, indent=2, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
