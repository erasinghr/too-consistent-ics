#!/usr/bin/env python
"""Phase 2, step 2.5 - score a live attack run: physics arms on the telemetry, consequence
from the ground truth, and a pointer to Arm N on the paired pcap.

For one run directory (data/raw/attack/<rung>_<ts>/ holding baseline_*.csv and truth.csv)
and the injector sidecar (reports/phase2_live_<rung>_<ts>.json, for the onset time):

  physics arms  - run the 2.2b detector (scripts/phase2_detector.py, thresholds from
                  reports/phase2_detector_calibration.json) over the RECORDED telemetry,
                  which for a fabrication is the attacker's feed. Report, per arm, whether
                  anything fired after onset and the delay. For R3 the expected result is
                  SILENCE: the covert limit, now on live data.
  consequence   - from truth.csv (the real twin on :5030): peak real level, and time from
                  onset until the real tank passes the 550 mm interlock / 600 mm overflow /
                  20 mm dry-out. This is the physical damage the silent telemetry hid.
  network arm   - scored separately on the pcap:
                      python scripts/phase2_net_arm.py <attack pcap> --score
                  (printed as the next step; Arm N should fire on the proxy's added latency).

    python scripts/phase2_score_live.py                      # newest run, newest sidecar
    python scripts/phase2_score_live.py --run data/raw/attack/R3-fab-matched_...Z
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_baseline_dataset import add_derived                       # noqa: E402
from phase2_detector import BLOCK, bool_blocks, component_alarms, episodes, load_detector, statistics  # noqa: E402

RESIDUAL = set("CMVP")       # alarms available only one sample after their index (central diff)


def regularize(df: pd.DataFrame) -> pd.DataFrame:
    """Snap live telemetry to a 1 Hz grid: each row to its nearest second, keep the last of
    any doubles, set ts to exact seconds. Transport through the proxy makes the publisher's
    1 Hz poll slip and occasionally double up (sub-second dt), which a central-difference rate
    residual amplifies; the simulator null (and the released day) are regular 1 Hz, so this is
    the matched basis. Gaps survive as dt>1; only sub-second doubles are collapsed."""
    t0 = float(df["ts"].iloc[0])
    sec = np.rint(df["ts"].to_numpy() - t0).astype(int)
    df = df.assign(_sec=sec).drop_duplicates("_sec", keep="last")
    df = df.assign(ts=t0 + df["_sec"].astype(float)).drop(columns="_sec").reset_index(drop=True)
    return df


def dt_summary(ts: np.ndarray) -> dict:
    dt = np.diff(ts)
    return {"rows": int(len(ts) + 0), "mean_dt": round(float(dt.mean()), 3),
            "min_dt": round(float(dt.min()), 3), "max_dt": round(float(dt.max()), 3),
            "sub_0p5s": int((dt < 0.5).sum()), "over_1p5s": int((dt > 1.5).sum())}


def load_telemetry(run: Path, plant: dict) -> pd.DataFrame:
    files = sorted(glob.glob(str(run / "baseline_*.csv")))
    if not files:
        raise SystemExit(f"no baseline_*.csv in {run}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.sort_values("ts").drop_duplicates(subset=["seq"]).reset_index(drop=True)


def onset_unix(sidecar: Path) -> float:
    rec = json.loads(sidecar.read_text())
    iso = rec.get("onset_wallclock_utc")
    if not iso:
        raise SystemExit(f"{sidecar} has no onset_wallclock_utc - did the attack arm?")
    return datetime.fromisoformat(iso).timestamp()


def onset_from_log(log: Path, run_date_iso: str) -> float | None:
    """Fallback when no sidecar: the injector prints '[HH:MM:SS] onset: ...' to attack_proxy.log.
    The process clock is UTC on this stack (RUNID and CSV timestamps agree), so combine that
    time with the run's date (from the first telemetry row) to get the onset instant."""
    import re
    if not log.exists():
        return None
    t = None
    for line in log.read_text().splitlines():
        m = re.search(r"\[(\d\d:\d\d:\d\d)\]\s*onset", line)
        if m:
            t = m.group(1)
    if t is None:
        return None
    return datetime.fromisoformat(f"{run_date_iso}T{t}+00:00").timestamp()


def first_after(alarm: np.ndarray, i0: int) -> float:
    idx = np.flatnonzero(alarm[i0:])
    return float(idx[0]) if idx.size else math.inf


def consequence(run: Path, onset: float) -> dict:
    tp = run / "truth.csv"
    if not tp.exists():
        return {"truth": "missing"}
    ts, h1, h2, pump = [], [], [], []
    with open(tp) as f:
        for row in csv.DictReader(f):
            ts.append(float(row["ts_unix"])); h1.append(float(row["h1_true_mm"]))
            h2.append(float(row["h2_true_mm"])); pump.append(int(row["pump"]))
    ts, h1, h2 = np.array(ts), np.array(h1), np.array(h2)
    after = ts >= onset
    hmax = np.maximum(h1, h2)

    def t_cross(cond):
        m = cond & after
        return float(ts[m][0] - onset) if m.any() else None
    return {
        "rows": len(ts),
        "peak_real_h1_mm": round(float(h1.max()), 1),
        "peak_real_level_mm": round(float(hmax.max()), 1),
        "min_real_h1_mm": round(float(h1[after].min()), 1) if after.any() else None,
        "t_to_HH_550_s": t_cross(hmax >= 550),
        "t_to_overflow_600_s": t_cross(hmax >= 600),
        "t_to_dry_20_s": t_cross(h1 <= 20),
        "real_out_of_band_s": int(((h1[after] < 240) | (h1[after] > 360)).sum()) if after.any() else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/twin.yaml")
    ap.add_argument("--run", default=None, help="attack run dir; default newest under data/raw/attack/")
    ap.add_argument("--sidecar", default=None, help="injector sidecar JSON; default newest reports/phase2_live_*.json")
    ap.add_argument("--detector", default="reports/phase2_detector_calibration.json")
    ap.add_argument("--onset-iso", default=None, help="override onset, e.g. 2026-10-30T14:16:21+00:00")
    ap.add_argument("--proxy-log", default="logs/attack_proxy.log", help="fallback onset source")
    ap.add_argument("--no-regularize", action="store_true",
                    help="score raw live timestamps (default: snap to 1 Hz, removing transport jitter)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    plant = cfg["plant"]
    run = Path(a.run) if a.run else Path(sorted(glob.glob("data/raw/attack/*/"))[-1])
    thr, params = load_detector(a.detector)
    p = {k: params[k] for k in ("window", "mean_window", "alg_window", "k", "mask_pad")}
    iso = lambda t: datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")
    print(f"run      {run}")

    raw = load_telemetry(run, plant)
    jitter = dt_summary(raw["ts"].to_numpy())
    print(f"telemetry sampling (raw): mean dt {jitter['mean_dt']}s, min {jitter['min_dt']}s, "
          f"max {jitter['max_dt']}s, {jitter['sub_0p5s']} gaps <0.5s, {jitter['over_1p5s']} >1.5s")
    if a.no_regularize:
        df = add_derived(raw, plant); reg_note = "raw timestamps (--no-regularize)"
    else:
        df = add_derived(regularize(raw), plant)
        reg_note = f"regularised to 1 Hz ({len(raw)} -> {len(df)} rows)"
    print(f"residuals computed on: {reg_note}")
    # resolve onset: --onset-iso, else the sidecar JSON, else the proxy log + the run's date
    run_date = datetime.fromtimestamp(float(raw["ts"].iloc[0]), timezone.utc).strftime("%Y-%m-%d")
    scj = ([a.sidecar] if a.sidecar else sorted(glob.glob("reports/phase2_live_*.json")))
    if a.onset_iso:
        onset, src = datetime.fromisoformat(a.onset_iso).timestamp(), "--onset-iso"
    elif scj:
        onset, src = onset_unix(Path(scj[-1])), Path(scj[-1]).name
    else:
        onset = onset_from_log(Path(a.proxy_log), run_date)
        if onset is None:
            raise SystemExit("no onset found: no sidecar JSON, and no 'onset' line in "
                             f"{a.proxy_log}. Pass --onset-iso YYYY-MM-DDTHH:MM:SS+00:00 "
                             "(the time from attack_proxy.log's onset line).")
        src = f"{a.proxy_log} (no sidecar)"
    t0, t1 = float(raw["ts"].iloc[0]), float(raw["ts"].iloc[-1])
    if not (t0 <= onset <= t1):
        print(f"WARNING: onset {iso(onset)} is outside the telemetry span {iso(t0)}..{iso(t1)} "
              "- check clock/timezone or pass --onset-iso")
    print(f"onset    {iso(onset)}   (from {src})")
    i0 = int(np.searchsorted(df["ts"].to_numpy(), onset))
    hours = len(df) / 3600
    print(f"telemetry {len(df)} rows ({hours:.2f} h), onset at sample {i0}")

    st = statistics(df, plant, p)
    print(f"\nphysics arms on the RECORDED (for R3, fabricated) telemetry  [thresholds: {Path(a.detector).name}]")
    report_arms = {}
    for arm in ("A", "B"):
        comps = component_alarms(st, thr, arm)
        comps = {k: (np.r_[False, v[:-1]] if k[0] in RESIDUAL else v) for k, v in comps.items()}
        al = np.logical_or.reduce(list(comps.values()))
        after = al.copy(); after[:i0] = False
        before = al.copy(); before[i0:] = False
        d = first_after(al, i0)
        fired = [k for k, v in comps.items() if v[i0:].any()]
        report_arms[arm] = {
            "alarm_after_onset": bool(after.any()),
            "delay_s": None if math.isinf(d) else d,
            "episodes_after_onset": episodes(bool_blocks(after)),
            "pre_onset_alarm": bool(before.any()),
            "components_fired_after_onset": fired,
        }
        verdict = "SILENT (covert)" if not after.any() else f"fired at +{d:.0f}s via {'+'.join(fired)}"
        print(f"  Arm {arm}: {verdict}   [pre-onset alarms: {'yes' if before.any() else 'no'}]")

    cons = consequence(run, onset)
    print("\nconsequence from the TRUE plant (truth.csv):")
    if cons.get("truth") == "missing":
        print("  truth.csv missing - no ground truth for this run")
    else:
        print(f"  peak real level {cons['peak_real_level_mm']} mm (h1 {cons['peak_real_h1_mm']}, "
              f"min h1 after onset {cons['min_real_h1_mm']})")
        print(f"  time from onset -> HH 550 mm: {cons['t_to_HH_550_s']} s   "
              f"overflow 600 mm: {cons['t_to_overflow_600_s']} s   dry 20 mm: {cons['t_to_dry_20_s']} s")
        print(f"  real level out of the 240-360 band for {cons['real_out_of_band_s']} s after onset")

    pcaps = sorted(glob.glob("data/raw/pcap_attack/*.pcap")) + sorted(glob.glob("data/raw/pcap_attack/*.pcapng"))
    print("\nnetwork arm - score the paired pcap next:")
    if pcaps:
        print(f"    python scripts/phase2_net_arm.py {pcaps[-1]} --score")
    else:
        print("    python scripts/phase2_net_arm.py <attack pcap> --score   (no pcap found under data/raw/pcap_attack/)")

    headline = ("physics SILENT, real plant driven past the interlock - the covert attack the fused "
                "detector needs the network arm for"
                if (not report_arms["A"]["alarm_after_onset"] and not report_arms["B"]["alarm_after_onset"]
                    and cons.get("t_to_HH_550_s") is not None)
                else "see per-arm results above")
    rep = {
        "step": "phase2 2.5 live-run scoring (physics arms + consequence)",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": str(run), "onset_source": src, "detector": a.detector,
        "onset_utc": iso(onset), "telemetry_rows": int(len(df)), "onset_sample": i0,
        "sampling_jitter_raw": jitter, "regularized": not a.no_regularize,
        "physics_arms": report_arms, "consequence": cons, "headline": headline,
    }
    out = Path(a.out) if a.out else run / "score_physics.json"
    out.write_text(json.dumps(rep, indent=2, default=lambda x: None if isinstance(x, float) and not math.isfinite(x) else x))
    print(f"\nheadline: {headline}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
