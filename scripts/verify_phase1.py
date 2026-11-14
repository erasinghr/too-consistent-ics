#!/usr/bin/env python3
"""Hybrid AI-Quantum Cognitive Security Framework Phase 1 - exit criteria verification.

Applies the standing three-question phase check:

  Q1  What did this phase produce?      -> artefact checks A1-A3
  Q2  How do I know it works?           -> live checks L1-L4 and physics check P1
  Q3  What does the next phase consume? -> handover check H1

Run it with the full stack up (twin + PLC + publisher + recorder):

    python scripts/verify_phase1.py --config configs/twin.yaml

Add --offline to check only the recorded dataset (no live services required).
Writes reports/phase1_verification.json and exits non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

RESULTS: list[dict] = []


def check(cid: str, name: str, ok: bool, detail: str = "", critical: bool = True):
    RESULTS.append({"id": cid, "check": name, "pass": bool(ok),
                    "detail": detail, "critical": critical})
    mark = "PASS" if ok else ("FAIL" if critical else "WARN")
    print(f"  [{mark}] {cid}  {name}" + (f"  -- {detail}" if detail else ""))
    return ok


# ---------------------------------------------------------------- artefacts
def artefact_checks(cfg, root: Path):
    print("\nQ1  What did Phase 1 produce?")
    mo = root / "edge/twin/TwoTank.mo"
    fmu = root / cfg["run"]["fmu_path"]
    st = root / "edge/plc/two_tank.st"
    check("A1", "Modelica plant model present", mo.exists(), str(mo))
    check("A2", "FMU exported", fmu.exists(),
          str(fmu) if fmu.exists() else "not exported - python engine only",
          critical=False)
    check("A3", "PLC Structured Text present", st.exists(), str(st))


# --------------------------------------------------------------- live checks
def live_checks(cfg, sample_s: float):
    print("\nQ2  How do I know it works?")
    from pymodbus.client import ModbusTcpClient
    mb, ctl = cfg["modbus"], cfg["control"]
    host = "127.0.0.1" if mb["host"] in ("0.0.0.0", "") else mb["host"]
    c = ModbusTcpClient(host, port=int(mb["port"]))
    if not check("L1", "twin reachable on Modbus/TCP", c.connect(),
                 f"{host}:{mb['port']}"):
        return None

    unit = int(mb["unit_id"])
    h1, h2, pump, t0 = [], [], [], time.perf_counter()
    while time.perf_counter() - t0 < sample_s:
        rr = c.read_input_registers(0, 5, slave=unit)
        rc = c.read_coils(0, 3, slave=unit)
        if not rr.isError() and not rc.isError():
            h1.append(rr.registers[0]); h2.append(rr.registers[1])
            pump.append(int(rc.bits[0]))
        time.sleep(0.2)
    c.close()

    n = len(h1)
    check("L2", "levels are evolving (plant is running)",
          n > 5 and (max(h1) - min(h1)) >= 2,
          f"h1 range {min(h1)}-{max(h1)} mm over {n} samples")
    duty = sum(pump) / n if n else 0.0
    check("L3", "control loop is closed (pump modulating)",
          0.15 <= duty <= 0.85 or (0 < sum(pump) < n),
          f"pump duty {duty:.1%} over {sample_s:.0f}s")
    in_band = all(ctl["sp_low_mm"] - 60 <= v <= ctl["sp_high_mm"] + 60 for v in h1)
    check("L4", "tank 1 held inside the control band",
          in_band, f"band {ctl['sp_low_mm']}-{ctl['sp_high_mm']} mm (+/-60 mm tol)")
    return {"h1_mm": h1, "h2_mm": h2, "pump": pump, "duty": duty}


def mqtt_check(cfg, sample_s: float):
    mq = cfg["mqtt"]
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        return check("L5", "MQTT telemetry arriving at the fog tier", False,
                     "paho-mqtt not installed", critical=False)
    got = []
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="phase1-verify")
    except AttributeError:
        client = mqtt.Client(client_id="phase1-verify")
    client.on_message = lambda c, u, m: got.append(m.payload)
    try:
        client.connect(mq["host"], int(mq["port"]), keepalive=15)
    except Exception as exc:
        return check("L5", "MQTT telemetry arriving at the fog tier", False,
                     f"broker unreachable: {exc}")
    client.subscribe(mq["topic"], qos=int(mq.get("qos", 0)))
    client.loop_start(); time.sleep(sample_s); client.loop_stop(); client.disconnect()
    expected = mq["publish_hz"] * sample_s
    ok = len(got) >= 0.8 * expected
    detail = f"{len(got)} frames in {sample_s:.0f}s (expected ~{expected:.0f})"
    if got:
        try:
            frame = json.loads(got[-1])
            required = {"seq", "ts", "h1_m", "h2_m", "pump", "valve", "alarm"}
            missing = required - set(frame)
            ok = ok and not missing
            detail += f"; schema {'complete' if not missing else 'missing ' + str(missing)}"
        except Exception as exc:
            ok = False; detail += f"; malformed payload: {exc}"
    return check("L5", "MQTT telemetry arriving at the fog tier", ok, detail)


# ------------------------------------------------------------ physics check
def physics_check(cfg, root: Path, live):
    sys.path.insert(0, str(root / "edge/twin"))
    from physics import PlantParams, analytic_equilibrium
    p = PlantParams.from_config(cfg)

    # Prefer the recorded baseline: it spans many fill/drain cycles, whereas the
    # short live window may catch the plant mid-transient.
    h1_mean = h2_mean = None
    src = "recorded baseline"
    files = sorted(glob.glob(str(root / cfg["storage"]["baseline_dir"] / "baseline_*.csv")))
    if files:
        import csv as _csv
        rows = [r for f in files for r in _csv.DictReader(open(f))]
        rows = [r for r in rows if int(r["pump"]) in (0, 1)]
        if len(rows) >= 60:
            h1_mean = sum(float(r["h1_m"]) for r in rows) / len(rows)
            h2_mean = sum(float(r["h2_m"]) for r in rows) / len(rows)
    if h1_mean is None and live and live["h1_mm"]:
        src = "live sample"
        h1_mean = sum(live["h1_mm"]) / len(live["h1_mm"]) / 1000.0
        h2_mean = sum(live["h2_mm"]) / len(live["h2_mm"]) / 1000.0
    if h1_mean is None:
        return check("P1", "physics matches closed-form equilibrium", False, "no data")
    predicted = analytic_equilibrium(p, h1_mean)
    err = abs(h2_mean - predicted) / predicted if predicted else 1.0
    return check("P1", "physics matches closed-form equilibrium h2* = (Cv12/Cv2o)^2 h1",
                 err <= 0.10,
                 f"[{src}] h1_mean={h1_mean:.3f} m, h2_mean={h2_mean:.3f} m, "
                 f"predicted={predicted:.3f} m, error={err:.1%}")


# ----------------------------------------------------------- handover check
def handover_check(cfg, root: Path, min_hours: float):
    print("\nQ3  What does Phase 2 consume from Phase 1?")
    raw = root / cfg["storage"]["baseline_dir"]
    files = sorted(glob.glob(str(raw / "baseline_*.csv")))
    if not check("H1", "benign baseline captured", bool(files),
                 f"{len(files)} file(s) in {raw}"):
        return
    import csv as _csv
    rows = 0
    t_first = t_last = None
    bad = 0
    for f in files:
        for r in _csv.DictReader(open(f)):
            try:
                ts = float(r["ts"])
                float(r["h1_m"]); float(r["h2_m"]); int(r["pump"])
            except Exception:
                bad += 1
                continue
            t_first = ts if t_first is None else min(t_first, ts)
            t_last = ts if t_last is None else max(t_last, ts)
            rows += 1
    hours = (t_last - t_first) / 3600.0 if rows > 1 else 0.0
    check("H2", f"baseline covers at least {min_hours:g} h of process time",
          hours >= min_hours, f"{rows:,} rows spanning {hours:.2f} h")
    check("H3", "no malformed rows", bad == 0, f"{bad} malformed")

    dvc_dir = (root / ".dvc").is_dir()
    check("H4", "data-version-control initialised", dvc_dir,
          f"{root / '.dvc'} present" if dvc_dir else "run: dvc init",
          critical=False)

    pointers = sorted((root / "data" / "processed").glob("baseline_*.parquet.dvc"))
    check("H5", "baseline dataset tracked by DVC", bool(pointers),
          ", ".join(x.name for x in pointers) if pointers
          else "run: dvc add data/processed/baseline_<version>.parquet && dvc push",
          critical=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/twin.yaml")
    ap.add_argument("--root", default=".")
    ap.add_argument("--sample-seconds", type=float, default=20.0)
    ap.add_argument("--min-hours", type=float, default=1.0)
    ap.add_argument("--offline", action="store_true", help="skip live service checks")
    a = ap.parse_args()

    root = Path(a.root).resolve()
    cfg = yaml.safe_load(open(a.config))
    print(f"Hybrid AI-Quantum Cognitive Security Framework Phase 1 verification  -  {datetime.now(timezone.utc).isoformat()}")
    print(f"root={root}  config={a.config}")

    artefact_checks(cfg, root)
    live = None
    if a.offline:
        print("\nQ2  (offline mode - live checks skipped)")
    else:
        live = live_checks(cfg, a.sample_seconds)
        mqtt_check(cfg, min(a.sample_seconds, 10.0))
    physics_check(cfg, root, live)
    handover_check(cfg, root, a.min_hours)

    failed = [r for r in RESULTS if not r["pass"] and r["critical"]]
    warned = [r for r in RESULTS if not r["pass"] and not r["critical"]]
    out = Path(root / "reports"); out.mkdir(exist_ok=True)
    payload = {"utc": datetime.now(timezone.utc).isoformat(), "config": a.config,
               "checks": RESULTS, "failed": len(failed), "warnings": len(warned),
               "verdict": "PASS" if not failed else "FAIL"}
    (out / "phase1_verification.json").write_text(json.dumps(payload, indent=2))

    print(f"\n{'='*66}\nVERDICT: {payload['verdict']}  "
          f"({len(RESULTS)-len(failed)-len(warned)} passed, "
          f"{len(failed)} failed, {len(warned)} warnings)")
    print(f"report written to {out/'phase1_verification.json'}\n{'='*66}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
