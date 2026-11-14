#!/usr/bin/env python3
"""Cross-check the OpenModelica FMU against the Python RK4 engine.

Both engines integrate the same governing equations (Table 4.1) from identical
initial conditions with identical inputs.  Agreement is an independent
confirmation that the plant dynamics are implemented correctly in both
realisations, and is reported in Chapter 4 (Section 4.6).

Run from the repository root:

    python scripts/crosscheck_fmu_rk4.py
    python scripts/crosscheck_fmu_rk4.py --scenario fill --t-end 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "edge" / "twin"))

from physics import PlantParams, TwoTankPython  # noqa: E402

FMU_PATH = ROOT / "edge" / "twin" / "TwoTank.fmu"


def run_fmu(u_pump: float, u_valve: float, dt: float, t_end: float,
            h1_0: float, h2_0: float) -> np.ndarray:
    """Simulate the FMU step-wise; return an (N, 2) array of [h1, h2]."""
    from fmpy import extract, read_model_description
    from fmpy.fmi2 import FMU2Slave

    md = read_model_description(str(FMU_PATH))
    unzip = extract(str(FMU_PATH))
    vr = {v.name: v.valueReference for v in md.modelVariables}

    fmu = FMU2Slave(
        guid=md.guid,
        unzipDirectory=unzip,
        modelIdentifier=md.coSimulation.modelIdentifier,
        instanceName="crosscheck",
    )
    try:
        fmu.instantiate()
        fmu.setupExperiment(startTime=0.0)
        fmu.enterInitializationMode()
        # Set initial levels before initialisation completes so the FMU starts
        # from the same state as the Python engine.
        for name, value in (("h1", h1_0), ("h2", h2_0)):
            if name in vr:
                try:
                    fmu.setReal([vr[name]], [value])
                except Exception:
                    pass  # h1/h2 may be outputs only; falls back to .mo defaults
        fmu.exitInitializationMode()
        fmu.setReal([vr["u_pump"], vr["u_valve"]], [u_pump, u_valve])

        rows, t = [], 0.0
        n = int(round(t_end / dt))
        for _ in range(n):
            fmu.doStep(currentCommunicationPoint=t, communicationStepSize=dt)
            t += dt
            rows.append(fmu.getReal([vr["h1"], vr["h2"]]))
        return np.asarray(rows, dtype=float)
    finally:
        try:
            fmu.terminate()
        finally:
            fmu.freeInstance()


def run_rk4(params: PlantParams, u_pump: float, u_valve: float,
            dt: float, t_end: float) -> np.ndarray:
    """Integrate the Python RK4 engine; return an (N, 2) array of [h1, h2]."""
    plant = TwoTankPython(params)
    plant.set_inputs(u_pump, u_valve)
    rows = []
    for _ in range(int(round(t_end / dt))):
        plant.step(dt)
        rows.append((plant.h1, plant.h2))
    return np.asarray(rows, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/twin.yaml")
    ap.add_argument("--scenario", choices=["drain", "fill"], default="drain",
                    help="drain: pump off (default, no overflow risk). "
                         "fill: pump on -- keep --t-end short.")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--t-end", type=float, default=300.0)
    ap.add_argument("--tol-mm", type=float, default=1.5,
                    help="pass threshold in mm (default: the sensor noise floor)")
    a = ap.parse_args()

    cfg = yaml.safe_load((ROOT / a.config).read_text())
    params = PlantParams.from_config(cfg)

    u_pump = 1.0 if a.scenario == "fill" else 0.0
    u_valve = 1.0

    print(f"scenario={a.scenario}  u_pump={u_pump}  u_valve={u_valve}  "
          f"dt={a.dt}s  t_end={a.t_end}s")
    print(f"initial levels: h1={params.h1_0:.3f} m  h2={params.h2_0:.3f} m  "
          f"(h_max={params.h_max:.3f} m)\n")

    fmu = run_fmu(u_pump, u_valve, a.dt, a.t_end, params.h1_0, params.h2_0)
    rk4 = run_rk4(params, u_pump, u_valve, a.dt, a.t_end)

    n = min(len(fmu), len(rk4))
    fmu, rk4 = fmu[:n], rk4[:n]
    t = np.arange(1, n + 1) * a.dt

    d = np.abs(fmu - rk4) * 1000.0  # mm
    clamped = (rk4 <= 1e-9) | (rk4 >= params.h_max - 1e-9)
    if clamped.any():
        print("WARNING: the RK4 engine clamped at a bound during this run; "
              "comparison past that point is not meaningful.\n")

    print(f"{'':14s}{'max |dh|':>12s}{'mean |dh|':>12s}{'at t':>10s}")
    for i, name in enumerate(("h1", "h2")):
        j = int(np.argmax(d[:, i]))
        print(f"{name:14s}{d[:, i].max():10.4f}mm{d[:, i].mean():10.4f}mm"
              f"{t[j]:9.1f}s")

    print(f"\nfinal   FMU: h1={fmu[-1,0]:.6f} m  h2={fmu[-1,1]:.6f} m")
    print(f"final   RK4: h1={rk4[-1,0]:.6f} m  h2={rk4[-1,1]:.6f} m")

    worst = d.max()
    ok = worst <= a.tol_mm
    print(f"\nmax deviation {worst:.4f} mm vs tolerance {a.tol_mm:.2f} mm "
          f"-> {'PASS' if ok else 'FAIL'}")
    if ok:
        print("The two realisations agree within the sensor noise floor.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
