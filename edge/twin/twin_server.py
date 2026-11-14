#!/usr/bin/env python3
"""Hybrid AI-Quantum Cognitive Security Framework Phase 1 - digital-twin process (edge tier).

Runs the two-tank plant model in real time and exposes it on the network as a
Modbus/TCP *slave*, exactly as a real instrumented skid would appear to a PLC:

    Input registers 0..4  (read by the PLC)  : levels and flows
    Coils           0..2  (written by the PLC): pump, valve V12, alarm lamp

OpenPLC is configured (Slave Devices -> Generic Modbus TCP Device) to poll this
server, which maps the registers to %IW100..%IW104 and the coils to
%QX100.0..%QX100.2 inside the ST program.

Two engines are available:
    --engine fmu     FMI 2.0 co-simulation of TwoTank.fmu via fmpy  (default)
    --engine python  RK4 integration of the same equations          (fallback)

Usage
-----
    python edge/twin/twin_server.py --config configs/twin.yaml
    python edge/twin/twin_server.py --config configs/twin.yaml \
           --engine python --realtime-factor 60 --duration 3600
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import random
import signal
import sys
import threading
import time
from pathlib import Path

import yaml
from pymodbus.datastore import (ModbusSequentialDataBlock, ModbusServerContext,
                                ModbusSlaveContext)
from pymodbus.server import StartTcpServer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from physics import PlantParams, TwoTankPython  # noqa: E402

LOG = logging.getLogger("twin")

# Modbus function-code keys used by pymodbus datastore
CO, IR = 1, 4                # Modbus function codes: read coils / read input registers

# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------


class FMUEngine:
    """FMI 2.0 co-simulation wrapper around TwoTank.fmu (via fmpy)."""

    def __init__(self, fmu_path: str, p: PlantParams):
        from fmpy import extract, read_model_description
        from fmpy.fmi2 import FMU2Slave

        fmu_path = str(fmu_path)
        if not os.path.exists(fmu_path):
            raise FileNotFoundError(
                f"{fmu_path} not found. Export it first:  cd edge/twin && omc export_fmu.mos"
            )
        self.md = read_model_description(fmu_path)
        self.vr = {v.name: v.valueReference for v in self.md.modelVariables}
        for name in ("u_pump", "u_valve", "h1", "h2", "q_in", "q12", "q_out"):
            if name not in self.vr:
                raise KeyError(f"variable '{name}' missing from FMU - re-export TwoTank.mo")
        self.unzipdir = extract(fmu_path)
        self.fmu = FMU2Slave(
            guid=self.md.guid,
            unzipDirectory=self.unzipdir,
            modelIdentifier=self.md.coSimulation.modelIdentifier,
            instanceName="two_tank_twin",
        )
        self.fmu.instantiate()
        self.fmu.setupExperiment(startTime=0.0)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()
        self.t = 0.0
        self.p = p
        LOG.info("FMU engine ready (%s, FMI %s)", self.md.modelName, self.md.fmiVersion)

    def set_inputs(self, u_pump: float, u_valve: float) -> None:
        self.fmu.setReal([self.vr["u_pump"], self.vr["u_valve"]],
                         [float(bool(u_pump)), float(bool(u_valve))])

    def step(self, dt: float) -> None:
        self.fmu.doStep(currentCommunicationPoint=self.t, communicationStepSize=dt)
        self.t += dt

    def outputs(self) -> dict:
        vals = self.fmu.getReal([self.vr[n] for n in ("h1", "h2", "q_in", "q12", "q_out")])
        return dict(zip(("h1", "h2", "q_in", "q12", "q_out"), vals))

    def close(self) -> None:
        try:
            self.fmu.terminate()
            self.fmu.freeInstance()
        except Exception:  # pragma: no cover - best effort on shutdown
            pass


class PythonEngine:
    """RK4 fallback with the identical model (see physics.py)."""

    def __init__(self, p: PlantParams):
        self.m = TwoTankPython(p)
        self.p = p
        LOG.info("Python RK4 engine ready (fallback mode)")

    def set_inputs(self, u_pump, u_valve): self.m.set_inputs(u_pump, u_valve)
    def step(self, dt): self.m.step(dt)
    def outputs(self): return self.m.outputs()
    def close(self): pass

    @property
    def t(self): return self.m.t


# ---------------------------------------------------------------------------
# Sensor model
# ---------------------------------------------------------------------------


class SensorModel:
    """Adds Gaussian noise and transducer quantisation to the true state."""

    def __init__(self, cfg: dict, seed: int = 42):
        s = cfg.get("sensors", {})
        self.level_std = float(s.get("level_noise_std", 0.0))
        self.flow_std = float(s.get("flow_noise_std", 0.0))
        self.quant_mm = int(s.get("quantisation_mm", 1))
        self.rng = random.Random(seed)

    def level_mm(self, h_true: float) -> int:
        h = h_true + self.rng.gauss(0.0, self.level_std)
        mm = int(round(h * 1000.0 / self.quant_mm) * self.quant_mm)
        return max(0, min(mm, 65535))

    def flow_clpm(self, q_true: float) -> int:
        """m3/s -> centilitres per minute (q * 60000 L/min * 100)."""
        q = q_true + self.rng.gauss(0.0, self.flow_std)
        return max(0, min(int(round(q * 6_000_000.0)), 65535))


# ---------------------------------------------------------------------------
# Twin process
# ---------------------------------------------------------------------------


class DigitalTwin:
    def __init__(self, cfg: dict, overrides: dict):
        self.cfg = cfg
        run = cfg["run"]
        self.dt = float(overrides.get("dt") or run["dt"])
        self.rtf = float(overrides.get("realtime_factor") or run["realtime_factor"])
        self.duration = float(overrides.get("duration") if overrides.get("duration") is not None
                              else run.get("duration", 0))
        engine = overrides.get("engine") or run["engine"]
        self.params = PlantParams.from_config(cfg)
        self.stop = threading.Event()

        if engine == "fmu":
            self.engine = FMUEngine(run["fmu_path"], self.params)
        elif engine == "python":
            self.engine = PythonEngine(self.params)
        else:
            raise ValueError(f"unknown engine '{engine}' (use fmu or python)")

        self.sensors = SensorModel(cfg, seed=int(run.get("seed", 42)))

        # ---- Modbus datastore ------------------------------------------------
        mb = cfg["modbus"]
        defaults = list(mb.get("coil_defaults", [0, 1, 0]))
        self.slave = ModbusSlaveContext(
            co=ModbusSequentialDataBlock(0, defaults + [0] * 13),
            di=ModbusSequentialDataBlock(0, [0] * 16),
            hr=ModbusSequentialDataBlock(0, [0] * 16),
            ir=ModbusSequentialDataBlock(0, [0] * 16),
            zero_mode=True,
        )
        self.context = ModbusServerContext(slaves={int(mb["unit_id"]): self.slave},
                                           single=False)
        self.unit = int(mb["unit_id"])
        self.host, self.port = mb["host"], int(mb["port"])
        self.steps = 0

    # ------------------------------------------------------------------ modbus
    def read_coils(self):
        vals = self.slave.getValues(CO, 0, count=3)
        return bool(vals[0]), bool(vals[1]), bool(vals[2])

    def write_registers(self, o: dict):
        regs = [
            self.sensors.level_mm(o["h1"]),
            self.sensors.level_mm(o["h2"]),
            self.sensors.flow_clpm(o["q_in"]),
            self.sensors.flow_clpm(o["q12"]),
            self.sensors.flow_clpm(o["q_out"]),
        ]
        self.slave.setValues(IR, 0, regs)

    # -------------------------------------------------------------------- loop
    def run_simulation(self):
        LOG.info("simulation loop: dt=%.3fs realtime_factor=%.1f duration=%s",
                 self.dt, self.rtf, self.duration or "unbounded")
        wall_start = time.perf_counter()
        sim_t = 0.0
        # prime the registers so the PLC sees valid data on its first poll
        self.write_registers(self.engine.outputs())
        while not self.stop.is_set():
            pump, valve, _alarm = self.read_coils()
            self.engine.set_inputs(pump, valve)
            self.engine.step(self.dt)
            out = self.engine.outputs()
            self.write_registers(out)
            sim_t += self.dt
            self.steps += 1

            if self.steps % int(max(1, 60 / self.dt)) == 0:
                LOG.info("t=%7.1fs  h1=%.3fm h2=%.3fm  pump=%d valve=%d",
                         sim_t, out["h1"], out["h2"], int(pump), int(valve))

            if self.duration and sim_t >= self.duration:
                LOG.info("requested duration reached (%.0f s of process time)", sim_t)
                break

            target = wall_start + sim_t / self.rtf
            slack = target - time.perf_counter()
            if slack > 0:
                self.stop.wait(slack)
            elif slack < -1.0:
                LOG.warning("simulation is %.1fs behind real time - lower realtime_factor", -slack)
        self.engine.close()
        LOG.info("simulation stopped after %d steps (%.1f s process time)", self.steps, sim_t)

    def serve_forever(self):
        sim = threading.Thread(target=self.run_simulation, name="sim", daemon=True)
        sim.start()

        def _shutdown(signum, _frame):
            LOG.info("signal %s received - shutting down", signum)
            self.stop.set()
        for s in (signal.SIGINT, signal.SIGTERM):
            signal.signal(s, _shutdown)

        LOG.info("Modbus/TCP slave listening on %s:%d (unit %d)", self.host, self.port, self.unit)
        watchdog = threading.Thread(target=self._watchdog, daemon=True)
        watchdog.start()
        StartTcpServer(context=self.context, address=(self.host, self.port))

    def _watchdog(self):
        """Stop the Modbus server once the simulation thread has finished."""
        while not self.stop.is_set():
            time.sleep(0.5)
        time.sleep(0.5)
        os._exit(0)


# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description="Hybrid AI-Quantum Cognitive Security Framework Phase 1 digital twin")
    ap.add_argument("--config", default="configs/twin.yaml")
    ap.add_argument("--engine", choices=["fmu", "python"], default=None)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--realtime-factor", dest="realtime_factor", type=float, default=None)
    ap.add_argument("--duration", type=float, default=None,
                    help="process seconds to simulate; 0 = unbounded")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    cfg = yaml.safe_load(open(a.config))
    if a.port:
        cfg["modbus"]["port"] = a.port
    twin = DigitalTwin(cfg, vars(a))
    twin.serve_forever()


if __name__ == "__main__":
    main()
