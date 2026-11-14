"""Pure-Python reference implementation of the TwoTank plant.

This is a *fallback and cross-check* for edge/twin/TwoTank.mo. It integrates
exactly the same equations with a fixed-step RK4 scheme, so that:

  * Phase 1 can proceed if the OpenModelica toolchain is unavailable, and
  * the FMU can be validated against an independent implementation
    (scripts/verify_phase1.py compares the two trajectories).

Equations (Torricelli):
    q_in  = Qpump * u_pump
    q12   = u_valve * Cv12 * sqrt(2 g h1)
    q_out = Cv2o * sqrt(2 g h2)
    dh1/dt = (q_in - q12) / A1
    dh2/dt = (q12 - q_out) / A2
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PlantParams:
    A1: float = 0.0154
    A2: float = 0.0154
    Cv12: float = 1.2e-5
    Cv2o: float = 1.2e-5
    Qpump: float = 6.0e-5
    g: float = 9.81
    h1_0: float = 0.30
    h2_0: float = 0.30
    h_max: float = 0.60

    @classmethod
    def from_config(cls, cfg: dict) -> "PlantParams":
        p = cfg.get("plant", {})
        return cls(**{k: float(v) for k, v in p.items() if k in cls.__annotations__})


class TwoTankPython:
    """Fixed-step RK4 integrator for the two-tank plant."""

    def __init__(self, p: PlantParams):
        self.p = p
        self.h1 = p.h1_0
        self.h2 = p.h2_0
        self.t = 0.0
        self.u_pump = 0.0
        self.u_valve = 1.0

    # ------------------------------------------------------------------ model
    def _flows(self, h1: float, h2: float):
        p = self.p
        h1 = max(h1, 1e-9)
        h2 = max(h2, 1e-9)
        q_in = p.Qpump * self.u_pump
        q12 = self.u_valve * p.Cv12 * math.sqrt(2.0 * p.g * h1)
        q_out = p.Cv2o * math.sqrt(2.0 * p.g * h2)
        return q_in, q12, q_out

    def _deriv(self, h1: float, h2: float):
        q_in, q12, q_out = self._flows(h1, h2)
        return (q_in - q12) / self.p.A1, (q12 - q_out) / self.p.A2

    # ------------------------------------------------------------------- step
    def set_inputs(self, u_pump: float, u_valve: float) -> None:
        self.u_pump = 1.0 if u_pump else 0.0
        self.u_valve = 1.0 if u_valve else 0.0

    def step(self, dt: float) -> None:
        h1, h2 = self.h1, self.h2
        k1 = self._deriv(h1, h2)
        k2 = self._deriv(h1 + 0.5 * dt * k1[0], h2 + 0.5 * dt * k1[1])
        k3 = self._deriv(h1 + 0.5 * dt * k2[0], h2 + 0.5 * dt * k2[1])
        k4 = self._deriv(h1 + dt * k3[0], h2 + dt * k3[1])
        self.h1 = h1 + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        self.h2 = h2 + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        # physical bounds: tanks cannot go negative and overflow at h_max
        self.h1 = min(max(self.h1, 0.0), self.p.h_max)
        self.h2 = min(max(self.h2, 0.0), self.p.h_max)
        self.t += dt

    # ---------------------------------------------------------------- outputs
    def outputs(self) -> dict:
        q_in, q12, q_out = self._flows(self.h1, self.h2)
        return {"h1": self.h1, "h2": self.h2, "q_in": q_in, "q12": q12, "q_out": q_out}


def analytic_equilibrium(p: PlantParams, h1_mean: float) -> float:
    """Steady-state level of tank 2 for a given mean level in tank 1.

    At equilibrium q12 = q_out, hence
        Cv12 * sqrt(2 g h1) = Cv2o * sqrt(2 g h2)
        h2* = (Cv12 / Cv2o)^2 * h1
    This closed form is the validation target used in Chapter 4.
    """
    return (p.Cv12 / p.Cv2o) ** 2 * h1_mean
