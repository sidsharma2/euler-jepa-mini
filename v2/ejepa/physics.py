"""Euler-work rotor with a lagged, delayed actuator.

Truth model (SI units, per episode i with hidden parameters theta_i):

    J d(omega)/dt = mdot r (k_u a - k_w omega) - c0 - c2 omega^2      (Euler torque - load)
    tau_a da/dt   = u(t - d dt) - a                                    (actuator lag + transport delay)

omega  rotor speed [rad/s]          a     actuator position (swirl command) [-]
u      commanded control [-]        J     inertia [kg m^2]
mdot   mass flow [kg/s]             r     representative radius [m]
k_u    swirl gain [m/s]             k_w   speed feedback [m/(s rad/s)] = [m]
c0     constant load [N m]          c2    quadratic load [N m s^2]
tau_a  actuator time constant [s]   d     transport delay [samples]

The command is zero-order-held over each sample dt and integrated with RK4 on
`substeps` internal steps.  Everything is vectorised over episodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RotorConstants:
    radius_m: float = 0.25
    swirl_gain_m_s: float = 100.0
    load_torque_N_m: float = 0.15
    blade_count: int = 11
    dt_s: float = 0.01
    substeps: int = 10
    u_min: float = 0.25
    u_max: float = 0.95


CONST = RotorConstants()

# Hidden-parameter ranges.  "id" is the training distribution; "ood" moves the
# quadratic load and inertia outside it (held-out dynamical regime).
PARAM_RANGES = {
    "id": {
        "mass_flow_kg_s": (0.70, 0.90),
        "inertia_kg_m2": (0.017, 0.023),
        "speed_feedback_m": (0.12, 0.18),
        "quad_load_N_m_s2": (0.0, 1.5e-5),
        "actuator_tau_s": (0.03, 0.25),
        "delay_steps": (0, 4),
    },
    "ood": {
        "mass_flow_kg_s": (0.70, 0.90),
        "inertia_kg_m2": (0.023, 0.028),
        "speed_feedback_m": (0.12, 0.18),
        "quad_load_N_m_s2": (1.5e-5, 3.0e-5),
        "actuator_tau_s": (0.03, 0.25),
        "delay_steps": (0, 4),
    },
}

PARAM_NAMES = ("mass_flow_kg_s", "inertia_kg_m2", "speed_feedback_m",
               "quad_load_N_m_s2", "actuator_tau_s", "delay_steps")


@dataclass
class EpisodeParams:
    mass_flow_kg_s: np.ndarray
    inertia_kg_m2: np.ndarray
    speed_feedback_m: np.ndarray
    quad_load_N_m_s2: np.ndarray
    actuator_tau_s: np.ndarray
    delay_steps: np.ndarray
    const: RotorConstants = field(default=CONST)

    @property
    def n(self) -> int:
        return len(self.mass_flow_kg_s)

    def as_matrix(self) -> np.ndarray:
        return np.column_stack([getattr(self, k).astype(np.float64) for k in PARAM_NAMES])

    def subset(self, index) -> "EpisodeParams":
        return EpisodeParams(*(getattr(self, k)[index] for k in PARAM_NAMES), const=self.const)

    @staticmethod
    def from_matrix(matrix: np.ndarray, const: RotorConstants = CONST) -> "EpisodeParams":
        cols = [matrix[:, i] for i in range(matrix.shape[1])]
        cols[-1] = cols[-1].astype(np.int64)
        return EpisodeParams(*cols, const=const)


def sample_params(n: int, rng: np.random.Generator, regime: str = "id") -> EpisodeParams:
    ranges = PARAM_RANGES[regime]
    values = {}
    for name in PARAM_NAMES:
        low, high = ranges[name]
        if name == "delay_steps":
            values[name] = rng.integers(low, high + 1, size=n)
        else:
            values[name] = rng.uniform(low, high, size=n)
    return EpisodeParams(**values)


def nominal_params(n: int = 1) -> EpisodeParams:
    """Mid-range 'catalogue' parameters, used by the nominal-physics baseline."""
    mid = {k: 0.5 * (lo + hi) for k, (lo, hi) in PARAM_RANGES["id"].items()}
    return EpisodeParams(**{k: np.full(n, mid[k]) for k in PARAM_NAMES[:-1]},
                         delay_steps=np.full(n, 2, dtype=np.int64))


def euler_torque(p: EpisodeParams, omega, a):
    """Fluid (Euler) torque tau_E = mdot r Delta C_theta [N m]."""
    c = p.const
    return p.mass_flow_kg_s * c.radius_m * (c.swirl_gain_m_s * a - p.speed_feedback_m * omega)


def load_torque(p: EpisodeParams, omega):
    return p.const.load_torque_N_m + p.quad_load_N_m_s2 * omega * omega


def rhs(p: EpisodeParams, omega, a, u_eff):
    domega = (euler_torque(p, omega, a) - load_torque(p, omega)) / p.inertia_kg_m2
    da = (u_eff - a) / p.actuator_tau_s
    return domega, da


def rk4_sample(p: EpisodeParams, omega, a, u_eff):
    """Advance (omega, a) by one sample dt under zero-order-held u_eff."""
    h = p.const.dt_s / p.const.substeps
    for _ in range(p.const.substeps):
        k1w, k1a = rhs(p, omega, a, u_eff)
        k2w, k2a = rhs(p, omega + 0.5 * h * k1w, a + 0.5 * h * k1a, u_eff)
        k3w, k3a = rhs(p, omega + 0.5 * h * k2w, a + 0.5 * h * k2a, u_eff)
        k4w, k4a = rhs(p, omega + h * k3w, a + h * k3a, u_eff)
        omega = omega + h * (k1w + 2 * k2w + 2 * k3w + k4w) / 6
        a = a + h * (k1a + 2 * k2a + 2 * k3a + k4a) / 6
    return omega, a


def equilibrium_speed(p: EpisodeParams, a):
    """Steady omega for actuator position a: c2 w^2 + mdot r k_w w + c0 - mdot r k_u a = 0."""
    c = p.const
    b = p.mass_flow_kg_s * c.radius_m * p.speed_feedback_m
    rhs_const = p.mass_flow_kg_s * c.radius_m * c.swirl_gain_m_s * a - c.load_torque_N_m
    c2 = p.quad_load_N_m_s2
    linear = rhs_const / b
    safe = np.maximum(c2, 1e-30)
    quad = (-b + np.sqrt(b * b + 4 * safe * rhs_const)) / (2 * safe)
    return np.where(c2 > 1e-12, quad, linear)


class RotorSim:
    """Stateful, vectorised simulator used both for datasets and closed-loop control."""

    def __init__(self, p: EpisodeParams, omega0, a0, u_hist0):
        self.p = p
        self.omega = np.asarray(omega0, dtype=np.float64).copy()
        self.a = np.asarray(a0, dtype=np.float64).copy()
        dmax = int(PARAM_RANGES["ood"]["delay_steps"][1]) + 1
        # Command pipeline: buffer[:, j] is the command issued j samples ago.
        self.buffer = np.repeat(np.asarray(u_hist0, dtype=np.float64)[:, None], dmax + 1, axis=1)
        self.rows = np.arange(p.n)

    def step(self, u_cmd):
        u_cmd = np.clip(np.asarray(u_cmd, dtype=np.float64), self.p.const.u_min, self.p.const.u_max)
        self.buffer = np.roll(self.buffer, 1, axis=1)
        self.buffer[:, 0] = u_cmd
        u_eff = self.buffer[self.rows, self.p.delay_steps]
        self.omega, self.a = rk4_sample(self.p, self.omega, self.a, u_eff)
        return self.omega.copy(), self.a.copy()


def initial_state(p: EpisodeParams, rng: np.random.Generator):
    u0 = rng.uniform(0.35, 0.85, size=p.n)
    omega0 = equilibrium_speed(p, u0) * (1.0 + 0.03 * rng.standard_normal(p.n))
    return omega0, u0.copy(), u0


def excitation_commands(n: int, steps: int, rng: np.random.Generator, u0=None) -> np.ndarray:
    """Piecewise-constant / ramp command sequences (system-identification style excitation)."""
    c = CONST
    u = np.empty((n, steps))
    for i in range(n):
        t, level = 0, (u0[i] if u0 is not None else rng.uniform(c.u_min, c.u_max))
        while t < steps:
            hold = int(rng.uniform(0.15, 0.8) / c.dt_s)
            nxt = rng.uniform(c.u_min, c.u_max)
            seg = min(hold, steps - t)
            if rng.random() < 0.3:
                u[i, t:t + seg] = np.linspace(level, nxt, seg)
            else:
                u[i, t:t + seg] = nxt
            level, t = nxt, t + seg
    return u


def simulate(p: EpisodeParams, u_cmd: np.ndarray, omega0, a0, u_hist0):
    """Open-loop simulation; returns omega, a with shape (N, T+1) (index 0 = initial)."""
    sim = RotorSim(p, omega0, a0, u_hist0)
    n, steps = u_cmd.shape
    omega = np.empty((n, steps + 1)); a = np.empty((n, steps + 1))
    omega[:, 0], a[:, 0] = sim.omega, sim.a
    for k in range(steps):
        omega[:, k + 1], a[:, k + 1] = sim.step(u_cmd[:, k])
    return omega, a
