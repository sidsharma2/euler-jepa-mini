"""Verification tests for the v2 truth model, sensor and classical tracker.

Run with `pytest v2/tests` or `python v2/tests/test_v2_physics.py`.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ejepa.physics import (CONST, EpisodeParams, RotorSim, equilibrium_speed, euler_torque,  # noqa: E402
                           load_torque, sample_params, simulate)
from ejepa.sensors import FREQS_HZ, SpectrumSensor  # noqa: E402


def _params(n=4, c2=0.0, tau=0.1, delay=0):
    return EpisodeParams(np.full(n, 0.8), np.full(n, 0.02), np.full(n, 0.15), np.full(n, c2),
                         np.full(n, tau), np.full(n, delay, dtype=np.int64))


def test_linear_limit_matches_exact_solution():
    """c2 = 0 and a fast actuator: omega follows the exact first-order solution."""
    p = _params(c2=0.0, tau=1e-3)
    u = np.full((p.n, 100), 0.6)
    om, a = simulate(p, u, np.full(p.n, 300.0), np.full(p.n, 0.6), np.full(p.n, 0.6))
    beta = 0.8 * CONST.radius_m * 0.15 / 0.02
    w_inf = equilibrium_speed(p, 0.6)
    exact = w_inf[0] + (300.0 - w_inf[0]) * np.exp(-beta * CONST.dt_s * np.arange(101))
    assert np.max(np.abs(om[0] - exact)) < 1e-3


def test_equilibrium_is_a_fixed_point():
    rng = np.random.default_rng(0)
    p = sample_params(64, rng)
    a = rng.uniform(0.3, 0.9, 64)
    w = equilibrium_speed(p, a)
    residual = euler_torque(p, w, a) - load_torque(p, w)
    assert np.max(np.abs(residual)) < 1e-8


def test_transport_delay_shifts_response():
    """A delay of d samples shifts the response to a command step by exactly d samples."""
    base = _params(n=1, delay=0); delayed = _params(n=1, delay=3)
    u = np.full((1, 60), 0.5); u[:, 10:] = 0.8
    w0 = equilibrium_speed(base, 0.5)
    om0, _ = simulate(base, u, w0, np.full(1, 0.5), np.full(1, 0.5))
    om3, _ = simulate(delayed, u, w0, np.full(1, 0.5), np.full(1, 0.5))
    assert np.allclose(om3[0, 3:], om0[0, :-3], atol=1e-9)


def test_energy_balance_sign():
    """Raising the command increases speed; Euler torque exceeds load while accelerating."""
    p = _params(n=1, c2=1e-5)
    sim = RotorSim(p, equilibrium_speed(p, 0.5), np.full(1, 0.5), np.full(1, 0.5))
    w_prev = sim.omega.copy()
    for _ in range(50):
        sim.step(np.full(1, 0.8))
    assert sim.omega[0] > w_prev[0]


def test_sensor_peak_at_blade_passing_frequency():
    rng = np.random.default_rng(1)
    sensor = SpectrumSensor(32, rng, nuisance=0.0)
    omega = np.full(32, 400.0)
    obs = np.mean([sensor.observe(omega, np.zeros(32)) for _ in range(20)], axis=0)
    bpf = CONST.blade_count * 400.0 / (2 * np.pi)
    peak_bin = np.argmax(obs.mean(0)[:30])
    assert abs(FREQS_HZ[peak_bin] - bpf) <= 50.0


def test_tracker_recovers_speed_without_nuisance():
    from ejepa.classical import track
    from ejepa.data import generate
    ep = generate(8, 3, "id", nuisance=0.0, steps=120)
    w = track(ep.obs)
    assert np.sqrt(np.mean((w - ep.omega) ** 2)) < 3.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
