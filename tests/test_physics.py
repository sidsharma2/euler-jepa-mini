"""Verification tests for the independent remediation physics helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from physics_utils import forward_euler_step, safe_exact_step


PARAMS = dict(
    mass_flow_rate_kg_s=0.8,
    radius_m=0.25,
    inertia_kg_m2=0.02,
    swirl_gain_m_s=8.0,
    speed_feedback_m_per_rad=0.01,
    load_torque_N_m=0.15,
)


def test_exact_step_converges_to_equilibrium():
    alpha = PARAMS["mass_flow_rate_kg_s"] * PARAMS["radius_m"] * PARAMS["swirl_gain_m_s"] / PARAMS["inertia_kg_m2"]
    beta = PARAMS["mass_flow_rate_kg_s"] * PARAMS["radius_m"] * PARAMS["speed_feedback_m_per_rad"] / PARAMS["inertia_kg_m2"]
    gamma = PARAMS["load_torque_N_m"] / PARAMS["inertia_kg_m2"]
    equilibrium = (alpha * 0.8 - gamma) / beta
    omega = 350.0
    for _ in range(100000):
        omega = safe_exact_step(omega, 0.8, 0.01, **PARAMS)
    assert abs(omega - equilibrium) < 1e-8


def test_forward_euler_has_first_order_convergence():
    exact = safe_exact_step(400.0, 0.8, 0.4, **PARAMS)
    errors = []
    for subdivisions in (10, 20, 40):
        dt = 0.4 / subdivisions
        omega = 400.0
        for _ in range(subdivisions):
            omega = forward_euler_step(omega, 0.8, dt, **PARAMS)
        errors.append(abs(omega - exact))
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert 1.5 < errors[0] / errors[1] < 2.5


def test_beta_zero_uses_linear_limit():
    expected = 400.0 + (0.8 * 0.25 * 8.0 / 0.02 * 0.7 - 0.15 / 0.02) * 0.1
    actual = safe_exact_step(400.0, 0.7, 0.1, speed_feedback_m_per_rad=0.0, **{k: v for k, v in PARAMS.items() if k != "speed_feedback_m_per_rad"})
    assert actual == expected


def test_window_alignment_and_constant_control():
    sequence = np.column_stack((np.arange(200, dtype=float), np.full(200, 0.7)))
    context = sequence[:20]
    target = sequence[20:30]
    assert np.array_equal(context[-1], sequence[19])
    assert np.array_equal(target[0], sequence[20])
    assert np.intersect1d(np.arange(20), np.arange(20, 30)).size == 0
    assert np.all(target[:, 1] == context[-1, 1])


def test_beta_invariant_under_common_mass_and_inertia_scaling():
    def beta(mdot, inertia):
        return mdot * PARAMS["radius_m"] * PARAMS["speed_feedback_m_per_rad"] / inertia
    assert np.isclose(beta(0.8, 0.02), beta(1.6, 0.04))


def test_equilibrium_invariant_when_load_scales_consistently():
    def equilibrium(mdot, inertia):
        alpha = mdot * PARAMS["radius_m"] * PARAMS["swirl_gain_m_s"] / inertia
        beta = mdot * PARAMS["radius_m"] * PARAMS["speed_feedback_m_per_rad"] / inertia
        gamma = PARAMS["load_torque_N_m"] / inertia
        return (alpha * 0.8 - gamma) / beta
    # With fixed load torque, omega_eq is not invariant; scale load with flow.
    def scaled_equilibrium(mdot, inertia, load):
        alpha = mdot * PARAMS["radius_m"] * PARAMS["swirl_gain_m_s"] / inertia
        beta_value = mdot * PARAMS["radius_m"] * PARAMS["speed_feedback_m_per_rad"] / inertia
        return (alpha * 0.8 - load / inertia) / beta_value
    assert np.isclose(scaled_equilibrium(0.8, 0.02, 0.15), scaled_equilibrium(1.6, 0.04, 0.30))
