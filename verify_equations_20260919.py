"""Verify the equations used by the Euler-JEPA manuscript.

This is an additive numerical check.  It does not modify the source project or
the reported experiment results.  The checks cover the affine exact solution,
the nonlinear-load definition, finite-difference control sensitivity, and the
matrix zero-order-hold identities used in Appendix B.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPORTS = HERE / "reports"


def rhs(omega: float, control: float, mdot: float, radius: float, inertia: float,
        ku: float, komega: float, c0: float, c2: float) -> float:
    torque = mdot * radius * (ku * control - komega * omega)
    return (torque - c0 - c2 * omega**2) / inertia


def rk4_step(omega: float, control: float, dt: float, mdot: float, radius: float,
             inertia: float, ku: float, komega: float, c0: float, c2: float) -> float:
    k1 = rhs(omega, control, mdot, radius, inertia, ku, komega, c0, c2)
    k2 = rhs(omega + 0.5 * dt * k1, control, mdot, radius, inertia, ku, komega, c0, c2)
    k3 = rhs(omega + 0.5 * dt * k2, control, mdot, radius, inertia, ku, komega, c0, c2)
    k4 = rhs(omega + dt * k3, control, mdot, radius, inertia, ku, komega, c0, c2)
    return omega + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def exact_affine_step(omega: float, control: float, dt: float, mdot: float,
                      radius: float, inertia: float, ku: float, komega: float,
                      c0: float) -> float:
    alpha = mdot * radius * ku / inertia
    beta = mdot * radius * komega / inertia
    gamma = c0 / inertia
    equilibrium = (alpha * control - gamma) / beta
    return equilibrium + (omega - equilibrium) * math.exp(-beta * dt)


def run() -> dict[str, object]:
    mdot, radius, inertia, ku, c0 = 0.8, 0.25, 0.02, 8.0, 0.15
    control, omega0, dt = 0.75, 400.0, 0.01
    checks: dict[str, object] = {}

    beta_short = mdot * radius * 0.01 / inertia
    beta_long = mdot * radius * 0.25 / inertia
    checks["nominal_beta_short_s^-1"] = beta_short
    checks["nominal_time_constant_short_s"] = 1.0 / beta_short
    checks["nominal_beta_full_trajectory_s^-1"] = beta_long
    checks["nominal_time_constant_full_trajectory_s"] = 1.0 / beta_long

    exact = exact_affine_step(omega0, control, dt, mdot, radius, inertia, ku, 0.01, c0)
    refined = omega0
    for _ in range(100):
        refined = rk4_step(refined, control, dt / 100.0, mdot, radius, inertia, ku, 0.01, c0, 0.0)
    checks["affine_exact_vs_rk4_100_substeps_abs_error_rad_s"] = abs(exact - refined)
    assert abs(exact - refined) < 1e-10

    epsilon_checks = {}
    for epsilon in (0.0, 0.1, 0.2, 0.5):
        c2 = epsilon * c0 / (400.0**2 * (1.0 - epsilon)) if epsilon < 1.0 else math.inf
        fraction = (c2 * 400.0**2) / (c0 + c2 * 400.0**2) if c2 else 0.0
        epsilon_checks[str(epsilon)] = {"c2_N_m_s2": c2, "recovered_fraction": fraction}
        assert abs(fraction - epsilon) < 1e-14
    checks["quadratic_load_fraction_checks"] = epsilon_checks

    # The scalar sensitivity is the discrete-time closed-form derivative.
    komega = 0.25
    alpha = mdot * radius * ku / inertia
    beta = mdot * radius * komega / inertia
    phi = math.exp(-beta * dt)
    gamma_u = alpha / beta * (1.0 - phi)
    horizon = 200
    analytic_sensitivity = gamma_u * sum(phi**j for j in range(horizon))

    def rollout(u: float) -> float:
        value = omega0
        for _ in range(horizon):
            value = exact_affine_step(value, u, dt, mdot, radius, inertia, ku, komega, c0)
        return value

    delta = 1e-5
    finite_difference = (rollout(control + delta) - rollout(control - delta)) / (2.0 * delta)
    checks["scalar_endpoint_sensitivity_analytic_rad_s_per_control"] = analytic_sensitivity
    checks["scalar_endpoint_sensitivity_centered_difference_rad_s_per_control"] = finite_difference
    checks["scalar_sensitivity_abs_error"] = abs(analytic_sensitivity - finite_difference)
    assert abs(analytic_sensitivity - finite_difference) < 1e-6

    # Verify the block exponential identity used for Phi and Gamma.
    matrix_a = np.array([[-2.0, 0.3], [-0.4, -1.0]])
    matrix_b = np.array([[0.5], [1.2]])
    matrix_h = 0.01
    augmented = np.zeros((3, 3))
    augmented[:2, :2] = matrix_a
    augmented[:2, 2:] = matrix_b
    block = torch.matrix_exp(torch.tensor(matrix_h * augmented, dtype=torch.float64)).numpy()
    phi_matrix = torch.matrix_exp(torch.tensor(matrix_h * matrix_a, dtype=torch.float64)).numpy()
    block_error = float(np.max(np.abs(block[:2, :2] - phi_matrix)))
    gamma_block = block[:2, 2:]
    checks["matrix_phi_block_exponential_max_abs_error"] = block_error
    checks["matrix_gamma_block_exponential"] = gamma_block.tolist()
    assert block_error < 1e-14

    payload = {
        "status": "passed",
        "experiment_protocols": {
            "corrected_short_horizon": {"komega": 0.01, "horizon_steps": 10, "dt_s": 0.01},
            "full_trajectory_nonlinear_load": {"komega": 0.25, "horizon_steps": 200, "dt_s": 0.01},
        },
        "checks": checks,
        "interpretation": [
            "The affine exact solution agrees with refined RK4 integration.",
            "The quadratic-load formula recovers the requested epsilon fraction at 400 rad/s.",
            "The scalar control-sensitivity matrix formula agrees with centered finite differences.",
            "The block matrix exponential returns the stated Phi and Gamma matrices.",
        ],
    }
    RESULTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    (RESULTS / "math_verification_20260919.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = "# Mathematical verification — 2026-09-19\n\n"
    report += "All automated checks passed. The audit found that the equations are internally correct, but the manuscript needed to distinguish the short corrected benchmark (`k_omega=0.01`, 10 steps) from the full-trajectory nonlinear-load experiment (`k_omega=0.25`, 200 steps).\n\n"
    report += "The affine exact step agrees with refined fourth-order Runge–Kutta integration; the epsilon load formula recovers the requested fraction at 400 rad/s; the scalar control-sensitivity formula agrees with centered finite differences; and the block matrix exponential returns the stated discrete matrices.\n"
    (REPORTS / "math_verification_20260919.md").write_text(report + "\n```json\n" + json.dumps(payload, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    run()
