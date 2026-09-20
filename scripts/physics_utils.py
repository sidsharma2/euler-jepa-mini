"""Small independent physics helpers for remediation tests and extensions."""

from __future__ import annotations

import math


def safe_exact_step(
    omega_rad_s: float,
    control: float,
    dt_s: float,
    mass_flow_rate_kg_s: float,
    radius_m: float,
    inertia_kg_m2: float,
    swirl_gain_m_s: float,
    speed_feedback_m_per_rad: float,
    load_torque_N_m: float,
) -> float:
    """Advance the affine rotor ODE, including the beta -> 0 limit."""
    alpha = mass_flow_rate_kg_s * radius_m * swirl_gain_m_s / inertia_kg_m2
    beta = mass_flow_rate_kg_s * radius_m * speed_feedback_m_per_rad / inertia_kg_m2
    gamma = load_torque_N_m / inertia_kg_m2
    forcing = alpha * control - gamma
    if abs(beta) < 1e-12:
        return omega_rad_s + forcing * dt_s
    equilibrium = forcing / beta
    return equilibrium + (omega_rad_s - equilibrium) * math.exp(-beta * dt_s)


def forward_euler_step(
    omega_rad_s: float,
    control: float,
    dt_s: float,
    mass_flow_rate_kg_s: float,
    radius_m: float,
    inertia_kg_m2: float,
    swirl_gain_m_s: float,
    speed_feedback_m_per_rad: float,
    load_torque_N_m: float,
) -> float:
    """Advance the same affine ODE with explicit Euler."""
    torque_N_m = mass_flow_rate_kg_s * radius_m * (swirl_gain_m_s * control - speed_feedback_m_per_rad * omega_rad_s)
    return omega_rad_s + dt_s * (torque_N_m - load_torque_N_m) / inertia_kg_m2
