"""Full-trajectory diagnostics for the nonlinear-load discrepancy study.

This diagnostic intentionally does not manufacture trajectories from endpoint
predictions. Physics trajectories are evaluated directly; learned endpoint-only
models are marked unavailable until trained to predict complete trajectories.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
HERE = SCRIPT_DIR.parent
SOURCE = SCRIPT_DIR / "run_experiment.py"
RESULTS = HERE / "results"
REPORTS = HERE / "reports"
DT_S = 0.01
K = 0.25
C0 = 0.15
HORIZON = 200


def load_source():
    spec = importlib.util.spec_from_file_location("euler_discrepancy_diag_source", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rhs(omega, control, params, c2):
    return (params.mass_flow_rate_kg_s * params.radius_m * (params.swirl_gain_m_s * control - params.speed_feedback_s * omega) - C0 - c2 * omega**2) / params.inertia_kg_m2


def rk4_step(omega, control, params, dt, c2):
    k1 = rhs(omega, control, params, c2)
    k2 = rhs(omega + dt * k1 / 2, control, params, c2)
    k3 = rhs(omega + dt * k2 / 2, control, params, c2)
    k4 = rhs(omega + dt * k3, control, params, c2)
    return omega + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def truth_trajectory(omega0, control, params, c2, steps=HORIZON):
    values = [omega0]
    for _ in range(steps):
        values.append(rk4_step(values[-1], control, params, DT_S / 10, c2))
    return np.asarray(values)


def linear_trajectory(omega0, control, params, c2, steps=HORIZON, calibrated=False):
    if calibrated:
        # Fit the wrong affine model to the first 20 samples of the truth.
        observed = truth_trajectory(omega0, control, params, c2, 20)
        rho, intercept = np.linalg.lstsq(np.column_stack((observed[:-1], np.ones(20))), observed[1:], rcond=None)[0]
    else:
        rho = np.exp(-(params.mass_flow_rate_kg_s * params.radius_m * K / params.inertia_kg_m2) * DT_S)
        equilibrium = (params.mass_flow_rate_kg_s * params.radius_m * params.swirl_gain_m_s * control - C0) / (params.mass_flow_rate_kg_s * params.radius_m * K)
        intercept = equilibrium * (1 - rho)
    values = [omega0]
    for _ in range(steps):
        values.append(rho * values[-1] + intercept)
    return np.asarray(values)


def ode_residual(trajectory, control, params, c2):
    derivative = np.diff(trajectory) / DT_S
    predicted_torque = params.inertia_kg_m2 * derivative
    physical_torque = params.mass_flow_rate_kg_s * params.radius_m * (params.swirl_gain_m_s * control - params.speed_feedback_s * trajectory[:-1]) - C0 - c2 * trajectory[:-1] ** 2
    return float(np.sqrt(np.mean((predicted_torque - physical_torque) ** 2)))


def main():
    module = load_source()
    rng = np.random.default_rng(19)
    rows, horizon_rows, sensitivity_rows = [], [], []
    for epsilon in (0, 0.02, 0.05, 0.10, 0.20, 0.50):
        c2 = epsilon * C0 / (400.0**2 * max(1e-12, 1 - epsilon))
        for sequence in range(30):
            params = module.Parameters(mass_flow_rate_kg_s=float(rng.uniform(.70, .90)), inertia_kg_m2=float(rng.uniform(.017, .023)), speed_feedback_s=K)
            omega0 = float(rng.uniform(350, 450)); control = float(rng.uniform(.45, 1.05))
            truth = truth_trajectory(omega0, control, params, c2)
            for method, prediction in (("rk4_oracle", truth), ("nominal_physics", linear_trajectory(omega0, control, module.Parameters(speed_feedback_s=K), c2)), ("calibrated_physics", linear_trajectory(omega0, control, params, c2, calibrated=True))):
                rows.append({'epsilon':epsilon,'sequence_index':sequence,'method':method,'full_trajectory_rmse_rad_s':float(np.sqrt(np.mean((prediction-truth)**2))),'ode_residual_N_m':ode_residual(prediction,control,params,c2)})
                for horizon in (10, 50, 100, 200): horizon_rows.append({'epsilon':epsilon,'sequence_index':sequence,'method':method,'horizon_steps':horizon,'horizon_error_rad_s':float(abs(prediction[horizon]-truth[horizon]))})
                delta=1e-4; plus=(truth_trajectory(omega0,control+delta,params,c2)[10]-truth[10])/delta; sensitivity_rows.append({'epsilon':epsilon,'sequence_index':sequence,'method':method,'control_sensitivity_rad_s_per_control':float(plus)})
    RESULTS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    for name, data in [('discrepancy_full_trajectory_diagnostics.csv',rows),('discrepancy_horizon_errors.csv',horizon_rows),('discrepancy_control_sensitivity.csv',sensitivity_rows)]:
        with (RESULTS/name).open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=data[0].keys()); writer.writeheader(); writer.writerows(data)
    payload={'experiment':'nonlinear_load_full_trajectory_diagnostics','epsilons':[0,.02,.05,.10,.20,.50],'methods_evaluated':['rk4_oracle','nominal_physics','calibrated_physics'],'learned_methods':'not available because the existing Task 6 runner produced endpoint-only predictions; no endpoint was converted into a fabricated trajectory','rows':len(rows),'horizon_rows':len(horizon_rows),'sensitivity_rows':len(sensitivity_rows)}
    (RESULTS/'discrepancy_diagnostics.json').write_text(json.dumps(payload,indent=2),encoding='utf-8'); (REPORTS/'discrepancy_diagnostics.md').write_text('# Discrepancy full-trajectory diagnostics\n\n'+json.dumps(payload,indent=2),encoding='utf-8'); print(json.dumps(payload,indent=2))


if __name__ == '__main__': main()
