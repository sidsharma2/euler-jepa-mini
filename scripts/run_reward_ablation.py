"""Reward/cost ablation using the verified analytical Euler simulator.

This is a control-objective experiment, not RL training. Random shooting is
used with the exact known dynamics to show how reward terms change the selected
control sequence before a learned world model or policy is introduced.
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
ORIGINAL_SCRIPT = SCRIPT_DIR / "run_experiment.py"
RESULTS = HERE / "results"


def load_module():
    spec = importlib.util.spec_from_file_location("original_euler_experiment", ORIGINAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load original experiment: {ORIGINAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def simulate_batch(module, params, initial_omega: float, controls: np.ndarray, dt_s: float) -> np.ndarray:
    """Vectorized exact simulation for candidate control sequences."""
    alpha = params.mass_flow_rate_kg_s * params.radius_m * params.swirl_gain_m_s / params.inertia_kg_m2
    beta = params.mass_flow_rate_kg_s * params.radius_m * params.speed_feedback_s / params.inertia_kg_m2
    gamma = params.load_torque_N_m / params.inertia_kg_m2
    equilibrium = (alpha * controls - gamma) / beta
    omega = np.empty((controls.shape[0], controls.shape[1] + 1), dtype=float)
    omega[:, 0] = initial_omega
    for step in range(controls.shape[1]):
        omega[:, step + 1] = equilibrium[:, step] + (omega[:, step] - equilibrium[:, step]) * np.exp(-beta * dt_s)
    return omega


def main() -> None:
    module = load_module()
    params = module.Parameters()
    rng = np.random.default_rng(11)
    candidates = rng.uniform(0.45, 1.05, size=(50000, 40))
    omega = simulate_batch(module, params, initial_omega=400.0, controls=candidates, dt_s=0.01)
    target = 430.0
    tracking = np.mean(((omega[:, 1:] - target) / 30.0) ** 2, axis=1)
    terminal = ((omega[:, -1] - target) / 30.0) ** 2
    effort = np.mean(((candidates - 0.75) / 0.30) ** 2, axis=1)
    rate = np.mean(np.diff(candidates, axis=1) ** 2 / 0.30**2, axis=1)
    constraint = np.mean(np.maximum(0.0, np.abs(omega[:, 1:] - 550.0) - 150.0) ** 2 / 30.0**2, axis=1)
    objective_terms = {
        "tracking_only": (tracking + 2.0 * terminal, 0.0, 0.0, 0.0),
        "tracking_plus_effort": (tracking + 2.0 * terminal + 0.10 * effort, 0.10, 0.0, 0.0),
        "tracking_effort_smooth_constraints": (tracking + 2.0 * terminal + 0.10 * effort + 0.50 * rate + 5.0 * constraint, 0.10, 0.50, 5.0),
    }
    rows = []
    for name, (cost, effort_weight, rate_weight, constraint_weight) in objective_terms.items():
        index = int(np.argmin(cost))
        selected_controls = candidates[index]
        selected_omega = omega[index]
        rows.append({
            "objective": name,
            "cost": float(cost[index]),
            "tracking_term": float(tracking[index]),
            "terminal_error_rad_s": float(abs(selected_omega[-1] - target)),
            "mean_control": float(selected_controls.mean()),
            "max_control": float(selected_controls.max()),
            "control_rate_rms": float(np.sqrt(np.mean(np.diff(selected_controls) ** 2))),
            "max_omega_rad_s": float(selected_omega.max()),
            "effort_weight": effort_weight,
            "rate_weight": rate_weight,
            "constraint_weight": constraint_weight,
        })
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "reward_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "experiment": "reward_cost_ablation_with_exact_euler_dynamics",
        "candidate_sequences": int(candidates.shape[0]),
        "horizon_steps": int(candidates.shape[1]),
        "target_omega_rad_s": target,
        "control_bounds": [0.45, 1.05],
        "rows": rows,
        "interpretation": "Reward shaping must be judged by tracking, effort, smoothness, constraints, and terminal error separately; objective value alone is not a physical success metric.",
    }
    (RESULTS / "reward_ablation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
