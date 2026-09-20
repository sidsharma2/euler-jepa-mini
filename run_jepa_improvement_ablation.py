"""Run additive JEPA-mini improvement ablations at epsilon=0.5.

This experiment keeps the Task 6 data generation, train/validation split,
three random seeds, optimizer, and 1000-step budget fixed. It tests two
changes independently and together:

1. a differentiable rotor ordinary-differential-equation residual penalty;
2. a predictor-variance regularizer inspired by the checked-in JEPA training
   implementation, to discourage latent collapse.

The existing Task 6 implementation and source project are not modified.
Results are written to additive files in this derived project.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPORTS = HERE / "reports"
BASELINE_SCRIPT = HERE / "run_discrepancy_trajectory_training.py"
EPSILON = 0.5
SEEDS = (7, 19, 31)
ODE_WEIGHT = 0.05
LATENT_VARIANCE_WEIGHT = 0.01
ODE_SCALE_N_M = 1.0


def load_baseline_module():
    """Load the existing Task 6 module without executing its main function."""
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("task6_baseline", BASELINE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load baseline script: {BASELINE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def effective_rank(values: torch.Tensor) -> float:
    """Return the entropy effective rank of a batch of latent vectors."""
    centered = values - values.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    probabilities = singular_values / singular_values.sum().clamp_min(1.0e-12)
    return float(torch.exp(-(probabilities * torch.log(probabilities.clamp_min(1.0e-12))).sum()).item())


def latent_variance_penalty(predicted_latent: torch.Tensor) -> torch.Tensor:
    """Penalize predictor dimensions whose batch standard deviation is below one."""
    standard_deviation = torch.sqrt(predicted_latent.var(dim=0, unbiased=False) + 1.0e-4)
    return torch.mean(torch.relu(1.0 - standard_deviation))


def ode_residual_penalty(
    prediction_normalized: torch.Tensor,
    context: torch.Tensor,
    calibrated_base: torch.Tensor,
    parameters: dict[str, torch.Tensor],
    c2: float,
    baseline_module,
) -> torch.Tensor:
    """Compute a differentiable torque-balance residual for the predicted path."""
    forecast = calibrated_base + prediction_normalized * 600.0
    initial_speed = context[:, -1:, 0] * 600.0
    trajectory = torch.cat((initial_speed, forecast), dim=1)
    previous_speed = trajectory[:, :-1]
    derivative = (trajectory[:, 1:] - previous_speed) / baseline_module.DT
    control = context[:, -1:, 1].expand_as(previous_speed)
    rhs_torque = (
        parameters["mass_flow"]
        * parameters["radius"]
        * (parameters["swirl_gain"] * control - baseline_module.K * previous_speed)
        - baseline_module.C0
        - c2 * previous_speed.square()
    )
    residual_N_m = parameters["inertia"] * derivative - rhs_torque
    return torch.mean((residual_N_m / ODE_SCALE_N_M).square())


def parameter_tensors(parameter_list, device: torch.device) -> dict[str, torch.Tensor]:
    """Convert source-project parameter records to broadcastable tensors."""
    return {
        "mass_flow": torch.tensor([item.mass_flow_rate_kg_s for item in parameter_list], device=device).unsqueeze(1),
        "radius": torch.tensor([item.radius_m for item in parameter_list], device=device).unsqueeze(1),
        "swirl_gain": torch.tensor([item.swirl_gain_m_s for item in parameter_list], device=device).unsqueeze(1),
        "inertia": torch.tensor([item.inertia_kg_m2 for item in parameter_list], device=device).unsqueeze(1),
    }


def train_one(
    baseline_module,
    context: np.ndarray,
    future: np.ndarray,
    calibrated_base: np.ndarray,
    parameter_list,
    c2: float,
    seed: int,
    ode_weight: float,
    latent_variance_weight: float,
):
    """Train one JEPA-style model and select its checkpoint by validation RMSE."""
    torch.manual_seed(seed)
    model = baseline_module.JEPATrajectory().to(baseline_module.DEVICE)
    model.target_encoder.load_state_dict(model.context_encoder.state_dict())
    model.target_project.load_state_dict(model.context_project.state_dict())
    for parameter in model.target_encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.target_project.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    split = int(len(context) * 0.8)
    device = baseline_module.DEVICE
    context_tensor = torch.tensor(context, device=device)
    target_tensor = torch.tensor(future, device=device)
    base_tensor = torch.tensor(calibrated_base, device=device)
    residual_target = torch.tensor((future[:, :, 0] * 600.0 - calibrated_base) / 600.0, device=device)
    train_parameters = parameter_tensors(parameter_list, device)
    history = []
    best_state = None
    best_validation_rmse = float("inf")
    best_step = 0

    for step in range(1, 1001):
        model.train()
        prediction, latent_prediction = model(context_tensor[:split])
        with torch.no_grad():
            target_latent = baseline_module.layer_norm_target(model.encode_target(target_tensor[:split]))
        trajectory_loss = torch.mean((prediction - residual_target[:split]) ** 2)
        latent_loss = torch.mean(torch.abs(latent_prediction - target_latent))
        ode_loss = ode_residual_penalty(
            prediction,
            context_tensor[:split],
            base_tensor[:split],
            {key: value[:split] for key, value in train_parameters.items()},
            c2,
            baseline_module,
        )
        variance_loss = latent_variance_penalty(latent_prediction)
        loss = trajectory_loss + 0.05 * latent_loss + ode_weight * ode_loss + latent_variance_weight * variance_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        momentum = 1.0 - (1.0 - 0.998) * (np.cos(np.pi * step / 1000.0) + 1.0) / 2.0
        with torch.no_grad():
            for target_parameter, context_parameter in zip(model.target_encoder.parameters(), model.context_encoder.parameters()):
                target_parameter.mul_(momentum).add_(context_parameter, alpha=1.0 - momentum)
            for target_parameter, context_parameter in zip(model.target_project.parameters(), model.context_project.parameters()):
                target_parameter.mul_(momentum).add_(context_parameter, alpha=1.0 - momentum)

        if step % 25 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                validation_prediction, _ = model(context_tensor[split:])
                validation_rmse = torch.sqrt(
                    torch.mean(((validation_prediction - residual_target[split:]) * 600.0).square())
                ).item()
            history.append(
                {
                    "seed": seed,
                    "gradient_step": step,
                    "trajectory_loss": float(trajectory_loss.item()),
                    "latent_loss": float(latent_loss.item()),
                    "ode_loss": float(ode_loss.item()),
                    "variance_loss": float(variance_loss.item()),
                    "total_loss": float(loss.item()),
                    "validation_trajectory_rmse_rad_s": validation_rmse,
                }
            )
            if validation_rmse < best_validation_rmse:
                best_validation_rmse = validation_rmse
                best_step = step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_step, best_validation_rmse


def evaluate(baseline_module, model, context, test_future, test_base, test_parameters, c2):
    """Evaluate trajectory error, physical residual, latent rank, and sensitivity."""
    prediction = baseline_module.predict(model, "jepa_residual", context, test_base)
    truth = test_future[:, :, 0] * 600.0
    full_rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
    endpoint_rmse = baseline_module.rmse(prediction[:, -1], truth[:, -1])
    ode_rmse = float(np.mean([
        baseline_module.ode_residual(path, item_context, params, c2)
        for path, item_context, params in zip(prediction, context, test_parameters)
    ]))
    with torch.no_grad():
        latent = model.encode_context(torch.tensor(context, device=baseline_module.DEVICE))
    rank = effective_rank(latent)

    delta = 1.0e-3
    plus_context = context.copy()
    plus_context[:, :, 1] += delta
    plus_base = np.asarray([baseline_module.calibrated_trajectory(item) for item in plus_context])
    plus_prediction = baseline_module.predict(model, "jepa_residual", plus_context, plus_base)
    truth_plus = []
    for item_context, params in zip(plus_context, test_parameters):
        value = float(item_context[-1, 0] * 600.0)
        control = float(item_context[-1, 1])
        for _ in range(baseline_module.HORIZON * baseline_module.SUBSTEPS):
            value = baseline_module.rk4_step(value, control, params, baseline_module.DT / baseline_module.SUBSTEPS, c2)
        truth_plus.append(value)
    derivative = (plus_prediction[:, -1] - prediction[:, -1]) / delta
    true_derivative = (np.asarray(truth_plus) - truth[:, -1]) / delta
    sensitivity_rmse = baseline_module.rmse(derivative, true_derivative)
    return {
        "full_trajectory_rmse_rad_s": full_rmse,
        "endpoint_rmse_rad_s": endpoint_rmse,
        "ode_residual_rmse_N_m": ode_rmse,
        "latent_effective_rank": rank,
        "control_sensitivity_rmse_rad_s_per_control": sensitivity_rmse,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a nonempty list of dictionaries as a CSV file."""
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the fixed-protocol 2x2 regularization ablation."""
    torch.set_num_threads(1)
    baseline_module = load_baseline_module()
    source_module = baseline_module.load_source()
    c2 = EPSILON * baseline_module.C0 / (400.0**2 * (1.0 - EPSILON))
    train_context, train_future, train_parameters = baseline_module.generate(source_module, 120, 101, c2)
    test_context, test_future, test_parameters = baseline_module.generate(source_module, 30, 202, c2)
    train_base = np.asarray([baseline_module.calibrated_trajectory(item) for item in train_context])
    test_base = np.asarray([baseline_module.calibrated_trajectory(item) for item in test_context])

    configurations = [
        ("baseline", 0.0, 0.0),
        ("ode_regularized", ODE_WEIGHT, 0.0),
        ("latent_variance_regularized", 0.0, LATENT_VARIANCE_WEIGHT),
        ("combined_regularization", ODE_WEIGHT, LATENT_VARIANCE_WEIGHT),
    ]
    result_rows = []
    history_rows = []
    for configuration, ode_weight, variance_weight in configurations:
        for seed in SEEDS:
            model, history, best_step, best_validation_rmse = train_one(
                baseline_module,
                train_context,
                train_future,
                train_base,
                train_parameters,
                c2,
                seed,
                ode_weight,
                variance_weight,
            )
            metrics = evaluate(
                baseline_module,
                model,
                test_context,
                test_future,
                test_base,
                test_parameters,
                c2,
            )
            result_rows.append(
                {
                    "configuration": configuration,
                    "seed": seed,
                    "epsilon": EPSILON,
                    "ode_weight": ode_weight,
                    "latent_variance_weight": variance_weight,
                    "best_gradient_step": best_step,
                    "best_validation_rmse_rad_s": best_validation_rmse,
                    **metrics,
                }
            )
            for item in history:
                history_rows.append({"configuration": configuration, **item})

    RESULTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    write_csv(RESULTS / "jepa_improvement_ablation.csv", result_rows)
    write_csv(RESULTS / "jepa_improvement_history.csv", history_rows)
    payload = {
        "experiment": "jepa_mini_improvement_ablation",
        "epsilon": EPSILON,
        "seeds": list(SEEDS),
        "context_steps": baseline_module.CONTEXT,
        "horizon_steps": baseline_module.HORIZON,
        "dt_s": baseline_module.DT,
        "device": str(baseline_module.DEVICE),
        "configurations": [
            {"name": name, "ode_weight": ode, "latent_variance_weight": variance}
            for name, ode, variance in configurations
        ],
        "provenance": {
            "task6_script": str(BASELINE_SCRIPT),
            "source_project": str(baseline_module.SOURCE),
            "official_jepa_training_reference": "facebookresearch/jepa/app/vjepa/train.py",
            "sparse_dynamics_reference": "dynamicslab/pysindy",
        },
        "result_rows": result_rows,
        "limitations": [
            "Synthetic scalar rotor benchmark only",
            "Constant-control 20-step context and 200-step forecast",
            "No experimental validation",
            "Regularization weights were selected as a small ablation, not globally optimized",
        ],
    }
    (RESULTS / "jepa_improvement_ablation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = """# JEPA-mini improvement ablation

This additive experiment keeps the corrected Task 6 data, split, seeds, 2-second forecast, optimizer, and 1000-step budget fixed at `epsilon = 0.5`. It tests two changes suggested by the present failure modes:

* `ode_regularized`: adds a differentiable residual of the rotor torque balance to the training loss;
* `latent_variance_regularized`: adds a predictor-variance penalty modeled on the checked-in official JEPA training code;
* `combined_regularization`: applies both terms.

The checked-in official reference was inspected at [facebookresearch/jepa `app/vjepa/train.py](https://github.com/facebookresearch/jepa/blob/main/app/vjepa/train.py). The local PySINDy repository was also inspected as a possible independent sparse-dynamics comparator; it was not inserted into this ablation, so the result remains attributable to the JEPA-mini change alone.

The primary acceptance criteria are full-trajectory RMSE, ODE residual, finite-difference control sensitivity, and effective rank of the learned context representation. A lower latent loss alone is not treated as an improvement.

Results are in `results/jepa_improvement_ablation.csv`; training histories are in `results/jepa_improvement_history.csv` and the full provenance payload is in `results/jepa_improvement_ablation.json`.
"""
    (REPORTS / "jepa_improvement_ablation.md").write_text(report, encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
