"""Retrain full-trajectory residual models for the nonlinear Euler rotor.

This is an additive Task 6 experiment.  The source project is imported but
never modified.  Checkpoints are selected on a validation split; the test
set is used only once for the reported metrics.
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

from metrics import bootstrap_ci, r2, rmse, skill_vs_persistence

SCRIPT_DIR = Path(__file__).resolve().parent
HERE = SCRIPT_DIR.parent
SOURCE = SCRIPT_DIR / "run_experiment.py"
RESULTS, REPORTS, FIGURES = HERE / "results", HERE / "reports", HERE / "figures"
DT, SUBSTEPS, CONTEXT, HORIZON, K, C0 = 0.01, 10, 20, 200, 0.25, 0.15
EPS_VALUES = (0.0, 0.10, 0.20, 0.50)
SEEDS = (7, 19, 31)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_source():
    spec = importlib.util.spec_from_file_location("euler_source_task6", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source module: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rhs(omega, control, params, c2):
    torque = params.mass_flow_rate_kg_s * params.radius_m * (
        params.swirl_gain_m_s * control - params.speed_feedback_s * omega
    )
    return (torque - C0 - c2 * omega * omega) / params.inertia_kg_m2


def rk4_step(omega, control, params, dt, c2):
    k1 = rhs(omega, control, params, c2)
    k2 = rhs(omega + dt * k1 / 2, control, params, c2)
    k3 = rhs(omega + dt * k2 / 2, control, params, c2)
    k4 = rhs(omega + dt * k3, control, params, c2)
    return omega + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def generate(module, count, seed, c2):
    rng = np.random.default_rng(seed)
    contexts, futures, params_list = [], [], []
    for _ in range(count):
        params = module.Parameters(
            mass_flow_rate_kg_s=float(rng.uniform(.70, .90)),
            inertia_kg_m2=float(rng.uniform(.017, .023)),
            speed_feedback_s=K,
        )
        control = float(rng.uniform(.45, 1.05))
        omega = [float(rng.uniform(350, 450))]
        for _ in range((CONTEXT + HORIZON) * SUBSTEPS):
            omega.append(rk4_step(omega[-1], control, params, DT / SUBSTEPS, c2))
        values = np.asarray(omega)[::SUBSTEPS]
        contexts.append(np.column_stack((values[:CONTEXT] / 600, np.full(CONTEXT, control))))
        futures.append(np.column_stack((values[CONTEXT:CONTEXT + HORIZON] / 600, np.full(HORIZON, control))))
        params_list.append(params)
    return np.asarray(contexts, dtype=np.float32), np.asarray(futures, dtype=np.float32), params_list


def calibrated_trajectory(context):
    speed = context[:, 0] * 600
    design = np.column_stack((speed[:-1], np.ones(len(speed) - 1)))
    rho, intercept = np.linalg.lstsq(design, speed[1:], rcond=None)[0]
    result = []
    value = float(speed[-1])
    for _ in range(HORIZON):
        value = rho * value + intercept
        result.append(value)
    return np.asarray(result)


def nominal_trajectory(module, context):
    params = module.Parameters(speed_feedback_s=K)
    value, control = float(context[-1, 0] * 600), float(context[-1, 1])
    result = []
    for _ in range(HORIZON):
        value = module.exact_step(params, value, control, DT)
        result.append(value)
    return np.asarray(result)


class GRUResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(2, 48, batch_first=True)
        self.head = nn.Sequential(nn.Linear(48, 96), nn.GELU(), nn.Linear(96, HORIZON))

    def forward(self, context):
        _, hidden = self.gru(context)
        return self.head(hidden[-1])


class JEPATrajectory(nn.Module):
    """JEPA-style latent predictor with a supervised full-trajectory head."""
    def __init__(self):
        super().__init__()
        self.context_encoder = nn.GRU(2, 48, batch_first=True)
        self.target_encoder = nn.GRU(2, 48, batch_first=True)
        self.context_project = nn.Sequential(nn.Linear(48, 24), nn.GELU(), nn.Linear(24, 24))
        self.target_project = nn.Sequential(nn.Linear(48, 24), nn.GELU(), nn.Linear(24, 24))
        self.predictor = nn.Sequential(nn.Linear(25, 48), nn.GELU(), nn.Linear(48, 24))
        self.trajectory_head = nn.Sequential(nn.Linear(25, 96), nn.GELU(), nn.Linear(96, HORIZON))

    def encode_context(self, context):
        _, hidden = self.context_encoder(context)
        return self.context_project(hidden[-1])

    def encode_target(self, future):
        _, hidden = self.target_encoder(future)
        return self.target_project(hidden[-1])

    def forward(self, context):
        latent = self.encode_context(context)
        control = context[:, -1:, 1]
        features = torch.cat((latent, control), dim=1)
        return self.trajectory_head(features), self.predictor(features)


def layer_norm_target(value):
    return torch.nn.functional.layer_norm(value, (value.shape[-1],))


def train_model(kind, context, future, base, seed):
    torch.manual_seed(seed)
    model = GRUResidual() if kind == "gru_residual" else JEPATrajectory()
    model.to(DEVICE)
    if kind == "jepa_residual":
        # JEPA target starts as a copy and is updated only by EMA; it never
        # receives prediction-loss gradients.
        model.target_encoder.load_state_dict(model.context_encoder.state_dict())
        model.target_project.load_state_dict(model.context_project.state_dict())
        for parameter in model.target_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in model.target_project.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    split = int(len(context) * 0.8)
    x = torch.tensor(context, device=DEVICE)
    y = torch.tensor((future[:, :, 0] * 600 - base) / 600, device=DEVICE)
    target_input = torch.tensor(future, device=DEVICE)
    history, best_state, best_val, best_step = [], None, float("inf"), 0
    for step in range(1, 1001):
        model.train()
        if kind == "gru_residual":
            prediction = model(x[:split])
            loss = torch.mean((prediction - y[:split]) ** 2)
        else:
            prediction, latent_prediction = model(x[:split])
            with torch.no_grad():
                target_latent = layer_norm_target(model.encode_target(target_input[:split]))
            loss_trajectory = torch.mean((prediction - y[:split]) ** 2)
            loss_latent = torch.mean(torch.abs(latent_prediction - target_latent))
            loss = loss_trajectory + 0.05 * loss_latent
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if kind == "jepa_residual":
            momentum = 1.0 - (1.0 - 0.998) * (np.cos(np.pi * step / 1000) + 1.0) / 2.0
            with torch.no_grad():
                for target_parameter, context_parameter in zip(model.target_encoder.parameters(), model.context_encoder.parameters()):
                    target_parameter.mul_(momentum).add_(context_parameter, alpha=1.0 - momentum)
                for target_parameter, context_parameter in zip(model.target_project.parameters(), model.context_project.parameters()):
                    target_parameter.mul_(momentum).add_(context_parameter, alpha=1.0 - momentum)
        if step % 25 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                if kind == "gru_residual":
                    val_prediction = model(x[split:])
                else:
                    val_prediction, _ = model(x[split:])
                val_rmse = torch.sqrt(torch.mean(((val_prediction - y[split:]) * 600) ** 2)).item()
            history.append({"model": kind, "seed": seed, "gradient_step": step,
                            "train_loss": float(loss.item()), "validation_trajectory_rmse_rad_s": val_rmse})
            if val_rmse < best_val:
                best_val, best_step = val_rmse, step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No validation checkpoint was produced")
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_step, best_val


def predict(model, kind, context, base):
    with torch.no_grad():
        x = torch.tensor(context, device=DEVICE)
        correction = model(x)[0] if kind == "jepa_residual" else model(x)
    return base + correction.detach().cpu().numpy() * 600


def ode_residual(prediction, context, params, c2):
    trajectory = np.concatenate(([context[-1, 0] * 600], prediction))
    controls = np.full(HORIZON, context[-1, 1])
    residual = []
    for index in range(HORIZON):
        derivative = (trajectory[index + 1] - trajectory[index]) / DT
        residual.append(params.inertia_kg_m2 * derivative - (params.mass_flow_rate_kg_s * params.radius_m *
                      (params.swirl_gain_m_s * controls[index] - K * trajectory[index]) - C0 - c2 * trajectory[index] ** 2))
    return float(np.sqrt(np.mean(np.square(residual))))


def bootstrap_metric(pred, truth, sequence_ids):
    low, high = bootstrap_ci(pred[:, -1], truth[:, -1], sequence_ids)
    return low, high


def main():
    torch.set_num_threads(1)
    module = load_source()
    metric_rows, horizon_rows, sensitivity_rows, history_rows = [], [], [], []
    optimum_rows = []
    for epsilon in EPS_VALUES:
        c2 = epsilon * C0 / (400.0 ** 2 * max(1e-12, 1 - epsilon))
        train_context, train_future, train_params = generate(module, 120, 101, c2)
        test_context, test_future, test_params = generate(module, 30, 202, c2)
        train_base = np.asarray([calibrated_trajectory(c) for c in train_context])
        test_base = np.asarray([calibrated_trajectory(c) for c in test_context])
        truth = test_future[:, :, 0] * 600
        persistence = np.repeat(test_context[:, -1:, 0] * 600, HORIZON, axis=1)
        baseline_predictions = {
            "nominal_physics": np.asarray([nominal_trajectory(module, c) for c in test_context]),
            "calibrated_physics": test_base,
        }
        learned = {}
        learned_models = {}
        for kind in ("gru_residual", "jepa_residual"):
            for seed in SEEDS:
                model, history, best_step, best_val = train_model(kind, train_context, train_future, train_base, seed)
                method = f"{kind}_seed{seed}"
                learned[method] = predict(model, kind, test_context, test_base)
                learned_models[method] = (model, kind)
                optimum_rows.append({"epsilon": epsilon, "method": method, "seed": seed,
                                     "best_gradient_step": best_step, "best_validation_trajectory_rmse_rad_s": best_val})
                for item in history:
                    item["epsilon"] = epsilon; history_rows.append(item)
        all_predictions = {**baseline_predictions, **learned}
        for method, prediction in all_predictions.items():
            seq_ids = np.arange(len(prediction))
            endpoint_low, endpoint_high = bootstrap_ci(prediction[:, -1], truth[:, -1], seq_ids)
            full_low, full_high = bootstrap_ci(np.sqrt(np.mean((prediction - truth) ** 2, axis=1)),
                                                np.zeros(len(prediction)), seq_ids)
            if method.startswith("gru_residual") or method.startswith("jepa_residual"):
                mean_ode = float(np.mean([ode_residual(p, c, pa, c2) for p, c, pa in zip(prediction, test_context, test_params)]))
            elif method == "calibrated_physics":
                mean_ode = float(np.mean([ode_residual(p, c, pa, c2) for p, c, pa in zip(prediction, test_context, test_params)]))
            else:
                mean_ode = float(np.mean([ode_residual(p, c, pa, c2) for p, c, pa in zip(prediction, test_context, test_params)]))
            endpoint_truth = truth[:, -1]
            metric_rows.append({"epsilon": epsilon, "method": method, "endpoint_rmse_rad_s": rmse(prediction[:, -1], endpoint_truth),
                                "endpoint_r2": r2(prediction[:, -1], endpoint_truth),
                                "full_trajectory_rmse_rad_s": float(np.sqrt(np.mean((prediction - truth) ** 2))),
                                "endpoint_skill_vs_persistence": skill_vs_persistence(prediction[:, -1], endpoint_truth, persistence[:, -1]),
                                "ode_residual_rmse_N_m": mean_ode, "endpoint_rmse_ci95_low_rad_s": endpoint_low,
                                "endpoint_rmse_ci95_high_rad_s": endpoint_high, "full_rmse_ci95_low_rad_s": full_low,
                                "full_rmse_ci95_high_rad_s": full_high})
            for horizon in (10, 50, 100, 200):
                errors = np.sqrt(np.mean((prediction[:, :horizon] - truth[:, :horizon]) ** 2, axis=1))
                horizon_rows.append({"epsilon": epsilon, "method": method, "horizon_steps": horizon,
                                     "horizon_seconds": horizon * DT, "trajectory_rmse_rad_s": float(np.sqrt(np.mean(errors ** 2))),
                                     "trajectory_rmse_ci95_low_rad_s": bootstrap_ci(errors, np.zeros(len(errors)), seq_ids)[0],
                                     "trajectory_rmse_ci95_high_rad_s": bootstrap_ci(errors, np.zeros(len(errors)), seq_ids)[1]})
        # Control sensitivity is a finite-difference endpoint derivative check.
        delta = 1e-3
        for method, prediction in all_predictions.items():
            plus_context = test_context.copy(); plus_context[:, :, 1] += delta
            if method == "nominal_physics":
                plus = np.asarray([nominal_trajectory(module, c) for c in plus_context])
            elif method == "calibrated_physics":
                plus = np.asarray([calibrated_trajectory(c) for c in plus_context])
            else:
                model, kind = learned_models[method]
                plus_base = np.asarray([calibrated_trajectory(c) for c in plus_context])
                plus = predict(model, kind, plus_context, plus_base)
            truth_plus = []
            for c, pa in zip(plus_context, test_params):
                value = float(c[-1, 0] * 600); control = float(c[-1, 1])
                for _ in range(HORIZON * SUBSTEPS): value = rk4_step(value, control, pa, DT / SUBSTEPS, c2)
                truth_plus.append(value)
            truth_plus = np.asarray(truth_plus)
            derivative = (plus[:, -1] - prediction[:, -1]) / delta
            true_derivative = (truth_plus - truth[:, -1]) / delta
            sensitivity_rows.append({"epsilon": epsilon, "method": method, "control_delta": delta,
                                     "endpoint_control_sensitivity_rmse_rad_s_per_control": rmse(derivative, true_derivative)})
    RESULTS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    def write_csv(path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    write_csv(RESULTS / "discrepancy_trajectory_training.csv", metric_rows)
    write_csv(RESULTS / "discrepancy_trajectory_horizon.csv", horizon_rows)
    write_csv(RESULTS / "discrepancy_trajectory_sensitivity.csv", sensitivity_rows)
    write_csv(RESULTS / "discrepancy_trajectory_optima.csv", optimum_rows)
    write_csv(RESULTS / "discrepancy_trajectory_training_history.csv", history_rows)
    payload = {"experiment": "task6_full_trajectory_retraining", "device": str(DEVICE), "horizon_steps": HORIZON,
               "context_steps": CONTEXT, "dt_s": DT, "epsilons": list(EPS_VALUES), "seeds": list(SEEDS),
               "metrics": metric_rows, "optima": optimum_rows,
               "limitations": ["Synthetic nonlinear-load data", "One context window per sequence", "No experimental validation",
                               "Learned sensitivity uses frozen test predictions and is diagnostic only"]}
    (RESULTS / "discrepancy_trajectory_training.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = """# Task 6 full-trajectory JEPA retraining\n\n"""
    report += "This run trains a GRU residual model and a JEPA-style hybrid with a target encoder, stop-gradient layer-normalized target, EMA-compatible architecture, L1 latent loss, and a full 200-step trajectory head. Checkpoints are selected by validation trajectory RMSE.\n\n"
    report += "Training-framework references: [I-JEPA](https://arxiv.org/abs/2301.08243), [V-JEPA 2](https://arxiv.org/abs/2506.09985), [official JEPA training code](https://github.com/facebookresearch/jepa/blob/main/app/vjepa/train.py), and [official V-JEPA 2 training code](https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/train.py).\n\n"
    report += "The CSV files contain endpoint and full-trajectory RMSE, R², skill versus persistence, ODE residuals in N m, horizon errors through 2 s, bootstrap intervals, control sensitivity, and validation-selected training optima.\n"
    (REPORTS / "discrepancy_trajectory_training.md").write_text(report + "\n```json\n" + json.dumps(payload, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps({"device": str(DEVICE), "rows": len(metric_rows), "optima": optimum_rows}, indent=2))


if __name__ == "__main__":
    main()
