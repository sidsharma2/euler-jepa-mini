"""Additive, corrected architecture comparison for the Euler-JEPA study.

This file intentionally leaves ``run_improvement_experiments.py`` unchanged.
The direct regression baseline has no JEPA encoder in its computation path.
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
ORIGINAL_SCRIPT = SCRIPT_DIR / "run_experiment.py"
RESULTS = HERE / "results"
REPORTS = HERE / "reports"
DT_S = 0.01
HORIZON = 10


def load_original_module():
    spec = importlib.util.spec_from_file_location("original_euler_experiment_v2", ORIGINAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load original experiment: {ORIGINAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_constant_sequences(module, n_sequences: int, seed: int, steps: int = 200):
    rng = np.random.default_rng(seed)
    sequences, parameters = [], []
    for _ in range(n_sequences):
        params = module.Parameters(
            mass_flow_rate_kg_s=float(rng.uniform(0.70, 0.90)),
            inertia_kg_m2=float(rng.uniform(0.017, 0.023)),
            speed_feedback_s=float(rng.uniform(0.0085, 0.0115)),
        )
        control = float(rng.uniform(0.45, 1.05))
        omega = module.simulate(params, np.full(steps, control, dtype=np.float32), DT_S, float(rng.uniform(350, 450)))
        sequences.append(np.column_stack((omega[:-1] / 600.0, np.full(steps, control))))
        parameters.append(params)
    return np.asarray(sequences, dtype=np.float32), parameters


def make_data(module):
    train_sequences, train_parameters = make_constant_sequences(module, 120, 7)
    test_sequences, test_parameters = make_constant_sequences(module, 30, 19)
    train_contexts, train_targets, train_meta = module.make_windows(train_sequences, train_parameters)
    test_contexts, test_targets, test_meta = module.make_windows(test_sequences, test_parameters)
    return train_contexts, train_targets, train_meta, test_contexts, test_targets, test_meta


class ActionConditionedPredictor(nn.Module):
    def __init__(self, embedding_size: int = 24):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(embedding_size + 1, 48), nn.GELU(), nn.Linear(48, embedding_size))

    def forward(self, embedding, control):
        return self.network(torch.cat((embedding, control[:, None]), dim=1))


def train_jepa(module, contexts, targets, variant: str, seed: int, gradient_steps: int = 180):
    seed_everything(seed)
    context_tensor = torch.from_numpy(contexts).to(module.DEVICE)
    target_tensor = torch.from_numpy(targets).to(module.DEVICE)
    context_encoder = module.Encoder().to(module.DEVICE)
    target_encoder = module.Encoder().to(module.DEVICE)
    target_encoder.load_state_dict(context_encoder.state_dict())
    predictor = ActionConditionedPredictor().to(module.DEVICE) if variant == "action_conditioned_latent_probe" else module.Predictor().to(module.DEVICE)
    optimizer = torch.optim.AdamW(list(context_encoder.parameters()) + list(predictor.parameters()), lr=2e-3)
    losses = []
    for _ in range(gradient_steps):
        context_embedding = context_encoder(context_tensor)
        with torch.no_grad():
            target_embedding = target_encoder(target_tensor)
        if variant == "action_conditioned_latent_probe":
            prediction = predictor(context_embedding, context_tensor[:, -1, 1])
        else:
            prediction = predictor(context_embedding)
        loss = torch.mean((prediction - target_embedding) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for target_parameter, context_parameter in zip(target_encoder.parameters(), context_encoder.parameters()):
                target_parameter.mul_(0.99).add_(context_parameter, alpha=0.01)
        losses.append(float(loss.detach().cpu()))
    return context_encoder, predictor, losses


def fit_probe(module, features, labels, nonlinear: bool = False, steps_override: int | None = None):
    seed_everything(2026)
    if nonlinear:
        model = nn.Sequential(nn.Linear(features.shape[1], 48), nn.GELU(), nn.Linear(48, 1)).to(module.DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        steps = 500
    else:
        model = nn.Linear(features.shape[1], 1).to(module.DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        steps = 250
    if steps_override is not None:
        steps = steps_override
    for _ in range(steps):
        prediction = model(features).squeeze(-1)
        loss = torch.mean((prediction - labels) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def nominal(module, contexts, metadata):
    return np.asarray([module.nominal_analytical_endpoint(c, p, HORIZON, DT_S) * 600.0 for c, (_, _, p) in zip(contexts, metadata)])


def evaluate_variant(module, variant, encoder, predictor, train_contexts, train_targets, test_contexts, test_targets, train_nominal, test_nominal):
    train_context_tensor = torch.from_numpy(train_contexts).to(module.DEVICE)
    test_context_tensor = torch.from_numpy(test_contexts).to(module.DEVICE)
    with torch.no_grad():
        train_embedding = encoder(train_context_tensor)
        test_embedding = encoder(test_context_tensor)
        if variant == "predicted_latent_probe":
            train_features, test_features = predictor(train_embedding), predictor(test_embedding)
        elif variant == "action_conditioned_latent_probe":
            train_features = predictor(train_embedding, train_context_tensor[:, -1, 1])
            test_features = predictor(test_embedding, test_context_tensor[:, -1, 1])
        else:
            train_features, test_features = train_embedding, test_embedding
        train_labels = torch.from_numpy(train_targets[:, -1, 0].astype(float)).float().to(module.DEVICE)
        if variant in {"physics_residual_probe", "physics_residual_mlp"}:
            train_labels = train_labels - torch.from_numpy(train_nominal / 600.0).float().to(module.DEVICE)
    probe = fit_probe(module, train_features.detach(), train_labels, nonlinear=variant == "physics_residual_mlp")
    with torch.no_grad():
        prediction = probe(test_features.detach()).squeeze(-1).cpu().numpy() * 600.0
    if variant in {"physics_residual_probe", "physics_residual_mlp"}:
        prediction += test_nominal
    return prediction


def main() -> None:
    module = load_original_module()
    train_contexts, train_targets, train_meta, test_contexts, test_targets, test_meta = make_data(module)
    truth = test_targets[:, -1, 0].astype(float) * 600.0
    persistence = test_contexts[:, -1, 0].astype(float) * 600.0
    sequence_index = np.asarray([item[0] for item in test_meta], dtype=int)
    train_nominal = nominal(module, train_contexts, train_meta)
    test_nominal = nominal(module, test_contexts, test_meta)
    variants = ["context_probe", "predicted_latent_probe", "action_conditioned_latent_probe", "physics_residual_probe", "physics_residual_mlp"]
    rows = []
    for variant in variants:
        for seed in (7, 17, 27):
            encoder, predictor, losses = train_jepa(module, train_contexts, train_targets, variant, seed)
            prediction = evaluate_variant(module, variant, encoder, predictor, train_contexts, train_targets, test_contexts, test_targets, train_nominal, test_nominal)
            low, high = bootstrap_ci(prediction, truth, sequence_index)
            rows.append({"variant": variant, "seed": seed, "rmse_rad_s": rmse(prediction, truth), "r2": r2(prediction, truth), "skill_vs_persistence": skill_vs_persistence(prediction, truth, persistence), "rmse_ci95_low_rad_s": low, "rmse_ci95_high_rad_s": high, "final_latent_loss": losses[-1], "gradient_steps": len(losses)})
    raw_train = torch.from_numpy(train_contexts.reshape(len(train_contexts), -1)).float().to(module.DEVICE)
    raw_test = torch.from_numpy(test_contexts.reshape(len(test_contexts), -1)).float().to(module.DEVICE)
    raw_train_labels = torch.from_numpy(train_targets[:, -1, 0].astype(float)).float().to(module.DEVICE)
    direct = fit_probe(module, raw_train, raw_train_labels, nonlinear=True, steps_override=3000)
    with torch.no_grad():
        direct_prediction = direct(raw_test).squeeze(-1).cpu().numpy() * 600.0
    low, high = bootstrap_ci(direct_prediction, truth, sequence_index)
    rows.append({"variant": "direct_endpoint_regression", "seed": 2026, "rmse_rad_s": rmse(direct_prediction, truth), "r2": r2(direct_prediction, truth), "skill_vs_persistence": skill_vs_persistence(direct_prediction, truth, persistence), "rmse_ci95_low_rad_s": low, "rmse_ci95_high_rad_s": high, "final_latent_loss": None, "gradient_steps": None})
    oracle_values = []
    for context, (_, _, params) in zip(test_contexts, test_meta):
        omega = float(context[-1, 0] * 600.0)
        for _ in range(HORIZON):
            omega = module.exact_step(params, omega, float(context[-1, 1]), DT_S)
        oracle_values.append(omega)
    for name, prediction in (("nominal_analytical", test_nominal), ("true_parameter_oracle", np.asarray(oracle_values))):
        low, high = bootstrap_ci(prediction, truth, sequence_index)
        rows.append({"variant": name, "seed": None, "rmse_rad_s": rmse(prediction, truth), "r2": r2(prediction, truth), "skill_vs_persistence": skill_vs_persistence(prediction, truth, persistence), "rmse_ci95_low_rad_s": low, "rmse_ci95_high_rad_s": high, "final_latent_loss": None, "gradient_steps": None})
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "improvement_experiments_v2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    summary = []
    for variant in [*variants, "direct_endpoint_regression", "nominal_analytical", "true_parameter_oracle"]:
        values = [row for row in rows if row["variant"] == variant]
        metrics = np.asarray([row["rmse_rad_s"] for row in values])
        summary.append({"variant": variant, "mean_rmse_rad_s": float(metrics.mean()), "std_rmse_rad_s": float(metrics.std(ddof=1)) if len(metrics) > 1 else 0.0, "n_runs": len(metrics)})
    with (RESULTS / "improvement_summary_v2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    payload = {"experiment": "euler_jepa_architecture_and_probe_alignment_v2", "optimisation": "full-batch, one gradient step per iteration", "seeds": [7, 17, 27], "train_sequences": 120, "test_sequences": 30, "train_windows": len(train_contexts), "test_windows": len(test_contexts), "variants": variants + ["direct_endpoint_regression", "nominal_analytical", "true_parameter_oracle"], "rows": rows, "summary": summary, "limitations": ["synthetic data", "in-distribution held-out sequences", "no experimental validation"]}
    (RESULTS / "improvement_experiments_v2.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (REPORTS / "improvement_experiments_v2_run.log").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
