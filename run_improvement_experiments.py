"""Controlled architecture and evaluation experiments for the Euler JEPA benchmark.

The important comparison is between probing the context embedding (the original
evaluation) and probing the predictor's forecast embedding (the representation
that JEPA is actually trained to produce). An action-conditioned predictor is
also tested because the control is part of the physical state-transition law.
The original project and corrected benchmark are imported read-only.
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
ORIGINAL_SCRIPT = HERE / "run_experiment.py"
RESULTS = HERE / "results"


def load_original_module():
    spec = importlib.util.spec_from_file_location("original_euler_experiment", ORIGINAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load original experiment: {ORIGINAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ActionConditionedPredictor(nn.Module):
    """Predict a target-window embedding from context embedding and control."""

    def __init__(self, embedding_size: int = 24):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_size + 1, 48),
            nn.GELU(),
            nn.Linear(48, embedding_size),
        )

    def forward(self, embedding: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((embedding, control[:, None]), dim=1))


def make_constant_sequences(module, n_sequences: int, seed: int, steps: int = 200):
    """Generate the same corrected constant-control protocol used by the benchmark."""
    rng = np.random.default_rng(seed)
    sequences = []
    parameters = []
    for _ in range(n_sequences):
        params = module.Parameters(
            mass_flow_rate_kg_s=float(rng.uniform(0.70, 0.90)),
            inertia_kg_m2=float(rng.uniform(0.017, 0.023)),
            speed_feedback_s=float(rng.uniform(0.0085, 0.0115)),
        )
        control = float(rng.uniform(0.45, 1.05))
        controls = np.full(steps, control, dtype=np.float32)
        omega = module.simulate(
            params, controls, dt_s=0.01,
            omega0_rad_s=float(rng.uniform(350.0, 450.0)),
        )
        sequences.append(np.column_stack((omega[:-1] / 600.0, controls)))
        parameters.append(params)
    return np.asarray(sequences, dtype=np.float32), parameters


def train_variant(module, contexts, targets, variant: str, seed: int, epochs: int = 180):
    torch.manual_seed(seed)
    context_tensor = torch.from_numpy(contexts).to(module.DEVICE)
    target_tensor = torch.from_numpy(targets).to(module.DEVICE)
    context_encoder = module.Encoder().to(module.DEVICE)
    target_encoder = module.Encoder().to(module.DEVICE)
    target_encoder.load_state_dict(context_encoder.state_dict())
    predictor = (
        module.Predictor().to(module.DEVICE)
        if variant == "predicted_latent_probe"
        else ActionConditionedPredictor().to(module.DEVICE)
    )
    optimizer = torch.optim.AdamW(
        list(context_encoder.parameters()) + list(predictor.parameters()), lr=2e-3
    )
    losses = []
    for _ in range(epochs):
        context_embedding = context_encoder(context_tensor)
        with torch.no_grad():
            target_embedding = target_encoder(target_tensor)
        if variant == "predicted_latent_probe":
            prediction = predictor(context_embedding)
        else:
            prediction = predictor(context_embedding, context_tensor[:, -1, 1])
        loss = torch.mean((prediction - target_embedding) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for target_parameter, context_parameter in zip(
                target_encoder.parameters(), context_encoder.parameters()
            ):
                target_parameter.mul_(0.99).add_(context_parameter, alpha=0.01)
        losses.append(float(loss.detach().cpu()))
    return context_encoder, predictor, losses


def fit_linear_probe(module, embeddings: torch.Tensor, labels: torch.Tensor) -> nn.Module:
    """Fit the same small linear downstream probe used by the benchmark."""
    probe = nn.Linear(embeddings.shape[1], 1).to(module.DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=5e-3)
    for _ in range(250):
        prediction = probe(embeddings).squeeze(-1)
        loss = torch.mean((prediction - labels) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return probe


def fit_nonlinear_probe(module, embeddings: torch.Tensor, labels: torch.Tensor) -> nn.Module:
    """Fit a small nonlinear residual head without changing the JEPA encoder."""
    probe = nn.Sequential(
        nn.Linear(embeddings.shape[1], 48), nn.GELU(), nn.Linear(48, 1)
    ).to(module.DEVICE)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(500):
        prediction = probe(embeddings).squeeze(-1)
        loss = torch.mean((prediction - labels) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return probe


def evaluate(module, encoder, predictor, variant, train_contexts, train_targets, test_contexts, test_targets, train_nominal, test_nominal):
    with torch.no_grad():
        train_context_tensor = torch.from_numpy(train_contexts).to(module.DEVICE)
        test_context_tensor = torch.from_numpy(test_contexts).to(module.DEVICE)
        train_context_embedding = encoder(train_context_tensor)
        test_context_embedding = encoder(test_context_tensor)
        if variant in {"context_probe", "direct_endpoint_probe", "physics_residual_probe", "physics_residual_mlp"}:
            train_features = train_context_embedding
            test_features = test_context_embedding
        elif variant == "predicted_latent_probe":
            train_features = predictor(train_context_embedding)
            test_features = predictor(test_context_embedding)
        else:
            train_features = predictor(train_context_embedding, train_context_tensor[:, -1, 1])
            test_features = predictor(test_context_embedding, test_context_tensor[:, -1, 1])
        train_labels = torch.from_numpy(train_targets[:, -1, 0]).to(module.DEVICE)
        test_labels = torch.from_numpy(test_targets[:, -1, 0]).to(module.DEVICE)
        if variant in {"physics_residual_probe", "physics_residual_mlp"}:
            train_labels = train_labels - torch.from_numpy(train_nominal).to(module.DEVICE)
    probe = (
        fit_nonlinear_probe(module, train_features.detach(), train_labels)
        if variant == "physics_residual_mlp"
        else fit_linear_probe(module, train_features.detach(), train_labels)
    )
    with torch.no_grad():
        prediction = probe(test_features).squeeze(-1).cpu().numpy()
    if variant in {"physics_residual_probe", "physics_residual_mlp"}:
        prediction = prediction + test_nominal
    truth = test_labels.cpu().numpy()
    return float(np.sqrt(np.mean((prediction - truth) ** 2)) * 600.0)


def main() -> None:
    module = load_original_module()
    train_sequences, train_parameters = make_constant_sequences(module, 120, seed=7)
    test_sequences, test_parameters = make_constant_sequences(module, 30, seed=19)
    train_contexts, train_targets, _ = module.make_windows(train_sequences, train_parameters)
    test_contexts, test_targets, _ = module.make_windows(test_sequences, test_parameters)

    train_meta = module.make_windows(train_sequences, train_parameters)[2]
    test_meta = module.make_windows(test_sequences, test_parameters)[2]

    def nominal_values(contexts, metadata):
        return np.asarray([
            module.nominal_analytical_endpoint(context, params, horizon=10, dt_s=0.01)
            for context, (_, _, params) in zip(contexts, metadata)
        ])

    train_nominal = nominal_values(train_contexts, train_meta)
    test_nominal = nominal_values(test_contexts, test_meta)
    variants = [
        "context_probe",
        "predicted_latent_probe",
        "action_conditioned_latent_probe",
        "direct_endpoint_probe",
        "physics_residual_probe",
        "physics_residual_mlp",
    ]
    rows = []
    for variant in variants:
        for seed in (7, 17, 27):
            encoder, predictor, losses = train_variant(
                module, train_contexts, train_targets,
                "predicted_latent_probe" if variant != "action_conditioned_latent_probe" else "action_conditioned",
                seed,
            )
            rmse = evaluate(
                module, encoder, predictor, variant,
                train_contexts, train_targets, test_contexts, test_targets,
                train_nominal, test_nominal,
            )
            rows.append({
                "variant": variant,
                "seed": seed,
                "test_rmse_rad_s": rmse,
                "final_latent_loss": losses[-1],
                "epochs": len(losses),
            })

    nominal = []
    oracle = []
    truth = test_targets[:, -1, 0] * 600.0
    for context, (_, _, params) in zip(test_contexts, test_meta):
        nominal.append(module.nominal_analytical_endpoint(context, params, horizon=10, dt_s=0.01) * 600.0)
        omega = float(context[-1, 0] * 600.0)
        for _ in range(10):
            omega = module.exact_step(params, omega, float(context[-1, 1]), 0.01)
        oracle.append(omega)
    rows.extend([
        {"variant": "nominal_analytical", "seed": None, "test_rmse_rad_s": float(np.sqrt(np.mean((np.asarray(nominal) - truth) ** 2))), "final_latent_loss": None, "epochs": None},
        {"variant": "true_parameter_oracle", "seed": None, "test_rmse_rad_s": float(np.sqrt(np.mean((np.asarray(oracle) - truth) ** 2))), "final_latent_loss": None, "epochs": None},
    ])
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "improvement_experiments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary_rows = []
    for variant in variants:
        values = np.asarray([
            row["test_rmse_rad_s"] for row in rows if row["variant"] == variant
        ])
        summary_rows.append({
            "variant": variant,
            "mean_rmse_rad_s": float(values.mean()),
            "std_rmse_rad_s": float(values.std(ddof=1)),
            "n_seeds": int(values.size),
        })
    summary_rows.extend([
        {"variant": "nominal_analytical", "mean_rmse_rad_s": rows[-2]["test_rmse_rad_s"], "std_rmse_rad_s": 0.0, "n_seeds": 1},
        {"variant": "true_parameter_oracle", "mean_rmse_rad_s": rows[-1]["test_rmse_rad_s"], "std_rmse_rad_s": 0.0, "n_seeds": 1},
    ])
    with (RESULTS / "improvement_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    summary = {
        "experiment": "euler_jepa_architecture_and_probe_alignment",
        "seeds": [7, 17, 27],
        "train_sequences": 120,
        "test_sequences": 30,
        "train_windows": int(len(train_contexts)),
        "test_windows": int(len(test_contexts)),
        "variants": variants,
        "evaluation": "linear probe fitted on the same feature type used at test time",
        "control_protocol": "constant control per sequence; control is observed in the context",
        "results_csv": str(RESULTS / "improvement_experiments.csv"),
        "summary_csv": str(RESULTS / "improvement_summary.csv"),
        "summary_rows": summary_rows,
        "rows": rows,
        "interpretation": "The predicted-latent variants test the JEPA forecast representation directly. Results remain synthetic representation-learning evidence, not physical validation.",
    }
    (RESULTS / "improvement_experiments.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
