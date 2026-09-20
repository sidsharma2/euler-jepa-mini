"""Training-trend and representation-collapse diagnostics for the Euler JEPA.

This is a controlled ablation of the current one-target JEPA objective. It
compares prediction-only training with prediction plus variance/covariance
regularization, logging train/holdout latent error and embedding health during
training. The script does not modify the original experiment or raw inputs.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


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
        omega = module.simulate(
            params, np.full(steps, control, dtype=np.float32), 0.01,
            float(rng.uniform(350.0, 450.0)),
        )
        sequences.append(np.column_stack((omega[:-1] / 600.0, np.full(steps, control, dtype=np.float32))))
        parameters.append(params)
    return np.asarray(sequences, dtype=np.float32), parameters


def variance_loss(embedding: torch.Tensor, target_std: float = 0.05) -> torch.Tensor:
    standard_deviation = torch.sqrt(embedding.var(dim=0, unbiased=False) + 1e-04)
    return torch.relu(target_std - standard_deviation).mean()


def covariance_loss(embedding: torch.Tensor) -> torch.Tensor:
    centered = embedding - embedding.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(embedding.shape[0] - 1, 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    return torch.mean(off_diagonal.square())


def embedding_diagnostics(embedding: torch.Tensor) -> dict[str, float]:
    centered = embedding - embedding.mean(dim=0, keepdim=True)
    standard_deviation = centered.std(dim=0, unbiased=False)
    singular_values = torch.linalg.svdvals(centered)
    effective_rank = float((singular_values.sum() ** 2 / (singular_values.square().sum() + 1e-12)).detach().cpu())
    return {
        "embedding_std_mean": float(standard_deviation.mean().detach().cpu()),
        "embedding_std_min": float(standard_deviation.min().detach().cpu()),
        "embedding_std_max": float(standard_deviation.max().detach().cpu()),
        "embedding_effective_rank": effective_rank,
    }


def train(module, train_contexts, train_targets, test_contexts, test_targets, seed: int, variant: str, epochs: int = 180):
    torch.manual_seed(seed)
    train_context = torch.from_numpy(train_contexts).to(module.DEVICE)
    train_target = torch.from_numpy(train_targets).to(module.DEVICE)
    test_context = torch.from_numpy(test_contexts).to(module.DEVICE)
    test_target = torch.from_numpy(test_targets).to(module.DEVICE)
    context_encoder = module.Encoder().to(module.DEVICE)
    target_encoder = module.Encoder().to(module.DEVICE)
    target_encoder.load_state_dict(context_encoder.state_dict())
    predictor = module.Predictor().to(module.DEVICE)
    optimizer = torch.optim.AdamW(
        list(context_encoder.parameters()) + list(predictor.parameters()), lr=2e-3
    )
    regularized = variant == "prediction_plus_vc"
    history = []
    for epoch in range(1, epochs + 1):
        context_embedding = context_encoder(train_context)
        with torch.no_grad():
            target_embedding = target_encoder(train_target)
        prediction = predictor(context_embedding)
        prediction_loss = torch.mean((prediction - target_embedding) ** 2)
        # The target encoder is stop-gradient. Regularization must act on the
        # trainable context representation or it cannot change the model.
        var = variance_loss(context_embedding) if regularized else torch.zeros((), device=module.DEVICE)
        cov = covariance_loss(context_embedding) if regularized else torch.zeros((), device=module.DEVICE)
        total_loss = prediction_loss + 10.0 * var + 100.0 * cov if regularized else prediction_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            for target_parameter, context_parameter in zip(target_encoder.parameters(), context_encoder.parameters()):
                target_parameter.mul_(0.99).add_(context_parameter, alpha=0.01)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            with torch.no_grad():
                test_prediction = predictor(context_encoder(test_context))
                test_target_embedding = target_encoder(test_target)
                test_prediction_loss = torch.mean((test_prediction - test_target_embedding) ** 2)
                diagnostics = embedding_diagnostics(context_embedding.detach())
            history.append({
                "variant": variant,
                "seed": seed,
                "epoch": epoch,
                "train_prediction_loss": float(prediction_loss.detach().cpu()),
                "test_prediction_loss": float(test_prediction_loss.detach().cpu()),
                "variance_loss": float(var.detach().cpu()),
                "covariance_loss": float(cov.detach().cpu()),
                **diagnostics,
            })
    with torch.no_grad():
        train_embedding = context_encoder(train_context)
        test_embedding = context_encoder(test_context)
        train_labels = torch.from_numpy(train_targets[:, -1, 0]).to(module.DEVICE)
        test_labels = torch.from_numpy(test_targets[:, -1, 0]).to(module.DEVICE)
    probe = module.fit_probe(train_embedding.detach(), train_labels)
    with torch.no_grad():
        endpoint_prediction = probe(test_embedding).squeeze(-1).cpu().numpy()
    endpoint_truth = test_labels.cpu().numpy()
    endpoint_rmse = float(np.sqrt(np.mean((endpoint_prediction - endpoint_truth) ** 2)) * 600.0)
    return history, endpoint_rmse


def main() -> None:
    module = load_module()
    train_sequences, train_parameters = make_constant_sequences(module, 120, seed=7)
    test_sequences, test_parameters = make_constant_sequences(module, 30, seed=19)
    train_contexts, train_targets, _ = module.make_windows(train_sequences, train_parameters)
    test_contexts, test_targets, _ = module.make_windows(test_sequences, test_parameters)
    all_history, final_results = [], []
    for variant in ("prediction_only", "prediction_plus_vc"):
        for seed in (7, 17, 27):
            history, endpoint_rmse = train(
                module, train_contexts, train_targets, test_contexts, test_targets,
                seed, variant,
            )
            all_history.extend(history)
            final_results.append({
                "variant": variant,
                "seed": seed,
                "endpoint_probe_rmse_rad_s": endpoint_rmse,
                **{key: value for key, value in history[-1].items() if key not in {"variant", "seed", "epoch"}},
            })
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "training_diagnostics_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_history[0].keys())
        writer.writeheader()
        writer.writerows(all_history)
    for variant in ("prediction_only", "prediction_plus_vc"):
        variant_history = [row for row in all_history if row["variant"] == variant]
        epochs = sorted({row["epoch"] for row in variant_history})
        trend_rows = []
        for epoch in epochs:
            rows_at_epoch = [row for row in variant_history if row["epoch"] == epoch]
            trend_rows.append({
                "epoch": epoch,
                "mean_train_prediction_loss": float(np.mean([row["train_prediction_loss"] for row in rows_at_epoch])),
                "mean_test_prediction_loss": float(np.mean([row["test_prediction_loss"] for row in rows_at_epoch])),
                "mean_embedding_std": float(np.mean([row["embedding_std_mean"] for row in rows_at_epoch])),
                "mean_effective_rank": float(np.mean([row["embedding_effective_rank"] for row in rows_at_epoch])),
            })
        with (RESULTS / f"training_trend_{variant}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=trend_rows[0].keys())
            writer.writeheader()
            writer.writerows(trend_rows)
    summary = {
        "experiment": "jepa_prediction_vs_variance_covariance_regularization",
        "seeds": [7, 17, 27],
        "train_sequences": 120,
        "test_sequences": 30,
        "train_windows": int(len(train_contexts)),
        "test_windows": int(len(test_contexts)),
        "regularized_loss": "prediction MSE + 10*variance hinge + 100*off-diagonal covariance penalty",
        "final_results": final_results,
        "trend_interpretation": "Good training lowers holdout prediction loss without collapsing embedding variance or effective rank. Bad training lowers train loss while holdout loss rises, or drives embedding standard deviation and effective rank toward zero.",
    }
    (RESULTS / "training_diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
