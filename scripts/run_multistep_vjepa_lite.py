"""Small action-conditioned, multi-step JEPA inspired by V-JEPA training.

This is not a reproduction of V-JEPA's video transformer. It translates the
relevant structural ideas to the scalar Euler system: a sequence encoder,
stop-gradient EMA target encoder, a recurrent action-conditioned predictor, and
loss over the full future horizon rather than one pooled target embedding.
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


class SequenceEncoder(nn.Module):
    """Encode a sequence and expose both final and per-step latent tokens."""

    def __init__(self, latent_size: int = 24):
        super().__init__()
        self.gru = nn.GRU(input_size=2, hidden_size=48, batch_first=True)
        self.projection = nn.Sequential(nn.Linear(48, 48), nn.GELU(), nn.Linear(48, latent_size))

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs, hidden = self.gru(sequence)
        return self.projection(hidden[-1]), self.projection(outputs)


class ActionConditionedRollout(nn.Module):
    """Autoregressively predict future latent tokens from state embedding and actions."""

    def __init__(self, latent_size: int = 24):
        super().__init__()
        self.cell = nn.GRUCell(latent_size + 1, latent_size)
        self.output = nn.Sequential(nn.LayerNorm(latent_size), nn.Linear(latent_size, latent_size))

    def forward(self, initial: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        hidden = initial
        previous = initial
        predictions = []
        for step in range(controls.shape[1]):
            hidden = self.cell(torch.cat((previous, controls[:, step, None]), dim=1), hidden)
            previous = self.output(hidden)
            predictions.append(previous)
        return torch.stack(predictions, dim=1)


def fit_probe(module, features: torch.Tensor, labels: torch.Tensor) -> nn.Module:
    probe = nn.Linear(features.shape[1], 1).to(module.DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=5e-3)
    for _ in range(300):
        loss = torch.mean((probe(features).squeeze(-1) - labels) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return probe


def train(module, train_contexts, train_targets, test_contexts, test_targets, seed: int, train_horizon: int, epochs: int = 720):
    torch.manual_seed(seed)
    train_context = torch.from_numpy(train_contexts).to(module.DEVICE)
    train_target = torch.from_numpy(train_targets).to(module.DEVICE)
    test_context = torch.from_numpy(test_contexts).to(module.DEVICE)
    test_target = torch.from_numpy(test_targets).to(module.DEVICE)
    context_encoder = SequenceEncoder().to(module.DEVICE)
    target_encoder = SequenceEncoder().to(module.DEVICE)
    target_encoder.load_state_dict(context_encoder.state_dict())
    predictor = ActionConditionedRollout().to(module.DEVICE)
    optimizer = torch.optim.AdamW(
        list(context_encoder.parameters()) + list(predictor.parameters()),
        lr=2e-3, weight_decay=1e-4,
    )
    history = []
    for epoch in range(1, epochs + 1):
        context_embedding, _ = context_encoder(train_context)
        with torch.no_grad():
            _, target_tokens = target_encoder(train_target)
        predictions = predictor(context_embedding, train_target[:, :, 1])
        horizon = min(train_horizon, predictions.shape[1])
        loss = torch.mean((predictions[:, :horizon] - target_tokens[:, :horizon]) ** 2)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(context_encoder.parameters()) + list(predictor.parameters()), 1.0)
        optimizer.step()
        with torch.no_grad():
            for target_parameter, context_parameter in zip(target_encoder.parameters(), context_encoder.parameters()):
                target_parameter.mul_(0.99).add_(context_parameter, alpha=0.01)
        if epoch == 1 or epoch % 60 == 0 or epoch == epochs:
            with torch.no_grad():
                test_context_embedding, _ = context_encoder(test_context)
                _, test_target_tokens = target_encoder(test_target)
                test_predictions = predictor(test_context_embedding, test_target[:, :, 1])
                test_loss = torch.mean((test_predictions[:, :horizon] - test_target_tokens[:, :horizon]) ** 2)
            history.append({
                "train_horizon": train_horizon,
                "seed": seed,
                "epoch": epoch,
                "train_latent_loss": float(loss.detach().cpu()),
                "test_latent_loss": float(test_loss.detach().cpu()),
            })
    with torch.no_grad():
        train_context_embedding, _ = context_encoder(train_context)
        test_context_embedding, _ = context_encoder(test_context)
        train_predicted_tokens = predictor(train_context_embedding, train_target[:, :, 1])
        test_predicted_tokens = predictor(test_context_embedding, test_target[:, :, 1])
        train_endpoint_features = train_predicted_tokens[:, -1]
        test_endpoint_features = test_predicted_tokens[:, -1]
        train_labels = train_target[:, -1, 0]
        test_labels = test_target[:, -1, 0]
    probe = fit_probe(module, train_endpoint_features.detach(), train_labels)
    with torch.no_grad():
        prediction = probe(test_endpoint_features).squeeze(-1).cpu().numpy()
    truth = test_labels.cpu().numpy()
    endpoint_rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)) * 600.0)
    return history, endpoint_rmse


def main() -> None:
    module = load_module()
    train_sequences, train_parameters = make_constant_sequences(module, 120, seed=7)
    test_sequences, test_parameters = make_constant_sequences(module, 30, seed=19)
    train_contexts, train_targets, _ = module.make_windows(train_sequences, train_parameters)
    test_contexts, test_targets, _ = module.make_windows(test_sequences, test_parameters)
    rows, history = [], []
    for train_horizon in (1, 10):
        for seed in (7, 17, 27):
            run_history, endpoint_rmse = train(
                module, train_contexts, train_targets, test_contexts, test_targets,
                seed, train_horizon, epochs=720,
            )
            history.extend(run_history)
            rows.append({
                "train_horizon": train_horizon,
                "seed": seed,
                "endpoint_probe_rmse_rad_s": endpoint_rmse,
                "final_train_latent_loss": run_history[-1]["train_latent_loss"],
                "final_test_latent_loss": run_history[-1]["test_latent_loss"],
            })
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "multistep_vjepa_lite.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (RESULTS / "multistep_vjepa_lite_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "experiment": "action_conditioned_multistep_vjepa_lite",
        "epochs": 720,
        "seeds": [7, 17, 27],
        "train_horizons": [1, 10],
        "architecture": "GRU sequence encoder + EMA target encoder + action-conditioned GRUCell latent rollout",
        "rows": rows,
        "interpretation": "This translates V-JEPA temporal/action-conditioned structure to the scalar Euler benchmark; it is not a V-JEPA transformer reproduction.",
    }
    (RESULTS / "multistep_vjepa_lite.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
