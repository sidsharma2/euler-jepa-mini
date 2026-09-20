"""Physics-first Euler turbomachinery transient plus a tiny JEPA experiment.

The synthetic system is intentionally simple: a rotor with inertia J receives
Euler work through a prescribed tangential-velocity change.  The exact
discrete-time solution is used as the data generator and analytical baseline.
The JEPA learns a representation of a recent state/control block and predicts
the representation of the next block.  A frozen linear probe converts that
representation into a future rotor-speed prediction.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class Parameters:
    mass_flow_rate_kg_s: float = 0.8
    radius_m: float = 0.25
    inertia_kg_m2: float = 0.02
    swirl_gain_m_s: float = 8.0
    speed_feedback_s: float = 0.01
    load_torque_N_m: float = 0.15


def euler_torque(params: Parameters, omega_rad_s: float, control: float) -> float:
    """Euler torque from tau = mdot * r * Delta C_theta."""
    delta_c_theta_m_s = params.swirl_gain_m_s * control - params.speed_feedback_s * omega_rad_s
    return params.mass_flow_rate_kg_s * params.radius_m * delta_c_theta_m_s


def exact_step(params: Parameters, omega_rad_s: float, control: float, dt_s: float) -> float:
    """Exact step for the linearized rotor equation under constant control."""
    alpha = params.mass_flow_rate_kg_s * params.radius_m * params.swirl_gain_m_s / params.inertia_kg_m2
    beta = params.mass_flow_rate_kg_s * params.radius_m * params.speed_feedback_s / params.inertia_kg_m2
    gamma = params.load_torque_N_m / params.inertia_kg_m2
    omega_equilibrium = (alpha * control - gamma) / beta
    return omega_equilibrium + (omega_rad_s - omega_equilibrium) * math.exp(-beta * dt_s)


def simulate(params: Parameters, controls: np.ndarray, dt_s: float, omega0_rad_s: float) -> np.ndarray:
    omega = np.empty(len(controls) + 1, dtype=np.float32)
    omega[0] = omega0_rad_s
    for index, control in enumerate(controls):
        omega[index + 1] = exact_step(params, float(omega[index]), float(control), dt_s)
    return omega


def make_sequences(n_sequences: int, seed: int, steps: int = 200) -> tuple[np.ndarray, list[Parameters]]:
    rng = np.random.default_rng(seed)
    sequences = []
    parameters = []
    for _ in range(n_sequences):
        params = Parameters(
            mass_flow_rate_kg_s=float(rng.uniform(0.70, 0.90)),
            inertia_kg_m2=float(rng.uniform(0.017, 0.023)),
            speed_feedback_s=float(rng.uniform(0.0085, 0.0115)),
        )
        controls = np.repeat(rng.uniform(0.45, 1.05, size=steps // 20), 20).astype(np.float32)
        controls = controls[:steps]
        omega = simulate(params, controls, dt_s=0.01, omega0_rad_s=float(rng.uniform(350.0, 450.0)))
        # Each record contains normalized omega and the dimensionless control.
        sequences.append(np.column_stack((omega[:-1] / 600.0, controls)))
        parameters.append(params)
    return np.asarray(sequences, dtype=np.float32), parameters


def make_windows(sequences: np.ndarray, parameters: list[Parameters], context: int = 20, horizon: int = 10):
    contexts, targets, meta = [], [], []
    # Window starts are aligned with control blocks, so the nominal analytical
    # baseline can assume the current command persists through the horizon.
    for sequence_index, sequence in enumerate(sequences):
        for start in range(0, sequence.shape[0] - context - horizon + 1, 20):
            contexts.append(sequence[start : start + context])
            targets.append(sequence[start + context : start + context + horizon])
            meta.append((sequence_index, start, parameters[sequence_index]))
    return np.asarray(contexts), np.asarray(targets), meta


class Encoder(nn.Module):
    def __init__(self, input_size: int = 2, hidden_size: int = 48, embedding_size: int = 24):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.projection = nn.Sequential(nn.Linear(hidden_size, 48), nn.GELU(), nn.Linear(48, embedding_size))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(sequence)
        return self.projection(hidden[-1])


class Predictor(nn.Module):
    def __init__(self, embedding_size: int = 24):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_size, 48), nn.GELU(), nn.Linear(48, embedding_size)
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.network(embedding)


def train_jepa(contexts: np.ndarray, targets: np.ndarray, epochs: int = 180) -> tuple[Encoder, Predictor, list[float]]:
    torch.manual_seed(7)
    context_tensor = torch.from_numpy(contexts).to(DEVICE)
    target_tensor = torch.from_numpy(targets).to(DEVICE)
    context_encoder = Encoder().to(DEVICE)
    target_encoder = Encoder().to(DEVICE)
    target_encoder.load_state_dict(context_encoder.state_dict())
    predictor = Predictor().to(DEVICE)
    optimizer = torch.optim.AdamW(list(context_encoder.parameters()) + list(predictor.parameters()), lr=2e-3)
    losses = []
    for _ in range(epochs):
        context_embedding = context_encoder(context_tensor)
        with torch.no_grad():
            target_embedding = target_encoder(target_tensor)
        prediction = predictor(context_embedding)
        loss = torch.mean((prediction - target_embedding) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # EMA target encoder: the JEPA target is not backpropagated through.
        with torch.no_grad():
            for target_parameter, context_parameter in zip(target_encoder.parameters(), context_encoder.parameters()):
                target_parameter.mul_(0.99).add_(context_parameter, alpha=0.01)
        losses.append(float(loss.detach().cpu()))
    return context_encoder, predictor, losses


def fit_probe(embeddings: torch.Tensor, labels: torch.Tensor, epochs: int = 250) -> nn.Module:
    probe = nn.Linear(embeddings.shape[1], 1).to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=5e-3)
    for _ in range(epochs):
        prediction = probe(embeddings).squeeze(-1)
        loss = torch.mean((prediction - labels) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return probe


def nominal_analytical_endpoint(context: np.ndarray, params: Parameters, horizon: int, dt_s: float) -> float:
    omega = float(context[-1, 0] * 600.0)
    control = float(context[-1, 1])
    nominal = Parameters()
    for _ in range(horizon):
        omega = exact_step(nominal, omega, control, dt_s)
    return omega / 600.0


def main() -> None:
    torch.set_float32_matmul_precision("high")
    RESULTS.mkdir(parents=True, exist_ok=True)
    train_sequences, train_parameters = make_sequences(220, seed=7)
    test_sequences, test_parameters = make_sequences(60, seed=19)
    train_contexts, train_targets, _ = make_windows(train_sequences, train_parameters)
    test_contexts, test_targets, test_meta = make_windows(test_sequences, test_parameters)

    context_encoder, predictor, losses = train_jepa(train_contexts, train_targets)
    with torch.no_grad():
        train_embeddings = context_encoder(torch.from_numpy(train_contexts).to(DEVICE))
        test_embeddings = context_encoder(torch.from_numpy(test_contexts).to(DEVICE))
        train_labels = torch.from_numpy(train_targets[:, -1, 0]).to(DEVICE)
        test_labels = torch.from_numpy(test_targets[:, -1, 0]).to(DEVICE)
    probe = fit_probe(train_embeddings.detach(), train_labels)
    with torch.no_grad():
        ml_prediction = probe(test_embeddings).squeeze(-1).cpu().numpy()

    analytical_prediction = np.asarray(
        [nominal_analytical_endpoint(context, params, horizon=10, dt_s=0.01)
         for context, (_, _, params) in zip(test_contexts, test_meta)]
    )
    truth = test_labels.cpu().numpy()
    ml_rmse = float(np.sqrt(np.mean((ml_prediction - truth) ** 2)) * 600.0)
    analytical_rmse = float(np.sqrt(np.mean((analytical_prediction - truth) ** 2)) * 600.0)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(losses, color="tab:blue")
    axes[0].set(title="JEPA representation loss", xlabel="Epoch", ylabel="MSE")
    axes[0].grid(alpha=0.25)
    axes[1].scatter(truth * 600.0, ml_prediction * 600.0, s=12, alpha=0.55, label="JEPA + probe")
    limits = [min(truth.min(), ml_prediction.min()) * 600.0, max(truth.max(), ml_prediction.max()) * 600.0]
    axes[1].plot(limits, limits, "k--", linewidth=1, label="Perfect prediction")
    axes[1].set(title="Future rotor-speed prediction", xlabel="True omega (rad/s)", ylabel="Predicted omega (rad/s)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(RESULTS / "euler_jepa_results.png", dpi=160)
    plt.close(figure)

    metrics = {
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "train_windows": int(len(train_contexts)),
        "test_windows": int(len(test_contexts)),
        "jepa_final_loss": losses[-1],
        "ml_probe_rmse_rad_s": ml_rmse,
        "nominal_analytical_rmse_rad_s": analytical_rmse,
        "physics": {
            "equations": ["delta_h0 = U * delta_Ctheta", "J*domega/dt = mdot*r*delta_Ctheta - tau_load"],
            "dt_s": 0.01,
            "context_steps": 20,
            "forecast_steps": 10,
        },
        "nominal_parameters": asdict(Parameters()),
        "interpretation": "The analytical baseline uses nominal parameters; synthetic truth varies inertia, flow, and feedback. The ML probe forecasts the future endpoint from the JEPA context embedding.",
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
