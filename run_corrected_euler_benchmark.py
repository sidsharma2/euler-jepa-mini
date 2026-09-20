"""Run a corrected, well-posed open-loop Euler--JEPA benchmark.

The original experiment changed the control every 20 samples while asking a
model to predict the next 10 samples without supplying the future control. This
variant keeps control constant over each whole sequence, so the forecast is
well-posed and the nominal analytical baseline uses information available in
the context. It also reports an oracle analytical baseline and exact-vs-forward
Euler integration error.

The original project is imported read-only. All new artifacts are written under
the research-workflow example directory.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ORIGINAL_SCRIPT = HERE / "run_experiment.py"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
REPORTS = HERE / "reports"


def load_original_module():
    """Load the original experiment implementation without executing main()."""
    spec = importlib.util.spec_from_file_location("original_euler_experiment", ORIGINAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load original experiment: {ORIGINAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_constant_sequences(module, n_sequences: int, seed: int, steps: int = 200):
    """Generate sequences whose control is known and constant over the forecast."""
    rng = np.random.default_rng(seed)
    sequences = []
    parameters = []
    controls_used = []
    for _ in range(n_sequences):
        params = module.Parameters(
            mass_flow_rate_kg_s=float(rng.uniform(0.70, 0.90)),
            inertia_kg_m2=float(rng.uniform(0.017, 0.023)),
            speed_feedback_s=float(rng.uniform(0.0085, 0.0115)),
        )
        control = float(rng.uniform(0.45, 1.05))
        controls = np.full(steps, control, dtype=np.float32)
        omega = module.simulate(
            params,
            controls,
            dt_s=0.01,
            omega0_rad_s=float(rng.uniform(350.0, 450.0)),
        )
        sequences.append(np.column_stack((omega[:-1] / 600.0, controls)))
        parameters.append(params)
        controls_used.append(control)
    return np.asarray(sequences, dtype=np.float32), parameters, controls_used


def forward_euler_step(module, params, omega_rad_s: float, control: float, dt_s: float) -> float:
    """Advance the physical ODE with explicit Euler for verification only."""
    torque = module.euler_torque(params, omega_rad_s, control)
    domega_dt = (torque - params.load_torque_N_m) / params.inertia_kg_m2
    return omega_rad_s + dt_s * domega_dt


def forward_euler_trajectory(module, params, control: float, steps: int, dt_s: float, omega0: float):
    omega = np.empty(steps + 1, dtype=float)
    omega[0] = omega0
    for index in range(steps):
        omega[index + 1] = forward_euler_step(module, params, omega[index], control, dt_s)
    return omega


def oracle_endpoint(module, context: np.ndarray, params, horizon: int, dt_s: float) -> float:
    """Exact endpoint using the true parameters; this is a diagnostic oracle."""
    omega = float(context[-1, 0] * 600.0)
    control = float(context[-1, 1])
    for _ in range(horizon):
        omega = module.exact_step(params, omega, control, dt_s)
    return omega / 600.0


def main() -> None:
    module = load_original_module()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    train_sequences, train_parameters, _ = make_constant_sequences(module, 120, seed=7)
    test_sequences, test_parameters, _ = make_constant_sequences(module, 30, seed=19)
    train_contexts, train_targets, _ = module.make_windows(train_sequences, train_parameters)
    test_contexts, test_targets, test_meta = module.make_windows(test_sequences, test_parameters)

    context_encoder, _predictor, losses = module.train_jepa(train_contexts, train_targets)
    with torch.no_grad():
        train_embeddings = context_encoder(torch.from_numpy(train_contexts).to(module.DEVICE))
        test_embeddings = context_encoder(torch.from_numpy(test_contexts).to(module.DEVICE))
        train_labels = torch.from_numpy(train_targets[:, -1, 0]).to(module.DEVICE)
        test_labels = torch.from_numpy(test_targets[:, -1, 0]).to(module.DEVICE)
    probe = module.fit_probe(train_embeddings.detach(), train_labels)
    with torch.no_grad():
        ml_prediction = probe(test_embeddings).squeeze(-1).cpu().numpy()

    nominal_prediction = np.asarray([
        module.nominal_analytical_endpoint(context, params, horizon=10, dt_s=0.01)
        for context, (_, _, params) in zip(test_contexts, test_meta)
    ])
    oracle_prediction = np.asarray([
        oracle_endpoint(module, context, params, horizon=10, dt_s=0.01)
        for context, (_, _, params) in zip(test_contexts, test_meta)
    ])
    truth = test_labels.cpu().numpy()

    # Independent integration check on one representative trajectory.
    check_params = train_parameters[0]
    check_control = float(train_sequences[0, 0, 1])
    exact = module.simulate(check_params, np.full(200, check_control), 0.01, 400.0)
    forward = forward_euler_trajectory(module, check_params, check_control, 200, 0.01, 400.0)
    integration_error = float(np.max(np.abs(exact - forward)))

    def rmse_rad_s(prediction):
        return float(np.sqrt(np.mean((prediction - truth) ** 2)) * 600.0)

    metrics = {
        "benchmark": "corrected_open_loop_constant_control",
        "device": str(module.DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "train_sequences": len(train_sequences),
        "test_sequences": len(test_sequences),
        "train_windows": len(train_contexts),
        "test_windows": len(test_contexts),
        "random_seeds": {"data_train": 7, "data_test": 19, "jepa": 7},
        "control_protocol": "one constant control per sequence; future control is therefore known from the observed context",
        "jepa_final_loss": float(losses[-1]),
        "ml_probe_rmse_rad_s": rmse_rad_s(ml_prediction),
        "nominal_analytical_rmse_rad_s": rmse_rad_s(nominal_prediction),
        "oracle_analytical_rmse_rad_s": rmse_rad_s(oracle_prediction),
        "max_exact_vs_forward_euler_error_rad_s": integration_error,
        "equations": [
            "J*domega/dt = mdot*r*delta_Ctheta - tau_load",
            "delta_Ctheta = k_u*u - k_omega*omega",
            "delta_h0 = U*delta_Ctheta",
            "U = r*omega",
        ],
        "interpretation": "The corrected task is well-posed. The oracle isolates parameter mismatch, while the nominal baseline tests robustness to parameter variation. Neither result is experimental validation.",
    }

    with (RESULTS / "corrected_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_omega_rad_s", "jepa_probe_omega_rad_s", "nominal_analytical_omega_rad_s", "oracle_analytical_omega_rad_s"])
        writer.writerows(zip(truth * 600.0, ml_prediction * 600.0, nominal_prediction * 600.0, oracle_prediction * 600.0))
    (RESULTS / "corrected_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": False,
    })
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(losses, color="black", linewidth=1.2)
    axes[0].set(xlabel="Epoch", ylabel="JEPA representation loss", title="Latent prediction training")
    axes[1].scatter(truth * 600.0, ml_prediction * 600.0, marker="o", facecolors="none", edgecolors="black", s=22, label="JEPA + probe")
    axes[1].scatter(truth * 600.0, nominal_prediction * 600.0, marker="s", facecolors="none", edgecolors="black", s=22, label="Nominal analytical")
    axes[1].scatter(truth * 600.0, oracle_prediction * 600.0, marker="D", facecolors="black", edgecolors="black", s=12, label="Oracle analytical")
    limits = [min(truth.min(), ml_prediction.min(), nominal_prediction.min(), oracle_prediction.min()) * 600.0,
              max(truth.max(), ml_prediction.max(), nominal_prediction.max(), oracle_prediction.max()) * 600.0]
    axes[1].plot(limits, limits, "k--", linewidth=0.9, label="Perfect prediction")
    axes[1].set(xlabel="True $\\omega$ (rad/s)", ylabel="Predicted $\\omega$ (rad/s)", title="Corrected open-loop forecast")
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURES / "corrected_euler_benchmark.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / "corrected_euler_benchmark.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    log = {
        "original_script": ORIGINAL_SCRIPT.name,
        "new_artifact_root": ".",
        "source_project_modified": False,
        "parameters": asdict(module.Parameters()),
        "metrics": metrics,
    }
    (REPORTS / "corrected_run_log.txt").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
