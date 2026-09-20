"""Run the missing baseline suite on the corrected Euler-JEPA dataset."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

from metrics import bootstrap_ci, r2, rmse, skill_vs_persistence


SCRIPT_DIR = Path(__file__).resolve().parent
HERE = SCRIPT_DIR.parent
ORIGINAL_SCRIPT = SCRIPT_DIR / "run_experiment.py"
RESULTS = HERE / "results"
REPORTS = HERE / "reports"
DT_S = 0.01
CONTEXT_STEPS = 20
HORIZON_STEPS = 10


def load_original_module():
    """Import the source implementation without calling its main function."""
    spec = importlib.util.spec_from_file_location("original_euler_experiment_baseline", ORIGINAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load original experiment: {ORIGINAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_constant_sequences(module, n_sequences: int, seed: int, steps: int = 200):
    """Reproduce the corrected constant-control data generator exactly."""
    rng = np.random.default_rng(seed)
    sequences, parameters = [], []
    for _ in range(n_sequences):
        params = module.Parameters(
            mass_flow_rate_kg_s=float(rng.uniform(0.70, 0.90)),
            inertia_kg_m2=float(rng.uniform(0.017, 0.023)),
            speed_feedback_s=float(rng.uniform(0.0085, 0.0115)),
        )
        control = float(rng.uniform(0.45, 1.05))
        controls = np.full(steps, control, dtype=np.float32)
        omega = module.simulate(params, controls, DT_S, float(rng.uniform(350.0, 450.0)))
        sequences.append(np.column_stack((omega[:-1] / 600.0, controls)))
        parameters.append(params)
    return np.asarray(sequences, dtype=np.float32), parameters


def window_data(module, sequences, parameters):
    """Build windows and retain sequence IDs separately from source metadata."""
    contexts, targets, metadata = module.make_windows(
        sequences, parameters, context=CONTEXT_STEPS, horizon=HORIZON_STEPS
    )
    sequence_index = np.asarray([item[0] for item in metadata], dtype=int)
    return contexts, targets, metadata, sequence_index


def nominal_prediction(module, contexts, metadata):
    return np.asarray([
        module.nominal_analytical_endpoint(context, params, HORIZON_STEPS, DT_S) * 600.0
        for context, (_, _, params) in zip(contexts, metadata)
    ])


def oracle_prediction(module, contexts, metadata):
    values = []
    for context, (_, _, params) in zip(contexts, metadata):
        omega = float(context[-1, 0] * 600.0)
        control = float(context[-1, 1])
        for _ in range(HORIZON_STEPS):
            omega = module.exact_step(params, omega, control, DT_S)
        values.append(omega)
    return np.asarray(values)


def least_squares_sysid(module, contexts):
    """Fit the affine discrete map omega_next = rho*omega + c per window."""
    predictions = []
    for context in contexts:
        omega = context[:, 0].astype(float) * 600.0
        design = np.column_stack((omega[:-1], np.ones(omega.size - 1)))
        rho, intercept = np.linalg.lstsq(design, omega[1:], rcond=None)[0]
        future = float(omega[-1])
        for _ in range(HORIZON_STEPS):
            future = rho * future + intercept
        predictions.append(future)
    return np.asarray(predictions)


def fit_ols(train_contexts, train_targets, test_contexts):
    """Fit an ordinary least-squares endpoint predictor on raw context."""
    train_features = np.column_stack((
        train_contexts.reshape(len(train_contexts), -1),
        np.ones(len(train_contexts)),
    ))
    test_features = np.column_stack((
        test_contexts.reshape(len(test_contexts), -1),
        np.ones(len(test_contexts)),
    ))
    coefficients = np.linalg.lstsq(
        train_features, train_targets[:, -1, 0].astype(float) * 600.0, rcond=None
    )[0]
    return test_features @ coefficients


def main() -> None:
    module = load_original_module()
    train_sequences, train_parameters = make_constant_sequences(module, 120, 7)
    test_sequences, test_parameters = make_constant_sequences(module, 30, 19)
    train_contexts, train_targets, _, _ = window_data(module, train_sequences, train_parameters)
    test_contexts, test_targets, test_metadata, sequence_index = window_data(module, test_sequences, test_parameters)
    truth = test_targets[:, -1, 0].astype(float) * 600.0
    persistence = test_contexts[:, -1, 0].astype(float) * 600.0
    previous = test_contexts[:, -2, 0].astype(float) * 600.0
    predictions = {
        "persistence": persistence,
        "two_point_extrapolation": persistence + HORIZON_STEPS * (persistence - previous),
        "ols_raw_context": fit_ols(train_contexts, train_targets, test_contexts),
        "nominal_analytical": nominal_prediction(module, test_contexts, test_metadata),
        "true_parameter_oracle": oracle_prediction(module, test_contexts, test_metadata),
        "least_squares_sysid": least_squares_sysid(module, test_contexts),
    }
    nominal_train = nominal_prediction(module, train_contexts, window_data(module, train_sequences, train_parameters)[2])
    residual_target = train_targets[:, -1, 0].astype(float) * 600.0 - nominal_train
    train_features = np.column_stack((train_contexts.reshape(len(train_contexts), -1), np.ones(len(train_contexts))))
    test_features = np.column_stack((test_contexts.reshape(len(test_contexts), -1), np.ones(len(test_contexts))))
    residual_coefficients = np.linalg.lstsq(train_features, residual_target, rcond=None)[0]
    predictions["ols_residual_on_nominal"] = predictions["nominal_analytical"] + test_features @ residual_coefficients

    rows = []
    for name, prediction in predictions.items():
        ci_low, ci_high = bootstrap_ci(prediction, truth, sequence_index)
        rows.append({
            "predictor": name,
            "rmse_rad_s": rmse(prediction, truth),
            "r2": r2(prediction, truth),
            "skill_vs_persistence": skill_vs_persistence(prediction, truth, persistence),
            "rmse_ci95_low_rad_s": ci_low,
            "rmse_ci95_high_rad_s": ci_high,
            "test_sequences": int(np.unique(sequence_index).size),
            "test_windows": int(len(truth)),
        })
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "baseline_suite.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "experiment": "corrected_euler_rotor_baseline_suite",
        "seeds": {"train": 7, "test": 19},
        "train_sequences": 120,
        "test_sequences": 30,
        "train_windows": int(len(train_contexts)),
        "test_windows": int(len(test_contexts)),
        "context_steps": CONTEXT_STEPS,
        "horizon_steps": HORIZON_STEPS,
        "bootstrap": {"resampled_unit": "complete test sequence", "n": 10000, "alpha": 0.05},
        "rows": rows,
        "limitations": ["synthetic data", "in-distribution held-out sequences", "no experimental validation"],
    }
    (RESULTS / "baseline_suite.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (REPORTS / "baseline_suite_run.log").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
