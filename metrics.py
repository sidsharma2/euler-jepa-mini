"""Metrics used by the additive Euler-JEPA remediation experiments."""

from __future__ import annotations

import numpy as np


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Return RMSE in the units of the supplied arrays."""
    prediction = np.asarray(prediction, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def r2(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Return coefficient of determination, guarding constant targets."""
    prediction = np.asarray(prediction, dtype=float)
    truth = np.asarray(truth, dtype=float)
    denominator = np.sum((truth - np.mean(truth)) ** 2)
    if denominator == 0.0:
        return float("nan")
    return float(1.0 - np.sum((truth - prediction) ** 2) / denominator)


def skill_vs_persistence(
    prediction: np.ndarray, truth: np.ndarray, persistence_prediction: np.ndarray
) -> float:
    """Return 1 - MSE(prediction) / MSE(persistence)."""
    prediction = np.asarray(prediction, dtype=float)
    truth = np.asarray(truth, dtype=float)
    persistence_prediction = np.asarray(persistence_prediction, dtype=float)
    denominator = np.mean((persistence_prediction - truth) ** 2)
    if denominator == 0.0:
        return float("nan")
    return float(1.0 - np.mean((prediction - truth) ** 2) / denominator)


def bootstrap_ci(
    prediction: np.ndarray,
    truth: np.ndarray,
    sequence_index: np.ndarray,
    n: int = 10_000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> tuple[float, float]:
    """Bootstrap an RMSE CI by resampling complete test sequences.

    Windows from one sequence are kept together. This preserves the effective
    sample size of the corrected benchmark instead of treating overlapping
    windows as independent observations.
    """
    prediction = np.asarray(prediction, dtype=float)
    truth = np.asarray(truth, dtype=float)
    sequence_index = np.asarray(sequence_index)
    unique_sequences = np.unique(sequence_index)
    if unique_sequences.size == 0:
        raise ValueError("sequence_index must contain at least one sequence")
    rng = np.random.default_rng(seed)
    sequence_positions = {
        sequence: np.flatnonzero(sequence_index == sequence)
        for sequence in unique_sequences
    }
    sampled = rng.integers(0, unique_sequences.size, size=(n, unique_sequences.size))
    values = np.empty(n, dtype=float)
    for index, sample in enumerate(sampled):
        positions = np.concatenate([sequence_positions[unique_sequences[i]] for i in sample])
        values[index] = rmse(prediction[positions], truth[positions])
    return (
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    )
