"""Measure whether the current JEPA benefits from substantially more epochs."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
HERE = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_training_diagnostics as diagnostics  # noqa: E402


def main() -> None:
    module = diagnostics.load_module()
    train_sequences, train_parameters = diagnostics.make_constant_sequences(module, 120, seed=7)
    test_sequences, test_parameters = diagnostics.make_constant_sequences(module, 30, seed=19)
    train_contexts, train_targets, _ = module.make_windows(train_sequences, train_parameters)
    test_contexts, test_targets, _ = module.make_windows(test_sequences, test_parameters)
    rows = []
    for variant in ("prediction_only", "prediction_plus_vc"):
        for epochs in (180, 720, 1440):
            history, endpoint_rmse = diagnostics.train(
                module, train_contexts, train_targets, test_contexts, test_targets,
                seed=7, variant=variant, epochs=epochs,
            )
            final = history[-1]
            rows.append({
                "variant": variant,
                "epochs": epochs,
                "endpoint_probe_rmse_rad_s": endpoint_rmse,
                "train_prediction_loss": final["train_prediction_loss"],
                "test_prediction_loss": final["test_prediction_loss"],
                "embedding_std_mean": final["embedding_std_mean"],
                "embedding_effective_rank": final["embedding_effective_rank"],
            })
    output = HERE / "results" / "epoch_scaling.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(rows)


if __name__ == "__main__":
    main()
