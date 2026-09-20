"""Plot the additive JEPA-mini improvement ablation."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "results" / "jepa_improvement_ablation.csv"
OUTPUT = HERE / "results" / "figures" / "jepa_improvement_ablation"


def main() -> None:
    """Save a vector/Png summary of the controlled ablation."""
    data = pd.read_csv(CSV)
    labels = {
        "baseline": "baseline",
        "ode_regularized": "ODE penalty",
        "latent_variance_regularized": "latent variance",
        "combined_regularization": "combined",
    }
    order = list(labels)
    positions = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), sharex=True)
    metrics = [
        ("full_trajectory_rmse_rad_s", "full-trajectory RMSE (rad s$^{-1}$)"),
        ("ode_residual_rmse_N_m", "ODE residual (N m)"),
        ("latent_effective_rank", "latent effective rank"),
    ]
    markers = ["o", "s", "^"]
    for axis, (column, ylabel), marker in zip(axes, metrics, markers):
        for index, name in enumerate(order):
            values = data.loc[data["configuration"] == name, column].to_numpy()
            axis.scatter(
                np.full(values.shape, positions[index]),
                values,
                marker=marker,
                facecolors="none" if index == 0 else "black",
                edgecolors="black",
                s=34,
                zorder=3,
            )
            axis.plot(
                positions[index],
                values.mean(),
                marker="_",
                color="black",
                markersize=12,
                markeredgewidth=1.2,
                zorder=4,
            )
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions)
        axis.set_xticklabels([labels[name] for name in order], rotation=20, ha="right")
        axis.tick_params(axis="x", pad=2)
        axis.grid(False)
    fig.suptitle("JEPA-mini improvement ablation at $\\epsilon=0.5$", y=1.02)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
