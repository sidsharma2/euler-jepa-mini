"""Journal-quality matplotlib style shared by every v2 figure (STIX serif, PDF + PNG)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {
    "jepa": "#1f4e79", "jepa_ft": "#5b8fc9", "recon": "#b5562b", "sup": "#3a7d44",
    "greybox": "#6b6b6b", "nominal": "#a8a8a8", "oracle": "#000000", "persistence": "#c9b458",
    "pi": "#8c6bb1", "random": "#bdbdbd", "truth": "#000000",
}
LABELS = {
    "jepa": "JEPA world model", "jepa_ft": "JEPA + fine-tune", "recon": "Reconstruction world model",
    "sup": "Supervised", "greybox": "Tracker + grey-box (true ODE form)", "nominal": "Nominal physics",
    "oracle": "Oracle (true parameters)", "persistence": "Persistence", "pi": "PI on tracked speed",
    "random": "Random encoder",
}


def apply_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "axes.linewidth": 0.7, "lines.linewidth": 1.3, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight",
        "figure.dpi": 110, "figure.subplot.wspace": 0.34,
    })


def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
