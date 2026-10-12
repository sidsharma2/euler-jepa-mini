"""Compile every TikZ/pgfplots figure and export it for the paper (PDF) and the web (SVG).

    python export_plot_data.py      # results -> data/*.csv
    python build_figures.py         # plots/fig_*.tex -> figures/*.pdf, svg/*.svg, png/*.png
"""
import shutil
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
PLOTS, FIGS, SVG, PNG = PAPER / "plots", PAPER / "figures", PAPER / "svg", PAPER / "png"


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="ignore")


def main(only=None):
    for d in (FIGS, SVG, PNG):
        d.mkdir(exist_ok=True)
    failed = []
    for tex in sorted(PLOTS.glob("fig_*.tex")):
        if only and tex.stem not in only:
            continue
        r = run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name], PLOTS)
        pdf = PLOTS / f"{tex.stem}.pdf"
        if r.returncode != 0 or not pdf.exists():
            err = [l for l in r.stdout.splitlines() if l.startswith("!")][:2]
            failed.append((tex.stem, err)); print("FAILED", tex.stem, err); continue
        shutil.copy(pdf, FIGS / pdf.name)
        # SVG with glyphs converted to paths: renders identically in any browser (portfolio).
        run(["pdftocairo", "-svg", str(pdf), str(SVG / f"{tex.stem}.svg")], PLOTS)
        run(["pdftoppm", "-png", "-r", "200", "-singlefile", str(pdf), str(PNG / tex.stem)], PLOTS)
        for ext in (".aux", ".log"):
            (PLOTS / f"{tex.stem}{ext}").unlink(missing_ok=True)
        print("ok", tex.stem)
    return failed


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1:] or None) else 0)
