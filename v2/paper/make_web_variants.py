"""Generate 16:9 web-gallery variants of selected figures (plots/fig_web_*.tex).

The paper figures are sized for an IEEE page; a portfolio carousel slide is 16:9, so wide two-panel
charts leave large white bands there.  The variants change only sizes, never data or styling.
"""
from pathlib import Path

PLOTS = Path(__file__).resolve().parent / "plots"
VARIANTS = {
    # name: (source, [(old, new), ...])
    "fig_web_labels": ("fig_labels.tex", [("panel, xmode=log, ymode=log, rad,", "panel, height=6.8cm, xmode=log, ymode=log, rad,")]),
    "fig_web_mpc": ("fig_mpc.tex", [("width=6.4cm, height=6.2cm,", "width=6.6cm, height=7.4cm,")]),
    "fig_web_problem": ("fig_problem.tex", [("width=12cm, xmin=0, xmax=3.99]", "width=15cm, xmin=0, xmax=3.99]"),
                                            ("at={(group c1r2.south west)}, width=12cm,", "at={(group c1r2.south west)}, width=15cm,")]),
}

for name, (src, pairs) in VARIANTS.items():
    text = (PLOTS / src).read_text(encoding="utf-8")
    for old, new in pairs:
        assert old in text, (name, old)
        text = text.replace(old, new)
    (PLOTS / f"{name}.tex").write_text(text, encoding="utf-8")
    print("wrote", name)
