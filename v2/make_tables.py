"""LaTeX + Markdown result tables generated from results/*.csv (no hand-typed numbers)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES, OUT = HERE / "results", HERE / "results" / "tables"
sys.path.insert(0, str(HERE))
from make_figures import load_metrics  # noqa: E402

ROWS = [  # label, source, run regex (learned) or method (classical)
    ("Oracle (true parameters, true state)", "c", "oracle"),
    ("Tracker + grey-box ID (true ODE form)", "c", "greybox"),
    ("Nominal physics + tracker", "c", "nominal"),
    ("Persistence", "c", "persistence"),
    ("JEPA world model, linear probe (300)", "l", r"jepa_long_s\d\+probe300"),
    ("JEPA world model, MLP probe (300)", "l", r"jepa_long_s\d\+mlp300"),
    ("Reconstruction WM, linear probe (300)", "l", r"recon_long_s\d\+probe300"),
    ("Reconstruction WM, MLP probe (300)", "l", r"recon_long_s\d\+mlp300"),
    ("Supervised, 300 labels", "l", r"sup_n300_long_s\d"),
    ("JEPA + fine-tune, 300 labels", "l", r"jepaft_n300_long_s\d"),
    ("Supervised, all 3000 labels", "l", r"sup_all_long_s\d"),
]
ABLATIONS = [  # seed-0 development runs
    ("JEPA, long horizon (main), linear probe", "l", r"jepa_long_s0\+probe300"),
    ("JEPA, short horizon (K=64, 2 s crops)", "l", r"jepa_s0\+probe300"),
    ("JEPA, random frozen trunk (no stage 1)", "l", r"jepa_rand_s0\+probe300"),
    ("JEPA, adaptive context embedding", "l", r"jepa_ctx_s0\+probe300"),
    ("Reconstruction, short horizon", "l", r"recon_s0\+probe300"),
    ("Supervised all labels, short horizon", "l", r"sup_all_s0"),
]


def pick(learned, classical, src, key, test):
    if src == "c":
        return classical[(classical.method == key) & (classical.test == test)]
    return learned[(learned.test == test) & learned.run.str.fullmatch(key)]


def fmt(series, digits=2):
    s = series.dropna()
    if s.empty:
        return "--"
    if len(s) == 1:
        return f"{s.iloc[0]:.{digits}f}"
    return f"{s.mean():.{digits}f} $\\pm$ {s.std():.{digits}f}"


def md(series, digits=2):
    return re.sub(r" \$\\pm\$ ", " ± ", fmt(series, digits))


def main_table(learned, classical, rows=None):
    cols = [("filter_rmse", "test_id", "Filter"), ("rmse_h25", "test_id", "0.25 s"), ("rmse_h100", "test_id", "1 s"),
            ("rmse_h200", "test_id", "2 s"), ("rmse_h200", "test_ood", "2 s OOD"),
            ("ode_residual_Nm", "test_id", "Torque res."), ("sens_rel_err_final", "test_id", "Gain err.")]
    tex = ["\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
           "Method & " + " & ".join(c[2] for c in cols) + "\\\\", "\\midrule"]
    mdl = ["| Method | " + " | ".join(c[2] for c in cols) + " |", "|---|" + "---:|" * len(cols)]
    for label, src, key in (rows or ROWS):
        cells_t, cells_m = [], []
        for metric, test, _ in cols:
            sub = pick(learned, classical, src, key, test)
            series = sub[metric] if metric in sub else pd.Series(dtype=float)
            d = 3 if metric == "sens_rel_err_final" else 2
            cells_t.append(fmt(series, d)); cells_m.append(md(series, d))
        if all(c == "--" for c in cells_t):
            continue
        tex.append(label + " & " + " & ".join(cells_t) + "\\\\")
        mdl.append(f"| {label} | " + " | ".join(cells_m) + " |")
    tex += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(tex), "\n".join(mdl)


def label_table(learned, metric="rmse_h200"):
    sub = learned[learned.test == "test_id"]
    fams = [("JEPA + probe", lambda n: rf"jepa_long_s\d\+probe{n}"), ("Recon. + probe", lambda n: rf"recon_long_s\d\+probe{n}"),
            ("Supervised", lambda n: rf"sup_n{n}_long_s\d" if n != 3000 else r"sup_all_long_s\d"),
            ("JEPA + fine-tune", lambda n: rf"jepaft_n{n}_long_s\d")]
    ns = (10, 30, 100, 300, 3000)
    tex = ["\\begin{tabular}{l" + "r" * len(ns) + "}", "\\toprule",
           "Labelled episodes & " + " & ".join(map(str, ns)) + "\\\\", "\\midrule"]
    mdl = ["| Labelled episodes | " + " | ".join(map(str, ns)) + " |", "|---|" + "---:|" * len(ns)]
    for name, pat in fams:
        cells = [sub[sub.run.str.fullmatch(pat(n))][metric] for n in ns]
        tex.append(name + " & " + " & ".join(fmt(c) for c in cells) + "\\\\")
        mdl.append(f"| {name} | " + " | ".join(md(c) for c in cells) + " |")
    tex += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(tex), "\n".join(mdl)


def nuisance_table(learned, classical, metric="filter_rmse"):
    lams = (0, 0.5, 1, 2, 3)
    rows = [("JEPA (trained $\\lambda=1$)", "l", r"jepa_long_s\d\+probe300"), ("Recon. (trained $\\lambda=1$)", "l", r"recon_long_s\d\+probe300"),
            ("JEPA (trained $\\lambda=2.5$)", "l", r"jepa_hi_long_s\d\+probe300"), ("Recon. (trained $\\lambda=2.5$)", "l", r"recon_hi_long_s\d\+probe300"),
            ("Supervised, all labels", "l", r"sup_all_long_s\d"), ("Tracker / grey-box", "c", "greybox")]
    tex = ["\\begin{tabular}{l" + "r" * len(lams) + "}", "\\toprule",
           "$\\lambda$ at test & " + " & ".join(f"{l:g}" for l in lams) + "\\\\", "\\midrule"]
    mdl = ["| λ at test | " + " | ".join(f"{l:g}" for l in lams) + " |", "|---|" + "---:|" * len(lams)]
    for label, src, key in rows:
        cells = [pick(learned, classical, src, key, f"test_lam{l:g}").get(metric, pd.Series(dtype=float)) for l in lams]
        if all(c.dropna().empty for c in cells):
            continue
        tex.append(label + " & " + " & ".join(fmt(c) for c in cells) + "\\\\")
        mdl.append(f"| {label.replace('$', '').replace(chr(92) + 'lambda', 'λ')} | " + " | ".join(md(c) for c in cells) + " |")
    tex += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(tex), "\n".join(mdl)


def probe_table():
    files = sorted(RES.glob("param_probes*.csv"), key=lambda q: (q.stem.count("_") == 1, q.stat().st_mtime))
    if not files:
        return "", ""
    df = pd.concat([pd.read_csv(p) for p in files]).drop_duplicates("run", keep="last")
    df["family"] = df.run.str.replace(r"_s\d$", "", regex=True)
    cols = ["mass_flow_kg_s", "inertia_kg_m2", "speed_feedback_m", "quad_load_N_m_s2", "actuator_tau_s", "delay_steps"]
    names = ["$\\dot m$", "$J$", "$k_\\omega$", "$c_2$", "$\\tau_a$", "$d$"]
    tex = ["\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule", "Latent & " + " & ".join(names) + "\\\\", "\\midrule"]
    mdl = ["| Latent | ṁ | J | k_ω | c₂ | τ_a | d |", "|---|---:|---:|---:|---:|---:|---:|"]
    for fam, g in df.groupby("family", sort=False):
        tex.append(fam.replace("_", "\\_") + " & " + " & ".join(fmt(g[c]) for c in cols) + "\\\\")
        mdl.append(f"| {fam} | " + " | ".join(md(g[c]) for c in cols) + " |")
    tex += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(tex), "\n".join(mdl)


def mpc_table(test="test_id"):
    p = RES / ("mpc_tracking.csv" if test == "test_id" else f"mpc_tracking_{test}.csv")
    if not p.exists():
        return "", ""
    df = pd.read_csv(p)
    if "excess_mse_vs_oracle" not in df:
        return "", ""
    df = df.sort_values("excess_mse_vs_oracle")
    tex = ["\\begin{tabular}{lrrr}", "\\toprule",
           "Controller & Tracking RMSE & Excess MSE vs oracle [95\\% CI] & Steady RMSE\\\\", "\\midrule"]
    mdl = ["| Controller | Tracking RMSE | Excess MSE vs oracle (rad²/s²) [paired 95 % CI] | Steady RMSE |",
           "|---|---:|---:|---:|"]
    for _, r in df.iterrows():
        name = r.controller.replace("_", "\\_")
        ex = f"{r.excess_mse_vs_oracle:.1f} [{r.excess_mse_ci_lo:.1f}, {r.excess_mse_ci_hi:.1f}]"
        tex.append(f"{name} & {r.tracking_rmse:.2f} & {ex} & {r.steady_rmse:.2f}\\\\")
        mdl.append(f"| {r.controller} | {r.tracking_rmse:.2f} | {ex} | {r.steady_rmse:.2f} |")
    tex += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(tex), "\n".join(mdl)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    learned, classical = load_metrics()
    blocks = {"main": main_table(learned, classical), "ablations": main_table(learned, classical, ABLATIONS), "labels_forecast": label_table(learned, "rmse_h200"),
              "labels_filter": label_table(learned, "filter_rmse"), "nuisance_filter": nuisance_table(learned, classical),
              "nuisance_forecast": nuisance_table(learned, classical, "rmse_h200"), "param_probes": probe_table(),
              "mpc": mpc_table(), "mpc_lam3": mpc_table("test_lam3")}
    summary = ["# Euler-JEPA v2 result tables (generated by make_tables.py)", "",
               "All errors are rotor-speed RMSE in rad/s unless stated. Mean ± s.d. over seeds where several exist."]
    for name, (tex, mdt) in blocks.items():
        if tex:
            (OUT / f"{name}.tex").write_text(tex, encoding="utf-8")
            summary += ["", f"## {name}", "", mdt]
    (OUT / "tables.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
