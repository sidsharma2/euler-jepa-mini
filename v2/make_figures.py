"""Figures for Euler-JEPA v2, generated only from recorded CSV/NPZ results (results/ -> figures/)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ejepa.plot_style import COLORS, LABELS, apply_style, save_fig  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from ejepa.data import load_split  # noqa: E402
from ejepa.evaluate import T0, HORIZON  # noqa: E402
from ejepa.physics import CONST  # noqa: E402

RES, FIG, CACHE = HERE / "results", HERE / "figures", HERE / "cache"
HZ = (10, 25, 50, 100, 200)


def load_metrics():
    frames = [pd.read_csv(p) for p in sorted(RES.glob("metrics_learned*.csv"), key=lambda q: (q.stem.count("_") == 1, q.stat().st_mtime))]
    learned = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["run", "test"], keep="last")
    classical = pd.read_csv(RES / "metrics_classical.csv")
    return learned, classical


def family(run: str) -> str:
    if run.startswith("jepaft"):
        return "jepa_ft"
    if run.startswith("jepa"):
        return "jepa"
    if run.startswith("recon"):
        return "recon"
    if run.startswith("sup"):
        return "sup"
    return run.split("_")[0]


def fig_problem():
    ep = load_split("test_id", CACHE)
    i = 3
    fig, ax = plt.subplots(2, 1, figsize=(6.6, 4.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    t = np.arange(ep.obs.shape[1]) * CONST.dt_s
    ax[0].imshow(ep.obs[i].T.astype(float), aspect="auto", origin="lower", cmap="magma",
                 extent=[0, t[-1], 0, 3.2], vmin=-1.2, vmax=2.0)
    for h in (1, 2, 3):
        ax[0].plot(t, h * CONST.blade_count * ep.omega[i] / (2 * np.pi) / 1e3, color="w", lw=0.6, ls="--")
    ax[0].set_ylabel("Frequency (kHz)")
    ax[0].text(0.02, 0.93, "casing-microphone log spectrum (model input)", transform=ax[0].transAxes,
               color="w", fontsize=7.5, va="top")
    ax[1].plot(t, ep.omega[i], color="k", label=r"rotor speed $\omega$ (hidden)")
    ax[1].set_ylabel(r"$\omega$ (rad s$^{-1}$)")
    ax2 = ax[1].twinx(); ax2.plot(t, ep.u[i], color=COLORS["recon"], lw=0.9, label="command $u$ (known)")
    ax2.set_ylabel("$u$ (–)")
    ax[1].axvline(T0 * CONST.dt_s, color="0.5", lw=0.6, ls=":")
    ax[1].set_xlabel("Time (s)")
    lines = ax[1].get_lines()[:1] + ax2.get_lines()
    ax[1].legend(lines, [l.get_label() for l in lines], loc="lower left", ncol=2)
    save_fig(fig, FIG / "fig_problem")


def fig_horizon(learned, classical):
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=False)
    for j, test in enumerate(("test_id", "test_ood")):
        a = ax[j]
        for m in ("nominal", "persistence", "greybox"):
            r = classical[(classical.method == m) & (classical.test == test)].iloc[0]
            a.plot(np.array(HZ) * CONST.dt_s, [r[f"rmse_h{h}"] for h in HZ], color=COLORS[m], marker="o", ms=2.5,
                   ls="--" if m != "greybox" else "-", label=LABELS[m])
        groups = {"jepa": "jepa_long_s*+probe300", "recon": "recon_long_s*+probe300", "sup": "sup_all_long_s*"}
        for fam, pat in groups.items():
            sub = learned[(learned.test == test) & learned.run.str.fullmatch(pat.replace("+", r"\+").replace("*", r"\d"))]
            if sub.empty:
                continue
            vals = np.array([[r[f"rmse_h{h}"] for h in HZ] for _, r in sub.iterrows()])
            x = np.array(HZ) * CONST.dt_s
            a.plot(x, vals.mean(0), color=COLORS[fam], marker="o", ms=2.5,
                   label=LABELS[fam] + (" (all labels)" if fam == "sup" else " (300-label probe)"))
            if len(vals) > 1:
                a.fill_between(x, vals.min(0), vals.max(0), color=COLORS[fam], alpha=0.18, lw=0)
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel("Forecast horizon (s)"); a.set_ylabel(r"RMSE (rad s$^{-1}$)")
        a.set_title("In-distribution" if test == "test_id" else "Held-out regime (higher $c_2$, $J$)")
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, -0.12))
    save_fig(fig, FIG / "fig_horizon")


def fig_label_efficiency(learned):
    sub = learned[learned.test == "test_id"].copy()
    sub["family"] = sub.run.map(family)
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.8))
    for metric, a, title in (("filter_rmse", ax[0], "State estimation (filtering)"),
                             ("rmse_h200", ax[1], "2 s open-loop forecast")):
        for fam in ("jepa", "recon", "sup", "jepa_ft"):
            f = sub[(sub.family == fam) & sub.run.str.contains("_long_") & ~sub.run.str.contains("_hi_")
                    & ~sub.run.str.contains("mlp")]
            if fam == "sup":
                pass
            if f.empty:
                continue
            g = f.groupby("labelled")[metric]
            x = np.array(sorted(g.groups)); m = g.mean().loc[x].values
            lo, hi = g.min().loc[x].values, g.max().loc[x].values
            a.plot(x, m, color=COLORS[fam], marker="o", ms=3, label=LABELS[fam])
            a.fill_between(x, lo, hi, color=COLORS[fam], alpha=0.18, lw=0)
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel("Labelled training episodes (speed labels)"); a.set_ylabel(r"RMSE (rad s$^{-1}$)")
        a.set_title(title)
    ax[0].legend(fontsize=6.5)
    save_fig(fig, FIG / "fig_label_efficiency")


def fig_nuisance(learned, classical):
    lams = (0, 0.5, 1, 2, 3)
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.8))
    for metric, a, title in (("filter_rmse", ax[0], "State estimation"), ("rmse_h200", ax[1], "2 s forecast")):
        for fam, pat in (("jepa", r"jepa_long_s\d\+probe300"), ("recon", r"recon_long_s\d\+probe300"), ("sup", r"sup_all_long_s\d")):
            vals = []
            for lam in lams:
                s = learned[(learned.test == f"test_lam{lam:g}") & learned.run.str.fullmatch(pat)][metric]
                vals.append((s.mean(), s.min(), s.max()) if len(s) else (np.nan,) * 3)
            v = np.array(vals)
            a.plot(lams, v[:, 0], color=COLORS[fam], marker="o", ms=3, label=LABELS[fam])
            a.fill_between(lams, v[:, 1], v[:, 2], color=COLORS[fam], alpha=0.18, lw=0)
        gb = [classical[(classical.method == "greybox") & (classical.test == f"test_lam{lam:g}")][metric].mean()
              for lam in lams]
        a.plot(lams, gb, color=COLORS["greybox"], marker="s", ms=3, label=LABELS["greybox"])
        a.axvline(1.0, color="0.6", lw=0.6, ls=":")
        a.set_yscale("log"); a.set_xlabel(r"Nuisance level $\lambda$ (training: $\lambda=1$)")
        a.set_ylabel(r"RMSE (rad s$^{-1}$)"); a.set_title(title)
    ax[0].legend(fontsize=6.5)
    save_fig(fig, FIG / "fig_nuisance")


def fig_step_response():
    cl = np.load(RES / "classical_test_id.npz")
    true = cl["sens_true"]
    t = np.arange(1, HORIZON + 1) * CONST.dt_s
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for i in (0, 5, 9):
        ax[0].plot(t, true[i], color="k", lw=1.8)
        ax[0].plot(t, cl["sens_greybox"][i], color=COLORS["greybox"], ls=":", lw=1.0)
        for name, fam in (("jepa_long_s0+probe300", "jepa"), ("recon_long_s0+probe300", "recon"), ("sup_all_long_s0", "sup")):
            p = RES / f"pred_{name}_test_id.npz"
            if p.exists():
                ax[0].plot(t, np.load(p)["sens"][i], color=COLORS[fam], lw=1.0, ls="--")
    ax[0].set_xlabel("Time after step (s)"); ax[0].set_ylabel(r"$\partial\hat\omega/\partial u$ (rad s$^{-1}$)")
    ax[0].set_title("Step response, three test episodes (black: truth)")
    # Error distribution of the final sensitivity.
    data, names = [], []
    for name, fam in (("jepa_long_s0+probe300", "jepa"), ("recon_long_s0+probe300", "recon"), ("sup_all_long_s0", "sup")):
        p = RES / f"pred_{name}_test_id.npz"
        if p.exists():
            s = np.load(p)["sens"]
            data.append(np.abs(s[:, -1] - true[:, -1]) / np.abs(true[:, -1])); names.append(fam)
    data.append(np.abs(cl["sens_greybox"][:, -1] - true[:, -1]) / np.abs(true[:, -1])); names.append("greybox")
    bp = ax[1].boxplot(data, showfliers=False, widths=0.55, patch_artist=True)
    for patch, n in zip(bp["boxes"], names):
        patch.set_facecolor(COLORS[n]); patch.set_alpha(0.55)
    ax[1].set_xticks(range(1, len(names) + 1), [LABELS[n].split(" (")[0].replace(" world model", "") for n in names],
                     rotation=15, fontsize=7)
    ax[1].set_ylabel("Relative error of 2 s gain"); ax[1].set_yscale("log")
    ax[1].set_title("Control-authority error (v1 failure mode)")
    save_fig(fig, FIG / "fig_step_response")


def fig_param_probes():
    df = pd.concat([pd.read_csv(p) for p in sorted(RES.glob("param_probes*.csv"), key=lambda q: (q.stem.count("_") == 1, q.stat().st_mtime))]).drop_duplicates("run", keep="last")
    df = df[(df.run.str.contains("_long_") | df.run.str.startswith("random")) & ~df.run.str.contains("_hi_")]
    df["family"] = df.run.map(lambda r: "random" if r.startswith("random") else family(r))
    cols = ["mass_flow_kg_s", "inertia_kg_m2", "speed_feedback_m", "quad_load_N_m_s2", "actuator_tau_s", "delay_steps"]
    names = [r"$\dot m$", "$J$", r"$k_\omega$", "$c_2$", r"$\tau_a$", "$d$"]
    fams = [f for f in ("jepa", "recon", "sup", "random") if f in set(df.family)]
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    w = 0.8 / len(fams)
    for j, fam in enumerate(fams):
        g = df[df.family == fam][cols]
        ax.bar(np.arange(len(cols)) + (j - (len(fams) - 1) / 2) * w, g.mean().clip(lower=-0.2), w,
               yerr=g.std().fillna(0), color=COLORS[fam], alpha=0.85, label=LABELS[fam], capsize=1.5,
               error_kw={"lw": 0.6})
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(cols)), names); ax.set_ylabel(r"Linear-probe $R^2$ at $t=2$ s")
    ax.set_ylim(-0.2, 1.0); ax.legend(ncol=2, fontsize=6.5)
    ax.set_title("Hidden physical parameters recoverable from the latent belief")
    save_fig(fig, FIG / "fig_param_probes")


MPC_SHOW = [  # controller, family, label
    ("sup_all_long_s0", "sup", "Supervised, 3000 labels"),
    ("greybox", "greybox", "Tracker + grey-box (true ODE)"),
    ("jepaft_n300_long_s0", "jepa_ft", "JEPA + fine-tune, 300 labels"),
    ("jepa_long_s0", "jepa", "JEPA world model, 300-label probe"),
    ("recon_long_s0", "recon", "Reconstruction WM, 300-label probe"),
    ("sup_n300_long_s0", "sup", "Supervised, 300 labels"),
    ("pi", "pi", "PI on tracked speed"),
]


def fig_mpc():
    tests = [("test_id", r"$\lambda=1$ (training nuisance)"), ("test_lam3", r"$\lambda=3$ (heavy nuisance)")]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for a, (test, title) in zip(ax, tests):
        df = pd.read_csv(RES / ("mpc_tracking.csv" if test == "test_id" else f"mpc_tracking_{test}.csv")).set_index("controller")
        rows = [r for r in MPC_SHOW if r[0] in df.index]
        y = np.arange(len(rows))
        val = np.array([max(df.loc[r[0], "excess_mse_vs_oracle"], 0.3) for r in rows])
        lo = np.array([max(df.loc[r[0], "excess_mse_ci_lo"], 0.3) for r in rows])
        hi = np.array([df.loc[r[0], "excess_mse_ci_hi"] for r in rows])
        a.barh(y, val, color=[COLORS[r[1]] for r in rows], alpha=0.85)
        a.errorbar(val, y, xerr=[val - lo, hi - val], fmt="none", ecolor="k", lw=0.7, capsize=1.5)
        a.set_xscale("log"); a.set_xlim(1, 1e4)
        a.set_yticks(y, [r[2] for r in rows], fontsize=7)
        a.set_title(title); a.set_xlabel(r"Excess tracking MSE vs oracle (rad$^2$ s$^{-2}$)")
    ax[0].invert_yaxis()
    fig.subplots_adjust(wspace=0.06)
    save_fig(fig, FIG / "fig_mpc")

    tr = np.load(RES / "mpc_traces_test_lam3.npz")
    t = (T0 + np.arange(tr["ref"].shape[1])) * CONST.dt_s
    fig, a = plt.subplots(figsize=(3.4, 2.6))
    i = 2
    a.plot(t, tr["ref"][i], color="k", lw=1.0, ls="--", label="set-point")
    for name, fam, lab in (("oracle", "oracle", "Oracle"), ("greybox", "greybox", "Grey-box"), ("pi", "pi", "PI"),
                           ("jepaft_n300_long_s0", "jepa_ft", "JEPA + FT (300)"), ("sup_n300_long_s0", "sup", "Supervised (300)")):
        if f"omega_{name}" in tr:
            a.plot(t, tr[f"omega_{name}"][i], color=COLORS[fam], lw=1.0, label=lab)
    a.set_xlabel("Time (s)"); a.set_ylabel(r"$\omega$ (rad s$^{-1}$)"); a.legend(fontsize=5.8, ncol=2, loc="lower left")
    a.set_title(r"Closed-loop tracking at $\lambda=3$ (one episode)")
    save_fig(fig, FIG / "fig_mpc_trace")


def fig_example_forecast():
    ep = load_split("test_id", CACHE)
    cl = np.load(RES / "classical_test_id.npz")
    t = np.arange(ep.omega.shape[1]) * CONST.dt_s
    tf = (T0 + 1 + np.arange(HORIZON)) * CONST.dt_s
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=False)
    for a, i in zip(axes, (1, 7, 12)):
        a.plot(t, ep.omega[i], color="k", lw=1.4, label="truth")
        a.plot(tf, cl["greybox"][i], color=COLORS["greybox"], ls=":", label="grey-box")
        for name, fam in (("jepa_long_s0+probe300", "jepa"), ("recon_long_s0+probe300", "recon"), ("sup_all_long_s0", "sup")):
            p = RES / f"pred_{name}_test_id.npz"
            if p.exists():
                a.plot(tf, np.load(p)["pred"][i], color=COLORS[fam], ls="--", lw=1.0, label=LABELS[fam].split(" (")[0])
        a.axvline(T0 * CONST.dt_s, color="0.6", lw=0.6, ls=":")
        a.set_xlabel("Time (s)")
    axes[0].set_ylabel(r"$\omega$ (rad s$^{-1}$)"); axes[0].legend(fontsize=6)
    save_fig(fig, FIG / "fig_example_forecast")


if __name__ == "__main__":
    apply_style()
    learned, classical = load_metrics()
    for fn in (fig_problem, lambda: fig_horizon(learned, classical), lambda: fig_label_efficiency(learned),
               lambda: fig_nuisance(learned, classical), fig_step_response, fig_param_probes, fig_mpc,
               fig_example_forecast):
        try:
            fn()
        except Exception as exc:  # keep going; report which figure lacked data
            print("figure failed:", getattr(fn, "__name__", "lambda"), repr(exc))
    print("figures written to", FIG)
