"""Export tidy CSV tables for the pgfplots figures (paper/data/*.csv) from v2/results.

Every number plotted in the paper comes from these files; nothing is typed by hand.
Seed aggregates are written as <name>_m (mean), <name>_lo (min), <name>_hi (max).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent
V2 = PAPER.parent
RES, RUNS, LOGS, DATA = V2 / "results", V2 / "runs", V2 / "logs", PAPER / "data"
sys.path.insert(0, str(V2))

HZ = (10, 25, 50, 100, 200)
LAMS = (0, 0.5, 1, 2, 3)
NAN = float("nan")


def learned_metrics():
    files = sorted(RES.glob("metrics_learned*.csv"), key=lambda q: (q.stem.count("_") == 1, q.stat().st_mtime))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.drop_duplicates(subset=["run", "test"], keep="last")


def agg(df, pattern, test, col):
    s = df[(df.test == test) & df.run.str.fullmatch(pattern)][col].dropna()
    return (s.mean(), s.min(), s.max()) if len(s) else (NAN, NAN, NAN)


def put(row, name, triple):
    row[f"{name}_m"], row[f"{name}_lo"], row[f"{name}_hi"] = triple


def write(name, rows):
    DATA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(DATA / f"{name}.csv", index=False, na_rep="nan", float_format="%.6g")


MAIN = {"jepa": r"jepa_long_s\d\+probe300", "jepamlp": r"jepa_long_s\d\+mlp300", "recon": r"recon_long_s\d\+probe300",
        "reconmlp": r"recon_long_s\d\+mlp300", "sup": r"sup_all_long_s\d", "sup300": r"sup_n300_long_s\d",
        "ft": r"jepaft_n300_long_s\d"}


def export_horizon(L, C):
    for test in ("test_id", "test_ood"):
        rows = []
        for h in HZ:
            row = {"h": h * 0.01}
            for m in ("greybox", "nominal", "persistence"):
                row[m] = C[(C.method == m) & (C.test == test)][f"rmse_h{h}"].iloc[0]
            for k, pat in MAIN.items():
                put(row, k, agg(L, pat, test, f"rmse_h{h}"))
            rows.append(row)
        write(f"horizon_{test}", rows)


def export_labels(L):
    fams = {"jepa": lambda n: rf"jepa_long_s\d\+probe{n}", "recon": lambda n: rf"recon_long_s\d\+probe{n}",
            "sup": lambda n: rf"sup_n{n}_long_s\d" if n != 3000 else r"sup_all_long_s\d",
            "ft": lambda n: rf"jepaft_n{n}_long_s\d"}
    for metric in ("filter_rmse", "rmse_h200"):
        rows = []
        for n in (10, 30, 100, 300, 3000):
            row = {"n": n}
            for k, pat in fams.items():
                put(row, k, agg(L, pat(n), "test_id", metric))
            rows.append(row)
        write(f"labels_{metric}", rows)


def export_nuisance(L, C, H):
    pats = {"jepa": MAIN["jepa"], "recon": MAIN["recon"], "sup": MAIN["sup"],
            "jepahi": r"jepa_hi_long_s\d\+probe300", "reconhi": r"recon_hi_long_s\d\+probe300",
            "jepamix": r"jepa_mix_long_s\d\+probe300", "reconmix": r"recon_mix_long_s\d\+probe300"}
    for metric in ("filter_rmse", "rmse_h200"):
        rows = []
        for lam in LAMS:
            test = f"test_lam{lam:g}"
            row = {"lam": lam}
            for k, pat in pats.items():
                put(row, k, agg(L, pat, test, metric))
            g = C[(C.method == "greybox") & (C.test == test)]
            row["greybox"] = g[metric].iloc[0] if len(g) and metric in g else NAN
            for sensor in ("jepa_long_s0", "jepaft_n300_long_s0", "tracker_window"):
                h = H[(H.run == f"hybrid[{sensor}]") & (H.test == test)] if H is not None else []
                row[f"hyb_{sensor.split('_')[0]}"] = h[metric].iloc[0] if len(h) and metric in h else NAN
            rows.append(row)
        write(f"nuisance_{metric}", rows)


def export_step(L):
    cl = np.load(RES / "classical_test_id.npz")
    true = cl["sens_true"]
    t = np.arange(1, true.shape[1] + 1) * 0.01
    rows = {"t": t}
    eps = (0, 5, 9)
    for i in eps:
        rows[f"true{i}"] = true[i]
        rows[f"grey{i}"] = cl["sens_greybox"][i]
    gains = []
    for key, run in (("jepa", "jepa_long_s0+probe300"), ("recon", "recon_long_s0+probe300"), ("sup", "sup_all_long_s0")):
        p = RES / f"pred_{run}_test_id.npz"
        if p.exists():
            s = np.load(p)["sens"]
            for i in eps:
                rows[f"{key}{i}"] = s[i]
            e = np.abs(s[:, -1] - true[:, -1]) / np.abs(true[:, -1])
            gains.append({"method": key, "q25": np.quantile(e, .25), "med": np.median(e), "q75": np.quantile(e, .75)})
    e = np.abs(cl["sens_greybox"][:, -1] - true[:, -1]) / np.abs(true[:, -1])
    gains.append({"method": "grey", "q25": np.quantile(e, .25), "med": np.median(e), "q75": np.quantile(e, .75)})
    pd.DataFrame(rows).iloc[::2].to_csv(DATA / "step_examples.csv", index=False, float_format="%.5g")
    write("step_gain_error", gains)
    for g in gains:
        write(f"step_gain_error_{g['method']}", [g])


def export_probes():
    files = sorted(RES.glob("param_probes*.csv"), key=lambda q: (q.stem.count("_") == 1, q.stat().st_mtime))
    df = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates("run", keep="last")
    cols = ["mass_flow_kg_s", "inertia_kg_m2", "speed_feedback_m", "quad_load_N_m_s2", "actuator_tau_s", "delay_steps"]
    names = ["mdot", "J", "kw", "c2", "tau", "d"]
    fams = {"jepa": r"jepa_long_s\d", "recon": r"recon_long_s\d", "sup": r"sup_all_long_s\d", "random": r"random_s\d"}
    rows = []
    for c, nm in zip(cols, names):
        row = {"param": nm}
        for k, pat in fams.items():
            s = df[df.run.str.fullmatch(pat)][c]
            row[f"{k}_m"], row[f"{k}_sd"] = (s.mean(), s.std() if len(s) > 1 else 0.0) if len(s) else (NAN, NAN)
        rows.append(row)
    write("param_probes", rows)


MPC_SHOW = [("sup_all_long_s0", "Supervised (3000 labels)", "sup"), ("greybox", "Tracker + grey-box (true ODE)", "grey"),
            ("jepaft_n300_long_s0", "JEPA + fine-tune (300 labels)", "ft"),
            ("jepa_long_s0", "JEPA WM (300-label probe)", "jepa"),
            ("jepa_hi_long_s0", "JEPA trained at $\\lambda{=}2.5$", "jepahi"),
            ("recon_long_s0", "Reconstruction WM (300-label probe)", "recon"),
            ("sup_n300_long_s0", "Supervised (300 labels)", "sup300"), ("pi", "PI on tracked speed", "pi")]


def export_mpc():
    for test, f in (("id", "mpc_tracking.csv"), ("lam3", "mpc_tracking_test_lam3.csv")):
        if not (RES / f).exists():
            continue
        df = pd.read_csv(RES / f).set_index("controller")
        rows = []
        for k, (c, label, color) in enumerate(c for c in MPC_SHOW if c[0] in df.index):
            r = df.loc[c]
            rows.append({"y": k, "label": label, "color": color, "excess": max(r.excess_mse_vs_oracle, 0.3),
                         "lo": max(r.excess_mse_ci_lo, 0.3), "hi": r.excess_mse_ci_hi,
                         "track": r.tracking_rmse, "steady": r.steady_rmse})
        df_out = pd.DataFrame(rows)
        df_out["err_minus"] = df_out.excess - df_out.lo
        df_out["err_plus"] = df_out.hi - df_out.excess
        df_out.to_csv(DATA / f"mpc_{test}.csv", index=False, float_format="%.5g")
    p = RES / "mpc_traces_test_lam3.npz"
    if p.exists():
        tr = np.load(p)
        i = 2
        rows = {"t": (199 + np.arange(tr["ref"].shape[1])) * 0.01, "ref": tr["ref"][i]}
        for k, name in (("oracle", "oracle"), ("grey", "greybox"), ("pi", "pi"), ("ft", "jepaft_n300_long_s0"),
                        ("sup300", "sup_n300_long_s0")):
            if f"omega_{name}" in tr:
                rows[k] = tr[f"omega_{name}"][i]
        pd.DataFrame(rows).to_csv(DATA / "mpc_trace_lam3.csv", index=False, float_format="%.5g")


def export_ablation(L):
    abl = [("Main: two-stage JEPA (long horizon)", r"jepa_long_s0\+probe300"),
           ("Full-horizon rollouts (2 s)", r"jepa_xlong_s0\+probe300"),
           ("Nuisance-randomised training", r"jepa_mix_long_s0\+probe300"),
           ("Short horizon (0.64 s on 2 s crops)", r"jepa_s0\+probe300"),
           ("No stage-1 pre-training", r"jepa_rand_s0\+probe300"),
           ("Adaptive context embedding", r"jepa_ctx_s0\+probe300")]
    rows = []
    for k, (label, pat) in enumerate(abl):
        r = {"y": k, "label": label}
        for test, key in (("test_id", "id"), ("test_ood", "ood"), ("test_lam3", "lam3")):
            r[key] = agg(L, pat, test, "rmse_h200")[0]
        rows.append(r)
    kept = [r for r in rows if not np.isnan(r["id"])]
    for k, r in enumerate(kept):
        r["y"] = k
    write("ablation", kept)


def export_hybrid():
    p = RES / "metrics_hybrid.csv"
    if not p.exists():
        return None
    H = pd.read_csv(p)
    L = learned_metrics()
    C = pd.read_csv(RES / "metrics_classical.csv")
    rows = []
    for k, (test, label) in enumerate((("test_lam0", "$\\lambda{=}0$"), ("test_id", "$\\lambda{=}1$"),
                                       ("test_lam2", "$\\lambda{=}2$"), ("test_lam3", "$\\lambda{=}3$"),
                                       ("test_ood", "held-out"))):
        def hyb(sensor):
            h = H[(H.run == f"hybrid[{sensor}]") & (H.test == test)]
            return h.rmse_h200.iloc[0] if len(h) else NAN
        g = C[(C.method == "greybox") & (C.test == test)]
        rows.append({"x": k, "label": label, "grey": g.rmse_h200.iloc[0] if len(g) else NAN,
                     "hyb_tracker": hyb("tracker_window"), "hyb_jepa": hyb("jepa_long_s0"),
                     "hyb_ft": hyb("jepaft_n300_long_s0"), "jepa_wm": agg(L, r"jepa_long_s0\+probe300", test, "rmse_h200")[0]})
    write("hybrid", rows)
    return H


def export_ensemble():
    p = RES / "ensemble_spread_skill.csv"
    if p.exists():
        df = pd.read_csv(p)
        for (fam, test), g in df.groupby(["family", "test"]):
            g[["spread", "rmse"]].to_csv(DATA / f"spread_skill_{fam}_{test}.csv", index=False, float_format="%.5g")


def export_training_curves():
    rows = {}
    for key, run in (("jepa", "jepa_long_s0"), ("recon", "recon_long_s0"), ("sup", "sup_all_long_s0")):
        h = pd.read_csv(RUNS / run / "history.csv")
        rows[key] = h.set_index("step")[["val_filter_rmse", "val_roll_rmse", "eff_rank"]]
    steps = sorted(set().union(*[r.index for r in rows.values()]))
    out = {"step": steps}
    for key, h in rows.items():
        h = h.reindex(steps)
        for c in h.columns:
            out[f"{key}_{c}"] = h[c].values
    pd.DataFrame(out).to_csv(DATA / "training_curves.csv", index=False, na_rep="nan", float_format="%.5g")


def parse_log(path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.search(r"(\{\"step\".*\})", line)
        if m:
            rows.append(json.loads(m.group(1)))
    return pd.DataFrame(rows)


def export_collapse():
    naive = parse_log(LOGS / "diag_jepa.log")
    two = pd.read_csv(RUNS / "jepa_long_s0" / "history.csv")
    two = two[two.step <= 1500]
    rows = []
    for _, r in naive.iterrows():
        rows.append({"step": r.step, "naive_rank": r.eff_rank, "naive_filter": r.val_filter_rmse})
    pd.DataFrame(rows).to_csv(DATA / "collapse_naive.csv", index=False, float_format="%.5g")
    two[["step", "eff_rank", "val_filter_rmse"]].rename(columns={"eff_rank": "two_rank", "val_filter_rmse": "two_filter"}) \
        .to_csv(DATA / "collapse_twostage.csv", index=False, float_format="%.5g")


def export_examples():
    from ejepa.data import load_split
    ep = load_split("test_id", V2 / "cache")
    cl = np.load(RES / "classical_test_id.npz")
    t = np.arange(400) * 0.01
    for i in (1, 7, 12):
        rows = {"t": t, "truth": ep.omega[i]}
        fut = np.full(400, np.nan)
        fut[200:] = cl["greybox"][i]; rows["grey"] = fut.copy()
        for key, run in (("jepa", "jepa_long_s0+probe300"), ("recon", "recon_long_s0+probe300"), ("sup", "sup_all_long_s0")):
            p = RES / f"pred_{run}_test_id.npz"
            if p.exists():
                fut = np.full(400, np.nan); fut[200:] = np.load(p)["pred"][i]; rows[key] = fut
        pd.DataFrame(rows).iloc[::2].to_csv(DATA / f"example_{i}.csv", index=False, na_rep="nan", float_format="%.5g")
    # Spectrogram raster + blade-passing overlay for the problem figure.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    i = 3
    plt.imsave(DATA / "spectrogram.png", ep.obs[i].T.astype(float)[::-1], cmap="magma", vmin=-1.2, vmax=2.0)
    bpf = {"t": t[::2]}
    for h in (1, 2, 3):
        bpf[f"h{h}"] = (h * 11 * ep.omega[i] / (2 * np.pi) / 1e3)[::2]
    bpf["omega"] = ep.omega[i][::2]; bpf["u"] = ep.u[i][::2]
    pd.DataFrame(bpf).to_csv(DATA / "problem_episode.csv", index=False, float_format="%.5g")


def export_numbers(L, C, H):
    """LaTeX macros for every number quoted in the paper prose (data/numbers.tex)."""
    out = {}

    def m(pattern, test="test_id", col="rmse_h200"):
        return agg(L, pattern, test, col)[0]

    def sd(pattern, test="test_id", col="rmse_h200"):
        s = L[(L.test == test) & L.run.str.fullmatch(pattern)][col].dropna()
        return s.std() if len(s) > 1 else NAN

    def c(method, test="test_id", col="rmse_h200"):
        r = C[(C.method == method) & (C.test == test)]
        return r[col].iloc[0] if len(r) else NAN

    J, Jm, R, Rm, S, S3, F3 = (MAIN[k] for k in ("jepa", "jepamlp", "recon", "reconmlp", "sup", "sup300", "ft"))
    out.update(nJepaH=m(J), nJepaHsd=sd(J), nJepaMlpH=m(Jm), nReconH=m(R), nReconMlpH=m(Rm), nSupAllH=m(S),
               nSupThreeH=m(S3), nFtThreeH=m(F3), nGreyH=c("greybox"), nNominalH=c("nominal"),
               nGreyOOD=c("greybox", "test_ood"), nJepaOOD=m(J, "test_ood"), nSupAllOOD=m(S, "test_ood"),
               nGreyLamZero=c("greybox", "test_lam0"), nGreyLamThree=c("greybox", "test_lam3"),
               nJepaLamZero=m(J, "test_lam0"), nJepaLamThree=m(J, "test_lam3"), nReconLamThree=m(R, "test_lam3"),
               nSupAllLamThree=m(S, "test_lam3"), nSupThreeLamThree=m(S3, "test_lam3"), nFtThreeLamThree=m(F3, "test_lam3"),
               nJepaProbeTen=m(r"jepa_long_s\d\+probe10"), nSupTen=m(r"sup_n10_long_s\d"), nFtTen=m(r"jepaft_n10_long_s\d"),
               nJepaFilt=m(J, col="filter_rmse"), nJepaMlpFilt=m(Jm, col="filter_rmse"), nReconFilt=m(R, col="filter_rmse"),
               nSupAllFilt=m(S, col="filter_rmse"), nTrackerFilt=c("greybox", col="filter_rmse"),
               nGainJepa=m(J, col="sens_rel_err_final"), nGainRecon=m(R, col="sens_rel_err_final"),
               nGainSup=m(S, col="sens_rel_err_final"), nGainGrey=c("greybox", col="sens_rel_err_final"),
               nAblShort=m(r"jepa_s0\+probe300"), nAblRand=m(r"jepa_rand_s0\+probe300"), nAblCtx=m(r"jepa_ctx_s0\+probe300"),
               nAblXlong=m(r"jepa_xlong_s0\+probe300"), nAblMix=m(r"jepa_mix_long_s0\+probe300"),
               nAblMixLamThree=m(r"jepa_mix_long_s0\+probe300", "test_lam3"), nAblMixLamZero=m(r"jepa_mix_long_s0\+probe300", "test_lam0"),
               nAblHiLamThree=m(r"jepa_hi_long_s0\+probe300", "test_lam3"), nAblJepaSzLamThree=m(r"jepa_long_s0\+probe300", "test_lam3"),
               nAblReconMixLamThree=m(r"recon_mix_long_s0\+probe300", "test_lam3"))
    # Seed-0 counterparts for like-for-like comparison with single-seed ablations and round-2 runs.
    s0, mix, hi = r"jepa_long_s0\+probe300", r"jepa_mix_long_s0\+mlp300", r"jepa_hi_long_s0\+mlp300"
    out.update(nJepaSzH=m(s0), nJepaSzLamZero=m(s0, "test_lam0"), nJepaSzOOD=m(s0, "test_ood"),
               nJepaSzMlpFiltLamThree=m(r"jepa_long_s0\+mlp300", "test_lam3", "filter_rmse"),
               nMixMlpFiltLamThree=m(mix, "test_lam3", "filter_rmse"), nHiMlpFiltLamThree=m(hi, "test_lam3", "filter_rmse"),
               nMixMlpH=m(mix), nJepaSzMlpH=m(r"jepa_long_s0\+mlp300"), nMixMlpLamThree=m(mix, "test_lam3"),
               nJepaSzMlpLamThree=m(r"jepa_long_s0\+mlp300", "test_lam3"),
               nXlongOOD=m(r"jepa_xlong_s0\+probe300", "test_ood"), nXlongLamThree=m(r"jepa_xlong_s0\+probe300", "test_lam3"),
               nJepaSzH_sd_note=0)
    out.pop("nJepaSzH_sd_note")
    out["nLabelRatioTen"] = out["nSupTen"] / out["nJepaProbeTen"]
    out["nSeeds"] = len(L[(L.test == "test_id") & L.run.str.fullmatch(J)])
    probes = pd.read_csv(DATA / "param_probes.csv").set_index("param")
    out.update(nRtwoKwJepa=probes.loc["kw", "jepa_m"], nRtwoTauJepa=probes.loc["tau", "jepa_m"],
               nRtwoTauSup=probes.loc["tau", "sup_m"], nRtwoJJepa=probes.loc["J", "jepa_m"])
    for test, key in (("id", "Id"), ("lam3", "LamThree")):
        p = DATA / f"mpc_{test}.csv"
        if p.exists():
            d = pd.read_csv(p).set_index("color")
            for col, name in (("ft", "Ft"), ("sup300", "SupThree"), ("grey", "Grey"), ("jepa", "Jepa"),
                              ("recon", "Recon"), ("sup", "SupAll"), ("pi", "Pi"), ("jepahi", "JepaHi")):
                out[f"nMpc{name}{key}"] = d.loc[col, "excess"] if col in d.index else NAN
            out[f"nMpcRatio{key}"] = out[f"nMpcSupThree{key}"] / out[f"nMpcFt{key}"]
    for sensor, name in (("jepa_long_s0", "Jepa"), ("jepaft_n300_long_s0", "Ft"), ("tracker_window", "Tracker"),
                         ("sup_n300_long_s0", "SupThree")):
        for test, key in (("test_id", "Id"), ("test_ood", "OOD"), ("test_lam3", "LamThree"), ("test_lam2", "LamTwo"),
                          ("test_lam0", "LamZero")):
            h = H[(H.run == f"hybrid[{sensor}]") & (H.test == test)] if H is not None else []
            out[f"nHyb{name}{key}"] = h.rmse_h200.iloc[0] if len(h) else NAN
    e = RES / "metrics_ensemble.csv"
    E = pd.read_csv(e) if e.exists() else None
    for fam, name in (("jepa_long", "Jepa"), ("sup_all_long", "Sup"), ("recon_long", "Recon")):
        for test, key in (("test_id", "Id"), ("test_lam3", "LamThree"), ("test_ood", "OOD")):
            r = E[(E.family == fam) & (E.test == test)] if E is not None else []
            ok = len(r) > 0
            out[f"nEns{name}{key}"] = r.ensemble_rmse_h200.iloc[0] if ok else NAN
            out[f"nEnsSingle{name}{key}"] = r.single_rmse_mean.iloc[0] if ok else NAN
            out[f"nEnsCov{name}{key}"] = 100 * r.coverage_2sigma.iloc[0] if ok else NAN
            out[f"nEnsCorr{name}{key}"] = r.spread_error_corr.iloc[0] if ok else NAN
    lines = ["% Auto-generated by export_plot_data.py -- do not edit."]
    for k, v in out.items():
        if isinstance(v, (int, np.integer)):
            txt = str(int(v))
        elif v is None or (isinstance(v, float) and np.isnan(v)):
            txt = "\\textbf{n/a}"
        else:
            txt = f"{v:.2f}" if abs(v) < 1 else (f"{v:.1f}" if abs(v) < 100 else f"{v:.0f}")
        lines.append(f"\\newcommand{{\\{k}}}{{{txt}}}")
    (DATA / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    L = learned_metrics()
    C = pd.read_csv(RES / "metrics_classical.csv")
    H = export_hybrid()
    for fn in (lambda: export_horizon(L, C), lambda: export_labels(L), lambda: export_nuisance(L, C, H),
               lambda: export_step(L), export_probes, export_mpc, lambda: export_ablation(L), export_ensemble,
               export_training_curves, export_collapse, export_examples):
        try:
            fn()
        except Exception as exc:
            print("export failed:", getattr(fn, "__name__", "lambda"), repr(exc))
    if H is not None:
        H.to_csv(DATA / "hybrid_raw.csv", index=False, float_format="%.5g")
    try:
        export_numbers(L, C, H)
    except Exception as exc:
        print("export failed: numbers", repr(exc))
    print("exported:", sorted(p.name for p in DATA.glob("*")))
