"""Euler-JEPA v2 experiment driver.

    python run_v2.py classical          # tracker + grey-box + nominal + oracle baselines
    python run_v2.py train [--only PAT]  # every learned model in RUNS (skips finished runs)
    python run_v2.py eval  [--only PAT]  # metrics for every finished run
    python run_v2.py mpc                 # closed-loop tracking comparison
    python run_v2.py all

Each stage writes CSV/JSON under results/ and logs under logs/; finished runs are not retrained.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ejepa import classical as C  # noqa: E402
from ejepa.data import load_split  # noqa: E402
from ejepa.evaluate import (HORIZON, T0, bootstrap_ci, filtered_speed, forecast, mlp_speed_probe,  # noqa: E402
                            ode_residual, param_probe, rmse, speed_probe, step_response, true_step_response)
from ejepa.physics import nominal_params, simulate  # noqa: E402
from ejepa.train import DEVICE, TrainConfig, load_model, train  # noqa: E402

CACHE, RUNS_DIR, RESULTS, LOGS = HERE / "cache", HERE / "runs", HERE / "results", HERE / "logs"
SEEDS = (0, 1, 2)
LABEL_BUDGETS = (10, 30, 100, 300)
TEST_SETS = ("test_id", "test_ood", "test_lam0", "test_lam0.5", "test_lam1", "test_lam2", "test_lam3")
HORIZONS = (10, 25, 50, 100, 200)
STEPS = 6000


# Long-horizon variant: 1.28 s rollouts on 3.2 s crops; 8 anchors per episode keeps memory equal.
LONG = dict(rollout_k=128, crop=320, steps=8000, anchors=8)


SWEEP = dict(LONG, steps=2000)


def run_specs():
    """Ordered queue.

    Main configuration (all seeds): the long-horizon setting for JEPA, reconstruction and
    supervised alike.  Seed 0 additionally keeps the development runs (short horizon, random
    trunk, adaptive context) as ablations, the full label sweep and the high-nuisance runs.
    """
    head, sweep, extra = [], [], []
    for s in SEEDS:
        head.append(dict(name=f"jepa1_s{s}", method="jepa1", seed=s, train="train", val="val", steps=3000))
        head.append(dict(name=f"jepa_long_s{s}", method="jepa", seed=s, train="train", val="val",
                         init_from=f"jepa1_s{s}/encoder.pt", **LONG))
        head.append(dict(name=f"recon_long_s{s}", method="recon", seed=s, train="train", val="val", **LONG))
        head.append(dict(name=f"sup_all_long_s{s}", method="sup", seed=s, train="train", val="val",
                         labelled=3000, **LONG))
        for n in (LABEL_BUDGETS if s == 0 else (10, 100)):
            sweep.append(dict(name=f"sup_n{n}_long_s{s}", method="sup", seed=s, train="train", val="val",
                              labelled=n, **SWEEP))
            sweep.append(dict(name=f"jepaft_n{n}_long_s{s}", method="jepa_ft", seed=s, train="train", val="val",
                              labelled=n, lr=3e-4, init_from=f"jepa_long_s{s}/model.pt", **SWEEP))
    # Seed-0 development runs kept as ablations (short horizon = K 64 on 2 s crops).
    extra += [dict(name="jepa_s0", method="jepa", seed=0, train="train", val="val", init_from="jepa1_s0/encoder.pt"),
              dict(name="recon_s0", method="recon", seed=0, train="train", val="val"),
              dict(name="sup_all_s0", method="sup", seed=0, train="train", val="val", labelled=3000),
              dict(name="jepa_rand_s0", method="jepa", seed=0, train="train", val="val"),
              dict(name="jepa_ctx_s0", method="jepa", seed=0, train="train", val="val",
                   init_from="jepa1_s0/encoder.pt", ctx_dim=16),
              dict(name="jepa1_hi_s0", method="jepa1", seed=0, train="train_hi", val="val_hi", steps=3000),
              dict(name="jepa_hi_long_s0", method="jepa", seed=0, train="train_hi", val="val_hi",
                   init_from="jepa1_hi_s0/encoder.pt", **LONG),
              dict(name="recon_hi_long_s0", method="recon", seed=0, train="train_hi", val="val_hi", **LONG)]
    # Round-2 improvements (seed 0): full-horizon training and nuisance-randomised pre-training.
    improve = [dict(name="jepa_xlong_s0", method="jepa", seed=0, train="train", val="val",
                    init_from="jepa1_s0/encoder.pt", rollout_k=200, crop=400, steps=8000, anchors=4),
               dict(name="jepa1_mix_s0", method="jepa1", seed=0, train="train_mix", val="val_mix", steps=3000),
               dict(name="jepa_mix_long_s0", method="jepa", seed=0, train="train_mix", val="val_mix",
                    init_from="jepa1_mix_s0/encoder.pt", **LONG),
               dict(name="recon_mix_long_s0", method="recon", seed=0, train="train_mix", val="val_mix", **LONG)]
    seed0 = [x for x in head + sweep if x["seed"] == 0]
    rest_head = [x for x in head if x["seed"] != 0]
    rest_sweep = [x for x in sweep if x["seed"] != 0]
    return seed0 + extra + rest_head + improve + rest_sweep


def log_to(name):
    LOGS.mkdir(exist_ok=True)
    fh = (LOGS / f"{name}.log").open("a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); fh.write(line + "\n"); fh.flush()
    return log


def provenance():
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    except OSError:
        rev = ""
    return {"git_head": rev, "python": platform.python_version(), "torch": torch.__version__,
            "device": str(DEVICE), "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "agent": "Claude Code (claude-opus-5-5)"}


# ----------------------------------------------------------------------------- classical
def stage_classical(log):
    RESULTS.mkdir(exist_ok=True)
    val = load_split("val", CACHE)
    best = None
    for kappa in (0.02, 0.08, 0.2):
        for sharp in (1.0, 2.0, 4.0):
            err = rmse(C.track(val.obs[:100], kappa=kappa, sharp=sharp), val.omega[:100])
            log(f"tracker kappa={kappa} sharp={sharp} val rmse={err:.3f}")
            if best is None or err < best[0]:
                best = (err, kappa, sharp)
    _, kappa, sharp = best
    (RESULTS / "tracker_config.json").write_text(json.dumps({"kappa": kappa, "sharp": sharp, "val_rmse": best[0]}))
    rows = []
    for name in TEST_SETS:
        ep = load_split(name, CACHE)
        u, up0 = ep.u.astype(np.float64), ep.u_prev0.astype(np.float64)
        t = time.time()
        w_ctx = C.track(ep.obs[:, :T0 + 1], kappa=kappa, sharp=sharp)          # context only
        w_all = C.track(ep.obs, kappa=kappa, sharp=sharp, causal=True)        # causal filter, full episode
        fit = C.fit_greybox(w_ctx, u, up0, fit_steps=T0)
        preds = {"greybox": C.greybox_forecast(fit, u, up0, T0, HORIZON)}
        # Nominal physics: catalogue parameters, initialised from the tracked context.
        nom = nominal_params(ep.n)
        om, _ = simulate(nom, u, w_ctx[:, 0], up0, up0)
        preds["nominal"] = om[:, T0 + 1:T0 + 1 + HORIZON]
        # Oracle: true parameters and true initial state (verifies the protocol, ~0 error).
        om, _ = simulate(ep.params, u, ep.omega[:, 0], ep.act[:, 0], ep.u_prev0)
        preds["oracle"] = om[:, T0 + 1:T0 + 1 + HORIZON]
        preds["persistence"] = np.repeat(w_ctx[:, -1:], HORIZON, 1)
        truth = ep.omega[:, T0 + 1:T0 + 1 + HORIZON]
        # Step response of the grey-box model by central finite difference.
        resp = []
        for sgn in (1, -1):
            uu = u.copy(); uu[:, T0:T0 + HORIZON] += sgn * 1e-3
            resp.append(C.greybox_forecast(fit, u, up0, T0, HORIZON, u_override=uu))
        sens = (resp[0] - resp[1]) / 2e-3
        true_sens = true_step_response(ep)
        np.savez_compressed(RESULTS / f"classical_{name}.npz", **preds, tracked_causal=w_all,
                            sens_greybox=sens, sens_true=true_sens, delay=fit["delay"], theta=fit["theta"])
        for method, pred in preds.items():
            rows.append(metric_row(method, "classical", name, ep, pred, truth,
                                   filt=w_all if method == "greybox" else None,
                                   sens=sens if method == "greybox" else None, true_sens=true_sens))
        log(f"classical {name}: " + ", ".join(f"{r['method']}={r['rmse_h200']:.2f}" for r in rows[-4:])
            + f" ({time.time() - t:.0f}s)")
    write_rows(RESULTS / "metrics_classical.csv", rows)


def metric_row(method, run, test, ep, pred, truth, filt=None, sens=None, true_sens=None):
    per_ep = np.mean((pred - truth) ** 2, axis=1)
    lo, hi = bootstrap_ci(per_ep)
    row = {"method": method, "run": run, "test": test, "rmse_h200": float(np.sqrt(per_ep.mean())),
           "rmse_h200_ci_lo": lo, "rmse_h200_ci_hi": hi,
           "median_ep_rmse": float(np.median(np.sqrt(per_ep)))}
    for h in HORIZONS:
        row[f"rmse_h{h}"] = float(rmse(pred[:, :h], truth[:, :h]))
    row["ode_residual_Nm"] = float(np.mean(ode_residual(pred, ep)))
    if filt is not None:
        row["filter_rmse"] = float(rmse(filt[:, 20:], ep.omega[:, 20:]))
    if sens is not None:
        row["sens_rmse"] = float(rmse(sens, true_sens))
        row["sens_rel_err_final"] = float(np.median(np.abs(sens[:, -1] - true_sens[:, -1]) / (np.abs(true_sens[:, -1]) + 1e-9)))
    return row


def write_rows(path, rows):
    import csv
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


# ----------------------------------------------------------------------------- training
def selected(specs, only):
    return [s for s in specs if not only or any(fnmatch.fnmatch(s["name"], p) for p in only.split(","))]


def stage_train(log, only=None):
    cache = {}
    for spec in selected(run_specs(), only):
        out = RUNS_DIR / spec["name"]
        if (out / "config.json").exists():               # written last by every training stage
            continue
        for split in (spec["train"], spec["val"]):
            if split not in cache:
                cache[split] = load_split(split, CACHE)
        cfg = TrainConfig(method=spec["method"], seed=spec["seed"], steps=spec.get("steps", STEPS),
                          lr=spec.get("lr", 1e-3), labelled=spec.get("labelled"), tag=spec["name"],
                          rollout_k=spec.get("rollout_k", 64), crop=spec.get("crop", 200),
                          anchors=spec.get("anchors", 24), ctx_dim=spec.get("ctx_dim", 0),
                          init_from=str(RUNS_DIR / spec["init_from"]) if spec.get("init_from") else None)
        run_log = log_to(spec["name"])
        run_log(f"training {spec} on {DEVICE}")
        t = time.time()
        train(cfg, cache[spec["train"]], cache[spec["val"]], out, log=run_log)
        (out / "provenance.json").write_text(json.dumps({**provenance(), "spec": spec,
                                                         "wall_s": time.time() - t}, indent=2))
        log(f"done {spec['name']} in {time.time() - t:.0f}s")


# ----------------------------------------------------------------------------- evaluation
def stage_eval(log, only=None):
    RESULTS.mkdir(exist_ok=True)
    train_ep = load_split("train", CACHE)
    tests = {name: load_split(name, CACHE) for name in TEST_SETS}
    true_sens = {name: true_step_response(ep) for name, ep in tests.items()}
    rows, probe_rows = [], []
    for spec in selected(run_specs(), only):
        run_dir = RUNS_DIR / spec["name"]
        if not (run_dir / "model.pt").exists():
            continue
        model = load_model(run_dir)
        val = load_split(spec["val"], CACHE)
        # Self-supervised models are decoded with a ridge probe fitted on N labelled training
        # episodes; for the main runs N = 300 (also evaluated at every label budget below).
        budgets = (300,) if model.method in ("sup", "jepa_ft") else LABEL_BUDGETS + (3000,)
        for n_lab in budgets:
            probe = None
            if model.method in ("jepa", "recon"):
                probe = speed_probe(model, train_ep.subset(np.arange(n_lab)), val)
            label = spec["name"] if model.method in ("sup", "jepa_ft") else f"{spec['name']}+probe{n_lab}"
            for name, ep in tests.items():
                if model.method in ("sup", "jepa_ft") or n_lab == 300 or name == "test_id":
                    pred = forecast(model, ep, probe)
                    truth = ep.omega[:, T0 + 1:T0 + 1 + HORIZON]
                    sens = (step_response(model, ep, probe) if name in ("test_id", "test_ood") and n_lab in (300,)
                            else None)
                    row = metric_row(spec["method"], label, name, ep, pred, truth,
                                     filt=filtered_speed(model, ep, probe), sens=sens, true_sens=true_sens[name])
                    row.update({"seed": spec["seed"], "labelled": spec.get("labelled", n_lab),
                                "train_split": spec["train"], "spec": spec["name"]})
                    rows.append(row)
                    if name == "test_id" and n_lab in (300, 3000) or model.method in ("sup", "jepa_ft") and name == "test_id":
                        np.savez_compressed(RESULTS / f"pred_{label}_{name}.npz", pred=pred, sens=sens if sens is not None else 0)
            log(f"eval {label}: " + ", ".join(f"{r['test']}={r['rmse_h200']:.2f}" for r in rows if r["run"] == label))
        # Non-linear (MLP) probe with 300 labels: how much speed information is in z beyond linear access?
        if model.method in ("jepa", "recon"):
            mprobe = mlp_speed_probe(model, train_ep.subset(np.arange(300)), val)
            label = f"{spec['name']}+mlp300"
            for name in ("test_id", "test_ood", "test_lam3"):
                ep = tests[name]
                pred = forecast(model, ep, mprobe)
                row = metric_row(spec["method"], label, name, ep, pred, ep.omega[:, T0 + 1:T0 + 1 + HORIZON],
                                 filt=filtered_speed(model, ep, mprobe))
                row.update({"seed": spec["seed"], "labelled": 300, "train_split": spec["train"], "spec": spec["name"]})
                rows.append(row)
            log(f"eval {label}: " + ", ".join(f"{r['test']}={r['rmse_h200']:.2f}/filt {r['filter_rmse']:.2f}"
                                              for r in rows if r["run"] == label))
        # Hidden-parameter probes (300 labelled episodes) for the main models.
        if not spec["name"].startswith(("sup_n", "jepaft_")):
            r2 = param_probe(model, train_ep.subset(np.arange(300)), tests["test_id"])
            probe_rows.append({"run": spec["name"], "method": spec["method"], "seed": spec["seed"], **r2})
            log(f"param probe {spec['name']}: " + json.dumps({k: round(v, 3) for k, v in r2.items()}))
    # Random-initialisation control: an untrained encoder with the same architecture.
    from ejepa.models import WorldModel
    for s in SEEDS:
        torch.manual_seed(s)
        rand = WorldModel("jepa").to(DEVICE).eval()
        r2 = param_probe(rand, train_ep.subset(np.arange(300)), tests["test_id"])
        probe_rows.append({"run": f"random_s{s}", "method": "random", "seed": s, **r2})
        probe = speed_probe(rand, train_ep.subset(np.arange(300)), load_split("val", CACHE))
        filt = filtered_speed(rand, tests["test_id"], probe)
        rows.append({"method": "random", "run": f"random_s{s}+probe300", "test": "test_id", "seed": s,
                     "filter_rmse": float(rmse(filt[:, 20:], tests["test_id"].omega[:, 20:])), "labelled": 300})
    suffix = "" if not only else "_" + only.replace("*", "x").replace(",", "_")
    write_rows(RESULTS / f"metrics_learned{suffix}.csv", rows)
    write_rows(RESULTS / f"param_probes{suffix}.csv", probe_rows)
    (RESULTS / "provenance_eval.json").write_text(json.dumps(provenance(), indent=2))


# ----------------------------------------------------------------------------- hybrid
HYBRID_START = 30          # learned beliefs need a short warm-up; all hybrid fits start here


def stage_hybrid(log, tests=("test_id", "test_ood", "test_lam0", "test_lam2", "test_lam3")):
    """Learned perception + physics dynamics.

    A causal speed estimate from a learned belief replaces the comb tracker as the input to
    grey-box identification of the true ODE form (fit window [HYBRID_START, T0]); the fitted
    model then forecasts.  Control: the comb tracker fitted over the identical window."""
    train_ep, val = load_split("train", CACHE), load_split("val", CACHE)
    sensors = {}
    for name in ("jepa_long_s0", "jepaft_n300_long_s0", "sup_n300_long_s0"):
        if (RUNS_DIR / name / "model.pt").exists():
            model = load_model(RUNS_DIR / name)
            probe = mlp_speed_probe(model, train_ep.subset(np.arange(300)), val) if model.method == "jepa" else None
            sensors[name] = (model, probe)
    kappa_sharp = json.loads((RESULTS / "tracker_config.json").read_text())
    rows = []
    for test in tests:
        ep = load_split(test, CACHE)
        u, up0 = ep.u.astype(np.float64), ep.u_prev0.astype(np.float64)
        truth = ep.omega[:, T0 + 1:T0 + 1 + HORIZON]
        true_sens = true_step_response(ep)
        estimates = {"tracker_window": C.track(ep.obs[:, :T0 + 1], kappa=kappa_sharp["kappa"], sharp=kappa_sharp["sharp"])}
        for name, (model, probe) in sensors.items():
            estimates[name] = filtered_speed(model, ep, probe)[:, :T0 + 1].astype(np.float64)
        for name, w_hat in estimates.items():
            t = time.time()
            fit = C.fit_greybox(w_hat, u, up0, fit_steps=T0, iters=300, t_start=HYBRID_START)
            pred = C.greybox_forecast(fit, u, up0, T0, HORIZON)
            resp = []
            for sgn in (1, -1):
                uu = u.copy(); uu[:, T0:T0 + HORIZON] += sgn * 1e-3
                resp.append(C.greybox_forecast(fit, u, up0, T0, HORIZON, u_override=uu))
            sens = (resp[0] - resp[1]) / 2e-3
            label = f"hybrid[{name}]"
            row = metric_row("hybrid", label, test, ep, pred, truth, sens=sens, true_sens=true_sens)
            row["sensor_rmse_context"] = float(rmse(w_hat[:, HYBRID_START:], ep.omega[:, HYBRID_START:T0 + 1]))
            rows.append(row)
            log(f"hybrid {test} {name}: sensor {row['sensor_rmse_context']:.2f} -> 2 s forecast {row['rmse_h200']:.2f} "
                f"({time.time() - t:.0f}s)")
            write_rows(RESULTS / "metrics_hybrid.csv", rows)


# ----------------------------------------------------------------------------- ensemble UQ
def stage_ensemble(log, families=("jepa_long", "recon_long", "sup_all_long"),
                   tests=("test_id", "test_ood", "test_lam3")):
    """Seed ensembles: ensemble-mean accuracy and whether member spread predicts error."""
    train_ep, val = load_split("train", CACHE), load_split("val", CACHE)
    rows, calib = [], []
    for fam in families:
        members = [RUNS_DIR / f"{fam}_s{s}" for s in SEEDS if (RUNS_DIR / f"{fam}_s{s}" / "model.pt").exists()]
        if len(members) < 2:
            continue
        models = []
        for m in members:
            model = load_model(m)
            probe = speed_probe(model, train_ep.subset(np.arange(300)), val) if model.method in ("jepa", "recon") else None
            models.append((model, probe))
        for test in tests:
            ep = load_split(test, CACHE)
            truth = ep.omega[:, T0 + 1:T0 + 1 + HORIZON]
            preds = np.stack([forecast(mo, ep, pr) for mo, pr in models])          # (M,N,H)
            mean, sd = preds.mean(0), preds.std(0, ddof=1)
            err = np.abs(mean - truth)
            single = [float(rmse(p, truth)) for p in preds]
            row = {"family": fam, "test": test, "members": len(models),
                   "ensemble_rmse_h200": float(rmse(mean, truth)), "single_rmse_mean": float(np.mean(single)),
                   "spread_mean": float(sd.mean()),
                   "coverage_2sigma": float(np.mean(err <= 2 * sd)),
                   "spread_error_corr": float(np.corrcoef(sd.ravel(), err.ravel())[0, 1])}
            rows.append(row)
            # Binned spread-skill: predicted spread vs realised RMSE (per episode-time point).
            edges = np.quantile(sd.ravel(), np.linspace(0, 1, 11))
            idx = np.clip(np.digitize(sd.ravel(), edges[1:-1]), 0, 9)
            for b in range(10):
                sel = idx == b
                calib.append({"family": fam, "test": test, "bin": b, "spread": float(sd.ravel()[sel].mean()),
                              "rmse": float(np.sqrt(np.mean(err.ravel()[sel] ** 2)))})
            log(f"ensemble {fam} {test}: " + json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}))
    write_rows(RESULTS / "metrics_ensemble.csv", rows)
    write_rows(RESULTS / "ensemble_spread_skill.csv", calib)


# ----------------------------------------------------------------------------- MPC
def paired_excess(mse_a, mse_ref, n_boot: int = 4000, seed: int = 0):
    """Mean per-episode excess MSE over a reference controller, with a paired bootstrap 95 % CI."""
    d = np.asarray(mse_a) - np.asarray(mse_ref)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def stage_mpc(log, n_episodes: int = 40):
    for test, val_name in (("test_id", "val"), ("test_lam3", "val_hi")):
        stage_mpc_one(log, test, val_name, n_episodes)


def stage_mpc_one(log, test, val_name, n_episodes):
    from ejepa import mpc
    ep = load_split(test, CACHE).subset(np.arange(n_episodes))
    ref = mpc.setpoints(ep, np.random.default_rng(7))
    train_ep, val = load_split("train", CACHE), load_split("val", CACHE)
    # PI gains tuned on validation episodes (grid search) so the model-free baseline is fair;
    # for the lambda = 3 test they are tuned on the high-nuisance validation split.
    val_ep = load_split(val_name, CACHE).subset(np.arange(12))
    val_ref = mpc.setpoints(val_ep, np.random.default_rng(3))
    best = None
    for kp in (1e-3, 2e-3, 4e-3):
        for ki in (2e-3, 5e-3, 1e-2, 2e-2):
            om, _ = mpc.run_closed_loop(None, val_ep, val_ref, seed=5, kind="pi", pi_gains=(kp, ki))
            err = rmse(om, val_ref)
            if best is None or err < best[0]:
                best = (err, (kp, ki))
    pi_gains = best[1]
    log(f"PI gains tuned on validation: Kp={pi_gains[0]}, Ki={pi_gains[1]} (val rmse {best[0]:.2f})")
    controllers = {"oracle": (mpc.OracleController(), "mpc"), "greybox": (mpc.GreyboxController(), "mpc"),
                   "pi": (None, "pi")}
    for name in ("jepa_s0", "jepa_long_s0", "recon_s0", "recon_long_s0", "sup_all_s0", "sup_all_long_s0",
                 "sup_n300_long_s0", "jepaft_n300_long_s0", "jepa_hi_long_s0", "recon_hi_long_s0"):
        if (RUNS_DIR / name / "model.pt").exists():
            model = load_model(RUNS_DIR / name)
            # Self-supervised models plan through the 300-label MLP probe (better than linear on validation).
            probe = (mlp_speed_probe(model, train_ep.subset(np.arange(300)), val)
                     if model.method in ("jepa", "recon") else None)
            controllers[name] = (mpc.WorldModelController(model, probe), "mpc")
    rows, traces, mse_ref = [], {"ref": ref}, None
    seg = ref.shape[1] // mpc.SEGMENTS
    steady = slice(int(0.6 * seg), seg)                         # last 40 % of each segment
    for name, (ctrl, kind) in controllers.items():
        t = time.time()
        omega, u = mpc.run_closed_loop(ctrl, ep, ref, seed=11, kind=kind, pi_gains=pi_gains)
        err = omega - ref
        mse_ep = np.mean(err ** 2, 1)
        steady_ep = np.mean(err.reshape(n_episodes, mpc.SEGMENTS, -1)[:, :, steady] ** 2, axis=(1, 2))
        if name == "oracle":
            mse_ref, steady_ref = mse_ep, steady_ep
        lo, hi = bootstrap_ci(mse_ep)
        ex, ex_lo, ex_hi = paired_excess(mse_ep, mse_ref)
        sx, sx_lo, sx_hi = paired_excess(steady_ep, steady_ref)
        rows.append({"test": test, "controller": name, "tracking_rmse": float(np.sqrt(mse_ep.mean())),
                     "tracking_rmse_ci_lo": lo, "tracking_rmse_ci_hi": hi,
                     "steady_rmse": float(np.sqrt(steady_ep.mean())),
                     "excess_mse_vs_oracle": ex, "excess_mse_ci_lo": ex_lo, "excess_mse_ci_hi": ex_hi,
                     "excess_steady_mse_vs_oracle": sx, "excess_steady_ci_lo": sx_lo, "excess_steady_ci_hi": sx_hi,
                     "control_tv": float(np.mean(np.abs(np.diff(u, axis=1)).sum(1))),
                     "wall_s": time.time() - t})
        traces[f"omega_{name}"], traces[f"u_{name}"] = omega, u
        log(f"mpc[{test}] {name}: track {rows[-1]['tracking_rmse']:.2f} steady {rows[-1]['steady_rmse']:.2f} "
            f"excess {ex:.1f} [{ex_lo:.1f}, {ex_hi:.1f}] rad^2/s^2")
    suffix = "" if test == "test_id" else f"_{test}"
    write_rows(RESULTS / f"mpc_tracking{suffix}.csv", rows)
    np.savez_compressed(RESULTS / f"mpc_traces{suffix}.npz", **traces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("classical", "train", "eval", "mpc", "hybrid", "ensemble", "all"))
    ap.add_argument("--only", default=None, help="comma-separated fnmatch patterns over run names")
    args = ap.parse_args()
    log = log_to(f"stage_{args.stage}")
    log(json.dumps(provenance()))
    if args.stage in ("classical", "all"):
        stage_classical(log)
    if args.stage in ("train", "all"):
        stage_train(log, args.only)
    if args.stage in ("eval", "all"):
        stage_eval(log, args.only)
    if args.stage in ("hybrid", "all"):
        stage_hybrid(log)
    if args.stage in ("ensemble", "all"):
        stage_ensemble(log)
    if args.stage in ("mpc", "all"):
        stage_mpc(log)


if __name__ == "__main__":
    main()
