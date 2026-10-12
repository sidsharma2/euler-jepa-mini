"""Evaluation: frozen linear probes, open-loop latent rollouts, control sensitivity, ODE residuals.

Protocol (fixed for every method):
  context   frames 0..T0 (T0 = 199, i.e. 2.0 s) and the commands issued before them
  forecast  omega at 200..399 (2.0 s) given the *future* commands (open loop, no new frames)
  decoding  jepa / recon: a ridge probe z -> omega fitted on labelled training episodes only;
            sup / jepa_ft: their trained speed head.
"""
from __future__ import annotations

import numpy as np
import torch

from .data import Episodes, T_CONTEXT, denorm_omega, norm_omega
from .physics import PARAM_NAMES, euler_torque, load_torque, simulate
from .train import DEVICE

T0 = T_CONTEXT - 1
HORIZON = 200


@torch.no_grad()
def encode(model, ep: Episodes, upto: int | None = None, batch: int = 50) -> np.ndarray:
    u_prev = ep.u_prev()
    t = ep.obs.shape[1] if upto is None else upto
    out = []
    for s in range(0, ep.n, batch):
        obs = torch.from_numpy(ep.obs[s:s + batch, :t].astype(np.float32)).to(DEVICE)
        up = torch.from_numpy(u_prev[s:s + batch, :t]).to(DEVICE)
        z, _ = model.encoder(obs, up)
        out.append(z.cpu().numpy())
    return np.concatenate(out)


def fit_ridge(z, y, lam):
    """Closed-form ridge with intercept. z (M,D), y (M,) or (M,k)."""
    mu, ymu = z.mean(0), y.mean(0)
    zc, yc = z - mu, y - ymu
    w = np.linalg.solve(zc.T @ zc + lam * len(z) * np.eye(z.shape[1]), zc.T @ yc)
    return {"w": w, "b": ymu - mu @ w}


def apply_ridge(probe, z):
    return z @ probe["w"] + probe["b"]


LAMBDAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


def speed_probe(model, labelled: Episodes, val: Episodes, t_min: int = 20):
    """Fit z -> omega on labelled episodes, choose lambda on validation filtering error."""
    zl, zv = encode(model, labelled), encode(model, val)
    xl = zl[:, t_min:].reshape(-1, zl.shape[-1]); yl = norm_omega(labelled.omega[:, t_min:]).reshape(-1)
    xv = zv[:, t_min:].reshape(-1, zv.shape[-1]); yv = norm_omega(val.omega[:, t_min:]).reshape(-1)
    best = None
    for lam in LAMBDAS:
        probe = fit_ridge(xl, yl, lam)
        err = np.sqrt(np.mean((apply_ridge(probe, xv) - yv) ** 2))
        if best is None or err < best[0]:
            best = (err, lam, probe)
    probe = best[2]; probe["lam"] = best[1]
    return probe


def mlp_speed_probe(model, labelled: Episodes, val: Episodes, t_min: int = 20, epochs: int = 30, seed: int = 0):
    """Non-linear probe: 2-layer MLP z -> omega on frozen latents; epoch chosen on validation."""
    torch.manual_seed(seed)
    zl = torch.from_numpy(encode(model, labelled)[:, t_min:].reshape(-1, model.encoder.latent_dim)).to(DEVICE)
    yl = torch.from_numpy(norm_omega(labelled.omega[:, t_min:]).reshape(-1).astype(np.float32)).to(DEVICE)
    zv = torch.from_numpy(encode(model, val)[:, t_min:].reshape(-1, model.encoder.latent_dim)).to(DEVICE)
    yv = torch.from_numpy(norm_omega(val.omega[:, t_min:]).reshape(-1).astype(np.float32)).to(DEVICE)
    net = torch.nn.Sequential(torch.nn.Linear(zl.shape[1], 128), torch.nn.GELU(),
                              torch.nn.Linear(128, 128), torch.nn.GELU(), torch.nn.Linear(128, 1)).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    best = (float("inf"), None)
    for _ in range(epochs):
        perm = torch.randperm(len(zl), device=DEVICE)
        for j in range(0, len(zl), 2048):
            b = perm[j:j + 2048]
            loss = ((net(zl[b]).squeeze(-1) - yl[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            err = float(((net(zv).squeeze(-1) - yv) ** 2).mean().sqrt())
        if err < best[0]:
            best = (err, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return {"mlp": net}


def decoder_fn(model, probe=None):
    if model.method in ("sup", "jepa_ft"):
        return lambda z: model.speed_head(z).squeeze(-1)
    if "mlp" in probe:
        return lambda z: probe["mlp"](z).squeeze(-1)
    w = torch.tensor(probe["w"], dtype=torch.float32, device=DEVICE)
    b = torch.tensor(probe["b"], dtype=torch.float32, device=DEVICE)
    return lambda z: z @ w + b


@torch.no_grad()
def filtered_speed(model, ep: Episodes, probe=None) -> np.ndarray:
    """Causal speed estimate omega_hat_k = decode(z_k) for every frame (state estimation)."""
    dec = decoder_fn(model, probe)
    z = torch.from_numpy(encode(model, ep)).to(DEVICE)
    return denorm_omega(dec(z).cpu().numpy())


def context_start(model, ep: Episodes, t0: int = T0, t_min: int = 16):
    """Belief at t0 and (adaptive models) the machine embedding from context transitions only."""
    zc = encode(model, ep, upto=t0 + 1)
    c = None
    if getattr(model, "ctx_dim", 0):
        with torch.no_grad():
            c = model.context(torch.from_numpy(zc[:, t_min:]).to(DEVICE),
                              torch.from_numpy(ep.u[:, t_min:t0 + 1].astype(np.float32)).to(DEVICE))
    return torch.from_numpy(zc[:, t0]).to(DEVICE), c


def _set_ctx(model, c, s, batch):
    if c is not None:
        model.predictor.set_context(c[s:s + batch])


def forecast(model, ep: Episodes, probe=None, t0: int = T0, horizon: int = HORIZON,
             u_override: np.ndarray | None = None, batch: int = 100, grad: bool = False):
    """Open-loop rollout from the belief at t0 under commands u[t0 : t0+horizon]."""
    dec = decoder_fn(model, probe)
    u = ep.u if u_override is None else u_override
    z0, c = context_start(model, ep, t0)
    outs = []
    for s in range(0, ep.n, batch):
        useq = torch.from_numpy(u[s:s + batch, t0:t0 + horizon].astype(np.float32)).to(DEVICE)
        _set_ctx(model, c, s, batch)
        with torch.set_grad_enabled(grad):
            outs.append(denorm_omega(dec(model.predictor.rollout(z0[s:s + batch], useq)).cpu().numpy()))
    return np.concatenate(outs)


@torch.no_grad()
def step_response(model, ep: Episodes, probe=None, t0: int = T0, horizon: int = HORIZON, batch: int = 100,
                  delta: float = 5e-3):
    """d omega_hat(t0+k) / d delta for a step delta added to every future command.

    Central finite difference through the latent rollout (the model is smooth; this matches
    autograd to well below the reported error and is ~100x cheaper than per-step autograd)."""
    dec = decoder_fn(model, probe)
    z0, c = context_start(model, ep, t0)
    outs = []
    for s in range(0, ep.n, batch):
        useq = torch.from_numpy(ep.u[s:s + batch, t0:t0 + horizon].astype(np.float32)).to(DEVICE)
        _set_ctx(model, c, s, batch)
        plus = dec(model.predictor.rollout(z0[s:s + batch], useq + delta))
        minus = dec(model.predictor.rollout(z0[s:s + batch], useq - delta))
        outs.append(((plus - minus) * 100.0 / (2 * delta)).cpu().numpy())   # rad/s per unit command
    return np.concatenate(outs)


def true_step_response(ep: Episodes, t0: int = T0, horizon: int = HORIZON, delta: float = 1e-3):
    """Finite-difference truth from the simulator (central difference)."""
    base = np.asarray(ep.u, dtype=np.float64)
    res = []
    for sgn in (+1, -1):
        u = base.copy(); u[:, t0:t0 + horizon] += sgn * delta
        omega, _ = simulate(ep.params, u, ep.omega[:, 0], ep.act[:, 0], ep.u_prev0)
        res.append(omega[:, t0 + 1:t0 + 1 + horizon])
    return (res[0] - res[1]) / (2 * delta)


def ode_residual(pred, ep: Episodes, t0: int = T0):
    """RMS torque residual [N m] of a forecast using the TRUE parameters and actuator path."""
    p = ep.params
    traj = np.concatenate([ep.omega[:, t0:t0 + 1], pred], axis=1).astype(np.float64)
    a = ep.act[:, t0:t0 + traj.shape[1] - 1].astype(np.float64)
    w0, w1 = traj[:, :-1], traj[:, 1:]
    wm = 0.5 * (w0 + w1)
    # Trapezoidal torque residual J dw/dt - (tau_E - tau_load) per sample.
    tau = (euler_torque(p_col(p), wm, a) - load_torque(p_col(p), wm))
    res = p.inertia_kg_m2[:, None] * (w1 - w0) / p.const.dt_s - tau
    return np.sqrt(np.mean(res ** 2, axis=1))


class _Col:
    def __init__(self, p):
        self._p = p
        self.const = p.const

    def __getattr__(self, name):
        return getattr(self._p, name)[:, None]


def p_col(p):
    return _Col(p)


def param_probe(model, labelled: Episodes, test: Episodes, t_window=(150, T0 + 1), lam=1e-3):
    """Linear probe z_{t} -> hidden physical parameters (standardised); R^2 per parameter at t0."""
    zl = encode(model, labelled, upto=t_window[1])[:, t_window[0]:t_window[1]]
    y = labelled.params.as_matrix()
    mu, sd = y.mean(0), y.std(0) + 1e-12
    yl = np.repeat(((y - mu) / sd)[:, None, :], zl.shape[1], axis=1).reshape(-1, y.shape[1])
    probe = fit_ridge(zl.reshape(-1, zl.shape[-1]), yl, lam)
    zt = encode(model, test, upto=T0 + 1)[:, T0]
    pred = apply_ridge(probe, zt) * sd + mu
    truth = test.params.as_matrix()
    r2 = 1 - ((pred - truth) ** 2).sum(0) / ((truth - truth.mean(0)) ** 2).sum(0)
    return dict(zip(PARAM_NAMES, r2.tolist()))


def rmse(a, b, axis=None):
    return np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2, axis=axis))


def bootstrap_ci(per_episode, n_boot: int = 2000, seed: int = 0):
    """95 % CI of the RMS over episodes from per-episode mean-squared errors."""
    rng = np.random.default_rng(seed)
    mse = np.asarray(per_episode)
    idx = rng.integers(0, len(mse), size=(n_boot, len(mse)))
    stats = np.sqrt(mse[idx].mean(1))
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


_QUICK_CACHE = {}


@torch.no_grad()
def quick_validation(model, train_ep: Episodes, val_ep: Episodes, n_label: int = 64, n_val: int = 64):
    """Cheap diagnostic during training: probe on 64 labelled train episodes, val on 64 episodes."""
    model.eval()
    lab, val = train_ep.subset(np.arange(n_label)), val_ep.subset(np.arange(n_val))
    probe = None if model.method in ("sup", "jepa_ft") else speed_probe(model, lab, val)
    filt = filtered_speed(model, val, probe)
    pred = forecast(model, val, probe, horizon=100)
    out = {"val_filter_rmse": float(rmse(filt[:, 20:], val.omega[:, 20:])),
           "val_roll_rmse": float(rmse(pred, val.omega[:, T0 + 1:T0 + 101]))}
    model.train()
    return out
