"""Classical, structure-aware baseline pipeline (deliberately strong).

1. Harmonic-comb speed tracker.  For a grid of candidate speeds, sum the log-spectrum at
   the first three blade-passing harmonics h N_b omega / (2 pi); smooth over time with a
   Viterbi (max-product) pass under a slew-rate limit.  `causal=True` returns the forward
   (filtered) argmax only, for closed-loop use.
2. Grey-box system identification.  Fit the *true* model structure
       d(omega)/dt = alpha a - beta omega - g0 - g2 omega^2,   tau_a da/dt = u(t - d dt) - a
   to the tracked speed over the context window by output-error minimisation (batched RK4 in
   torch, Adam, Huber loss), enumerating the transport delay d.  The forecast simulates the
   fitted model forward under the known future commands.
This baseline knows the governing equation; the learned models do not.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .physics import CONST, PARAM_RANGES
from .sensors import BIN_HZ, N_BINS

GRID = np.arange(150.0, 751.0, 1.0)
HARMONIC_W = (1.0, 0.8, 0.6)


def comb_scores(obs: np.ndarray) -> np.ndarray:
    """obs (N,T,64) normalised dB -> score (N,T,G)."""
    x = obs.astype(np.float32)
    score = np.zeros(obs.shape[:2] + (len(GRID),), np.float32)
    for h, w in zip((1, 2, 3), HARMONIC_W):
        pos = h * CONST.blade_count * GRID / (2 * np.pi) / BIN_HZ - 0.5
        inside = (pos >= 0) & (pos <= N_BINS - 1)
        p = np.clip(pos, 0, N_BINS - 1)
        lo = np.floor(p).astype(int); hi = np.minimum(lo + 1, N_BINS - 1); fr = (p - lo).astype(np.float32)
        val = x[..., lo] * (1 - fr) + x[..., hi] * fr
        score += w * np.where(inside, val, -1.0)
    return score


def track(obs: np.ndarray, kappa: float = 0.02, max_step: int = 8, sharp: float = 4.0,
          causal: bool = False, batch: int = 50) -> np.ndarray:
    out = []
    for s in range(0, obs.shape[0], batch):
        em = sharp * comb_scores(obs[s:s + batch])                 # log-emission (N,T,G)
        n, t_len, g = em.shape
        offsets = np.arange(-max_step, max_step + 1)
        pen = kappa * offsets.astype(np.float32) ** 2
        delta = em[:, 0].copy()
        back = np.zeros((n, t_len, g), np.int8)
        filt = np.empty((n, t_len), np.int64); filt[:, 0] = delta.argmax(1)
        for t in range(1, t_len):
            best = np.full((n, g), -np.inf, np.float32); arg = np.zeros((n, g), np.int8)
            for j, o in enumerate(offsets):
                shifted = np.full((n, g), -np.inf, np.float32)
                if o >= 0:
                    shifted[:, o:] = delta[:, :g - o] if o > 0 else delta
                else:
                    shifted[:, :o] = delta[:, -o:]
                cand = shifted - pen[j]
                better = cand > best
                best = np.where(better, cand, best); arg = np.where(better, j, arg)
            delta = best + em[:, t]
            delta -= delta.max(1, keepdims=True)
            back[:, t] = arg
            filt[:, t] = delta.argmax(1)
        if causal:
            out.append(GRID[filt])
            continue
        path = np.empty((n, t_len), np.int64); path[:, -1] = delta.argmax(1)
        rows = np.arange(n)
        for t in range(t_len - 1, 0, -1):
            path[:, t - 1] = path[:, t] - offsets[back[rows, t, path[:, t]]]
        out.append(GRID[np.clip(path, 0, g - 1)])
    return np.concatenate(out).astype(np.float64)


# ----------------------------------------------------------------------------- grey box
DELAYS = tuple(range(PARAM_RANGES["id"]["delay_steps"][0], PARAM_RANGES["id"]["delay_steps"][1] + 1))


def _unpack(theta, omega_init):
    alpha = 1000.0 * torch.exp(theta[:, 0])
    beta = 1.5 * torch.exp(theta[:, 1])
    g0 = 7.5 + 10.0 * theta[:, 2]
    g2 = 1e-3 * F.softplus(theta[:, 3] - 2.0)
    tau = 0.1 * torch.exp(theta[:, 4])
    w0 = omega_init + 10.0 * theta[:, 5]
    return alpha, beta, g0, g2, tau, w0


def greybox_simulate(theta, omega_init, a_init, u_eff, dt=CONST.dt_s):
    """Batched RK4 (one step per sample).  u_eff (B,T) already delayed. Returns omega (B,T+1), a."""
    alpha, beta, g0, g2, tau, w = _unpack(theta, omega_init)
    a = a_init
    ws, as_ = [w], [a]

    def f(w, a, u):
        return alpha * a - beta * w - g0 - g2 * w * w, (u - a) / tau

    for k in range(u_eff.shape[1]):
        u = u_eff[:, k]
        k1 = f(w, a, u); k2 = f(w + 0.5 * dt * k1[0], a + 0.5 * dt * k1[1], u)
        k3 = f(w + 0.5 * dt * k2[0], a + 0.5 * dt * k2[1], u); k4 = f(w + dt * k3[0], a + dt * k3[1], u)
        w = w + dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
        a = a + dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
        ws.append(w); as_.append(a)
    return torch.stack(ws, 1), torch.stack(as_, 1)


def delayed(u, u_prev0, d):
    """u_eff[k] = u[k-d] with the pre-episode command for k < d."""
    if d == 0:
        return u
    pad = np.repeat(u_prev0[:, None], d, axis=1)
    return np.concatenate([pad, u[:, :-d]], axis=1)


def fit_greybox(omega_hat, u, u_prev0, fit_steps: int, iters: int = 400, device=None, t_start: int = 0):
    """Fit per-episode grey-box parameters to a speed estimate over [t_start, fit_steps].

    All candidate delays are fitted in one batch (N * n_delays independent problems) on the GPU
    when available; the delay with the lowest output-error loss is kept per episode.  With
    t_start > 0 (e.g. a learned causal estimate that needs a warm-up) the actuator state at
    t_start is unknown and is fitted as a seventh parameter."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n, nd = omega_hat.shape[0], len(DELAYS)
    ue = np.concatenate([delayed(u, u_prev0, d)[:, t_start:fit_steps] for d in DELAYS])
    ue = torch.tensor(ue, dtype=torch.float32, device=device)
    target = torch.tensor(np.tile(omega_hat[:, t_start:fit_steps + 1], (nd, 1)), dtype=torch.float32, device=device)
    w_init = target[:, 0].clone()
    a_guess = u_prev0 if t_start == 0 else u[:, t_start - 1]
    a_guess = torch.tensor(np.tile(a_guess, nd), dtype=torch.float32, device=device)
    theta = torch.zeros(n * nd, 7, dtype=torch.float32, device=device, requires_grad=True)
    free_a0 = t_start > 0

    def a0_of(th):
        return a_guess + 0.3 * torch.tanh(th[:, 6]) if free_a0 else a_guess

    opt = torch.optim.Adam([theta], lr=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=0.002)
    for _ in range(iters):
        sim, _ = greybox_simulate(theta, w_init, a0_of(theta), ue)
        loss_each = F.huber_loss(sim, target, delta=5.0, reduction="none").mean(1)
        opt.zero_grad(); loss_each.sum().backward(); opt.step(); sched.step()
    with torch.no_grad():
        a0 = a0_of(theta)
        sim, _ = greybox_simulate(theta, w_init, a0, ue)
        losses = F.huber_loss(sim, target, delta=5.0, reduction="none").mean(1).reshape(nd, n).cpu().numpy()
    best = losses.argmin(0)
    pick = lambda x: x.detach().cpu().numpy().astype(np.float64).reshape(nd, n, *x.shape[1:])[best, np.arange(n)]  # noqa: E731
    return {"theta": pick(theta), "delay": np.asarray(DELAYS)[best], "w_init": omega_hat[:, t_start].astype(np.float64),
            "a_init": pick(a0), "fit_loss": losses.min(0), "t_start": t_start}


def greybox_forecast(fit, u, u_prev0, t0: int, horizon: int, u_override=None):
    """Simulate the fitted model from t_start through t0 + horizon; return omega[t0+1 : t0+1+H]."""
    u_use = u if u_override is None else u_override
    ts = int(fit.get("t_start", 0))
    out = np.empty((u.shape[0], horizon))
    for d in np.unique(fit["delay"]):
        idx = np.where(fit["delay"] == d)[0]
        ue = torch.tensor(delayed(u_use[idx], u_prev0[idx], int(d))[:, ts:t0 + horizon], dtype=torch.float64)
        with torch.no_grad():
            sim, _ = greybox_simulate(torch.tensor(fit["theta"][idx]), torch.tensor(fit["w_init"][idx]),
                                      torch.tensor(fit["a_init"][idx]), ue)
        out[idx] = sim[:, t0 + 1 - ts:t0 + 1 - ts + horizon].numpy()
    return out


def greybox_params(fit):
    alpha, beta, g0, g2, tau, w0 = (x.numpy() for x in _unpack(torch.tensor(fit["theta"]), torch.tensor(fit["w_init"])))
    return {"alpha": alpha, "beta": beta, "g0": g0, "g2": g2, "tau": tau, "delay": fit["delay"]}
