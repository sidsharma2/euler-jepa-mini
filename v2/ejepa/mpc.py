"""Closed-loop speed tracking with cross-entropy-method (CEM) MPC, in the true simulator.

Task: from t = 2 s, track a piecewise-constant speed set-point schedule (two 1 s
segments) using only the acoustic spectrogram.  Set-points are drawn inside each
episode's reachable steady band [omega_inf(u_min + 0.05), omega_inf(u_max - 0.05)], computed
from the true parameters *only to design feasible tasks* (the v1 audit found an
infeasible target; this guards against that).

Controllers share the same planner, horizon, sample budget and cost; they differ only in
the model used to predict speed:
  world model  latent belief (GRU state stepped online) + action-conditioned predictor
  greybox      comb tracker (causal) + grey-box model fitted on the first 2 s, with a
               speed-correction observer
  oracle       true parameters and true state (upper bound)
  pi           PI controller on the causal comb-tracker speed (no model, no planner)
"""
from __future__ import annotations

import numpy as np
import torch

from .classical import comb_scores, GRID, delayed, fit_greybox, _unpack
from .data import Episodes, denorm_omega
from .evaluate import T0, decoder_fn
from .physics import CONST, RotorSim, equilibrium_speed, euler_torque
from .sensors import SpectrumSensor
from .train import DEVICE

H_PLAN = 50            # 0.5 s look-ahead
KNOTS = 5              # piecewise-constant knots over the horizon
REPLAN = 5             # replan every 50 ms
N_SAMPLES, N_ELITE, N_ITERS = 128, 16, 4
W_DU = 2e3             # rad^2/s^2 per (unit command change)^2  -- smoothness penalty
T_CTRL = 200           # 2 s of closed-loop control after the 2 s context


SEGMENTS = 2           # two 1 s set-point segments (longer than the ~0.7 s rotor time constant)


def setpoints(ep: Episodes, rng: np.random.Generator, segments: int = SEGMENTS):
    """Moderate set-point steps (5-20 % of the reachable band) from the speed at t = 2 s.

    Large steps make every controller -- the oracle included -- actuator-saturation-limited,
    which measures the actuator rather than the model (found in the first MPC run).  Moderate
    steps keep the task feasible within each 1 s segment, so steady-state error measures how
    accurately each controller knows the speed and the plant."""
    lo = equilibrium_speed(ep.params, CONST.u_min + 0.05)
    hi = equilibrium_speed(ep.params, CONST.u_max - 0.05)
    band = hi - lo
    level, levels = ep.omega[:, T0].astype(np.float64), []
    for _ in range(segments):
        step = band * rng.uniform(0.05, 0.20, ep.n) * rng.choice([-1.0, 1.0], ep.n)
        level = np.clip(level + step, lo + 0.05 * band, hi - 0.05 * band)
        levels.append(level)
    return np.repeat(np.stack(levels, 1), T_CTRL // segments, axis=1)   # (N, T_CTRL)


def _expand_knots(knots, horizon=H_PLAN):
    return np.repeat(knots, horizon // knots.shape[-1], axis=-1)


def cem_plan(predict, ref, u_last, mean0, rng):
    """predict(U (N,S,H)) -> omega (N,S,H).  ref (N,H).  Returns best knots (N,KNOTS)."""
    n = ref.shape[0]
    mean, std = mean0.copy(), np.full((n, KNOTS), 0.25)
    for _ in range(N_ITERS):
        knots = np.clip(mean[:, None] + std[:, None] * rng.standard_normal((n, N_SAMPLES, KNOTS)),
                        CONST.u_min, CONST.u_max)
        knots[:, 0] = mean                                                # keep the incumbent
        u = _expand_knots(knots)
        w = predict(u)
        du = np.diff(np.concatenate([np.repeat(u_last[:, None, None], N_SAMPLES, 1), knots], -1), axis=-1)
        cost = np.mean((w - ref[:, None]) ** 2, -1) + W_DU * np.sum(du ** 2, -1)
        elite = np.take_along_axis(knots, np.argsort(cost, 1)[:, :N_ELITE, None], 1)
        mean, std = elite.mean(1), np.maximum(elite.std(1), 0.02)
    return mean


def ep_u_ctx(u_prev_ctx):
    """Commands u[k] aligned with context latents z[k] (u_prev[k] = u[k-1]); last entry padded."""
    return np.concatenate([u_prev_ctx[:, 1:], u_prev_ctx[:, -1:]], 1).astype(np.float32)


class WorldModelController:
    def __init__(self, model, probe=None):
        self.model, self.dec = model, decoder_fn(model, probe)

    def reset(self, ep, obs_ctx, u_prev_ctx):
        from .models import RECEPTIVE
        with torch.no_grad():
            z, self.h = self.model.encoder(torch.from_numpy(obs_ctx.astype(np.float32)).to(DEVICE),
                                           torch.from_numpy(u_prev_ctx.astype(np.float32)).to(DEVICE))
            self.z = z[:, -1]
            self.c = None
            if getattr(self.model, "ctx_dim", 0):
                self.c = self.model.context(z[:, 16:], torch.from_numpy(ep_u_ctx(u_prev_ctx)).to(DEVICE)[:, 16:])
        # Causal trunk receptive field: the last RECEPTIVE frames reproduce the batch computation.
        self.buf = torch.from_numpy(obs_ctx[:, -(RECEPTIVE - 1):].astype(np.float32)).to(DEVICE)

    @torch.no_grad()
    def observe(self, obs_t, u_prev_t):
        enc = self.model.encoder
        self.buf = torch.cat([self.buf, torch.from_numpy(obs_t.astype(np.float32)).to(DEVICE)[:, None]], 1)
        hmap = enc.trunk(self.buf)[:, :, -1:]                       # newest frame only
        z, self.h = enc.from_features(enc.readout(hmap),
                                      torch.from_numpy(u_prev_t.astype(np.float32)).to(DEVICE)[:, None], self.h)
        self.z = z[:, -1]
        self.buf = self.buf[:, 1:]

    @torch.no_grad()
    def predict(self, u):                       # u (N,S,H)
        n, s, h = u.shape
        z = self.z[:, None].expand(n, s, -1).reshape(n * s, -1)
        if self.c is not None:
            self.model.predictor.set_context(self.c.repeat_interleave(s, dim=0))
        traj = self.model.predictor.rollout(z, torch.from_numpy(u.reshape(n * s, h).astype(np.float32)).to(DEVICE))
        return denorm_omega(self.dec(traj).cpu().numpy()).reshape(n, s, h)


class GreyboxController:
    """Fitted grey-box model + causal comb tracker with a constant-gain speed observer."""

    def __init__(self, gain: float = 0.2):
        self.gain = gain

    def reset(self, ep, obs_ctx, u_prev_ctx):
        from .classical import track
        w_hat = track(obs_ctx)                                       # smoothed over the context
        u_ctx = np.concatenate([u_prev_ctx[:, 1:], u_prev_ctx[:, -1:]], 1)  # u[k] for k < T0+1 (last unknown yet)
        fit = fit_greybox(w_hat, u_ctx.astype(np.float64), u_prev_ctx[:, 0].astype(np.float64), fit_steps=T0, iters=300)
        self.theta = torch.tensor(fit["theta"]); self.delay = fit["delay"]
        self.alpha, self.beta, self.g0, self.g2, self.tau, _ = (x.numpy() for x in _unpack(self.theta, torch.zeros(len(self.delay))))
        # Re-simulate the fitted model over the context to get its actuator state.
        from .classical import greybox_simulate
        # Commands u[T0-6 .. T0-1] (u_prev_ctx[:, k] = u[k-1]).
        self.u_hist = [list(u_prev_ctx[i, -6:]) for i in range(len(self.delay))]
        ue = np.stack([delayed(u_ctx[i:i + 1], u_prev_ctx[i:i + 1, 0], int(self.delay[i]))[0] for i in range(len(self.delay))])
        with torch.no_grad():
            sim, a = greybox_simulate(self.theta, torch.tensor(fit["w_init"]), torch.tensor(fit["a_init"]),
                                      torch.tensor(ue[:, :T0], dtype=torch.float64))
        self.w, self.a = sim[:, -1].numpy(), a[:, -1].numpy()
        self.scores_prev = None
        self.track_delta = None
        self._init_tracker(obs_ctx)

    def _init_tracker(self, obs_ctx):
        from .classical import track
        self.w_meas = track(obs_ctx, causal=True)[:, -1]

    def _f(self, w, a, u):
        return self.alpha * a - self.beta * w - self.g0 - self.g2 * w * w, (u - a) / self.tau

    def _step(self, w, a, u, dt=CONST.dt_s):
        k1 = self._f(w, a, u); k2 = self._f(w + 0.5 * dt * k1[0], a + 0.5 * dt * k1[1], u)
        k3 = self._f(w + 0.5 * dt * k2[0], a + 0.5 * dt * k2[1], u); k4 = self._f(w + dt * k3[0], a + dt * k3[1], u)
        return w + dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6, a + dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6

    def observe(self, obs_t, u_prev_t):
        # Advance the model one step with the command actually issued, then correct speed.
        for i in range(len(self.delay)):
            self.u_hist[i].append(float(u_prev_t[i]))
        u_eff = np.array([self.u_hist[i][-1 - int(self.delay[i])] for i in range(len(self.delay))])
        self.w, self.a = self._step(self.w, self.a, u_eff)
        # Causal comb measurement, gated to a +/- 25 rad/s window around the model prediction.
        sc = comb_scores(obs_t[:, None])[:, 0]
        win = np.abs(GRID[None] - self.w[:, None]) <= 25
        meas = GRID[np.argmax(np.where(win, sc, -np.inf), 1)]
        self.w = self.w + self.gain * (meas - self.w)

    def predict(self, u):
        n, s, h = u.shape
        w = np.repeat(self.w[:, None], s, 1); a = np.repeat(self.a[:, None], s, 1)
        hist = np.array([self.u_hist[i][-6:] for i in range(n)])        # last commands (oldest first)
        out = np.empty((n, s, h))
        for k in range(h):
            u_eff = np.empty((n, s))
            for i in range(n):
                d = int(self.delay[i])
                u_eff[i] = u[i, :, k - d] if k - d >= 0 else hist[i, len(hist[i]) + (k - d)]
            w, a = self._step_b(w, a, u_eff)
            out[:, :, k] = w
        return out

    def _step_b(self, w, a, u):
        al, be, g0, g2, ta = (x[:, None] for x in (self.alpha, self.beta, self.g0, self.g2, self.tau))
        dt = CONST.dt_s
        def f(w, a): return al * a - be * w - g0 - g2 * w * w, (u - a) / ta
        k1 = f(w, a); k2 = f(w + 0.5 * dt * k1[0], a + 0.5 * dt * k1[1])
        k3 = f(w + 0.5 * dt * k2[0], a + 0.5 * dt * k2[1]); k4 = f(w + dt * k3[0], a + dt * k3[1])
        return w + dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6, a + dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6


class OracleController:
    """True parameters and true state (simulator copy)."""

    def reset(self, ep, obs_ctx, u_prev_ctx):
        self.p = ep.params

    def bind(self, sim):
        self.sim = sim

    def observe(self, obs_t, u_prev_t):
        pass

    def predict(self, u):
        n, s, h = u.shape
        rep = lambda x: np.repeat(x, s, 0)  # noqa: E731
        from .physics import EpisodeParams, PARAM_NAMES
        p = EpisodeParams(*(rep(getattr(self.p, k)) for k in PARAM_NAMES))
        sim = RotorSim(p, rep(self.sim.omega), rep(self.sim.a), rep(self.sim.buffer[:, 0]))
        sim.buffer = rep(self.sim.buffer)
        out = np.empty((n * s, h))
        flat = u.reshape(n * s, h)
        for k in range(h):
            out[:, k], _ = sim.step(flat[:, k])
        return out.reshape(n, s, h)


def run_closed_loop(controller, ep: Episodes, ref: np.ndarray, seed: int = 0, kind: str = "mpc",
                    pi_gains=(2e-3, 4e-3)):
    """Replay the episode's first 2 s (identical commands, fresh sensor noise), then control."""
    rng = np.random.default_rng(seed)
    n = ep.n
    sim = RotorSim(ep.params, ep.omega[:, 0], ep.act[:, 0], ep.u_prev0)
    sensor = SpectrumSensor(n, np.random.default_rng(seed + 1), ep.nuisance)
    obs_ctx = np.empty((n, T0 + 1, 64), np.float32)
    for k in range(T0 + 1):
        obs_ctx[:, k] = sensor.observe(sim.omega, euler_torque(ep.params, sim.omega, sim.a))
        if k < T0:
            sim.step(ep.u[:, k])
    u_prev_ctx = ep.u_prev()[:, :T0 + 1]
    if isinstance(controller, OracleController):
        controller.bind(sim)
    if kind == "mpc":
        controller.reset(ep, obs_ctx, u_prev_ctx)
    u_last = ep.u[:, T0 - 1].astype(np.float64)
    mean = np.repeat(u_last[:, None], KNOTS, 1)
    omega_log, u_log = np.empty((n, T_CTRL)), np.empty((n, T_CTRL))
    plan_u = None
    integ = np.zeros(n)
    w_meas = None
    if kind == "pi":
        from .classical import track
        w_meas = track(obs_ctx, causal=True)[:, -1]
        u_ff = ep.u[:, T0 - 1].astype(np.float64)
    for k in range(T_CTRL):
        if kind == "pi":
            # u = u_ff + Kp e + Ki int(e) dt, with conditional-integration anti-windup.
            err = ref[:, k] - w_meas
            u_raw = u_ff + pi_gains[0] * err + pi_gains[1] * (integ + err * CONST.dt_s)
            saturated = (u_raw > CONST.u_max) | (u_raw < CONST.u_min)
            integ = np.where(saturated, integ, integ + err * CONST.dt_s)
            u_cmd = np.clip(u_raw, CONST.u_min, CONST.u_max)
        else:
            if k % REPLAN == 0:
                window = ref[:, k:k + H_PLAN]
                if window.shape[1] < H_PLAN:
                    window = np.concatenate([window, np.repeat(window[:, -1:], H_PLAN - window.shape[1], 1)], 1)
                mean = cem_plan(controller.predict, window, u_last, mean, rng)
                plan_u = _expand_knots(mean[:, None])[:, 0]
                mean = np.concatenate([mean[:, 1:], mean[:, -1:]], 1)  # warm start (shifted)
                j = 0
            u_cmd = plan_u[:, j]; j += 1
        omega_log[:, k] = sim.omega
        sim.step(u_cmd)
        u_log[:, k] = u_cmd
        obs = sensor.observe(sim.omega, euler_torque(ep.params, sim.omega, sim.a))
        if kind == "pi":
            sc = comb_scores(obs[:, None])[:, 0]
            win = np.abs(GRID[None] - w_meas[:, None]) <= 25
            w_meas = 0.7 * w_meas + 0.3 * GRID[np.argmax(np.where(win, sc, -np.inf), 1)]
        else:
            controller.observe(obs, u_cmd)
        u_last = u_cmd
    return omega_log, u_log
