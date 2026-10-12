"""Episode generation and caching.

Alignment convention (used everywhere):
    obs[:, k]    spectrum frame of state omega[:, k] at time k dt
    u[:, k]      command issued at time k, held over [k, k+1)
    omega[:, k]  true speed at time k (label; never given to self-supervised models)
The encoder at time k ingests (obs[:, k], u[:, k-1]); the predictor maps (z_k, u_k) -> z_{k+1}.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .physics import (CONST, EpisodeParams, RotorSim, euler_torque, excitation_commands,
                      initial_state, sample_params)
from .sensors import SpectrumSensor

T_EPISODE = 400          # 4 s at 100 Hz
T_CONTEXT = 200          # forecasts start at 2 s
GENERATOR_VERSION = "v2.0"


@dataclass
class Episodes:
    obs: np.ndarray        # (N, T, 64) float16
    u: np.ndarray          # (N, T) float32 commands
    u_prev0: np.ndarray    # (N,) command held before t = 0
    omega: np.ndarray      # (N, T) float32 true speed
    act: np.ndarray        # (N, T) float32 true actuator position
    params: EpisodeParams
    nuisance: float
    regime: str

    @property
    def n(self) -> int:
        return self.obs.shape[0]

    def u_prev(self) -> np.ndarray:
        """Command known to the encoder at step k, i.e. u[k-1] (u_prev0 at k = 0)."""
        return np.concatenate([self.u_prev0[:, None], self.u[:, :-1]], axis=1).astype(np.float32)

    def subset(self, index) -> "Episodes":
        return Episodes(self.obs[index], self.u[index], self.u_prev0[index], self.omega[index],
                        self.act[index], self.params.subset(index), self.nuisance, self.regime)


def rollout_episodes(params: EpisodeParams, u_cmd: np.ndarray, rng: np.random.Generator,
                     nuisance: float, omega0, a0, u_hist0):
    sim = RotorSim(params, omega0, a0, u_hist0)
    sensor = SpectrumSensor(params.n, rng, nuisance)
    n, steps = u_cmd.shape
    obs = np.empty((n, steps, 64), np.float16)
    omega = np.empty((n, steps), np.float32); act = np.empty((n, steps), np.float32)
    for k in range(steps):
        omega[:, k], act[:, k] = sim.omega, sim.a
        obs[:, k] = sensor.observe(sim.omega, euler_torque(params, sim.omega, sim.a))
        sim.step(u_cmd[:, k])
    return obs, omega, act


def generate(n: int, seed: int, regime: str = "id", nuisance: float = 1.0,
             steps: int = T_EPISODE, chunk: int = 4000) -> Episodes:
    rng = np.random.default_rng(seed)
    params = sample_params(n, rng, regime)
    omega0, a0, u0 = initial_state(params, rng)
    u_cmd = excitation_commands(n, steps, rng, u0).astype(np.float32)
    # nuisance == "mix": per-episode lambda ~ U(0, 3) (nuisance randomisation), drawn from a
    # separate generator so the physics stream is identical to a fixed-lambda split.
    lam = (np.random.default_rng(seed + 7).uniform(0.0, 3.0, n) if nuisance == "mix" else None)
    parts = []
    for start in range(0, n, chunk):
        idx = slice(start, min(n, start + chunk))
        parts.append(rollout_episodes(params.subset(idx), u_cmd[idx], rng, nuisance if lam is None else lam[idx],
                                      omega0[idx], a0[idx], u0[idx]))
    obs, omega, act = (np.concatenate(x) for x in zip(*parts))
    return Episodes(obs, u_cmd, u0.astype(np.float32), omega, act, params,
                    -1.0 if lam is not None else nuisance, regime)


SPLITS = {
    # name: (count, seed, regime, nuisance)
    "train": (3000, 1001, "id", 1.0),
    "val": (300, 1002, "id", 1.0),
    "test_id": (300, 1003, "id", 1.0),
    "test_ood": (300, 1004, "ood", 1.0),
    "train_hi": (3000, 1011, "id", 2.5),
    "val_hi": (300, 1012, "id", 2.5),
    "train_mix": (3000, 1021, "id", "mix"),     # per-episode lambda ~ U(0, 3)
    "val_mix": (300, 1022, "id", "mix"),
}
for _lam in (0.0, 0.5, 1.0, 2.0, 3.0):
    SPLITS[f"test_lam{_lam:g}"] = (300, 1100 + int(10 * _lam), "id", _lam)


def _key(name: str) -> str:
    spec = json.dumps([GENERATOR_VERSION, SPLITS[name], T_EPISODE, CONST.__dict__], sort_keys=True)
    return hashlib.sha1(spec.encode()).hexdigest()[:10]


def load_split(name: str, cache_dir: Path) -> Episodes:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}_{_key(name)}.npz"
    if path.exists():
        z = np.load(path)
        params = EpisodeParams.from_matrix(z["params"])
        return Episodes(z["obs"], z["u"], z["u_prev0"], z["omega"], z["act"], params,
                        float(z["nuisance"]) if z["nuisance"].dtype.kind in "fi" else -1.0, str(z["regime"]))
    count, seed, regime, nuisance = SPLITS[name]
    ep = generate(count, seed, regime, nuisance)
    np.savez(path, obs=ep.obs, u=ep.u, u_prev0=ep.u_prev0, omega=ep.omega, act=ep.act,
             params=ep.params.as_matrix(), nuisance=nuisance, regime=regime)
    return ep


# Speed normalisation used by every learned head and probe.
OMEGA_REF, OMEGA_SCALE = 400.0, 100.0


def norm_omega(omega):
    return (omega - OMEGA_REF) / OMEGA_SCALE


def denorm_omega(value):
    return value * OMEGA_SCALE + OMEGA_REF
