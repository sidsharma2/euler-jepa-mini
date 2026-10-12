"""Synthetic casing-acoustic spectrogram sensor.

The learned models never observe omega directly.  Each 10 ms frame is a 64-bin
log-power spectrum (0-3.2 kHz, 50 Hz bins: a 20 ms Hann window at 50 % overlap)
of a casing microphone.  It contains:

  signal    blade-passing tones at h * N_b * omega / (2 pi), h = 1, 2, 3, whose level
            follows a fan-law speed term plus a steady-loading term in Euler torque
            (so the spectrum carries acceleration information, not only speed);
  floor     speed-dependent pink broadband flow noise;
  nuisance  (scaled by lambda) telegraph distractor tones from neighbouring machinery
            that appear at random frequencies, drift, and switch off; impulsive
            broadband bursts; per-episode calibration offset and slow gain drift;
  noise     Welch periodogram fluctuation (chi^2 with 8 degrees of freedom).

This is a declared synthetic observation model, chosen because tacholess speed
estimation from casing acoustics/vibration is a real monitoring problem.  It is not
a calibrated model of any specific machine.
"""
from __future__ import annotations

import numpy as np

from .physics import CONST

N_BINS = 64
BIN_HZ = 50.0
FREQS_HZ = (np.arange(N_BINS) + 0.5) * BIN_HZ
MAX_DISTRACTORS = 6
DB_OFFSET, DB_SCALE = 50.0, 15.0


class SpectrumSensor:
    """`nuisance` is a scalar lambda, or an (n,) array of per-episode lambdas (nuisance randomisation)."""

    def __init__(self, n: int, rng: np.random.Generator, nuisance=1.0):
        self.n, self.rng = n, rng
        self.per_episode = np.ndim(nuisance) > 0
        self.lam_vec = np.asarray(nuisance, dtype=np.float64) * np.ones(n)
        self.lam = float(self.lam_vec.max()) if self.per_episode else float(nuisance)
        self.gain_db = 2.0 * rng.standard_normal(n)              # calibration offset
        self.drift_db = np.zeros(n)                               # slow gain drift
        self.floor_db = 2.0 * rng.standard_normal(n)
        if self.per_episode:
            count = np.minimum(rng.poisson(1.5 * self.lam_vec), MAX_DISTRACTORS)
        else:
            count = np.minimum(rng.poisson(1.5 * self.lam, size=n), MAX_DISTRACTORS) if self.lam > 0 else np.zeros(n, int)
        self.active_slot = np.arange(MAX_DISTRACTORS)[None, :] < count[:, None]   # slot exists
        self.on = rng.random((n, MAX_DISTRACTORS)) < 0.4
        self.freq = rng.uniform(150.0, 3100.0, size=(n, MAX_DISTRACTORS))
        self.level = 66.0 + 3.0 * rng.standard_normal((n, MAX_DISTRACTORS))
        self.burst = np.zeros(n)

    def observe(self, omega, tau_euler):
        rng, lam = self.rng, self.lam
        f = FREQS_HZ[None, :]
        omega = np.maximum(np.asarray(omega, dtype=np.float64), 1.0)
        power = np.zeros((self.n, N_BINS))
        # Blade-passing tones.
        bpf = CONST.blade_count * omega / (2.0 * np.pi)
        speed_db = 50.0 * np.log10(omega / 400.0)
        loading_db = 0.6 * np.asarray(tau_euler)
        for h in (1, 2, 3):
            level = 70.0 + speed_db + loading_db - 7.0 * (h - 1)
            centre = h * bpf * (1.0 + 0.002 * rng.standard_normal(self.n))
            power += 10 ** (level[:, None] / 10) * np.exp(-0.5 * ((f - centre[:, None]) / 30.0) ** 2)
        # Pink broadband floor.
        floor = 42.0 + 20.0 * np.log10(omega / 400.0)[:, None] + self.floor_db[:, None] - 10.0 * np.log10(f / 500.0)
        power += 10 ** (floor / 10)
        if lam > 0:
            # Telegraph distractor tones: switch on at a new random frequency, drift, switch off.
            switch_on = (~self.on) & (rng.random(self.on.shape) < 0.02)
            switch_off = self.on & (rng.random(self.on.shape) < 0.03)
            self.freq = np.where(switch_on, rng.uniform(150.0, 3100.0, self.freq.shape), self.freq)
            self.level = np.where(switch_on, 66.0 + 3.0 * rng.standard_normal(self.level.shape), self.level)
            self.on = (self.on | switch_on) & ~switch_off
            self.freq = np.clip(self.freq + 5.0 * rng.standard_normal(self.freq.shape), 100.0, 3150.0)
            lam_db = 10 * np.log10(np.maximum(self.lam_vec, 1e-12))[:, None] if self.per_episode else 10 * np.log10(lam)
            amp = np.where(self.on & self.active_slot, 10 ** ((self.level + lam_db) / 10), 0.0)
            power += np.einsum("nd,ndb->nb", amp,
                               np.exp(-0.5 * ((f[None] - self.freq[:, :, None]) / 30.0) ** 2))
            # Impulsive broadband bursts.
            start = rng.random(self.n) < 0.01 * (self.lam_vec if self.per_episode else lam)
            self.burst = np.where(start, 2.0, np.maximum(self.burst - 1.0, 0.0))
            power *= np.where(self.burst > 0, 10 ** 0.8, 1.0)[:, None]
            self.drift_db = np.clip(self.drift_db + 0.05 * rng.standard_normal(self.n), -3.0, 3.0)
            gain = self.gain_db + self.drift_db
            if self.per_episode:
                gain = np.where(self.lam_vec > 0, gain, 0.0)        # clean sensor when lambda = 0
        else:
            gain = np.zeros(self.n)
        power *= rng.gamma(4.0, 0.25, size=power.shape)            # chi^2_8 / 8
        db = 10.0 * np.log10(power) + gain[:, None]
        return ((db - DB_OFFSET) / DB_SCALE).astype(np.float32)
