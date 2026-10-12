"""Euler-JEPA v2: action-conditioned latent world models for a partially observed rotor.

Package layout
--------------
physics    Euler-work rotor + actuator lag/delay simulator (RK4, SI units).
sensors    Synthetic casing-acoustic spectrogram sensor with structured nuisance.
data       Episode generation, dataset splits, on-disk caching.
models     Belief encoder, action-conditioned predictor, JEPA / reconstruction / supervised heads.
train      Training loops for every learned method.
classical  Harmonic-comb speed tracker and batched grey-box system identification.
evaluate   Probes, open-loop rollouts, control sensitivity, ODE residuals.
mpc        Cross-entropy-method MPC on any model, closed loop in the true simulator.
"""
