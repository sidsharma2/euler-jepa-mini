# Euler-JEPA v2 — an action-conditioned latent world model for a rotor you can only hear

v2 is a redesign of the Euler-JEPA mini study. The v1 code, results and paper in the
parent folder are unchanged; v2 is additive and its numbers are **not comparable** with v1,
because the benchmark asks a different and harder question.

## Why v1 could not show what JEPA is for

The v1 audit (parent `reports/main.tex`) was honest, and its negative result was structural:

| v1 design choice | Consequence |
| --- | --- |
| Rotor speed observed directly and noise-free | Nothing for representation learning to do; supervised regression on the scored quantity must win. |
| JEPA latent prediction was an auxiliary term (weight 0.05); the forecast came from a separate head on the context embedding | The JEPA objective could not improve the forecast by construction; it could only regularise or harm the shared encoder. |
| Control held constant within every sequence | The control derivative is not identifiable from the data, so every learned model was control-insensitive. Perturbing *u* alone at test time was an off-manifold input. |
| Full-trajectory protocol: $k_\omega = 0.25$, $\omega_0\approx400$ rad/s | Equilibrium speed is about 21 rad/s, so every sequence was a spin-down far from any operating point. |
| 96 training sequences, single latent summary, effective rank about 1 of 24 | Collapse-prone and data-starved. |

## What v2 does instead

**Benchmark (synthetic, declared).** The same Euler-work rotor balance, now with a
lagged and delayed actuator, hidden per-episode parameters, and time-varying commands:

$$
J\,\dot\omega = \dot m\,r\,(k_u a - k_\omega\omega) - c_0 - c_2\omega^2,\qquad
\tau_a\,\dot a = u(t-d\,\Delta t) - a .
$$

$\dot m, J, k_\omega, c_2, \tau_a, d$ are drawn per episode and never shown to the models.
Operating points are consistent: the equilibrium speeds lie between about 130 and 740 rad/s, and
$1/\beta$ ranges from 0.5 to 1 s.

**Partial observation.** Models never see $\omega$. They receive a 64-bin casing-microphone
log spectrum every 10 ms (0–3.2 kHz). The spectrum contains blade-passing tones at
$hN_b\omega/2\pi$ whose level depends on speed and Euler torque, broadband flow noise, and
nuisance at level $\lambda$: distractor tones that hop in frequency, impulsive bursts, and
calibration drift. This is a declared synthetic model of tacholess speed sensing, not a calibrated
machine.

**Model.** A causal belief encoder $E_\theta$ (frame CNN followed by a GRU over spectra and past commands)
produces a LayerNorm latent $z_t$. An action-conditioned residual predictor
$z_{t+1}=P_\phi(z_t,u_t)$ is trained purely self-supervised:

* **multi-step latent rollout loss** $\sum_{k=1}^{K}\lVert P_\phi^{k}(z_t,u_{t:t+k})-\operatorname{sg}\,\bar E(x_{0:t+k})\rVert_1$ with $K=32$ (0.32 s),
* **one-step teacher forcing** at every time step,
* **EMA target encoder** $\bar\theta\leftarrow m\bar\theta+(1-m)\theta$ (cosine $m$: 0.996 → 1),
* **anti-collapse**: a variance hinge and a covariance penalty on $z$ (VICReg-style), with effective rank logged.

There is no reconstruction and there are no labels. Speed is read out afterwards by a **frozen linear probe**
fitted on a small labelled subset. Forecasts are **open-loop latent rollouts** under known
future commands. Control uses **CEM-MPC directly in latent space**, as in V-JEPA 2-AC.

**Controls (same encoder, predictor, optimiser, data and step budget).**

* *Reconstruction world model:* the same rollouts, but every latent is decoded back to the
  spectrum (observation prediction). This isolates the JEPA ingredient.
* *Supervised:* the same rollouts, regressing true speed. Run with all 3000 labelled episodes (upper
  bound) and with 10–300 labelled episodes (label-matched).
* *JEPA + fine-tune:* JEPA pre-training followed by supervised fine-tuning on 10–300 labels.
* *Classical, structure-aware:* a harmonic-comb speed tracker with Viterbi smoothing, followed by batched
  grey-box system identification of the **true ODE form** (including a delay search), then
  simulation. This baseline is given knowledge the learned models do not have.
* *Nominal physics, persistence, oracle* (true parameters and state, which verifies the protocol), a
  *random-encoder* probe control, and a *PI controller* for the control task.

**Evaluation.**

1. State estimation: causal filtering RMSE of $\hat\omega_t$ from spectra.
2. Open-loop forecast: 2 s ahead from a 2 s context, RMSE against horizon, with bootstrap CIs.
3. Held-out dynamical regime: $c_2$ and $J$ outside the training range.
4. Nuisance sweep: $\lambda = 0$ to 3 at test, models trained at $\lambda=1$ and $\lambda=2.5$.
5. Label efficiency: 10, 30, 100, 300 and 3000 labelled episodes.
6. Control authority: the step response $\partial\hat\omega(t_0+k)/\partial u$ by autograd through
   the latent rollout, compared with central finite differences of the true simulator. This is
   the v1 failure mode, now tested on data where it is identifiable.
7. Physics consistency: the RMS torque residual of the forecasts under the true parameters.
8. Hidden-parameter probes: linear $R^2$ for $\dot m, J, k_\omega, c_2, \tau_a, d$ from $z_{t=2\,\mathrm s}$.
9. Closed-loop speed tracking with CEM-MPC in the true simulator, using reachable set-points only.

## Reproduce

```powershell
conda activate rl-gpu        # torch 2.10 + CUDA 12.6, numpy, matplotlib, pandas
python make_data.py          # deterministic dataset splits -> cache/
python run_v2.py classical
python run_v2.py train       # skips finished runs; --only "jepa_s*" to filter
python run_v2.py eval
python run_v2.py mpc
python make_figures.py
python tests/test_v2_physics.py
```

## Results

> **Final (2026-10-13).** The JEPA, reconstruction and fully supervised world models report the
> mean ± s.d. over 3 seeds. The supervised label sweep, JEPA + fine-tune, the closed-loop study,
> the hybrid and all ablations use seed 0, and seed-0 numbers are compared with seed-0 numbers.
> Every number comes from `results/` via `make_tables.py` (`results/tables/tables.md`) and
> `paper/export_plot_data.py` (`paper/data/numbers.tex`). Errors are rotor-speed RMSE in rad/s over
> 300 held-out episodes. The paper is [`paper/main.pdf`](paper/main.pdf), and every figure is in
> [`paper/svg/`](paper/svg/).

![Forecast error against horizon](paper/svg/fig_horizon.svg)

### What JEPA buys you here

1. **Label efficiency (the headline).** A linear probe on the frozen JEPA latent, fitted on **10
   labelled episodes**, forecasts 2 s ahead with **13.2 ± 1.4**. Supervised training from scratch
   on the same 10 episodes gives **40.6**, which is 3.1× worse. JEPA pre-training followed by
   fine-tuning beats supervised training at every budget up to 300 episodes.

   | Labelled episodes | 10 | 30 | 100 | 300 | 3000 |
   |---|---:|---:|---:|---:|---:|
   | Supervised from scratch (seed 0) | 40.6 | 29.1 | 17.0 | 14.7 | 7.3 ± 0.2 |
   | Frozen JEPA + linear probe (3 seeds) | **13.2 ± 1.4** | **11.5 ± 0.8** | 11.3 ± 1.0 | 11.3 ± 1.1 | 11.3 ± 1.1 |
   | JEPA + fine-tune (seed 0) | 18.9 | 14.0 | 11.5 | **9.5** | — |

   ![Label efficiency](paper/svg/fig_labels.svg)

2. **Robustness to nuisance.** Raising test nuisance from λ = 0 to λ = 3 moves the JEPA 2 s
   forecast from 10.7 to 13.4 (+25 %). The classical pipeline goes from 4.1 to 14.3 (+250 %),
   because its harmonic tracker locks onto distractor tones. The reconstruction world model reaches
   23.1. With 300 labels, JEPA + fine-tune holds 12.9 at λ = 3, while supervised training reaches 27.5.

3. **Planning in latent space works.** CEM-MPC on the JEPA latent tracks speed set-points from the
   spectrogram alone. This is the excess tracking MSE over an oracle controller (40 episodes,
   rad²/s², paired 95 % bootstrap CIs, seed 0):

   | Controller | λ = 1 | λ = 3 |
   |---|---:|---:|
   | Supervised, 3000 labels | 3.9 [1.3, 6.6] | 20.8 [11.7, 31.8] |
   | Tracker + grey-box (true ODE form) | 6.6 [2.9, 11.2] | 14.5 [4.6, 27.7] |
   | JEPA + fine-tune, 300 labels | 7.7 [3.7, 12.3] | 28.6 [19.2, 39.1] |
   | JEPA world model, 300-label probe | 7.9 [3.5, 12.9] | 38.6 [18.5, 69.9] |
   | JEPA trained at λ = 2.5 | 13.5 | 32.0 |
   | Reconstruction world model, 300-label probe | 22.7 [9.2, 44.8] | 85.8 [29.2, 185.9] |
   | Supervised, 300 labels | 22.2 [16.4, 28.6] | 224 [120, 357] |
   | PI on tracked speed | 344 [258, 442] | 2444 [584, 5607] |

   With 300 labels, JEPA pre-training cuts excess closed-loop error 2.9× at λ = 1 and 7.8× at λ = 3.

   ![Closed-loop excess error](paper/svg/fig_mpc.svg)

4. **Control authority is now learned (the v1 failure mode).** The median relative error of the
   2 s step-response gain ∂ω/∂u is 0.10 for JEPA, 0.26 for reconstruction, 0.04 for full supervision
   and 0.03 for the grey-box. v1's learned models were control-insensitive.

5. **JEPA against reconstruction.** The architecture, data and budget are the same; only the
   objective differs. With linear probes, JEPA gives 11.3 ± 1.1 against 18.2 ± 1.0 for the 2 s
   forecast. With MLP probes the gap shrinks to 10.2 against 11.5. So much of the advantage is in how
   the speed information is organised, not only in whether it is present.

### Where it does not win (stated plainly)

* **In-distribution accuracy.** The classical pipeline, given the exact ODE form, forecasts with
  4.6. Full supervision with 3000 labels reaches 7.3 ± 0.2. JEPA's advantage is labels and
  robustness, not peak accuracy.
* **Machine identification.** Linear-probe R² from the latent at t = 2 s is 0.76 for the
  steady-state gain k_ω and 0.22 for the actuator lag τ_a. It is about 0 for the inertia J, the mass
  flow ṁ and the delay d. The latent captures state better than it identifies the machine.
* **Uncertainty.** A three-seed ensemble cuts error by about 9 % (10.3 against 11.3), but it is
  over-confident. Only 64 % of errors fall within ±2σ, and spread and error correlate at only 0.28.

### Second-round improvements (seed 0)

| Variant | In dist. | Held-out | λ = 3 | Verdict |
|---|---:|---:|---:|---|
| Main: long-horizon training (1.28 s rollouts) | 10.1 | 14.4 | 12.9 | largest single gain (was 14.7) |
| Full-horizon rollouts (2 s on 4 s crops) | 11.5 | 14.8 | 14.0 | no further gain |
| Nuisance-randomised pre-training (λ ~ U[0,3]) | 11.2 | 15.5 | 13.2 | filtering at λ = 3: 10.3 → **6.0** (MLP probe); forecasts barely move |
| Hybrid: JEPA belief → grey-box ID of the true ODE | **7.0** | 11.0 | 16.4 | −30 % in distribution; worse at λ = 3 (sensor error amplified) |
| No stage-1 pre-training (short horizon) | 15.7 | 16.1 | 20.1 | pre-training matters |
| Adaptive context embedding (short horizon) | 16.1 | 19.1 | 18.9 | negative result |

The pattern across these variants is consistent. Perception is no longer the bottleneck:
randomisation made the speed estimate much more robust, but forecasts did not follow. Long-horizon
error comes from incomplete machine identification. Physics dynamics help when perception is good
and hurt when it is not.

![JEPA perception + physics dynamics](paper/svg/fig_hybrid.svg)

### Failure modes found while building it

* **Action shortcut.** A single-stage JEPA whose EMA target encoder also sees the commands
  collapses onto the command history (effective rank 2.6 of 48). Speed becomes unrecoverable.
  VICReg regularisation did not fix this; observation-only targets did.
  ![Collapse](paper/svg/fig_collapse.svg)
* **Read-out bottleneck.** A hand-built spatial-soft-argmax read-out that stage 1 never trains
  threw away the speed information: MLP probe 18.1 on the read-out vs 5.1 on the full trunk map.
* **Rollout drift.** Probe-decoded rollouts diverged beyond the training horizon until a TD-MPC2
  latent-consistency term was added (applied to every method).
* **Infeasible control task.** The first MPC task used set-point steps of up to 80 % of the
  reachable band. Every controller, the oracle included, was saturation-limited, so it was
  replaced. The archived results are in `results/mpc_tracking_saturated_task_v0.csv`.

### Scope note

The supervised and fine-tuned label-sweep runs for seeds 1–2, and a nuisance-randomised
reconstruction control, were dropped for compute reasons. The GPU is shared with other local
jobs; see `logs/scope_trim.txt`. Everything runs on one laptop RTX 3050 Ti (4 GB).

### Figures

Every figure is a standalone TikZ/pgfplots file in [`paper/plots/`](paper/plots/). Each reads CSVs
exported from `results/` and is built to PDF (paper), SVG (web) and PNG with
`python paper/export_plot_data.py && python paper/build_figures.py`. The earlier matplotlib versions
are in `figures/`.
