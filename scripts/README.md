# Script index

All scripts are intended to be run from the repository root. They write derived
outputs to the root-level `results/`, `figures/`, and `reports/` directories.

## Core experiments

- `run_experiment.py` — compact synthetic benchmark and JEPA-style model.
- `run_corrected_euler_benchmark.py` — corrected short-horizon analytical benchmark.
- `run_discrepancy_trajectory_training.py` — full-trajectory nonlinear-load comparison used by the paper.

## Baselines and diagnostics

- `run_baseline_suite.py` — persistence, extrapolation, ordinary least squares, and physics baselines.
- `run_discrepancy_benchmark.py` — quadratic-load discrepancy sweep.
- `run_discrepancy_diagnostics.py` — full-trajectory residual diagnostics.
- `run_training_diagnostics.py` — training-history and horizon diagnostics.
- `run_reachability_control.py` — reachability-aware control diagnostic.
- `run_regime_sweep.py` — physical-regime and forecast-horizon sweep.

## Model-development experiments

- `run_improvement_experiments.py` and `run_improvement_experiments_v2.py` — additive improvement studies.
- `run_jepa_improvement_ablation.py` — JEPA-style ablation study.
- `run_multistep_vjepa_lite.py` — multistep latent-prediction prototype.
- `run_reward_ablation.py` — objective/reward ablation.
- `run_target_norm_ablation.py` — target-normalization ablation.
- `run_epoch_scaling.py` — training-budget scaling study.

## Utilities

- `metrics.py` — reusable evaluation metrics.
- `physics_utils.py` — lightweight physics-step utilities.
- `plot_jepa_improvement_ablation.py` — plot generation from saved ablation results.
- `verify_equations_20260919.py` — independent analytical and matrix-equation audit.

The scripts are deliberately kept as transparent entry points rather than
hidden behind a package command. This makes each experiment easy to inspect,
rerun, and cite in a research workflow.
