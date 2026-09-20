# Mathematical verification — 2026-09-19

All automated checks passed. The audit found that the equations are internally correct, but the manuscript needed to distinguish the short corrected benchmark (`k_omega=0.01`, 10 steps) from the full-trajectory nonlinear-load experiment (`k_omega=0.25`, 200 steps).

The affine exact step agrees with refined fourth-order Runge–Kutta integration; the epsilon load formula recovers the requested fraction at 400 rad/s; the scalar control-sensitivity formula agrees with centered finite differences; and the block matrix exponential returns the stated discrete matrices.

```json
{
  "status": "passed",
  "experiment_protocols": {
    "corrected_short_horizon": {
      "komega": 0.01,
      "horizon_steps": 10,
      "dt_s": 0.01
    },
    "full_trajectory_nonlinear_load": {
      "komega": 0.25,
      "horizon_steps": 200,
      "dt_s": 0.01
    }
  },
  "checks": {
    "nominal_beta_short_s^-1": 0.1,
    "nominal_time_constant_short_s": 10.0,
    "nominal_beta_full_trajectory_s^-1": 2.5,
    "nominal_time_constant_full_trajectory_s": 0.4,
    "affine_exact_vs_rk4_100_substeps_abs_error_rad_s": 1.1368683772161603e-13,
    "quadratic_load_fraction_checks": {
      "0.0": {
        "c2_N_m_s2": 0.0,
        "recovered_fraction": 0.0
      },
      "0.1": {
        "c2_N_m_s2": 1.0416666666666667e-07,
        "recovered_fraction": 0.1
      },
      "0.2": {
        "c2_N_m_s2": 2.3437499999999998e-07,
        "recovered_fraction": 0.19999999999999998
      },
      "0.5": {
        "c2_N_m_s2": 9.374999999999999e-07,
        "recovered_fraction": 0.5
      }
    },
    "scalar_endpoint_sensitivity_analytic_rad_s_per_control": 31.784385696029265,
    "scalar_endpoint_sensitivity_centered_difference_rad_s_per_control": 31.784385695488023,
    "scalar_sensitivity_abs_error": 5.412417181105411e-10,
    "matrix_phi_block_exponential_max_abs_error": 0.0,
    "matrix_gamma_block_exponential": [
      [
        0.004968142825051984
      ],
      [
        0.011930275168535105
      ]
    ]
  },
  "interpretation": [
    "The affine exact solution agrees with refined RK4 integration.",
    "The quadratic-load formula recovers the requested epsilon fraction at 400 rad/s.",
    "The scalar control-sensitivity matrix formula agrees with centered finite differences.",
    "The block matrix exponential returns the stated Phi and Gamma matrices."
  ]
}
```
