# Data sources and software citations

## Data classification

This project does not ingest a private experimental dataset or an external raw
dataset. Every trajectory used by the Euler-JEPA experiments is synthetic and
is generated from the declared rotor equation by the repository scripts.

The main generators are:

- [`scripts/run_experiment.py`](scripts/run_experiment.py) for the compact corrected benchmark;
- [`scripts/run_discrepancy_trajectory_training.py`](scripts/run_discrepancy_trajectory_training.py)
  for the full-trajectory nonlinear-load experiment;
- the other `scripts/run_*.py` files for additive baselines, ablations, and diagnostics.

The synthetic data should therefore be interpreted as software-verification
and method-development data. They do not represent measured compressor,
turbine, combustor, or engine behaviour.

## External data boundary

The UIUC airfoil data used in the separate `airfoil-polar-baseline` project are
not included here and are not used to train, calibrate, or validate the
Euler-JEPA model. No UIUC-derived quantity appears in the reported rotor
experiments.

## Scientific method references

- Assran et al., “Self-Supervised Learning from Images with a Joint-Embedding
  Predictive Architecture,” [arXiv:2301.08243](https://arxiv.org/abs/2301.08243).
- Bardes et al., “Revisiting Feature Prediction for Learning Visual
  Representations from Video,” [arXiv:2404.08471](https://arxiv.org/abs/2404.08471).
- Chen et al., “Neural Ordinary Differential Equations,”
  [arXiv:1806.07366](https://arxiv.org/abs/1806.07366).
- Brunton, Proctor, and Kutz, “Discovering governing equations from data by
  sparse identification of nonlinear dynamical systems,”
  [doi:10.1073/pnas.1517384113](https://doi.org/10.1073/pnas.1517384113).

The complete bibliography used by the paper is in
[`reports/references.bib`](reports/references.bib).

## Software citations

The implementation uses and should be cited through the following project
pages:

- [Python](https://www.python.org/)
- [NumPy](https://numpy.org/)
- [PyTorch](https://pytorch.org/)
- [Matplotlib](https://matplotlib.org/)
- [pytest](https://pytest.org/)
- [LaTeX](https://www.latex-project.org/)
- [PGFPlots](https://pgfplots.sourceforge.net/)

## Provenance limitations

The public manifest records the current repository-level lineage. Earlier
local runs were reconstructed from surviving artifacts and should not be
treated as complete original run logs. Results remain exploratory, and the
paper does not claim physical validation.
