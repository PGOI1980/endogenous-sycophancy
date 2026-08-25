# When Sycophancy Becomes Endogenous: Simulation Code

Companion code for the paper:

**Gallacher, P. (2026). _When Sycophancy Becomes Endogenous: A Coupled Feedback Loop Model of Delusional Spiralling._**

This repository contains the simulation scripts, saved outputs, and figure-generation code used to replicate the baseline model from Chandra et al. (2026) and extend it with an endogenous sycophancy update rule.

## Repository layout

```text
endogenous-sycophancy/
  README.md
  LICENSE
  CITATION.cff
  requirements.txt
  src/
    __init__.py
    io_utils.py
    simulation_core.py
  scripts/
    01_baseline_replication.py
    02_coupled_model.py
    03_informed_user.py
    04_sensitivity_analysis.py
    05_adaptive_informed_user.py
    06_confidence_intervals.py
    07_generate_figures.py
    08_generate_revision_figures.py
    09_mitigations.py
    10_graded_signal.py
  results/
  figures/
  tests/
    test_smoke.py
```

## Environment

- Python 3.10 or newer recommended
- NumPy
- SciPy
- Matplotlib
- pytest for smoke tests

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Model parameters

These follow the paper and the Chandra et al. baseline:

- `H in {0, 1}`, with true state `H = 1`
- prior `p(H=0) = p(H=1) = 0.5`
- `k = 2` data points per round
- `p(D=1 | H=0) = 0.4`, `p(D=1 | H=1) = 0.6`
- `T = 100` rounds
- `N_SIMS = 10_000` simulations per condition
- catastrophic spiralling threshold: `p(H=0) >= 0.99`
- baseline coupling strength: `alpha = 0.02`

## Reproducibility notes

- Scripts use fixed NumPy seeds for deterministic reruns.
- Results are written to `results/`.
- Figures are written to `figures/`.
- The numbered scripts are intended to be run in order.
- The main numerical scripts save canonical JSON outputs that the plotting scripts consume.

## Run order

```bash
python scripts/01_baseline_replication.py
python scripts/02_coupled_model.py
python scripts/03_informed_user.py
python scripts/04_sensitivity_analysis.py
python scripts/05_adaptive_informed_user.py
python scripts/06_confidence_intervals.py
python scripts/07_generate_figures.py
python scripts/08_generate_revision_figures.py
python scripts/09_mitigations.py
python scripts/10_graded_signal.py
```

## Expected outputs

- `01_baseline_replication.py` -> `results/baseline_results.json`
- `02_coupled_model.py` -> `results/coupled_results.json`
- `03_informed_user.py` -> `results/informed_results.json`
- `04_sensitivity_analysis.py` -> `results/sensitivity_results.json`
- `05_adaptive_informed_user.py` -> `results/adaptive_results.json`
- `06_confidence_intervals.py` -> `results/confidence_intervals.json`
- `07_generate_figures.py` -> main manuscript figures in `figures/`
- `08_generate_revision_figures.py` -> revision figures in `figures/`
- `09_mitigations.py` -> `results/mitigation_results.json` and `figures/figure9_mitigations.png`
- `10_graded_signal.py` -> `results/graded_signal_results.json`

Scripts 09 and 10 are documented RECONSTRUCTIONS of scripts lost between revision rounds, calibrated against the archived result set; see their headers for the verification record.

## Smoke test

```bash
pytest -q
```

## Public release notes

This repository is organized for portability:

- no hardcoded local absolute paths
- relative `results/` and `figures/` directories
- simple smoke tests
- metadata files for licensing and citation

## Citation

Please cite the companion paper and, if relevant, this code repository.
