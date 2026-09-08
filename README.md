# When Sycophancy Becomes Endogenous

Companion code for *When Sycophancy Becomes Endogenous: A Coupled Feedback Loop Model of Delusional Spiralling*. Version `3.0.0-review` contains the revised simulation, its saved results, and the source for every manuscript figure.

The model uses a binary world state, a Bayesian user, and a bot whose sycophantic response propensity can change after agreement with the user's expressed opinion. The revised paper distinguishes that agreement signal from the user's actual posterior movement. It also separates rising exposure to sycophancy from dependence on the current user.

## Reproduce the revised analyses

Use Python 3.10 or later. Install the dependencies, run the model checks, generate the numerical results, then draw the figures:

```bash
python -m pip install -r requirements.txt
python scripts/check_model.py
python scripts/run_all.py
python scripts/make_figures.py
```

The model checks use ordinary Python assertions and do not require pytest. They can also be run with `python -m pytest` if pytest is installed. The analyses use 10,000 conversations per condition and 100 rounds per conversation. Execution time depends on the machine, especially for the 101-point inference checks.

## Files and their roles

| Path | Contents |
| --- | --- |
| `src/model.py` | Shared simulation and probability-conserving diffusion |
| `src/statistics.py` | Wilson intervals, exact paired tests, and Holm adjustment |
| `scripts/run_all.py` | All revised numerical experiments |
| `scripts/make_figures.py` | Figures 1-9 and supplementary Figure S1, drawn from saved results |
| `scripts/check_model.py` | Standalone runner for model identities |
| `tests/test_model.py` | Probability, boundary, and equivalent-setting checks |
| `results/` | Counts, intervals, paired comparisons, traces, and event arrays |
| `figures/` | Publication figures in PNG and SVG formats |
| `docs/methods.md` | Numerical specification, comparison families, and knowledge assumptions |
| `docs/revision_record.md` | Changes to the implementation and interpretation |
| `legacy/v2/` | The supplied earlier source and saved outputs, retained for provenance |

## Sampling and interpretation

Each revised condition starts `numpy.random.default_rng(42)` and consumes six uniform arrays per round in a fixed order. Repeated specifications therefore give the same result, and comparisons pair simulated conversations through common random inputs. The tests use discordant outcomes, rather than treating paired conditions as independent binomial samples. Donor controls use an independent seed, 4242; signal permutations and fair signals use seed 987.

`results/manifest.json` records settings, dependency versions, and source hashes. `results/events.npz` retains the Boolean event arrays for the principal comparisons. Saved rate summaries include integer counts and denominators; zero observed crossings are not represented as proof of zero underlying probability.

The reported rates quantify the specified simulation. They do not estimate clinical incidence or the frequency of harm among deployed-chatbot users. A catastrophic spiral is an operational threshold crossing, `P(H=0) >= 0.99` at any round, while the actual world is `H=1`.

## Changes from the supplied v2 archive

The active analyses replace truncated Gaussian diffusion with reflection that preserves each world-state marginal. They add exact-rule and oracle learners, matched mitigation controls, component comparisons, donor schedule replay, shuffled signals, and fair feedback. The logistic endpoints are explicitly absorbing. All active numerical claims and figures use one common implementation.

The earlier scripts are under `legacy/v2`; they are not inputs to the revised pipeline. Their original notes disclose that the mitigation and graded-signal scripts reconstructed code lost between revision rounds. Those notes are preserved. The revised estimates are new paired reruns, not numerical targets fitted to the earlier output.

The manuscript's anonymised review copy can be submitted with a supplementary archive containing the active source and results. Public repository metadata identifies the author and should be handled according to the journal's submission requirements.

## Version and citation

This is a review version. The pull request records the exact source proposed for the manuscript revision; a final release should identify the accepted manuscript version and its corresponding commit. Citation metadata is supplied in `CITATION.cff`.
