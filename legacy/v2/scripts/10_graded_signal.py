"""
Graded satisfaction signal robustness check (Section 5).

RECONSTRUCTION NOTE. The original graded-signal script was lost between the
July and August 2026 revision rounds. This script is a reconstruction built
from the manuscript's specification -- s(t) equals the magnitude of the
posterior shift toward the user's expressed belief, replacing the binary +/-1
of the main analysis -- and verified against the archived result set (graded
coupled spiralling 0.8 vs binary 3.5 per cent at pi_0 = 0.0; 17.1 vs 23.6 per
cent at pi_0 = 0.3) at Monte Carlo tolerance in August 2026.

Signal definition: after the user's Bayesian update,
    s(t) = P(t)(H = h*) - P(t-1)(H = h*)
i.e. the signed change in the probability the user assigns to their own
expressed belief h*. Confirming responses give s(t) > 0, disconfirming
responses s(t) < 0, with magnitude equal to the belief shift (|s| <= ~0.1 per
round in this parameterisation, against the binary signal's constant 1.0).
The pi update rule is unchanged: pi(t+1) = clip(pi(t) + alpha * s(t), 0, 1),
alpha = 0.02.

Model mechanics are otherwise identical to src/simulation_core.run_coupled_naive
(hallucinating-sycophant mode, alpha = 0.02, T = 100, N = 10,000).
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import save_json

T = 100
N_SIMS = 10_000
EPSILON = 0.01
P1_H0 = 0.4
P1_H1 = 0.6
ALPHA = 0.02
SEED = 42


def run_graded(pi_init: float, alpha: float = ALPHA, graded: bool = True,
               n_sims: int = N_SIMS, seed: int | None = SEED) -> dict:
    """Coupled model with graded (posterior-shift) or binary satisfaction signal.

    alpha = 0.0 gives the fixed-pi baseline (signal irrelevant).
    """
    rng = np.random.default_rng(seed)
    prior_h0 = np.full(n_sims, 0.5)
    pi = np.full(n_sims, float(pi_init))
    ever_spiraled = np.zeros(n_sims, dtype=bool)

    for _ in range(T):
        h_star = (rng.random(n_sims) >= prior_h0).astype(int)
        d1 = (rng.random(n_sims) < P1_H1).astype(int)
        d2 = (rng.random(n_sims) < P1_H1).astype(int)
        is_syc = rng.random(n_sims) < pi

        syc_d = h_star.copy()
        imp_d = np.where(rng.random(n_sims) < 0.5, d1, d2)
        obs_d = np.where(is_syc, syc_d, imp_d)

        lik_h0_obs = np.where(obs_d == 1, P1_H0, 1 - P1_H0)
        lik_h1_obs = np.where(obs_d == 1, P1_H1, 1 - P1_H1)
        confirmed = np.where(h_star == 0, lik_h0_obs > lik_h1_obs, lik_h1_obs > lik_h0_obs)

        prev = prior_h0.copy()
        unnorm_h0 = lik_h0_obs * prior_h0
        unnorm_h1 = lik_h1_obs * (1 - prior_h0)
        prior_h0 = unnorm_h0 / (unnorm_h0 + unnorm_h1)
        prior_h0 = np.clip(prior_h0, 1e-15, 1 - 1e-15)
        ever_spiraled |= prior_h0 >= (1 - EPSILON)

        if alpha > 0.0:
            if graded:
                # Signed posterior shift toward the expressed belief h*.
                satisfaction = np.where(h_star == 0, prior_h0 - prev, prev - prior_h0)
            else:
                satisfaction = np.where(confirmed, 1.0, -1.0)
            pi = np.clip(pi + alpha * satisfaction, 0.0, 1.0)

    return {
        "spiral_rate": float(ever_spiraled.mean()),
        "mean_final_pi": float(pi.mean()),
    }


PI0_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def main() -> None:
    results: dict = {"seed": SEED, "alpha": ALPHA, "n_sims": N_SIMS, "rows": {}}

    print("=== Graded vs binary satisfaction signal: spiral rates ===")
    print(f"{'pi_0':>5} | {'fixed':>8} | {'graded':>8} | {'binary':>8} | {'graded final pi':>15}")
    print("-" * 58)

    for pi0 in PI0_GRID:
        r_fixed = run_graded(pi0, alpha=0.0)
        r_graded = run_graded(pi0, graded=True)
        r_binary = run_graded(pi0, graded=False)
        results["rows"][f"pi0_{pi0}"] = {
            "fixed": r_fixed["spiral_rate"],
            "graded_coupled": r_graded["spiral_rate"],
            "binary_coupled": r_binary["spiral_rate"],
            "graded_final_pi": r_graded["mean_final_pi"],
            "binary_final_pi": r_binary["mean_final_pi"],
        }
        print(
            f"{pi0:>5.1f} | {r_fixed['spiral_rate']:>8.4f} | "
            f"{r_graded['spiral_rate']:>8.4f} | {r_binary['spiral_rate']:>8.4f} | "
            f"{r_graded['mean_final_pi']:>15.4f}"
        )

    save_json(results, "graded_signal_results.json")
    print("\nSaved results/graded_signal_results.json")


if __name__ == "__main__":
    main()
