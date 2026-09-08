"""
Add confidence intervals to main results.
Use Wilson score intervals and Fisher's exact test for coupled vs fixed differences.
"""
from pathlib import Path
import sys
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import save_json

np.random.seed(42)

T = 100
N_SIMS = 10000
EPSILON = 0.01
P1_H0 = 0.4
P1_H1 = 0.6


def run_with_ci(pi_init, alpha, n_sims=N_SIMS):
    """Run simulation and return rate with 95% Wilson CI."""
    prior_h0 = np.full(n_sims, 0.5)
    pi = np.full(n_sims, float(pi_init))
    ever_spiraled = np.zeros(n_sims, dtype=bool)

    for _ in range(T):
        h_star = (np.random.random(n_sims) >= prior_h0).astype(int)
        d1 = (np.random.random(n_sims) < P1_H1).astype(int)
        d2 = (np.random.random(n_sims) < P1_H1).astype(int)

        is_syc = np.random.random(n_sims) < pi
        syc_d = h_star.copy()
        imp_d = np.where(np.random.random(n_sims) < 0.5, d1, d2)
        obs_d = np.where(is_syc, syc_d, imp_d)

        lik_h0 = np.where(obs_d == 1, P1_H0, 1 - P1_H0)
        lik_h1 = np.where(obs_d == 1, P1_H1, 1 - P1_H1)
        unnorm_h0 = lik_h0 * prior_h0
        unnorm_h1 = lik_h1 * (1 - prior_h0)
        prior_h0 = unnorm_h0 / (unnorm_h0 + unnorm_h1)
        prior_h0 = np.clip(prior_h0, 1e-15, 1 - 1e-15)
        ever_spiraled |= (prior_h0 >= (1 - EPSILON))

        if alpha > 0:
            confirmed = (obs_d == h_star)
            sat = np.where(confirmed, 1.0, -1.0)
            pi = np.clip(pi + alpha * sat, 0.0, 1.0)

    rate = float(ever_spiraled.mean())
    n_spiral = int(ever_spiraled.sum())

    z = 1.96
    n = n_sims
    p_hat = rate
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    ci_low = max(0.0, float(centre - margin))
    ci_high = min(1.0, float(centre + margin))

    return {
        "rate": rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_spiral": n_spiral,
        "final_pi": pi.tolist(),
    }


def main():
    print("=== Main results with 95% Wilson CIs (naive user, hallucinating bot) ===")
    print(f"{'pi_init':>7} | {'Fixed rate':>10} {'[95% CI]':>18} | {'Coupled rate':>12} {'[95% CI]':>18} | {'p-value':>10}")
    print("-" * 95)

    pi_test = [round(x * 0.1, 1) for x in range(11)]
    alpha = 0.02
    summary = {"main_results": {}, "final_pi_distributions": {}}

    for pi_init in pi_test:
        fixed = run_with_ci(pi_init, 0.0)
        coupled = run_with_ci(pi_init, alpha)
        table = [
            [fixed["n_spiral"], N_SIMS - fixed["n_spiral"]],
            [coupled["n_spiral"], N_SIMS - coupled["n_spiral"]],
        ]
        _, p_val = stats.fisher_exact(table)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(
            f"{pi_init:>7.1f} | {fixed['rate']:>10.4f} [{fixed['ci_low']:.4f}, {fixed['ci_high']:.4f}] | "
            f"{coupled['rate']:>12.4f} [{coupled['ci_low']:.4f}, {coupled['ci_high']:.4f}] | {p_val:>8.2e} {sig}"
        )
        summary["main_results"][str(pi_init)] = {
            "fixed": {k: v for k, v in fixed.items() if k != "final_pi"},
            "coupled": {k: v for k, v in coupled.items() if k != "final_pi"},
            "fisher_p_value": float(p_val),
            "significance": sig,
        }
        summary["final_pi_distributions"][str(pi_init)] = coupled["final_pi"]

    save_json(summary, "confidence_intervals.json")
    print("\nDone. Results saved.")


if __name__ == "__main__":
    main()
