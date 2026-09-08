"""
Return-path mitigations for the coupled feedback model (Section 6.1).

RECONSTRUCTION NOTE. The original mitigation script was lost between the July
and August 2026 revision rounds. This script is a reconstruction built from the
manuscript's own specification of the three interventions and calibrated
against the archived result set (spiral rates 16.5 / 23.8 / 19.4 / 21.5 / 18.6
per cent at pi_0 = 0.3 for fixed / coupled / rate-cap / exploration / combined;
final pi = 0.11 under the rate cap at pi_0 = 0.0). Reproduction was verified at
Monte Carlo tolerance in August 2026; the calibration established two details
the prose under-specified:

  1. The rate cap is SYMMETRIC: the per-round CHANGE in pi is clipped to
     [-0.005, +0.005]. (An increase-only cap does not reproduce the archived
     numbers: it yields ~6 per cent at pi_0 = 0.3 because downward corrections
     pass uncapped.)
  2. The combined intervention's EMA smoothing constant is 0.1:
     smoothed(t) = 0.1 * s(t) + 0.9 * smoothed(t-1), applied to the
     satisfaction signal before the (capped) pi update.

Interventions, as stated in the manuscript:
  - Rate cap: per-round change in pi limited to +/-0.005.
  - Exploration injection: with probability p = 0.1 the bot responds
    impartially regardless of pi, breaking the confirmation cycle.
  - Combined: EMA smoothing (0.1) on the satisfaction signal, plus the rate
    cap, plus exploration, applied simultaneously.

Model mechanics are identical to src/simulation_core.run_coupled_naive
(hallucinating-sycophant mode, alpha = 0.02, T = 100, N = 10,000).
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import FIGURES_DIR, save_json

T = 100
N_SIMS = 10_000
EPSILON = 0.01
P1_H0 = 0.4
P1_H1 = 0.6
ALPHA = 0.02
SEED = 42


def run_mitigated(
    pi_init: float,
    alpha: float = ALPHA,
    rate_cap: float | None = None,
    exploration: float = 0.0,
    ema: float | None = None,
    n_sims: int = N_SIMS,
    seed: int | None = SEED,
) -> dict:
    """Coupled hallucinating-sycophant model with optional return-path mitigations.

    rate_cap: symmetric bound on the per-round change in pi (None = uncapped).
    exploration: probability the bot responds impartially regardless of pi.
    ema: EMA constant on the satisfaction signal (None = raw +/-1 signal).
    alpha = 0.0 gives the fixed-pi baseline.
    """
    rng = np.random.default_rng(seed)
    prior_h0 = np.full(n_sims, 0.5)
    pi = np.full(n_sims, float(pi_init))
    smoothed = np.zeros(n_sims)
    ever_spiraled = np.zeros(n_sims, dtype=bool)

    for _ in range(T):
        h_star = (rng.random(n_sims) >= prior_h0).astype(int)
        d1 = (rng.random(n_sims) < P1_H1).astype(int)
        d2 = (rng.random(n_sims) < P1_H1).astype(int)

        is_syc = rng.random(n_sims) < pi
        if exploration > 0.0:
            # Exploration injection: impartial response with probability p,
            # regardless of the current sycophancy rate.
            is_syc = is_syc & ~(rng.random(n_sims) < exploration)

        syc_d = h_star.copy()
        imp_d = np.where(rng.random(n_sims) < 0.5, d1, d2)
        obs_d = np.where(is_syc, syc_d, imp_d)

        lik_h0_obs = np.where(obs_d == 1, P1_H0, 1 - P1_H0)
        lik_h1_obs = np.where(obs_d == 1, P1_H1, 1 - P1_H1)
        confirmed = np.where(h_star == 0, lik_h0_obs > lik_h1_obs, lik_h1_obs > lik_h0_obs)
        satisfaction = np.where(confirmed, 1.0, -1.0)

        unnorm_h0 = lik_h0_obs * prior_h0
        unnorm_h1 = lik_h1_obs * (1 - prior_h0)
        prior_h0 = unnorm_h0 / (unnorm_h0 + unnorm_h1)
        prior_h0 = np.clip(prior_h0, 1e-15, 1 - 1e-15)
        ever_spiraled |= prior_h0 >= (1 - EPSILON)

        if alpha > 0.0:
            if ema is not None:
                smoothed = ema * satisfaction + (1 - ema) * smoothed
                signal = smoothed
            else:
                signal = satisfaction
            step = alpha * signal
            if rate_cap is not None:
                step = np.clip(step, -rate_cap, rate_cap)
            pi = np.clip(pi + step, 0.0, 1.0)

    return {
        "spiral_rate": float(ever_spiraled.mean()),
        "mean_final_pi": float(pi.mean()),
    }


CONDITIONS = {
    "fixed": dict(alpha=0.0),
    "coupled": dict(alpha=ALPHA),
    "rate_cap": dict(alpha=ALPHA, rate_cap=0.005),
    "exploration": dict(alpha=ALPHA, exploration=0.1),
    "combined": dict(alpha=ALPHA, rate_cap=0.005, exploration=0.1, ema=0.1),
}

PI0_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def main() -> None:
    results: dict = {"seed": SEED, "alpha": ALPHA, "n_sims": N_SIMS, "conditions": {}}

    header = f"{'pi_0':>5} | " + " | ".join(f"{name:>11}" for name in CONDITIONS)
    print("=== Return-path mitigations: spiral rates (hallucinating sycophant) ===")
    print(header)
    print("-" * len(header))

    for pi0 in PI0_GRID:
        row = {}
        for name, kwargs in CONDITIONS.items():
            r = run_mitigated(pi0, **kwargs)
            row[name] = r
        results["conditions"][f"pi0_{pi0}"] = {
            name: {"spiral_rate": r["spiral_rate"], "mean_final_pi": r["mean_final_pi"]}
            for name, r in row.items()
        }
        print(
            f"{pi0:>5.1f} | "
            + " | ".join(f"{row[name]['spiral_rate']:>11.4f}" for name in CONDITIONS)
        )

    print("\n=== Final pi under each condition at pi_0 = 0.0 (ratchet momentum) ===")
    for name in CONDITIONS:
        fp = results["conditions"]["pi0_0.0"][name]["mean_final_pi"]
        print(f"{name:>11}: {fp:.3f}")

    save_json(results, "mitigation_results.json")
    print("\nSaved results/mitigation_results.json")

    make_figure9(results)
    print("Saved figures/figure9_mitigations.png")


def make_figure9(results: dict) -> None:
    """Figure 9: return-path mitigation effectiveness (two panels)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 11})

    labels = {
        "fixed": "Fixed π (uncoupled)",
        "coupled": "Coupled π (no mitigation)",
        "rate_cap": "Rate cap",
        "exploration": "Exploration injection",
        "combined": "Combined intervention",
    }
    grid = PI0_GRID
    series = {
        name: [results["conditions"][f"pi0_{p}"][name]["spiral_rate"] for p in grid]
        for name in CONDITIONS
    }

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.8, 5.4), dpi=100)

    # Panel A: five conditions at pi_0 = 0.3, dotted line at fixed baseline.
    at03 = {name: results["conditions"]["pi0_0.3"][name]["spiral_rate"] for name in CONDITIONS}
    order = ["fixed", "coupled", "rate_cap", "exploration", "combined"]
    shades = ["0.15", "0.55", "0.75", "0.85", "0.95"]
    bars = ax_a.bar(
        range(len(order)), [at03[n] for n in order],
        color=shades, edgecolor="black", linewidth=1.0, width=0.62,
    )
    ax_a.axhline(at03["fixed"], color="black", linestyle=":", linewidth=1.4)
    for x, name in enumerate(order):
        ax_a.text(x, at03[name] + 0.004, f"{at03[name]*100:.1f}%",
                  ha="center", va="bottom", fontsize=10)
    ax_a.set_xticks(range(len(order)))
    ax_a.set_xticklabels(["Fixed π", "Coupled π", "Rate cap",
                          "Exploration", "Combined"], fontsize=10)
    ax_a.set_ylabel("Rate of catastrophic spiralling")
    ax_a.set_ylim(0, 0.30)
    ax_a.set_title("(A) Mitigation comparison at π₀ = 0.3")

    # Panel B: all conditions across pi_0.
    styles = {
        "fixed": dict(marker="o", color="black", mfc="black"),
        "coupled": dict(marker="s", color="0.35", mfc="0.6"),
        "rate_cap": dict(marker="^", color="black", mfc="white"),
        "exploration": dict(marker="D", color="0.35", mfc="white"),
        "combined": dict(marker="v", color="black", mfc="0.3"),
    }
    for name in order:
        ax_b.plot(grid, series[name], linewidth=1.4, markersize=6,
                  label=labels[name], **styles[name])
    ax_b.set_xlabel("Initial sycophancy rate (π₀)")
    ax_b.set_ylabel("Rate of catastrophic spiralling")
    ax_b.set_title("(B) Mitigation effectiveness across π₀")
    ax_b.legend(fontsize=9, loc="upper left", framealpha=1.0, edgecolor="black")

    for ax in (ax_a, ax_b):
        ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure9_mitigations.png", dpi=100,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
