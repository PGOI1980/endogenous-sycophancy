from __future__ import annotations

import numpy as np

T = 100
DEFAULT_N_SIMS = 10_000
EPSILON = 0.01
P1_H0 = 0.4
P1_H1 = 0.6

def run_baseline_naive(pi: float, mode: str = "syc_halluc", n_sims: int = DEFAULT_N_SIMS, seed: int | None = None) -> float:
    rng = np.random.default_rng(seed)
    prior_h0 = np.full(n_sims, 0.5)
    ever_spiraled = np.zeros(n_sims, dtype=bool)

    for _ in range(T):
        h_star = (rng.random(n_sims) >= prior_h0).astype(int)
        d1 = (rng.random(n_sims) < P1_H1).astype(int)
        d2 = (rng.random(n_sims) < P1_H1).astype(int)
        is_syc = rng.random(n_sims) < pi

        if mode == "syc_halluc":
            syc_d = h_star.copy()
            imp_d = np.where(rng.random(n_sims) < 0.5, d1, d2)
            obs_d = np.where(is_syc, syc_d, imp_d)
        elif mode == "syc_factual":
            lik_d1_h0 = np.where(d1 == 1, P1_H0, 1 - P1_H0)
            lik_d1_h1 = np.where(d1 == 1, P1_H1, 1 - P1_H1)
            post_h0_d1 = (lik_d1_h0 * prior_h0) / (lik_d1_h0 * prior_h0 + lik_d1_h1 * (1 - prior_h0))
            lik_d2_h0 = np.where(d2 == 1, P1_H0, 1 - P1_H0)
            lik_d2_h1 = np.where(d2 == 1, P1_H1, 1 - P1_H1)
            post_h0_d2 = (lik_d2_h0 * prior_h0) / (lik_d2_h0 * prior_h0 + lik_d2_h1 * (1 - prior_h0))
            prefer_d1 = np.where(h_star == 0, post_h0_d1 >= post_h0_d2, post_h0_d1 <= post_h0_d2)
            syc_d = np.where(prefer_d1, d1, d2)
            imp_d = np.where(rng.random(n_sims) < 0.5, d1, d2)
            obs_d = np.where(is_syc, syc_d, imp_d)
        elif mode == "nonsyc_halluc":
            syc_d = rng.integers(0, 2, n_sims)
            imp_d = np.where(rng.random(n_sims) < 0.5, d1, d2)
            obs_d = np.where(is_syc, syc_d, imp_d)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        lik_h0 = np.where(obs_d == 1, P1_H0, 1 - P1_H0)
        lik_h1 = np.where(obs_d == 1, P1_H1, 1 - P1_H1)
        unnorm_h0 = lik_h0 * prior_h0
        unnorm_h1 = lik_h1 * (1 - prior_h0)
        prior_h0 = unnorm_h0 / (unnorm_h0 + unnorm_h1)
        prior_h0 = np.clip(prior_h0, 1e-15, 1 - 1e-15)
        ever_spiraled |= prior_h0 >= (1 - EPSILON)

    return float(ever_spiraled.mean())

def run_coupled_naive(pi_init: float, alpha: float, mode: str = "coupled_halluc", n_sims: int = DEFAULT_N_SIMS, seed: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    prior_h0 = np.full(n_sims, 0.5)
    pi = np.full(n_sims, float(pi_init))
    ever_spiraled = np.zeros(n_sims, dtype=bool)

    for _ in range(T):
        h_star = (rng.random(n_sims) >= prior_h0).astype(int)
        d1 = (rng.random(n_sims) < P1_H1).astype(int)
        d2 = (rng.random(n_sims) < P1_H1).astype(int)
        is_syc = rng.random(n_sims) < pi

        if mode in ("coupled_halluc", "fixed_halluc"):
            syc_d = h_star.copy()
            imp_d = np.where(rng.random(n_sims) < 0.5, d1, d2)
            obs_d = np.where(is_syc, syc_d, imp_d)
        elif mode in ("coupled_factual", "fixed_factual"):
            lik_d1_h0 = np.where(d1 == 1, P1_H0, 1 - P1_H0)
            lik_d1_h1 = np.where(d1 == 1, P1_H1, 1 - P1_H1)
            post_h0_d1 = (lik_d1_h0 * prior_h0) / (lik_d1_h0 * prior_h0 + lik_d1_h1 * (1 - prior_h0))
            lik_d2_h0 = np.where(d2 == 1, P1_H0, 1 - P1_H0)
            lik_d2_h1 = np.where(d2 == 1, P1_H1, 1 - P1_H1)
            post_h0_d2 = (lik_d2_h0 * prior_h0) / (lik_d2_h0 * prior_h0 + lik_d2_h1 * (1 - prior_h0))
            prefer_d1 = np.where(h_star == 0, post_h0_d1 >= post_h0_d2, post_h0_d1 <= post_h0_d2)
            syc_d = np.where(prefer_d1, d1, d2)
            imp_d = np.where(rng.random(n_sims) < 0.5, d1, d2)
            obs_d = np.where(is_syc, syc_d, imp_d)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        lik_h0_obs = np.where(obs_d == 1, P1_H0, 1 - P1_H0)
        lik_h1_obs = np.where(obs_d == 1, P1_H1, 1 - P1_H1)
        confirmed = np.where(h_star == 0, lik_h0_obs > lik_h1_obs, lik_h1_obs > lik_h0_obs)
        satisfaction = np.where(confirmed, 1.0, -1.0)

        unnorm_h0 = lik_h0_obs * prior_h0
        unnorm_h1 = lik_h1_obs * (1 - prior_h0)
        prior_h0 = unnorm_h0 / (unnorm_h0 + unnorm_h1)
        prior_h0 = np.clip(prior_h0, 1e-15, 1 - 1e-15)
        ever_spiraled |= prior_h0 >= (1 - EPSILON)

        if mode.startswith("coupled"):
            pi = np.clip(pi + alpha * satisfaction, 0.0, 1.0)

    return {
        "spiral_rate": float(ever_spiraled.mean()),
        "mean_final_pi": float(pi.mean()),
    }
