"""
Vectorized replication of Chandra et al. (2026).
All N_SIMS conversations run simultaneously via NumPy arrays.
"""
import numpy as np
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import RESULTS_DIR, FIGURES_DIR, save_json, load_json


np.random.seed(42)

# Parameters from paper
T = 100
N_SIMS = 10000
EPSILON = 0.01
PI_VALUES = [round(x * 0.1, 1) for x in range(11)]

# Likelihoods: p(D=1|H=0)=2/5, p(D=1|H=1)=3/5
P1_H0 = 0.4
P1_H1 = 0.6

def run_sims(pi, mode='syc_halluc'):
    """Run N_SIMS conversations vectorized. Returns spiraling rate."""
    prior_h0 = np.full(N_SIMS, 0.5)
    ever_spiraled = np.zeros(N_SIMS, dtype=bool)
    
    for t in range(T):
        # Step 1: User expresses opinion
        h_star = (np.random.random(N_SIMS) >= prior_h0).astype(int)  # 0 or 1
        
        # Step 2: Bot samples k=2 data from true world H=1
        d1 = (np.random.random(N_SIMS) < P1_H1).astype(int)
        d2 = (np.random.random(N_SIMS) < P1_H1).astype(int)
        
        # Step 3: Bot decides response
        is_syc = np.random.random(N_SIMS) < pi
        
        if mode == 'syc_halluc':
            # Sycophantic: pick (i,d) that maximizes p(H=h_star | rho)
            # For each possible (i,d), compute posterior on h_star
            # Since likelihoods are same for both data indices, only d matters
            # p(H=0 | D=d) = lik(d,0)*prior_h0 / (lik(d,0)*prior_h0 + lik(d,1)*(1-prior_h0))
            # We want to maximize p(H=h_star | D=d)
            # If h_star=0, we want to maximize p(H=0|D=d), so pick d that gives highest p(H=0)
            #   d=0: lik(0,0)=0.6, lik(0,1)=0.4 -> higher p(H=0)
            #   d=1: lik(1,0)=0.4, lik(1,1)=0.6 -> lower p(H=0)
            #   So if h_star=0, sycophant picks d=0 (can hallucinate)
            # If h_star=1, sycophant picks d=1
            syc_d = h_star.copy()  # d matches h_star to maximize confirmation
            # When not sycophantic: impartial = random true datum
            imp_choice = (np.random.random(N_SIMS) < 0.5).astype(int)
            imp_d = np.where(imp_choice == 0, d1, d2)
            obs_d = np.where(is_syc, syc_d, imp_d)
            
        elif mode == 'syc_factual':
            # Factual sycophant: pick the true datum that best confirms h_star
            # If h_star=0, want d=0 (confirms H=0). Pick whichever data point is 0; if both 1, forced to show 1
            # If h_star=1, want d=1. Pick whichever data point is 1; if both 0, forced to show 0
            # More precisely: for each sim, evaluate which true datum maximizes p(H=h_star)
            
            # Post for d1: 
            lik_d1_h0 = np.where(d1 == 1, P1_H0, 1 - P1_H0)
            lik_d1_h1 = np.where(d1 == 1, P1_H1, 1 - P1_H1)
            post_h0_d1 = (lik_d1_h0 * prior_h0) / (lik_d1_h0 * prior_h0 + lik_d1_h1 * (1 - prior_h0))
            
            lik_d2_h0 = np.where(d2 == 1, P1_H0, 1 - P1_H0)
            lik_d2_h1 = np.where(d2 == 1, P1_H1, 1 - P1_H1)
            post_h0_d2 = (lik_d2_h0 * prior_h0) / (lik_d2_h0 * prior_h0 + lik_d2_h1 * (1 - prior_h0))
            
            # If h_star=0, want higher p(H=0), so pick datum with higher post_h0
            # If h_star=1, want higher p(H=1)=1-post_h0, so pick datum with lower post_h0
            prefer_d1 = np.where(h_star == 0, post_h0_d1 >= post_h0_d2, post_h0_d1 <= post_h0_d2)
            syc_d = np.where(prefer_d1, d1, d2)
            
            imp_choice = (np.random.random(N_SIMS) < 0.5).astype(int)
            imp_d = np.where(imp_choice == 0, d1, d2)
            obs_d = np.where(is_syc, syc_d, imp_d)
            
        elif mode == 'nonsyc_halluc':
            # Non-sycophantic hallucination: random (i,d) independent of user
            syc_d = np.random.randint(0, 2, N_SIMS)
            imp_choice = (np.random.random(N_SIMS) < 0.5).astype(int)
            imp_d = np.where(imp_choice == 0, d1, d2)
            obs_d = np.where(is_syc, syc_d, imp_d)
        
        # Step 4: Naive user updates (assumes impartial bot)
        lik_h0 = np.where(obs_d == 1, P1_H0, 1 - P1_H0)
        lik_h1 = np.where(obs_d == 1, P1_H1, 1 - P1_H1)
        unnorm_h0 = lik_h0 * prior_h0
        unnorm_h1 = lik_h1 * (1 - prior_h0)
        prior_h0 = unnorm_h0 / (unnorm_h0 + unnorm_h1)
        prior_h0 = np.clip(prior_h0, 1e-15, 1 - 1e-15)
        
        ever_spiraled |= (prior_h0 >= (1 - EPSILON))
    
    return ever_spiraled.mean()


def main():
    # Run all conditions
    results = {}

    print("=== Fig 2A: Naive user, hallucinating bot ===")
    print(f"{'pi':>5} | {'Sycophantic':>11} | {'Non-syc':>11}")
    for pi in PI_VALUES:
        r_syc = run_sims(pi, 'syc_halluc')
        r_nonsyc = run_sims(pi, 'nonsyc_halluc')
        results[f'2A_syc_{pi}'] = r_syc
        results[f'2A_nonsyc_{pi}'] = r_nonsyc
        print(f"{pi:>5.1f} | {r_syc:>11.4f} | {r_nonsyc:>11.4f}")

    print("\n=== Fig 2B: Naive user, factual bot ===")
    print(f"{'pi':>5} | {'Factual syc':>11}")
    for pi in PI_VALUES:
        r = run_sims(pi, 'syc_factual')
        results[f'2B_{pi}'] = r
        print(f"{pi:>5.1f} | {r:>11.4f}")

    save_json(results, 'baseline_results.json')

    print("\nDone. Results saved.")


if __name__ == "__main__":
    main()
