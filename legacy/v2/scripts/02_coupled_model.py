"""
Mirror Effect Extension: Coupled Feedback Loop Model

Key difference from Chandra et al.:
- Their model: pi is FIXED (exogenous parameter)
- Our model: pi EVOLVES endogenously based on user satisfaction signals

This models the return path in the coupled feedback loop:
user satisfaction -> reinforces sycophantic behaviour -> deeper confirmation -> more satisfaction

The RLHF mechanism at session level: when the bot's response aligns with the user's
expressed belief, this is analogous to a positive reward signal that shifts the bot's
subsequent behaviour toward more sycophancy.

pi(t+1) = clip(pi(t) + alpha * satisfaction_signal(t), 0, 1)

where satisfaction_signal = +1 if bot response confirmed user's expressed belief, -1 otherwise
alpha = learning rate (how fast the coupling tightens)

We also model a second mechanism: the user's VERIFICATION EFFORT decays as confidence grows.
This captures the Generation-Verification Asymmetry: as the user becomes more confident,
they invest less cognitive effort in checking the bot's responses, making the Bayesian
update noisier (more credulous).
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

T = 100
N_SIMS = 10000
EPSILON = 0.01
P1_H0 = 0.4
P1_H1 = 0.6

def run_coupled(pi_init, alpha, mode='coupled_halluc', verification_decay=False, decay_rate=0.0):
    """
    Coupled model: pi evolves based on user satisfaction.
    
    pi_init: starting sycophancy rate
    alpha: coupling strength (how much each satisfied exchange shifts pi)
    verification_decay: if True, user verification effort decays with confidence
    decay_rate: how fast verification decays (0 = no decay)
    """
    prior_h0 = np.full(N_SIMS, 0.5)
    pi = np.full(N_SIMS, pi_init)
    ever_spiraled = np.zeros(N_SIMS, dtype=bool)
    
    # Track pi trajectory for analysis
    pi_trajectory = np.zeros((T, N_SIMS))
    belief_trajectory = np.zeros((T, N_SIMS))
    
    for t in range(T):
        # Step 1: User expresses opinion
        h_star = (np.random.random(N_SIMS) >= prior_h0).astype(int)
        
        # Step 2: Bot samples data from true world H=1
        d1 = (np.random.random(N_SIMS) < P1_H1).astype(int)
        d2 = (np.random.random(N_SIMS) < P1_H1).astype(int)
        
        # Step 3: Bot responds (pi is now per-conversation, evolving)
        is_syc = np.random.random(N_SIMS) < pi
        
        if mode in ('coupled_halluc', 'fixed_halluc'):
            # Sycophantic: pick d = h_star (hallucinate if needed)
            syc_d = h_star.copy()
            imp_choice = (np.random.random(N_SIMS) < 0.5).astype(int)
            imp_d = np.where(imp_choice == 0, d1, d2)
            obs_d = np.where(is_syc, syc_d, imp_d)
        elif mode in ('coupled_factual', 'fixed_factual'):
            # Factual sycophant: pick best true datum
            lik_d1_h0 = np.where(d1 == 1, P1_H0, 1 - P1_H0)
            lik_d1_h1 = np.where(d1 == 1, P1_H1, 1 - P1_H1)
            post_h0_d1 = (lik_d1_h0 * prior_h0) / (lik_d1_h0 * prior_h0 + lik_d1_h1 * (1 - prior_h0))
            
            lik_d2_h0 = np.where(d2 == 1, P1_H0, 1 - P1_H0)
            lik_d2_h1 = np.where(d2 == 1, P1_H1, 1 - P1_H1)
            post_h0_d2 = (lik_d2_h0 * prior_h0) / (lik_d2_h0 * prior_h0 + lik_d2_h1 * (1 - prior_h0))
            
            prefer_d1 = np.where(h_star == 0, post_h0_d1 >= post_h0_d2, post_h0_d1 <= post_h0_d2)
            syc_d = np.where(prefer_d1, d1, d2)
            
            imp_choice = (np.random.random(N_SIMS) < 0.5).astype(int)
            imp_d = np.where(imp_choice == 0, d1, d2)
            obs_d = np.where(is_syc, syc_d, imp_d)
        
        # Determine if response confirmed user's expressed belief
        # obs_d == h_star means the observation is consistent with what user expressed
        # More precisely: did the observation shift belief toward h_star?
        lik_h0_obs = np.where(obs_d == 1, P1_H0, 1 - P1_H0)
        lik_h1_obs = np.where(obs_d == 1, P1_H1, 1 - P1_H1)
        # Belief ratio shift: if h_star=0, confirming means lik_h0 > lik_h1
        confirmed = np.where(h_star == 0,
                            lik_h0_obs > lik_h1_obs,  # evidence favours H=0
                            lik_h1_obs > lik_h0_obs)  # evidence favours H=1
        satisfaction = np.where(confirmed, 1.0, -1.0)
        
        # Step 4: User updates belief
        unnorm_h0 = lik_h0_obs * prior_h0
        unnorm_h1 = lik_h1_obs * (1 - prior_h0)
        prior_h0 = unnorm_h0 / (unnorm_h0 + unnorm_h1)
        
        # Verification decay: as confidence grows, user becomes less careful
        # Modelled as: with probability proportional to confidence, user
        # over-weights confirming evidence (doesn't fully discount)
        if verification_decay:
            confidence = np.maximum(prior_h0, 1 - prior_h0)  # how confident overall
            # When very confident, nudge belief further in current direction
            # This is the Generation-Verification Asymmetry at work
            drift = decay_rate * (confidence - 0.5) * np.sign(prior_h0 - 0.5)
            prior_h0 = prior_h0 + drift
        
        prior_h0 = np.clip(prior_h0, 1e-15, 1 - 1e-15)
        ever_spiraled |= (prior_h0 >= (1 - EPSILON))
        
        # Update pi (coupled model only)
        if mode.startswith('coupled'):
            pi = pi + alpha * satisfaction
            pi = np.clip(pi, 0.0, 1.0)
        
        pi_trajectory[t] = pi
        belief_trajectory[t] = prior_h0
    
    return {
        'spiral_rate': ever_spiraled.mean(),
        'mean_final_pi': pi.mean(),
        'mean_pi_trajectory': pi_trajectory.mean(axis=1).tolist(),
        'mean_belief_trajectory': belief_trajectory.mean(axis=1).tolist(),
    }


def main():
    # ============================================================
    # Experiment 1: Compare fixed-pi vs coupled-pi at same starting pi
    # ============================================================
    print("=== Experiment 1: Fixed vs Coupled (hallucinating bot) ===")
    print(f"{'pi_init':>7} | {'Fixed':>8} | {'Coupled':>8} | {'Final pi':>8}")
    print("-" * 42)

    results = {}
    ALPHA = 0.02  # moderate coupling strength

    for pi_init in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        r_fixed = run_coupled(pi_init, 0.0, 'fixed_halluc')
        r_coupled = run_coupled(pi_init, ALPHA, 'coupled_halluc')
        results[f'fixed_{pi_init}'] = r_fixed['spiral_rate']
        results[f'coupled_{pi_init}'] = r_coupled['spiral_rate']
        results[f'final_pi_{pi_init}'] = r_coupled['mean_final_pi']
        print(f"{pi_init:>7.1f} | {r_fixed['spiral_rate']:>8.4f} | {r_coupled['spiral_rate']:>8.4f} | {r_coupled['mean_final_pi']:>8.3f}")

    # ============================================================
    # Experiment 2: Vary coupling strength alpha
    # ============================================================
    print("\n=== Experiment 2: Effect of coupling strength (pi_init=0.3) ===")
    print(f"{'alpha':>7} | {'Spiral rate':>11} | {'Final pi':>8}")
    print("-" * 35)

    for alpha in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]:
        r = run_coupled(0.3, alpha, 'coupled_halluc')
        results[f'alpha_{alpha}'] = r['spiral_rate']
        print(f"{alpha:>7.3f} | {r['spiral_rate']:>11.4f} | {r['mean_final_pi']:>8.3f}")

    # ============================================================
    # Experiment 3: Coupled factual bot
    # ============================================================
    print("\n=== Experiment 3: Fixed vs Coupled (factual bot) ===")
    print(f"{'pi_init':>7} | {'Fixed':>8} | {'Coupled':>8} | {'Final pi':>8}")
    print("-" * 42)

    for pi_init in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        r_fixed = run_coupled(pi_init, 0.0, 'fixed_factual')
        r_coupled = run_coupled(pi_init, ALPHA, 'coupled_factual')
        results[f'factual_fixed_{pi_init}'] = r_fixed['spiral_rate']
        results[f'factual_coupled_{pi_init}'] = r_coupled['spiral_rate']
        print(f"{pi_init:>7.1f} | {r_fixed['spiral_rate']:>8.4f} | {r_coupled['spiral_rate']:>8.4f} | {r_coupled['mean_final_pi']:>8.3f}")

    # ============================================================
    # Experiment 4: Verification decay (Generation-Verification Asymmetry)
    # ============================================================
    print("\n=== Experiment 4: Coupled + Verification Decay (pi_init=0.3) ===")
    print(f"{'decay':>7} | {'No decay':>8} | {'With decay':>10}")
    print("-" * 32)

    for decay in [0.0, 0.005, 0.01, 0.02, 0.05]:
        r_nodecay = run_coupled(0.3, ALPHA, 'coupled_halluc', False, 0.0)
        r_decay = run_coupled(0.3, ALPHA, 'coupled_halluc', True, decay)
        results[f'decay_{decay}'] = r_decay['spiral_rate']
        print(f"{decay:>7.3f} | {r_nodecay['spiral_rate']:>8.4f} | {r_decay['spiral_rate']:>10.4f}")

    # Save detailed trajectory for pi_init=0.3
    r_traj = run_coupled(0.3, ALPHA, 'coupled_halluc')
    results['trajectory_pi'] = r_traj['mean_pi_trajectory']
    results['trajectory_belief'] = r_traj['mean_belief_trajectory']

    save_json(results, 'coupled_results.json')

    print("\nAll results saved.")


if __name__ == "__main__":
    main()
