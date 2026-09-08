"""
Adaptive Informed User: responds to reviewer criticism.

The original informed user maintains a joint posterior over (H, pi),
treating pi as FIXED. Under coupling, pi actually drifts, so the
user's model is misspecified.

The adaptive informed user models pi as potentially nonstationary:
at each round, the posterior over pi undergoes diffusion (random walk
in logit space), spreading the distribution before the new observation
is incorporated. This allows the user to track a drifting pi.

We test: does the mitigation failure persist when the user can
adapt to changing pi?
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
k = 2

# Discretize pi for the user's inference
PI_BINS = np.linspace(0, 1, 21)  # 21 bins from 0 to 1
N_PI = len(PI_BINS)
DELTA_PI = PI_BINS[1] - PI_BINS[0]

def likelihood_obs(d, h):
    if h == 1:
        return P1_H1 if d == 1 else (1 - P1_H1)
    else:
        return P1_H0 if d == 1 else (1 - P1_H0)

def diffuse_pi_posterior(joint, sigma_diffusion):
    """
    Apply random-walk diffusion to the pi dimension of the joint posterior.
    This models the user's belief that pi might change between rounds.
    
    For each H value, convolve the pi-marginal with a discrete Gaussian kernel.
    sigma_diffusion controls how much the user expects pi to drift per round.
    """
    if sigma_diffusion == 0:
        return joint
    
    # Build discrete Gaussian kernel over PI_BINS
    kernel_width = int(4 * sigma_diffusion / DELTA_PI) + 1
    kernel_range = np.arange(-kernel_width, kernel_width + 1)
    kernel = np.exp(-0.5 * (kernel_range * DELTA_PI / sigma_diffusion) ** 2)
    kernel /= kernel.sum()
    
    # Convolve each sim's pi distribution for each H value
    from scipy.ndimage import convolve1d
    for h in range(2):
        # joint[:, h, :] has shape (N_SIMS, N_PI)
        joint[:, h, :] = convolve1d(joint[:, h, :], kernel, axis=1, mode='constant', cval=0)
    
    # Renormalize
    total = joint.sum(axis=(1, 2), keepdims=True)
    total = np.maximum(total, 1e-300)
    joint /= total
    
    return joint

def run_adaptive_informed(true_pi, mode='halluc', coupled=False, alpha=0.0,
                          sigma_diffusion=0.0):
    """
    Run informed user with optional diffusion (adaptive tracking of pi).
    
    sigma_diffusion=0: original static informed user (Chandra et al.)
    sigma_diffusion>0: adaptive informed user who expects pi to drift
    """
    joint = np.ones((N_SIMS, 2, N_PI)) / (2 * N_PI)
    bot_pi = np.full(N_SIMS, true_pi)
    ever_spiraled = np.zeros(N_SIMS, dtype=bool)
    
    for t in range(T):
        # Diffuse pi posterior (adaptive user expects pi may change)
        if sigma_diffusion > 0:
            joint = diffuse_pi_posterior(joint, sigma_diffusion)
        
        p_h0 = joint[:, 0, :].sum(axis=1)
        p_h1 = joint[:, 1, :].sum(axis=1)
        total = p_h0 + p_h1
        p_h0 /= total
        
        h_star = (np.random.random(N_SIMS) >= p_h0).astype(int)
        
        d1 = (np.random.random(N_SIMS) < P1_H1).astype(int)
        d2 = (np.random.random(N_SIMS) < P1_H1).astype(int)
        
        is_syc = np.random.random(N_SIMS) < bot_pi
        
        if mode == 'halluc':
            syc_d = h_star.copy()
            imp_d = np.where(np.random.random(N_SIMS) < 0.5, d1, d2)
            obs_d = np.where(is_syc, syc_d, imp_d)
        elif mode == 'factual':
            p_d0_h = np.array([1 - P1_H0, 1 - P1_H1])
            best_for_hstar = np.where(
                h_star == 0,
                np.where((d1 == 0) | (d2 == 0), 0, 1),
                np.where((d1 == 1) | (d2 == 1), 1, 0)
            )
            imp_d = np.where(np.random.random(N_SIMS) < 0.5, d1, d2)
            obs_d = np.where(is_syc, best_for_hstar, imp_d)
        
        # Update joint posterior
        for h in [0, 1]:
            lik_d_h = likelihood_obs(1, h)
            for pi_idx, pi_val in enumerate(PI_BINS):
                if mode == 'halluc':
                    p_d_given_h = np.where(obs_d == 1, lik_d_h, 1 - lik_d_h)
                    indicator = (obs_d == h_star).astype(float)
                    p_obs = pi_val * indicator + (1 - pi_val) * p_d_given_h
                elif mode == 'factual':
                    p_d1_h = lik_d_h
                    p_d0_h_val = 1 - lik_d_h
                    p_wanted_h = np.where(h_star == 0, p_d0_h_val, p_d1_h)
                    p_unwanted_h = 1 - p_wanted_h
                    is_wanted = (obs_d == h_star)
                    p_syc_shows_wanted = 1 - p_unwanted_h**k
                    p_syc_shows_unwanted = p_unwanted_h**k
                    p_syc_obs = np.where(is_wanted, p_syc_shows_wanted, p_syc_shows_unwanted)
                    p_d_given_h = np.where(obs_d == 1, lik_d_h, 1 - lik_d_h)
                    p_obs = pi_val * p_syc_obs + (1 - pi_val) * p_d_given_h
                
                joint[:, h, pi_idx] *= p_obs
        
        total = joint.sum(axis=(1, 2), keepdims=True)
        total = np.maximum(total, 1e-300)
        joint /= total
        
        p_h0_marginal = joint[:, 0, :].sum(axis=1)
        ever_spiraled |= (p_h0_marginal >= (1 - EPSILON))
        
        if coupled:
            confirmed = (obs_d == h_star)
            sat = np.where(confirmed, 1.0, -1.0)
            bot_pi = np.clip(bot_pi + alpha * sat, 0.0, 1.0)
    
    return ever_spiraled.mean()


def main():
    # ============================================================
    # Main experiments
    # ============================================================
    PI_TEST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ALPHA = 0.02

    # Test multiple diffusion levels
    DIFFUSION_LEVELS = [0.0, 0.02, 0.05, 0.1, 0.2]

    print("=== Experiment: Adaptive vs Static Informed User (hallucinating bot, coupled) ===")
    print(f"{'pi_init':>7} | {'Static':>8} | {'sig=0.02':>8} | {'sig=0.05':>8} | {'sig=0.1':>8} | {'sig=0.2':>8}")
    print("-" * 60)

    results = {}
    for pi in PI_TEST:
        row = f"{pi:>7.1f}"
        for sigma in DIFFUSION_LEVELS:
            label = f"static" if sigma == 0 else f"sig={sigma}"
            r = run_adaptive_informed(pi, 'halluc', coupled=True, alpha=ALPHA,
                                       sigma_diffusion=sigma)
            results[f'adaptive_halluc_{pi}_{sigma}'] = r
            row += f" | {r:>8.4f}"
        print(row)

    print("\n=== Experiment: Adaptive vs Static Informed User (factual bot, coupled) ===")
    print(f"{'pi_init':>7} | {'Static':>8} | {'sig=0.02':>8} | {'sig=0.05':>8} | {'sig=0.1':>8} | {'sig=0.2':>8}")
    print("-" * 60)

    for pi in PI_TEST:
        row = f"{pi:>7.1f}"
        for sigma in DIFFUSION_LEVELS:
            r = run_adaptive_informed(pi, 'factual', coupled=True, alpha=ALPHA,
                                       sigma_diffusion=sigma)
            results[f'adaptive_factual_{pi}_{sigma}'] = r
            row += f" | {r:>8.4f}"
        print(row)

    # Also run the fixed-pi baseline for comparison
    print("\n=== Control: Adaptive Informed User with FIXED pi (no coupling) ===")
    print(f"{'pi_init':>7} | {'Static':>8} | {'sig=0.05':>8} | {'sig=0.1':>8}")
    print("-" * 40)

    for pi in [0.0, 0.1, 0.2, 0.3, 0.5]:
        row = f"{pi:>7.1f}"
        for sigma in [0.0, 0.05, 0.1]:
            r = run_adaptive_informed(pi, 'halluc', coupled=False, alpha=0.0,
                                       sigma_diffusion=sigma)
            results[f'adaptive_fixed_{pi}_{sigma}'] = r
            row += f" | {r:>8.4f}"
        print(row)

    save_json(results, 'adaptive_results.json')

    print("\nDone.")


if __name__ == "__main__":
    main()
