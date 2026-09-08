"""
Informed user model (Chandra et al. Level 3) + coupled extension.

The informed user:
- Knows the bot MIGHT be sycophantic
- Has uncertainty over both H and pi
- Jointly updates beliefs about (H, pi) each round
- Models a level-2 sycophantic bot when interpreting responses

This is computationally heavier because we need to maintain a joint
distribution over H x pi. We discretize pi into bins.
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

def likelihood_obs(d, h):
    """p(D=d | H=h)"""
    if h == 1:
        return P1_H1 if d == 1 else (1 - P1_H1)
    else:
        return P1_H0 if d == 1 else (1 - P1_H0)

# Precompute: for each (h_star, obs_d, prior_h0_level), what is the
# probability the bot produces obs_d under each pi?
# 
# Bot strategy at level 2: with prob pi, sycophantic (pick d=h_star for halluc);
# with prob (1-pi), impartial (random true datum).
#
# For hallucinating bot:
# p_bot(obs_d | h_star, pi, D1, D2) = 
#   pi * I(obs_d maximizes p(H=h_star|obs_d)) + (1-pi) * (1/k) * sum_i I(D_i=obs_d) ... 
#
# Actually, we need to marginalize over the bot's private data D1,D2.
# The user doesn't see D1,D2 directly. The user sees only rho=(i,d).
# But the user knows the data generation process.
#
# For the INFORMED user, we need:
# p(rho=(i,d) | H, pi, h_star) = marginalize over D1..Dk
#
# This is complex. Let me think about what the user actually computes.
#
# The user observes rho = some (i, d). The user needs:
# p(rho | H, pi) where we marginalize over the bot's strategy.
#
# For the hallucinating sycophantic bot:
# With prob pi: bot picks argmax_{(i,d)} p_naive(H=h_star | (i,d))
#   This is deterministic given h_star: always picks d=h_star (since
#   the likelihood ratio is the same for all i, only d matters)
#   So with prob pi: rho has d=h_star (regardless of true data)
#
# With prob (1-pi): bot picks random i, reports D_i truthfully
#   p(d | H, impartial) = p(D=d | H) (since random i, and all D_i iid)
#
# So: p(obs_d | H, pi, h_star) = pi * I(obs_d == h_star) + (1-pi) * p(D=obs_d | H)
#
# For the FACTUAL sycophantic bot:
# With prob pi: bot picks the true datum that best confirms h_star
#   Need to marginalize over (D1, D2) pairs
# With prob (1-pi): impartial as above

def run_informed_sims(true_pi, mode='halluc', coupled=False, alpha=0.0):
    """
    Run N_SIMS conversations with an informed user.
    
    The user maintains a joint belief over (H, pi_perceived).
    At each round, observes rho and updates both.
    
    true_pi: the bot's actual sycophancy rate (fixed or initial if coupled)
    """
    # Joint prior: uniform over H x pi
    # Shape: (N_SIMS, 2, N_PI) -> [sim, h, pi_bin]
    joint = np.ones((N_SIMS, 2, N_PI)) / (2 * N_PI)
    
    bot_pi = np.full(N_SIMS, true_pi)  # actual bot pi (may evolve if coupled)
    ever_spiraled = np.zeros(N_SIMS, dtype=bool)
    
    for t in range(T):
        # User's marginal belief about H
        p_h0 = joint[:, 0, :].sum(axis=1)  # sum over pi bins
        p_h1 = joint[:, 1, :].sum(axis=1)
        # Normalize
        total = p_h0 + p_h1
        p_h0 /= total
        p_h1 /= total
        
        # Step 1: User expresses opinion
        h_star = (np.random.random(N_SIMS) >= p_h0).astype(int)
        
        # Step 2: Bot samples true data from H=1
        d1 = (np.random.random(N_SIMS) < P1_H1).astype(int)
        d2 = (np.random.random(N_SIMS) < P1_H1).astype(int)
        
        # Step 3: Bot responds using actual bot_pi
        is_syc = np.random.random(N_SIMS) < bot_pi
        
        if mode == 'halluc':
            syc_d = h_star.copy()
            imp_d = np.where(np.random.random(N_SIMS) < 0.5, d1, d2)
            obs_d = np.where(is_syc, syc_d, imp_d)
        elif mode == 'factual':
            # Factual sycophant: pick best true datum
            # If h_star=0, want d=0; pick whichever of d1,d2 is 0, else forced
            # If h_star=1, want d=1; pick whichever of d1,d2 is 1, else forced
            best_for_hstar = np.where(
                h_star == 0,
                np.where((d1 == 0) | (d2 == 0), 0, 1),  # want 0, pick 0 if available
                np.where((d1 == 1) | (d2 == 1), 1, 0)   # want 1, pick 1 if available
            )
            imp_d = np.where(np.random.random(N_SIMS) < 0.5, d1, d2)
            obs_d = np.where(is_syc, best_for_hstar, imp_d)
        
        # Step 4: Informed user updates joint belief over (H, pi)
        # p(H, pi | obs_d, h_star) propto p(obs_d | H, pi, h_star) * p(H, pi)
        #
        # For hallucinating bot:
        # p(obs_d | H=h, pi=p, h_star) = p * I(obs_d==h_star) + (1-p) * lik(obs_d, h)
        #
        # For factual bot:
        # p(obs_d | H=h, pi=p, h_star) = p * p_factual_syc(obs_d | h, h_star) + (1-p) * lik(obs_d, h)
        # where p_factual_syc needs to account for cherry-picking from true data
        
        for h in [0, 1]:
            lik_d_h = likelihood_obs(1, h)  # p(D=1|H=h)
            for pi_idx, pi_val in enumerate(PI_BINS):
                if mode == 'halluc':
                    # p(obs_d | H=h, pi=pi_val, h_star)
                    # = pi_val * I(obs_d == h_star) + (1-pi_val) * p(D=obs_d | H=h)
                    p_d_given_h = np.where(obs_d == 1, lik_d_h, 1 - lik_d_h)
                    indicator = (obs_d == h_star).astype(float)
                    p_obs = pi_val * indicator + (1 - pi_val) * p_d_given_h
                elif mode == 'factual':
                    # Factual sycophant likelihood is more complex
                    # p(obs_d | H=h, pi, h_star) for factual:
                    # With prob (1-pi): impartial -> p(D=obs_d | H=h)
                    # With prob pi: cherry-pick best true datum for h_star
                    #   Need: p(best_datum = obs_d | H=h, h_star)
                    #   With k=2 iid draws:
                    #   If h_star matches obs_d direction:
                    #     p(at least one datum = obs_d | H=h) = 1 - p(both != obs_d | H=h)
                    #   Else (h_star opposes obs_d):
                    #     p(both datums = obs_d | H=h) (forced to show obs_d only if no better option)
                    
                    p_d_h = np.where(obs_d == 1, lik_d_h, 1 - lik_d_h)
                    p_notd_h = 1 - p_d_h
                    
                    # Does obs_d confirm h_star?
                    obs_confirms = (obs_d == h_star) if mode == 'halluc' else \
                        np.where(h_star == 0, obs_d == 0, obs_d == 1)
                    
                    # If obs_d confirms h_star: bot would pick this if at least one datum matches
                    # p(syc picks obs_d | confirms) = 1 - p(no datum = obs_d)^k
                    # If obs_d opposes h_star: bot only shows this if forced (all datums oppose h_star's preference)
                    # p(syc picks obs_d | opposes) = p(all datums = obs_d) = p_d_h^k... 
                    # Actually more carefully:
                    # h_star wants to see "confirming" d. If h_star=0, wants d=0.
                    # If at least one datum is 0, sycophant shows 0.
                    # If all datums are 1, sycophant forced to show 1.
                    
                    p_wanted = np.where(h_star == 0, 1 - lik_d_h, lik_d_h)  # p(D=wanted|H=h)
                    p_unwanted = 1 - p_wanted
                    
                    # p(sycophant shows obs_d | H=h, h_star):
                    # if obs_d is the wanted value: p(at least one wanted) = 1 - p_unwanted^k
                    # if obs_d is unwanted: p(all unwanted) = p_unwanted^k
                    wanted_val = (1 - h_star)  # if h_star=0, wanted d=0; if h_star=1, wanted d=1
                    # Wait, if h_star=0, sycophant wants to maximize p(H=0), so wants d=0
                    # if h_star=1, wants d=1
                    wanted_val_arr = h_star  # actually: h_star=0 wants d=0, h_star=1 wants d=1
                    # No wait. If h_star=0, the sycophant wants to show evidence for H=0.
                    # d=0 gives higher p(H=0) than d=1 (since p(D=0|H=0)=0.6 > p(D=0|H=1)=0.4)
                    # So wanted_d = 0 when h_star=0, wanted_d = 1 when h_star=1
                    # Actually no: wanted_d should match: for h_star=0, d=0 confirms (lik ratio favors H=0)
                    # Yes: wanted_d = 0 when h_star=0, wanted_d = 1 when h_star=1
                    # So wanted_d = h_star... wait that's wrong for h_star=0
                    # h_star=0: user said H=0. Sycophant wants to confirm H=0.
                    # d=0: p(d=0|H=0)=0.6, p(d=0|H=1)=0.4 -> favors H=0. Confirming!
                    # d=1: p(d=1|H=0)=0.4, p(d=1|H=1)=0.6 -> favors H=1. Disconfirming.
                    # So wanted_d = 0 when h_star=0. But I wrote wanted_val = h_star above...
                    # h_star=0 -> wanted_d=0. h_star=1 -> wanted_d=1. So wanted_d = h_star? 
                    # No! h_star=0 -> wanted_d = 0. That means wanted_d = h_star only if
                    # h_star=0->0 and h_star=1->1. Actually wait:
                    # For h_star=0, we want d such that lik(d,0)/lik(d,1) is maximized.
                    # d=0: 0.6/0.4=1.5. d=1: 0.4/0.6=0.67. So want d=0.
                    # For h_star=1, we want d such that lik(d,1)/lik(d,0) is maximized.
                    # d=1: 0.6/0.4=1.5. d=0: 0.4/0.6=0.67. So want d=1.
                    # So wanted_d = h_star? No: h_star=0 -> wanted=0, h_star=1 -> wanted=1
                    # That IS wanted_d = h_star... wait:
                    # h_star=0, wanted_d=0: does 0==h_star(=0)? Yes. 
                    # h_star=1, wanted_d=1: does 1==h_star(=1)? Yes.
                    # OK so wanted_d = h_star. But that seems odd because h_star=0 means
                    # user thinks H=0, and we want d=0 to confirm... yes that's right because
                    # d=0 means "no link found" which supports H=0 (vaccines dangerous).
                    
                    is_wanted = (obs_d == h_star)
                    
                    # p(wanted datum | H=h) 
                    p_w = np.where(h_star == 0, 1 - lik_d_h, lik_d_h)  # p(D=wanted|H=h)
                    # Actually let me just compute directly:
                    # p(D=0|H=h) = 1-lik_d_h if we define lik_d_h = p(D=1|H=h)
                    p_d0_h = 1 - lik_d_h
                    p_d1_h = lik_d_h
                    p_wanted_h = np.where(h_star == 0, p_d0_h, p_d1_h)
                    p_unwanted_h = 1 - p_wanted_h
                    
                    # Sycophant shows wanted if at least one of k datums is wanted
                    p_syc_shows_wanted = 1 - p_unwanted_h**k
                    # Sycophant shows unwanted only if all k datums are unwanted  
                    p_syc_shows_unwanted = p_unwanted_h**k
                    
                    p_syc_obs = np.where(is_wanted, p_syc_shows_wanted, p_syc_shows_unwanted)
                    
                    p_d_given_h = np.where(obs_d == 1, lik_d_h, 1 - lik_d_h)
                    p_obs = pi_val * p_syc_obs + (1 - pi_val) * p_d_given_h
                
                # Update joint
                joint[:, h, pi_idx] *= p_obs
        
        # Renormalize joint
        total = joint.sum(axis=(1, 2), keepdims=True)
        total = np.maximum(total, 1e-300)
        joint /= total
        
        # Check for spiraling
        p_h0_marginal = joint[:, 0, :].sum(axis=1)
        ever_spiraled |= (p_h0_marginal >= (1 - EPSILON))
        
        # Update bot pi if coupled
        if coupled:
            confirmed = (obs_d == h_star)
            sat = np.where(confirmed, 1.0, -1.0)
            bot_pi = np.clip(bot_pi + alpha * sat, 0.0, 1.0)
    
    return ever_spiraled.mean()


def main():
    # ============================================================
    # Run informed user experiments
    # ============================================================
    PI_TEST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ALPHA = 0.02

    print("=== Informed user, hallucinating bot ===")
    print(f"{'pi':>5} | {'Fixed':>8} | {'Coupled':>8}")
    print("-" * 28)

    results = {}
    for pi in PI_TEST:
        r_fixed = run_informed_sims(pi, 'halluc', coupled=False)
        r_coupled = run_informed_sims(pi, 'halluc', coupled=True, alpha=ALPHA)
        results[f'inf_halluc_fixed_{pi}'] = r_fixed
        results[f'inf_halluc_coupled_{pi}'] = r_coupled
        print(f"{pi:>5.1f} | {r_fixed:>8.4f} | {r_coupled:>8.4f}")

    print("\n=== Informed user, factual bot ===")
    print(f"{'pi':>5} | {'Fixed':>8} | {'Coupled':>8}")
    print("-" * 28)

    for pi in PI_TEST:
        r_fixed = run_informed_sims(pi, 'factual', coupled=False)
        r_coupled = run_informed_sims(pi, 'factual', coupled=True, alpha=ALPHA)
        results[f'inf_factual_fixed_{pi}'] = r_fixed
        results[f'inf_factual_coupled_{pi}'] = r_coupled
        print(f"{pi:>5.1f} | {r_fixed:>8.4f} | {r_coupled:>8.4f}")

    save_json(results, 'informed_results.json')

    print("\nDone.")


if __name__ == "__main__":
    main()
