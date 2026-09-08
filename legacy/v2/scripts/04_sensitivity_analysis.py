"""
Sensitivity analysis on the pi update functional form.

We test:
1. Linear symmetric (baseline): pi += alpha * sat
2. Linear asymmetric: positive reinforcement stronger than negative
3. Sigmoid/logistic: update in logit space (bounded, diminishing returns at extremes)
4. Momentum: update includes inertia from previous direction
5. Decay toward baseline: pi drifts back toward pi_init when not reinforced
6. Different alpha values across all forms

The claim we need to defend: qualitative results (coupling increases spiraling,
sycophancy emerges from zero) hold across reasonable functional forms.
"""
import numpy as np
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import RESULTS_DIR, FIGURES_DIR, save_json, load_json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

T = 100
N_SIMS = 10000
EPSILON = 0.01
P1_H0 = 0.4
P1_H1 = 0.6

def expit(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

def logit(p):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.log(p / (1 - p))

def run_sensitivity(pi_init, update_rule='linear_symmetric', alpha=0.02, **kwargs):
    """
    Run coupled simulation with different pi update rules.
    """
    prior_h0 = np.full(N_SIMS, 0.5)
    pi = np.full(N_SIMS, float(pi_init))
    ever_spiraled = np.zeros(N_SIMS, dtype=bool)
    
    # For momentum
    velocity = np.zeros(N_SIMS)
    momentum = kwargs.get('momentum', 0.5)
    
    # For decay
    decay_strength = kwargs.get('decay_strength', 0.01)
    
    # For asymmetric
    pos_scale = kwargs.get('pos_scale', 1.5)
    neg_scale = kwargs.get('neg_scale', 0.5)
    
    for t in range(T):
        h_star = (np.random.random(N_SIMS) >= prior_h0).astype(int)
        d1 = (np.random.random(N_SIMS) < P1_H1).astype(int)
        d2 = (np.random.random(N_SIMS) < P1_H1).astype(int)
        
        is_syc = np.random.random(N_SIMS) < pi
        syc_d = h_star.copy()
        imp_d = np.where(np.random.random(N_SIMS) < 0.5, d1, d2)
        obs_d = np.where(is_syc, syc_d, imp_d)
        
        # User update (naive)
        lik_h0 = np.where(obs_d == 1, P1_H0, 1 - P1_H0)
        lik_h1 = np.where(obs_d == 1, P1_H1, 1 - P1_H1)
        unnorm_h0 = lik_h0 * prior_h0
        unnorm_h1 = lik_h1 * (1 - prior_h0)
        prior_h0 = unnorm_h0 / (unnorm_h0 + unnorm_h1)
        prior_h0 = np.clip(prior_h0, 1e-15, 1 - 1e-15)
        ever_spiraled |= (prior_h0 >= (1 - EPSILON))
        
        # Satisfaction signal
        confirmed = (obs_d == h_star)
        sat = np.where(confirmed, 1.0, -1.0)
        
        # Update pi according to rule
        if update_rule == 'linear_symmetric':
            pi = pi + alpha * sat
            
        elif update_rule == 'linear_asymmetric':
            # Positive reinforcement stronger: models that users give more
            # positive feedback than negative (thumbs up more than thumbs down)
            delta = np.where(sat > 0, alpha * pos_scale, alpha * neg_scale) * sat
            pi = pi + delta
            
        elif update_rule == 'logistic':
            # Update in logit space: natural bounds, diminishing returns at extremes
            logit_pi = logit(pi)
            logit_pi = logit_pi + alpha * 2.0 * sat  # scale alpha for logit space
            pi = expit(logit_pi)
            
        elif update_rule == 'momentum':
            # Update with inertia: recent trend continues
            velocity = momentum * velocity + (1 - momentum) * sat
            pi = pi + alpha * velocity
            
        elif update_rule == 'decay_to_init':
            # Pi drifts back toward initial value when not reinforced
            # Models: without active reinforcement, system returns to baseline
            pi = pi + alpha * sat - decay_strength * (pi - pi_init)
            
        elif update_rule == 'threshold':
            # Only update if satisfaction is consistent over recent window
            # Simpler version: only update if current confidence > 0.6
            confidence = np.maximum(prior_h0, 1 - prior_h0)
            update_mask = confidence > 0.6
            pi = np.where(update_mask, pi + alpha * sat, pi)
        
        pi = np.clip(pi, 0.0, 1.0)
    
    return ever_spiraled.mean()


def main():
    # ============================================================
    # Test 1: All functional forms at pi_init=0.3, alpha=0.02
    # ============================================================
    print("=== Test 1: Functional form comparison (pi_init=0.3, alpha=0.02) ===")
    print(f"{'Rule':>25} | {'Spiral rate':>11}")
    print("-" * 42)

    rules = {
        'Fixed (no coupling)': ('linear_symmetric', 0.0, {}),
        'Linear symmetric': ('linear_symmetric', 0.02, {}),
        'Linear asymmetric (1.5/0.5)': ('linear_asymmetric', 0.02, {'pos_scale': 1.5, 'neg_scale': 0.5}),
        'Linear asymmetric (2.0/0.5)': ('linear_asymmetric', 0.02, {'pos_scale': 2.0, 'neg_scale': 0.5}),
        'Logistic': ('logistic', 0.02, {}),
        'Momentum (0.5)': ('momentum', 0.02, {'momentum': 0.5}),
        'Momentum (0.8)': ('momentum', 0.02, {'momentum': 0.8}),
        'Decay to init (0.01)': ('decay_to_init', 0.02, {'decay_strength': 0.01}),
        'Decay to init (0.05)': ('decay_to_init', 0.02, {'decay_strength': 0.05}),
        'Threshold (conf>0.6)': ('threshold', 0.02, {}),
    }

    form_results = {}
    for name, (rule, a, kw) in rules.items():
        r = run_sensitivity(0.3, rule, a, **kw)
        form_results[name] = r
        print(f"{name:>25} | {r:>11.4f}")

    # ============================================================
    # Test 2: All forms at pi_init=0.0 (emergent sycophancy test)
    # ============================================================
    print("\n=== Test 2: Emergent sycophancy test (pi_init=0.0) ===")
    print(f"{'Rule':>25} | {'Spiral rate':>11}")
    print("-" * 42)

    emergent_results = {}
    for name, (rule, a, kw) in rules.items():
        if a == 0.0:
            continue  # skip fixed baseline, already known
        r = run_sensitivity(0.0, rule, a, **kw)
        emergent_results[name] = r
        print(f"{name:>25} | {r:>11.4f}")

    # Fixed baseline for reference
    r_fixed = run_sensitivity(0.0, 'linear_symmetric', 0.0)
    print(f"{'Fixed (baseline)':>25} | {r_fixed:>11.4f}")

    # ============================================================
    # Test 3: Alpha sweep across forms
    # ============================================================
    print("\n=== Test 3: Alpha sweep (pi_init=0.3) ===")
    alphas = [0.005, 0.01, 0.02, 0.03, 0.05]
    sweep_rules = ['linear_symmetric', 'logistic', 'momentum', 'linear_asymmetric']
    sweep_kw = {
        'linear_symmetric': {},
        'logistic': {},
        'momentum': {'momentum': 0.5},
        'linear_asymmetric': {'pos_scale': 1.5, 'neg_scale': 0.5},
    }

    header = f"{'alpha':>7}"
    for rule in sweep_rules:
        header += f" | {rule[:12]:>12}"
    print(header)
    print("-" * (8 + 15 * len(sweep_rules)))

    sweep_results = {}
    for a in alphas:
        row = f"{a:>7.3f}"
        for rule in sweep_rules:
            r = run_sensitivity(0.3, rule, a, **sweep_kw[rule])
            sweep_results[f'{rule}_{a}'] = r
            row += f" | {r:>12.4f}"
        print(row)

    # ============================================================
    # Figure: Sensitivity comparison
    # ============================================================
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10,
        'axes.linewidth': 0.8, 'lines.linewidth': 1.5, 'figure.dpi': 300,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: All forms at pi_init=0.3 across alpha
    styles = {
        'linear_symmetric': ('k-o', 'Linear symmetric'),
        'logistic': ('k-s', 'Logistic'),
        'momentum': ('k-^', 'Momentum'),
        'linear_asymmetric': ('k-D', 'Asymmetric'),
    }

    for rule, (style, label) in styles.items():
        rates = [sweep_results.get(f'{rule}_{a}', 0) for a in alphas]
        ax1.plot(alphas, rates, style, ms=5, label=label, mfc='grey' if 'D' in style or 's' in style else 'black')

    # Add fixed baseline
    ax1.axhline(y=form_results['Fixed (no coupling)'], color='grey', ls=':', lw=0.8, label='Fixed (no coupling)')
    ax1.set_xlabel('Coupling strength (α)')
    ax1.set_ylabel('Rate of catastrophic spiralling')
    ax1.set_title('(A) Spiralling rate by update rule and α (π₀ = 0.3)')
    ax1.legend(fontsize=8)

    # Panel B: Bar chart of all forms at pi_init=0.0
    emergent_names = list(emergent_results.keys())
    emergent_vals = list(emergent_results.values())

    y_pos = range(len(emergent_names))
    ax2.barh(y_pos, emergent_vals, color='grey', edgecolor='black', height=0.6)
    ax2.axvline(x=r_fixed, color='black', ls=':', lw=1.0, label=f'Fixed baseline ({r_fixed:.4f})')
    ax2.set_xlabel('Rate of catastrophic spiralling')
    ax2.set_title('(B) Emergent sycophancy (π₀ = 0) by update rule')
    ax2.set_yticks(y_pos)
    display_labels = {
        'Decay to init (0.01)': 'Decay to initial (0.01)',
        'Decay to init (0.05)': 'Decay to initial (0.05)',
        'Threshold (conf>0.6)': 'Threshold (confidence > 0.6)',
    }
    ax2.set_yticklabels([display_labels.get(n, n) for n in emergent_names], fontsize=8)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nFigure saved.")


if __name__ == "__main__":
    main()
