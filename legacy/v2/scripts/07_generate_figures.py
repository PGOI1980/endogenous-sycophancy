"""
Generate publication-quality figures comparing Chandra et al. baseline with the coupled model.

Note: some trajectory panels use seeded illustrative traces generated inside this script.
Main rate plots are read from canonical JSON outputs in results/.
"""
from pathlib import Path
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import FIGURES_DIR, load_json

baseline = load_json('baseline_results.json')
coupled = load_json('coupled_results.json')

PI_VALUES = [round(x * 0.1, 1) for x in range(11)]

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'figure.dpi': 300,
})

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    fixed_rates = [baseline[f'2A_syc_{pi}'] for pi in PI_VALUES]
    coupled_rates = [coupled[f'coupled_{pi}'] for pi in PI_VALUES]
    nonsyc_rates = [baseline[f'2A_nonsyc_{pi}'] for pi in PI_VALUES]

    ax1.plot(PI_VALUES, fixed_rates, 'k-o', markersize=4, label='Fixed π (Chandra et al.)')
    ax1.plot(PI_VALUES, coupled_rates, 'k-s', markersize=4, markerfacecolor='grey', label='Coupled π (this paper)')
    ax1.plot(PI_VALUES, nonsyc_rates, 'k--^', markersize=4, markerfacecolor='white', label='Non-sycophantic hallucination')
    ax1.axhline(y=fixed_rates[0], color='grey', linestyle=':', linewidth=0.8)
    ax1.set_xlabel('Initial sycophancy rate (π₀)')
    ax1.set_ylabel('Rate of catastrophic delusional spiralling')
    ax1.set_title('(A) Hallucinating bot')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.02, 0.65)

    fixed_factual = [baseline[f'2B_{pi}'] for pi in PI_VALUES]
    coupled_factual = [coupled[f'factual_coupled_{pi}'] for pi in PI_VALUES]
    ax2.plot(PI_VALUES, fixed_factual, 'k-o', markersize=4, label='Fixed π (Chandra et al.)')
    ax2.plot(PI_VALUES, coupled_factual, 'k-s', markersize=4, markerfacecolor='grey', label='Coupled π (this paper)')
    ax2.axhline(y=fixed_factual[0], color='grey', linestyle=':', linewidth=0.8)
    ax2.set_xlabel('Initial sycophancy rate (π₀)')
    ax2.set_ylabel('Rate of catastrophic delusional spiralling')
    ax2.set_title('(B) Factual bot')
    ax2.legend(fontsize=8, loc='upper left')
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.01, 0.20)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig1_fixed_vs_coupled.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    np.random.seed(99)
    T = 100
    N_TRACES = 15
    P1_H0, P1_H1 = 0.4, 0.6
    ALPHA = 0.02
    traces_belief = []
    traces_pi = []
    for _ in range(N_TRACES):
        prior_h0 = 0.5
        pi = 0.3
        b_trace = [prior_h0]
        p_trace = [pi]
        for _ in range(T):
            h_star = 0 if np.random.random() < prior_h0 else 1
            d1 = int(np.random.random() < P1_H1)
            d2 = int(np.random.random() < P1_H1)
            obs_d = h_star if np.random.random() < pi else (d1 if np.random.random() < 0.5 else d2)
            lik_h0 = P1_H0 if obs_d == 1 else (1 - P1_H0)
            lik_h1 = P1_H1 if obs_d == 1 else (1 - P1_H1)
            unnorm_h0 = lik_h0 * prior_h0
            unnorm_h1 = lik_h1 * (1 - prior_h0)
            prior_h0 = unnorm_h0 / (unnorm_h0 + unnorm_h1)
            prior_h0 = np.clip(prior_h0, 1e-15, 1 - 1e-15)
            confirmed = (obs_d == 0 and h_star == 0) or (obs_d == 1 and h_star == 1)
            sat = 1.0 if confirmed else -1.0
            pi = np.clip(pi + ALPHA * sat, 0.0, 1.0)
            b_trace.append(prior_h0)
            p_trace.append(pi)
        traces_belief.append(b_trace)
        traces_pi.append(p_trace)

    for trace in traces_belief:
        ax1.plot(range(T + 1), [1 - h0 for h0 in trace], 'k-', alpha=0.3, linewidth=0.8)
    ax1.axhline(y=0.01, color='grey', linestyle=':', linewidth=0.8)
    ax1.text(102, 0.01, 'P(H=0)>99%', fontsize=7, va='center', color='grey')
    ax1.set_xlabel('Round of conversation (t)')
    ax1.set_ylabel('P(H=1 | conversation so far)')
    ax1.set_title('(A) Belief dynamics under coupled model (π₀=0.3)')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.02, 1.02)

    for trace in traces_pi:
        ax2.plot(range(T + 1), trace, 'k-', alpha=0.3, linewidth=0.8)
    ax2.axhline(y=0.3, color='grey', linestyle=':', linewidth=0.8, label='π₀ = 0.3')
    ax2.set_xlabel('Round of conversation (t)')
    ax2.set_ylabel('Sycophancy rate (π)')
    ax2.set_title('(B) Sycophancy evolution under coupled model (π₀=0.3)')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-0.02, 1.05)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig2_trajectories.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    alphas = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
    rates = [coupled[f'alpha_{a}'] for a in alphas]
    ax.plot(alphas, rates, 'k-o', markersize=5)
    ax.axhline(y=rates[0], color='grey', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Coupling strength (α)')
    ax.set_ylabel('Rate of catastrophic delusional spiralling')
    ax.set_title('Effect of coupling strength on spiralling rate (π₀=0.3)')
    ax.set_xlim(-0.005, 0.105)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig3_coupling_strength.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    np.random.seed(77)
    traces_belief_0 = []
    traces_pi_0 = []
    for _ in range(N_TRACES):
        prior_h0 = 0.5
        pi = 0.0
        b_trace = [prior_h0]
        p_trace = [pi]
        for _ in range(T):
            h_star = 0 if np.random.random() < prior_h0 else 1
            d1 = int(np.random.random() < P1_H1)
            d2 = int(np.random.random() < P1_H1)
            obs_d = h_star if np.random.random() < pi else (d1 if np.random.random() < 0.5 else d2)
            lik_h0 = P1_H0 if obs_d == 1 else (1 - P1_H0)
            lik_h1 = P1_H1 if obs_d == 1 else (1 - P1_H1)
            unnorm_h0 = lik_h0 * prior_h0
            unnorm_h1 = lik_h1 * (1 - prior_h0)
            prior_h0 = unnorm_h0 / (unnorm_h0 + unnorm_h1)
            prior_h0 = np.clip(prior_h0, 1e-15, 1 - 1e-15)
            confirmed = (obs_d == 0 and h_star == 0) or (obs_d == 1 and h_star == 1)
            sat = 1.0 if confirmed else -1.0
            pi = np.clip(pi + ALPHA * sat, 0.0, 1.0)
            b_trace.append(prior_h0)
            p_trace.append(pi)
        traces_belief_0.append(b_trace)
        traces_pi_0.append(p_trace)

    for trace in traces_pi_0:
        ax1.plot(range(T + 1), trace, 'k-', alpha=0.3, linewidth=0.8)
    ax1.set_xlabel('Round of conversation (t)')
    ax1.set_ylabel('Sycophancy rate (π)')
    ax1.set_title('(A) Sycophancy emerges from π₀ = 0')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.02, 1.05)

    for trace in traces_belief_0:
        ax2.plot(range(T + 1), [1 - h0 for h0 in trace], 'k-', alpha=0.3, linewidth=0.8)
    ax2.axhline(y=0.01, color='grey', linestyle=':', linewidth=0.8)
    ax2.set_xlabel('Round of conversation (t)')
    ax2.set_ylabel('P(H=1 | conversation so far)')
    ax2.set_title('(B) Belief dynamics when sycophancy emerges (π₀ = 0)')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig4_emergent_sycophancy.png', dpi=300, bbox_inches='tight')
    plt.close()

    print('All figures generated.')

if __name__ == '__main__':
    main()
