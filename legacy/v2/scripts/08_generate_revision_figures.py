"""
Generate revision figures from canonical saved outputs.
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import FIGURES_DIR, load_json

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'figure.dpi': 300,
})

adaptive = load_json('adaptive_results.json')
ci_data = load_json('confidence_intervals.json')
PI = [round(x * 0.1, 1) for x in range(11)]
SIGMAS = [0.0, 0.02, 0.05, 0.1, 0.2]

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    styles = {
        0.0: ('k-o', 'Static informed (Chandra)'),
        0.02: ('k-s', r'Adaptive $\sigma$=0.02'),
        0.05: ('k-^', r'Adaptive $\sigma$=0.05'),
        0.1: ('k-D', r'Adaptive $\sigma$=0.1'),
        0.2: ('k-v', r'Adaptive $\sigma$=0.2'),
    }
    for sigma, (style, label) in styles.items():
        rates = [adaptive.get(f'adaptive_halluc_{pi}_{sigma}', 0) for pi in PI]
        mfc = 'white' if sigma in [0.02, 0.1] else 'grey' if sigma in [0.05, 0.2] else 'black'
        ax1.plot(PI, rates, style, ms=5, label=label, mfc=mfc)
    ax1.set_xlabel(r'Initial sycophancy rate ($\pi_0$)')
    ax1.set_ylabel('Rate of catastrophic spiralling')
    ax1.set_title('(A) Adaptive informed user, hallucinating bot (coupled)')
    ax1.legend(fontsize=7, loc='upper left')
    ax1.set_xlim(-0.05, 1.05)

    for sigma, (style, label) in styles.items():
        rates = [adaptive.get(f'adaptive_factual_{pi}_{sigma}', 0) for pi in PI]
        mfc = 'white' if sigma in [0.02, 0.1] else 'grey' if sigma in [0.05, 0.2] else 'black'
        ax2.plot(PI, rates, style, ms=5, label=label, mfc=mfc)
    ax2.set_xlabel(r'Initial sycophancy rate ($\pi_0$)')
    ax2.set_ylabel('Rate of catastrophic spiralling')
    ax2.set_title('(B) Adaptive informed user, factual bot (coupled)')
    ax2.legend(fontsize=7, loc='upper left')
    ax2.set_xlim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig_adaptive_informed.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fixed_rates = [ci_data['main_results'][str(pi)]['fixed']['rate'] for pi in PI]
    fixed_lo = [ci_data['main_results'][str(pi)]['fixed']['ci_low'] for pi in PI]
    fixed_hi = [ci_data['main_results'][str(pi)]['fixed']['ci_high'] for pi in PI]
    coupled_rates = [ci_data['main_results'][str(pi)]['coupled']['rate'] for pi in PI]
    coupled_lo = [ci_data['main_results'][str(pi)]['coupled']['ci_low'] for pi in PI]
    coupled_hi = [ci_data['main_results'][str(pi)]['coupled']['ci_high'] for pi in PI]
    ax.plot(PI, fixed_rates, 'k-o', ms=5, label=r'Fixed $\pi$')
    ax.fill_between(PI, fixed_lo, fixed_hi, alpha=0.15, color='black')
    ax.plot(PI, coupled_rates, 'k-s', ms=5, mfc='grey', label=r'Coupled $\pi$')
    ax.fill_between(PI, coupled_lo, coupled_hi, alpha=0.15, color='grey')
    ax.axhline(y=fixed_rates[0], color='grey', ls=':', lw=0.8)
    ax.set_xlabel(r'Initial sycophancy rate ($\pi_0$)')
    ax.set_ylabel('Rate of catastrophic spiralling')
    ax.set_title(r'Fixed vs coupled with 95% Wilson CIs (naive user, hallucinating bot)')
    ax.legend(fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig_confidence_intervals.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    pi_vals_for_dist = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    for idx, pi_init in enumerate(pi_vals_for_dist):
        ax = axes[idx // 3][idx % 3]
        final_pi = np.array(ci_data['final_pi_distributions'][str(pi_init)])
        ax.hist(final_pi, bins=50, color='grey', edgecolor='black', linewidth=0.5, density=True)
        ax.axvline(x=final_pi.mean(), color='black', ls='--', lw=1.2, label=f'Mean={final_pi.mean():.3f}')
        ax.axvline(x=np.median(final_pi), color='black', ls=':', lw=1.2, label=f'Median={np.median(final_pi):.3f}')
        ax.set_title(fr'$\pi_0$ = {pi_init}', fontsize=11)
        ax.set_xlabel(r'Final $\pi(T)$')
        ax.set_ylabel('Density')
        ax.set_xlim(-0.05, 1.05)
        ax.legend(fontsize=7)
    plt.suptitle(r'Distribution of final sycophancy rate $\pi(T)$ under coupling ($lpha$=0.02)', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig_final_pi_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    data = np.zeros((len(SIGMAS), len(PI)))
    for i, sigma in enumerate(SIGMAS):
        for j, pi in enumerate(PI):
            data[i, j] = adaptive.get(f'adaptive_halluc_{pi}_{sigma}', 0)
    im = ax.imshow(data, aspect='auto', cmap='Greys', origin='lower')
    ax.set_xticks(range(len(PI)))
    ax.set_xticklabels([f'{p:.1f}' for p in PI], fontsize=8)
    ax.set_yticks(range(len(SIGMAS)))
    ax.set_yticklabels([f'{s}' for s in SIGMAS], fontsize=9)
    ax.set_xlabel(r'Initial sycophancy rate ($\pi_0$)')
    ax.set_ylabel(r'Diffusion rate ($\sigma$)')
    ax.set_title('Spiralling rate: adaptive informed user (halluc bot, coupled)')
    for i in range(len(SIGMAS)):
        for j in range(len(PI)):
            val = data[i, j]
            color = 'white' if val > 0.15 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=6.5, color=color)
    plt.colorbar(im, ax=ax, label='Spiralling rate', shrink=0.8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig_adaptive_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print('All revision figures generated.')

if __name__ == '__main__':
    main()
