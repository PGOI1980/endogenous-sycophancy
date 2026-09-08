"""Monte Carlo summaries and tests for paired simulation outcomes."""
import numpy as np
from scipy.stats import binomtest


def summary(run):
    n = len(run.events)
    k = int(run.events.sum())
    p = k/n
    z = 1.959963984540054
    den = 1+z*z/n
    centre = (p+z*z/(2*n))/den
    width = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return dict(n=n, count=k, rate=p, wilson=[max(0.,centre-width),min(1.,centre+width)],
                mean_final_pi=float(run.final_pi.mean()),
                median_final_pi=float(np.median(run.final_pi)),
                n_at_one=int(np.count_nonzero(run.final_pi == 1)))


def paired(a, b):
    """b minus a; exact McNemar test on discordant conversation outcomes."""
    lost = int(np.count_nonzero(a.events & ~b.events))
    gained = int(np.count_nonzero(~a.events & b.events))
    discordant = lost+gained
    p = float(binomtest(gained, discordant).pvalue) if discordant else 1.
    return dict(lost=lost, gained=gained, difference=(gained-lost)/len(a.events), p=p)


def holm(comparisons):
    order = sorted(range(len(comparisons)), key=lambda i:comparisons[i]["p"])
    ceiling = 0.
    for rank, idx in enumerate(order):
        ceiling = max(ceiling, min(1., (len(order)-rank)*comparisons[idx]["p"]))
        comparisons[idx]["p_holm"] = ceiling


def exact_impartial_probability(turns=100):
    """Absorb at log-odds <= -log(99); each datum changes odds by 1.5."""
    mass={0:1.}
    absorbed=0.
    for _ in range(turns):
        next_mass={}
        for score,probability in mass.items():
            for step,chance in [(1,.6),(-1,.4)]:
                target=score+step
                if target <= -12:
                    absorbed += probability*chance
                else:
                    next_mass[target]=next_mass.get(target,0.)+probability*chance
        mass=next_mass
    return absorbed
