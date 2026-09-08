"""One implementation shared by all revised experiments.

All conditions consume six uniform random arrays per turn, in the same order.
Reusing a seed therefore pairs simulated conversations across conditions.
The binary feedback signal records response agreement, not posterior movement.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import convolve1d
from scipy.special import expit, logit


def diffuse(joint, sigma, grid):
    """Reflecting Gaussian diffusion; preserves each simulation's H marginal."""
    if sigma == 0:
        return joint
    dx = grid[1] - grid[0]
    radius = int(np.ceil(4 * sigma / dx))
    offsets = np.arange(-radius, radius + 1) * dx
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    return convolve1d(joint, kernel, axis=-1, mode="reflect")


def observation_likelihood(obs, opinion, propensity, bot):
    """P(d | H, pi, h*), shape (N, 2, M), marginalising private data."""
    p1 = np.array([0.4, 0.6])[None, :, None]
    impartial = np.where(obs[:, None, None] == 1, p1, 1 - p1)
    agreement = (obs == opinion)[:, None, None]
    if bot == "halluc":
        syc = agreement.astype(float)
    elif bot == "factual":
        wanted = np.where(opinion[:, None, None] == 1, p1, 1 - p1)
        syc = np.where(agreement, 1 - (1 - wanted) ** 2, (1 - wanted) ** 2)
    else:
        raise ValueError(bot)
    propensity = np.asarray(propensity)
    if propensity.ndim == 1:
        propensity = propensity[None, None, :]
    elif propensity.ndim == 2:
        propensity = propensity[:, None, :]
    return propensity * syc + (1 - propensity) * impartial


@dataclass
class Run:
    events: np.ndarray
    final_pi: np.ndarray
    first_crossing: np.ndarray
    pi_traces: np.ndarray
    belief_traces: np.ndarray
    schedule: np.ndarray | None = None
    signals: np.ndarray | None = None


def simulate(pi0, *, bot="halluc", learner="naive", alpha=0.02,
             rule="linear", sigma=0.0, bins=21, n=10000, turns=100,
             seed=42, signal="agreement", positive=1.5, negative=0.5,
             momentum=0.5, decay=0.01, confidence_gate=0.6,
             exploration=0.0, cap=None, ema=None, schedule=None,
             external_signals=None, record=False):
    """Run one condition; count any >=99% false-confidence crossing.

Learners: naive; static (fixed unknown pi); diffusion (unknown changing pi);
aware (known alpha and agreement rule, unknown initial pi on a grid);
oracle (known initial pi and known agreement rule).
Schedules and external signals are controls for naive users only.
The aware/oracle likelihood includes an announced exploration probability.
"""
    if learner not in {"naive", "static", "diffusion", "aware", "oracle"}:
        raise ValueError(learner)
    if learner in {"aware", "oracle"} and (rule != "linear" or signal != "agreement" or cap is not None or ema is not None):
        raise ValueError("Rule-aware inference requires the stated linear agreement rule")
    if bot == "random" and learner != "naive":
        raise ValueError("Random hallucination is a naive-user control")
    rng = np.random.default_rng(seed)
    p0 = np.full(n, 0.5)
    pi = np.full(n, float(pi0))
    grid = np.linspace(0, 1, bins)
    if learner == "oracle":
        joint = np.ones((n, 2, 1)) / 2
        candidates = np.full((n, 1), float(pi0))
    else:
        joint = np.ones((n, 2, bins)) / (2 * bins)
        candidates = np.broadcast_to(grid, (n, bins)).copy()
    events = np.zeros(n, dtype=bool)
    crossing = np.full(n, -1, dtype=int)
    velocity = np.zeros(n)
    smooth = np.zeros(n)
    pi_traces = np.zeros((turns + 1, min(15, n)))
    belief_traces = np.zeros_like(pi_traces)
    pi_traces[0] = pi0
    belief_traces[0] = 0.5
    schedules = np.empty((turns, n)) if record else None
    signals = np.empty((turns, n), dtype=np.int8) if record else None
    for t in range(turns):
        # A transition of pi cannot supply evidence about the fixed world state.
        if learner == "diffusion" and t > 0:
            joint = diffuse(joint, sigma, grid)
        if learner != "naive":
            p0 = joint[:, 0].sum(axis=1)
        u = rng.random((6, n))
        opinion = (u[0] >= p0).astype(int)
        d1, d2 = (u[1] < 0.6).astype(int), (u[2] < 0.6).astype(int)
        if schedule is not None:
            pi = np.broadcast_to(schedule[t], (n,)).copy()
        if record:
            schedules[t] = pi
        syc = (u[3] < pi) & (u[5] >= exploration)
        impartial = np.where(u[4] < 0.5, d1, d2)
        if bot == "halluc":
            preferred = opinion
        elif bot == "factual":
            preferred = np.where(d1 == opinion, d1, d2)
        elif bot == "random":
            preferred = (u[5] < 0.5).astype(int)
        else:
            raise ValueError(bot)
        obs = np.where(syc, preferred, impartial)
        agreement = np.where(obs == opinion, 1.0, -1.0)
        before = p0.copy()
        if learner == "naive":
            l0 = np.where(obs == 1, 0.4, 0.6)
            l1 = 1 - l0
            p0 = l0 * p0 / (l0 * p0 + l1 * (1 - p0))
            p0 = np.clip(p0, 1e-15, 1 - 1e-15)
        else:
            perceived = candidates if learner in {"aware", "oracle"} else grid
            lik = observation_likelihood(obs, opinion, (1-exploration)*perceived, bot)
            joint *= lik
            total = joint.sum(axis=(1, 2), keepdims=True)
            if np.any(total <= 0):
                raise FloatingPointError("Observation has zero probability under every hypothesis")
            joint /= total
            p0 = joint[:, 0].sum(axis=1)
        events |= p0 >= 0.99
        if signal == "agreement":
            sat = agreement
        elif signal == "graded":
            sat = np.where(opinion == 0, p0-before, before-p0)
        elif signal == "posterior_sign":
            delta = np.where(opinion == 0, p0-before, before-p0)
            sat = np.where(delta > 0, 1., np.where(delta < 0, -1., 0.))
        else:
            raise ValueError(signal)
        if record:
            signals[t] = agreement
        if external_signals is not None:
            sat = external_signals[t]
        if ema is not None:
            smooth = ema * sat + (1-ema) * smooth
            sat = smooth
        if schedule is None:
            if rule == "linear":
                step = alpha * sat
            elif rule == "asymmetric":
                step = alpha * np.where(sat > 0, positive, negative) * sat
            elif rule == "logistic":
                # Exact endpoints are absorbing; no arbitrary numerical floor.
                inner = (pi > 0) & (pi < 1)
                nxt = pi.copy()
                nxt[inner] = expit(logit(pi[inner]) + 2 * alpha * sat[inner])
                step = nxt-pi
            elif rule == "momentum":
                velocity = momentum * velocity + (1-momentum)*sat
                step = alpha * velocity
            elif rule == "decay":
                step = alpha*sat - decay*(pi-pi0)
            elif rule == "threshold":
                step = alpha*sat*(np.maximum(p0,1-p0) > confidence_gate)
            else:
                raise ValueError(rule)
            if cap is not None:
                step = np.clip(step, -cap, cap)
            pi = np.clip(pi+step, 0., 1.)
        if learner in {"aware", "oracle"}:
            candidates = np.clip(candidates + alpha*agreement[:, None], 0., 1.)
        crossing[(crossing < 0) & (pi >= 0.5)] = t+1
        pi_traces[t+1] = pi[:15]
        belief_traces[t+1] = 1-p0[:15]
    return Run(events, pi, crossing, pi_traces, belief_traces, schedules, signals)
