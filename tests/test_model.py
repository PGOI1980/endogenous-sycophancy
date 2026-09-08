import numpy as np
from src.model import diffuse, simulate, observation_likelihood


def test_diffusion_preserves_world_marginal_and_probability():
    rng = np.random.default_rng(1)
    j = rng.random((12,2,21));j /= j.sum(axis=(1,2),keepdims=True)
    for sigma in [.02,.05,.1,.2]:
        d = diffuse(j,sigma,np.linspace(0,1,21))
        np.testing.assert_allclose(d.sum(axis=2),j.sum(axis=2),atol=1e-14)
        assert np.all(d >= 0)


def test_observation_probabilities_sum_to_one():
    for bot in ['halluc','factual']:
        opinion = np.array([0,1])
        probs = sum(observation_likelihood(np.full(2,d),opinion,np.linspace(0,1,21),bot) for d in [0,1])
        np.testing.assert_allclose(probs,1.)


def test_equivalent_settings_have_identical_outcomes():
    opts=dict(n=500,seed=13)
    a=simulate(0.,alpha=0.,bot='halluc',**opts)
    b=simulate(0.,alpha=0.,bot='factual',**opts)
    np.testing.assert_array_equal(a.events,b.events)
    np.testing.assert_array_equal(a.belief_traces,b.belief_traces)
    a=simulate(.3,cap=.005,**opts);b=simulate(.3,alpha=.005,**opts)
    np.testing.assert_array_equal(a.events,b.events)
    np.testing.assert_array_equal(a.final_pi,b.final_pi)


def test_logistic_zero_is_absorbing():
    opts=dict(n=500,seed=9)
    a=simulate(0.,rule='logistic',**opts);b=simulate(0.,alpha=0.,**opts)
    np.testing.assert_array_equal(a.final_pi,0.)
    np.testing.assert_array_equal(a.events,b.events)


def test_zero_diffusion_equals_static():
    a=simulate(.3,learner='static',n=500)
    b=simulate(.3,learner='diffusion',sigma=0.,n=500)
    np.testing.assert_array_equal(a.events,b.events)
    np.testing.assert_allclose(a.belief_traces,b.belief_traces)


def test_oracle_impartial_equals_naive():
    a=simulate(0.,alpha=0.,learner='naive',n=500)
    b=simulate(0.,alpha=0.,learner='oracle',n=500)
    np.testing.assert_array_equal(a.events,b.events)
    np.testing.assert_allclose(a.belief_traces,b.belief_traces,atol=1e-12)
