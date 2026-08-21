"""Tests for memory/certification.py -- the paper's exact confidence-bound
and cannot-link construction (arXiv-2605.10870v1, Appendix app:certificates).
Uses hand-seeded ContextActionStats at LARGE n (not n=1): with realistic
project-scale N/A/t values, a single observation gives a confidence radius
> 1.0 (the bound is trivial), so cannot_link can never certify anything at
n=1 -- this is the theory's own "certification exploration" requirement,
not a bug, and these tests exercise it directly."""
import math

from memory.certification import ContextActionStats, cannot_link, confidence_radius, lcb, ucb

ACTIONS = ["ACTION_0", "ACTION_1", "ACTION_2"]
DELTA = 0.05


def _seed(stats: ContextActionStats, x: str, a: str, n: int, mu: float) -> None:
    """Feeds n observations of context x, action a, all with the same
    reward mu (0.0 or 1.0), so the resulting empirical mean is exactly mu."""
    reward = 1.0 if mu >= 0.5 else 0.0
    for _ in range(n):
        stats.update(x, a, reward)


def test_confidence_radius_shrinks_with_more_observations():
    small_n_radius = confidence_radius(n=1, n_contexts=2, n_actions=3, t=100, delta=DELTA)
    large_n_radius = confidence_radius(n=200, n_contexts=2, n_actions=3, t=100, delta=DELTA)
    assert large_n_radius < small_n_radius


def test_confidence_radius_is_inf_for_zero_observations():
    assert confidence_radius(n=0, n_contexts=2, n_actions=3, t=100, delta=DELTA) == float("inf")


def test_ucb_is_one_and_lcb_is_zero_for_unobserved_pair():
    stats = ContextActionStats()
    assert ucb(stats, "x", "ACTION_0", n_actions=3, t=10, delta=DELTA) == 1.0
    assert lcb(stats, "x", "ACTION_0", n_actions=3, t=10, delta=DELTA) == 0.0


def test_single_observation_never_certifies_a_conflict():
    stats = ContextActionStats()
    _seed(stats, "x", "ACTION_0", n=1, mu=1.0)
    _seed(stats, "x2", "ACTION_1", n=1, mu=1.0)
    assert cannot_link(stats, "x", "x2", ACTIONS, t=2, delta=DELTA, epsilon=0.3) is False


def test_many_observations_with_clearly_divergent_best_actions_certifies_a_conflict():
    stats = ContextActionStats()
    # x strongly prefers ACTION_0, is bad at ACTION_1/ACTION_2
    _seed(stats, "x", "ACTION_0", n=200, mu=1.0)
    _seed(stats, "x", "ACTION_1", n=200, mu=0.0)
    _seed(stats, "x", "ACTION_2", n=200, mu=0.0)
    # x2 strongly prefers ACTION_1, is bad at ACTION_0/ACTION_2
    _seed(stats, "x2", "ACTION_1", n=200, mu=1.0)
    _seed(stats, "x2", "ACTION_0", n=200, mu=0.0)
    _seed(stats, "x2", "ACTION_2", n=200, mu=0.0)
    assert cannot_link(stats, "x", "x2", ACTIONS, t=400, delta=DELTA, epsilon=0.3) is True


def test_many_observations_with_the_same_best_action_never_certifies_a_conflict():
    stats = ContextActionStats()
    _seed(stats, "x", "ACTION_0", n=200, mu=1.0)
    _seed(stats, "x", "ACTION_1", n=200, mu=0.0)
    _seed(stats, "x2", "ACTION_0", n=200, mu=1.0)
    _seed(stats, "x2", "ACTION_1", n=200, mu=0.0)
    assert cannot_link(stats, "x", "x2", ACTIONS, t=400, delta=DELTA, epsilon=0.3) is False


def test_context_action_stats_running_mean_is_correct():
    stats = ContextActionStats()
    stats.update("x", "ACTION_0", 1.0)
    stats.update("x", "ACTION_0", 0.0)
    stats.update("x", "ACTION_0", 1.0)
    assert stats.n("x", "ACTION_0") == 3
    assert math.isclose(stats.mu("x", "ACTION_0"), 2 / 3)


def test_n_contexts_counts_distinct_contexts_not_observations():
    stats = ContextActionStats()
    stats.update("x", "ACTION_0", 1.0)
    stats.update("x", "ACTION_1", 0.0)
    stats.update("x2", "ACTION_0", 1.0)
    assert stats.n_contexts() == 2
