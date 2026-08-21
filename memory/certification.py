"""Certified incompatibility math for Decision-Aware memory, implementing
arXiv-2605.10870v1's exact confidence-bound construction (Appendix
app:certificates, not the paper's separate LoCoMo-specific answer-conflict
heuristic in app:practical-split-trigger) -- this project's environment
gives repeated per-(context,action) reward feedback directly, exactly what
the theoretical construction assumes.
"""
import math


class ContextActionStats:
    """Tracks n_t(x,a) and mu_hat_t(x,a) for every observed
    (micro-context, action) pair. x is a string key (see
    decision_aware_mem.context_key); a is an action label string."""

    def __init__(self):
        self._n: dict[str, dict[str, int]] = {}
        self._mu: dict[str, dict[str, float]] = {}

    def update(self, x: str, a: str, reward: float) -> None:
        n_table = self._n.setdefault(x, {})
        mu_table = self._mu.setdefault(x, {})
        n = n_table.get(a, 0)
        mu = mu_table.get(a, 0.0)
        new_n = n + 1
        new_mu = mu + (reward - mu) / new_n
        n_table[a] = new_n
        mu_table[a] = new_mu

    def n(self, x: str, a: str) -> int:
        return self._n.get(x, {}).get(a, 0)

    def mu(self, x: str, a: str) -> float:
        return self._mu.get(x, {}).get(a, 0.0)

    def n_contexts(self) -> int:
        """N_t: number of distinct micro-contexts observed so far."""
        return len(self._n)


def confidence_radius(n: int, n_contexts: int, n_actions: int, t: int, delta: float) -> float:
    """c_t(x,a), eq:conf-radius-app. +inf for n=0 (the trivial [0,1] bound
    applies to unobserved pairs per the paper's own convention)."""
    if n <= 0:
        return float("inf")
    return math.sqrt(math.log(4 * n_contexts * n_actions * t**2 / delta) / (2 * n))


def ucb(stats: ContextActionStats, x: str, a: str, n_actions: int, t: int, delta: float) -> float:
    n = stats.n(x, a)
    if n <= 0:
        return 1.0
    radius = confidence_radius(n, stats.n_contexts(), n_actions, t, delta)
    return min(1.0, stats.mu(x, a) + radius)


def lcb(stats: ContextActionStats, x: str, a: str, n_actions: int, t: int, delta: float) -> float:
    n = stats.n(x, a)
    if n <= 0:
        return 0.0
    radius = confidence_radius(n, stats.n_contexts(), n_actions, t, delta)
    return max(0.0, stats.mu(x, a) - radius)


def _lcb_star(stats: ContextActionStats, x: str, actions: list[str], n_actions: int, t: int, delta: float) -> float:
    return max(lcb(stats, x, a, n_actions, t, delta) for a in actions)


def _lower_gap(stats: ContextActionStats, x: str, a: str, actions: list[str], t: int, delta: float) -> float:
    """Delta_under_t(x,a) = LCB*_t(x) - UCB_t(x,a)."""
    n_actions = len(actions)
    return _lcb_star(stats, x, actions, n_actions, t, delta) - ucb(stats, x, a, n_actions, t, delta)


def cannot_link(
    stats: ContextActionStats,
    x: str,
    x2: str,
    actions: list[str],
    t: int,
    delta: float,
    epsilon: float,
) -> bool:
    """True iff the data certify x and x' cannot share a memory slot of
    radius <= epsilon: d_under_t(x,x') = min_a max(Delta_under_t(x,a),
    Delta_under_t(x',a)) > epsilon."""
    d_under = min(
        max(_lower_gap(stats, x, a, actions, t, delta), _lower_gap(stats, x2, a, actions, t, delta))
        for a in actions
    )
    return d_under > epsilon
