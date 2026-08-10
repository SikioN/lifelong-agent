"""Synthetic support-ticket environment for DeMem-lite.

Each ticket has a descriptive topic (surface text, paraphrased) and a
ground-truth correct_action. For tenants in the "override" regime, the
correct action is remapped through a fixed derangement permutation to a
*different* topic's action — this decouples semantic similarity (driven by
topic content) from decision utility (driven by correct_action), per the
alpha-permutation Decoupled Bandit construction in docs/materials/PLAN.md.
"""
import numpy as np

N_TOPICS = 8
ACTION_LABELS = [f"ACTION_{i}" for i in range(N_TOPICS)]


def build_default_action_map(seed: int) -> dict[int, str]:
    """Random (seeded) bijection topic_id -> action label.

    Deliberately non-obvious: action labels are abstract codes, not
    semantically related to topic content, so a pretrained LM cannot guess
    the rule from world knowledge alone (see PLAN.md "утечка знаний из
    претрейна" risk).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_TOPICS)
    return {topic_id: ACTION_LABELS[perm[topic_id]] for topic_id in range(N_TOPICS)}


def _random_derangement(n: int, rng: np.random.Generator) -> list[int]:
    """A permutation of range(n) with no fixed points. Rejection sampling —
    converges in a handful of draws on average (P(derangement) -> 1/e)."""
    while True:
        perm = rng.permutation(n)
        if all(perm[i] != i for i in range(n)):
            return perm.tolist()


def build_mismatch_permutation(seed: int) -> dict[int, int]:
    """Fixed derangement topic_id -> topic_id used to compute the override
    regime's effective decision-identity. No fixed points, so override
    *always* routes to a genuinely different topic's action."""
    rng = np.random.default_rng(seed)
    derangement = _random_derangement(N_TOPICS, rng)
    return {topic_id: derangement[topic_id] for topic_id in range(N_TOPICS)}
