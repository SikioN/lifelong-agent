"""Tests for agent/policy.py. Loads the real, already-cached
Qwen2.5-0.5B-Instruct model once per test session (module-scoped fixture) --
these tests are slower than pure-logic tests elsewhere in the suite, matching
the precedent set by tests/test_manipulation_check.py loading a real
SentenceTransformer. Model load takes ~4s locally; do not mock it out, the
whole point of Stage 2 is proving real scoring behavior."""
import numpy as np
import pytest

from agent.policy import ClosedSetPolicy


@pytest.fixture(scope="module")
def policy() -> ClosedSetPolicy:
    return ClosedSetPolicy()


def test_score_candidates_returns_one_finite_score_per_candidate(policy):
    scores = policy.score_candidates("The sky is", ["blue", "purple"])
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))


def test_score_candidates_is_deterministic(policy):
    scores_a = policy.score_candidates("Hello,", ["world", "there"])
    scores_b = policy.score_candidates("Hello,", ["world", "there"])
    assert np.allclose(scores_a, scores_b)


def test_predict_follows_a_strong_explicit_echo_instruction(policy):
    prompt = (
        "Repeat back exactly the following code and nothing else: ACTION_0\n"
        "Your response:"
    )
    prediction = policy.predict(prompt, ["ACTION_0", "ACTION_5"])
    assert prediction == "ACTION_0"


def test_predict_returns_one_of_the_candidates(policy):
    prediction = policy.predict("The capital of France is", ["Paris", "Berlin", "Madrid"])
    assert prediction in ["Paris", "Berlin", "Madrid"]
