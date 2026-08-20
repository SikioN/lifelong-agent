"""Tests for agent/policy.py. Loads the real, already-cached
Qwen2.5-0.5B-Instruct model once per test session (module-scoped fixture) --
these tests are slower than pure-logic tests elsewhere in the suite, matching
the precedent set by tests/test_manipulation_check.py loading a real
SentenceTransformer. Model load takes ~4s locally; do not mock it out, the
whole point of Stage 2 is proving real scoring behavior."""
import numpy as np
import pytest

from agent.policy import ClosedSetPolicy, calibrate_scores


@pytest.fixture(scope="module")
def policy() -> ClosedSetPolicy:
    return ClosedSetPolicy("Qwen/Qwen2.5-0.5B-Instruct")


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


def test_score_candidates_batch_matches_unbatched(policy):
    prompts = [
        "The capital of France is",
        "Repeat back exactly the following code and nothing else: ACTION_2\nYour response:",
    ]
    candidates = ["Paris", "Berlin"]
    batched = policy.score_candidates_batch(prompts, candidates)
    unbatched = np.stack([policy.score_candidates(p, candidates) for p in prompts])
    assert batched.shape == unbatched.shape == (2, 2)
    assert np.allclose(batched, unbatched, atol=1e-3)


def test_score_candidates_batch_handles_very_different_prompt_lengths(policy):
    short_prompt = "Hi"
    long_prompt = "This is a much longer prompt with many more tokens in it than the short one, " * 3
    candidates = ["yes", "no"]
    batched = policy.score_candidates_batch([short_prompt, long_prompt], candidates)
    unbatched = np.stack(
        [policy.score_candidates(p, candidates) for p in [short_prompt, long_prompt]]
    )
    assert np.allclose(batched, unbatched, atol=1e-3)


def test_calibrate_scores_single_vector_subtracts_prior():
    raw_scores = np.array([1.0, 2.0, 3.0])
    prior = np.array([0.5, 0.5, 0.5])
    calibrated = calibrate_scores(raw_scores, prior)
    assert np.allclose(calibrated, [0.5, 1.5, 2.5])


def test_calibrate_scores_batched_broadcasts_prior_per_row():
    raw_scores = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    prior = np.array([1.0, 1.0, 1.0])
    calibrated = calibrate_scores(raw_scores, prior)
    assert np.allclose(calibrated, [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])


def test_calibrate_scores_removes_a_biased_prior_that_would_flip_the_argmax():
    # candidate 1 has a much higher prior for no task-relevant reason; the
    # actual task signal (equal across all three) carries no preference.
    prior = np.array([1.0, 5.0, 2.0])
    signal = np.array([0.0, 0.0, 0.0])
    raw_scores = prior + signal
    assert np.argmax(raw_scores) == 1  # uncalibrated: biased toward candidate 1
    calibrated = calibrate_scores(raw_scores, prior)
    assert np.allclose(calibrated, [0.0, 0.0, 0.0])  # bias removed


def test_measure_label_prior_returns_one_finite_score_per_candidate(policy):
    prior = policy.measure_label_prior("Pick one:", ["ACTION_0", "ACTION_1", "ACTION_2"])
    assert prior.shape == (3,)
    assert np.all(np.isfinite(prior))


def test_measure_label_prior_is_deterministic(policy):
    prior_a = policy.measure_label_prior("Pick one:", ["ACTION_0", "ACTION_1"])
    prior_b = policy.measure_label_prior("Pick one:", ["ACTION_0", "ACTION_1"])
    assert np.allclose(prior_a, prior_b)
