"""Tests for experiments/diagnose_label_bias.py, run against the real
cached Qwen2.5-0.5B-Instruct model with a tiny n_tickets so the suite
stays fast. The real backbone sweep across multiple/larger models runs on
Kaggle as an execution-only task, not here."""
import pytest

from agent.policy import ClosedSetPolicy
from env.generator import ACTION_LABELS
from experiments.diagnose_label_bias import diagnose_bias


@pytest.fixture(scope="module")
def policy() -> ClosedSetPolicy:
    return ClosedSetPolicy("Qwen/Qwen2.5-0.5B-Instruct")


def test_diagnose_bias_returns_all_expected_keys(policy):
    result = diagnose_bias(policy, n_tickets=8)
    assert set(result.keys()) == {
        "near_ceiling_accuracy",
        "raw_chance_accuracy",
        "calibrated_chance_accuracy",
        "seconds_per_step",
        "label_prior",
    }


def test_diagnose_bias_accuracies_are_in_unit_interval(policy):
    result = diagnose_bias(policy, n_tickets=8)
    for key in ("near_ceiling_accuracy", "raw_chance_accuracy", "calibrated_chance_accuracy"):
        assert 0.0 <= result[key] <= 1.0


def test_diagnose_bias_label_prior_has_one_entry_per_action(policy):
    result = diagnose_bias(policy, n_tickets=8)
    assert len(result["label_prior"]) == len(ACTION_LABELS)
