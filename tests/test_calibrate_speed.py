"""Tests for experiments/calibrate_speed.py's pure functions, run against
the real cached Qwen2.5-0.5B-Instruct model but with a tiny n_tickets so the
suite stays fast. The real gate-deciding run (larger N, full grid estimate)
happens in Task 5 as a standalone script execution, not as a pytest assert."""
import pytest

from agent.policy import ClosedSetPolicy
from experiments.calibrate_speed import (
    measure_batch_seconds_per_step,
    run_chance_check,
    run_near_ceiling_check,
)


@pytest.fixture(scope="module")
def policy() -> ClosedSetPolicy:
    return ClosedSetPolicy()


def test_run_near_ceiling_check_returns_accuracy_in_unit_interval(policy):
    accuracy = run_near_ceiling_check(policy, n_tickets=8)
    assert 0.0 <= accuracy <= 1.0


def test_run_chance_check_returns_accuracy_in_unit_interval(policy):
    accuracy = run_chance_check(policy, n_tickets=8)
    assert 0.0 <= accuracy <= 1.0


def test_measure_batch_seconds_per_step_returns_a_positive_float(policy):
    seconds_per_step = measure_batch_seconds_per_step(policy, batch_size=4)
    assert seconds_per_step > 0.0
