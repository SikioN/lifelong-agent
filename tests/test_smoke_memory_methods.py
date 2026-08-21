"""Tests for experiments/smoke_memory_methods.py's pure orchestration logic
-- run against the real cached Qwen2.5-0.5B-Instruct model with a tiny
n_steps so the suite stays fast, matching the existing project convention."""
import pytest

from agent.policy import ClosedSetPolicy
from env.generator import ACTION_LABELS, TicketGenerator
from experiments.calibrate_speed import neutral_prompt
from experiments.smoke_memory_methods import MEMORY_METHOD_NAMES, run_smoke
from memory.random_mem import RandomMemory


@pytest.fixture(scope="module")
def policy() -> ClosedSetPolicy:
    return ClosedSetPolicy("Qwen/Qwen2.5-0.5B-Instruct")


def test_memory_method_names_has_five_entries():
    assert set(MEMORY_METHOD_NAMES) == {"random", "recency", "semantic", "oracle", "decision_aware"}


def test_run_smoke_returns_accuracy_in_unit_interval(policy):
    prior = policy.measure_label_prior(neutral_prompt(), ACTION_LABELS)
    generator = TicketGenerator(alpha=0.0, seed=1, n_tenants=5)
    memory = RandomMemory(budget=4, seed=0)
    accuracy = run_smoke(
        "random", memory, policy, prior, generator=generator, n_steps=5,
    )
    assert 0.0 <= accuracy <= 1.0


from memory.decision_aware_mem import DecisionAwareMemory


def test_run_smoke_works_for_decision_aware_certified(policy):
    prior = policy.measure_label_prior(neutral_prompt(), ACTION_LABELS)
    generator = TicketGenerator(alpha=0.0, seed=1, n_tenants=5)
    memory = DecisionAwareMemory(budget=4, action_space=ACTION_LABELS)
    accuracy = run_smoke(
        "decision_aware", memory, policy, prior, generator=generator, n_steps=5,
    )
    assert 0.0 <= accuracy <= 1.0


def test_run_smoke_works_for_decision_aware_naive(policy):
    prior = policy.measure_label_prior(neutral_prompt(), ACTION_LABELS)
    generator = TicketGenerator(alpha=0.0, seed=1, n_tenants=5)
    memory = DecisionAwareMemory(
        budget=4, action_space=ACTION_LABELS, certified=False, split_on_conflict=False,
    )
    accuracy = run_smoke(
        "decision_aware", memory, policy, prior, generator=generator, n_steps=5,
    )
    assert 0.0 <= accuracy <= 1.0
