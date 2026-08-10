"""Tests for env/generator.py: vocabulary, permutations, tenants, and the
TicketGenerator stream. Grouped by task in this file per the Stage 1 plan."""
import numpy as np
import pytest

from env.generator import (
    N_TOPICS,
    ACTION_LABELS,
    build_default_action_map,
    build_mismatch_permutation,
)


def test_action_labels_has_one_per_topic():
    assert len(ACTION_LABELS) == N_TOPICS
    assert len(set(ACTION_LABELS)) == N_TOPICS  # all unique


def test_default_action_map_is_a_bijection_onto_action_labels():
    mapping = build_default_action_map(seed=42)
    assert len(mapping) == N_TOPICS
    assert set(mapping.keys()) == set(range(N_TOPICS))
    assert set(mapping.values()) == set(ACTION_LABELS)


def test_default_action_map_is_deterministic_given_seed():
    assert build_default_action_map(seed=7) == build_default_action_map(seed=7)


def test_default_action_map_differs_across_seeds():
    # not a hard guarantee in general, but true for these two seeds — pins
    # down that seed actually changes the mapping (catches a hardcoded stub)
    assert build_default_action_map(seed=1) != build_default_action_map(seed=2)


def test_mismatch_permutation_is_a_derangement():
    perm = build_mismatch_permutation(seed=42)
    assert len(perm) == N_TOPICS
    assert set(perm.keys()) == set(range(N_TOPICS))
    assert set(perm.values()) == set(range(N_TOPICS))  # bijection
    for topic_id, mapped_id in perm.items():
        assert mapped_id != topic_id, "mismatch permutation must have no fixed points"


def test_mismatch_permutation_is_deterministic_given_seed():
    assert build_mismatch_permutation(seed=7) == build_mismatch_permutation(seed=7)
