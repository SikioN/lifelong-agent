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


from env.generator import TOPIC_TEMPLATES, render_ticket_text


def test_topic_templates_cover_every_topic_with_multiple_variants():
    assert set(TOPIC_TEMPLATES.keys()) == set(range(N_TOPICS))
    for topic_id, templates in TOPIC_TEMPLATES.items():
        assert len(templates) >= 3, f"topic {topic_id} needs paraphrase variety"
        assert len(set(templates)) == len(templates)  # no duplicate templates


def test_render_ticket_text_uses_a_topic_template():
    rng = np.random.default_rng(0)
    text = render_ticket_text(topic_id=1, tenant_id="T0001", rng=rng)
    body = text.split("\n\n")[0]
    assert body in TOPIC_TEMPLATES[1]


def test_render_ticket_text_tenant_marker_is_low_salience():
    rng = np.random.default_rng(0)
    text = render_ticket_text(topic_id=0, tenant_id="T0042", rng=rng)
    lines = [line for line in text.split("\n") if line.strip()]
    assert "T0042" not in lines[0], "tenant id must not appear in the first line"
    assert "T0042" in text, "tenant id must still be present somewhere"


def test_render_ticket_text_is_deterministic_given_rng_state():
    text_a = render_ticket_text(topic_id=2, tenant_id="T0001", rng=np.random.default_rng(5))
    text_b = render_ticket_text(topic_id=2, tenant_id="T0001", rng=np.random.default_rng(5))
    assert text_a == text_b


from env.generator import Tenant, build_tenants, resolve_correct_action


def test_build_tenants_returns_requested_count_with_unique_ids():
    tenants = build_tenants(n_tenants=40, alpha=0.3, seed=1)
    assert len(tenants) == 40
    assert len({t.tenant_id for t in tenants}) == 40


def test_build_tenants_override_fraction_matches_alpha_at_scale():
    tenants = build_tenants(n_tenants=3000, alpha=0.3, seed=1)
    frac_override = sum(t.override for t in tenants) / len(tenants)
    assert abs(frac_override - 0.3) < 0.03


def test_build_tenants_alpha_zero_means_no_override():
    tenants = build_tenants(n_tenants=200, alpha=0.0, seed=1)
    assert all(not t.override for t in tenants)


def test_build_tenants_alpha_one_means_all_override():
    tenants = build_tenants(n_tenants=200, alpha=1.0, seed=1)
    assert all(t.override for t in tenants)


def test_resolve_correct_action_default_regime_uses_direct_mapping():
    default_map = build_default_action_map(seed=42)
    perm = build_mismatch_permutation(seed=43)
    tenant = Tenant(tenant_id="T0", override=False)
    assert resolve_correct_action(0, tenant, default_map, perm) == default_map[0]


def test_resolve_correct_action_override_regime_uses_permuted_mapping():
    default_map = build_default_action_map(seed=42)
    perm = build_mismatch_permutation(seed=43)
    tenant = Tenant(tenant_id="T0", override=True)
    expected = default_map[perm[0]]
    assert resolve_correct_action(0, tenant, default_map, perm) == expected
    # perm has no fixed points (Task 1), so override always changes the result
    assert resolve_correct_action(0, tenant, default_map, perm) != default_map[0]
