"""Tests for agent/prompt_templates.py and the TOPIC_NAMES addition to
env/generator.py. Pure string/dict logic — no model loading, fast."""
from env.generator import ACTION_LABELS, N_TOPICS, TOPIC_NAMES, build_default_action_map
from agent.prompt_templates import COMPLETION_ANCHOR, build_prompt, render_rule_context


def test_topic_names_covers_every_topic():
    assert set(TOPIC_NAMES.keys()) == set(range(N_TOPICS))
    assert all(isinstance(name, str) and name.strip() for name in TOPIC_NAMES.values())


def test_topic_names_are_unique():
    assert len(set(TOPIC_NAMES.values())) == N_TOPICS


def test_render_rule_context_has_one_line_per_topic_with_correct_mapping():
    action_map = build_default_action_map(seed=42)
    rule_context = render_rule_context(action_map)
    for topic_id, action in action_map.items():
        assert f"{TOPIC_NAMES[topic_id]} -> {action}" in rule_context


def test_build_prompt_includes_ticket_text_verbatim():
    prompt = build_prompt("some ticket body text", ACTION_LABELS)
    assert "some ticket body text" in prompt


def test_build_prompt_ends_with_completion_anchor():
    prompt = build_prompt("x", ACTION_LABELS, rule_context="RULE", memory_context="MEM")
    assert prompt.endswith(COMPLETION_ANCHOR)


def test_build_prompt_without_rule_or_memory_omits_both_blocks():
    prompt = build_prompt("ticket body", ACTION_LABELS)
    assert "Company routing policy" not in prompt
    assert "MEM_MARKER" not in prompt


def test_build_prompt_with_rule_context_includes_it_verbatim():
    action_map = build_default_action_map(seed=1)
    rule_context = render_rule_context(action_map)
    prompt = build_prompt("ticket body", ACTION_LABELS, rule_context=rule_context)
    assert rule_context in prompt


def test_build_prompt_with_memory_context_includes_it_verbatim():
    prompt = build_prompt("ticket body", ACTION_LABELS, memory_context="MEM_MARKER precedent text")
    assert "MEM_MARKER precedent text" in prompt


def test_build_prompt_lists_full_action_space():
    prompt = build_prompt("ticket body", ACTION_LABELS)
    for action in ACTION_LABELS:
        assert action in prompt
