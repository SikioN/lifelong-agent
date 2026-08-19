"""Prompt assembly for the frozen-backbone policy (agent/policy.py).

Combines ticket text with two independent optional blocks — a rule-in-context
block (topic->action table, used by Stage 2's near-ceiling check) and a
memory-content block (unused until Stage 3's memory methods) — into a single
prompt string that always ends in COMPLETION_ANCHOR. agent/policy.py appends
candidate action labels directly after this string for closed-set scoring.
"""
from env.generator import TOPIC_NAMES

COMPLETION_ANCHOR = "\nAction:"


def render_rule_context(default_action_map: dict[int, str]) -> str:
    """Human-readable routing rule, one line per topic, keyed by TOPIC_NAMES
    so the model matches ticket content to a rule line without ever seeing
    the numeric topic_id directly."""
    lines = ["Company routing policy (topic -> required action):"]
    for topic_id in sorted(default_action_map):
        lines.append(f"- {TOPIC_NAMES[topic_id]} -> {default_action_map[topic_id]}")
    return "\n".join(lines)


def build_prompt(
    ticket_text: str,
    action_space: list[str],
    rule_context: str = "",
    memory_context: str = "",
) -> str:
    """Assemble the full prompt seen by the policy.

    rule_context and memory_context are independent and both optional:
    Stage 2's chance control passes neither, Stage 2's near-ceiling check
    passes only rule_context, Stage 3+ memory methods will pass only
    memory_context (the rule itself is not handed to the policy directly
    once memory is expected to carry that information).
    """
    parts = []
    if rule_context:
        parts.append(rule_context)
    if memory_context:
        parts.append(memory_context)
    parts.append(f"Ticket:\n{ticket_text}")
    parts.append(
        "Respond with exactly one action label from this list: "
        + ", ".join(action_space)
    )
    parts.append(COMPLETION_ANCHOR)
    return "\n\n".join(parts)
