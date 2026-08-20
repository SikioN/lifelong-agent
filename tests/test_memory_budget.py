"""Tests for memory/base.py: the shared interface every Stage 3 memory
method subclasses. The budget invariant tested here is on retrieve()'s
output (how many precedents get shown to the policy at once), not on
internal storage size -- see the plan's Architecture note for why (the
paper's Feature-RAG baseline has unbounded storage by design)."""
import pytest

from env.generator import Ticket
from memory.base import BaseMemory, MemorySlot, format_precedents


def _make_ticket(step: int = 0) -> Ticket:
    return Ticket(
        step=step,
        tenant_id="T0000",
        topic_id=0,
        text="billing invoice charge dispute: duplicate",
        correct_action="ACTION_3",
        is_override=False,
    )


def test_memory_slot_holds_text_action_correct():
    slot = MemorySlot(text="some ticket", action="ACTION_1", correct=True)
    assert slot.text == "some ticket"
    assert slot.action == "ACTION_1"
    assert slot.correct is True


def test_format_precedents_empty_list_returns_empty_string():
    assert format_precedents([]) == ""


def test_format_precedents_includes_text_and_action_for_each_slot():
    slots = [
        MemorySlot(text="ticket A", action="ACTION_1", correct=True),
        MemorySlot(text="ticket B", action="ACTION_2", correct=False),
    ]
    result = format_precedents(slots)
    assert "ticket A" in result
    assert "ACTION_1" in result
    assert "ticket B" in result
    assert "ACTION_2" in result


def test_format_precedents_collapses_multiline_ticket_text_to_one_line():
    from env.generator import TicketGenerator

    generator = TicketGenerator(alpha=1.0, seed=3, n_tenants=5)
    ticket = generator.sample(0)
    assert "\n" in ticket.text  # real generator output is multi-line

    slots = [MemorySlot(text=ticket.text, action="ACTION_1", correct=True)]
    result = format_precedents(slots)

    # Header line + exactly one line per precedent -> exactly one newline
    # for a single-slot list, and no embedded newline inside the rendered
    # precedent line itself.
    lines = result.split("\n")
    assert len(lines) == 2
    assert "\n" not in lines[1]
    assert "->" in lines[1] and "ACTION_1" in lines[1]


def test_base_memory_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseMemory(budget=4)


def test_base_memory_subclass_must_implement_write_and_retrieve():
    class Incomplete(BaseMemory):
        pass

    with pytest.raises(TypeError):
        Incomplete(budget=4)
