"""Tests for memory/decision_aware_mem.py's routing skeleton (Task 2) --
conflict detection and slot splitting land in Task 3; these tests only
exercise context_key, Slot, and write/retrieve's routing + pooled-action
lookup, with all observations kept decision-COMPATIBLE (no conflicts) so
routing behavior can be tested in isolation."""
import pytest
from sentence_transformers import SentenceTransformer

from env.generator import Ticket
from memory.decision_aware_mem import DecisionAwareMemory, context_key


@pytest.fixture(scope="module")
def encoder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def _ticket(step: int, text: str) -> Ticket:
    return Ticket(
        step=step, tenant_id="T0000", topic_id=0, text=text,
        correct_action="ACTION_0", is_override=False,
    )


def test_context_key_is_the_first_line_of_ticket_text():
    text = "billing invoice charge dispute: duplicate\n\n— submitted via support portal (ref: T0009)"
    assert context_key(text) == "billing invoice charge dispute: duplicate"


def test_context_key_of_single_line_text_is_the_whole_text():
    assert context_key("no newline here") == "no newline here"


def test_new_micro_context_creates_a_new_slot_while_under_budget(encoder):
    mem = DecisionAwareMemory(budget=4, encoder=encoder, action_space=["ACTION_0", "ACTION_1"])
    mem.write(_ticket(0, "topic A ticket"), action="ACTION_0", correct=True)
    mem.write(_ticket(1, "topic B ticket"), action="ACTION_1", correct=True)
    assert len(mem.slots) == 2


def test_slots_never_exceed_budget(encoder):
    mem = DecisionAwareMemory(budget=2, encoder=encoder, action_space=["ACTION_0", "ACTION_1"])
    for i in range(6):
        mem.write(_ticket(i, f"unique topic {i}"), action="ACTION_0", correct=True)
    assert len(mem.slots) <= 2


def test_repeated_writes_of_the_same_micro_context_reuse_its_slot(encoder):
    mem = DecisionAwareMemory(budget=4, encoder=encoder, action_space=["ACTION_0"])
    mem.write(_ticket(0, "topic A ticket"), action="ACTION_0", correct=True)
    mem.write(_ticket(1, "topic A ticket"), action="ACTION_0", correct=True)
    assert len(mem.slots) == 1
    assert mem.slots[0].members == {"topic A ticket"}


def test_retrieve_on_empty_memory_returns_empty_string(encoder):
    mem = DecisionAwareMemory(budget=4, encoder=encoder, action_space=["ACTION_0"])
    assert mem.retrieve(_ticket(0, "brand new topic")) == ""


def test_retrieve_after_writes_names_the_best_pooled_action(encoder):
    mem = DecisionAwareMemory(budget=4, encoder=encoder, action_space=["ACTION_0", "ACTION_1"])
    for i in range(10):
        mem.write(_ticket(i, "topic A ticket"), action="ACTION_0", correct=True)
    context = mem.retrieve(_ticket(99, "topic A ticket"))
    assert "ACTION_0" in context
