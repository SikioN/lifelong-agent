"""Tests for memory/semantic_mem.py. Loads the real, already-cached
all-MiniLM-L6-v2 encoder (same one env/manipulation_check.py already uses)
-- this is a small (22M param) embedding model, not the causal LM policy,
so it's cheap and safe to load and encode with locally."""
import pytest
from sentence_transformers import SentenceTransformer

from env.generator import Ticket
from memory.semantic_mem import SemanticMemory


@pytest.fixture(scope="module")
def encoder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def _ticket(step: int, text: str) -> Ticket:
    return Ticket(
        step=step, tenant_id="T0000", topic_id=0, text=text,
        correct_action="ACTION_0", is_override=False,
    )


def test_write_does_not_evict_past_budget(encoder):
    mem = SemanticMemory(budget=2, encoder=encoder)
    for step in range(5):
        mem.write(_ticket(step, text=f"ticket about topic {step}"), action="ACTION_1", correct=True)
    assert len(mem.slots) == 5  # unbounded storage -- budget doesn't cap writes


def test_retrieve_never_exceeds_budget(encoder):
    mem = SemanticMemory(budget=2, encoder=encoder)
    for step in range(5):
        mem.write(_ticket(step, text=f"ticket about topic {step}"), action="ACTION_1", correct=True)
    context = mem.retrieve(_ticket(99, text="a query ticket"))
    assert context.count('- "') <= 2


def test_retrieve_prefers_the_most_similar_stored_tickets(encoder):
    mem = SemanticMemory(budget=1, encoder=encoder)
    mem.write(
        _ticket(0, text="billing invoice charge dispute: duplicate"),
        action="ACTION_3", correct=True,
    )
    mem.write(
        _ticket(1, text="it would be great if the app supported dark mode"),
        action="ACTION_6", correct=True,
    )
    # query is lexically/semantically closer to the billing ticket
    context = mem.retrieve(_ticket(99, text="I was charged twice for the same order"))
    assert "ACTION_3" in context
    assert "ACTION_6" not in context


def test_retrieve_on_empty_memory_returns_empty_string(encoder):
    mem = SemanticMemory(budget=4, encoder=encoder)
    assert mem.retrieve(_ticket(0, text="anything")) == ""
