"""Tests for memory/recency_mem.py -- a plain FIFO budget-K memory: once
full, a new write evicts the oldest slot."""
from env.generator import Ticket
from memory.recency_mem import RecencyMemory


def _ticket(step: int, text: str = "some ticket") -> Ticket:
    return Ticket(
        step=step, tenant_id="T0000", topic_id=0, text=text,
        correct_action="ACTION_0", is_override=False,
    )


def test_write_never_exceeds_budget():
    mem = RecencyMemory(budget=3)
    for step in range(10):
        mem.write(_ticket(step), action="ACTION_1", correct=True)
    assert len(mem.slots) == 3


def test_write_past_budget_evicts_the_oldest_slot():
    mem = RecencyMemory(budget=2)
    mem.write(_ticket(0, text="first"), action="ACTION_0", correct=True)
    mem.write(_ticket(1, text="second"), action="ACTION_1", correct=True)
    mem.write(_ticket(2, text="third"), action="ACTION_2", correct=True)
    assert [s.text for s in mem.slots] == ["second", "third"]


def test_retrieve_never_exceeds_budget():
    mem = RecencyMemory(budget=2)
    for step in range(10):
        mem.write(_ticket(step, text=f"ticket {step}"), action="ACTION_1", correct=True)
    context = mem.retrieve(_ticket(99))
    assert context.count('- "') <= 2


def test_retrieve_on_empty_memory_returns_empty_string():
    mem = RecencyMemory(budget=4)
    assert mem.retrieve(_ticket(0)) == ""
