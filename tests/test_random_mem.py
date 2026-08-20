"""Tests for memory/random_mem.py -- a genuine budget-K memory where a
full memory's new write replaces a uniformly-random existing slot (not
the oldest, which would make this identical to Recency)."""
from env.generator import Ticket
from memory.random_mem import RandomMemory


def _ticket(step: int, text: str = "some ticket") -> Ticket:
    return Ticket(
        step=step, tenant_id="T0000", topic_id=0, text=text,
        correct_action="ACTION_0", is_override=False,
    )


def test_write_under_budget_just_appends():
    mem = RandomMemory(budget=4, seed=0)
    mem.write(_ticket(0), action="ACTION_1", correct=True)
    mem.write(_ticket(1), action="ACTION_2", correct=False)
    assert len(mem.slots) == 2


def test_write_never_exceeds_budget():
    mem = RandomMemory(budget=3, seed=0)
    for step in range(10):
        mem.write(_ticket(step), action="ACTION_1", correct=True)
    assert len(mem.slots) == 3


def test_write_past_budget_is_deterministic_given_seed():
    mem_a = RandomMemory(budget=3, seed=42)
    mem_b = RandomMemory(budget=3, seed=42)
    for step in range(10):
        mem_a.write(_ticket(step, text=f"ticket {step}"), action=f"ACTION_{step % 8}", correct=True)
        mem_b.write(_ticket(step, text=f"ticket {step}"), action=f"ACTION_{step % 8}", correct=True)
    assert [s.text for s in mem_a.slots] == [s.text for s in mem_b.slots]


def test_retrieve_never_exceeds_budget():
    mem = RandomMemory(budget=2, seed=0)
    for step in range(10):
        mem.write(_ticket(step, text=f"ticket {step}"), action="ACTION_1", correct=True)
    context = mem.retrieve(_ticket(99))
    # each precedent line starts with "- \"" per format_precedents
    assert context.count('- "') <= 2


def test_retrieve_on_empty_memory_returns_empty_string():
    mem = RandomMemory(budget=4, seed=0)
    assert mem.retrieve(_ticket(0)) == ""
