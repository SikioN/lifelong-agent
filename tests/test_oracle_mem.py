"""Tests for memory/oracle_mem.py -- the paper's Oracle upper-bound
baseline: privileged direct access to the ground-truth decision identity,
not a realistic memory method that learns from write()."""
from env.generator import TicketGenerator
from memory.oracle_mem import OracleMemory


def test_retrieve_reveals_the_correct_action_directly():
    generator = TicketGenerator(alpha=1.0, seed=5, n_tenants=5)
    oracle = OracleMemory(budget=4, generator=generator)
    ticket = generator.sample(0)
    context = oracle.retrieve(ticket)
    assert ticket.correct_action in context


def test_retrieve_works_for_override_and_non_override_tenants():
    generator = TicketGenerator(alpha=0.5, seed=7, n_tenants=20)
    oracle = OracleMemory(budget=4, generator=generator)
    tickets = [generator.sample(step) for step in range(20)]
    assert any(t.is_override for t in tickets)
    assert any(not t.is_override for t in tickets)
    for ticket in tickets:
        context = oracle.retrieve(ticket)
        assert ticket.correct_action in context


def test_write_is_a_safe_no_op():
    generator = TicketGenerator(alpha=0.0, seed=1, n_tenants=5)
    oracle = OracleMemory(budget=4, generator=generator)
    ticket = generator.sample(0)
    oracle.write(ticket, action="ACTION_0", correct=True)  # must not raise
    assert oracle.retrieve(ticket) != ""  # still works after a write() call
