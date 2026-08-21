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


def test_certified_conflict_splits_a_shared_slot(encoder):
    # NOTE: unlike a naive read of the plan brief's sketch, cannot_link
    # (memory/certification.py) can only certify a conflict once BOTH
    # contexts carry informative (non-trivial) confidence bounds on BOTH
    # actions -- an action that a context has literally never tried gets
    # the loosest possible [0,1] bound (LCB=0, UCB=1), which can never
    # exceed EPSILON_CONFLICT no matter how many times the *other* action
    # is tried (see tests/test_certification.py's own seeding pattern,
    # which always seeds both a good AND a bad action per context). So
    # each topic must be written with BOTH actions -- correct=True for its
    # preferred action, correct=False for the other -- for the certified
    # math to ever produce a positive margin.
    mem2 = DecisionAwareMemory(budget=3, encoder=encoder, action_space=["ACTION_0", "ACTION_1"])
    # two singleton slots to fill the budget partially
    mem2.write(_ticket(10, "topic C ticket"), action="ACTION_0", correct=True)
    mem2.write(_ticket(11, "topic D ticket"), action="ACTION_0", correct=True)
    # third slot: force A and B to share it directly (topic A/B have no
    # textual similarity in general, so seed the shared slot explicitly
    # for a deterministic test rather than relying on routing).
    slot = mem2._route(context_key("topic A ticket"), _ticket(20, "topic A ticket"))
    slot.add_member(context_key("topic B ticket"), mem2._embedding_for(context_key("topic B ticket"), _ticket(21, "topic B ticket")))
    assert len(slot.members) == 2
    for _ in range(200):
        # topic A strongly wants ACTION_0 and is confirmed bad at ACTION_1
        mem2.write(_ticket(20, "topic A ticket"), action="ACTION_0", correct=True)
        mem2.write(_ticket(20, "topic A ticket"), action="ACTION_1", correct=False)
        # topic B strongly wants ACTION_1 and is confirmed bad at ACTION_0
        mem2.write(_ticket(21, "topic B ticket"), action="ACTION_1", correct=True)
        mem2.write(_ticket(21, "topic B ticket"), action="ACTION_0", correct=False)
    final_slot = mem2._slot_for(context_key("topic A ticket"))
    assert context_key("topic B ticket") not in final_slot.members


def test_naive_mode_overwrites_instead_of_splitting(encoder):
    mem = DecisionAwareMemory(
        budget=3, encoder=encoder, action_space=["ACTION_0", "ACTION_1"],
        certified=False, split_on_conflict=False,
    )
    slot = mem._route(context_key("topic A ticket"), _ticket(0, "topic A ticket"))
    slot.add_member(context_key("topic B ticket"), mem._embedding_for(context_key("topic B ticket"), _ticket(1, "topic B ticket")))
    mem.write(_ticket(0, "topic A ticket"), action="ACTION_0", correct=True)
    mem.write(_ticket(1, "topic B ticket"), action="ACTION_1", correct=True)
    # naive/overwrite: the slot's membership collapses to just the most
    # recently written, conflicting context -- no versioning, old member
    # discarded entirely.
    final_slot = mem._slot_for(context_key("topic B ticket"))
    assert final_slot.members == {context_key("topic B ticket")}


def test_slots_still_never_exceed_budget_after_conflicts(encoder):
    mem = DecisionAwareMemory(budget=2, encoder=encoder, action_space=["ACTION_0", "ACTION_1"])
    for i in range(30):
        text = f"unique topic {i}"
        action = "ACTION_0" if i % 2 == 0 else "ACTION_1"
        for _ in range(5):
            mem.write(_ticket(i, text), action=action, correct=True)
    assert len(mem.slots) <= 2
