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


def test_certified_and_naive_modes_diverge_under_a_returning_context(encoder):
    """H2's actual comparison: same shared slot (topic A + topic B,
    decision-incompatible: A wants ACTION_0, B wants ACTION_1), same
    conflicting input, but the two policies react on different evidence.

    NOTE on what "diverge" means here: both policies eventually converge
    on separated slots for A and B once budget headroom lets a fresh slot
    be created for whichever context gets displaced (self-healing) -- the
    underlying per-context stats (memory._stats) persist independently of
    slot membership, so final slot topology after many rounds is NOT the
    distinguishing signal (an earlier version of this test asserted final
    topology and was wrong: naive settles into the same two-slot shape as
    certified within ~2 rounds here, just via a cruder path). The REAL,
    deterministic divergence is WHEN each policy acts: naive collapses the
    shared slot after a SINGLE disagreement with no statistical
    justification (could be noise), while certified requires accumulated
    evidence and refuses to touch the slot until cannot_link's confidence
    bound actually certifies the conflict.

    Each topic tries BOTH actions under certified mode (one good, one
    bad) -- certification requires the competing action to actually be
    tried on both contexts, or the untried action's trivial UCB=1 bound
    prevents cannot_link from ever certifying anything (see Task 3's
    review). Naive mode only needs a single sample per topic, per
    _naive_conflict's own (simpler, already-approved) logic."""
    def build_shared_slot(mem):
        slot = mem._route(context_key("topic A ticket"), _ticket(0, "topic A ticket"))
        slot.add_member(
            context_key("topic B ticket"),
            mem._embedding_for(context_key("topic B ticket"), _ticket(1, "topic B ticket")),
        )
        return slot

    certified_mem = DecisionAwareMemory(budget=3, encoder=encoder, action_space=["ACTION_0", "ACTION_1"])
    certified_slot = build_shared_slot(certified_mem)
    naive_mem = DecisionAwareMemory(
        budget=3, encoder=encoder, action_space=["ACTION_0", "ACTION_1"],
        certified=False, split_on_conflict=False,
    )
    naive_slot = build_shared_slot(naive_mem)

    # Round 1, same conflicting input to both. Naive: a single disagreement
    # is enough -- the shared slot is overwritten immediately, keeping only
    # whichever context was written last (topic B).
    naive_mem.write(_ticket(0, "topic A ticket"), action="ACTION_0", correct=True)
    naive_mem.write(_ticket(1, "topic B ticket"), action="ACTION_1", correct=True)
    assert naive_slot.members == {context_key("topic B ticket")}

    # Certified: one round of (admittedly slim) evidence is not enough to
    # certify a conflict -- the slot must remain shared, unlike naive's
    # immediate, statistically unjustified overwrite above.
    certified_mem.write(_ticket(0, "topic A ticket"), action="ACTION_0", correct=True)
    certified_mem.write(_ticket(0, "topic A ticket"), action="ACTION_1", correct=False)
    certified_mem.write(_ticket(1, "topic B ticket"), action="ACTION_1", correct=True)
    certified_mem.write(_ticket(1, "topic B ticket"), action="ACTION_0", correct=False)
    assert certified_slot.members == {context_key("topic A ticket"), context_key("topic B ticket")}

    # Keep accumulating real evidence for certified mode until the conflict
    # is genuinely certified -- unlike naive, this takes real evidence, not
    # a single sample.
    for _ in range(199):
        certified_mem.write(_ticket(0, "topic A ticket"), action="ACTION_0", correct=True)
        certified_mem.write(_ticket(0, "topic A ticket"), action="ACTION_1", correct=False)
        certified_mem.write(_ticket(1, "topic B ticket"), action="ACTION_1", correct=True)
        certified_mem.write(_ticket(1, "topic B ticket"), action="ACTION_0", correct=False)

    # certified: the conflict is now statistically certified, so topic B
    # was split into its own slot -- topic A's original slot no longer
    # contains topic B, and still recommends topic A's own best action.
    final_certified_slot = certified_mem._slot_for(context_key("topic A ticket"))
    assert context_key("topic B ticket") not in final_certified_slot.members
    certified_context = certified_mem.retrieve(_ticket(99, "topic A ticket"))
    assert "ACTION_0" in certified_context
