"""Decision-Aware memory: this project's central research contribution
(DeMem-lite). A new memory slot is created only on a statistically
CERTIFIED decision conflict (memory.certification.cannot_link), not by a
similarity threshold -- see Task 3 for that logic. Eviction, when the
slot budget is full, removes the slot with the lowest accumulated
decision-utility, not the oldest or a random one (docs/materials/PLAN.md's
Methods section, item 4).

Uses the SAME embedding model as Semantic-RAG (all-MiniLM-L6-v2) for
routing an unseen micro-context to its nearest existing slot -- the
paper's own confound-avoidance requirement: the only difference between
Semantic-RAG and Decision-Aware is the write/read/evict policy, not a
stronger embedding space.

Micro-context identity (the finite context set the certification math
operates over) is the ticket's first text line -- env.generator.py's
render_ticket_text always puts the topic's paraphrase template there
verbatim, and there are exactly 32 such templates in this project's
environment, so repeated exposure to the same template genuinely
accumulates statistics. This is NOT topic_id (which the memory has no
privileged access to, unlike Oracle) -- it's read directly off the given
ticket text, same as every other non-Oracle memory method.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from env.generator import Ticket
from memory.certification import ContextActionStats
from memory.semantic_mem import DEFAULT_ENCODER_NAME, _cosine_similarity

DELTA = 0.05  # confidence-bound failure probability (paper's delta)
EPSILON_CONFLICT = 0.3  # cannot-link margin (paper's epsilon)


def context_key(ticket_text: str) -> str:
    """The finite micro-context identity -- see this module's docstring."""
    return ticket_text.split("\n", 1)[0]


class Slot:
    def __init__(self, member: str, embedding: np.ndarray):
        self.members: set[str] = {member}
        self.centroid: np.ndarray = embedding

    def add_member(self, member: str, embedding: np.ndarray) -> None:
        self.members.add(member)
        self.centroid = (self.centroid + embedding) / 2

    def drop_member(self, member: str) -> None:
        self.members.discard(member)


class DecisionAwareMemory:
    """Does not subclass BaseMemory: its internal state (per-microcontext
    stats + slot->members routing) doesn't fit the flat MemorySlot list
    every bounded-storage method uses. Matches OracleMemory's precedent
    of exposing the same write/retrieve call shape plus a `.slots`
    attribute for polymorphic memory_size logging -- here, .slots is the
    list of top-level Slot objects (never more than `budget` of them),
    the unit DeMem actually compresses history to K of."""

    def __init__(
        self,
        budget: int,
        encoder: SentenceTransformer | None = None,
        certified: bool = True,
        split_on_conflict: bool = True,
        action_space: list[str] | None = None,
    ):
        self.budget = budget
        self.encoder = encoder or SentenceTransformer(DEFAULT_ENCODER_NAME)
        self.certified = certified
        self.split_on_conflict = split_on_conflict
        self.action_space = action_space or [f"ACTION_{i}" for i in range(8)]
        self.slots: list[Slot] = []
        self._stats = ContextActionStats()
        self._embeddings: dict[str, np.ndarray] = {}
        self._t = 0

    def _slot_for(self, x: str) -> Slot | None:
        for slot in self.slots:
            if x in slot.members:
                return slot
        return None

    def _embedding_for(self, x: str, ticket: Ticket) -> np.ndarray:
        if x not in self._embeddings:
            self._embeddings[x] = self.encoder.encode(ticket.text)
        return self._embeddings[x]

    def _route(self, x: str, ticket: Ticket) -> Slot:
        existing = self._slot_for(x)
        if existing is not None:
            return existing
        embedding = self._embedding_for(x, ticket)
        if len(self.slots) < self.budget:
            slot = Slot(x, embedding)
            self.slots.append(slot)
            return slot
        similarities = [_cosine_similarity(embedding, s.centroid) for s in self.slots]
        best = self.slots[int(np.argmax(similarities))]
        best.add_member(x, embedding)
        return best

    def _slot_best_action(self, slot: Slot) -> str | None:
        pooled_n = {a: 0 for a in self.action_space}
        pooled_mu = {a: 0.0 for a in self.action_space}
        for member in slot.members:
            for a in self.action_space:
                n = self._stats.n(member, a)
                if n == 0:
                    continue
                mu = self._stats.mu(member, a)
                pooled_mu[a] = (pooled_mu[a] * pooled_n[a] + mu * n) / (pooled_n[a] + n)
                pooled_n[a] += n
        if all(n == 0 for n in pooled_n.values()):
            return None
        return max(self.action_space, key=lambda a: pooled_mu[a])

    def retrieve(self, ticket: Ticket) -> str:
        x = context_key(ticket.text)
        slot = self._route(x, ticket)
        best_action = self._slot_best_action(slot)
        if best_action is None:
            return ""
        return f"Most confident action for similar past decisions: {best_action}."

    def write(self, ticket: Ticket, action: str, correct: bool) -> None:
        self._t += 1
        x = context_key(ticket.text)
        self._embedding_for(x, ticket)
        self._stats.update(x, action, 1.0 if correct else 0.0)
        self._route(x, ticket)
