"""Common interface for Stage 3 memory methods: write/retrieve under a
budget K. The budget invariant every method must satisfy is on retrieve()'s
output (never more than `budget` formatted precedents shown to the policy
at once) -- not on internal storage size. This is paper-faithful, not a
simplification: the paper's Feature-RAG baseline (arXiv-2605.10870v1,
neurips_2026.tex line 2812) has an explicitly UNBOUNDED growing memory
bank and retrieves only the top-k nearest neighbors per decision. Random-K,
Recency, and Oracle do cap storage at `budget` (their designs have no
reason to retain more) -- Semantic-RAG (Task 5) is the one method with
unbounded storage, by design, matching the paper.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

from env.generator import Ticket


@dataclass(frozen=True)
class MemorySlot:
    """One retained unit of experience: what the ticket said, what action
    the agent took, and whether that action was correct."""
    text: str
    action: str
    correct: bool


def format_precedents(slots: list[MemorySlot]) -> str:
    """Renders a list of MemorySlots into the memory_context string passed
    to agent.prompt_templates.build_prompt. Empty list -> empty string
    (build_prompt already handles an empty memory_context by omitting the
    block entirely)."""
    if not slots:
        return ""
    lines = ["Past precedents (ticket -> action taken):"]
    for slot in slots:
        outcome = "correct" if slot.correct else "incorrect"
        lines.append(f'- "{slot.text}" -> {slot.action} ({outcome})')
    return "\n".join(lines)


class BaseMemory(ABC):
    """Common write/retrieve interface for Stage 3 memory methods. Each
    subclass fully owns its own write() (bounded-storage subclasses cap
    self.slots at self.budget internally; unbounded-storage subclasses like
    Semantic-RAG simply append) -- the base class only fixes the interface
    and provides the shared budget attribute, not the storage policy."""

    def __init__(self, budget: int):
        self.budget = budget
        self.slots: list[MemorySlot] = []

    @abstractmethod
    def write(self, ticket: Ticket, action: str, correct: bool) -> None:
        """Records the outcome of one decision so future retrieve() calls
        can draw on it."""
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, ticket: Ticket) -> str:
        """Returns a memory_context string (possibly "") for this ticket,
        built from self.slots -- must never represent more than
        self.budget precedents."""
        raise NotImplementedError
