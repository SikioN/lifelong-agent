"""Random-K: the paper's Random Partition control baseline
(arXiv-2605.10870v1, neurips_2026.tex line 2855) -- verifies that DeMem's
gains don't come from partitioning/retaining SOME past examples alone.
A genuine budget-K memory: once full, a new write replaces a
uniformly-random existing slot, not the oldest one."""
import numpy as np

from env.generator import Ticket
from memory.base import BaseMemory, MemorySlot, format_precedents


class RandomMemory(BaseMemory):
    def __init__(self, budget: int, seed: int = 0):
        super().__init__(budget)
        self._rng = np.random.default_rng(seed)

    def write(self, ticket: Ticket, action: str, correct: bool) -> None:
        slot = MemorySlot(text=ticket.text, action=action, correct=correct)
        if len(self.slots) < self.budget:
            self.slots.append(slot)
        else:
            replace_idx = int(self._rng.integers(0, self.budget))
            self.slots[replace_idx] = slot

    def retrieve(self, ticket: Ticket) -> str:
        return format_precedents(self.slots)
