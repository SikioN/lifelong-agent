"""Recency (FIFO): the simplest bounded-history baseline -- once full, a
new write evicts the oldest slot. K most recent experiences, always."""
from env.generator import Ticket
from memory.base import BaseMemory, MemorySlot, format_precedents


class RecencyMemory(BaseMemory):
    def write(self, ticket: Ticket, action: str, correct: bool) -> None:
        self.slots.append(MemorySlot(text=ticket.text, action=action, correct=correct))
        while len(self.slots) > self.budget:
            self.slots.pop(0)

    def retrieve(self, ticket: Ticket) -> str:
        return format_precedents(self.slots)
