"""Semantic-RAG: the paper's Feature-RAG baseline (arXiv-2605.10870v1,
neurips_2026.tex line 2812) -- the exact "similarity" the paper's DeMem
method argues against. Storage is UNBOUNDED (write() always appends,
matching the paper's growing memory bank B_t = {(x_i, mu_i)}_{i<=t}); the
budget only bounds retrieve()'s top-k-by-cosine-similarity output.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from env.generator import Ticket
from memory.base import BaseMemory, MemorySlot, format_precedents

DEFAULT_ENCODER_NAME = "all-MiniLM-L6-v2"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class SemanticMemory(BaseMemory):
    def __init__(self, budget: int, encoder: SentenceTransformer | None = None):
        super().__init__(budget)
        self.encoder = encoder or SentenceTransformer(DEFAULT_ENCODER_NAME)
        self._embeddings: list[np.ndarray] = []

    def write(self, ticket: Ticket, action: str, correct: bool) -> None:
        self.slots.append(MemorySlot(text=ticket.text, action=action, correct=correct))
        self._embeddings.append(self.encoder.encode(ticket.text))

    def retrieve(self, ticket: Ticket) -> str:
        if not self.slots:
            return ""
        query_embedding = self.encoder.encode(ticket.text)
        similarities = [_cosine_similarity(query_embedding, e) for e in self._embeddings]
        top_k_indices = np.argsort(similarities)[::-1][: self.budget]
        top_slots = [self.slots[i] for i in top_k_indices]
        return format_precedents(top_slots)
