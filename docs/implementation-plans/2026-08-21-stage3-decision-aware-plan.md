# Stage 3 Decision-Aware Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Decision-Aware memory — this project's central research contribution — as the 5th Stage 3 baseline, so `docs/materials/PLAN.md`'s Stage 3 Go/No-Go gate ("smoke-прогон T=20 по всем методам при α=0 и α=0.5" over **all 5 methods**, not 4) can actually be satisfied. Plan A (`docs/implementation-plans/2026-08-20-stage3-memory-baselines-plan.md`) deliberately deferred this method as genuinely novel and under-specified; this plan resolves that design and implements it.

**Architecture:** `memory/certification.py` implements the paper's exact confidence-bound/cannot-link math (arXiv-2605.10870v1, Appendix `app:certificates`, not the separate LoCoMo-specific answer-conflict heuristic in `app:practical-split-trigger` — our environment gives repeated per-`(context, action)` reward feedback directly, which is exactly what the theoretical construction assumes, so there is no need for the paper's one-pass-benchmark simplification). `memory/decision_aware_mem.py` builds on it: up to `budget` top-level memory slots, each holding a set of "micro-context" keys; a new observation is routed to an existing slot (by micro-context membership, or by embedding-centroid similarity for a genuinely new micro-context) and only split into its own slot when the accumulated statistics **certify** a decision conflict with another member of the same slot — never on similarity alone. Eviction, when the slot budget is full, removes the slot with the lowest accumulated decision-utility, not the oldest or a random one. Two boolean flags (`certified`, `split_on_conflict`) switch between DeMem's certified-split behavior (H1) and PLAN.md's H2 "naive" comparison point (evict/overwrite on the first observed disagreement, no statistical margin, no versioning) — H2's actual experiment (non-stationary/circular regimes) is Stage 5's job, using `env.generator.TicketGenerator`'s already-existing `drift_period` parameter; this plan only builds the memory method both H1 and H2 will use.

**Tech Stack:** Python 3.11, `numpy`, `sentence_transformers` (`all-MiniLM-L6-v2`, same encoder as `memory/semantic_mem.py` — reused, not reimplemented, per the paper's confound-avoidance requirement), `pytest`. No new dependencies.

**Spec:** `docs/materials/PLAN.md` ("Методы памяти" item 4, the Stage 3 row of "Этапы выполнения", H2 in "Гипотезы"). `docs/materials/arXiv-2605.10870v1/neurips_2026.tex`: confidence bounds and cannot-link construction at `app:certificates` (lines ~1500-1585, formulas below); the informal description at lines 589-652.

## Global Constraints

- **Micro-context identity is the ticket's first text line, not `topic_id`.** `env/generator.py`'s `render_ticket_text` always puts the topic's paraphrase template on the first line (there are exactly 32 templates: 8 topics × 4 phrasings, see `env/generator.py`'s `TOPIC_TEMPLATES`). This is legitimate input already visible to every non-Oracle memory method (it's literally `ticket.text`) — not privileged information like Oracle's `topic_id`/`tenant.override` access. Repeated exposure to the same template across steps is what lets the confidence bounds actually shrink; a finer key (e.g. the full multi-line text, which includes a per-tenant reference suffix) would never repeat and the certification math would never certify anything.
- **Certification math (verbatim from the paper, Appendix `app:certificates`):** for pull count `n_t(x,a)` and empirical mean `mu_hat_t(x,a)`, confidence radius `c_t(x,a) = sqrt(log(4*N*A*t^2/delta) / (2*n_t(x,a)))` for `n_t(x,a) > 0` (else the trivial `[0,1]` bound applies — `UCB=1`, `LCB=0`). `UCB_t(x,a) = min(1, mu_hat + c_t)`, `LCB_t(x,a) = max(0, mu_hat - c_t)`. `UCB*_t(x) = max_a UCB_t(x,a)`, `LCB*_t(x) = max_a LCB_t(x,a)`. `Delta_under_t(x,a) = LCB*_t(x) - UCB_t(x,a)`. `d_under_t(x,x') = min_a max(Delta_under_t(x,a), Delta_under_t(x',a))`. Cannot-link holds whenever `d_under_t(x,x') > epsilon`.
- `DELTA = 0.05` (confidence-bound failure probability), `EPSILON_CONFLICT = 0.3` (cannot-link margin) — fixed constants, not tuned per-run.
- Eviction, when the slot budget is full, always picks the slot with the lowest accumulated decision-utility (pooled best-action empirical mean across its members) — never recency or random. This is what PLAN.md's Methods section explicitly requires for Decision-Aware, in contrast to Random-K/Recency.
- Shared embedding model: `sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")`, reusing `memory/semantic_mem.py`'s `DEFAULT_ENCODER_NAME` constant and `_cosine_similarity` helper — not a second copy.
- `DecisionAwareMemory` does NOT subclass `BaseMemory` (its internal state doesn't fit the flat `MemorySlot` list every bounded-storage method uses), matching `OracleMemory`'s precedent — but it exposes the same `write`/`retrieve` call shape plus a `.slots` attribute (here: the list of top-level `Slot` objects, capped at `budget`) so Stage 4's future `memory_size` logging works uniformly across all five methods.
- No LoRA / weight fine-tuning. No notebooks carrying real logic. Python pinned `>=3.11,<3.13`.
- **This plan's own Go/No-Go check** (Task 5): a unit-level smoke run of `DecisionAwareMemory` alongside the other four methods via `experiments/smoke_memory_methods.py`'s existing `run_smoke()`, at small `n_steps`, both `certified=True` and `certified=False`, no crashes. The full T=20/both-alpha real gate run (Stage 3's actual PLAN.md gate) is a separate execution-only step after this plan lands, same posture as Plan A's Task 8 (requires the human partner's confirmation of where to run it against the real causal LM).

---

## Task 1: `memory/certification.py` — confidence bounds and cannot-link test

**Files:**
- Create: `memory/certification.py`
- Create: `tests/test_certification.py`
- Modify: `.gitignore` (whitelist the new test file)

**Interfaces:**
- Produces: `ContextActionStats` (class: `.update(x: str, a: str, reward: float) -> None`, `.n(x: str, a: str) -> int`, `.mu(x: str, a: str) -> float`, `.n_contexts() -> int`), `confidence_radius(n: int, n_contexts: int, n_actions: int, t: int, delta: float) -> float`, `ucb(stats: ContextActionStats, x: str, a: str, n_actions: int, t: int, delta: float) -> float`, `lcb(...)` (same signature), `cannot_link(stats: ContextActionStats, x: str, x2: str, actions: list[str], t: int, delta: float, epsilon: float) -> bool`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_certification.py`:

```python
"""Tests for memory/certification.py -- the paper's exact confidence-bound
and cannot-link construction (arXiv-2605.10870v1, Appendix app:certificates).
Uses hand-seeded ContextActionStats at LARGE n (not n=1): with realistic
project-scale N/A/t values, a single observation gives a confidence radius
> 1.0 (the bound is trivial), so cannot_link can never certify anything at
n=1 -- this is the theory's own "certification exploration" requirement,
not a bug, and these tests exercise it directly."""
import math

from memory.certification import ContextActionStats, cannot_link, confidence_radius, lcb, ucb

ACTIONS = ["ACTION_0", "ACTION_1", "ACTION_2"]
DELTA = 0.05


def _seed(stats: ContextActionStats, x: str, a: str, n: int, mu: float) -> None:
    """Feeds n observations of context x, action a, all with the same
    reward mu (0.0 or 1.0), so the resulting empirical mean is exactly mu."""
    reward = 1.0 if mu >= 0.5 else 0.0
    for _ in range(n):
        stats.update(x, a, reward)


def test_confidence_radius_shrinks_with_more_observations():
    small_n_radius = confidence_radius(n=1, n_contexts=2, n_actions=3, t=100, delta=DELTA)
    large_n_radius = confidence_radius(n=200, n_contexts=2, n_actions=3, t=100, delta=DELTA)
    assert large_n_radius < small_n_radius


def test_confidence_radius_is_inf_for_zero_observations():
    assert confidence_radius(n=0, n_contexts=2, n_actions=3, t=100, delta=DELTA) == float("inf")


def test_ucb_is_one_and_lcb_is_zero_for_unobserved_pair():
    stats = ContextActionStats()
    assert ucb(stats, "x", "ACTION_0", n_actions=3, t=10, delta=DELTA) == 1.0
    assert lcb(stats, "x", "ACTION_0", n_actions=3, t=10, delta=DELTA) == 0.0


def test_single_observation_never_certifies_a_conflict():
    stats = ContextActionStats()
    _seed(stats, "x", "ACTION_0", n=1, mu=1.0)
    _seed(stats, "x2", "ACTION_1", n=1, mu=1.0)
    assert cannot_link(stats, "x", "x2", ACTIONS, t=2, delta=DELTA, epsilon=0.3) is False


def test_many_observations_with_clearly_divergent_best_actions_certifies_a_conflict():
    stats = ContextActionStats()
    # x strongly prefers ACTION_0, is bad at ACTION_1/ACTION_2
    _seed(stats, "x", "ACTION_0", n=200, mu=1.0)
    _seed(stats, "x", "ACTION_1", n=200, mu=0.0)
    _seed(stats, "x", "ACTION_2", n=200, mu=0.0)
    # x2 strongly prefers ACTION_1, is bad at ACTION_0/ACTION_2
    _seed(stats, "x2", "ACTION_1", n=200, mu=1.0)
    _seed(stats, "x2", "ACTION_0", n=200, mu=0.0)
    _seed(stats, "x2", "ACTION_2", n=200, mu=0.0)
    assert cannot_link(stats, "x", "x2", ACTIONS, t=400, delta=DELTA, epsilon=0.3) is True


def test_many_observations_with_the_same_best_action_never_certifies_a_conflict():
    stats = ContextActionStats()
    _seed(stats, "x", "ACTION_0", n=200, mu=1.0)
    _seed(stats, "x", "ACTION_1", n=200, mu=0.0)
    _seed(stats, "x2", "ACTION_0", n=200, mu=1.0)
    _seed(stats, "x2", "ACTION_1", n=200, mu=0.0)
    assert cannot_link(stats, "x", "x2", ACTIONS, t=400, delta=DELTA, epsilon=0.3) is False


def test_context_action_stats_running_mean_is_correct():
    stats = ContextActionStats()
    stats.update("x", "ACTION_0", 1.0)
    stats.update("x", "ACTION_0", 0.0)
    stats.update("x", "ACTION_0", 1.0)
    assert stats.n("x", "ACTION_0") == 3
    assert math.isclose(stats.mu("x", "ACTION_0"), 2 / 3)


def test_n_contexts_counts_distinct_contexts_not_observations():
    stats = ContextActionStats()
    stats.update("x", "ACTION_0", 1.0)
    stats.update("x", "ACTION_1", 0.0)
    stats.update("x2", "ACTION_0", 1.0)
    assert stats.n_contexts() == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_certification.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'memory.certification'`.

- [ ] **Step 3: Write minimal implementation**

Create `memory/certification.py`:

```python
"""Certified incompatibility math for Decision-Aware memory, implementing
arXiv-2605.10870v1's exact confidence-bound construction (Appendix
app:certificates, not the paper's separate LoCoMo-specific answer-conflict
heuristic in app:practical-split-trigger) -- this project's environment
gives repeated per-(context,action) reward feedback directly, exactly what
the theoretical construction assumes.
"""
import math


class ContextActionStats:
    """Tracks n_t(x,a) and mu_hat_t(x,a) for every observed
    (micro-context, action) pair. x is a string key (see
    decision_aware_mem.context_key); a is an action label string."""

    def __init__(self):
        self._n: dict[str, dict[str, int]] = {}
        self._mu: dict[str, dict[str, float]] = {}

    def update(self, x: str, a: str, reward: float) -> None:
        n_table = self._n.setdefault(x, {})
        mu_table = self._mu.setdefault(x, {})
        n = n_table.get(a, 0)
        mu = mu_table.get(a, 0.0)
        new_n = n + 1
        new_mu = mu + (reward - mu) / new_n
        n_table[a] = new_n
        mu_table[a] = new_mu

    def n(self, x: str, a: str) -> int:
        return self._n.get(x, {}).get(a, 0)

    def mu(self, x: str, a: str) -> float:
        return self._mu.get(x, {}).get(a, 0.0)

    def n_contexts(self) -> int:
        """N_t: number of distinct micro-contexts observed so far."""
        return len(self._n)


def confidence_radius(n: int, n_contexts: int, n_actions: int, t: int, delta: float) -> float:
    """c_t(x,a), eq:conf-radius-app. +inf for n=0 (the trivial [0,1] bound
    applies to unobserved pairs per the paper's own convention)."""
    if n <= 0:
        return float("inf")
    return math.sqrt(math.log(4 * n_contexts * n_actions * t**2 / delta) / (2 * n))


def ucb(stats: ContextActionStats, x: str, a: str, n_actions: int, t: int, delta: float) -> float:
    n = stats.n(x, a)
    if n <= 0:
        return 1.0
    radius = confidence_radius(n, stats.n_contexts(), n_actions, t, delta)
    return min(1.0, stats.mu(x, a) + radius)


def lcb(stats: ContextActionStats, x: str, a: str, n_actions: int, t: int, delta: float) -> float:
    n = stats.n(x, a)
    if n <= 0:
        return 0.0
    radius = confidence_radius(n, stats.n_contexts(), n_actions, t, delta)
    return max(0.0, stats.mu(x, a) - radius)


def _lcb_star(stats: ContextActionStats, x: str, actions: list[str], n_actions: int, t: int, delta: float) -> float:
    return max(lcb(stats, x, a, n_actions, t, delta) for a in actions)


def _lower_gap(stats: ContextActionStats, x: str, a: str, actions: list[str], t: int, delta: float) -> float:
    """Delta_under_t(x,a) = LCB*_t(x) - UCB_t(x,a)."""
    n_actions = len(actions)
    return _lcb_star(stats, x, actions, n_actions, t, delta) - ucb(stats, x, a, n_actions, t, delta)


def cannot_link(
    stats: ContextActionStats,
    x: str,
    x2: str,
    actions: list[str],
    t: int,
    delta: float,
    epsilon: float,
) -> bool:
    """True iff the data certify x and x' cannot share a memory slot of
    radius <= epsilon: d_under_t(x,x') = min_a max(Delta_under_t(x,a),
    Delta_under_t(x',a)) > epsilon."""
    d_under = min(
        max(_lower_gap(stats, x, a, actions, t, delta), _lower_gap(stats, x2, a, actions, t, delta))
        for a in actions
    )
    return d_under > epsilon
```

- [ ] **Step 4: Add the `.gitignore` allowlist entry**

Edit `.gitignore`, add after the `!/tests/test_smoke_memory_methods.py` line:

```
!/tests/test_certification.py
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_certification.py -v`
Expected: PASS (8 tests).

- [ ] **Step 6: Commit**

```bash
git add memory/certification.py tests/test_certification.py .gitignore
git commit -m "Add memory/certification.py: paper's exact confidence-bound/cannot-link math for Decision-Aware"
```

---

## Task 2: `memory/decision_aware_mem.py` — slot routing skeleton (no conflict detection yet)

**Files:**
- Create: `memory/decision_aware_mem.py`
- Create: `tests/test_decision_aware_mem.py`
- Modify: `.gitignore` (whitelist the new test file)

**Interfaces:**
- Consumes: `memory.certification.ContextActionStats` (Task 1); `memory.semantic_mem.DEFAULT_ENCODER_NAME`, `memory.semantic_mem._cosine_similarity` (already landed); `env.generator.Ticket`.
- Produces: `context_key(ticket_text: str) -> str`, `Slot` (class: `.members: set[str]`, `.centroid: np.ndarray`, `.add_member(member: str, embedding: np.ndarray) -> None`, `.drop_member(member: str) -> None`), `DecisionAwareMemory` (class: `__init__(self, budget: int, encoder=None, certified: bool = True, split_on_conflict: bool = True, action_space: list[str] | None = None)`, `.slots: list[Slot]`, `.retrieve(self, ticket: Ticket) -> str`, `.write(self, ticket: Ticket, action: str, correct: bool) -> None` — write's conflict-resolution logic lands in Task 3, this task only needs write() to update stats and route, not yet split anything).

This task builds routing and the empirical-best-action lookup; Task 3 adds the certified-split behavior.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_decision_aware_mem.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_decision_aware_mem.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'memory.decision_aware_mem'`.

- [ ] **Step 3: Write minimal implementation**

Create `memory/decision_aware_mem.py`:

```python
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
```

- [ ] **Step 4: Add the `.gitignore` allowlist entry**

Edit `.gitignore`, add after the `!/tests/test_certification.py` line:

```
!/tests/test_decision_aware_mem.py
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_decision_aware_mem.py -v`
Expected: PASS (7 tests).

- [ ] **Step 6: Commit**

```bash
git add memory/decision_aware_mem.py tests/test_decision_aware_mem.py .gitignore
git commit -m "Add memory/decision_aware_mem.py: slot routing skeleton (no conflict detection yet)"
```

---

## Task 3: certified conflict detection and split-on-conflict (H1 behavior)

**Files:**
- Modify: `memory/decision_aware_mem.py`
- Modify: `tests/test_decision_aware_mem.py`

**Interfaces:**
- Consumes: `memory.certification.cannot_link` (Task 1).
- Produces: `DecisionAwareMemory._resolve_conflicts(self, slot: Slot, x: str) -> None` (called from `write()`), `DecisionAwareMemory._slot_value(self, slot: Slot) -> float` (accumulated decision-utility, used for eviction).

This task makes `certified=True, split_on_conflict=True` (the default constructor values from Task 2) actually split a slot when a certified conflict is found among its members, and evict the lowest-decision-utility other slot to make room if the budget is already full.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_decision_aware_mem.py`:

```python
from memory.decision_aware_mem import DecisionAwareMemory, context_key


def test_certified_conflict_splits_a_shared_slot(encoder):
    mem = DecisionAwareMemory(budget=4, encoder=encoder, action_space=["ACTION_0", "ACTION_1"])
    # force both micro-contexts into the SAME slot by filling the budget
    # with unrelated singleton slots first, then routing two decision-
    # incompatible contexts into the one remaining slot via similarity
    # (simplify: budget=1, so any second micro-context is forced to share
    # the sole slot with the first).
    mem_forced = DecisionAwareMemory(budget=1, encoder=encoder, action_space=["ACTION_0", "ACTION_1"])
    for _ in range(200):
        mem_forced.write(_ticket(0, "topic A ticket"), action="ACTION_0", correct=True)
        mem_forced.write(_ticket(1, "topic B ticket"), action="ACTION_1", correct=True)
    # topic A strongly wants ACTION_0, topic B strongly wants ACTION_1 --
    # once both are forced into the one shared slot (budget=1), a
    # certified conflict must eventually split them apart. Since budget=1,
    # "splitting" here means the conflict is DETECTED (we can observe this
    # indirectly: the slot's members set should shrink to exclude one of
    # the two, since there's no room to actually create a second slot and
    # _split_or_overwrite's eviction path has no OTHER slot to evict, so
    # the split is a no-op re-add -- this test instead uses budget=3 so
    # eviction has somewhere to go).
    mem2 = DecisionAwareMemory(budget=3, encoder=encoder, action_space=["ACTION_0", "ACTION_1"])
    # two singleton slots to fill the budget partially
    mem2.write(_ticket(10, "topic C ticket"), action="ACTION_0", correct=True)
    mem2.write(_ticket(11, "topic D ticket"), action="ACTION_0", correct=True)
    # third slot: force A and B to share it by using budget=3 with 2 filled
    # -- the third write's context (topic A) opens the 3rd slot; then many
    # repeated topic B writes, routed by similarity, join that same slot
    # (topic A/B share no textual similarity in general, but at budget=3
    # every 4th distinct context has nowhere new to go and must join an
    # existing slot -- topic B will route to whichever slot's centroid it's
    # closest to, which may not be topic A's; to make the test
    # deterministic, seed both into the exact same slot directly).
    slot = mem2._route(context_key("topic A ticket"), _ticket(20, "topic A ticket"))
    slot.add_member(context_key("topic B ticket"), mem2._embedding_for(context_key("topic B ticket"), _ticket(21, "topic B ticket")))
    assert len(slot.members) == 2
    for _ in range(200):
        mem2.write(_ticket(20, "topic A ticket"), action="ACTION_0", correct=True)
        mem2.write(_ticket(21, "topic B ticket"), action="ACTION_1", correct=True)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_decision_aware_mem.py -v`
Expected: `test_certified_conflict_splits_a_shared_slot` and `test_naive_mode_overwrites_instead_of_splitting` FAIL (the topic-B member is never removed from the shared slot, since `write()` doesn't call any conflict-resolution logic yet). The other, already-passing tests from Task 2 continue to PASS.

- [ ] **Step 3: Write minimal implementation**

In `memory/decision_aware_mem.py`, add `cannot_link` to the imports:

```python
from memory.certification import ContextActionStats, cannot_link
```

Then add these two methods to `DecisionAwareMemory`, and call `_resolve_conflicts` from `write()`:

```python
    def write(self, ticket: Ticket, action: str, correct: bool) -> None:
        self._t += 1
        x = context_key(ticket.text)
        self._embedding_for(x, ticket)
        self._stats.update(x, action, 1.0 if correct else 0.0)
        slot = self._route(x, ticket)
        self._resolve_conflicts(slot, x)

    def _resolve_conflicts(self, slot: Slot, x: str) -> None:
        for x2 in slot.members - {x}:
            is_conflict = (
                cannot_link(self._stats, x, x2, self.action_space, self._t, DELTA, EPSILON_CONFLICT)
                if self.certified
                else self._naive_conflict(x, x2)
            )
            if is_conflict:
                self._split_or_overwrite(slot, x, x2)
                return

    def _naive_conflict(self, x: str, x2: str) -> bool:
        """H2's naive comparison point: any disagreement between the two
        contexts' current best empirical action, no statistical margin."""
        best_x = max(self.action_space, key=lambda a: self._stats.mu(x, a))
        best_x2 = max(self.action_space, key=lambda a: self._stats.mu(x2, a))
        return (
            best_x != best_x2
            and self._stats.n(x, best_x) > 0
            and self._stats.n(x2, best_x2) > 0
        )

    def _split_or_overwrite(self, slot: Slot, x: str, x2: str) -> None:
        embedding = self._embeddings[x]
        if self.split_on_conflict:
            slot.drop_member(x)
            if len(self.slots) < self.budget:
                self.slots.append(Slot(x, embedding))
                return
            other_slots = [s for s in self.slots if s is not slot]
            if not other_slots:
                slot.add_member(x, embedding)
                return
            weakest = min(other_slots, key=self._slot_value)
            self.slots.remove(weakest)
            self.slots.append(Slot(x, embedding))
        else:
            slot.members = {x}
            slot.centroid = embedding

    def _slot_value(self, slot: Slot) -> float:
        """Accumulated decision-utility: pooled best-action empirical mean
        across the slot's members (PLAN.md: eviction is by decision-
        utility, not recency/random)."""
        best_action = self._slot_best_action(slot)
        if best_action is None:
            return 0.0
        values = [
            self._stats.mu(m, best_action)
            for m in slot.members
            if self._stats.n(m, best_action) > 0
        ]
        return sum(values) / len(values) if values else 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_decision_aware_mem.py -v`
Expected: PASS (10 tests: the 7 from Task 2 plus 3 new).

- [ ] **Step 5: Commit**

```bash
git add memory/decision_aware_mem.py tests/test_decision_aware_mem.py
git commit -m "Add certified conflict detection, split-on-conflict, and utility-based eviction to DecisionAwareMemory"
```

---

## Task 4: naive/overwrite H2 variant — divergence test

**Files:**
- Modify: `tests/test_decision_aware_mem.py`

**Interfaces:**
- Consumes: `DecisionAwareMemory(certified=False, split_on_conflict=False)` from Task 3 (no new production code — Task 3 already implements the naive/overwrite path; this task proves the two configurations genuinely diverge under the same input, which is the actual point of exposing these flags for H2).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_decision_aware_mem.py`:

```python
def test_certified_and_naive_modes_diverge_under_a_returning_context(encoder):
    """H2's actual comparison: after a slot is forced to share topic A and
    topic B (decision-incompatible), certified mode SPLITS (topic B moves
    to its own slot, topic A's slot keeps its history), while naive mode
    OVERWRITES (topic A's history is discarded entirely, replaced by
    topic B). If topic A's context reappears later, certified mode still
    has a slot recommending its original best action; naive mode has
    nothing -- topic A's slot was overwritten, and its own routing lookup
    now returns whatever the (empty-history) slot most recently learned
    from an unrelated context, not topic A's actual best action."""
    def build_shared_slot(mem):
        slot = mem._route(context_key("topic A ticket"), _ticket(0, "topic A ticket"))
        slot.add_member(
            context_key("topic B ticket"),
            mem._embedding_for(context_key("topic B ticket"), _ticket(1, "topic B ticket")),
        )
        return slot

    certified_mem = DecisionAwareMemory(budget=3, encoder=encoder, action_space=["ACTION_0", "ACTION_1"])
    build_shared_slot(certified_mem)
    naive_mem = DecisionAwareMemory(
        budget=3, encoder=encoder, action_space=["ACTION_0", "ACTION_1"],
        certified=False, split_on_conflict=False,
    )
    build_shared_slot(naive_mem)

    for _ in range(200):
        certified_mem.write(_ticket(0, "topic A ticket"), action="ACTION_0", correct=True)
        certified_mem.write(_ticket(1, "topic B ticket"), action="ACTION_1", correct=True)
        naive_mem.write(_ticket(0, "topic A ticket"), action="ACTION_0", correct=True)
        naive_mem.write(_ticket(1, "topic B ticket"), action="ACTION_1", correct=True)

    # certified: topic A still has its own slot with its own history
    certified_context = certified_mem.retrieve(_ticket(99, "topic A ticket"))
    assert "ACTION_0" in certified_context

    # naive: topic A's slot was overwritten by topic B at the first
    # disagreement -- topic A's original history is gone, and re-routing
    # topic A now either lands on an unrelated/empty slot or a slot whose
    # learned action no longer reflects topic A's own best action.
    naive_slot = naive_mem._slot_for(context_key("topic A ticket"))
    assert naive_slot is None or context_key("topic A ticket") not in naive_slot.members
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_decision_aware_mem.py::test_certified_and_naive_modes_diverge_under_a_returning_context -v`
Expected: PASS already, actually — if Task 3 is correctly implemented this test should pass without further code changes, since it only exercises existing behavior. Run it to confirm; if it fails, that means Task 3's split/overwrite logic has a bug — stop and re-examine `_split_or_overwrite` and `_resolve_conflicts` rather than editing this test to force a pass.

- [ ] **Step 3: Confirm and commit**

If the test passes as expected (no new implementation needed — this task is a verification task proving Task 3's two modes genuinely diverge, the actual thing H2 will exploit):

```bash
git add tests/test_decision_aware_mem.py
git commit -m "Add test proving certified vs naive Decision-Aware modes diverge (H2's basis)"
```

If it fails, treat it as a real Task 3 bug: fix `memory/decision_aware_mem.py`'s `_split_or_overwrite`/`_resolve_conflicts`, re-run the full `tests/test_decision_aware_mem.py` suite, and commit the fix and the test together with a message describing what was wrong.

---

## Task 5: integrate as the 5th memory method

**Files:**
- Modify: `experiments/smoke_memory_methods.py`
- Modify: `tests/test_smoke_memory_methods.py`

**Interfaces:**
- Consumes: `memory.decision_aware_mem.DecisionAwareMemory` (Tasks 2-3); `experiments.smoke_memory_methods.run_smoke` (already exists, unchanged signature).
- Produces: `MEMORY_METHOD_NAMES` extended to 5 entries; `build_memory()` extended with a `"decision_aware"` branch.

- [ ] **Step 1: Write the failing test**

Edit `tests/test_smoke_memory_methods.py`. Update the existing test:

```python
def test_memory_method_names_has_four_entries():
    assert set(MEMORY_METHOD_NAMES) == {"random", "recency", "semantic", "oracle"}
```

to:

```python
def test_memory_method_names_has_five_entries():
    assert set(MEMORY_METHOD_NAMES) == {"random", "recency", "semantic", "oracle", "decision_aware"}
```

Then append:

```python
from memory.decision_aware_mem import DecisionAwareMemory


def test_run_smoke_works_for_decision_aware_certified(policy):
    prior = policy.measure_label_prior(neutral_prompt(), ACTION_LABELS)
    generator = TicketGenerator(alpha=0.0, seed=1, n_tenants=5)
    memory = DecisionAwareMemory(budget=4, action_space=ACTION_LABELS)
    accuracy = run_smoke(
        "decision_aware", memory, policy, prior, generator=generator, n_steps=5,
    )
    assert 0.0 <= accuracy <= 1.0


def test_run_smoke_works_for_decision_aware_naive(policy):
    prior = policy.measure_label_prior(neutral_prompt(), ACTION_LABELS)
    generator = TicketGenerator(alpha=0.0, seed=1, n_tenants=5)
    memory = DecisionAwareMemory(
        budget=4, action_space=ACTION_LABELS, certified=False, split_on_conflict=False,
    )
    accuracy = run_smoke(
        "decision_aware", memory, policy, prior, generator=generator, n_steps=5,
    )
    assert 0.0 <= accuracy <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_smoke_memory_methods.py -v`
Expected: FAIL — `test_memory_method_names_has_five_entries` fails (still 4 entries), and the two new `decision_aware` tests fail with `ValueError: unknown memory method: decision_aware` (from `build_memory`'s `raise ValueError` branch).

- [ ] **Step 3: Write minimal implementation**

Edit `experiments/smoke_memory_methods.py`. Add the import and extend `MEMORY_METHOD_NAMES` and `build_memory`:

```python
from memory.decision_aware_mem import DecisionAwareMemory
```

```python
MEMORY_METHOD_NAMES = ["random", "recency", "semantic", "oracle", "decision_aware"]
```

```python
def build_memory(method_name: str, budget: int, generator: TicketGenerator, seed: int = 0):
    if method_name == "random":
        return RandomMemory(budget=budget, seed=seed)
    if method_name == "recency":
        return RecencyMemory(budget=budget)
    if method_name == "semantic":
        return SemanticMemory(budget=budget)
    if method_name == "oracle":
        return OracleMemory(budget=budget, generator=generator)
    if method_name == "decision_aware":
        return DecisionAwareMemory(budget=budget, action_space=list(ACTION_LABELS))
    raise ValueError(f"unknown memory method: {method_name}")
```

(`ACTION_LABELS` is already imported at the top of `experiments/smoke_memory_methods.py` from `env.generator` — no new import needed for that name.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_smoke_memory_methods.py -v`
Expected: PASS (5 tests: the 2 unchanged, `test_memory_method_names_has_five_entries` renamed/updated, and 2 new `decision_aware` tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/smoke_memory_methods.py tests/test_smoke_memory_methods.py
git commit -m "Integrate DecisionAwareMemory as the 5th Stage 3 memory method"
```

---

## Self-Review

**1. Spec coverage** (against `docs/materials/PLAN.md`'s Methods item 4 and Stage 3 gate):
- Certified conflict via confidence bounds (not similarity threshold) → Task 1 (math) + Task 3 (wiring). ✓
- Shared embedding with Semantic-RAG → Task 2 imports `DEFAULT_ENCODER_NAME`/`_cosine_similarity` from `memory/semantic_mem.py` directly, no duplication. ✓
- Eviction by decision-utility, not recency/random → `_slot_value`/`_split_or_overwrite`'s `min(other_slots, key=self._slot_value)` in Task 3. ✓
- `certified`/`naive` and `split`/`overwrite` flags for H2 → constructor flags in Task 2, wired in Task 3, divergence proven in Task 4. ✓
- Stage 3 gate now covers all 5 methods → Task 5. ✓
- H2's actual non-stationary/circular-regime experiment → explicitly out of scope (Stage 5's job, using `env.generator.TicketGenerator`'s existing `drift_period` parameter, which already exists and needed no new environment work — confirmed by reading `env/generator.py` before writing this plan).

**2. Placeholder scan:** No TBD/TODO. Task 4's step 2 explicitly tells the executor what to do on either outcome (pass confirms Task 3; fail means go fix Task 3), rather than silently assuming success.

**3. Type consistency:** `DecisionAwareMemory.__init__(self, budget: int, encoder=None, certified: bool = True, split_on_conflict: bool = True, action_space: list[str] | None = None)` (Task 2) is used identically in Task 3's tests, Task 4's test, and Task 5's `build_memory()` branch. `context_key(ticket_text: str) -> str` (Task 2) is used consistently in Tasks 3-4's tests. `cannot_link(stats, x, x2, actions, t, delta, epsilon)` (Task 1) matches its call site in Task 3's `_resolve_conflicts` exactly (same argument order, same names).

---

**Plan complete and saved to `docs/implementation-plans/2026-08-21-stage3-decision-aware-plan.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks.

**2. Inline Execution** — execute tasks in this session directly, batched with checkpoints for review.

**Which approach?**

---

**Note on what's still outstanding after this plan:** once this lands, Stage 3's own gate (`docs/materials/PLAN.md`) needs a real execution step — a smoke run of all 5 methods at T=20, α∈{0, 0.5}, against the real causal LM, same posture as Plan A's Task 8 (needs your confirmation of where to run it — Colab, per your standing instruction). That real run is the actual Stage 3 close-out; this plan only builds what it will exercise.
