# Stage 3 Memory Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire label-prior calibration into `ClosedSetPolicy.predict()` (Stage 3's mandatory entry requirement per `docs/materials/PLAN.md`'s Stage 2 row), then implement the common `write/retrieve/evict` memory interface plus four baseline memory methods (Random-K, Recency, Semantic-RAG, Oracle) that Stage 4/5's experiments will compare against Decision-Aware (a separate, follow-up plan).

**Architecture:** `memory/base.py` defines `BaseMemory` (ABC) and `MemorySlot`. The tested invariant across all methods is on `retrieve()`'s output (never more than `budget` formatted precedents shown to the policy at once) — **not** on internal storage size. This distinction is paper-faithful, not a simplification: per `docs/materials/arXiv-2605.10870v1/neurips_2026.tex` (lines 2812-2830), the paper's Feature-RAG baseline explicitly maintains an **unbounded** growing memory bank `B_t = {(x_i, μ̂_i)}_{i≤t}` and retrieves only the top-k nearest neighbors per decision — the comparison Stage 4/5 needs is "smart K-slot compression vs. naive full retention + smart retrieval both showing K items," not "everyone stores exactly K items." Random-K, Recency, and Oracle *do* cap storage at `budget` (their own designs have no reason to retain more), matching the paper's Random Partition baseline (line 2855-2861, a genuine K-group partition). `memory/semantic_mem.py` is the one exception with unbounded storage.

**Tech Stack:** Python 3.11, `numpy`, `sentence-transformers` (`all-MiniLM-L6-v2`, already a project dependency, already used by `env/manipulation_check.py` — same encoder, per `docs/materials/PLAN.md`'s requirement that Semantic-RAG and the future Decision-Aware method share one embedding function), `pytest`. No new dependencies.

**Spec:** `docs/materials/PLAN.md` (Stage 3 row of "Этапы выполнения"; "Методы памяти" section; the Stage 2 finding about calibration wiring, under the Stage 3 row). `docs/materials/arXiv-2605.10870v1/neurips_2026.tex` (Feature-RAG at line 2812, Random Partition at line 2855 — the ground truth for what these baselines must faithfully adapt).

## Global Constraints

(Copied verbatim/paraphrased from the spec — apply to every task below)

- **Mandatory Stage 3 entry requirement** (from `docs/materials/PLAN.md`'s Stage 2 row): calibration must be wired into `ClosedSetPolicy.predict()` before any memory method is built, with the near-ceiling/calibrated-chance gate re-confirmed afterward — not left as a follow-up.
- **Budget invariant:** every memory method's `retrieve()` must never return more than `budget` (`K`) formatted precedents in its `memory_context` string — tested explicitly for each method, not assumed.
- **Shared embedding model:** `memory/semantic_mem.py` uses `all-MiniLM-L6-v2` (via `sentence_transformers.SentenceTransformer`), the same encoder `env/manipulation_check.py` already uses and the same one the future Decision-Aware method must share (per `docs/materials/PLAN.md`'s "общий embedding у Semantic-RAG и Decision-Aware" requirement) — do not introduce a second encoder.
- No LoRA / weight fine-tuning anywhere. No notebooks carrying real logic.
- Python pinned `>=3.11,<3.13`.
- **Stage 3's own Go/No-Go gate** (next plan, not this one): unit tests pass; a smoke run of T=20 across all methods at α=0 and α=0.5 completes with no crashes and no degenerate (0%/100%) accuracy. This plan builds the pieces that gate will exercise; the smoke run itself is this plan's Task 7.

---

## Task 1: Wire calibration into `ClosedSetPolicy.predict()`

**Files:**
- Modify: `agent/policy.py`
- Modify: `experiments/calibrate_speed.py`
- Modify: `tests/test_policy.py`

**Interfaces:**
- Consumes: existing `ClosedSetPolicy.score_candidates`, module-level `calibrate_scores` (both already in `agent/policy.py`, unchanged).
- Produces: `ClosedSetPolicy.predict(self, prompt: str, candidates: list[str], calibration_prior: np.ndarray | None) -> str` — `calibration_prior` is now a **required** parameter (no default) so every call site must explicitly state whether it wants calibrated or raw scoring, rather than silently defaulting either way. Later tasks (memory methods) pass a real prior; Task 1 itself updates the two existing Stage 2 callers to pass `None` (preserving their exact prior behavior).

- [ ] **Step 1: Write the failing tests**

Edit `tests/test_policy.py`. Find the existing test:

```python
def test_predict_follows_a_strong_explicit_echo_instruction(policy):
    prompt = (
        "Repeat back exactly the following code and nothing else: ACTION_0\n"
        "Your response:"
    )
    prediction = policy.predict(prompt, ["ACTION_0", "ACTION_5"])
    assert prediction == "ACTION_0"
```

Replace its call with the new required-parameter form (this test now fails to *compile correctly at the call site*, not just assert-fail, until Step 3 lands — but write it first per TDD):

```python
def test_predict_follows_a_strong_explicit_echo_instruction(policy):
    prompt = (
        "Repeat back exactly the following code and nothing else: ACTION_0\n"
        "Your response:"
    )
    prediction = policy.predict(prompt, ["ACTION_0", "ACTION_5"], calibration_prior=None)
    assert prediction == "ACTION_0"
```

Also update:

```python
def test_predict_returns_one_of_the_candidates(policy):
    prediction = policy.predict("The capital of France is", ["Paris", "Berlin", "Madrid"])
    assert prediction in ["Paris", "Berlin", "Madrid"]
```

to:

```python
def test_predict_returns_one_of_the_candidates(policy):
    prediction = policy.predict("The capital of France is", ["Paris", "Berlin", "Madrid"], calibration_prior=None)
    assert prediction in ["Paris", "Berlin", "Madrid"]
```

Then append two new tests:

```python
def test_predict_with_none_prior_matches_raw_argmax(policy):
    prompt = "Hello,"
    candidates = ["world", "there"]
    raw_scores = policy.score_candidates(prompt, candidates)
    import numpy as np
    expected = candidates[int(np.argmax(raw_scores))]
    assert policy.predict(prompt, candidates, calibration_prior=None) == expected


def test_predict_with_a_prior_applies_calibration():
    import numpy as np
    from agent.policy import calibrate_scores

    class _FakePolicy:
        def score_candidates(self, prompt, candidates):
            return np.array([1.0, 5.0, 2.0])
        predict = ClosedSetPolicy.predict

    fake = _FakePolicy()
    prior = np.array([1.0, 5.0, 2.0])  # matches raw_scores exactly -> calibrated scores all equal
    # argmax of an all-equal array is index 0 (numpy's tie-break rule) --
    # this proves the prior was actually subtracted, not ignored, since
    # the raw argmax (index 1, score 5.0) would differ from index 0.
    candidates = ["ACTION_0", "ACTION_1", "ACTION_2"]
    result = fake.predict(fake, "irrelevant prompt", candidates, calibration_prior=prior)
    assert result == "ACTION_0"
```

Note: this `_FakePolicy` trick avoids loading a real model for this specific unit test (it only exercises `predict()`'s own calibration-application logic, not real scoring) — the module-scoped `policy` fixture used by the other tests still loads the real cached model as before.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_policy.py -v`
Expected: FAIL — `TypeError: predict() missing 1 required positional argument: 'calibration_prior'` (or similar) for the two updated tests, and a real failure/error for the two new tests since `predict()` doesn't accept `calibration_prior` yet.

- [ ] **Step 3: Write minimal implementation**

In `agent/policy.py`, find:

```python
    def predict(self, prompt: str, candidates: list[str]) -> str:
        scores = self.score_candidates(prompt, candidates)
        return candidates[int(np.argmax(scores))]
```

Replace with:

```python
    def predict(self, prompt: str, candidates: list[str], calibration_prior: np.ndarray | None) -> str:
        """calibration_prior is required (not optional-with-a-default) so
        every call site states explicitly whether it wants calibrated or
        raw scoring -- see agent.policy.calibrate_scores and
        ClosedSetPolicy.measure_label_prior for how to obtain a real prior.
        Pass None for deliberately uncalibrated scoring (e.g. Stage 2's
        raw-chance diagnostic, which needs to see the uncalibrated signal)."""
        scores = self.score_candidates(prompt, candidates)
        if calibration_prior is not None:
            scores = calibrate_scores(scores, calibration_prior)
        return candidates[int(np.argmax(scores))]
```

- [ ] **Step 4: Update the two existing Stage 2 callers**

Edit `experiments/calibrate_speed.py`. In `run_near_ceiling_check`, find:

```python
        prediction = policy.predict(prompt, ACTION_LABELS)
```

Replace with:

```python
        prediction = policy.predict(prompt, ACTION_LABELS, calibration_prior=None)
```

In `run_chance_check`, find the same line (a second occurrence) and make the identical change:

```python
        prediction = policy.predict(prompt, ACTION_LABELS, calibration_prior=None)
```

(`run_chance_check_calibrated` already does its own manual `score_candidates` + `calibrate_scores` + `argmax` — leave it untouched; it doesn't call `predict()` at all.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_policy.py -v`
Expected: PASS (13 tests: the 11 already there, with 2 updated call sites, plus 2 new).

Run: `uv run pytest tests/test_calibrate_speed.py -v`
Expected: PASS (5 tests, unchanged behavior — `run_near_ceiling_check`/`run_chance_check`'s only change is an explicit `None` at an already-`None`-equivalent call site).

- [ ] **Step 6: Commit**

```bash
git add agent/policy.py experiments/calibrate_speed.py tests/test_policy.py
git commit -m "Wire calibration into ClosedSetPolicy.predict() (Stage 3 mandatory entry requirement)"
```

---

## Task 2: `memory/base.py` — common interface

**Files:**
- Create: `memory/__init__.py` (empty)
- Create: `memory/base.py`
- Create: `tests/test_memory_budget.py` (currently a Stage-0 placeholder — replace its content entirely)

**Interfaces:**
- Consumes: `env.generator.Ticket` (Stage 1, unchanged).
- Produces: `MemorySlot` (frozen dataclass: `text: str`, `action: str`, `correct: bool`), `BaseMemory` (ABC) with `__init__(self, budget: int)`, `.budget: int`, abstract `.write(self, ticket: Ticket, action: str, correct: bool) -> None`, abstract `.retrieve(self, ticket: Ticket) -> str`, and `format_precedents(slots: list[MemorySlot]) -> str` (module-level helper). Tasks 3-6 all subclass `BaseMemory` and use `format_precedents`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory_budget.py` (replacing the Stage-0 placeholder entirely):

```python
"""Tests for memory/base.py: the shared interface every Stage 3 memory
method subclasses. The budget invariant tested here is on retrieve()'s
output (how many precedents get shown to the policy at once), not on
internal storage size -- see the plan's Architecture note for why (the
paper's Feature-RAG baseline has unbounded storage by design)."""
import pytest

from env.generator import Ticket
from memory.base import BaseMemory, MemorySlot, format_precedents


def _make_ticket(step: int = 0) -> Ticket:
    return Ticket(
        step=step,
        tenant_id="T0000",
        topic_id=0,
        text="billing invoice charge dispute: duplicate",
        correct_action="ACTION_3",
        is_override=False,
    )


def test_memory_slot_holds_text_action_correct():
    slot = MemorySlot(text="some ticket", action="ACTION_1", correct=True)
    assert slot.text == "some ticket"
    assert slot.action == "ACTION_1"
    assert slot.correct is True


def test_format_precedents_empty_list_returns_empty_string():
    assert format_precedents([]) == ""


def test_format_precedents_includes_text_and_action_for_each_slot():
    slots = [
        MemorySlot(text="ticket A", action="ACTION_1", correct=True),
        MemorySlot(text="ticket B", action="ACTION_2", correct=False),
    ]
    result = format_precedents(slots)
    assert "ticket A" in result
    assert "ACTION_1" in result
    assert "ticket B" in result
    assert "ACTION_2" in result


def test_base_memory_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseMemory(budget=4)


def test_base_memory_subclass_must_implement_write_and_retrieve():
    class Incomplete(BaseMemory):
        pass

    with pytest.raises(TypeError):
        Incomplete(budget=4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_memory_budget.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'memory.base'`.

- [ ] **Step 3: Write minimal implementation**

Create `memory/__init__.py` with empty content.

Create `memory/base.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_memory_budget.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add memory/__init__.py memory/base.py tests/test_memory_budget.py
git commit -m "Add memory/base.py: common write/retrieve interface for Stage 3 memory methods"
```

---

## Task 3: `memory/random_mem.py` — Random-K

**Files:**
- Create: `memory/random_mem.py`
- Create: `tests/test_random_mem.py`
- Modify: `.gitignore` (whitelist the new test file)

**Interfaces:**
- Consumes: `BaseMemory`, `MemorySlot`, `format_precedents` (Task 2); `env.generator.Ticket` (Stage 1).
- Produces: `RandomMemory(BaseMemory)` with `__init__(self, budget: int, seed: int = 0)`.

Faithful to the paper's Random Partition baseline (line 2855-2861): a genuine budget-K memory, contents chosen without regard to recency or similarity -- once full, a new write replaces a uniformly-random existing slot rather than the oldest one (that distinction is what separates this from Recency, Task 4).

- [ ] **Step 1: Add the `.gitignore` allowlist entry**

Edit `.gitignore`, add after the `!/tests/test_kaggle_orchestrate.py` line:

```
!/tests/test_random_mem.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_random_mem.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_random_mem.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'memory.random_mem'`.

- [ ] **Step 4: Write minimal implementation**

Create `memory/random_mem.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_random_mem.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add .gitignore memory/random_mem.py tests/test_random_mem.py
git commit -m "Add memory/random_mem.py: Random-K baseline (paper's Random Partition control)"
```

---

## Task 4: `memory/recency_mem.py` — Recency (FIFO)

**Files:**
- Create: `memory/recency_mem.py`
- Create: `tests/test_recency_mem.py`
- Modify: `.gitignore` (whitelist the new test file)

**Interfaces:**
- Consumes: `BaseMemory`, `MemorySlot`, `format_precedents` (Task 2).
- Produces: `RecencyMemory(BaseMemory)` with `__init__(self, budget: int)`.

- [ ] **Step 1: Add the `.gitignore` allowlist entry**

Edit `.gitignore`, add after the `!/tests/test_random_mem.py` line:

```
!/tests/test_recency_mem.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_recency_mem.py`:

```python
"""Tests for memory/recency_mem.py -- a plain FIFO budget-K memory: once
full, a new write evicts the oldest slot."""
from env.generator import Ticket
from memory.recency_mem import RecencyMemory


def _ticket(step: int, text: str = "some ticket") -> Ticket:
    return Ticket(
        step=step, tenant_id="T0000", topic_id=0, text=text,
        correct_action="ACTION_0", is_override=False,
    )


def test_write_never_exceeds_budget():
    mem = RecencyMemory(budget=3)
    for step in range(10):
        mem.write(_ticket(step), action="ACTION_1", correct=True)
    assert len(mem.slots) == 3


def test_write_past_budget_evicts_the_oldest_slot():
    mem = RecencyMemory(budget=2)
    mem.write(_ticket(0, text="first"), action="ACTION_0", correct=True)
    mem.write(_ticket(1, text="second"), action="ACTION_1", correct=True)
    mem.write(_ticket(2, text="third"), action="ACTION_2", correct=True)
    assert [s.text for s in mem.slots] == ["second", "third"]


def test_retrieve_never_exceeds_budget():
    mem = RecencyMemory(budget=2)
    for step in range(10):
        mem.write(_ticket(step, text=f"ticket {step}"), action="ACTION_1", correct=True)
    context = mem.retrieve(_ticket(99))
    assert context.count('- "') <= 2


def test_retrieve_on_empty_memory_returns_empty_string():
    mem = RecencyMemory(budget=4)
    assert mem.retrieve(_ticket(0)) == ""
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_recency_mem.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'memory.recency_mem'`.

- [ ] **Step 4: Write minimal implementation**

Create `memory/recency_mem.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_recency_mem.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add .gitignore memory/recency_mem.py tests/test_recency_mem.py
git commit -m "Add memory/recency_mem.py: FIFO recency baseline"
```

---

## Task 5: `memory/semantic_mem.py` — Semantic-RAG

**Files:**
- Create: `memory/semantic_mem.py`
- Create: `tests/test_semantic_mem.py`
- Modify: `.gitignore` (whitelist the new test file)

**Interfaces:**
- Consumes: `BaseMemory`, `MemorySlot`, `format_precedents` (Task 2); `sentence_transformers.SentenceTransformer` (already a project dependency, already used by `env/manipulation_check.py`).
- Produces: `SemanticMemory(BaseMemory)` with `__init__(self, budget: int, encoder: SentenceTransformer | None = None)`.

Faithful to the paper's Feature-RAG baseline (line 2812-2830): storage is **unbounded** (`write()` always appends, never evicts) -- the budget only bounds what `retrieve()` returns (the top-`budget` nearest neighbors by cosine similarity to the current ticket's text). This is the one method in this plan where `len(self.slots)` legitimately exceeds `self.budget` -- that is correct, not a bug (see the plan's Architecture note).

- [ ] **Step 1: Add the `.gitignore` allowlist entry**

Edit `.gitignore`, add after the `!/tests/test_recency_mem.py` line:

```
!/tests/test_semantic_mem.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_semantic_mem.py`. This loads the real `all-MiniLM-L6-v2` encoder (already cached locally, matching the precedent set by `tests/test_manipulation_check.py`) -- **do not mock it out**, but this is a small (22M param) encoder, not the LLM policy, and loading/encoding a handful of short strings is fast and cheap (no risk of the OOM class of issue that motivated the no-local-LLM-execution directive, which was specifically about the multi-hundred-million-to-billion-parameter causal LM, not this encoder):

```python
"""Tests for memory/semantic_mem.py. Loads the real, already-cached
all-MiniLM-L6-v2 encoder (same one env/manipulation_check.py already uses)
-- this is a small (22M param) embedding model, not the causal LM policy,
so it's cheap and safe to load and encode with locally."""
import pytest
from sentence_transformers import SentenceTransformer

from env.generator import Ticket
from memory.semantic_mem import SemanticMemory


@pytest.fixture(scope="module")
def encoder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def _ticket(step: int, text: str) -> Ticket:
    return Ticket(
        step=step, tenant_id="T0000", topic_id=0, text=text,
        correct_action="ACTION_0", is_override=False,
    )


def test_write_does_not_evict_past_budget(encoder):
    mem = SemanticMemory(budget=2, encoder=encoder)
    for step in range(5):
        mem.write(_ticket(step, text=f"ticket about topic {step}"), action="ACTION_1", correct=True)
    assert len(mem.slots) == 5  # unbounded storage -- budget doesn't cap writes


def test_retrieve_never_exceeds_budget(encoder):
    mem = SemanticMemory(budget=2, encoder=encoder)
    for step in range(5):
        mem.write(_ticket(step, text=f"ticket about topic {step}"), action="ACTION_1", correct=True)
    context = mem.retrieve(_ticket(99, text="a query ticket"))
    assert context.count('- "') <= 2


def test_retrieve_prefers_the_most_similar_stored_tickets(encoder):
    mem = SemanticMemory(budget=1, encoder=encoder)
    mem.write(
        _ticket(0, text="billing invoice charge dispute: duplicate"),
        action="ACTION_3", correct=True,
    )
    mem.write(
        _ticket(1, text="it would be great if the app supported dark mode"),
        action="ACTION_6", correct=True,
    )
    # query is lexically/semantically closer to the billing ticket
    context = mem.retrieve(_ticket(99, text="I was charged twice for the same order"))
    assert "ACTION_3" in context
    assert "ACTION_6" not in context


def test_retrieve_on_empty_memory_returns_empty_string(encoder):
    mem = SemanticMemory(budget=4, encoder=encoder)
    assert mem.retrieve(_ticket(0, text="anything")) == ""
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_semantic_mem.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'memory.semantic_mem'`.

- [ ] **Step 4: Write minimal implementation**

Create `memory/semantic_mem.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_semantic_mem.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add .gitignore memory/semantic_mem.py tests/test_semantic_mem.py
git commit -m "Add memory/semantic_mem.py: Semantic-RAG baseline (paper's Feature-RAG, unbounded storage + top-k retrieval)"
```

---

## Task 6: `memory/oracle_mem.py` — Oracle

**Files:**
- Create: `memory/oracle_mem.py`
- Create: `tests/test_oracle_mem.py`
- Modify: `.gitignore` (whitelist the new test file)

**Interfaces:**
- Consumes: `BaseMemory`, `MemorySlot` (Task 2 -- Oracle does not need `format_precedents`, it constructs its own single-fact string); `env.generator.TicketGenerator`, `Ticket`, `resolve_correct_action` (Stage 1, unchanged).
- Produces: `OracleMemory(BaseMemory)` with `__init__(self, budget: int, generator: TicketGenerator)`.

Faithful to the paper's Oracle (line 2790-2796): "has access to the true decision identity" -- an upper bound, not a realistic memory method. It doesn't need to write/retrieve past experience at all; it directly computes the ground-truth correct action for the current ticket using privileged access to the generator's tenant/override state, and hands that fact to the policy as the "memory."

- [ ] **Step 1: Add the `.gitignore` allowlist entry**

Edit `.gitignore`, add after the `!/tests/test_semantic_mem.py` line:

```
!/tests/test_oracle_mem.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_oracle_mem.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_oracle_mem.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'memory.oracle_mem'`.

- [ ] **Step 4: Write minimal implementation**

Create `memory/oracle_mem.py`:

```python
"""Oracle: the paper's upper-bound baseline (arXiv-2605.10870v1,
neurips_2026.tex line 2790) -- "has access to the true decision identity".
Not a realistic memory method: write() is a no-op, and retrieve() uses
privileged access to the TicketGenerator's tenant/override state to
directly compute the ground-truth correct action for the current ticket,
handing it to the policy as a stated fact rather than a learned precedent.
"""
from env.generator import Ticket, TicketGenerator, resolve_correct_action


class OracleMemory:
    """Does not subclass BaseMemory: BaseMemory's write/retrieve contract
    is built around learning from past MemorySlots, and Oracle explicitly
    doesn't -- it needs a reference to the TicketGenerator instead, which
    no other memory method receives (and shouldn't, since a real agent
    can't observe it). Still implements the same write(ticket, action,
    correct)/retrieve(ticket) -> str shape so Stage 4/5's experiment
    runner can treat it uniformly with the BaseMemory subclasses."""

    def __init__(self, budget: int, generator: TicketGenerator):
        self.budget = budget
        self.generator = generator

    def write(self, ticket: Ticket, action: str, correct: bool) -> None:
        pass

    def retrieve(self, ticket: Ticket) -> str:
        tenant = self.generator._current_tenant(ticket.tenant_id, ticket.step)
        correct_action = resolve_correct_action(
            ticket.topic_id,
            tenant,
            self.generator.default_action_map,
            self.generator.tenant_mismatch_perms[ticket.tenant_id],
        )
        return f"Oracle precedent: the correct action for this exact situation is {correct_action}."
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_oracle_mem.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add .gitignore memory/oracle_mem.py tests/test_oracle_mem.py
git commit -m "Add memory/oracle_mem.py: Oracle upper-bound baseline"
```

---

## Task 7: Smoke-test integration across all four methods

**Files:**
- Create: `experiments/smoke_memory_methods.py`
- Create: `tests/test_smoke_memory_methods.py`
- Modify: `.gitignore` (whitelist the new test file)

**Interfaces:**
- Consumes: `RandomMemory`, `RecencyMemory`, `SemanticMemory`, `OracleMemory` (Tasks 3-6); `ClosedSetPolicy` (Stage 2, with Task 1's calibration wiring); `agent.prompt_templates.build_prompt`; `env.generator.TicketGenerator`.
- Produces: `run_smoke(method_name: str, memory, policy: ClosedSetPolicy, prior, alpha: float, budget: int, n_steps: int = 20, seed: int = 0) -> float` (returns accuracy over the smoke run), `MEMORY_METHOD_NAMES: list[str]`.

This is the actual Stage 3 Go/No-Go gate from `docs/materials/PLAN.md`: "smoke-прогон T=20 по всем методам при α=0 и α=0.5 без падений и без вырожденной (0%/100%) accuracy". It is an execution-only task once written -- the real run happens against the cached local encoder (cheap) and, per the project's current no-local-LLM-execution posture, the causal LM part should be run wherever Stage 2's gate was last run (Colab), not assumed safe locally without checking with the human partner first.

- [ ] **Step 1: Add the `.gitignore` allowlist entry**

Edit `.gitignore`, add after the `!/tests/test_oracle_mem.py` line:

```
!/tests/test_smoke_memory_methods.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_smoke_memory_methods.py`. This test uses the real cached causal LM (small, `Qwen2.5-0.5B-Instruct`, matching the existing project convention for local-safe LLM tests) at a tiny `n_steps`, mirroring the precedent already established by `tests/test_diagnose_label_bias.py` and friends:

```python
"""Tests for experiments/smoke_memory_methods.py's pure orchestration logic
-- run against the real cached Qwen2.5-0.5B-Instruct model with a tiny
n_steps so the suite stays fast, matching the existing project convention."""
import pytest

from agent.policy import ClosedSetPolicy
from env.generator import ACTION_LABELS, TicketGenerator
from experiments.calibrate_speed import neutral_prompt
from experiments.smoke_memory_methods import MEMORY_METHOD_NAMES, run_smoke
from memory.random_mem import RandomMemory


@pytest.fixture(scope="module")
def policy() -> ClosedSetPolicy:
    return ClosedSetPolicy("Qwen/Qwen2.5-0.5B-Instruct")


def test_memory_method_names_has_four_entries():
    assert set(MEMORY_METHOD_NAMES) == {"random", "recency", "semantic", "oracle"}


def test_run_smoke_returns_accuracy_in_unit_interval(policy):
    prior = policy.measure_label_prior(neutral_prompt(), ACTION_LABELS)
    generator = TicketGenerator(alpha=0.0, seed=1, n_tenants=5)
    memory = RandomMemory(budget=4, seed=0)
    accuracy = run_smoke(
        "random", memory, policy, prior, generator=generator, n_steps=5,
    )
    assert 0.0 <= accuracy <= 1.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_smoke_memory_methods.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'experiments.smoke_memory_methods'`.

- [ ] **Step 4: Write minimal implementation**

Create `experiments/smoke_memory_methods.py`:

```python
"""Stage 3 Go/No-Go gate: runs each of the four baseline memory methods
(Random-K, Recency, Semantic-RAG, Oracle) for real over a short ticket
stream at a given alpha, and reports accuracy -- no crashes, no degenerate
(0%/100%) accuracy is the acceptance bar per docs/materials/PLAN.md's
Stage 3 row. Decision-Aware is a separate, follow-up plan and is not
included here.

Run directly: `uv run python -m experiments.smoke_memory_methods`
"""
from agent.policy import ClosedSetPolicy
from agent.prompt_templates import build_prompt
from env.generator import ACTION_LABELS, TicketGenerator
from experiments.calibrate_speed import neutral_prompt
from memory.oracle_mem import OracleMemory
from memory.random_mem import RandomMemory
from memory.recency_mem import RecencyMemory
from memory.semantic_mem import SemanticMemory

MEMORY_METHOD_NAMES = ["random", "recency", "semantic", "oracle"]
SMOKE_N_STEPS = 20
SMOKE_BUDGET = 4
SMOKE_ALPHAS = [0.0, 0.5]


def build_memory(method_name: str, budget: int, generator: TicketGenerator, seed: int = 0):
    if method_name == "random":
        return RandomMemory(budget=budget, seed=seed)
    if method_name == "recency":
        return RecencyMemory(budget=budget)
    if method_name == "semantic":
        return SemanticMemory(budget=budget)
    if method_name == "oracle":
        return OracleMemory(budget=budget, generator=generator)
    raise ValueError(f"unknown memory method: {method_name}")


def run_smoke(
    method_name: str,
    memory,
    policy: ClosedSetPolicy,
    prior,
    generator: TicketGenerator,
    n_steps: int = SMOKE_N_STEPS,
) -> float:
    """Runs the agent loop for n_steps: retrieve memory context, predict
    (calibrated), score correctness, write the outcome back to memory.
    Returns overall accuracy."""
    correct = 0
    for step in range(n_steps):
        ticket = generator.sample(step)
        memory_context = memory.retrieve(ticket)
        prompt = build_prompt(ticket.text, ACTION_LABELS, memory_context=memory_context)
        prediction = policy.predict(prompt, ACTION_LABELS, calibration_prior=prior)
        is_correct = prediction == ticket.correct_action
        correct += int(is_correct)
        memory.write(ticket, action=prediction, correct=is_correct)
    return correct / n_steps


def main() -> None:
    policy = ClosedSetPolicy()
    prior = policy.measure_label_prior(neutral_prompt(), ACTION_LABELS)

    for alpha in SMOKE_ALPHAS:
        print(f"=== alpha={alpha} ===")
        for method_name in MEMORY_METHOD_NAMES:
            generator = TicketGenerator(alpha=alpha, seed=1, n_tenants=10)
            memory = build_memory(method_name, SMOKE_BUDGET, generator)
            accuracy = run_smoke(method_name, memory, policy, prior, generator)
            degenerate = " <-- DEGENERATE" if accuracy in (0.0, 1.0) else ""
            print(f"  {method_name:10s} accuracy={accuracy:.3f}{degenerate}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_smoke_memory_methods.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add .gitignore experiments/smoke_memory_methods.py tests/test_smoke_memory_methods.py
git commit -m "Add experiments/smoke_memory_methods.py: Stage 3 smoke-test gate across all 4 baseline methods"
```

---

## Task 8: Run the real Stage 3 smoke gate

**Files:** none new — execution-only, confirming the plan's built pieces actually satisfy `docs/materials/PLAN.md`'s Stage 3 Go/No-Go gate for real.

**Before this task:** confirm with the human partner where this should run. Everything in Tasks 1-7 was verified either with the small cached encoder (safe locally, per Task 5's note) or with tiny `n_steps` against the small cached `Qwen2.5-0.5B-Instruct` LLM in tests -- but running the FULL smoke gate (`T=20`, both alphas, all 4 methods, `experiments.smoke_memory_methods.main()`) is a larger real workload against the causal LM, and this project's standing posture after Stage 2 has been "no local LLM execution without checking first" at various points. Ask before running `main()` for real, and if the human partner wants it moved to Colab (matching Stage 2's precedent), adapt using the same `kaggle_runner.orchestrate` pattern (a thin script/notebook cell that clones the pinned commit and runs `python -m experiments.smoke_memory_methods`) rather than improvising a new mechanism.

- [ ] **Step 1: Confirm execution location with the human partner, then run**

Wherever agreed:

```bash
uv run python -m experiments.smoke_memory_methods
```

Expected: for both `alpha=0.0` and `alpha=0.5`, all four methods (`random`, `recency`, `semantic`, `oracle`) print an accuracy strictly between 0.0 and 1.0 (no `<-- DEGENERATE` marker), and the process exits 0 (no crash).

- [ ] **Step 2: Record the output as evidence**

```bash
mkdir -p experiments/smoke_output
uv run python -m experiments.smoke_memory_methods | tee experiments/smoke_output/stage3_smoke.txt
```

(Adjust the redirection target if this ran on Colab/elsewhere and output was captured differently -- the goal is a committed record of the real run, matching Stage 1/2's precedent of committing gate evidence.)

- [ ] **Step 3: Commit the evidence**

```bash
git add experiments/smoke_output/stage3_smoke.txt
git commit -m "Stage 3 smoke gate: real run evidence across all 4 baseline methods, both alphas"
```

**If the gate fails** (a crash, or a degenerate 0%/100% accuracy for any method/alpha combination): read the specific traceback or accuracy number before guessing at a fix.
- A crash inside `SemanticMemory` is most likely an encoder/embedding shape issue -- check `retrieve()`'s `np.argsort` call against an empty or single-element `self._embeddings` list.
- A degenerate 100% for `oracle` at any alpha is expected and correct (Oracle has privileged ground-truth access) -- do not treat this as a failure; the plan's "no degenerate accuracy" bar is about the three *non-oracle* methods, which don't have this privileged information. If this ambiguity blocks a clean pass/fail call, that's a real plan-text gap worth flagging to the human partner rather than silently deciding it either way.
- A degenerate 0%/100% for `random`/`recency`/`semantic` at `alpha=0.5` specifically may indicate `TicketGenerator`'s override behavior isn't varied enough at `n_tenants=10` over only 20 steps -- increasing `n_tenants` or `n_steps` in `SMOKE_N_STEPS`/`build_memory` calls is a reasonable first thing to try, not a code bug.

---

## Self-Review

**1. Spec coverage** (against `docs/materials/PLAN.md`'s Stage 3 row and "Методы памяти" section):
- Mandatory calibration-wiring entry requirement → Task 1. ✓
- Common `write/retrieve/evict` interface → Task 2 (`evict` folded into each subclass's `write()`, per the base class's design note -- not a separate abstract method, since Semantic-RAG's storage policy has no eviction at all; this is a deliberate, explained deviation from the literal three-method list, not an omission). ✓
- Random-K → Task 3. ✓
- Recency (FIFO) → Task 4. ✓
- Semantic-RAG (`all-MiniLM-L6-v2`, shared encoder requirement) → Task 5. ✓
- Oracle → Task 6. ✓
- "Бюджет ≤ K всегда" tested invariant → tested per-method on `retrieve()`'s output in Tasks 3-6 (not on storage, per the Architecture note's paper-grounded reasoning). ✓
- Stage 3 Go/No-Go gate (smoke run T=20, both alphas, no crashes/degenerate accuracy) → Tasks 7-8. ✓
- Decision-Aware (method 4) and its optional ablation (4b) → explicitly out of scope for this plan (a separate, follow-up plan, per the scope-check discussion with the human partner — its "certified conflict"/"decision-utility eviction" mechanism needs its own focused design, not a drive-by addition alongside four much-simpler baselines).

**2. Placeholder scan:** No TBD/TODO. Task 8's "if the gate fails" section names concrete things to check rather than vague advice, matching the established project pattern from Stage 1/2's plans.

**3. Type consistency:** `BaseMemory.__init__(self, budget: int)`, `.write(self, ticket: Ticket, action: str, correct: bool) -> None`, `.retrieve(self, ticket: Ticket) -> str` (Task 2) are implemented with identical signatures by `RandomMemory`/`RecencyMemory`/`SemanticMemory` (Tasks 3-5) and matched in shape (though not inheriting `BaseMemory`, by design) by `OracleMemory` (Task 6) — Task 7's `build_memory()` and `run_smoke()` call all four uniformly via this shared shape. `ClosedSetPolicy.predict(self, prompt, candidates, calibration_prior)` (Task 1) is called with the same three-argument form in Task 7's `run_smoke()`.

---

**Plan complete and saved to `docs/implementation-plans/2026-08-20-stage3-memory-baselines-plan.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — I execute tasks in this session directly, batched with checkpoints for your review.

**Which approach?**

---

**Note on scope:** this plan deliberately excludes Decision-Aware (the paper's actual novel method, and this project's central research contribution). Per the writing-plans skill's scope-check guidance, that deserves its own plan with focused design attention on its "certified conflict" and "decision-utility eviction" mechanisms — recommend brainstorming that one before writing it, given real algorithmic ambiguity remains (e.g., what statistical test counts as a "certified" conflict, how many repeated observations are required). Happy to start that whenever you'd like, separately from this plan's execution.
