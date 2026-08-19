# Stage 2: Policy Wrapper + Speed Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a frozen-backbone policy (`agent/policy.py`) that scores the fixed 8-label action space via closed-set causal-LM log-likelihood (no autoregressive generation, no fine-tuning), and a calibration script (`experiments/calibrate_speed.py`) that proves — with real numbers, not assumption — the Stage 2 hard gate from `docs/materials/PLAN.md`: near-ceiling accuracy when the topic→action rule is given directly in context, chance-level accuracy (≈1/8) with no rule and no memory, and a projected Stage 4+5 grid time that fits the 3–4 hour budget on this MacBook (CPU/MPS, no CUDA).

**Architecture:** `agent/prompt_templates.py` assembles a prompt from ticket text plus two independent optional blocks — a rule-in-context block (topic→action table, rendered via new `TOPIC_NAMES` human-readable labels in `env/generator.py`) and a memory-content block (unused until Stage 3, present now only as a plumbed-through parameter) — always ending in a fixed anchor string the policy appends candidate labels to. `agent/policy.py`'s `ClosedSetPolicy` wraps `Qwen/Qwen2.5-0.5B-Instruct` (already cached locally) frozen in eval mode; it scores each of the 8 action labels as a continuation of the prompt by diffing tokenized `prompt` against tokenized `prompt + " " + candidate` and summing log-softmax probabilities over the resulting token span — first an unbatched reference implementation, then a batched version that scores many (prompt × fixed candidate set) pairs in one padded forward pass, which is what makes the real Stage 4/5 grid computationally feasible. `experiments/calibrate_speed.py` runs both control checks and a throughput measurement against the real environment (`env.generator.TicketGenerator`) and hard-asserts the gate, mirroring the pattern already established by Stage 1's `env/manipulation_check.py`.

**Tech Stack:** Python 3.11 (pinned, unchanged), `torch` 2.13 (installed, MPS available / no CUDA), `transformers` 5.15 (installed — uses `dtype=` not the deprecated `torch_dtype=` kwarg, verified against this environment), `Qwen/Qwen2.5-0.5B-Instruct` (verified already present in the local HF cache — no download needed), `numpy`, `pytest`. No new dependencies.

**Spec:** `docs/materials/PLAN.md` (Stage 2 row of "Этапы выполнения"; "Итоговый рисёрч-дизайн" for the frozen-backbone/closed-set-scoring rationale; "Структура репозитория" for `agent/policy.py` and `agent/prompt_templates.py` responsibilities) and `docs/materials/T-Lab 2026. Lifelong Agents.md` (original assignment). Builds directly on `env/generator.py`'s `TicketGenerator` API from the already-gated Stage 1 plan (`docs/implementation-plans/2026-08-10-stage1-environment-and-manipulation-check.md`).

## Global Constraints

(Copied verbatim/paraphrased from `docs/materials/PLAN.md`, apply to every task below)

- Stage 2 goal: "Policy-обёртка + калибровка скорости" — frozen-backbone policy via **closed-set scoring**, **no LoRA on the critical path**. Artifacts: `agent/policy.py`, `experiments/calibrate_speed.py`.
- **Hard gate (blocking, do not proceed to Stage 3 until it passes):** near-ceiling accuracy when the rule is given in context at α=0 with no memory; chance control ≈ 1/|A| (1/8 = 0.125); estimated total time for the full Stage 4+5 grid ≤ 3–4 hours.
- Backbone: **Qwen2.5-0.5B-Instruct** (primary). Fallback if calibration shows it's too slow: `SmolLM2-135M/360M-Instruct` — this would be an **architectural amendment** to `PLAN.md` (per `docs/agent-system.md` §4), not a silent substitution.
- `agent/lora_pretrain.py` is explicitly optional/bonus, **not** part of Stage 2's critical path — do not build it here.
- The `topic → action` mapping must stay non-obvious (already enforced by Stage 1's `ACTION_i` abstract labels) — the rule-context text must render this mapping without leaking any semantic hint about *why* a topic maps to a given action.
- Python pinned `>=3.11,<3.13` (`.python-version` = `3.11`) — do not touch.
- No MCP servers anywhere in this project — plain Python + `uv` only.
- Plain `.py` scripts, no notebooks — deterministic reproducibility (already established in Stage 0/1).

---

## Task 1: Topic names + prompt assembly (`env/generator.py` addition, `agent/prompt_templates.py`)

**Files:**
- Modify: `env/generator.py` (additive only — do not touch any existing Stage 1 function/constant, the Stage 1 gate must stay green)
- Create: `agent/__init__.py` (empty)
- Create: `agent/prompt_templates.py`
- Create: `tests/test_prompt_templates.py`

**Interfaces:**
- Consumes: `N_TOPICS`, `ACTION_LABELS`, `build_default_action_map` from `env/generator.py` (Stage 1).
- Produces: `TOPIC_NAMES: dict[int, str]` (new module constant in `env/generator.py`), `COMPLETION_ANCHOR: str`, `render_rule_context(default_action_map: dict[int, str]) -> str`, `build_prompt(ticket_text: str, action_space: list[str], rule_context: str = "", memory_context: str = "") -> str` (all in `agent/prompt_templates.py`). Task 2/3/4 call `build_prompt(...)` and append candidate labels after the string it returns — the returned string **always** ends with `COMPLETION_ANCHOR`, this is load-bearing for the policy's tokenization-diff trick.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prompt_templates.py`:

```python
"""Tests for agent/prompt_templates.py and the TOPIC_NAMES addition to
env/generator.py. Pure string/dict logic — no model loading, fast."""
from env.generator import ACTION_LABELS, N_TOPICS, TOPIC_NAMES, build_default_action_map
from agent.prompt_templates import COMPLETION_ANCHOR, build_prompt, render_rule_context


def test_topic_names_covers_every_topic():
    assert set(TOPIC_NAMES.keys()) == set(range(N_TOPICS))
    assert all(isinstance(name, str) and name.strip() for name in TOPIC_NAMES.values())


def test_topic_names_are_unique():
    assert len(set(TOPIC_NAMES.values())) == N_TOPICS


def test_render_rule_context_has_one_line_per_topic_with_correct_mapping():
    action_map = build_default_action_map(seed=42)
    rule_context = render_rule_context(action_map)
    for topic_id, action in action_map.items():
        assert f"{TOPIC_NAMES[topic_id]} -> {action}" in rule_context


def test_build_prompt_includes_ticket_text_verbatim():
    prompt = build_prompt("some ticket body text", ACTION_LABELS)
    assert "some ticket body text" in prompt


def test_build_prompt_ends_with_completion_anchor():
    prompt = build_prompt("x", ACTION_LABELS, rule_context="RULE", memory_context="MEM")
    assert prompt.endswith(COMPLETION_ANCHOR)


def test_build_prompt_without_rule_or_memory_omits_both_blocks():
    prompt = build_prompt("ticket body", ACTION_LABELS)
    assert "Company routing policy" not in prompt
    assert "MEM_MARKER" not in prompt


def test_build_prompt_with_rule_context_includes_it_verbatim():
    action_map = build_default_action_map(seed=1)
    rule_context = render_rule_context(action_map)
    prompt = build_prompt("ticket body", ACTION_LABELS, rule_context=rule_context)
    assert rule_context in prompt


def test_build_prompt_with_memory_context_includes_it_verbatim():
    prompt = build_prompt("ticket body", ACTION_LABELS, memory_context="MEM_MARKER precedent text")
    assert "MEM_MARKER precedent text" in prompt


def test_build_prompt_lists_full_action_space():
    prompt = build_prompt("ticket body", ACTION_LABELS)
    for action in ACTION_LABELS:
        assert action in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_prompt_templates.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.prompt_templates'` (and `ImportError: cannot import name 'TOPIC_NAMES'`).

- [ ] **Step 3: Add `TOPIC_NAMES` to `env/generator.py`**

Append to `env/generator.py` (near `TOPIC_TEMPLATES`, after it — purely additive, do not reorder or edit existing code):

```python
TOPIC_NAMES: dict[int, str] = {
    0: "billing or invoice dispute",
    1: "password or login reset",
    2: "data export request",
    3: "account deletion request",
    4: "refund request",
    5: "shipping delay",
    6: "feature request",
    7: "bug report",
}
```

- [ ] **Step 4: Create the `agent` package and `prompt_templates.py`**

Create `agent/__init__.py` with empty content.

Create `agent/prompt_templates.py`:

```python
"""Prompt assembly for the frozen-backbone policy (agent/policy.py).

Combines ticket text with two independent optional blocks — a rule-in-context
block (topic->action table, used by Stage 2's near-ceiling check) and a
memory-content block (unused until Stage 3's memory methods) — into a single
prompt string that always ends in COMPLETION_ANCHOR. agent/policy.py appends
candidate action labels directly after this string for closed-set scoring.
"""
from env.generator import TOPIC_NAMES

COMPLETION_ANCHOR = "\nAction:"


def render_rule_context(default_action_map: dict[int, str]) -> str:
    """Human-readable routing rule, one line per topic, keyed by TOPIC_NAMES
    so the model matches ticket content to a rule line without ever seeing
    the numeric topic_id directly."""
    lines = ["Company routing policy (topic -> required action):"]
    for topic_id in sorted(default_action_map):
        lines.append(f"- {TOPIC_NAMES[topic_id]} -> {default_action_map[topic_id]}")
    return "\n".join(lines)


def build_prompt(
    ticket_text: str,
    action_space: list[str],
    rule_context: str = "",
    memory_context: str = "",
) -> str:
    """Assemble the full prompt seen by the policy.

    rule_context and memory_context are independent and both optional:
    Stage 2's chance control passes neither, Stage 2's near-ceiling check
    passes only rule_context, Stage 3+ memory methods will pass only
    memory_context (the rule itself is not handed to the policy directly
    once memory is expected to carry that information).
    """
    parts = []
    if rule_context:
        parts.append(rule_context)
    if memory_context:
        parts.append(memory_context)
    parts.append(f"Ticket:\n{ticket_text}")
    parts.append(
        "Respond with exactly one action label from this list: "
        + ", ".join(action_space)
    )
    parts.append(COMPLETION_ANCHOR)
    return "\n\n".join(parts)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_prompt_templates.py -v`
Expected: PASS (9 tests).

- [ ] **Step 6: Confirm the Stage 1 gate is still green (additive-change check)**

Run: `uv run pytest tests/test_env_generator.py tests/test_manipulation_check.py -v`
Expected: PASS, same test count as before (24 generator tests + manipulation_check tests) — proves the `TOPIC_NAMES` addition didn't disturb anything.

- [ ] **Step 7: Commit**

```bash
git add env/generator.py agent/__init__.py agent/prompt_templates.py tests/test_prompt_templates.py
git commit -m "Stage 2a: topic names + prompt assembly (rule-context, memory-context, completion anchor)"
```

---

## Task 2: `ClosedSetPolicy` core — unbatched candidate scoring

**Files:**
- Create: `agent/policy.py`
- Create: `tests/test_policy.py`

**Interfaces:**
- Consumes: nothing from this project (wraps `transformers.AutoModelForCausalLM`/`AutoTokenizer` directly).
- Produces: `DEFAULT_MODEL_NAME: str`, `class ClosedSetPolicy` with `__init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None)`, `.score_candidates(self, prompt: str, candidates: list[str]) -> np.ndarray` (shape `(len(candidates),)`), `.predict(self, prompt: str, candidates: list[str]) -> str`. Task 3 adds `.score_candidates_batch` to this same class; Task 4's `calibrate_speed.py` calls `.predict` and `.score_candidates_batch` — get these names/signatures right now, they're load-bearing.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_policy.py`:

```python
"""Tests for agent/policy.py. Loads the real, already-cached
Qwen2.5-0.5B-Instruct model once per test session (module-scoped fixture) --
these tests are slower than pure-logic tests elsewhere in the suite, matching
the precedent set by tests/test_manipulation_check.py loading a real
SentenceTransformer. Model load takes ~4s locally; do not mock it out, the
whole point of Stage 2 is proving real scoring behavior."""
import numpy as np
import pytest

from agent.policy import ClosedSetPolicy


@pytest.fixture(scope="module")
def policy() -> ClosedSetPolicy:
    return ClosedSetPolicy()


def test_score_candidates_returns_one_finite_score_per_candidate(policy):
    scores = policy.score_candidates("The sky is", ["blue", "purple"])
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))


def test_score_candidates_is_deterministic(policy):
    scores_a = policy.score_candidates("Hello,", ["world", "there"])
    scores_b = policy.score_candidates("Hello,", ["world", "there"])
    assert np.allclose(scores_a, scores_b)


def test_predict_follows_a_strong_explicit_echo_instruction(policy):
    prompt = (
        "Repeat back exactly the following code and nothing else: ACTION_0\n"
        "Your response:"
    )
    prediction = policy.predict(prompt, ["ACTION_0", "ACTION_5"])
    assert prediction == "ACTION_0"


def test_predict_returns_one_of_the_candidates(policy):
    prediction = policy.predict("The capital of France is", ["Paris", "Berlin", "Madrid"])
    assert prediction in ["Paris", "Berlin", "Madrid"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_policy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.policy'`.

- [ ] **Step 3: Write minimal implementation**

Create `agent/policy.py`:

```python
"""Frozen-backbone policy: closed-set action-label scoring via causal-LM
log-likelihood. No generation, no fine-tuning on the critical path (LoRA is
optional/bonus — see docs/materials/PLAN.md). Scoring candidates instead of
generating text is what makes the Stage 4/5 grid computationally feasible on
a MacBook without CUDA (batched forward passes, verified in Task 3).
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ClosedSetPolicy:
    """Frozen causal LM wrapped for closed-set (multiple-choice) scoring."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None):
        self.device = device or _select_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        self.model.to(self.device)
        self.model.eval()

    def _candidate_token_ids(self, prompt: str, candidate: str) -> list[int]:
        """Token ids representing `candidate` as a continuation of `prompt`,
        found by diffing tokenize(prompt) against tokenize(prompt + " " +
        candidate) -- robust to BPE merge behavior at the prompt/candidate
        seam, since we only trust the *difference*, not an assumed token
        count for `candidate` alone."""
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(prompt + " " + candidate, add_special_tokens=False)["input_ids"]
        assert full_ids[: len(prompt_ids)] == prompt_ids, (
            "tokenization of prompt is not a prefix of prompt+candidate; "
            "cannot isolate the candidate's token span"
        )
        return full_ids[len(prompt_ids):]

    @torch.no_grad()
    def score_candidates(self, prompt: str, candidates: list[str]) -> np.ndarray:
        """Sum log-prob of each candidate's tokens as a continuation of
        prompt. Unbatched reference implementation -- one forward pass per
        candidate; Task 3 adds a batched version for grid-scale throughput."""
        scores = np.zeros(len(candidates), dtype=np.float64)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for idx, candidate in enumerate(candidates):
            cand_ids = self._candidate_token_ids(prompt, candidate)
            input_ids = torch.tensor([prompt_ids + cand_ids], device=self.device)
            logits = self.model(input_ids).logits[0]  # (seq_len, vocab)
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            n_prompt = len(prompt_ids)
            total = 0.0
            for offset, token_id in enumerate(cand_ids):
                position = n_prompt + offset - 1  # logits[position] predicts token at position+1
                total += log_probs[position, token_id].item()
            scores[idx] = total
        return scores

    def predict(self, prompt: str, candidates: list[str]) -> str:
        scores = self.score_candidates(prompt, candidates)
        return candidates[int(np.argmax(scores))]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_policy.py -v`
Expected: PASS (4 tests). Model load is ~4s (verified locally, already cached — no download).

- [ ] **Step 5: Commit**

```bash
git add agent/policy.py tests/test_policy.py
git commit -m "Stage 2b: ClosedSetPolicy — unbatched closed-set candidate scoring"
```

---

## Task 3: Batched candidate scoring (grid-scale throughput)

**Files:**
- Modify: `agent/policy.py`
- Modify: `tests/test_policy.py`

**Interfaces:**
- Consumes: `ClosedSetPolicy` from Task 2.
- Produces: `ClosedSetPolicy.score_candidates_batch(self, prompts: list[str], candidates: list[str]) -> np.ndarray` (shape `(len(prompts), len(candidates))`). Task 4's `measure_batch_seconds_per_step` calls this exact method.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_policy.py`:

```python
def test_score_candidates_batch_matches_unbatched(policy):
    prompts = [
        "The capital of France is",
        "Repeat back exactly the following code and nothing else: ACTION_2\nYour response:",
    ]
    candidates = ["Paris", "Berlin"]
    batched = policy.score_candidates_batch(prompts, candidates)
    unbatched = np.stack([policy.score_candidates(p, candidates) for p in prompts])
    assert batched.shape == unbatched.shape == (2, 2)
    assert np.allclose(batched, unbatched, atol=1e-3)


def test_score_candidates_batch_handles_very_different_prompt_lengths(policy):
    short_prompt = "Hi"
    long_prompt = "This is a much longer prompt with many more tokens in it than the short one, " * 3
    candidates = ["yes", "no"]
    batched = policy.score_candidates_batch([short_prompt, long_prompt], candidates)
    unbatched = np.stack(
        [policy.score_candidates(p, candidates) for p in [short_prompt, long_prompt]]
    )
    assert np.allclose(batched, unbatched, atol=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_policy.py -v`
Expected: FAIL — `AttributeError: 'ClosedSetPolicy' object has no attribute 'score_candidates_batch'`.

- [ ] **Step 3: Write minimal implementation**

Append to `agent/policy.py` (inside the `ClosedSetPolicy` class — add this method after `predict`):

```python
    @torch.no_grad()
    def score_candidates_batch(self, prompts: list[str], candidates: list[str]) -> np.ndarray:
        """Same scoring as score_candidates, but for many prompts against the
        SAME fixed candidate set, in one padded batched forward pass. This is
        the throughput-critical path for the Stage 4/5 grid: right-padding +
        causal attention means padded positions never influence any real
        token's prediction, so per-row prompt/candidate lengths (tracked
        exactly, pre-padding) are all that's needed to index correctly."""
        pad_id = self.tokenizer.pad_token_id
        sequences: list[list[int]] = []
        prompt_lens: list[int] = []
        cand_lens: list[int] = []
        for prompt in prompts:
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            for candidate in candidates:
                cand_ids = self._candidate_token_ids(prompt, candidate)
                sequences.append(prompt_ids + cand_ids)
                prompt_lens.append(len(prompt_ids))
                cand_lens.append(len(cand_ids))

        max_len = max(len(seq) for seq in sequences)
        batch = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
        for row, seq in enumerate(sequences):
            batch[row, : len(seq)] = torch.tensor(seq)
            attention_mask[row, : len(seq)] = 1
        batch = batch.to(self.device)
        attention_mask = attention_mask.to(self.device)

        logits = self.model(input_ids=batch, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        flat_scores = np.zeros(len(sequences), dtype=np.float64)
        for row in range(len(sequences)):
            n_prompt = prompt_lens[row]
            n_cand = cand_lens[row]
            total = 0.0
            for offset in range(n_cand):
                position = n_prompt + offset - 1
                token_id = batch[row, n_prompt + offset].item()
                total += log_probs[row, position, token_id].item()
            flat_scores[row] = total

        return flat_scores.reshape(len(prompts), len(candidates))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_policy.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add agent/policy.py tests/test_policy.py
git commit -m "Stage 2c: batched closed-set candidate scoring for grid-scale throughput"
```

---

## Task 4: `experiments/calibrate_speed.py` — controls + grid time projection

**Files:**
- Create: `experiments/__init__.py` (empty)
- Create: `experiments/calibrate_speed.py`
- Create: `tests/test_calibrate_speed.py`

**Interfaces:**
- Consumes: `ClosedSetPolicy` (Task 2/3), `build_prompt`/`render_rule_context` (Task 1), `TicketGenerator`/`ACTION_LABELS` (Stage 1, `env/generator.py`).
- Produces: `N_CALIBRATION_TICKETS: int`, `NEAR_CEILING_THRESHOLD: float`, `CHANCE_TOLERANCE: float`, `TOTAL_GRID_STEPS: int`, `GRID_TIME_BUDGET_SECONDS: int`, `run_near_ceiling_check(policy, n_tickets=N_CALIBRATION_TICKETS) -> float`, `run_chance_check(policy, n_tickets=N_CALIBRATION_TICKETS) -> float`, `measure_batch_seconds_per_step(policy, batch_size=16) -> float`, `main() -> None`. Task 5 runs `main()` for real against the gate.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_calibrate_speed.py`:

```python
"""Tests for experiments/calibrate_speed.py's pure functions, run against
the real cached Qwen2.5-0.5B-Instruct model but with a tiny n_tickets so the
suite stays fast. The real gate-deciding run (larger N, full grid estimate)
happens in Task 5 as a standalone script execution, not as a pytest assert."""
import pytest

from agent.policy import ClosedSetPolicy
from experiments.calibrate_speed import (
    measure_batch_seconds_per_step,
    run_chance_check,
    run_near_ceiling_check,
)


@pytest.fixture(scope="module")
def policy() -> ClosedSetPolicy:
    return ClosedSetPolicy()


def test_run_near_ceiling_check_returns_accuracy_in_unit_interval(policy):
    accuracy = run_near_ceiling_check(policy, n_tickets=8)
    assert 0.0 <= accuracy <= 1.0


def test_run_chance_check_returns_accuracy_in_unit_interval(policy):
    accuracy = run_chance_check(policy, n_tickets=8)
    assert 0.0 <= accuracy <= 1.0


def test_measure_batch_seconds_per_step_returns_a_positive_float(policy):
    seconds_per_step = measure_batch_seconds_per_step(policy, batch_size=4)
    assert seconds_per_step > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_calibrate_speed.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'experiments.calibrate_speed'`.

- [ ] **Step 3: Write minimal implementation**

Create `experiments/__init__.py` with empty content.

Create `experiments/calibrate_speed.py`:

```python
"""Stage 2 hard gate: proves the frozen-backbone closed-set policy can (a)
apply an explicitly-given topic->action rule near-perfectly, and (b) sits at
chance without a rule or memory -- before any grid is run at scale. Also
projects total wall-clock time for the Stage 4+5 grid from a measured
per-step throughput, per docs/materials/PLAN.md's Stage 2 gate.

Run directly: `uv run python -m experiments.calibrate_speed`
"""
import time

from agent.policy import ClosedSetPolicy
from agent.prompt_templates import build_prompt, render_rule_context
from env.generator import ACTION_LABELS, TicketGenerator

N_CALIBRATION_TICKETS = 60
NEAR_CEILING_THRESHOLD = 0.85
CHANCE_TOLERANCE = 0.10  # accuracy must land within +/- this of 1/|A|

# Grid sizes computed directly from docs/materials/PLAN.md's "Этапы выполнения"
# Stage 4/5 rows (not the rougher "~50k+~14k" aside elsewhere in that doc --
# this calibration is exactly what should be trusted if the two disagree):
#   Stage 4 main:     alpha(5) x K(2)              x method(5) x seed(5) x T(400)
#   Stage 4 frontier:            K-frontier(4)      x method(5) x seed(5) x T(400)  [1 fixed high alpha]
#   Stage 5 (H2):      regime(2) x method(5) x seed(5) x T(750)
STAGE4_MAIN_STEPS = 5 * 2 * 5 * 5 * 400
STAGE4_FRONTIER_STEPS = 1 * 4 * 5 * 5 * 400
STAGE5_STEPS = 2 * 5 * 5 * 750
TOTAL_GRID_STEPS = STAGE4_MAIN_STEPS + STAGE4_FRONTIER_STEPS + STAGE5_STEPS  # 177,500

GRID_TIME_BUDGET_SECONDS = 4 * 3600  # upper end of PLAN.md's "3-4 hours"


def run_near_ceiling_check(policy: ClosedSetPolicy, n_tickets: int = N_CALIBRATION_TICKETS) -> float:
    """alpha=0, rule given in context, no memory -> the policy should apply
    the rule almost perfectly. This is a capability floor, not a memory
    result -- it must pass before any memory method comparison is meaningful."""
    generator = TicketGenerator(alpha=0.0, seed=777)
    rule_context = render_rule_context(generator.default_action_map)
    correct = 0
    for step in range(n_tickets):
        ticket = generator.sample(step)
        prompt = build_prompt(ticket.text, ACTION_LABELS, rule_context=rule_context)
        prediction = policy.predict(prompt, ACTION_LABELS)
        correct += int(prediction == ticket.correct_action)
    return correct / n_tickets


def run_chance_check(policy: ClosedSetPolicy, n_tickets: int = N_CALIBRATION_TICKETS) -> float:
    """No rule, no memory -> the model cannot know topic->action beyond
    guessing, since ACTION_i labels carry no world-knowledge-derivable
    semantics (Stage 1's non-obvious permutation). Accuracy should sit near
    1/|A|; a big deviation would mean the model is leaking pretrained
    knowledge or the scoring has a systematic bias (e.g. length bias)."""
    generator = TicketGenerator(alpha=0.0, seed=778)
    correct = 0
    for step in range(n_tickets):
        ticket = generator.sample(step)
        prompt = build_prompt(ticket.text, ACTION_LABELS)
        prediction = policy.predict(prompt, ACTION_LABELS)
        correct += int(prediction == ticket.correct_action)
    return correct / n_tickets


def measure_batch_seconds_per_step(policy: ClosedSetPolicy, batch_size: int = 16) -> float:
    """Wall-clock seconds per ticket-decision for one batched closed-set
    scoring call (batch_size tickets x len(ACTION_LABELS) candidates each),
    used to project total grid time. A "step" in the real grid is exactly
    one ticket's worth of scoring over the full action space."""
    generator = TicketGenerator(alpha=0.0, seed=779)
    rule_context = render_rule_context(generator.default_action_map)
    tickets = [generator.sample(step) for step in range(batch_size)]
    prompts = [build_prompt(t.text, ACTION_LABELS, rule_context=rule_context) for t in tickets]

    start = time.perf_counter()
    policy.score_candidates_batch(prompts, ACTION_LABELS)
    elapsed = time.perf_counter() - start
    return elapsed / batch_size


def main() -> None:
    policy = ClosedSetPolicy()

    near_ceiling_acc = run_near_ceiling_check(policy)
    chance_acc = run_chance_check(policy)
    seconds_per_step = measure_batch_seconds_per_step(policy)
    projected_total_seconds = seconds_per_step * TOTAL_GRID_STEPS

    chance_target = 1 / len(ACTION_LABELS)
    print(f"near-ceiling accuracy (rule given, alpha=0): {near_ceiling_acc:.3f}")
    print(f"chance accuracy (no rule, no memory):        {chance_acc:.3f}  (target ~{chance_target:.3f})")
    print(f"measured seconds/step (batched, batch=16):    {seconds_per_step:.4f}")
    print(f"projected total grid steps:                   {TOTAL_GRID_STEPS:,}")
    print(f"projected total grid time:                    {projected_total_seconds / 3600:.2f} hours")

    assert near_ceiling_acc >= NEAR_CEILING_THRESHOLD, (
        f"GATE FAILED: near-ceiling accuracy {near_ceiling_acc:.3f} < {NEAR_CEILING_THRESHOLD}"
    )
    assert abs(chance_acc - chance_target) <= CHANCE_TOLERANCE, (
        f"GATE FAILED: chance accuracy {chance_acc:.3f} not within "
        f"{CHANCE_TOLERANCE} of {chance_target:.3f}"
    )
    assert projected_total_seconds <= GRID_TIME_BUDGET_SECONDS, (
        f"GATE FAILED: projected grid time {projected_total_seconds / 3600:.2f}h "
        f"exceeds budget {GRID_TIME_BUDGET_SECONDS / 3600:.1f}h -- consider a "
        f"smaller backbone (SmolLM2-135M/360M-Instruct, per PLAN.md fallback)"
    )

    print("\nGATE PASSED.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_calibrate_speed.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/__init__.py experiments/calibrate_speed.py tests/test_calibrate_speed.py
git commit -m "Stage 2d: calibrate_speed — near-ceiling/chance controls + grid time projection"
```

---

## Task 5: Run calibration for real — confirm the Stage 2 gate

**Files:** none new — this task *executes* `experiments/calibrate_speed.py` end-to-end and records the outcome. No code changes unless the gate fails (see "If the gate fails" below).

- [ ] **Step 1: Run the full calibration**

```bash
cd "/Users/nmuravya/Projects/ai-res t bank"
uv run python -m experiments.calibrate_speed
```

Expected stdout: near-ceiling accuracy, chance accuracy, measured seconds/step, projected total grid steps and hours, then `GATE PASSED`. Non-zero exit / `AssertionError` means the gate failed.

- [ ] **Step 2: Record the calibration output as evidence**

```bash
mkdir -p experiments/calibration_output
uv run python -m experiments.calibrate_speed | tee experiments/calibration_output/stage2_calibration.txt
```

- [ ] **Step 3: Run the project's full test suite once more**

```bash
uv run pytest tests/test_env_generator.py tests/test_manipulation_check.py tests/test_prompt_templates.py tests/test_policy.py tests/test_calibrate_speed.py -v
```

Expected: all tests pass (Stage 1's 24+ generator/manipulation_check tests, plus Stage 2's 9 prompt_templates + 6 policy + 3 calibrate_speed tests).

- [ ] **Step 4: Commit the calibration evidence + update README status line**

Edit `README.md`, change the `## Status` line from `Stage 0 — repo scaffold.` (or whatever it currently says) to `Stage 2 — policy wrapper + speed calibration gate passed.`

```bash
git add experiments/calibration_output/stage2_calibration.txt README.md
git commit -m "Stage 2 gate: calibration results — near-ceiling/chance controls pass, grid time within budget"
```

**If the gate fails** (per `docs/agent-system.md`'s escalation protocol — these are the first 3 retry attempts before escalating to the Planner):

1. **Near-ceiling accuracy too low** → first check that `rule_context` unambiguously names every topic (compare `TOPIC_NAMES` wording against the actual phrasing in `TOPIC_TEMPLATES` — a mismatch in vocabulary between the rule line and the ticket text is the most likely cause of the model failing to match topic to rule). Try increasing `N_CALIBRATION_TICKETS` first to rule out noise before concluding the prompt itself is the problem. **If the prompt is already unambiguous and accuracy is still below threshold, this is a genuine capability-floor failure, not a prompt bug** — switch `DEFAULT_MODEL_NAME` in `agent/policy.py` to `Qwen/Qwen2.5-1.5B-Instruct` (already cached locally, confirmed — no download needed) and re-run Task 5. The take-home's own "Технологии" section states scale is not evaluated ("Масштаб модели не является частью оценки"), so upgrading here is not a compromise of any requirement — it only trades a bit of the Task 4 throughput measurement, which must be re-measured against the new backbone before re-checking the grid-time budget in point 3. Document the swap as an **architectural amendment** to `docs/materials/PLAN.md` (per `docs/agent-system.md` §4), same as any other backbone change.
2. **Chance accuracy far from 1/8** → verify no systematic length bias: print `len(policy._candidate_token_ids(prompt, c))` for each `c` in `ACTION_LABELS` on a sample prompt — Task 3's tokenizer check already confirmed all 8 labels are 3 tokens each for Qwen2.5-0.5B-Instruct, but re-verify if the backbone changes.
3. **Grid time exceeds budget** → this is the fallback path PLAN.md already anticipated: switch `DEFAULT_MODEL_NAME` in `agent/policy.py` to `HuggingFaceTB/SmolLM2-360M-Instruct` (note: only the base, non-Instruct `SmolLM2-135M` is currently cached locally — a fallback model needs a fresh download, confirm network access first) and re-run Task 5. Document this as an **architectural amendment** to `docs/materials/PLAN.md` (per `docs/agent-system.md` §4) — do not swap the model silently.

Points 1 and 3 pull in opposite directions (bigger-but-slower vs. smaller-but-faster) — if both trigger on the same run (near-ceiling fails *and* projected grid time is already over budget at 0.5B), that is not resolvable by a model swap alone and is an immediate Planner-level escalation: it would mean the closed-set-scoring approach itself, not just the backbone size, needs to change (e.g., a smaller K per step, a smaller grid, or a different scoring strategy).

If none of these resolve it within 3 attempts, this is a Planner-level escalation (per `docs/agent-system.md` §4) — the fix is architectural (e.g., rule phrasing, backbone choice), not a code bug.

---

## Self-Review

**1. Spec coverage** (against `PLAN.md` Stage 2 row + "Итоговый рисёрч-дизайн" + "Структура репозитория"):
- Frozen-backbone, closed-set scoring, no LoRA on critical path → Task 2/3 (`ClosedSetPolicy`, no generation anywhere, no training loop). ✓
- `agent/policy.py` artifact → Task 2/3. ✓
- `agent/prompt_templates.py` (rule-in-context + memory-content plumbing) → Task 1. ✓
- `experiments/calibrate_speed.py` artifact → Task 4. ✓
- Near-ceiling gate (rule given, α=0, no memory) → Task 4's `run_near_ceiling_check`, asserted in Task 5. ✓
- Chance/No-Info control ≈ 1/|A| → Task 4's `run_chance_check`, asserted in Task 5. ✓
- Grid time ≤ 3-4h → Task 4's `measure_batch_seconds_per_step` + `TOTAL_GRID_STEPS` projection, asserted in Task 5. ✓
- Backbone Qwen2.5-0.5B-Instruct with documented fallback → Task 2 (`DEFAULT_MODEL_NAME`) + Task 5's "if the gate fails" section. ✓
- `agent/lora_pretrain.py` explicitly out of scope → stated in Global Constraints, not built. ✓

**2. Placeholder scan:** No TBD/TODO; every step has real, runnable code with concrete assertions, not descriptions.

**3. Type consistency:** `ClosedSetPolicy.__init__/.score_candidates/.predict/.score_candidates_batch` are defined once (Task 2/3) and used with identical names/signatures in Task 4/5. `build_prompt`/`render_rule_context` (Task 1) are called identically in Task 4. `TOTAL_GRID_STEPS`, `NEAR_CEILING_THRESHOLD`, `CHANCE_TOLERANCE`, `GRID_TIME_BUDGET_SECONDS` are defined once in Task 4 and referenced only there and in Task 5's narrative — no drift between definition and usage.

---

**Plan complete and saved to `docs/implementation-plans/2026-08-19-stage2-policy-wrapper-and-speed-calibration.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Maps onto `docs/agent-system.md`'s Kодер role for Tasks 1-4 and the Верификатор role for the Stage 2 gate check in Task 5.

**2. Inline Execution** — I execute tasks in this session directly, batched with checkpoints for your review.

**Which approach?**
