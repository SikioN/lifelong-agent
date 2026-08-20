# Stage 2 Kaggle Redo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore Stage 2 to its last reviewed-clean state, add a bias-diagnosis/calibration mechanism to `ClosedSetPolicy` to investigate and fix the chance-accuracy anomaly found on the interrupted Kaggle run (raw 0.267 vs ~0.125 target), build reviewable Kaggle orchestration tooling, and use it to choose a backbone from real evidence and confirm the Stage 2 gate for real on Kaggle T4.

**Architecture:** `agent/policy.py` gets a pure `calibrate_scores` function plus a `measure_label_prior` convenience method on `ClosedSetPolicy`, keeping the policy prompt-agnostic (no new dependency on `agent/prompt_templates.py`). `experiments/calibrate_speed.py` gets a `neutral_prompt()` helper and a calibrated chance-check, and its `GRID_TIME_BUDGET_SECONDS` is re-grounded to a real Kaggle GPU session's wall-clock limit. `experiments/diagnose_label_bias.py` is new: it sweeps candidate backbones (via a `MODEL_NAMES` env var) and reports near-ceiling / raw-chance / calibrated-chance / throughput for each, so the final backbone is chosen from a table, not intuition. `kaggle_runner/orchestrate.py` is a small, unit-tested wrapper around the `kaggle` CLI (push/poll/pull) that renders a script-kernel payload with a pinned commit and entrypoint baked in as literals, avoiding any dependency on Kaggle's environment-variable support.

**Tech Stack:** Python 3.11 (local dev machine; Kaggle's own image Python version is out of this project's control — noted as a risk), `torch`/`transformers` (unchanged from Stage 2's original Tasks 1-4), `numpy`, `pytest`, `kaggle` CLI (already installed and credentialed on this machine, confirmed working).

**Spec:** `docs/implementation-plans/2026-08-20-stage2-kaggle-redo-design.md` — the plan below argues from this spec; executors should read both.

## Global Constraints

(Copied verbatim/paraphrased from the spec — apply to every task below)

- **Starting point is commit `e1fc413`** (the last Stage 2 state that was actually reviewed clean via `superpowers:subagent-driven-development`) — Task 1 restores `agent/policy.py`, `experiments/calibrate_speed.py`, and `README.md` to that exact content before any new work begins. Everything after `e1fc413` on the current branch (commits `730aa06`..`29a5e2b`) was unreviewed and is being replaced, not built on top of.
- **No force-push, no history rewrite.** All fixes are new commits on top of the current `master` tip — the unreviewed commits stay visible in history.
- `GRID_TIME_BUDGET_SECONDS = 9 * 3600` (one free-tier Kaggle GPU session) — not the original plan's CPU/MPS-era `4 * 3600`, and not the unreviewed session's unmotivated `20 * 3600`.
- `TOTAL_GRID_STEPS = 177,500`, computed verbatim from `docs/materials/PLAN.md`'s Stage 4/5 table (`5*2*5*5*400 + 1*4*5*5*400 + 2*5*5*750`) — never silently shrunk to make a run fit.
- `NEAR_CEILING_THRESHOLD = 0.85`, `CHANCE_TOLERANCE = 0.10` — the plan's original values. Never loosened to make a failing gate pass; if a backbone doesn't clear these, the fix is calibration or a different backbone, not a wider tolerance.
- **No notebooks carrying real logic.** The Kaggle kernel payload is a plain `.py` script (Kaggle `"kernel_type": "script"`), not a hand-edited `.ipynb` cell.
- **Single-GPU code path.** Kaggle's accelerator is `GPU_T4x2` (two GPUs), but this project's code only targets one (`cuda:0` via the existing `_select_device()`, unchanged) — no multi-GPU orchestration. This is a deliberate non-goal, not an oversight.
- No LoRA / weight fine-tuning anywhere in this plan.
- `agent/prompt_templates.py` from Stage 2's original Task 1 is untouched by this plan.

---

## Task 1: Restore Stage 2 baseline, remove unreviewed artifacts, fix repo hygiene

**Files:**
- Modify (restore from `e1fc413`): `agent/policy.py`, `experiments/calibrate_speed.py`, `README.md`
- Modify: `.gitignore` (add `__pycache__/` and `*.pyc`)
- Modify: `requirements.txt` (add missing `scipy>=1.11`, already present in `pyproject.toml` since Stage 1 but never synced here)
- Delete (tracked): `agent/__pycache__/*.pyc`, `env/__pycache__/*.pyc`, `experiments/__pycache__/*.pyc`, `experiments/calibration_output/stage2_calibration.txt`, `test_prompt.py`, `test_smol.py`, the entire current `kaggle_runner/` tree

**Interfaces:**
- Consumes: nothing (first task).
- Produces: a clean `agent/policy.py`/`experiments/calibrate_speed.py` matching `e1fc413` exactly, ready for Task 2/3 to build on; a `.gitignore` that won't let `__pycache__` be committed again.

- [ ] **Step 1: Restore the three files to their `e1fc413` content**

```bash
cd "/Users/nmuravya/Projects/ai-res t bank"
git checkout e1fc413 -- agent/policy.py experiments/calibrate_speed.py README.md
```

- [ ] **Step 2: Verify the restore is exact**

Run: `git diff e1fc413 -- agent/policy.py experiments/calibrate_speed.py README.md`
Expected: empty output (no diff) — confirms these three files now match `e1fc413` byte-for-byte.

- [ ] **Step 3: Add pycache patterns to `.gitignore`**

Edit `.gitignore`, add near the top (after the existing `.DS_Store` line):

```
__pycache__/
*.pyc
```

- [ ] **Step 4: Sync `requirements.txt` with `pyproject.toml`**

Edit `requirements.txt`, add after the `pytest>=8.0` line:

```
scipy>=1.11
```

- [ ] **Step 5: Remove tracked `__pycache__`/`.pyc` files from git**

```bash
git rm -r --cached agent/__pycache__ env/__pycache__ experiments/__pycache__
```

Expected: git reports the `.pyc` files as removed from the index (they remain gitignored on disk going forward, regenerated automatically by Python).

- [ ] **Step 6: Remove the stray root-level test scripts and the invalid calibration evidence**

```bash
git rm test_prompt.py test_smol.py experiments/calibration_output/stage2_calibration.txt
```

(`experiments/calibration_output/stage2_calibration.txt` recorded the outcome of the compromised gate — thresholds loosened, grid shrunk — so it does not represent valid Stage 2 evidence and must not be kept as if it did.)

- [ ] **Step 7: Remove the old `kaggle_runner/` tree entirely**

```bash
git rm -r kaggle_runner
rm -rf kaggle_runner
```

(The second command is a safety net for any nested/untracked content `git rm -r` doesn't reach — e.g. `kaggle_runner/output/lifelong-agent` was tracked as a git submodule-style gitlink from a nested `git clone`, and `kaggle_runner/output5/` was untracked. Task 5 rebuilds a clean `kaggle_runner/` from scratch.)

- [ ] **Step 8: Fix the stale README Status line**

Edit `README.md`, replace:

```markdown
## Status

Stage 0 — repo scaffold.
```

with:

```markdown
## Status

Stage 2 in progress — policy wrapper and calibration built and reviewed
(commit `e1fc413`). Execution moved to Kaggle T4 (no local GPU); see
[`docs/implementation-plans/2026-08-20-stage2-kaggle-redo-design.md`](docs/implementation-plans/2026-08-20-stage2-kaggle-redo-design.md).
```

- [ ] **Step 9: Run the full test suite to confirm a clean baseline**

```bash
uv run pytest tests/test_env_generator.py tests/test_manipulation_check.py tests/test_prompt_templates.py tests/test_policy.py tests/test_calibrate_speed.py -v
```

Expected: every test passes, zero failures (this is the same file set and code Task 4 of the original Stage 2 plan reviewed clean — restoring `e1fc413`'s content must reproduce that clean result).

- [ ] **Step 10: Commit**

```bash
git add .gitignore requirements.txt README.md agent/policy.py experiments/calibrate_speed.py
git commit -m "Restore Stage 2 to reviewed e1fc413 baseline, remove unreviewed Kaggle session artifacts

Reverts (forward, no history rewrite) commits 730aa06..29a5e2b: unreviewed
threshold loosening (CHANCE_TOLERANCE, GRID_TIME_BUDGET_SECONDS), grid
shrinkage (TOTAL_GRID_STEPS 177500->13000), committed .pyc files, stray
root-level test scripts, and the ad hoc kaggle_runner/ tree. See
docs/implementation-plans/2026-08-20-stage2-kaggle-redo-design.md."
```

---

## Task 2: `agent/policy.py` — label-prior calibration

**Files:**
- Modify: `agent/policy.py`
- Modify: `tests/test_policy.py`

**Interfaces:**
- Consumes: `ClosedSetPolicy.score_candidates` (restored in Task 1, unchanged).
- Produces: `calibrate_scores(raw_scores: np.ndarray, prior: np.ndarray) -> np.ndarray` (module-level function), `ClosedSetPolicy.measure_label_prior(self, prompt: str, candidates: list[str]) -> np.ndarray` (method). Task 3's `experiments/calibrate_speed.py` and Task 4's `experiments/diagnose_label_bias.py` both import and call these by these exact names.

- [ ] **Step 1: Write the failing tests**

Edit `tests/test_policy.py`, change the import line at the top from:

```python
from agent.policy import ClosedSetPolicy
```

to:

```python
from agent.policy import ClosedSetPolicy, calibrate_scores
```

Then append to the end of the file:

```python
def test_calibrate_scores_single_vector_subtracts_prior():
    raw_scores = np.array([1.0, 2.0, 3.0])
    prior = np.array([0.5, 0.5, 0.5])
    calibrated = calibrate_scores(raw_scores, prior)
    assert np.allclose(calibrated, [0.5, 1.5, 2.5])


def test_calibrate_scores_batched_broadcasts_prior_per_row():
    raw_scores = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    prior = np.array([1.0, 1.0, 1.0])
    calibrated = calibrate_scores(raw_scores, prior)
    assert np.allclose(calibrated, [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])


def test_calibrate_scores_removes_a_biased_prior_that_would_flip_the_argmax():
    # candidate 1 has a much higher prior for no task-relevant reason; the
    # actual task signal (equal across all three) carries no preference.
    prior = np.array([1.0, 5.0, 2.0])
    signal = np.array([0.0, 0.0, 0.0])
    raw_scores = prior + signal
    assert np.argmax(raw_scores) == 1  # uncalibrated: biased toward candidate 1
    calibrated = calibrate_scores(raw_scores, prior)
    assert np.allclose(calibrated, [0.0, 0.0, 0.0])  # bias removed


def test_measure_label_prior_returns_one_finite_score_per_candidate(policy):
    prior = policy.measure_label_prior("Pick one:", ["ACTION_0", "ACTION_1", "ACTION_2"])
    assert prior.shape == (3,)
    assert np.all(np.isfinite(prior))


def test_measure_label_prior_is_deterministic(policy):
    prior_a = policy.measure_label_prior("Pick one:", ["ACTION_0", "ACTION_1"])
    prior_b = policy.measure_label_prior("Pick one:", ["ACTION_0", "ACTION_1"])
    assert np.allclose(prior_a, prior_b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_policy.py -v`
Expected: FAIL — `ImportError: cannot import name 'calibrate_scores'`.

- [ ] **Step 3: Write minimal implementation**

Append to `agent/policy.py` (inside the `ClosedSetPolicy` class, after `predict`):

```python
    def measure_label_prior(self, prompt: str, candidates: list[str]) -> np.ndarray:
        """Convenience alias for score_candidates, named for the calibration
        use case: call this with a neutral prompt (no task-specific
        information) to find the model's baseline preference among
        candidates before any real signal is added. See calibrate_scores
        below for what to do with the result."""
        return self.score_candidates(prompt, candidates)
```

Then append after the end of the `ClosedSetPolicy` class (module level, outside the class body):

```python
def calibrate_scores(raw_scores: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Subtract a previously-measured label prior from raw closed-set
    scores, broadcasting over the last axis. Removes whatever preference
    among candidates exists independent of the prompt's actual content --
    see ClosedSetPolicy.measure_label_prior. Works for a single prompt's
    score vector (n_candidates,) and a batch's score matrix
    (n_prompts, n_candidates) alike, since numpy broadcasts a
    (n_candidates,) prior across either shape's last axis."""
    return raw_scores - prior
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_policy.py -v`
Expected: PASS (11 tests: the 6 already there plus 5 new).

- [ ] **Step 5: Commit**

```bash
git add agent/policy.py tests/test_policy.py
git commit -m "Add label-prior calibration (calibrate_scores, measure_label_prior) to ClosedSetPolicy"
```

---

## Task 3: `experiments/calibrate_speed.py` — calibrated chance-check + re-grounded grid-time budget

**Files:**
- Modify: `experiments/calibrate_speed.py`
- Modify: `tests/test_calibrate_speed.py`

**Interfaces:**
- Consumes: `calibrate_scores`, `ClosedSetPolicy.measure_label_prior` (Task 2).
- Produces: `neutral_prompt() -> str`, `run_chance_check_calibrated(policy: ClosedSetPolicy, prior: np.ndarray, n_tickets: int = N_CALIBRATION_TICKETS) -> float`, `GRID_TIME_BUDGET_SECONDS = 9 * 3600`. Task 4's `experiments/diagnose_label_bias.py` imports `neutral_prompt`, `run_chance_check_calibrated`, `run_chance_check`, `run_near_ceiling_check`, `measure_batch_seconds_per_step`, and `N_CALIBRATION_TICKETS` from this file by these exact names.

- [ ] **Step 1: Write the failing tests**

Edit `tests/test_calibrate_speed.py`, replace the import block at the top:

```python
import pytest

from agent.policy import ClosedSetPolicy
from experiments.calibrate_speed import (
    measure_batch_seconds_per_step,
    run_chance_check,
    run_near_ceiling_check,
)
```

with:

```python
import pytest

from agent.policy import ClosedSetPolicy
from agent.prompt_templates import build_prompt
from env.generator import ACTION_LABELS
from experiments.calibrate_speed import (
    measure_batch_seconds_per_step,
    neutral_prompt,
    run_chance_check,
    run_chance_check_calibrated,
    run_near_ceiling_check,
)
```

Then append to the end of the file:

```python
def test_neutral_prompt_matches_build_prompt_with_empty_ticket_and_no_rule_or_memory():
    assert neutral_prompt() == build_prompt("", ACTION_LABELS)


def test_run_chance_check_calibrated_returns_accuracy_in_unit_interval(policy):
    prior = policy.measure_label_prior(neutral_prompt(), ACTION_LABELS)
    accuracy = run_chance_check_calibrated(policy, prior, n_tickets=8)
    assert 0.0 <= accuracy <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_calibrate_speed.py -v`
Expected: FAIL — `ImportError: cannot import name 'neutral_prompt'`.

- [ ] **Step 3: Write minimal implementation**

Edit `experiments/calibrate_speed.py`. Change the import block at the top from:

```python
import time

from agent.policy import ClosedSetPolicy
from agent.prompt_templates import build_prompt, render_rule_context
from env.generator import ACTION_LABELS, TicketGenerator
```

to:

```python
import time

import numpy as np

from agent.policy import ClosedSetPolicy, calibrate_scores
from agent.prompt_templates import build_prompt, render_rule_context
from env.generator import ACTION_LABELS, TicketGenerator
```

Change the grid-time budget line from:

```python
GRID_TIME_BUDGET_SECONDS = 4 * 3600  # upper end of PLAN.md's "3-4 hours"
```

to:

```python
# Free-tier Kaggle GPU sessions cap at ~9h wall-clock; this replaces the
# original CPU/MPS-era 4h estimate now that Stage 2+ runs on Kaggle T4 (see
# docs/implementation-plans/2026-08-20-stage2-kaggle-redo-design.md). If a
# projected run exceeds this, split the grid across multiple kernel runs --
# do not raise this constant to make a run fit.
GRID_TIME_BUDGET_SECONDS = 9 * 3600  # one Kaggle free-tier GPU session
```

Add these two new functions after `run_chance_check` (keep `run_chance_check` itself unchanged -- it stays as the "raw", uncalibrated reference the diagnostic script compares against):

```python
def neutral_prompt() -> str:
    """The 'no information' prompt: same scaffold as a real ticket prompt,
    but with no ticket text, no rule, no memory -- isolates whatever score
    differences exist when nothing in the prompt justifies preferring one
    label over another. See agent.policy.ClosedSetPolicy.measure_label_prior."""
    return build_prompt("", ACTION_LABELS)


def run_chance_check_calibrated(
    policy: ClosedSetPolicy, prior: np.ndarray, n_tickets: int = N_CALIBRATION_TICKETS
) -> float:
    """Same condition as run_chance_check (no rule, no memory) but predicts
    from calibrate_scores(raw_scores, prior) instead of raw argmax -- this
    is the Stage 2 gate's real chance-condition check going forward, since
    calibration is now part of the policy's intended behavior, not an
    optional extra."""
    generator = TicketGenerator(alpha=0.0, seed=778)
    correct = 0
    for step in range(n_tickets):
        ticket = generator.sample(step)
        prompt = build_prompt(ticket.text, ACTION_LABELS)
        raw_scores = policy.score_candidates(prompt, ACTION_LABELS)
        calibrated = calibrate_scores(raw_scores, prior)
        prediction = ACTION_LABELS[int(np.argmax(calibrated))]
        correct += int(prediction == ticket.correct_action)
    return correct / n_tickets
```

Replace `main()` entirely with:

```python
def main() -> None:
    policy = ClosedSetPolicy()
    prior = policy.measure_label_prior(neutral_prompt(), ACTION_LABELS)

    near_ceiling_acc = run_near_ceiling_check(policy)
    chance_acc = run_chance_check_calibrated(policy, prior)
    seconds_per_step = measure_batch_seconds_per_step(policy)
    projected_total_seconds = seconds_per_step * TOTAL_GRID_STEPS

    chance_target = 1 / len(ACTION_LABELS)
    print(f"near-ceiling accuracy (rule given, alpha=0): {near_ceiling_acc:.3f}")
    print(f"calibrated chance accuracy (no rule, no memory): {chance_acc:.3f}  (target ~{chance_target:.3f})")
    print(f"measured seconds/step (batched, batch=16):    {seconds_per_step:.4f}")
    print(f"projected total grid steps:                   {TOTAL_GRID_STEPS:,}")
    print(f"projected total grid time:                    {projected_total_seconds / 3600:.2f} hours")

    assert near_ceiling_acc >= NEAR_CEILING_THRESHOLD, (
        f"GATE FAILED: near-ceiling accuracy {near_ceiling_acc:.3f} < {NEAR_CEILING_THRESHOLD}"
    )
    assert abs(chance_acc - chance_target) <= CHANCE_TOLERANCE, (
        f"GATE FAILED: calibrated chance accuracy {chance_acc:.3f} not within "
        f"{CHANCE_TOLERANCE} of {chance_target:.3f}"
    )
    assert projected_total_seconds <= GRID_TIME_BUDGET_SECONDS, (
        f"GATE FAILED: projected grid time {projected_total_seconds / 3600:.2f}h "
        f"exceeds budget {GRID_TIME_BUDGET_SECONDS / 3600:.1f}h -- split the grid "
        f"across multiple Kaggle kernel runs rather than raising this budget"
    )

    print("\nGATE PASSED.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_calibrate_speed.py -v`
Expected: PASS (5 tests: the 3 already there plus 2 new).

- [ ] **Step 5: Commit**

```bash
git add experiments/calibrate_speed.py tests/test_calibrate_speed.py
git commit -m "Wire calibration into the Stage 2 gate, re-ground GRID_TIME_BUDGET_SECONDS to a Kaggle GPU session (9h)"
```

---

## Task 4: `experiments/diagnose_label_bias.py` — backbone sweep

**Files:**
- Create: `experiments/diagnose_label_bias.py`
- Create: `tests/test_diagnose_label_bias.py`
- Modify: `.gitignore` (whitelist the new test file)

**Interfaces:**
- Consumes: `ClosedSetPolicy` (Task 1/2), `ACTION_LABELS` (Stage 1), `N_CALIBRATION_TICKETS`, `measure_batch_seconds_per_step`, `neutral_prompt`, `run_chance_check`, `run_chance_check_calibrated`, `run_near_ceiling_check` (Task 3).
- Produces: `DEFAULT_CANDIDATE_MODELS: list[str]`, `diagnose_bias(policy: ClosedSetPolicy, n_tickets: int = N_CALIBRATION_TICKETS) -> dict`. Task 6 runs this script's `main()` for real via `kaggle_runner`.

- [ ] **Step 1: Add the `.gitignore` allowlist entry**

Edit `.gitignore`, add after the `!/tests/test_calibrate_speed.py` line:

```
!/tests/test_diagnose_label_bias.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_diagnose_label_bias.py`:

```python
"""Tests for experiments/diagnose_label_bias.py, run against the real
cached Qwen2.5-0.5B-Instruct model with a tiny n_tickets so the suite
stays fast. The real backbone sweep across multiple/larger models runs on
Kaggle as an execution-only task, not here."""
import pytest

from agent.policy import ClosedSetPolicy
from env.generator import ACTION_LABELS
from experiments.diagnose_label_bias import diagnose_bias


@pytest.fixture(scope="module")
def policy() -> ClosedSetPolicy:
    return ClosedSetPolicy()


def test_diagnose_bias_returns_all_expected_keys(policy):
    result = diagnose_bias(policy, n_tickets=8)
    assert set(result.keys()) == {
        "near_ceiling_accuracy",
        "raw_chance_accuracy",
        "calibrated_chance_accuracy",
        "seconds_per_step",
        "label_prior",
    }


def test_diagnose_bias_accuracies_are_in_unit_interval(policy):
    result = diagnose_bias(policy, n_tickets=8)
    for key in ("near_ceiling_accuracy", "raw_chance_accuracy", "calibrated_chance_accuracy"):
        assert 0.0 <= result[key] <= 1.0


def test_diagnose_bias_label_prior_has_one_entry_per_action(policy):
    result = diagnose_bias(policy, n_tickets=8)
    assert len(result["label_prior"]) == len(ACTION_LABELS)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_diagnose_label_bias.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'experiments.diagnose_label_bias'`.

- [ ] **Step 4: Write minimal implementation**

Create `experiments/diagnose_label_bias.py`:

```python
"""Stage 2 bias diagnosis + backbone sweep.

Investigates the chance-accuracy anomaly found on the interrupted Kaggle
run (raw chance accuracy 0.267 vs ~0.125 target -- see
docs/implementation-plans/2026-08-20-stage2-kaggle-redo-design.md) before
any backbone is chosen: measures each candidate label's baseline
log-likelihood on a neutral prompt (ClosedSetPolicy.measure_label_prior
against calibrate_speed.neutral_prompt), and reports whether subtracting
that prior (calibrate_scores) closes the gap to chance. Then sweeps a list
of candidate backbones and reports near-ceiling / raw chance / calibrated
chance / throughput for each, so the final backbone is chosen from
evidence, not intuition. This script prints a comparison table -- it does
not assert a pass/fail gate itself; experiments/calibrate_speed.py's
main() is the gate, run afterward for whichever backbone this sweep
supports choosing.

Run locally (small model, cheap): `uv run python -m experiments.diagnose_label_bias`
Run a specific backbone: `MODEL_NAMES="Qwen/Qwen2.5-3B-Instruct" uv run python -m experiments.diagnose_label_bias`
Run several: `MODEL_NAMES="Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct" uv run python -m experiments.diagnose_label_bias`
"""
import json
import os
from pathlib import Path

from agent.policy import ClosedSetPolicy
from env.generator import ACTION_LABELS
from experiments.calibrate_speed import (
    N_CALIBRATION_TICKETS,
    measure_batch_seconds_per_step,
    neutral_prompt,
    run_chance_check,
    run_chance_check_calibrated,
    run_near_ceiling_check,
)

DEFAULT_CANDIDATE_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]
OUTPUT_DIR = Path(__file__).parent / "diagnostic_output"


def diagnose_bias(policy: ClosedSetPolicy, n_tickets: int = N_CALIBRATION_TICKETS) -> dict:
    """Runs near-ceiling, raw chance, and calibrated chance for one loaded
    policy, plus the label prior and measured throughput -- everything
    needed to judge whether this backbone is both capable and unbiased."""
    prior = policy.measure_label_prior(neutral_prompt(), ACTION_LABELS)
    near_ceiling = run_near_ceiling_check(policy, n_tickets=n_tickets)
    raw_chance = run_chance_check(policy, n_tickets=n_tickets)
    calibrated_chance = run_chance_check_calibrated(policy, prior, n_tickets=n_tickets)
    seconds_per_step = measure_batch_seconds_per_step(policy)
    return {
        "near_ceiling_accuracy": near_ceiling,
        "raw_chance_accuracy": raw_chance,
        "calibrated_chance_accuracy": calibrated_chance,
        "seconds_per_step": seconds_per_step,
        "label_prior": prior.tolist(),
    }


def main() -> None:
    raw = os.environ.get("MODEL_NAMES")
    candidate_models = raw.split(",") if raw else DEFAULT_CANDIDATE_MODELS

    chance_target = 1 / len(ACTION_LABELS)
    print(f"Backbone sweep -- chance target: {chance_target:.3f}\n")

    results = []
    for model_name in candidate_models:
        print(f"=== {model_name} ===")
        policy = ClosedSetPolicy(model_name)
        result = diagnose_bias(policy)
        result["model_name"] = model_name
        print(f"  near-ceiling accuracy:      {result['near_ceiling_accuracy']:.3f}")
        print(f"  raw chance accuracy:        {result['raw_chance_accuracy']:.3f}")
        print(f"  calibrated chance accuracy: {result['calibrated_chance_accuracy']:.3f}")
        print(f"  seconds/step:               {result['seconds_per_step']:.4f}")
        print(f"  label prior:                {[round(p, 2) for p in result['label_prior']]}\n")
        results.append(result)

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "backbone_sweep.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_diagnose_label_bias.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add .gitignore experiments/diagnose_label_bias.py tests/test_diagnose_label_bias.py
git commit -m "Add experiments/diagnose_label_bias.py: backbone sweep with bias diagnosis"
```

---

## Task 5: `kaggle_runner/` — Kaggle orchestration tooling

**Files:**
- Create: `kaggle_runner/__init__.py`
- Create: `kaggle_runner/kernel-metadata.json`
- Create: `kaggle_runner/orchestrate.py`
- Create: `tests/test_kaggle_orchestrate.py`
- Modify: `.gitignore` (whitelist the new test file)

**Interfaces:**
- Consumes: nothing from earlier tasks (this is infrastructure tooling, independent of Stage 2's ML code).
- Produces: `render_kernel_script(repo_commit: str, entrypoint: str) -> str`, `push(kernel_dir: Path, repo_commit: str, entrypoint: str) -> str`, `get_status(kernel_id: str) -> str`, `wait_for_completion(kernel_id: str, poll_interval_s: float = 30.0, timeout_s: float = 3600.0) -> str`, `pull_output(kernel_id: str, dest_dir: Path) -> Path`. Task 6 calls all four of `push`/`wait_for_completion`/`pull_output` directly against the real Kaggle account.

- [ ] **Step 1: Create the package init**

Create `kaggle_runner/__init__.py` with empty content.

- [ ] **Step 2: Create `kernel-metadata.json`**

Create `kaggle_runner/kernel-metadata.json`:

```json
{
  "id": "nmuravya/lifelong-agent-stage2",
  "title": "Lifelong Agent Stage 2",
  "code_file": "kernel_template.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "accelerator": "GPU_T4x2",
  "enable_internet": "true",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

(`"accelerator": "GPU_T4x2"` matches the value the prior session's kernel actually ran with successfully -- this project's code intentionally only uses one of the two GPUs, per the Global Constraints' single-GPU non-goal; `kernel_template.py` doesn't exist as a static file yet -- `push()` renders and writes it at push time, see Step 6. Note also Task 6's accelerator caveat: pushing via CLI does not reliably grant T4x2 in practice, regardless of this field -- the real T4x2 run is launched manually.)

- [ ] **Step 3: Add the `.gitignore` allowlist entry**

Edit `.gitignore`, add after the `!/tests/test_diagnose_label_bias.py` line:

```
!/tests/test_kaggle_orchestrate.py
```

- [ ] **Step 4: Write the failing tests**

Create `tests/test_kaggle_orchestrate.py`:

```python
"""Tests for kaggle_runner/orchestrate.py. Mocks subprocess.run so these
tests don't hit the real Kaggle API -- Task 6 does a real push/poll/pull
round trip against the live account as an execution-only step."""
import json
from unittest.mock import MagicMock, patch

import pytest

from kaggle_runner.orchestrate import (
    get_status,
    pull_output,
    push,
    render_kernel_script,
    wait_for_completion,
)


def test_render_kernel_script_bakes_in_commit_and_entrypoint():
    script = render_kernel_script("deadbeef", "experiments.calibrate_speed")
    assert 'REPO_COMMIT = "deadbeef"' in script
    assert 'ENTRYPOINT = "experiments.calibrate_speed"' in script


def test_push_renders_script_writes_it_and_calls_kaggle_kernels_push(tmp_path):
    metadata = {"id": "someuser/some-kernel"}
    (tmp_path / "kernel-metadata.json").write_text(json.dumps(metadata))
    with patch("kaggle_runner.orchestrate.subprocess.run") as mock_run:
        kernel_id = push(tmp_path, repo_commit="abc123", entrypoint="experiments.diagnose_label_bias")
    written_script = (tmp_path / "kernel_template.py").read_text()
    assert "abc123" in written_script
    assert "experiments.diagnose_label_bias" in written_script
    mock_run.assert_called_once_with(
        ["kaggle", "kernels", "push", "-p", str(tmp_path)], check=True
    )
    assert kernel_id == "someuser/some-kernel"


def test_get_status_parses_kaggle_cli_output():
    fake_result = MagicMock(stdout='someuser/some-kernel has status "complete"\n')
    with patch("kaggle_runner.orchestrate.subprocess.run", return_value=fake_result) as mock_run:
        status = get_status("someuser/some-kernel")
    mock_run.assert_called_once_with(
        ["kaggle", "kernels", "status", "someuser/some-kernel"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert status == "complete"


def test_wait_for_completion_polls_until_terminal_status():
    statuses = iter(["running", "running", "complete"])
    with patch("kaggle_runner.orchestrate.get_status", side_effect=lambda _id: next(statuses)):
        with patch("kaggle_runner.orchestrate.time.sleep") as mock_sleep:
            status = wait_for_completion("someuser/some-kernel", poll_interval_s=0.01, timeout_s=10)
    assert status == "complete"
    assert mock_sleep.call_count == 2


def test_wait_for_completion_raises_timeout_error_when_never_terminal():
    with patch("kaggle_runner.orchestrate.get_status", return_value="running"):
        with patch("kaggle_runner.orchestrate.time.monotonic", side_effect=[0.0, 0.0, 100.0]):
            with pytest.raises(TimeoutError):
                wait_for_completion("someuser/some-kernel", poll_interval_s=0.01, timeout_s=10)


def test_pull_output_calls_kaggle_kernels_output_and_creates_dest(tmp_path):
    dest = tmp_path / "out"
    with patch("kaggle_runner.orchestrate.subprocess.run") as mock_run:
        result = pull_output("someuser/some-kernel", dest)
    mock_run.assert_called_once_with(
        ["kaggle", "kernels", "output", "someuser/some-kernel", "-p", str(dest)],
        check=True,
    )
    assert result == dest
    assert dest.exists()
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `uv run pytest tests/test_kaggle_orchestrate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kaggle_runner.orchestrate'`.

- [ ] **Step 6: Write minimal implementation**

Create `kaggle_runner/orchestrate.py`:

```python
"""Local-side tooling to drive a Kaggle script-kernel run via the `kaggle`
CLI: push the kernel, poll until it finishes, and pull its output back.
Wraps subprocess calls to the CLI rather than talking to Kaggle's API
directly, since `kaggle` is already the credentialed, working entry point
on this machine (see docs/implementation-plans/2026-08-20-stage2-kaggle-redo-design.md).

The pushed kernel is a script-kernel (kernel_type: "script"), not a
notebook -- its content is generated by render_kernel_script() with the
target repo commit and entrypoint module baked in as literals, rather than
relying on Kaggle's environment-variable support (not verified for this
project's use case)."""
import json
import subprocess
import time
from pathlib import Path

TERMINAL_STATUSES = {"complete", "error", "cancelAcknowledged"}

KERNEL_SCRIPT_TEMPLATE = '''\
"""Kaggle script-kernel entrypoint, generated by kaggle_runner.orchestrate.push().
Not run locally, not imported by the rest of this project."""
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/SikioN/lifelong-agent.git"
REPO_COMMIT = {repo_commit!r}
ENTRYPOINT = {entrypoint!r}


def main() -> None:
    subprocess.run(["git", "clone", REPO_URL, "repo"], check=True)
    subprocess.run(["git", "-C", "repo", "checkout", REPO_COMMIT], check=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "repo/requirements.txt"],
        check=True,
    )
    subprocess.run([sys.executable, "-m", ENTRYPOINT], check=True, cwd="repo")

    # Kaggle captures whatever's in the working directory as kernel output --
    # copy the entrypoint's results here explicitly rather than relying on
    # where it happened to write them relative to the cloned repo.
    for output_dir_name in ("calibration_output", "diagnostic_output"):
        src = Path("repo/experiments") / output_dir_name
        if src.exists():
            shutil.copytree(src, output_dir_name, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
'''


def render_kernel_script(repo_commit: str, entrypoint: str) -> str:
    """Fills in the Kaggle script-kernel template with a pinned commit and
    the module to run -- baked in as literals so the pushed kernel doesn't
    depend on Kaggle's environment-variable support."""
    return KERNEL_SCRIPT_TEMPLATE.format(repo_commit=repo_commit, entrypoint=entrypoint)


def push(kernel_dir: Path, repo_commit: str, entrypoint: str) -> str:
    """Renders the kernel script for (repo_commit, entrypoint), writes it
    alongside kernel-metadata.json in kernel_dir, pushes via the kaggle
    CLI, and returns the kernel id read from kernel-metadata.json."""
    script = render_kernel_script(repo_commit, entrypoint)
    (kernel_dir / "kernel_template.py").write_text(script)
    subprocess.run(["kaggle", "kernels", "push", "-p", str(kernel_dir)], check=True)
    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text())
    return metadata["id"]


def get_status(kernel_id: str) -> str:
    """Returns the kernel's current run status, e.g. "running", "complete",
    "error" (see `kaggle kernels status --help` for the full set)."""
    result = subprocess.run(
        ["kaggle", "kernels", "status", kernel_id],
        check=True,
        capture_output=True,
        text=True,
    )
    # kaggle CLI prints e.g.: {kernel_id} has status "complete"
    return result.stdout.split('"')[1]


def wait_for_completion(kernel_id: str, poll_interval_s: float = 30.0, timeout_s: float = 3600.0) -> str:
    """Polls get_status until it reaches a terminal status or timeout_s
    elapses. Returns the terminal status; raises TimeoutError on timeout."""
    start = time.monotonic()
    while True:
        status = get_status(kernel_id)
        if status in TERMINAL_STATUSES:
            return status
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(
                f"kernel {kernel_id} did not finish within {timeout_s}s (last status: {status})"
            )
        time.sleep(poll_interval_s)


def pull_output(kernel_id: str, dest_dir: Path) -> Path:
    """Downloads the kernel's output files into dest_dir, returns dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "kernels", "output", kernel_id, "-p", str(dest_dir)],
        check=True,
    )
    return dest_dir
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_kaggle_orchestrate.py -v`
Expected: PASS (6 tests).

- [ ] **Step 8: Commit**

```bash
git add .gitignore kaggle_runner/__init__.py kaggle_runner/kernel-metadata.json kaggle_runner/orchestrate.py tests/test_kaggle_orchestrate.py
git commit -m "Add kaggle_runner/orchestrate.py: push/poll/pull wrapper around the kaggle CLI"
```

---

## Task 6: Real Kaggle round trip — diagnose bias, choose backbone, confirm the gate

**Files:** none new — this task *executes* the tooling from Tasks 1-5 against the real Kaggle account. Code changes happen only in the specific, narrow case called out in Step 5 (setting the chosen `DEFAULT_MODEL_NAME`) and if a platform issue in `kaggle_runner/orchestrate.py`'s generated script needs a fix (see troubleshooting below).

**Accelerator caveat (confirmed by the human partner before this task started):** pushing via the `kaggle` CLI reliably launches the run on a `P100`, regardless of the `"GPU_T4x2"` set in `kernel-metadata.json` — a known gap in how the CLI's push-triggered auto-run honors the accelerator field. The actual T4x2 run has to be started manually on kaggle.com, with the accelerator explicitly selected there. **This makes this task a real-time handoff between the agent and the human partner, not a fire-and-forget script run** — do not dispatch this as an unattended implementer expecting `push()` alone to produce the T4x2 run; whoever executes this task pushes the code, then stops and asks the human partner to launch it, then resumes once they confirm.

- [ ] **Step 1: Push the backbone-sweep code, then hand off to the human partner to launch it with T4x2**

```bash
cd "/Users/nmuravya/Projects/ai-res t bank"
git rev-parse HEAD   # note this commit hash -- this is what gets pinned into the kernel
```

```bash
uv run python -c "
from pathlib import Path
from kaggle_runner.orchestrate import push

commit = 'PASTE_THE_HASH_FROM_ABOVE'
kernel_id = push(Path('kaggle_runner'), repo_commit=commit, entrypoint='experiments.diagnose_label_bias')
print('pushed:', kernel_id)
"
```

This uploads the kernel and (per the caveat above) will auto-run on a P100 -- that auto-run's result does not matter and can be ignored. Stop here and tell the human partner: *"Pushed kernel `<kernel_id>`. Please open it on kaggle.com, set the accelerator to GPU T4 x2, and Save & Run All. Let me know once it's running (or done)."* Do not proceed to Step 2 until they confirm they've started the T4x2 run.

- [ ] **Step 2: Poll for completion and pull the results**

Once the human partner confirms the T4x2 run is underway:

```bash
uv run python -c "
from pathlib import Path
from kaggle_runner.orchestrate import wait_for_completion, pull_output

kernel_id = 'PASTE_THE_ID_FROM_STEP_1'
status = wait_for_completion(kernel_id, poll_interval_s=30, timeout_s=3600)
print('finished with status:', status)
pull_output(kernel_id, Path('kaggle_runner/runs/backbone-sweep'))
"
```

Expected: `status` prints `complete`; `kaggle_runner/runs/backbone-sweep/diagnostic_output/backbone_sweep.json` exists locally afterward. (`wait_for_completion`/`pull_output` only need the kernel id -- they work the same whether the run was launched by `push()` or manually on the website.)

- [ ] **Step 3: Inspect the sweep results**

```bash
cat kaggle_runner/runs/backbone-sweep/diagnostic_output/backbone_sweep.json
```

For each candidate model, read off `near_ceiling_accuracy`, `raw_chance_accuracy`, `calibrated_chance_accuracy`, `seconds_per_step`, `label_prior`. Confirm the diagnostic hypothesis: does `calibrated_chance_accuracy` land closer to `0.125` than `raw_chance_accuracy` did for every candidate? If not for a given model, its `label_prior` values (printed per-candidate) are the next thing to look at -- a genuinely flat prior with calibration still not helping would mean the bias isn't the simple prompt-independent kind `calibrate_scores` targets (see the design spec's risk section).

- [ ] **Step 4: Choose the backbone**

From the table, pick the smallest/fastest candidate that satisfies all three, using the **calibrated** chance accuracy (not raw):

- `near_ceiling_accuracy >= 0.85`
- `abs(calibrated_chance_accuracy - 0.125) <= 0.10`
- `seconds_per_step * 177_500 <= 32_400` (9h)

If more than one candidate qualifies, prefer the smaller/faster one (matches the original plan's "don't pay for a bigger model than needed" philosophy) -- being bigger is not itself a tiebreaker in favor of a candidate, per the design spec's finding that model size didn't reduce bias here.

- [ ] **Step 5: Set the chosen backbone**

Edit `agent/policy.py`, change:

```python
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
```

to the chosen model name, and add a one-line comment citing the evidence, e.g.:

```python
# Chosen from the Stage 2 backbone sweep (near-ceiling, calibrated chance,
# throughput) -- see kaggle_runner/runs/backbone-sweep/diagnostic_output/backbone_sweep.json
DEFAULT_MODEL_NAME = "<chosen model name>"
```

(If the sweep's own default backbone, `Qwen/Qwen2.5-0.5B-Instruct`, is the one chosen, this is a comment-only change -- still make it, so the evidence trail is explicit rather than implicit.)

- [ ] **Step 6: Run the local test suite once more**

```bash
uv run pytest tests/test_env_generator.py tests/test_manipulation_check.py tests/test_prompt_templates.py tests/test_policy.py tests/test_calibrate_speed.py tests/test_diagnose_label_bias.py tests/test_kaggle_orchestrate.py -v
```

Expected: all pass (these tests default to `Qwen/Qwen2.5-0.5B-Instruct` for their own fixtures regardless of `DEFAULT_MODEL_NAME`, so this just confirms the `agent/policy.py` edit didn't break anything).

- [ ] **Step 7: Commit the backbone choice**

```bash
git add agent/policy.py
git commit -m "Choose Stage 2 backbone from the real Kaggle sweep (see kaggle_runner/runs/backbone-sweep)"
git rev-parse HEAD   # note this new commit hash -- pin this into the final-gate kernel
```

- [ ] **Step 8: Push the final-gate code, then hand off again for the T4x2 launch**

```bash
uv run python -c "
from pathlib import Path
from kaggle_runner.orchestrate import push

commit = 'PASTE_THE_HASH_FROM_STEP_7'
kernel_id = push(Path('kaggle_runner'), repo_commit=commit, entrypoint='experiments.calibrate_speed')
print('pushed:', kernel_id)
"
```

Same handoff as Step 1: tell the human partner the kernel id, ask them to open it on kaggle.com, select GPU T4 x2, and Save & Run All. Wait for their confirmation before continuing.

- [ ] **Step 9: Poll, pull, and inspect the final gate output**

```bash
uv run python -c "
from pathlib import Path
from kaggle_runner.orchestrate import wait_for_completion, pull_output

kernel_id = 'PASTE_THE_ID_FROM_STEP_8'
status = wait_for_completion(kernel_id, poll_interval_s=30, timeout_s=3600)
print('finished with status:', status)
pull_output(kernel_id, Path('kaggle_runner/runs/final-gate'))
"
```

```bash
cat kaggle_runner/runs/final-gate/calibration_output/*.log 2>/dev/null || kaggle kernels output "$(python -c "import json; print(json.load(open('kaggle_runner/kernel-metadata.json'))['id'])")" -p /dev/stdout 2>&1 | tail -20
```

Confirm the log ends with `GATE PASSED.` printed for real, from `experiments/calibrate_speed.py`'s own asserts (`NEAR_CEILING_THRESHOLD=0.85`, `CHANCE_TOLERANCE=0.10`, `GRID_TIME_BUDGET_SECONDS=32,400`) against the chosen backbone -- not a loosened or shrunk version of that gate.

- [ ] **Step 10: Commit the pulled evidence**

```bash
git add kaggle_runner/runs/
git commit -m "Record real Kaggle Stage 2 gate evidence (backbone sweep + final calibration run, T4x2)"
```

**If any step fails:**
1. **A candidate's calibrated chance accuracy still fails the tolerance** after Step 3's inspection shows a non-flat, prompt-dependent bias -- this is the scenario the design spec's risk section anticipated (calibration-on-empty-prompt not fully closing the gap). This is a Planner-level finding, not a retry: it means either a richer calibration (e.g., averaging the prior over a few varied neutral-ish prompts instead of one fixed string) or accepting a wider, still-justified tolerance is needed -- do not silently loosen `CHANCE_TOLERANCE` to route around it; escalate with the actual numbers.
2. **No candidate fits the 9h budget** -- per the design spec, split the grid across multiple kernel runs (e.g. one run per `alpha` value) rather than raising `GRID_TIME_BUDGET_SECONDS`. This is a real architectural change to how Stage 4/5 will eventually run, worth a note in Task 7's `PLAN.md` amendment either way.
3. **`kaggle kernels push`/`status`/`output` fails outright** (auth, quota, malformed `kernel-metadata.json`) -- read the CLI's own error message first; these are usually self-explanatory (e.g. a quota exhausted message tells you directly). Fix `kaggle_runner/orchestrate.py` or `kernel-metadata.json` as needed and re-push; this is expected first-real-run friction, matching Stage 1's own precedent of iterating on `manipulation_check.py` based on real run output.
4. **The generated kernel script fails inside the Kaggle sandbox** (e.g. `pip install` from `requirements.txt` errors, `git clone` fails because the repo is private and `enable_internet`/auth isn't sufficient) -- read the pulled kernel log (`kaggle kernels output <id> -p <dir>` includes the run log), fix the issue in `kaggle_runner/orchestrate.py`'s `KERNEL_SCRIPT_TEMPLATE`, and re-run from Step 1. If the repo needs to be public or a token needs to be embedded for `git clone` to work from inside the Kaggle sandbox, that is itself a finding worth recording in Task 7's `PLAN.md` amendment.
5. **The human partner's manually-launched run (Step 1 or Step 8) uses a different accelerator than expected, or the manual launch step itself doesn't behave as described** -- the accelerator caveat above is based on one observed session; if reality differs (e.g. CLI push starts honoring `GPU_T4x2` directly, or the website's manual flow works differently than assumed), don't force the plan's described workaround -- note what actually happened and adjust the handoff for the remaining steps accordingly.

---

## Task 7: `PLAN.md` amendment, final verification

**Files:**
- Modify: `docs/materials/PLAN.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 6's actual results (chosen backbone, sweep numbers, final gate numbers) -- read them from `kaggle_runner/runs/backbone-sweep/diagnostic_output/backbone_sweep.json` and `kaggle_runner/runs/final-gate/`, do not guess or reuse this plan's placeholder text.
- Produces: nothing further tasks consume -- this is the closing task of the plan.

- [ ] **Step 1: Add the architectural amendment to `PLAN.md`**

Read `docs/materials/PLAN.md`'s existing "Архитектурная правка 1" and "Архитектурная правка 2" entries (in the Stage 1 row's notes, under "Этапы выполнения") for the established format. Add a new "**Архитектурная правка 3 (Stage 2, эскалация Планировщика → подтверждено человеком)**" entry immediately after them, covering (fill in the actual numbers from Task 6's evidence, not placeholder text):

- Reason for the change: no local GPU; Stage 2/4/5's real compute now runs on Kaggle T4 (free tier, `GPU_T4x2` requested, single-GPU code path used).
- The chance-accuracy bias found on the first real Kaggle run (raw `0.267` vs `~0.125` target) and the calibration fix added (`agent.policy.calibrate_scores` / `measure_label_prior`).
- The final chosen backbone and the evidence it was chosen from (cite `kaggle_runner/runs/backbone-sweep/diagnostic_output/backbone_sweep.json`).
- `GRID_TIME_BUDGET_SECONDS` re-grounded from the original plan's CPU/MPS-era `4*3600` to `9*3600` (one Kaggle GPU session), and what the actual projected grid time came out to for the chosen backbone.
- A pointer to `docs/implementation-plans/2026-08-20-stage2-kaggle-redo-design.md` and this plan file for full detail.

- [ ] **Step 2: Update `README.md`'s Status line**

Edit `README.md`, replace the Status section (set in Task 1 Step 8) with a line reflecting the real outcome, e.g.:

```markdown
## Status

Stage 2 complete -- policy wrapper, calibration, and Kaggle T4 execution
gate all reviewed and passing (backbone: `<chosen model name>`). See
[`docs/materials/PLAN.md`](docs/materials/PLAN.md) for the full research
plan and [`docs/implementation-plans/2026-08-20-stage2-kaggle-redo-design.md`](docs/implementation-plans/2026-08-20-stage2-kaggle-redo-design.md)
for how Stage 2's execution moved to Kaggle.
```

- [ ] **Step 3: Run the full local test suite one final time**

```bash
uv run pytest tests/test_env_generator.py tests/test_manipulation_check.py tests/test_prompt_templates.py tests/test_policy.py tests/test_calibrate_speed.py tests/test_diagnose_label_bias.py tests/test_kaggle_orchestrate.py -v
```

Expected: all pass, zero failures.

- [ ] **Step 4: Confirm a clean working tree**

```bash
git status --porcelain
```

Expected: empty (everything from this plan is committed; no stray `kaggle_runner/runs/` output, no leftover `.pyc`, nothing from the old unreviewed session remains).

- [ ] **Step 5: Commit**

```bash
git add docs/materials/PLAN.md README.md
git commit -m "Stage 2 Kaggle redo: PLAN.md architectural amendment + final status"
```

---

## Self-Review

**1. Spec coverage** (against `docs/implementation-plans/2026-08-20-stage2-kaggle-redo-design.md`):
- Section 1 (bias diagnosis + calibration in `agent/policy.py`) → Task 2. ✓
- Section 2 (backbone selection sweep) → Task 4, executed for real in Task 6. ✓
- Section 3 (grid-time budget re-grounded) → Task 3 (`GRID_TIME_BUDGET_SECONDS = 9*3600`, `TOTAL_GRID_STEPS` unchanged at 177,500 since Task 1 already restored it). ✓
- Section 4 (Kaggle orchestration tooling) → Task 5, exercised in Task 6. ✓
- Section 5 (repo cleanup) → Task 1. ✓
- Section 6 (`PLAN.md` amendment) → Task 7. ✓
- Testing approach (pure-function unit tests for `calibrate_scores`/`render_kernel_script`, real-model tests for `measure_label_prior`, mocked-subprocess tests for orchestration, one real execution-only round trip) → Tasks 2/4/5 for the unit tests, Task 6 for the real round trip. ✓
- Risk: Kaggle session limit not verified against the account → Task 6's troubleshooting section calls this out explicitly if timing doesn't match. ✓
- Risk: calibration assumes prompt-independent bias → Task 6 Step 2 explicitly directs inspecting whether calibration closes the gap, and the troubleshooting section's item 1 covers the case where it doesn't. ✓
- Risk: Kaggle image's preinstalled package versions may not match pins → `kernel_template.py`'s generated script explicitly `pip install`s from `requirements.txt` (fixed by Task 1) rather than trusting whatever's preinstalled. ✓

**2. Placeholder scan:** No TBD/TODO. Task 6 and Task 7 contain instructions like "paste the hash from above" and "fill in the actual numbers" because they depend on this task's own real-world output (the commit hash it just created, the Kaggle run's real results) -- this is the same pattern as the original Stage 1/Stage 2 plans' execution-only tasks, not a spec gap.

**3. Type consistency:** `calibrate_scores(raw_scores: np.ndarray, prior: np.ndarray) -> np.ndarray` and `ClosedSetPolicy.measure_label_prior(self, prompt: str, candidates: list[str]) -> np.ndarray` (Task 2) are used with identical signatures in Task 3 (`experiments/calibrate_speed.py`) and Task 4 (`experiments/diagnose_label_bias.py`). `neutral_prompt() -> str` and `run_chance_check_calibrated(policy, prior, n_tickets=...) -> float` (Task 3) are imported and called identically in Task 4. `push`/`get_status`/`wait_for_completion`/`pull_output` (Task 5) are called with matching signatures in Task 6's inline scripts.

---

**Plan complete and saved to `docs/implementation-plans/2026-08-20-stage2-kaggle-redo-plan.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — I execute tasks in this session directly, batched with checkpoints for your review.

**Which approach?**
