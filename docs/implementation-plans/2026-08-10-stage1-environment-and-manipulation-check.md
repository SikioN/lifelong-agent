# Stage 1: Environment Generator + Manipulation Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the synthetic ticket-stream environment (`env/generator.py`) that produces a controllable similarity/decision-utility (mis)match, and the standalone `env/manipulation_check.py` script that proves — with real embeddings, not by assumption — that Mode A (α≈0, similarity≈utility) and Mode B (α≥0.5, similarity misleading) actually differ empirically, per the hard gate in `docs/materials/PLAN.md` Stage 1.

**Architecture:** A pure-Python, dependency-light generator (`TicketGenerator`) produces `Ticket` records — support-ticket text + ground-truth `correct_action` — from 8 topics with a random (seeded, non-obvious) `topic → action` base rule, paraphrase templates for surface variation, and a per-tenant static override regime controlled by one continuous parameter `alpha`. Under override, a tenant's effective decision-identity is remapped through a fixed derangement permutation to a *different* topic's action — this single mechanism produces both halves of Mode B (similar topics needing different actions, dissimilar topics sharing an action). `manipulation_check.py` embeds sampled tickets with `all-MiniLM-L6-v2`, computes Spearman correlation between cosine similarity and action-agreement across an alpha sweep, and hard-asserts the gate thresholds from `PLAN.md`.

**Tech Stack:** Python 3.11 (pinned via `.python-version`), plain `.py` scripts (no notebooks — see note below), `numpy` for RNG/permutations, `sentence-transformers` (`all-MiniLM-L6-v2`) for embeddings, `scipy.stats.spearmanr` for correlation (new dependency, added in Task 5), `pandas`/`matplotlib` (`Agg` backend) for CSV/plot output, `pytest` for tests.

**On Colab/T4 vs. local MPS (your question):** Stage 1 has no GPU-bound step. The only model involved is the embedding encoder (`all-MiniLM-L6-v2`, 22M params) — it embeds 300 short sentences in low single-digit seconds on CPU alone; there is nothing here a T4 would meaningfully speed up, and no gradient computation happens in this stage. `PLAN.md` already settled on plain scripts over notebooks specifically for deterministic reproducibility (a fresh clone + one command should reproduce results, which is harder to guarantee from a notebook's mutable execution order). So Stage 1 stays local, CPU/MPS, `.py` scripts — **this is worth revisiting starting Stage 2**, where the causal LM policy (`Qwen2.5-0.5B-Instruct`) does real batched forward passes at grid scale, and MPS throughput becomes an actual open question that Stage 2's `calibrate_speed.py` is explicitly designed to answer. If that calibration shows local MPS is insufficient, an `.ipynb` on Colab T4 becomes the right call *then*, not now — I'll flag it explicitly as an option in the Stage 2 plan.

## Global Constraints

(Copied verbatim from `docs/materials/PLAN.md`, apply to every task below)

- Stage 1 goal: "α-permutation генератор тикетов + скрипт проверки". Artifacts: `env/generator.py`, `env/manipulation_check.py`, plot of Spearman ρ(similarity, action-agreement) vs α.
- **Hard gate (blocking, do not proceed to Stage 2 until it passes):** ρ > ~0.7 at α=0; ρ < ~0.15 at α≥0.5; cross-topic ("dissimilar"), same-action pairs must actually exist under override.
- Environment is built on the paper's α-permutation Decoupled-Bandit construction, not an ad hoc override table: each context has descriptive feature `x_t` (topic) and latent decision-identity `z_t`; with probability `1-α` they're aligned, with probability `α` `z_t` is remapped via a **fixed permutation** to a *different* topic's identity.
- `topic → action` base rule must be a **random, seeded, non-obvious** mapping (not an intuitive real-world pairing) — this is the fix for the "world-knowledge leakage" risk identified during design review.
- Tenant identifier must be rendered **low-salience** (not in the first line / not the dominant token) so the task doesn't trivialize to single-token parsing.
- Semantic-RAG and Decision-Aware memory (Stage 3) will later share **one** embedding function (`all-MiniLM-L6-v2`) — `manipulation_check.py` must use that same model, so its correlation numbers are representative of what Stage 3+ will actually see.
- Python is pinned to `>=3.11,<3.13` (`.python-version` = `3.11`) — a Stage 0 regression (uv defaulting to Python 3.14, hanging on `import torch` for minutes) already burned time on this; do not remove the pin.
- No MCP servers are used anywhere in this project (see `PLAN.md` "Инструменты и MCP") — everything here is plain Python + `uv`.

---

## Task 1: Topic/action vocabulary + seeded permutations

**Files:**
- Create: `env/__init__.py` (empty — makes `env` an explicit regular package)
- Create: `env/generator.py`
- Test: `tests/test_env_generator.py`
- Modify: `pyproject.toml` (add `[tool.pytest.ini_options]` so `from env.generator import ...` resolves when running `uv run pytest` from repo root)

**Interfaces:**
- Produces: `N_TOPICS: int` (module constant, value `8`), `ACTION_LABELS: list[str]` (module constant, `["ACTION_0", ..., "ACTION_7"]`), `build_default_action_map(seed: int) -> dict[int, str]`, `build_mismatch_permutation(seed: int) -> dict[int, int]`.
- Consumes: nothing (first task).

- [ ] **Step 1: Add pytest import path config**

Edit `pyproject.toml`, add this section (anywhere after `[project]`):

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
```

- [ ] **Step 2: Create the `env` package init**

Create `env/__init__.py` with empty content (0 bytes is fine, just needs to exist so `env` resolves as a regular package before `generator.py` exists).

- [ ] **Step 3: Write the failing test**

Create `tests/test_env_generator.py`:

```python
"""Tests for env/generator.py: vocabulary, permutations, tenants, and the
TicketGenerator stream. Grouped by task in this file per the Stage 1 plan."""
import numpy as np
import pytest

from env.generator import (
    N_TOPICS,
    ACTION_LABELS,
    build_default_action_map,
    build_mismatch_permutation,
)


def test_action_labels_has_one_per_topic():
    assert len(ACTION_LABELS) == N_TOPICS
    assert len(set(ACTION_LABELS)) == N_TOPICS  # all unique


def test_default_action_map_is_a_bijection_onto_action_labels():
    mapping = build_default_action_map(seed=42)
    assert len(mapping) == N_TOPICS
    assert set(mapping.keys()) == set(range(N_TOPICS))
    assert set(mapping.values()) == set(ACTION_LABELS)


def test_default_action_map_is_deterministic_given_seed():
    assert build_default_action_map(seed=7) == build_default_action_map(seed=7)


def test_default_action_map_differs_across_seeds():
    # not a hard guarantee in general, but true for these two seeds — pins
    # down that seed actually changes the mapping (catches a hardcoded stub)
    assert build_default_action_map(seed=1) != build_default_action_map(seed=2)


def test_mismatch_permutation_is_a_derangement():
    perm = build_mismatch_permutation(seed=42)
    assert len(perm) == N_TOPICS
    assert set(perm.keys()) == set(range(N_TOPICS))
    assert set(perm.values()) == set(range(N_TOPICS))  # bijection
    for topic_id, mapped_id in perm.items():
        assert mapped_id != topic_id, "mismatch permutation must have no fixed points"


def test_mismatch_permutation_is_deterministic_given_seed():
    assert build_mismatch_permutation(seed=7) == build_mismatch_permutation(seed=7)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_env_generator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'env.generator'` (generator.py doesn't exist yet).

- [ ] **Step 5: Write minimal implementation**

Create `env/generator.py`:

```python
"""Synthetic support-ticket environment for DeMem-lite.

Each ticket has a descriptive topic (surface text, paraphrased) and a
ground-truth correct_action. For tenants in the "override" regime, the
correct action is remapped through a fixed derangement permutation to a
*different* topic's action — this decouples semantic similarity (driven by
topic content) from decision utility (driven by correct_action), per the
alpha-permutation Decoupled Bandit construction in docs/materials/PLAN.md.
"""
import numpy as np

N_TOPICS = 8
ACTION_LABELS = [f"ACTION_{i}" for i in range(N_TOPICS)]


def build_default_action_map(seed: int) -> dict[int, str]:
    """Random (seeded) bijection topic_id -> action label.

    Deliberately non-obvious: action labels are abstract codes, not
    semantically related to topic content, so a pretrained LM cannot guess
    the rule from world knowledge alone (see PLAN.md "утечка знаний из
    претрейна" risk).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_TOPICS)
    return {topic_id: ACTION_LABELS[perm[topic_id]] for topic_id in range(N_TOPICS)}


def _random_derangement(n: int, rng: np.random.Generator) -> list[int]:
    """A permutation of range(n) with no fixed points. Rejection sampling —
    converges in a handful of draws on average (P(derangement) -> 1/e)."""
    while True:
        perm = rng.permutation(n)
        if all(perm[i] != i for i in range(n)):
            return perm.tolist()


def build_mismatch_permutation(seed: int) -> dict[int, int]:
    """Fixed derangement topic_id -> topic_id used to compute the override
    regime's effective decision-identity. No fixed points, so override
    *always* routes to a genuinely different topic's action."""
    rng = np.random.default_rng(seed)
    derangement = _random_derangement(N_TOPICS, rng)
    return {topic_id: derangement[topic_id] for topic_id in range(N_TOPICS)}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_env_generator.py -v`
Expected: PASS (6 tests).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml env/__init__.py env/generator.py tests/test_env_generator.py
git commit -m "Stage 1a: topic/action vocabulary + seeded base-rule and mismatch permutations"
```

---

## Task 2: Paraphrase templates + low-salience ticket text rendering

**Files:**
- Modify: `env/generator.py`
- Modify: `tests/test_env_generator.py`

**Interfaces:**
- Consumes: `N_TOPICS` from Task 1.
- Produces: `TOPIC_TEMPLATES: dict[int, list[str]]` (module constant), `render_ticket_text(topic_id: int, tenant_id: str, rng: np.random.Generator) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_env_generator.py`:

```python
from env.generator import TOPIC_TEMPLATES, render_ticket_text


def test_topic_templates_cover_every_topic_with_multiple_variants():
    assert set(TOPIC_TEMPLATES.keys()) == set(range(N_TOPICS))
    for topic_id, templates in TOPIC_TEMPLATES.items():
        assert len(templates) >= 3, f"topic {topic_id} needs paraphrase variety"
        assert len(set(templates)) == len(templates)  # no duplicate templates


def test_render_ticket_text_uses_a_topic_template():
    rng = np.random.default_rng(0)
    text = render_ticket_text(topic_id=1, tenant_id="T0001", rng=rng)
    body = text.split("\n\n")[0]
    assert body in TOPIC_TEMPLATES[1]


def test_render_ticket_text_tenant_marker_is_low_salience():
    rng = np.random.default_rng(0)
    text = render_ticket_text(topic_id=0, tenant_id="T0042", rng=rng)
    lines = [line for line in text.split("\n") if line.strip()]
    assert "T0042" not in lines[0], "tenant id must not appear in the first line"
    assert "T0042" in text, "tenant id must still be present somewhere"


def test_render_ticket_text_is_deterministic_given_rng_state():
    text_a = render_ticket_text(topic_id=2, tenant_id="T0001", rng=np.random.default_rng(5))
    text_b = render_ticket_text(topic_id=2, tenant_id="T0001", rng=np.random.default_rng(5))
    assert text_a == text_b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_env_generator.py -v`
Expected: FAIL — `ImportError: cannot import name 'TOPIC_TEMPLATES'` (and `render_ticket_text`).

- [ ] **Step 3: Write minimal implementation**

Append to `env/generator.py`:

```python
TOPIC_TEMPLATES: dict[int, list[str]] = {
    0: [  # billing_dispute
        "I was charged twice for the same order and need this fixed.",
        "There's a charge on my statement I don't recognize.",
        "My invoice total doesn't match what I was quoted.",
        "I noticed a duplicate transaction on my card this week.",
    ],
    1: [  # password_reset
        "I can't log into my account, the password reset link isn't arriving.",
        "My login keeps failing even after I reset my password.",
        "I'm locked out of my account and need help getting back in.",
        "The reset email for my password never showed up.",
    ],
    2: [  # data_export_request
        "I need a full export of my account data for my records.",
        "Can you send me a copy of all the data stored under my account?",
        "I'd like to download my complete usage history.",
        "Please provide an export of everything associated with my account.",
    ],
    3: [  # account_deletion
        "I want to permanently close my account and remove my data.",
        "Please delete my account, I no longer want to use the service.",
        "I'd like to cancel and have my account fully removed.",
        "Can you shut down my account and erase my information?",
    ],
    4: [  # refund_request
        "I'd like a refund for a purchase that didn't work as expected.",
        "The item I bought was defective, I want my money back.",
        "Please refund my last payment, I was not satisfied.",
        "I'm requesting a refund since the service didn't meet expectations.",
    ],
    5: [  # shipping_delay
        "My order was supposed to arrive last week and still hasn't shown up.",
        "The tracking hasn't updated in days and my package is late.",
        "I'm still waiting on a delivery that's well past the estimated date.",
        "My shipment seems stuck somewhere and hasn't moved.",
    ],
    6: [  # feature_request
        "It would be great if the app supported dark mode.",
        "Could you add the ability to export reports as PDF?",
        "I'd love to see keyboard shortcuts added to the editor.",
        "Please consider adding a bulk-edit option to the dashboard.",
    ],
    7: [  # bug_report
        "The app crashes every time I try to open the settings page.",
        "I found a bug where the totals don't add up correctly.",
        "The page freezes whenever I upload a large file.",
        "There's a glitch that logs me out randomly during use.",
    ],
}


def render_ticket_text(topic_id: int, tenant_id: str, rng: np.random.Generator) -> str:
    """Render ticket body + a low-salience tenant marker as a trailing
    signature line (not the first line, per PLAN.md's tenant-salience risk)."""
    template = rng.choice(TOPIC_TEMPLATES[topic_id])
    return f"{template}\n\n— submitted via support portal (ref: {tenant_id})"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_env_generator.py -v`
Expected: PASS (10 tests total).

- [ ] **Step 5: Commit**

```bash
git add env/generator.py tests/test_env_generator.py
git commit -m "Stage 1b: paraphrase templates + low-salience ticket text rendering"
```

---

## Task 3: Tenant override regime + correct_action resolution (core Mode A/B logic)

**Files:**
- Modify: `env/generator.py`
- Modify: `tests/test_env_generator.py`

**Interfaces:**
- Consumes: `build_default_action_map`, `build_mismatch_permutation` from Task 1.
- Produces: `Tenant` (frozen dataclass: `tenant_id: str`, `override: bool`), `build_tenants(n_tenants: int, alpha: float, seed: int) -> list[Tenant]`, `resolve_correct_action(topic_id: int, tenant: Tenant, default_action_map: dict[int, str], mismatch_perm: dict[int, int]) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_env_generator.py`:

```python
from env.generator import Tenant, build_tenants, resolve_correct_action


def test_build_tenants_returns_requested_count_with_unique_ids():
    tenants = build_tenants(n_tenants=40, alpha=0.3, seed=1)
    assert len(tenants) == 40
    assert len({t.tenant_id for t in tenants}) == 40


def test_build_tenants_override_fraction_matches_alpha_at_scale():
    tenants = build_tenants(n_tenants=3000, alpha=0.3, seed=1)
    frac_override = sum(t.override for t in tenants) / len(tenants)
    assert abs(frac_override - 0.3) < 0.03


def test_build_tenants_alpha_zero_means_no_override():
    tenants = build_tenants(n_tenants=200, alpha=0.0, seed=1)
    assert all(not t.override for t in tenants)


def test_build_tenants_alpha_one_means_all_override():
    tenants = build_tenants(n_tenants=200, alpha=1.0, seed=1)
    assert all(t.override for t in tenants)


def test_resolve_correct_action_default_regime_uses_direct_mapping():
    default_map = build_default_action_map(seed=42)
    perm = build_mismatch_permutation(seed=43)
    tenant = Tenant(tenant_id="T0", override=False)
    assert resolve_correct_action(0, tenant, default_map, perm) == default_map[0]


def test_resolve_correct_action_override_regime_uses_permuted_mapping():
    default_map = build_default_action_map(seed=42)
    perm = build_mismatch_permutation(seed=43)
    tenant = Tenant(tenant_id="T0", override=True)
    expected = default_map[perm[0]]
    assert resolve_correct_action(0, tenant, default_map, perm) == expected
    # perm has no fixed points (Task 1), so override always changes the result
    assert resolve_correct_action(0, tenant, default_map, perm) != default_map[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_env_generator.py -v`
Expected: FAIL — `ImportError: cannot import name 'Tenant'`.

- [ ] **Step 3: Write minimal implementation**

Append to `env/generator.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Tenant:
    tenant_id: str
    override: bool  # static regime: does this tenant's traffic use the mismatch permutation?


def build_tenants(n_tenants: int, alpha: float, seed: int) -> list[Tenant]:
    """alpha is the traffic-weighted fraction of tenants in override regime.
    Regime is static per tenant (not per-ticket) so it's learnable in-context
    across a tenant's history — this is what memory is supposed to exploit."""
    rng = np.random.default_rng(seed)
    tenants = []
    for i in range(n_tenants):
        tenant_id = f"T{i:04d}"
        override = bool(rng.random() < alpha)
        tenants.append(Tenant(tenant_id=tenant_id, override=override))
    return tenants


def resolve_correct_action(
    topic_id: int,
    tenant: Tenant,
    default_action_map: dict[int, str],
    mismatch_perm: dict[int, int],
) -> str:
    """The ground-truth label the policy/memory system is trying to predict."""
    if tenant.override:
        effective_topic = mismatch_perm[topic_id]
        return default_action_map[effective_topic]
    return default_action_map[topic_id]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_env_generator.py -v`
Expected: PASS (17 tests total).

- [ ] **Step 5: Commit**

```bash
git add env/generator.py tests/test_env_generator.py
git commit -m "Stage 1c: tenant override regime + correct_action resolution (Mode A/B core)"
```

---

## Task 4: `TicketGenerator` — sequential stream + drift schedule

**Files:**
- Modify: `env/generator.py`
- Modify: `tests/test_env_generator.py`

**Interfaces:**
- Consumes: everything from Tasks 1-3 (`build_default_action_map`, `build_mismatch_permutation`, `build_tenants`, `resolve_correct_action`, `render_ticket_text`).
- Produces: `Ticket` (frozen dataclass: `step: int`, `tenant_id: str`, `topic_id: int`, `text: str`, `correct_action: str`, `is_override: bool`), `TicketGenerator` class with `__init__(self, alpha: float, n_tenants: int = 40, seed: int = 0, drift_period: int | None = None)`, `.action_space -> list[str]` property, `.sample(step: int) -> Ticket`, `.stream(n_steps: int) -> Iterator[Ticket]`. This is the public API Stage 2 (`agent/policy.py`) and Stage 3 (`memory/*.py`) will consume — get names/types right here, they're load-bearing for every later stage.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_env_generator.py`:

```python
from env.generator import Ticket, TicketGenerator


def test_ticket_generator_stream_yields_requested_length():
    gen = TicketGenerator(alpha=0.3, seed=5)
    tickets = list(gen.stream(50))
    assert len(tickets) == 50
    assert all(isinstance(t, Ticket) for t in tickets)


def test_ticket_generator_is_deterministic_given_seed():
    stream_a = [t.correct_action for t in TicketGenerator(alpha=0.3, seed=5).stream(50)]
    stream_b = [t.correct_action for t in TicketGenerator(alpha=0.3, seed=5).stream(50)]
    assert stream_a == stream_b


def test_ticket_generator_alpha_zero_never_overrides():
    gen = TicketGenerator(alpha=0.0, seed=5)
    tickets = list(gen.stream(200))
    assert all(not t.is_override for t in tickets)


def test_ticket_generator_actions_are_within_action_space():
    gen = TicketGenerator(alpha=0.5, seed=5)
    for ticket in gen.stream(30):
        assert ticket.correct_action in gen.action_space


def test_ticket_generator_step_field_matches_position_in_stream():
    gen = TicketGenerator(alpha=0.2, seed=5)
    tickets = list(gen.stream(10))
    assert [t.step for t in tickets] == list(range(10))


def test_ticket_generator_drift_period_toggles_override_regime():
    # single tenant, base regime = override (alpha=1.0), drift_period=10:
    # steps 0-9 -> unflipped (True), 10-19 -> flipped (False), 20+ -> True again
    gen = TicketGenerator(alpha=1.0, seed=9, n_tenants=1, drift_period=10)
    tenant_id = next(iter(gen.tenants))
    regimes = [gen._current_tenant(tenant_id, step).override for step in (0, 5, 10, 15, 20)]
    assert regimes == [True, True, False, False, True]


def test_ticket_generator_no_drift_period_means_static_regime():
    gen = TicketGenerator(alpha=1.0, seed=9, n_tenants=1, drift_period=None)
    tenant_id = next(iter(gen.tenants))
    regimes = [gen._current_tenant(tenant_id, step).override for step in (0, 50, 500)]
    assert regimes == [True, True, True]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_env_generator.py -v`
Expected: FAIL — `ImportError: cannot import name 'Ticket'`.

- [ ] **Step 3: Write minimal implementation**

Append to `env/generator.py`:

```python
from typing import Iterator


@dataclass(frozen=True)
class Ticket:
    step: int
    tenant_id: str
    topic_id: int
    text: str
    correct_action: str
    is_override: bool


class TicketGenerator:
    """Sequential ticket stream with a controllable similarity/utility
    mismatch (alpha) and an optional drift schedule (for Stage 5's H2:
    recurring/circular concept test)."""

    def __init__(
        self,
        alpha: float,
        n_tenants: int = 40,
        seed: int = 0,
        drift_period: int | None = None,
    ):
        self.alpha = alpha
        self.drift_period = drift_period
        self.default_action_map = build_default_action_map(seed=seed)
        self.mismatch_perm = build_mismatch_permutation(seed=seed + 1)
        self.tenants: dict[str, Tenant] = {
            t.tenant_id: t for t in build_tenants(n_tenants, alpha, seed=seed + 2)
        }
        self._rng = np.random.default_rng(seed + 3)
        self._tenant_ids = list(self.tenants.keys())

    @property
    def action_space(self) -> list[str]:
        return list(ACTION_LABELS)

    def _current_tenant(self, tenant_id: str, step: int) -> Tenant:
        base = self.tenants[tenant_id]
        if self.drift_period is None:
            return base
        flips = step // self.drift_period
        override = base.override if flips % 2 == 0 else not base.override
        return Tenant(tenant_id=base.tenant_id, override=override)

    def sample(self, step: int) -> Ticket:
        tenant_id = self._rng.choice(self._tenant_ids)
        topic_id = int(self._rng.integers(0, N_TOPICS))
        tenant = self._current_tenant(tenant_id, step)
        text = render_ticket_text(topic_id, tenant_id, self._rng)
        correct_action = resolve_correct_action(
            topic_id, tenant, self.default_action_map, self.mismatch_perm
        )
        return Ticket(
            step=step,
            tenant_id=tenant_id,
            topic_id=topic_id,
            text=text,
            correct_action=correct_action,
            is_override=tenant.override,
        )

    def stream(self, n_steps: int) -> Iterator[Ticket]:
        for step in range(n_steps):
            yield self.sample(step)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_env_generator.py -v`
Expected: PASS (24 tests total).

- [ ] **Step 5: Commit**

```bash
git add env/generator.py tests/test_env_generator.py
git commit -m "Stage 1d: TicketGenerator stream API + drift schedule for H2"
```

---

## Task 5: Similarity/action-agreement correlation (unit-testable core of the gate)

**Files:**
- Modify: `pyproject.toml` (add `scipy` dependency)
- Create: `env/manipulation_check.py`
- Create: `tests/test_manipulation_check.py` (already has one Stage-0 smoke test — extend it, don't replace)

**Interfaces:**
- Consumes: `TicketGenerator` from Task 4.
- Produces: `cosine_similarity(a: np.ndarray, b: np.ndarray) -> float`, `compute_similarity_action_correlation(alpha: float, n_tickets: int, seed: int, encoder) -> float`, `SWEEP_ALPHAS: list[float]` (module constant, `[0.0, 0.15, 0.3, 0.5, 0.7]`), `N_PAIRS: int` (module constant, `3000`).

- [ ] **Step 1: Add the `scipy` dependency**

Edit `pyproject.toml`, add `"scipy>=1.11",` to the `dependencies` list (needed for `scipy.stats.spearmanr`).

Run: `uv sync`
Expected: installs scipy and its deps without touching torch/transformers versions.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_manipulation_check.py`:

```python
from env.manipulation_check import cosine_similarity, compute_similarity_action_correlation


def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-9


def test_correlation_is_high_at_alpha_zero_and_drops_at_high_alpha():
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    rho_aligned = compute_similarity_action_correlation(
        alpha=0.0, n_tickets=150, seed=0, encoder=encoder
    )
    rho_mismatched = compute_similarity_action_correlation(
        alpha=0.7, n_tickets=150, seed=0, encoder=encoder
    )
    assert rho_aligned > 0.5
    assert rho_mismatched < rho_aligned
```

Add `import numpy as np` to the top of `tests/test_manipulation_check.py` if not already present (it currently only imports `SentenceTransformer`).

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_manipulation_check.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'env.manipulation_check'`.

- [ ] **Step 4: Write minimal implementation**

Create `env/manipulation_check.py`:

```python
"""Stage 1 hard gate: proves that alpha actually decouples semantic
similarity from decision utility, using the real embedding model that
Stage 3's Semantic-RAG and Decision-Aware memory will both share.

Run directly: `uv run python env/manipulation_check.py`
"""
import matplotlib
matplotlib.use("Agg")  # headless — no display available when run via CLI/CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer

from env.generator import TicketGenerator

SWEEP_ALPHAS = [0.0, 0.15, 0.3, 0.5, 0.7]
N_TICKETS = 300
N_PAIRS = 3000
OUTPUT_DIR = Path(__file__).parent / "manipulation_check_output"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_similarity_action_correlation(
    alpha: float, n_tickets: int, seed: int, encoder: SentenceTransformer
) -> float:
    """Spearman rho between pairwise cosine similarity and whether the two
    tickets share the same correct_action, over N_PAIRS random pairs."""
    generator = TicketGenerator(alpha=alpha, seed=seed)
    tickets = [generator.sample(step) for step in range(n_tickets)]
    embeddings = encoder.encode([t.text for t in tickets])

    rng = np.random.default_rng(seed + 1000)
    idx_pairs = rng.integers(0, n_tickets, size=(N_PAIRS, 2))

    sims: list[float] = []
    agreements: list[int] = []
    for i, j in idx_pairs:
        if i == j:
            continue
        sims.append(cosine_similarity(embeddings[i], embeddings[j]))
        agreements.append(int(tickets[i].correct_action == tickets[j].correct_action))

    rho, _p_value = spearmanr(sims, agreements)
    return float(rho)


def check_cross_topic_same_action_pairs(alpha: float, seed: int, n_tickets: int = N_TICKETS) -> int:
    """Gate requirement: under override, dissimilar (different-topic)
    tickets sharing the same correct_action must actually exist."""
    generator = TicketGenerator(alpha=alpha, seed=seed)
    tickets = [generator.sample(step) for step in range(n_tickets)]
    count = sum(
        1
        for i in range(len(tickets))
        for j in range(i + 1, len(tickets))
        if tickets[i].topic_id != tickets[j].topic_id
        and tickets[i].correct_action == tickets[j].correct_action
    )
    assert count > 0, f"Gate failed: no cross-topic same-action pairs found at alpha={alpha}"
    print(f"cross-topic same-action pairs at alpha={alpha}: {count}")
    return count


def main() -> None:
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    rows = []
    for alpha in SWEEP_ALPHAS:
        rho = compute_similarity_action_correlation(
            alpha=alpha, n_tickets=N_TICKETS, seed=123, encoder=encoder
        )
        print(f"alpha={alpha:.2f}  rho={rho:.3f}")
        rows.append({"alpha": alpha, "rho": rho})

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)

    fig, ax = plt.subplots()
    ax.plot(df["alpha"], df["rho"], marker="o")
    ax.axhline(0.7, color="green", linestyle="--", linewidth=0.8, label="gate: rho>0.7 @ alpha=0")
    ax.axhline(0.15, color="red", linestyle="--", linewidth=0.8, label="gate: rho<0.15 @ alpha>=0.5")
    ax.set_xlabel("alpha (similarity/utility mismatch)")
    ax.set_ylabel("Spearman rho(similarity, action-agreement)")
    ax.set_title("Stage 1 manipulation check")
    ax.legend()
    fig.savefig(OUTPUT_DIR / "rho_vs_alpha.png", dpi=150)
    plt.close(fig)

    rho_at_zero = df.loc[df["alpha"] == 0.0, "rho"].item()
    assert rho_at_zero > 0.7, f"GATE FAILED: rho(alpha=0)={rho_at_zero:.3f}, need > 0.7"

    for alpha in (0.5, 0.7):
        rho_high = df.loc[df["alpha"] == alpha, "rho"].item()
        assert rho_high < 0.15, f"GATE FAILED: rho(alpha={alpha})={rho_high:.3f}, need < 0.15"

    check_cross_topic_same_action_pairs(alpha=0.5, seed=123)

    print(f"\nGATE PASSED. Results written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_manipulation_check.py -v`
Expected: PASS (all tests, including the pre-existing Stage 0 embedding smoke test).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock env/manipulation_check.py tests/test_manipulation_check.py
git commit -m "Stage 1e: similarity/action-agreement correlation + manipulation_check core"
```

---

## Task 6: Full sweep script — run for real, confirm the hard gate

**Files:** none new — this task *executes* `env/manipulation_check.py` end-to-end and records the outcome. No code changes unless the gate fails (see "If the gate fails" below).

- [ ] **Step 1: Run the full manipulation check**

```bash
cd "/Users/nmuravya/Desktop/ai-res t bank"
uv run python env/manipulation_check.py
```

Expected stdout: one `alpha=... rho=...` line per sweep value, the cross-topic pair count, and `GATE PASSED`. Non-zero exit / `AssertionError` means the gate failed.

- [ ] **Step 2: Inspect the output artifacts**

```bash
cat env/manipulation_check_output/results.csv
```
Confirm: `rho` at `alpha=0.0` is `>0.7`; `rho` at `alpha=0.5` and `alpha=0.7` are both `<0.15`; the trend across all 5 alphas is monotonically non-increasing (matches the paper's own headline mismatch-mechanism figure).

Open `env/manipulation_check_output/rho_vs_alpha.png` and visually confirm the curve crosses both dashed gate lines in the right direction.

- [ ] **Step 3: Run the full test suite once more**

```bash
uv run pytest tests/ --ignore=tests/antigravity --ignore=tests/brainstorm-server --ignore=tests/claude-code --ignore=tests/codex --ignore=tests/codex-plugin-sync --ignore=tests/explicit-skill-requests --ignore=tests/hooks --ignore=tests/kimi --ignore=tests/opencode --ignore=tests/pi --ignore=tests/shell-lint --ignore=tests/systematic-debugging -v
```
(The `--ignore` flags skip superpowers' own test subdirectories that live in the same physical `tests/` folder — only our `test_env_generator.py` and `test_manipulation_check.py` should run. Simpler day-to-day: `uv run pytest tests/test_env_generator.py tests/test_manipulation_check.py -v` and pass explicit paths, as Stage 0's README already documents.)

Expected: all tests pass (24 generator tests + manipulation_check tests).

- [ ] **Step 4: Commit the output artifacts as Stage 1 evidence**

```bash
git add env/manipulation_check_output/results.csv env/manipulation_check_output/rho_vs_alpha.png
git commit -m "Stage 1 gate: manipulation check results — alpha decouples similarity from utility"
```

**If the gate fails** (per `docs/agent-system.md`'s escalation protocol — this is the Coder's first 3 retry attempts before escalating to the Planner): the most likely causes, in order of likelihood, and what to try:
1. `rho(alpha=0)` too low → topic templates aren't semantically distinct enough for MiniLM to cluster cleanly. Try: increase `N_TICKETS`, or add 1-2 more templates per topic with more topic-specific vocabulary.
2. `rho(alpha>=0.5)` too high → derangement isn't decoupling enough, or `N_PAIRS` is too small for a stable correlation estimate. Try: increase `N_PAIRS` to 5000+; verify `build_mismatch_permutation` really has zero fixed points (Task 1 test already checks this, but re-verify with a different seed).
3. Cross-topic same-action assertion fails at `alpha=0.5` → statistically possible but unlikely with `n_tickets=300` and 8 topics; increase `n_tickets` in `check_cross_topic_same_action_pairs` before concluding something is structurally wrong.

If none of these resolve it within 3 attempts, this is a Planner-level escalation (per `docs/agent-system.md` §4) — the fix is architectural (e.g., topic/template design), not a code bug.

---

## Self-Review

**1. Spec coverage** (against `PLAN.md` Stage 1 row + "Итоговый рисёрч-дизайн" §2):
- α-permutation construction (not ad hoc override table) → Task 1 + Task 3. ✓
- Both halves of Mode B (similar→different action, dissimilar→same action) → Task 3's `resolve_correct_action` (similar topics diverge under override) + Task 5/6's `check_cross_topic_same_action_pairs` (dissimilar topics converge). ✓
- Non-obvious topic→action rule (world-knowledge leakage fix) → Task 1, abstract `ACTION_i` labels + random permutation. ✓
- Low-salience tenant marker → Task 2, tested explicitly (not first line). ✓
- Paraphrase pool (retrieval ≠ bag-of-words) → Task 2, 4 templates/topic. ✓
- Drift schedule (needed by Stage 5, listed in Stage 1's file responsibility) → Task 4, implemented and unit-tested now even though not exercised until Stage 5. ✓
- Manipulation check + hard gate thresholds → Task 5 (unit) + Task 6 (real run, real assertions). ✓
- Shared embedding model with future Stage 3 → Task 5 uses `all-MiniLM-L6-v2` explicitly. ✓

**2. Placeholder scan:** No TBD/TODO; every step has real code, not descriptions of code.

**3. Type consistency:** `Ticket`, `Tenant`, `TicketGenerator.action_space`/`.sample()`/`.stream()` are defined once in Task 4 and referenced identically (same names, same signatures) in Task 5/6 and in this document's "Interfaces" blocks — Stage 2/3 plans should import these exact names.

---

**Plan complete and saved to `docs/implementation-plans/2026-08-10-stage1-environment-and-manipulation-check.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Maps onto `docs/agent-system.md`'s Kоder role for Tasks 1-5 and the Verifier for the Stage 1 gate check in Task 6.

**2. Inline Execution** — I execute tasks in this session directly, batched with checkpoints for your review.

**Which approach?**
