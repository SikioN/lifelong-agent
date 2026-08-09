# DeMem-lite

Decision-aware memory for lifelong text agents: testing whether choosing what to
retain based on downstream decision utility beats plain similarity-based memory
retrieval (motivated by DeMem, arXiv 2605.10870). T-Lab take-home project.

Full research plan (H1/H2, environment, memory methods, all 8 stages): see
[`docs/materials/PLAN.md`](docs/materials/PLAN.md).

## Setup

```bash
uv venv && uv sync
```

or, without `uv`:

```bash
pip install -r requirements.txt
```

## Tests

```bash
uv run pytest tests/test_memory_budget.py tests/test_manipulation_check.py -v
```

Note: run these two files explicitly, not `pytest tests/` — the `tests/` folder
also contains unrelated third-party (superpowers) test files.

## Repository structure

- `env/` — task environment
- `agent/` — agent implementation
- `memory/` — memory method implementations
- `experiments/configs/` — experiment configurations
- `analysis/` — result analysis scripts
- `report/figures/` — generated figures for the final report

## Status

Stage 0 — repo scaffold.
