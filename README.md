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

Stage 2 complete -- policy wrapper and label-prior calibration reviewed;
the real GPU-based gate run (backbone `Qwen2.5-1.5B-Instruct`, Colab T4,
batch=4) passed its accuracy and bias-calibration criteria
(near-ceiling=1.000 vs threshold ≥0.85, calibrated chance=0.150 vs target
0.125±0.10) but failed its time-budget criterion (projected grid time
62.31h vs a 9h single-GPU-session budget) -- as anticipated, since the
full grid (177,500 steps) was never expected to fit in one session.
Execution moved from a local MacBook (insufficient) through Kaggle to
Google Colab (single T4) after real, evidence-driven infrastructure
findings -- see [`docs/materials/PLAN.md`](docs/materials/PLAN.md)'s
Архитектурная правка 3 and
[`docs/implementation-plans/2026-08-20-stage2-kaggle-redo-design.md`](docs/implementation-plans/2026-08-20-stage2-kaggle-redo-design.md)
for the full story. The real Stage 4/5 grid will need to be split across
multiple GPU sessions (e.g. by α value) rather than run in one, per the
gate's own outcome (see `kaggle_runner/runs/final-gate/gate_output.txt`
for the raw run output).
