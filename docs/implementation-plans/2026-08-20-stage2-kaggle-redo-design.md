# Stage 2 Kaggle Redo — Design

## Problem statement

Stage 2's implementation tasks (Tasks 1-4: `agent/prompt_templates.py`, `agent/policy.py`'s `ClosedSetPolicy`, `experiments/calibrate_speed.py`) were built and reviewed clean via `superpowers:subagent-driven-development`, landing at commit `e1fc413` on `master`. Task 5 — running the calibration for real to confirm the Stage 2 gate — was dispatched, but the session was interrupted mid-task. During the interruption, 9 further commits (`730aa06`..`29a5e2b`) landed on `master` and were pushed to `origin` (`github.com/SikioN/lifelong-agent`) with no task brief, no review, and no ledger entry. They:

- pivoted execution to Kaggle (introducing `kaggle_runner/`, `.ipynb` notebooks, nested repo clones) without documenting this as a plan amendment,
- weakened two gate thresholds specifically to make a failing gate pass (`CHANCE_TOLERANCE` 0.10→0.15, `GRID_TIME_BUDGET_SECONDS` 4h→20h) — explicitly forbidden in the Task 5 dispatch,
- shrank the experiment grid the gate is supposed to be calibrating for (`TOTAL_GRID_STEPS` 177,500→13,000, via seeds 5→1 and T 400/750→150/250),
- committed `.pyc` files and ad hoc root-level `test_*.py` scripts,
- never updated `docs/materials/PLAN.md` to reflect any of this.

Separately, the real Kaggle run (`Qwen2.5-3B-Instruct`) surfaced a genuine finding: chance-condition accuracy (no rule, no memory given) came back `0.267` against a target of `~0.125` — a statistically significant deviation (z≈3.3 at n=60, p<0.001), not noise. The human partner confirmed via brainstorming: local compute is genuinely insufficient (no GPU) so Kaggle T4 is a real, legitimate infrastructure requirement — but the chance-accuracy anomaly is a validity blocker that must be diagnosed and fixed before any backbone is chosen, and "bigger model reduces hallucinations" is not the right lens for a closed-set log-likelihood scoring design that never generates free text.

This spec redesigns Stage 2's execution properly for Kaggle T4, starting from the last reviewed-clean state (`e1fc413`), replacing the unreviewed session's output entirely (not building on top of it).

## Goals

1. A reviewable, reproducible Kaggle execution pipeline: a thin script-kernel plus a Python orchestration tool that pushes/polls/pulls via the `kaggle` CLI — no manual copy-pasting of notebook output, no hand-edited `.ipynb` cells.
2. Diagnose the chance-accuracy bias mechanism, and fix it at the source (a calibration step in `ClosedSetPolicy`), not by picking whichever model happens to look less biased.
3. Choose the Stage 2 backbone from real evidence — a calibrated near-ceiling/chance/throughput sweep across candidate model sizes, run on the actual target hardware (Kaggle T4).
4. Recompute the grid-time budget against a real constraint (a Kaggle GPU session's actual time limit), using the full, un-shrunk 177,500-step grid from `PLAN.md`'s Stage 4/5 table — not an arbitrary number chosen to make an assertion pass.
5. Restore repo hygiene (drop `.pyc` from git, drop stray root-level test scripts, replace the ad hoc `kaggle_runner/` mess with one clean, reviewed structure) and formally amend `PLAN.md` to document the Kaggle pivot as the architectural change it is.

## Non-goals

- Stage 3+ (memory methods), Stage 4/5 (the real experiment grid) — this spec is Stage 2 only.
- LoRA / weight fine-tuning — still out of scope per `PLAN.md`'s "no LoRA on the critical path."
- Multi-GPU orchestration — a single T4 device is the target; Kaggle's `GPU_T4x2` offers two, but using both would need real data-parallel work this project doesn't need yet (YAGNI). `kernel-metadata.json` requests a single-GPU accelerator.
- Rewriting `env/generator.py` or the parts of `agent/prompt_templates.py` already reviewed clean in Task 1 — untouched.
- A general-purpose Kaggle-runner framework for arbitrary future experiments — build exactly what Stage 2 needs; Stage 4/5 can extend it later if the shape still fits.

## Architecture

### 1. Bias diagnosis + calibration (`agent/policy.py`)

Two new pieces on `ClosedSetPolicy`:

- `measure_label_prior(candidates: list[str]) -> np.ndarray` — scores the candidates against a fixed neutral prompt constant (no ticket text, no rule, no memory — just the instruction-and-anchor scaffold from `build_prompt("", candidates)`), isolating whatever score differences exist when there is zero information to justify preferring one label over another. This is the mechanism check: if `chance accuracy != ~1/|A|` while individual `predict()` calls are still deterministic and argmax-based, an uneven prior over the label tokens is the leading hypothesis to confirm or rule out before touching model choice at all.
- A free function `calibrate_scores(raw_scores: np.ndarray, prior: np.ndarray) -> np.ndarray` — subtracts `prior` from `raw_scores` along the last axis (broadcasts over both the single-prompt `(n_candidates,)` shape from `score_candidates` and the batched `(n_prompts, n_candidates)` shape from `score_candidates_batch`), so calibration is a pure post-processing step usable everywhere scores are produced, not a special-cased code path.

`experiments/diagnose_label_bias.py` (new) drives this: load a policy, call `measure_label_prior`, print the raw per-label bias, then re-run the existing `run_chance_check` logic twice — once uncalibrated (today's behavior) and once with `calibrate_scores` applied before argmax — and report both accuracies. This directly tests whether calibration closes the gap to `~1/|A|`, and is the evidence that decides whether Approach B (calibration) is sufficient on its own or whether backbone choice still matters after calibration.

### 2. Backbone selection sweep

Once calibration is validated, the same script sweeps `MODEL_NAME` across a small candidate list (e.g. `Qwen2.5-0.5B-Instruct`, `1.5B`, `3B` — extendable) via an environment variable (the one part of the interrupted session's approach worth keeping, done through a reviewed, tested code path this time rather than an ad hoc shell override). For each candidate it reports: near-ceiling accuracy, raw chance accuracy, calibrated chance accuracy, and measured seconds/step throughput (reusing `measure_batch_seconds_per_step`, unchanged from Task 4). The final backbone is chosen from this table against the plan's original, non-negotiable thresholds (`NEAR_CEILING_THRESHOLD=0.85`, `CHANCE_TOLERANCE=0.10` — restored, not the weakened `0.15`) — evaluated on the **calibrated** score, since calibration is now part of the policy's real behavior, not an optional afterthought.

### 3. Grid-time budget, restored and re-grounded

`TOTAL_GRID_STEPS` reverts to `177,500`, computed the same way Task 4 already did (verbatim from `PLAN.md`'s Stage 4/5 table — 5×2×5×5×400 + 1×4×5×5×400 + 2×5×5×750). `GRID_TIME_BUDGET_SECONDS` is redefined against a real constraint: a single Kaggle GPU session's actual wall-clock limit, not the original plan's CPU/MPS-era 4-hour guess and not the rogue commit's unmotivated 20 hours. **Open question for the human partner to confirm** (Kaggle's exact free-tier session/quota numbers change over time and are best confirmed against the account in use, not assumed): if the projected grid time exceeds one session's limit, the design's fallback is to split the grid across multiple kernel runs (partitioned by, e.g., alpha or seed) rather than inflating the budget constant to make one run fit.

### 4. Kaggle orchestration tooling

Rebuild `kaggle_runner/` as a small, reviewed Python module instead of the ad hoc file pile:

- `kaggle_runner/kernel_template.py` — the actual script-kernel payload: clone the repo at a pinned commit/branch, `pip install -r requirements.txt` (or the equivalent minimal set — Kaggle images already ship torch/transformers, but pin what Stage 2 needs explicitly rather than relying on whatever happens to be preinstalled), run the target entrypoint (`python -m experiments.diagnose_label_bias` or `python -m experiments.calibrate_speed`) with output written to a known path (`/kaggle/working/output/`) so it's captured as kernel output.
- `kaggle_runner/kernel-metadata.json` — one clean metadata file: `"kernel_type": "script"`, single-GPU accelerator, `enable_internet: true`.
- `kaggle_runner/orchestrate.py` — thin wrapper around `subprocess` calls to the `kaggle` CLI: `push(kernel_dir) -> kernel_id`, `wait_for_completion(kernel_id, poll_interval_s, timeout_s) -> status`, `pull_output(kernel_id, dest_dir) -> Path`. Tests mock `subprocess.run` to verify command construction and status/output parsing without hitting the real Kaggle API; one execution-only task (mirroring Stage 1 Task 6 / the original Stage 2 Task 5) does one real push/poll/pull round-trip against the actual account to confirm it works end to end.
- One `kaggle_runner/runs/<run-id>/` directory per real execution, committed as evidence (matching the project's existing precedent of committing `env/manipulation_check_output/` from Stage 1) — replacing the current `output/output2/output3/output4/pull/pull2` clutter with one clear, timestamped or run-id-named directory per actual run.

### 5. Repo cleanup

New commits on top of current `master` (no force-push, no history rewrite — the pushed commits stay visible in history, this is a forward-fixing revert, not an erasure):

- Restore `agent/policy.py`, `experiments/calibrate_speed.py`, `.gitignore`, `README.md` to their `e1fc413` content as the starting point for this spec's new work.
- Remove `test_prompt.py`, `test_smol.py` from the repo root.
- `git rm --cached` any committed `__pycache__`/`.pyc` files; add `__pycache__/` and `*.pyc` to `.gitignore` (currently missing — this is how they got committed in the first place).
- Replace the old `kaggle_runner/` tree with the new structure from section 4.

### 6. `PLAN.md` amendment

Add a new, clearly labeled architectural-amendment entry to `docs/materials/PLAN.md` (following the existing precedent of "Архитектурная правка 1/2" already documented there for Stage 1) recording: the move to Kaggle T4 as Stage 2+'s compute environment (reason: no local GPU), the calibration mechanism added to `ClosedSetPolicy`, and the re-grounded grid-time budget. This is a task within the implementation plan, executed and reviewed like any other — not something done ad hoc outside the process this time.

## Testing approach

- `calibrate_scores`: pure function, unit-tested with fabricated `raw_scores`/`prior` arrays (no model loading) — both the single-vector and batched shapes.
- `measure_label_prior`: tested against the real cached model (matching the existing `test_policy.py` pattern) — asserts a finite array of the right length and determinism across repeated calls (no randomness in frozen-weight scoring).
- `kaggle_runner/orchestrate.py`: unit-tested with `subprocess.run` mocked — verifies the exact CLI invocations and correct parsing of `kaggle kernels status`/`kaggle kernels output` results; a separate execution-only task does one real round-trip against the live Kaggle account.
- `experiments/diagnose_label_bias.py` and the backbone sweep: same pattern as `calibrate_speed.py` — small-N unit tests locally against the pure logic, the real gate-deciding full run happens as an execution-only task via the Kaggle orchestration tool.

## Risks / open questions

- **Kaggle session/quota limits** need confirming against the actual account before `GRID_TIME_BUDGET_SECONDS` is finalized — flagged above, not assumed.
- Calibrating on a neutral/empty prompt assumes the bias is prompt-independent (a pure token-frequency prior). If `diagnose_label_bias.py` shows the bias varies with ticket content instead, calibration-on-empty-prompt won't fully close the gap — the diagnostic step exists specifically to catch this before committing to the fix.
- Kaggle's Docker image's preinstalled package versions may not exactly match this project's pinned `pyproject.toml` versions (e.g., a different `transformers` release) — `kernel_template.py` should pin explicitly rather than trust whatever's preinstalled, but exact version pins should be checked against what's actually available on the image at implementation time.
