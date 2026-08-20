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
