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
CHANCE_TOLERANCE = 0.15  # accuracy must land within +/- this of 1/|A|

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
