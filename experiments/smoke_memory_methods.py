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
