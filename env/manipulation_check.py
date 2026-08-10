"""Stage 1 hard gate: proves that alpha actually decouples semantic
similarity from decision utility, using the real embedding model that
Stage 3's Semantic-RAG and Decision-Aware memory will both share.

Run directly: `uv run python -m env.manipulation_check`
(module form required — running the file path directly puts env/ itself on
sys.path instead of the repo root, breaking the `from env.generator import`
below)
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
N_TENANTS = 300
OUTPUT_DIR = Path(__file__).parent / "manipulation_check_output"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_similarity_action_correlation(
    alpha: float, n_tickets: int, seed: int, encoder: SentenceTransformer
) -> float:
    """Spearman rho between pairwise cosine similarity and whether the two
    tickets share the same correct_action, over N_PAIRS random pairs."""
    generator = TicketGenerator(alpha=alpha, seed=seed, n_tenants=N_TENANTS)
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
    generator = TicketGenerator(alpha=alpha, seed=seed, n_tenants=N_TENANTS)
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
    ax.axhline(0.55, color="green", linestyle="--", linewidth=0.8, label="gate: rho>0.55 @ alpha=0")
    ax.axhline(0.15, color="red", linestyle="--", linewidth=0.8, label="gate: rho<0.15 @ alpha>=0.5")
    ax.set_xlabel("alpha (similarity/utility mismatch)")
    ax.set_ylabel("Spearman rho(similarity, action-agreement)")
    ax.set_title("Stage 1 manipulation check")
    ax.legend()
    fig.savefig(OUTPUT_DIR / "rho_vs_alpha.png", dpi=150)
    plt.close(fig)

    # Threshold is 0.55, not the original 0.7 in PLAN.md: with N_TOPICS=8 and
    # uniform random pair sampling, "same action" at alpha=0 is equivalent to
    # "same topic" (P=1/8), which caps achievable Spearman rho at ~0.57 even
    # under perfect separation. See docs/materials/PLAN.md Stage 1 escalation note.
    rho_at_zero = df.loc[df["alpha"] == 0.0, "rho"].item()
    assert rho_at_zero > 0.55, f"GATE FAILED: rho(alpha=0)={rho_at_zero:.3f}, need > 0.55"

    for alpha in (0.5, 0.7):
        rho_high = df.loc[df["alpha"] == alpha, "rho"].item()
        assert rho_high < 0.15, f"GATE FAILED: rho(alpha={alpha})={rho_high:.3f}, need < 0.15"

    check_cross_topic_same_action_pairs(alpha=0.5, seed=123)

    print(f"\nGATE PASSED. Results written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
