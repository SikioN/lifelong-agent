"""Stage 0 smoke test: confirms HF Hub access and the embedding model works."""
import numpy as np
from sentence_transformers import SentenceTransformer


def test_embedding_model_loads_and_encodes():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(["test sentence", "another sentence"])
    assert embeddings.shape == (2, 384)


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
