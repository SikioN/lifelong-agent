"""Stage 0 smoke test: confirms HF Hub access and the embedding model works."""
from sentence_transformers import SentenceTransformer


def test_embedding_model_loads_and_encodes():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(["test sentence", "another sentence"])
    assert embeddings.shape == (2, 384)
