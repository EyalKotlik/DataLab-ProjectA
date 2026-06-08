"""Integration tests for embed.py — loads MiniLM model (slow)."""
from __future__ import annotations

import numpy as np
import pytest

from embed import embed_queries, embed_texts


@pytest.mark.slow
class TestEmbedTexts:
    def test_shape(self):
        vectors = embed_texts(["hello world", "test sentence"])
        assert vectors.shape == (2, 384)

    def test_dtype(self):
        vectors = embed_texts(["hello"])
        assert vectors.dtype == np.float32

    def test_l2_normalized(self):
        vectors = embed_texts(["hello world", "another sentence", "third one"])
        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_empty_input(self):
        vectors = embed_texts([])
        assert vectors.shape == (0, 384)

    def test_single_text(self):
        vectors = embed_texts(["single"])
        assert vectors.shape == (1, 384)

    def test_deterministic(self):
        texts = ["Paris is the capital of France."]
        v1 = embed_texts(texts)
        v2 = embed_texts(texts)
        np.testing.assert_array_equal(v1, v2)


@pytest.mark.slow
class TestEmbedQueries:
    def test_same_output_as_embed_texts(self):
        queries = ["what is the capital of France?", "history of Rome"]
        vq = embed_queries(queries)
        vt = embed_texts(queries)
        np.testing.assert_array_equal(vq, vt)

    def test_shape(self):
        vq = embed_queries(["test query"])
        assert vq.shape == (1, 384)
