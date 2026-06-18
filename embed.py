"""Embedding utilities (sentence-transformers/all-MiniLM-L6-v2 only)."""
from __future__ import annotations

import logging
import time
from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from utils import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading model %s …", EMBEDDING_MODEL_NAME)
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Model loaded.")
    else:
        logger.debug("Model already cached — reusing.")
    return _model


def embed_texts(
    texts: Sequence[str],
    *,
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """Return L2-normalized embeddings, shape (n, dim)."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    logger.debug("embed_texts: %d texts, batch_size=%d", len(texts), batch_size)
    model = get_model()
    t0 = time.perf_counter()
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    result = np.asarray(vectors, dtype=np.float32)
    logger.info(
        "Embedding complete — shape=%s  [elapsed %.2fs]",
        result.shape,
        time.perf_counter() - t0,
    )
    return result


def embed_queries(
    queries: List[str],
    *,
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    return embed_texts(queries, batch_size=batch_size, show_progress=show_progress)
