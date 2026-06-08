"""Query-time retrieval (timed portion includes query embedding)."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from embed import embed_queries
from index import load_index
from utils import K_EVAL

logger = logging.getLogger(__name__)


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """
    Return ranked page_id lists (best first) for each query.

    Default: brute-force dot product on L2-normalized vectors.
    Replace with FAISS / reranking as needed.
    """
    logger.info("search_batch: %d queries, top_k=%d", len(queries), top_k)

    corpus_vectors, page_ids = load_index(artifacts_dir)
    logger.info(
        "search_batch: index loaded — %d vectors", len(page_ids)
    )

    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        logger.warning("search_batch: no query vectors produced — returning empty lists")
        return [[] for _ in queries]

    t0 = time.perf_counter()
    scores = query_vectors @ corpus_vectors.T
    ranked: List[List[int]] = []
    for row in scores:
        order = np.argsort(-row)
        seen: set[int] = set()
        ids: List[int] = []
        for idx in order:
            pid = page_ids[int(idx)]
            if pid in seen:
                continue
            seen.add(pid)
            ids.append(pid)
            if len(ids) >= top_k:
                break
        ranked.append(ids)
    logger.info(
        "search_batch: retrieval complete — %d ranked lists  [elapsed %.2fs]",
        len(ranked),
        time.perf_counter() - t0,
    )
    return ranked
