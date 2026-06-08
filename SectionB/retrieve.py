"""Query-time retrieval (timed portion includes query embedding).

Aggregation modes (select via AGGREGATE_MODE env var, default: lead_anchored):

  chunk_0_only   — rank by lead-chunk score only (diagnostic / baseline parity)
  lead_anchored  — lead score + LEAD_LAMBDA * max(other chunks) per page [DEFAULT]
  count_corrected— max chunk score - COUNT_BETA * log(n_chunks) per page
  mean_top2      — mean of top-2 chunk scores per page
  max            — plain max-of-chunks (original behaviour; known to regress)

Environment variables
---------------------
AGGREGATE_MODE   default lead_anchored
LEAD_LAMBDA      float, default 0.2
COUNT_BETA       float, default 0.1
"""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from embed import embed_queries
from index import load_index
from utils import K_EVAL

logger = logging.getLogger(__name__)

_AGGREGATE_MODE = os.environ.get("AGGREGATE_MODE", "lead_anchored")
_LEAD_LAMBDA = float(os.environ.get("LEAD_LAMBDA", "0.2"))
_COUNT_BETA = float(os.environ.get("COUNT_BETA", "0.1"))


def _rank_one(
    row: np.ndarray,
    page_ids: List[int],
    unique_pages: np.ndarray,
    page_inverse: np.ndarray,
    is_lead: np.ndarray,
    chunk_count: np.ndarray,
    top_k: int,
    mode: str,
    lead_lambda: float,
    count_beta: float,
) -> List[int]:
    """Return up to top_k ranked page_id ints for one query score row."""
    n_pages = len(unique_pages)

    if mode == "chunk_0_only":
        # Mask all non-lead vectors; rank by lead score only.
        masked = np.where(is_lead, row, -np.inf)
        order = np.argsort(-masked)
        seen: set = set()
        ids: List[int] = []
        for idx in order:
            if masked[int(idx)] == -np.inf:
                break
            pid = page_ids[int(idx)]
            if pid in seen:
                continue
            seen.add(pid)
            ids.append(pid)
            if len(ids) >= top_k:
                break
        return ids

    if mode == "lead_anchored":
        # page_score = score(chunk_0) + lead_lambda * max(score(other chunks))
        lead_score = np.full(n_pages, -np.inf, dtype=np.float32)
        np.maximum.at(lead_score, page_inverse[is_lead], row[is_lead])

        max_other = np.zeros(n_pages, dtype=np.float32)
        non_lead = ~is_lead
        if non_lead.any():
            np.maximum.at(max_other, page_inverse[non_lead], row[non_lead])

        # Pages that have no lead chunk (shouldn't occur with current build):
        no_lead = lead_score == -np.inf
        if no_lead.any():
            fallback = np.full(n_pages, -np.inf, dtype=np.float32)
            np.maximum.at(fallback, page_inverse, row)
            lead_score = np.where(no_lead, fallback, lead_score)

        page_score = lead_score + lead_lambda * np.maximum(max_other, 0.0)

    elif mode == "count_corrected":
        # page_score = max_chunk_score - count_beta * log(n_chunks)
        page_score = np.full(n_pages, -np.inf, dtype=np.float32)
        np.maximum.at(page_score, page_inverse, row)
        page_score -= count_beta * np.log(np.maximum(chunk_count, 1).astype(np.float32))

    elif mode == "mean_top2":
        # page_score = mean of top-2 chunk scores per page
        # Python dict approach — acceptable for diagnostic use
        page_chunks: dict = defaultdict(list)
        for i, (pid, s) in enumerate(zip(page_ids, row.tolist())):
            page_chunks[pid].append(s)
        page_score_dict = {}
        for pid, scores in page_chunks.items():
            top2 = sorted(scores, reverse=True)[:2]
            page_score_dict[pid] = sum(top2) / len(top2)
        ordered = sorted(page_score_dict, key=page_score_dict.__getitem__, reverse=True)
        return [int(p) for p in ordered[:top_k]]

    elif mode == "max":
        # Original behaviour (known regressor — kept for comparison only)
        page_score = np.full(n_pages, -np.inf, dtype=np.float32)
        np.maximum.at(page_score, page_inverse, row)

    else:
        raise ValueError(f"Unknown AGGREGATE_MODE: {mode!r}")

    # Common path for numpy-based modes
    order = np.argsort(-page_score)
    ids = []
    for idx in order:
        if page_score[int(idx)] == -np.inf:
            break
        ids.append(int(unique_pages[int(idx)]))
        if len(ids) >= top_k:
            break
    return ids


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
    aggregate: Optional[str] = None,
    lead_lambda: float = _LEAD_LAMBDA,
    count_beta: float = _COUNT_BETA,
) -> List[List[int]]:
    """
    Return ranked page_id lists (best first) for each query.

    Parameters
    ----------
    aggregate : str or None
        Aggregation mode.  None → use AGGREGATE_MODE env var (default
        'lead_anchored').  See module docstring for available modes.
    lead_lambda : float
        Weight for non-lead chunk bonus in 'lead_anchored' mode.
    count_beta : float
        Penalty coefficient in 'count_corrected' mode.
    """
    mode = aggregate if aggregate is not None else _AGGREGATE_MODE
    logger.info(
        "search_batch: %d queries, top_k=%d, aggregate=%s",
        len(queries), top_k, mode,
    )

    corpus_vectors, page_ids, chunk_ids = load_index(artifacts_dir)
    logger.info("search_batch: index loaded — %d vectors", len(page_ids))

    # Precompute page/chunk index structures (shared across all queries)
    page_ids_arr = np.array(page_ids, dtype=np.int64)
    chunk_ids_arr = np.array(chunk_ids, dtype=np.int32)
    unique_pages, page_inverse = np.unique(page_ids_arr, return_inverse=True)
    is_lead = chunk_ids_arr == 0
    chunk_count = np.bincount(page_inverse, minlength=len(unique_pages)).astype(np.int32)

    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        logger.warning("search_batch: no query vectors — returning empty lists")
        return [[] for _ in queries]

    t0 = time.perf_counter()
    scores = query_vectors @ corpus_vectors.T  # (n_queries, n_vectors)

    ranked: List[List[int]] = []
    for row in scores:
        ids = _rank_one(
            row, page_ids, unique_pages, page_inverse, is_lead, chunk_count,
            top_k, mode, lead_lambda, count_beta,
        )
        ranked.append(ids)

    logger.info(
        "search_batch: retrieval complete — %d ranked lists  [elapsed %.2fs]",
        len(ranked),
        time.perf_counter() - t0,
    )
    return ranked
