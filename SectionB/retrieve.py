"""Query-time retrieval (timed portion includes query embedding).

Dense aggregation modes (AGGREGATE_MODE env var, default: length_prior):

  length_prior   — lead-chunk only + β·log(real_word_count) penalty  [DEFAULT]
  count_corrected— max chunk score - COUNT_BETA * log(n_chunks)
  chunk_0_only   — lead-chunk score only (diagnostic / baseline parity)
  lead_anchored  — lead score + LEAD_LAMBDA * max(other chunks)
  mean_top2      — mean of top-2 chunk scores per page
  max            — plain max-of-chunks (known regressor, kept for reference)

Lexical fusion:
  USE_BM25=1 (default) fuses dense rankings with BM25 via Reciprocal Rank Fusion,
  but only for query tokens whose IDF ≥ BM25_MIN_IDF (genuine rare needles).
  Falls back to dense-only if artifacts/bm25.json.gz is absent or no needle tokens.

Environment variables
---------------------
AGGREGATE_MODE   default length_prior
COUNT_BETA       float, default 0.05  (penalty strength for length_prior and count_corrected)
LEAD_LAMBDA      float, default 0.2
USE_BM25         0/1, default 1
BM25_MIN_IDF     float, default 7.0   (IDF gate: only tokens with IDF≥this fire BM25)
BM25_WEIGHT      float, default 1.0   (BM25 contribution weight in RRF; <1 = dense-anchored)
RRF_K            float, default 60
"""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from bm25 import bm25_score_query, load_bm25, tokenize
from embed import embed_queries
from index import load_index
from utils import K_EVAL

logger = logging.getLogger(__name__)

_AGGREGATE_MODE = os.environ.get("AGGREGATE_MODE", "length_prior")
_COUNT_BETA = float(os.environ.get("COUNT_BETA", "0.05"))
_LEAD_LAMBDA = float(os.environ.get("LEAD_LAMBDA", "0.2"))
_USE_BM25 = os.environ.get("USE_BM25", "1").lower() in ("1", "true", "yes")
_BM25_MIN_IDF = float(os.environ.get("BM25_MIN_IDF", "7.0"))
_BM25_WEIGHT = float(os.environ.get("BM25_WEIGHT", "1.0"))
_RRF_K = float(os.environ.get("RRF_K", "60"))
_RRF_OVER_FETCH = 200   # candidates fetched from dense before RRF merge

# ---------------------------------------------------------------------------
# Word-count cache (per-page real content length from corpus JSON files)
# ---------------------------------------------------------------------------

_word_count_cache: Optional[Dict[int, int]] = None
_word_count_cache_entries_dir: Optional[Path] = None


def _load_word_counts(entries_dir: Optional[Path] = None) -> Dict[int, int]:
    """Return {page_id: content_word_count} for every corpus page (lazy-cached)."""
    global _word_count_cache, _word_count_cache_entries_dir
    from utils import ENTRIES_DIR, iter_entries
    root = entries_dir or ENTRIES_DIR
    if _word_count_cache is not None and _word_count_cache_entries_dir == root:
        logger.debug("_load_word_counts: cache hit (%d pages)", len(_word_count_cache))
        return _word_count_cache
    t0 = time.perf_counter()
    logger.info("_load_word_counts: scanning corpus for word counts …")
    wc: Dict[int, int] = {}
    for rec in iter_entries(root):
        wc[int(rec["page_id"])] = len(rec.get("content", "").split())
    _word_count_cache = wc
    _word_count_cache_entries_dir = root
    logger.info(
        "_load_word_counts: %d pages  [elapsed %.2fs]",
        len(wc), time.perf_counter() - t0,
    )
    return wc


# ---------------------------------------------------------------------------
# Per-query ranking (dense only)
# ---------------------------------------------------------------------------

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
    word_counts: Optional[Dict[int, int]] = None,
) -> List[int]:
    """Return up to top_k ranked page_id ints for one query score row (dense only)."""
    n_pages = len(unique_pages)

    if mode == "chunk_0_only":
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

    if mode == "length_prior":
        # One score per page from its lead chunk (chunk_id=0) only.
        # Penalise by real content word count: score -= β * log(word_count).
        # Long pages get demoted even when their lead vector is semantically similar.
        lead_scores = np.where(is_lead, row, -np.inf)
        page_score = np.full(n_pages, -np.inf, dtype=np.float32)
        np.maximum.at(page_score, page_inverse, lead_scores)
        if word_counts is not None:
            wc_arr = np.array(
                [max(word_counts.get(int(p), 1), 1) for p in unique_pages],
                dtype=np.float32,
            )
            page_score -= count_beta * np.log(wc_arr)
        else:
            # Fallback: use chunk count as proxy (same as count_corrected)
            page_score -= count_beta * np.log(np.maximum(chunk_count, 1).astype(np.float32))

    elif mode == "lead_anchored":
        lead_score = np.full(n_pages, -np.inf, dtype=np.float32)
        np.maximum.at(lead_score, page_inverse[is_lead], row[is_lead])
        max_other = np.zeros(n_pages, dtype=np.float32)
        non_lead = ~is_lead
        if non_lead.any():
            np.maximum.at(max_other, page_inverse[non_lead], row[non_lead])
        no_lead = lead_score == -np.inf
        if no_lead.any():
            fallback = np.full(n_pages, -np.inf, dtype=np.float32)
            np.maximum.at(fallback, page_inverse, row)
            lead_score = np.where(no_lead, fallback, lead_score)
        page_score = lead_score + lead_lambda * np.maximum(max_other, 0.0)

    elif mode == "count_corrected":
        page_score = np.full(n_pages, -np.inf, dtype=np.float32)
        np.maximum.at(page_score, page_inverse, row)
        page_score -= count_beta * np.log(np.maximum(chunk_count, 1).astype(np.float32))

    elif mode == "mean_top2":
        page_chunks: dict = defaultdict(list)
        for pid, s in zip(page_ids, row.tolist()):
            page_chunks[pid].append(s)
        page_score_dict = {
            pid: sum(sorted(sc, reverse=True)[:2]) / min(len(sc), 2)
            for pid, sc in page_chunks.items()
        }
        ordered = sorted(page_score_dict, key=page_score_dict.__getitem__, reverse=True)
        return [int(p) for p in ordered[:top_k]]

    elif mode == "max":
        page_score = np.full(n_pages, -np.inf, dtype=np.float32)
        np.maximum.at(page_score, page_inverse, row)

    else:
        raise ValueError(f"Unknown AGGREGATE_MODE: {mode!r}")

    order = np.argsort(-page_score)
    ids = []
    for idx in order:
        if page_score[int(idx)] == -np.inf:
            break
        ids.append(int(unique_pages[int(idx)]))
        if len(ids) >= top_k:
            break
    return ids


# ---------------------------------------------------------------------------
# RRF merge (BM25 + dense)
# ---------------------------------------------------------------------------

def _rrf_merge(
    dense_top: List[int],
    bm25_scores: Dict[int, float],
    top_k: int,
    k: float = _RRF_K,
    weight: float = 1.0,
) -> List[int]:
    """Reciprocal Rank Fusion of dense and BM25 rankings.

    weight < 1.0 makes this dense-anchored: BM25 can promote pages up the list
    but cannot override a confident dense top result.
    """
    over_fetch = max(len(dense_top), _RRF_OVER_FETCH)
    bm25_top = sorted(bm25_scores, key=bm25_scores.__getitem__, reverse=True)[:over_fetch]

    rrf: Dict[int, float] = {}
    for rank, pid in enumerate(dense_top):
        rrf[pid] = rrf.get(pid, 0.0) + 1.0 / (k + rank)
    for rank, pid in enumerate(bm25_top):
        rrf[pid] = rrf.get(pid, 0.0) + weight / (k + rank)

    return sorted(rrf, key=rrf.__getitem__, reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# Batch retrieval entry point
# ---------------------------------------------------------------------------

def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
    aggregate: Optional[str] = None,
    lead_lambda: float = _LEAD_LAMBDA,
    count_beta: float = _COUNT_BETA,
    use_bm25: Optional[bool] = None,
    bm25_min_idf: float = _BM25_MIN_IDF,
    bm25_weight: float = _BM25_WEIGHT,
) -> List[List[int]]:
    """Return ranked page_id lists (best first) for each query.

    Parameters
    ----------
    aggregate    : aggregation mode (None → AGGREGATE_MODE env var)
    use_bm25     : enable BM25 RRF fusion (None → USE_BM25 env var)
    bm25_min_idf : IDF gate — only query tokens with IDF ≥ this trigger BM25
    bm25_weight  : BM25 rank contribution weight in RRF (<1 → dense-anchored)
    """
    mode = aggregate if aggregate is not None else _AGGREGATE_MODE
    do_bm25 = use_bm25 if use_bm25 is not None else _USE_BM25
    logger.info(
        "search_batch: %d queries, top_k=%d, aggregate=%s, bm25=%s",
        len(queries), top_k, mode, do_bm25,
    )

    corpus_vectors, page_ids, chunk_ids = load_index(artifacts_dir)
    logger.info("search_batch: index loaded — %d vectors", len(page_ids))

    page_ids_arr = np.array(page_ids, dtype=np.int64)
    chunk_ids_arr = np.array(chunk_ids, dtype=np.int32)
    unique_pages, page_inverse = np.unique(page_ids_arr, return_inverse=True)
    is_lead = chunk_ids_arr == 0
    chunk_count = np.bincount(page_inverse, minlength=len(unique_pages)).astype(np.int32)

    # Load word counts (length_prior mode) — fast corpus scan, cached
    word_counts: Optional[Dict[int, int]] = None
    if mode == "length_prior":
        word_counts = _load_word_counts()

    # Load BM25 index once (lazy-cached)
    bm25_data: Optional[Any] = None
    if do_bm25:
        try:
            bm25_data = load_bm25(artifacts_dir)
        except FileNotFoundError:
            logger.warning("search_batch: bm25.json.gz not found — dense-only fallback")
            do_bm25 = False

    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        logger.warning("search_batch: no query vectors — returning empty lists")
        return [[] for _ in queries]

    t0 = time.perf_counter()
    scores = query_vectors @ corpus_vectors.T  # (n_queries, n_vectors)

    # Over-fetch from dense when fusing with BM25
    dense_top_k = _RRF_OVER_FETCH if do_bm25 else top_k

    ranked: List[List[int]] = []
    for row, query in zip(scores, queries):
        dense_top = _rank_one(
            row, page_ids, unique_pages, page_inverse, is_lead, chunk_count,
            dense_top_k, mode, lead_lambda, count_beta, word_counts,
        )
        if do_bm25 and bm25_data is not None:
            tokens = tokenize(query)
            # IDF gate: only fire BM25 when the query contains genuine rare tokens.
            # Years, common words have low IDF and do not benefit from lexical matching.
            needle_tokens = [
                t for t in tokens
                if bm25_data["idf"].get(t, 0.0) >= bm25_min_idf
            ]
            if needle_tokens:
                logger.debug(
                    "search_batch: BM25 firing on %d needle tokens: %s",
                    len(needle_tokens), needle_tokens[:5],
                )
                bm25_scores = bm25_score_query(needle_tokens, bm25_data)
                ids = _rrf_merge(dense_top, bm25_scores, top_k, weight=bm25_weight)
            else:
                ids = dense_top[:top_k]
        else:
            ids = dense_top[:top_k]
        ranked.append(ids)

    logger.info(
        "search_batch: retrieval complete — %d lists  [elapsed %.2fs]",
        len(ranked), time.perf_counter() - t0,
    )
    return ranked
