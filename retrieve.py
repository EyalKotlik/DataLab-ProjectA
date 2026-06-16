"""Query-time retrieval (timed portion includes query embedding).

DEFAULT pipeline — AGGREGATE_MODE=zfuse (mean NDCG@10 ≈ 0.43 on the 29 public queries;
see DIAGNOSIS.md for the full sweep that selected it):

  1. BM25 (un-gated, all query tokens) generates a candidate pool: each query's
     top ZFUSE_CAND_N pages, unioned across the batch.
  2. Dense score per page = its lead-chunk (chunk_id=0) cosine, with a length prior
     subtracted:  dense_raw = cos - ZFUSE_BETA * log(content_word_count).
     Short pages are favoured (relevant pages are short; distractors are long).
  3. Dense and BM25 scores are each z-score normalized over the candidate pool, then
     fused:  score = ZFUSE_DENSE_W * dense_z + (1 - ZFUSE_DENSE_W) * bm25_z.
  4. Top-10 page_ids by fused score.

  The length prior alone over-penalizes past β≈0.05; fused with BM25 (which re-anchors
  exact matches) it tolerates β=0.15 — the two signals are complementary.  Body chunks
  are unused by zfuse by default (only the lead chunk feeds the dense score).

Legacy dense aggregation modes (kept for A/B and the eval harness):
  length_prior, count_corrected, chunk_0_only, lead_anchored, mean_top2, max
  — these use the old per-query rank + gated-RRF path (see _rank_one / _rrf_merge).

Environment variables
---------------------
AGGREGATE_MODE   default zfuse
ZFUSE_DENSE_W    float, default 0.8   (dense weight in z-score fusion)
ZFUSE_BETA       float, default 0.15  (length-prior strength: -β·log(word_count))
ZFUSE_CAND_N     int,   default 300   (BM25 candidates per query before union)

Toggleable levers — ALL DEFAULT OFF; baseline output is byte-identical to 0.4338
----------------------------------------------------------------------------------
ZFUSE_CHUNK_AGG  "lead" | "max", default "lead"
                 lead: dense page score = max cosine over lead chunk (chunk_id=0) only.
                 max:  dense page score = max cosine over all content chunks (chunk_id>=0),
                       i.e. includes body chunks 1..5.  Re-tests the body-chunk hypothesis
                       on the corrected 29-query set (prior result used a corrupted set).
ZFUSE_TITLE_W    float, default 0.0   (L6 — title-chunk blend weight; 0.0 = off)
                 When > 0 blends the title-only embedding (chunk_id=-1) into the dense
                 signal:  dense_raw = cos_base + ZFUSE_TITLE_W * title_cos - β·log(wc)
                 Requires the current index (which stores chunk_id=-1 rows).
BM25_TITLE_BOOST float, default 1.0   (L7 — BM25 title-term TF multiplier; 1.0 = off)
                 Boosts title-term frequency in BM25 scoring; requires the current index
                 (which stores 3-element postings [doc_idx, tf, title_tf]).
BM25_K1          float (optional)     override stored k1 at query time (no rebuild)
BM25_B           float (optional)     override stored b at query time (no rebuild)
BM25_TEMPORAL    0/1, default 0       (L9 — decade→year prefix matching; 0 = off)
                 Emits "\x00"+first3 tokens for 4-digit year query tokens so that
                 "1820s"→token "1820" matches pages with year "1826" via prefix "182".
                 Requires the current index (which stores temporal prefix postings).

Legacy env vars (zfuse-irrelevant, used only by legacy dense aggregation modes)
COUNT_BETA       float, default 0.05  (legacy length_prior / count_corrected)
LEAD_LAMBDA      float, default 0.2   (legacy lead_anchored)
USE_BM25         0/1, default 1       (legacy RRF path)
BM25_MIN_IDF     float, default 7.0   (legacy IDF gate)
BM25_WEIGHT      float, default 1.0   (legacy RRF BM25 weight)
RRF_K            float, default 60    (legacy RRF)
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

# Default zfuse params
_AGGREGATE_MODE = os.environ.get("AGGREGATE_MODE", "zfuse")
_ZFUSE_DENSE_W = float(os.environ.get("ZFUSE_DENSE_W", "0.8"))
_ZFUSE_BETA = float(os.environ.get("ZFUSE_BETA", "0.15"))
_ZFUSE_CAND_N = int(os.environ.get("ZFUSE_CAND_N", "300"))

# Legacy mode params
_COUNT_BETA = float(os.environ.get("COUNT_BETA", "0.05"))
_LEAD_LAMBDA = float(os.environ.get("LEAD_LAMBDA", "0.2"))
_USE_BM25 = os.environ.get("USE_BM25", "1").lower() in ("1", "true", "yes")
_BM25_MIN_IDF = float(os.environ.get("BM25_MIN_IDF", "7.0"))
_BM25_WEIGHT = float(os.environ.get("BM25_WEIGHT", "1.0"))
_RRF_K = float(os.environ.get("RRF_K", "60"))
_RRF_OVER_FETCH = 200   # candidates fetched from dense before RRF merge

# Toggleable levers — all default to OFF (output identical to 0.4338 baseline)
_ZFUSE_CHUNK_AGG  = os.environ.get("ZFUSE_CHUNK_AGG", "lead")
_ZFUSE_TITLE_W    = float(os.environ.get("ZFUSE_TITLE_W", "0.0"))
_BM25_TITLE_BOOST = float(os.environ.get("BM25_TITLE_BOOST", "1.0"))
_BM25_K1_OVERRIDE = float(os.environ.get("BM25_K1")) if os.environ.get("BM25_K1") else None
_BM25_B_OVERRIDE  = float(os.environ.get("BM25_B"))  if os.environ.get("BM25_B")  else None
_BM25_TEMPORAL    = os.environ.get("BM25_TEMPORAL", "0").lower() in ("1", "true", "yes")


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
# z-score weighted fusion (DEFAULT pipeline)
# ---------------------------------------------------------------------------

def _zscore(x: np.ndarray) -> np.ndarray:
    sd = float(x.std())
    if sd < 1e-9:
        return np.zeros_like(x)
    return (x - float(x.mean())) / sd


def _zfuse_batch(
    queries: List[str],
    corpus_vectors: np.ndarray,
    page_ids: List[int],
    chunk_ids: List[int],
    word_counts: Optional[Dict[int, int]],
    bm25_data: Optional[Any],
    *,
    top_k: int,
    dense_w: float,
    beta: float,
    cand_n: int,
    chunk_agg: str = "lead",
    title_w: float = 0.0,
    title_boost: float = 1.0,
    k1_override: Optional[float] = None,
    b_override: Optional[float] = None,
    temporal: bool = False,
) -> List[List[int]]:
    """BM25-candidate + length-prior dense, fused by z-score.  See module docstring.

    Extra parameters (all default to OFF — no effect on baseline output)
    -------------------------------------------------------------------
    chunk_agg  : "lead" (default) uses only chunk_id=0; "max" takes max over
                 chunk_id>=0 (lead + body chunks, excludes title chunk at -1).
    title_w    : L6 — blend weight for title-only embedding (chunk_id=-1).
                 0.0 = off (default).
    title_boost: L7 — BM25 title-term TF multiplier; passed to bm25_score_query.
                 1.0 = off (default).
    k1_override, b_override: L7 — runtime k1/b overrides; None = use stored values.
    temporal   : L9 — decade→year prefix matching; False = off (default).
    """
    page_ids_arr = np.asarray(page_ids, dtype=np.int64)
    chunk_ids_arr = np.asarray(chunk_ids, dtype=np.int32)
    unique_pages, page_inverse = np.unique(page_ids_arr, return_inverse=True)
    n_pages = len(unique_pages)

    # --- Precompute row masks (once, before query loop) ---

    # Lead rows (chunk_id == 0) — always needed for "lead" mode and logging
    is_lead = chunk_ids_arr == 0
    lead_rows = np.where(is_lead)[0]
    lead_page = page_inverse[lead_rows]           # unique-page index per lead row

    # Content rows (chunk_id >= 0) — lead + body chunks, excludes title (-1)
    # Used when chunk_agg == "max"
    is_content = chunk_ids_arr >= 0
    content_rows = np.where(is_content)[0]
    content_page = page_inverse[content_rows]

    # L6: title rows (chunk_id == -1); inert when title_w == 0.0
    is_title = chunk_ids_arr == -1
    title_rows = np.where(is_title)[0]
    title_page = (
        page_inverse[title_rows]
        if title_rows.size > 0
        else np.empty(0, dtype=np.int64)
    )

    pid_to_uidx = {int(p): i for i, p in enumerate(unique_pages)}

    if word_counts is not None:
        wc = np.array([max(word_counts.get(int(p), 1), 1) for p in unique_pages], dtype=np.float32)
    else:
        wc = np.ones(n_pages, dtype=np.float32)
    log_wc = np.log(wc)

    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        logger.warning("_zfuse_batch: no query vectors — returning empty lists")
        return [[] for _ in queries]
    scores = query_vectors @ corpus_vectors.T    # (n_queries, n_vectors)

    # Pass 1: per-query BM25 scores (mapped to unique-page index) + batch-union candidate set
    per_query_bm25: List[Dict[int, float]] = []
    union: set = set()
    for q in queries:
        b_u: Dict[int, float] = {}
        if bm25_data is not None:
            raw = bm25_score_query(
                tokenize(q), bm25_data,
                title_boost=title_boost,
                k1_override=k1_override,
                b_override=b_override,
                temporal=temporal,
            )
            for pid, s in raw.items():
                ui = pid_to_uidx.get(int(pid))
                if ui is not None:
                    b_u[ui] = s
        per_query_bm25.append(b_u)
        union.update(sorted(b_u, key=b_u.get, reverse=True)[:cand_n])

    if not union:                                # no lexical signal at all → dense over all pages
        logger.warning("_zfuse_batch: empty BM25 candidate union — dense-only fallback")
        union = set(range(n_pages))
    cand = np.array(sorted(union), dtype=np.int64)

    ranked: List[List[int]] = []
    for qi, row in enumerate(scores):
        # --- Dense base: lead-only (default) or max-over-content-chunks ---
        if chunk_agg == "max":
            cos_base = np.full(n_pages, -np.inf, dtype=np.float32)
            np.maximum.at(cos_base, content_page, row[content_rows])
        else:  # "lead" — current default
            cos_base = np.full(n_pages, -np.inf, dtype=np.float32)
            np.maximum.at(cos_base, lead_page, row[lead_rows])

        # --- L6: blend title cosine into dense signal ---
        # title_cos initialised to 0 so pages without a title chunk contribute 0
        # (neutral); np.maximum.at updates only pages that have a title row.
        if title_w > 0.0 and title_rows.size > 0:
            title_cos = np.zeros(n_pages, dtype=np.float32)
            np.maximum.at(title_cos, title_page, row[title_rows])
            dense_raw = cos_base[cand] + title_w * title_cos[cand] - beta * log_wc[cand]
        else:
            dense_raw = cos_base[cand] - beta * log_wc[cand]

        valid = np.isfinite(dense_raw)
        cand_v = cand[valid]
        dz = _zscore(dense_raw[valid])

        b_u = per_query_bm25[qi]
        bvals = np.array([b_u.get(int(c), np.nan) for c in cand_v], dtype=np.float32)
        matched = ~np.isnan(bvals)
        bz = np.zeros(len(cand_v), dtype=np.float32)
        if matched.any():
            bz[matched] = _zscore(bvals[matched])

        fused = dense_w * dz + (1.0 - dense_w) * bz
        order = np.argsort(-fused)[:top_k]
        ranked.append([int(unique_pages[cand_v[j]]) for j in order])
    return ranked


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
    dense_w: float = _ZFUSE_DENSE_W,
    zfuse_beta: float = _ZFUSE_BETA,
    cand_n: int = _ZFUSE_CAND_N,
    # Toggleable levers — all default OFF (env-var-readable defaults)
    chunk_agg: str = _ZFUSE_CHUNK_AGG,
    title_w: float = _ZFUSE_TITLE_W,
    title_boost: float = _BM25_TITLE_BOOST,
    k1_override: Optional[float] = _BM25_K1_OVERRIDE,
    b_override: Optional[float] = _BM25_B_OVERRIDE,
    temporal: bool = _BM25_TEMPORAL,
) -> List[List[int]]:
    """Return ranked page_id lists (best first) for each query.

    Parameters
    ----------
    aggregate    : aggregation mode (None → AGGREGATE_MODE env var; default zfuse)
    dense_w      : zfuse dense weight; zfuse_beta: length-prior strength; cand_n: BM25 candidates
    use_bm25     : enable BM25 RRF fusion in legacy modes (None → USE_BM25 env var)
    bm25_min_idf : legacy IDF gate — only query tokens with IDF ≥ this trigger BM25
    bm25_weight  : legacy BM25 rank contribution weight in RRF (<1 → dense-anchored)

    Toggleable levers (zfuse mode only, all default OFF):
    chunk_agg    : "lead" or "max" — see ZFUSE_CHUNK_AGG env var
    title_w      : L6 title-chunk blend weight — see ZFUSE_TITLE_W env var
    title_boost  : L7 BM25 title-term TF multiplier — see BM25_TITLE_BOOST env var
    k1_override, b_override : L7 runtime BM25 k1/b override — see BM25_K1/BM25_B env vars
    temporal     : L9 decade→year prefix matching — see BM25_TEMPORAL env var
    """
    mode = aggregate if aggregate is not None else _AGGREGATE_MODE
    do_bm25 = use_bm25 if use_bm25 is not None else _USE_BM25
    logger.info(
        "search_batch: %d queries, top_k=%d, aggregate=%s, bm25=%s",
        len(queries), top_k, mode, do_bm25,
    )

    corpus_vectors, page_ids, chunk_ids = load_index(artifacts_dir)
    logger.info("search_batch: index loaded — %d vectors", len(page_ids))

    # DEFAULT path: z-score weighted fusion (BM25 candidates + length-prior dense).
    if mode == "zfuse":
        word_counts = _load_word_counts()
        try:
            bm25_data = load_bm25(artifacts_dir)
        except FileNotFoundError:
            logger.warning("search_batch: bm25.json.gz not found — zfuse runs dense-only")
            bm25_data = None
        t0 = time.perf_counter()
        ranked = _zfuse_batch(
            queries, corpus_vectors, page_ids, chunk_ids, word_counts, bm25_data,
            top_k=top_k, dense_w=dense_w, beta=zfuse_beta, cand_n=cand_n,
            chunk_agg=chunk_agg, title_w=title_w,
            title_boost=title_boost, k1_override=k1_override, b_override=b_override,
            temporal=temporal,
        )
        logger.info(
            "search_batch: zfuse complete — %d lists  [elapsed %.2fs]",
            len(ranked), time.perf_counter() - t0,
        )
        return ranked

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
