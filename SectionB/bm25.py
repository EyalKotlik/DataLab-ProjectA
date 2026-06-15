"""BM25 lexical retrieval — build-time index and query-time scoring.

Allowed packages only: numpy (stdlib re/json/gzip also used).
"""
from __future__ import annotations

import gzip
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils import ARTIFACTS_DIR, entry_text

logger = logging.getLogger(__name__)

BM25_FILE = "bm25.json.gz"
_K1 = 1.5
_B = 0.75
_MIN_DF = 1   # keep all terms — singletons are highest-IDF and critical for exact matches

_cache: Optional[Dict[str, Any]] = None
_cache_dir: Optional[Path] = None


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Lowercase tokeniser that preserves comma-separated numbers.

    "1,456,779" → ["1,456,779"]   (rare, high-IDF token)
    "seven-game" → ["seven", "game"]
    """
    return re.findall(r"[a-z]+|[0-9][0-9,]*[0-9]|[0-9]", text.lower())


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_bm25(records: List[Dict[str, Any]], out_dir: Path) -> None:
    """Compute BM25 index for the corpus and write artifacts/bm25.json.gz."""
    t0 = time.perf_counter()
    n_docs = len(records)
    logger.info("build_bm25: tokenising %d documents …", n_docs)

    page_ids: List[int] = []
    doc_lengths: List[int] = []
    doc_tfs: List[Dict[str, int]] = []

    for r in records:
        tokens = tokenize(entry_text(r))
        page_ids.append(int(r["page_id"]))
        doc_lengths.append(len(tokens))
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        doc_tfs.append(tf)

    avgdl = sum(doc_lengths) / n_docs if n_docs else 1.0

    # Build inverted index: term → [[doc_idx, raw_tf], ...]
    inv_idx: Dict[str, List[List[int]]] = {}
    for doc_idx, tf in enumerate(doc_tfs):
        for term, count in tf.items():
            if term not in inv_idx:
                inv_idx[term] = []
            inv_idx[term].append([doc_idx, count])

    # IDF (Robertson), drop low-frequency terms
    idf: Dict[str, float] = {}
    low_freq = [t for t, p in inv_idx.items() if len(p) < _MIN_DF]
    for t in low_freq:
        del inv_idx[t]
    for term, postings in inv_idx.items():
        df = len(postings)
        idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

    payload = {
        "page_ids": page_ids,
        "doc_lengths": doc_lengths,
        "avgdl": avgdl,
        "k1": _K1,
        "b": _B,
        "idf": idf,
        "inv_idx": inv_idx,
    }

    out_path = out_dir / BM25_FILE
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_mb = out_path.stat().st_size / 1e6
    logger.info(
        "build_bm25: %d terms, %d docs, %.1f MB gzip  [elapsed %.2fs]",
        len(idf), n_docs, size_mb, time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_bm25(artifacts_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Lazy-load and cache the BM25 index from artifacts/bm25.json.gz."""
    global _cache, _cache_dir
    root = artifacts_dir or ARTIFACTS_DIR
    if _cache is not None and _cache_dir == root:
        logger.debug("load_bm25: cache hit")
        return _cache
    t0 = time.perf_counter()
    path = root / BM25_FILE
    logger.info("load_bm25: loading %s …", path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    _cache = data
    _cache_dir = root
    logger.info(
        "load_bm25: %d terms, %d docs  [elapsed %.2fs]",
        len(data["idf"]), len(data["page_ids"]), time.perf_counter() - t0,
    )
    return data


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def bm25_score_query(
    query_tokens: List[str],
    data: Dict[str, Any],
) -> Dict[int, float]:
    """Return {page_id: BM25_score} for every page matching at least one token."""
    inv_idx = data["inv_idx"]
    idf = data["idf"]
    doc_lengths = data["doc_lengths"]
    page_ids = data["page_ids"]
    avgdl = data["avgdl"]
    k1 = data.get("k1", _K1)
    b = data.get("b", _B)

    scores: Dict[int, float] = {}
    for term in set(query_tokens):
        if term not in inv_idx:
            continue
        idf_val = idf.get(term, 0.0)
        if idf_val <= 0:
            continue
        for doc_idx, tf in inv_idx[term]:
            dl = doc_lengths[doc_idx]
            tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
            pid = page_ids[doc_idx]
            scores[pid] = scores.get(pid, 0.0) + idf_val * tf_norm
    return scores
