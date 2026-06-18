"""BM25 lexical retrieval — build-time index and query-time scoring.

Allowed packages only: numpy (stdlib re/json/gzip also used).

The rebuild bakes in two offline signals that are inert at query time unless their
respective env flags are set (see bm25_score_query docstring):

  L7 — field-weighted BM25 title boost
       Postings are stored as [doc_idx, raw_tf, title_tf] 3-tuples so that the runtime
       BM25_TITLE_BOOST env flag can multiply title-term frequency without a rebuild.
       Default BM25_TITLE_BOOST=1.0 → effective_tf = raw_tf (identical to before).

  L9 — temporal decade→year prefix matching
       4-digit year tokens (1000–2099) in each document generate a namespaced prefix
       posting "\x00" + first3digits (e.g. year "1826" → posting key "\x00182").
       The runtime BM25_TEMPORAL=1 flag emits the same prefix tokens for query-side
       year tokens so "1820s" → token "1820" → lookup "\x00182" → matches pages with
       any year 1820–1829. Default BM25_TEMPORAL=0 → query never emits "\x00" tokens
       → stored prefix postings are completely inert.
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

# Matches 4-digit year tokens in range 1000–2099 (L9)
_YEAR_RE = re.compile(r'^(1[0-9]{3}|20[0-9]{2})$')

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
    """Compute BM25 index for the corpus and write artifacts/bm25.json.gz.

    Postings format: [doc_idx, raw_tf, title_tf]  (3-tuple).
    title_tf is the count of this term in the title only; used by the L7 title-boost
    lever at query time.  With BM25_TITLE_BOOST=1.0 (default) the third element is
    never read and scoring is byte-identical to the old 2-tuple format.

    Temporal prefix postings (L9) are stored under keys "\x00" + first3digits and are
    never touched unless BM25_TEMPORAL=1 is set at query time.
    """
    t0 = time.perf_counter()
    n_docs = len(records)
    logger.info("build_bm25: tokenising %d documents …", n_docs)

    page_ids: List[int] = []
    doc_lengths: List[int] = []
    doc_tfs: List[Dict[str, int]] = []
    title_tfs: List[Dict[str, int]] = []    # L7: per-doc title-token counts
    temporal_tfs: List[Dict[str, int]] = [] # L9: per-doc year-prefix counts

    for r in records:
        full_tokens = tokenize(entry_text(r))
        title_tokens = tokenize(r.get("title", "") or "")

        page_ids.append(int(r["page_id"]))
        doc_lengths.append(len(full_tokens))

        # Regular TF over full document text
        tf: Dict[str, int] = {}
        for t in full_tokens:
            tf[t] = tf.get(t, 0) + 1
        doc_tfs.append(tf)

        # L7: TF over title tokens only (stored as 3rd posting element)
        t_tf: Dict[str, int] = {}
        for t in title_tokens:
            t_tf[t] = t_tf.get(t, 0) + 1
        title_tfs.append(t_tf)

        # L9: for every 4-digit year token, accumulate under its 3-char prefix key
        yr_tf: Dict[str, int] = {}
        for t in full_tokens:
            if _YEAR_RE.match(t):
                prefix = "\x00" + t[:3]
                yr_tf[prefix] = yr_tf.get(prefix, 0) + 1
        temporal_tfs.append(yr_tf)

    avgdl = sum(doc_lengths) / n_docs if n_docs else 1.0

    # Build inverted index: term → [[doc_idx, raw_tf, title_tf], ...]
    inv_idx: Dict[str, List[List[int]]] = {}
    for doc_idx, (tf, t_tf) in enumerate(zip(doc_tfs, title_tfs)):
        for term, count in tf.items():
            if term not in inv_idx:
                inv_idx[term] = []
            inv_idx[term].append([doc_idx, count, t_tf.get(term, 0)])

    # IDF (Robertson); drop low-frequency terms (with _MIN_DF=1 this is a no-op)
    idf: Dict[str, float] = {}
    low_freq = [t for t, p in inv_idx.items() if len(p) < _MIN_DF]
    for t in low_freq:
        del inv_idx[t]
    for term, postings in inv_idx.items():
        df = len(postings)
        idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

    # L9: merge temporal prefix postings into inv_idx under namespaced keys
    # Collect which prefix keys exist (for IDF computation)
    temporal_keys: set = set()
    for doc_idx, yr_tf in enumerate(temporal_tfs):
        for prefix, count in yr_tf.items():
            temporal_keys.add(prefix)
            if prefix not in inv_idx:
                inv_idx[prefix] = []
            inv_idx[prefix].append([doc_idx, count, 0])  # title_tf=0 for temporal

    # IDF for temporal prefix terms (computed on their own df after merge)
    for prefix in temporal_keys:
        if prefix in inv_idx:
            df = len(inv_idx[prefix])
            idf[prefix] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

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
        "build_bm25: %d terms (%d temporal prefix), %d docs, %.1f MB gzip  [elapsed %.2fs]",
        len(idf), len(temporal_keys), n_docs, size_mb, time.perf_counter() - t0,
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
    *,
    title_boost: float = 1.0,
    k1_override: Optional[float] = None,
    b_override: Optional[float] = None,
    temporal: bool = False,
) -> Dict[int, float]:
    """Return {page_id: BM25_score} for every page matching at least one token.

    Parameters
    ----------
    title_boost : float, default 1.0
        L7 lever.  Effective TF = raw_tf + (title_boost - 1) * title_tf.
        With 1.0 this collapses to raw_tf — output identical to before.
    k1_override, b_override : optional floats
        Override the stored k1/b values at query time (no rebuild required).
    temporal : bool, default False
        L9 lever.  When True, each 4-digit year token t in query_tokens also
        generates a lookup on the namespaced key "\x00" + t[:3] so that a
        decade query token like "1820" (from "1820s") matches pages containing
        any year sharing the first 3 digits (e.g. "1826" → "\x00182").
        When False (default) no namespaced tokens are emitted and the stored
        temporal prefix postings are completely inert.
    """
    inv_idx = data["inv_idx"]
    idf = data["idf"]
    doc_lengths = data["doc_lengths"]
    page_ids = data["page_ids"]
    avgdl = data["avgdl"]
    k1 = k1_override if k1_override is not None else data.get("k1", _K1)
    b = b_override if b_override is not None else data.get("b", _B)

    # L9: augment effective query token set with decade-prefix lookups
    effective_tokens: List[str] = list(query_tokens)
    if temporal:
        for t in set(query_tokens):
            if _YEAR_RE.match(t):
                effective_tokens.append("\x00" + t[:3])

    scores: Dict[int, float] = {}
    for term in set(effective_tokens):
        if term not in inv_idx:
            continue
        idf_val = idf.get(term, 0.0)
        if idf_val <= 0:
            continue
        for posting in inv_idx[term]:
            doc_idx = posting[0]
            raw_tf = posting[1]
            # 3rd element is title_tf; gracefully handle legacy 2-element postings
            t_tf = posting[2] if len(posting) > 2 else 0

            # L7: boost title-term frequency; with title_boost=1.0 → raw_tf unchanged
            effective_tf = raw_tf + (title_boost - 1.0) * t_tf

            dl = doc_lengths[doc_idx]
            tf_norm = effective_tf * (k1 + 1) / (effective_tf + k1 * (1 - b + b * dl / avgdl))
            pid = page_ids[doc_idx]
            scores[pid] = scores.get(pid, 0.0) + idf_val * tf_norm
    return scores
