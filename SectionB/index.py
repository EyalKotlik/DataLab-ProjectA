"""Offline index build and load (not timed at grading)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from bm25 import build_bm25
from chunk import Chunk, chunk_corpus
from embed import embed_texts
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, iter_entries

logger = logging.getLogger(__name__)

INDEX_VECTORS_NAME = "index_vectors.npy"
INDEX_META_NAME = "index_meta.json"


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Embed the full corpus and persist artifacts.

    Returns (vectors, page_ids) where row i corresponds to page_ids[i].
    For multi-chunk pipelines, store chunk metadata in index_meta.json and
    aggregate to page_id in retrieve.py.
    """
    out_dir = artifacts_dir or ensure_artifacts_dir()
    t_total = time.perf_counter()
    logger.info("build_index: starting — entries_dir=%s", entries_dir or "default")

    t0 = time.perf_counter()
    records = list(iter_entries(entries_dir))
    logger.info(
        "build_index: loaded %d corpus entries  [elapsed %.2fs]",
        len(records),
        time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    chunks: List[Chunk] = chunk_corpus(records)
    avg = len(chunks) / len(records) if records else 0
    logger.info(
        "build_index: %d chunks produced (avg %.1f per entry)  [elapsed %.2fs]",
        len(chunks),
        avg,
        time.perf_counter() - t0,
    )

    texts = [c.text for c in chunks]
    logger.info("build_index: embedding %d texts (batch_size=64) …", len(texts))
    t0 = time.perf_counter()
    vectors = embed_texts(texts, show_progress=True)
    logger.info(
        "build_index: embedding done — shape=%s  [elapsed %.2fs]",
        vectors.shape,
        time.perf_counter() - t0,
    )

    page_ids = [c.page_id for c in chunks]

    vectors_path = out_dir / INDEX_VECTORS_NAME
    logger.info("build_index: saving vectors → %s", vectors_path)
    np.save(vectors_path, vectors)

    meta = {
        "page_ids": page_ids,
        "chunk_ids": [c.chunk_id for c in chunks],
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "num_vectors": len(page_ids),
    }
    meta_path = out_dir / INDEX_META_NAME
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    t0 = time.perf_counter()
    logger.info("build_index: building BM25 index …")
    build_bm25(records, out_dir)
    logger.info("build_index: BM25 done  [elapsed %.2fs]", time.perf_counter() - t0)

    logger.info(
        "build_index: complete — %d vectors saved  [total elapsed %.2fs]",
        len(page_ids),
        time.perf_counter() - t_total,
    )
    return vectors, page_ids


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """Load precomputed vectors, page_id map, and chunk_id map from artifacts/.

    Returns
    -------
    vectors : np.ndarray, shape (n, 384)
    page_ids : list[int], length n  — corpus page_id for each row
    chunk_ids : list[int], length n — 0 for lead/first chunk, >0 for later windows
    """
    root = artifacts_dir or ARTIFACTS_DIR
    logger.debug("load_index: loading from %s", root)
    t0 = time.perf_counter()
    vectors = np.load(root / INDEX_VECTORS_NAME)
    meta = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
    page_ids = [int(x) for x in meta["page_ids"]]
    chunk_ids = [int(x) for x in meta.get("chunk_ids", [0] * len(page_ids))]
    logger.info(
        "load_index: %d vectors, dim=%d  [elapsed %.2fs]",
        vectors.shape[0],
        vectors.shape[1],
        time.perf_counter() - t0,
    )
    return vectors, page_ids, chunk_ids
