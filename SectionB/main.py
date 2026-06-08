"""
Section B entry point.

The autograder calls run(queries) once with all evaluation queries (batch of 50).
Query embedding + retrieval must complete within the time limit (GPU available).
"""
from __future__ import annotations

import logging
import os
import time
from typing import List

from index import build_index
from retrieve import search_batch
from utils import LOG_DIR, setup_logging

logger = logging.getLogger(__name__)


def run(queries: List[str]) -> List[List[int]]:
    """
    Rank corpus pages for each query.

    Parameters
    ----------
    queries : list[str]
        Batch of query strings (e.g. 50 hidden queries at grading time).

    Returns
    -------
    list[list[int]]
        One ranked list of page_id per query (most relevant first).
        Only the first 10 IDs per list are scored.
    """
    setup_logging()
    t0 = time.perf_counter()
    logger.info("run() called with %d queries", len(queries))
    result = search_batch(queries)
    logger.info("run() complete — elapsed %.2fs", time.perf_counter() - t0)
    return result


def build_offline_index() -> None:
    """Run once locally to create artifacts/ (not timed at grading)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    name = os.environ.get("BUILD_NAME", "")
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"build_{name}_{ts}.log" if name else f"build_{ts}.log"
    setup_logging(log_file=LOG_DIR / filename)

    t0 = time.perf_counter()
    label = f" [{name}]" if name else ""
    logger.info("build_offline_index() started%s — log: %s", label, LOG_DIR / filename)
    build_index()
    logger.info("build_offline_index() complete — total elapsed %.2fs", time.perf_counter() - t0)


if __name__ == "__main__":
    build_offline_index()
    print("Index built under artifacts/. Run: python scripts/eval_public.py")
