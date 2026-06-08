"""Cross-validation + bootstrap confidence interval over the public 50 queries.

Reports three numbers:
  full_mean  — mean NDCG@10 over all 50 queries (optimistic; what you tune to)
  cv_mean    — 5-fold cross-validated mean (honest; what to report in progress log)
  ci_95      — bootstrap 95% confidence interval on the per-query scores

The full_mean − cv_mean gap is the overfitting estimate.  Do not accept a
config change whose gain is smaller than the bootstrap CI width.

Usage (from SectionB/):
  python scripts/cv_eval.py
  AGGREGATE_MODE=length_prior COUNT_BETA=0.04 USE_BM25=0 python scripts/cv_eval.py
  AGGREGATE_MODE=length_prior COUNT_BETA=0.05 USE_BM25=1 BM25_MIN_IDF=7.0 python scripts/cv_eval.py

  # β sweep:
  for b in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.10; do
      COUNT_BETA=$b AGGREGATE_MODE=length_prior USE_BM25=0 python scripts/cv_eval.py
  done
"""
from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import List, Set

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from eval import load_query_file, ndcg_at_k
from main import run
from utils import PUBLIC_QUERIES_PATH

# Reproducible bootstrap seed
_SEED = 42
_N_BOOTSTRAP = 2000
_N_FOLDS = 5


def _bootstrap_ci(scores: List[float], n_boot: int = _N_BOOTSTRAP, seed: int = _SEED
                  ) -> tuple[float, float]:
    """95% bootstrap confidence interval for the mean."""
    rng = random.Random(seed)
    n = len(scores)
    boot_means = sorted(
        sum(rng.choices(scores, k=n)) / n
        for _ in range(n_boot)
    )
    lo = boot_means[int(0.025 * n_boot)]
    hi = boot_means[int(0.975 * n_boot)]
    return lo, hi


def main() -> None:
    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    ground_truth: List[Set[int]] = [r["relevant_page_ids"] for r in rows]
    n = len(queries)

    # Single run() call → per-query NDCG vector (no per-fold model reloads)
    t0 = time.perf_counter()
    ranked = run(queries)
    elapsed = time.perf_counter() - t0

    if len(ranked) != n:
        raise ValueError(f"run() returned {len(ranked)} lists, expected {n}")

    per_query = [ndcg_at_k(ranked[i], ground_truth[i]) for i in range(n)]

    # Full-set mean
    full_mean = sum(per_query) / n

    # 5-fold CV mean
    fold_size = n // _N_FOLDS
    cv_scores: List[float] = []
    for fold in range(_N_FOLDS):
        start = fold * fold_size
        end = start + fold_size if fold < _N_FOLDS - 1 else n
        held_out = per_query[start:end]
        cv_scores.append(sum(held_out) / len(held_out))
    cv_mean = sum(cv_scores) / _N_FOLDS

    # Bootstrap CI on per-query scores
    ci_lo, ci_hi = _bootstrap_ci(per_query)
    ci_width = ci_hi - ci_lo

    # Scoring config (for display)
    aggregate = os.environ.get("AGGREGATE_MODE", "length_prior")
    beta = os.environ.get("COUNT_BETA", "0.05")
    bm25 = os.environ.get("USE_BM25", "1")
    bm25_idf = os.environ.get("BM25_MIN_IDF", "7.0")
    bm25_w = os.environ.get("BM25_WEIGHT", "1.0")

    print(f"config:     AGGREGATE_MODE={aggregate} COUNT_BETA={beta} "
          f"USE_BM25={bm25} BM25_MIN_IDF={bm25_idf} BM25_WEIGHT={bm25_w}")
    print(f"n_queries:  {n}")
    print(f"full_mean:  {full_mean:.4f}  (optimistic — do not tune to this)")
    print(f"cv_mean:    {cv_mean:.4f}  ({_N_FOLDS}-fold, honest estimate)")
    print(f"ci_95:      [{ci_lo:.4f}, {ci_hi:.4f}]  width={ci_width:.4f}")
    print(f"overfit:    {full_mean - cv_mean:+.4f}  (full_mean − cv_mean)")
    print(f"elapsed:    {elapsed:.2f}s")
    print()
    print("Rule: accept a config change only if its CV gain > bootstrap CI width.")
    print(f"      Current CI width: {ci_width:.4f}  (~{ci_width:.2f} NDCG points)")


if __name__ == "__main__":
    main()
