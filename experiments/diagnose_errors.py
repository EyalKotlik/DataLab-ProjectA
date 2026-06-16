"""Step 1 — per-query error analysis (the gate for L2).

For each public query, reports NDCG@10 under the production zfuse config and the rank
of every relevant page in:
  (g) global dense  — lead-chunk cosine over ALL pages
  (b) BM25          — full corpus
  (p) candidate pool membership (BM25 top-cand_n union); 'OUT' = unreachable
  (f) final fused list

The key question: are any relevant pages OUT of the pool (rank in pool = OUT)?
If yes, L2 (global-dense union) has something to recover. If no, L2 is pointless.

Run:  conda run -n DataLab-ProjectA python experiments/diagnose_errors.py
"""
from __future__ import annotations

import numpy as np

from _lab import Lab


def rank_of(order_pages: list[int], pid: int) -> str:
    try:
        return str(order_pages.index(pid) + 1)
    except ValueError:
        return "—"


def main() -> None:
    lab = Lab()
    cand_n = 300

    baseline = lab.score_config(dense_w=0.8, beta=0.15, cand_n=cand_n)
    print(f"baseline mean NDCG@10 = {baseline:.4f}  (must ≈ 0.4338)\n")

    final_lists = lab.rank_lists(dense_w=0.8, beta=0.15, cand_n=cand_n)

    # candidate pool (same batch-union the scorer uses)
    pool: set = set()
    for b_u in lab.qbm:
        pool.update(sorted(b_u, key=b_u.get, reverse=True)[:cand_n])

    n_out = 0
    n_relevant = 0
    out_queries = []
    print(f"{'q':>2} {'ndcg':>5}  relevant-page diagnostics (g=global-dense b=bm25 p=pool f=final)")
    print("-" * 92)
    for qi, row in enumerate(lab.rows):
        rel = lab.rel[qi]
        ndcg = __import__("eval").ndcg_at_k(final_lists[qi], rel)
        # global dense order (page ids by lead cosine, desc)
        gd_order = [int(lab.unique_pages[j]) for j in np.argsort(-lab.cos_lead[qi])]
        # bm25 order
        b_u = lab.qbm[qi]
        bm_order = [int(lab.unique_pages[u]) for u in sorted(b_u, key=b_u.get, reverse=True)]
        parts = []
        q_has_out = False
        for pid in sorted(rel):
            n_relevant += 1
            ui = lab.pid_to_uidx.get(pid)
            in_pool = ui in pool if ui is not None else False
            if not in_pool:
                n_out += 1
                q_has_out = True
            parts.append(
                f"pid={pid} g={rank_of(gd_order, pid):>5} b={rank_of(bm_order, pid):>5} "
                f"p={'IN' if in_pool else 'OUT':>3} f={rank_of(final_lists[qi], pid):>3}"
            )
        if q_has_out:
            out_queries.append(qi)
        flag = "  <-- has OUT-of-pool relevant page" if q_has_out else ""
        print(f"{qi:>2} {ndcg:5.3f}  " + " | ".join(parts) + flag)

    print("-" * 92)
    print(f"relevant pages total = {n_relevant}; OUT of pool = {n_out} "
          f"across {len(out_queries)} queries: {out_queries}")
    print(f"\nGATE: L2 (dense union) can only help these {len(out_queries)} queries. "
          f"{'PROCEED to L2.' if out_queries else 'L2 is pointless — recall is not the limiter.'}")


if __name__ == "__main__":
    main()
