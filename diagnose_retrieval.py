"""Standalone retrieval diagnostics — why is Section B stuck at <0.3 NDCG@10?

Runs with numpy + stdlib only (NO sentence-transformers / GPU needed), so it can
be executed anywhere to reproduce the structural findings in DIAGNOSIS.md:

  * pure-BM25 NDCG@10 over the full corpus
  * BM25 recall@{1,3,10,50,100,500}  (is the relevant doc even retrievable?)
  * rank of the best relevant doc per query (where does the true answer land?)

Usage:
    python diagnose_retrieval.py

The point: BM25 recall@100 is high (~0.84) but recall@10 is ~0.52 — the relevant
page is almost always *retrievable* but badly *ranked*. The bottleneck is ranking
precision, not recall, which is the regime where dense+lexical fusion / reranking
help most. See DIAGNOSIS.md.
"""
from __future__ import annotations

import glob
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent
ENTRIES = sorted(glob.glob(str(STUDENT_ROOT / "data" / "Wikipedia Entries" / "*.json")))
QUERIES = STUDENT_ROOT / "data" / "public_queries.json"

K1, B = 1.5, 0.75


def tok(text: str):
    return re.findall(r"[a-z]+|[0-9][0-9,]*[0-9]|[0-9]", text.lower())


def build():
    page_ids, dl = [], []
    inv = defaultdict(list)
    for i, p in enumerate(ENTRIES):
        d = json.load(open(p))
        toks = tok(d.get("title", "") + " " + d.get("content", ""))
        tf = defaultdict(int)
        for w in toks:
            tf[w] += 1
        page_ids.append(int(d["page_id"]))
        dl.append(len(toks))
        for w, c in tf.items():
            inv[w].append((i, c))
    n = len(page_ids)
    avgdl = sum(dl) / n
    idf = {w: math.log((n - len(pl) + 0.5) / (len(pl) + 0.5) + 1) for w, pl in inv.items()}
    return page_ids, dl, avgdl, inv, idf


def bm25(qtoks, dl, avgdl, inv, idf):
    sc = defaultdict(float)
    for w in set(qtoks):
        if w not in inv:
            continue
        iv = idf[w]
        for di, tf in inv[w]:
            sc[di] += iv * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl[di] / avgdl))
    return sc


def ndcg(ranked, rel, k=10):
    seen, g = set(), []
    for pid in ranked:
        if pid in seen:
            continue
        seen.add(pid)
        g.append(1.0 if pid in rel else 0.0)
        if len(g) >= k:
            break
    dcg = g[0] if g else 0.0
    for i, x in enumerate(g[1:], start=2):
        dcg += x / math.log2(i)
    nrel = min(len(rel), k)
    idcg = sum((1 / math.log2(i) if i > 1 else 1) for i in range(1, nrel + 1))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    t0 = time.time()
    page_ids, dl, avgdl, inv, idf = build()
    print(f"indexed {len(page_ids)} docs in {time.time()-t0:.1f}s, vocab={len(idf)}")
    rows = json.load(open(QUERIES))

    scores, ranks = [], []
    recall_hits = {d: 0 for d in (1, 3, 10, 50, 100, 500)}
    for r in rows:
        rel = {int(x) for x in r["relevant_page_ids"]}
        sc = bm25(tok(r["query"]), dl, avgdl, inv, idf)
        order = [page_ids[di] for di in sorted(sc, key=sc.get, reverse=True)]
        scores.append(ndcg(order[:10], rel))
        rk = min((order.index(p) + 1 for p in rel if p in order), default=99999)
        ranks.append(rk)
        for d in recall_hits:
            if rel & set(order[:d]):
                recall_hits[d] += 1

    n = len(rows)
    print(f"\nPURE BM25 mean NDCG@10 = {sum(scores)/n:.4f}")
    for d in sorted(recall_hits):
        print(f"  recall@{d:<4d}: {recall_hits[d]}/{n} = {recall_hits[d]/n:.2f}")
    print("\nrank of best relevant doc per query (sorted):")
    print(" ", sorted(ranks))


if __name__ == "__main__":
    main()
