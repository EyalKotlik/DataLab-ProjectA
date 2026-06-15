"""Hybrid (BM25 + dense MiniLM) fusion ceiling diagnostic.

Unlike diagnose_retrieval.py this DOES need sentence-transformers (the fixed
all-MiniLM-L6-v2 model). It does NOT build the full index: it only embeds the
union of each query's BM25 top-N candidates (~a few thousand short docs), which
runs in ~1-2 min on CPU. That is enough to measure what a *properly fused*
ranking can reach, which is the question DIAGNOSIS.md is trying to answer.

Run inside the project env:
    conda activate DataLab-ProjectA-SectionB
    python diagnose_hybrid.py

Reports, over the BM25 candidate pool:
  * dense-only NDCG@10
  * RRF fusion at several k
  * z-score weighted fusion at several dense/lexical weights

Append the printed numbers to DIAGNOSIS.md so the trail stays current.
"""
from __future__ import annotations

import glob
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

STUDENT_ROOT = Path(__file__).resolve().parent
ENTRIES = sorted(glob.glob(str(STUDENT_ROOT / "data" / "Wikipedia Entries" / "*.json")))
QUERIES = STUDENT_ROOT / "data" / "public_queries.json"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CAND = 300          # BM25 candidates per query to embed
K1, B = 1.5, 0.75


def tok(text):
    return re.findall(r"[a-z]+|[0-9][0-9,]*[0-9]|[0-9]", text.lower())


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
    recs, page_ids, dl = [], [], []
    inv = defaultdict(list)
    for i, p in enumerate(ENTRIES):
        d = json.load(open(p))
        recs.append((d.get("title", ""), d.get("content", "")))
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

    def bm25(qtoks):
        sc = defaultdict(float)
        for w in set(qtoks):
            if w not in inv:
                continue
            iv = idf[w]
            for di, tf in inv[w]:
                sc[di] += iv * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl[di] / avgdl))
        return sc

    rows = json.load(open(QUERIES))
    qbm, cand = [], set()
    for r in rows:
        sc = bm25(tok(r["query"]))
        qbm.append(sc)
        for di in sorted(sc, key=sc.get, reverse=True)[:CAND]:
            cand.add(di)
    cand = sorted(cand)
    ci = {di: j for j, di in enumerate(cand)}
    print(f"candidate docs to embed: {len(cand)}")

    m = SentenceTransformer(MODEL)
    t0 = time.time()
    dvec = m.encode([recs[di][0] + "\n\n" + recs[di][1] for di in cand],
                    batch_size=128, convert_to_numpy=True, normalize_embeddings=True)
    qvec = m.encode([r["query"] for r in rows],
                    batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    print(f"embedded in {time.time()-t0:.1f}s")

    def evalrank(fn):
        return sum(ndcg(fn(qi), {int(x) for x in rows[qi]["relevant_page_ids"]})
                   for qi in range(len(rows))) / len(rows)

    def dense_rank(qi):
        order = np.argsort(-(dvec @ qvec[qi]))
        return [page_ids[cand[j]] for j in order[:10]]

    print("dense (over bm25 cand) NDCG@10 =", round(evalrank(dense_rank), 4))

    def rrf(qi, kk):
        dorder = [cand[j] for j in np.argsort(-(dvec @ qvec[qi]))]
        border = [di for di in sorted(qbm[qi], key=qbm[qi].get, reverse=True) if di in ci]
        rr = defaultdict(float)
        for rank, di in enumerate(dorder):
            rr[di] += 1.0 / (kk + rank)
        for rank, di in enumerate(border):
            rr[di] += 1.0 / (kk + rank)
        return [page_ids[di] for di in sorted(rr, key=rr.get, reverse=True)[:10]]

    for kk in (5, 10, 20, 60):
        print(f"RRF k={kk:<3d} NDCG@10 =", round(evalrank(lambda qi, kk=kk: rrf(qi, kk)), 4))

    def z(d):
        if not d:
            return {}
        v = np.array(list(d.values()))
        mu, sd = v.mean(), v.std() + 1e-9
        return {k: (val - mu) / sd for k, val in d.items()}

    def wfuse(qi, alpha):
        sims = dvec @ qvec[qi]
        dz = z({cand[j]: float(sims[j]) for j in range(len(cand))})
        bz = z({di: qbm[qi][di] for di in qbm[qi] if di in ci})
        fz = defaultdict(float)
        for di, v in dz.items():
            fz[di] += alpha * v
        for di, v in bz.items():
            fz[di] += (1 - alpha) * v
        return [page_ids[di] for di in sorted(fz, key=fz.get, reverse=True)[:10]]

    for a in (0.3, 0.5, 0.7):
        print(f"zscore fuse dense_w={a} NDCG@10 =", round(evalrank(lambda qi, a=a: wfuse(qi, a)), 4))


if __name__ == "__main__":
    main()
