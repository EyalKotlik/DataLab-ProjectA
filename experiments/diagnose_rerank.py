"""Sentence-granularity matching diagnostic — the next lever after fusion failed.

Context: full-doc dense (0.234), pure BM25 (0.233), and every fusion of the two
(best 0.263) all plateau together — they rank the SAME distractors high. The
untested structural change is the matching UNIT: answers live in one short lead
sentence, distractors are long multi-topic pages. Scoring against the best-matching
*sentence* (not the whole truncated doc) should sharpen precision where it's stuck.

Needs the project env (sentence-transformers). Does NOT rebuild the index — embeds
only the sentences of the BM25 candidate pool (~few-thousand docs). On GPU ~1-2 min.

    conda activate DataLab-ProjectA-SectionB   # or the -venv
    python diagnose_rerank.py

Reports, over the BM25 candidate pool:
  * lead_sentence : sim(query, first sentence of doc)
  * sent_max      : max over all sentences of sim(query, sentence)
  * sent_max + BM25 z-score fusion at a few weights
  * BM25 with decade expansion ("1820s" also matches 1820..1829) — targets the
    observed decade-vs-year temporal gaps.

Paste the printed numbers into docs/DIAGNOSIS.md.
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

STUDENT_ROOT = Path(__file__).resolve().parent.parent  # repo root (script lives in experiments/)
ENTRIES = sorted(glob.glob(str(STUDENT_ROOT / "data" / "Wikipedia Entries" / "*.json")))
QUERIES = STUDENT_ROOT / "data" / "public_queries.json"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CAND = 300
MAX_SENT = 15           # cap sentences embedded per candidate
K1, B = 1.5, 0.75


def tok(text):
    return re.findall(r"[a-z]+|[0-9][0-9,]*[0-9]|[0-9]", text.lower())


def expand_decades(query):
    """Add sibling years for any 'NNN0s' decade mention: 1820s -> 1820..1829."""
    extra = []
    for dec in re.findall(r"\b(\d{3})0s\b", query.lower()):
        extra += [f"{dec}{d}" for d in range(10)]
    return tok(query) + extra


def split_sentences(title, content):
    parts = re.split(r"(?<=[.!?])\s+|\n+", content)
    sents = [title] if title else []
    for p in parts:
        p = p.strip()
        if len(tok(p)) >= 4:        # drop section headers ("History.")
            sents.append(p)
    return sents[:MAX_SENT] or [title or content[:200]]


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
    print(f"candidate docs: {len(cand)}")

    # Build sentence list with owner map
    sent_texts, sent_owner, lead_idx = [], [], {}
    for di in cand:
        sents = split_sentences(*recs[di])
        lead_idx[di] = len(sent_texts)         # first sentence row for this doc
        for s in sents:
            sent_texts.append(s)
            sent_owner.append(di)
    sent_owner = np.array(sent_owner)
    print(f"sentences to embed: {len(sent_texts)}")

    m = SentenceTransformer(MODEL)
    t0 = time.time()
    svec = m.encode(sent_texts, batch_size=256, convert_to_numpy=True, normalize_embeddings=True)
    qvec = m.encode([r["query"] for r in rows], batch_size=64,
                    convert_to_numpy=True, normalize_embeddings=True)
    print(f"embedded in {time.time()-t0:.1f}s")

    def evalrank(fn):
        return sum(ndcg(fn(qi), {int(x) for x in rows[qi]["relevant_page_ids"]})
                   for qi in range(len(rows))) / len(rows)

    # per-candidate score helpers
    def sent_max_scores(qi):
        sims = svec @ qvec[qi]                  # one sim per sentence
        best = defaultdict(lambda: -1e9)
        for row, di in enumerate(sent_owner):
            if sims[row] > best[di]:
                best[di] = sims[row]
        return best

    def lead_scores(qi):
        return {di: float(svec[lead_idx[di]] @ qvec[qi]) for di in cand}

    def top10(score):
        order = sorted(score, key=score.get, reverse=True)[:10]
        return [page_ids[di] for di in order]

    print("lead_sentence NDCG@10 =", round(evalrank(lambda qi: top10(lead_scores(qi))), 4))
    print("sent_max      NDCG@10 =", round(evalrank(lambda qi: top10(sent_max_scores(qi))), 4))

    def z(d):
        v = np.array(list(d.values()))
        mu, sd = v.mean(), v.std() + 1e-9
        return {k: (val - mu) / sd for k, val in d.items()}

    def fuse(qi, alpha):
        dz = z(sent_max_scores(qi))
        bz = z({di: qbm[qi][di] for di in qbm[qi] if di in ci})
        f = defaultdict(float)
        for di, v in dz.items():
            f[di] += alpha * v
        for di, v in bz.items():
            f[di] += (1 - alpha) * v
        return top10(f)

    for a in (0.4, 0.6, 0.8):
        print(f"sent_max+BM25 z-fuse dense_w={a} NDCG@10 =",
              round(evalrank(lambda qi, a=a: fuse(qi, a)), 4))

    # decade-expanded BM25 (pure lexical, full corpus — independent of candidates)
    s = []
    for r in rows:
        sc = bm25(expand_decades(r["query"]))
        ranked = [page_ids[di] for di in sorted(sc, key=sc.get, reverse=True)[:10]]
        s.append(ndcg(ranked, {int(x) for x in r["relevant_page_ids"]}))
    print("BM25 + decade-expansion NDCG@10 =", round(sum(s) / len(s), 4))


if __name__ == "__main__":
    main()
