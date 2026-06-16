"""Shared lab harness for the FEASIBILITY experiments.

Loads the committed artifacts (corpus vectors + BM25) ONCE, embeds the 29 public
queries once (cached to query_vecs.npy), and exposes a faithful reimplementation of
`retrieve._zfuse_batch` (lead mode) parametrized by dense_w / beta / cand_n plus an
L2 `dense_union_m` knob.

Sanity contract: `score_config()` at the defaults (0.8, 0.15, 300, dense_union_m=0)
must reproduce the production 0.4338, or none of the sweeps are trustworthy.

All numpy after the one-time embed — runs in seconds, reproduces production exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from index import load_index            # noqa: E402
from bm25 import load_bm25, bm25_score_query, tokenize  # noqa: E402
from utils import load_public_queries    # noqa: E402
from eval import ndcg_at_k               # noqa: E402

QVEC_CACHE = HERE / "query_vecs.npy"


class Lab:
    """Holds all loaded data and precomputed arrays; methods score configs cheaply."""

    def __init__(self) -> None:
        self.vectors, page_ids, chunk_ids = load_index()
        self.bm25 = load_bm25()
        self.rows = load_public_queries()
        self.rel = [{int(p) for p in r["relevant_page_ids"]} for r in self.rows]

        # word counts (reuse production scanner so the length prior is identical)
        from retrieve import _load_word_counts
        wc_map = _load_word_counts()

        self.page_ids_arr = np.asarray(page_ids, dtype=np.int64)
        self.chunk_ids_arr = np.asarray(chunk_ids, dtype=np.int32)
        self.unique_pages, self.page_inverse = np.unique(self.page_ids_arr, return_inverse=True)
        self.n_pages = len(self.unique_pages)
        self.pid_to_uidx = {int(p): i for i, p in enumerate(self.unique_pages)}

        is_lead = self.chunk_ids_arr == 0
        self.lead_rows = np.where(is_lead)[0]
        self.lead_page = self.page_inverse[self.lead_rows]

        wc = np.array([max(wc_map.get(int(p), 1), 1) for p in self.unique_pages], dtype=np.float32)
        self.log_wc = np.log(wc)

        # query vectors (cache to avoid reloading the model each run)
        if QVEC_CACHE.exists():
            self.qv = np.load(QVEC_CACHE)
        else:
            from embed import embed_queries
            self.qv = embed_queries([r["query"] for r in self.rows])
            np.save(QVEC_CACHE, self.qv)

        # per-page lead cosine for every query: (n_queries, n_pages)
        scores = self.qv @ self.vectors.T            # (nq, n_vectors)
        cos = np.full((len(self.rows), self.n_pages), -np.inf, dtype=np.float32)
        # max over lead rows per page
        for qi in range(len(self.rows)):
            np.maximum.at(cos[qi], self.lead_page, scores[qi, self.lead_rows])
        self.cos_lead = cos                          # (nq, n_pages), -inf if no lead row

        # per-query BM25 scores mapped to unique-page index
        self.qbm: list[dict[int, float]] = []
        for r in self.rows:
            raw = bm25_score_query(tokenize(r["query"]), self.bm25)
            b_u: dict[int, float] = {}
            for pid, s in raw.items():
                ui = self.pid_to_uidx.get(int(pid))
                if ui is not None:
                    b_u[ui] = s
            self.qbm.append(b_u)

    # -- core scorer (mirrors retrieve._zfuse_batch, lead mode) --
    def rank_lists(self, dense_w=0.8, beta=0.15, cand_n=300, dense_union_m=0, top_k=10):
        # batch-union candidate pool: BM25 top-cand_n per query (+ optional dense top-M)
        union: set = set()
        for qi, b_u in enumerate(self.qbm):
            union.update(sorted(b_u, key=b_u.get, reverse=True)[:cand_n])
            if dense_union_m > 0:
                row = self.cos_lead[qi]
                topm = np.argpartition(-row, min(dense_union_m, self.n_pages - 1))[:dense_union_m]
                union.update(int(x) for x in topm)
        if not union:
            union = set(range(self.n_pages))
        cand = np.array(sorted(union), dtype=np.int64)

        out = []
        for qi in range(len(self.rows)):
            dense_raw = self.cos_lead[qi][cand] - beta * self.log_wc[cand]
            valid = np.isfinite(dense_raw)
            cand_v = cand[valid]
            dz = _z(dense_raw[valid])
            b_u = self.qbm[qi]
            bvals = np.array([b_u.get(int(c), np.nan) for c in cand_v], dtype=np.float32)
            matched = ~np.isnan(bvals)
            bz = np.zeros(len(cand_v), dtype=np.float32)
            if matched.any():
                bz[matched] = _z(bvals[matched])
            fused = dense_w * dz + (1.0 - dense_w) * bz
            order = np.argsort(-fused)[:top_k]
            out.append([int(self.unique_pages[cand_v[j]]) for j in order])
        return out

    def score_config(self, idxs=None, **kw) -> float:
        lists = self.rank_lists(**kw)
        if idxs is None:
            idxs = range(len(self.rows))
        s = [ndcg_at_k(lists[i], self.rel[i]) for i in idxs]
        return float(sum(s) / len(s)) if s else 0.0


def _z(x: np.ndarray) -> np.ndarray:
    sd = float(x.std())
    if sd < 1e-9:
        return np.zeros_like(x)
    return (x - float(x.mean())) / sd
