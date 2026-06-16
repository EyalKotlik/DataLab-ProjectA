# Section B — Retrieval pipeline

Retrieve relevant `page_id`s for a batch of natural-language queries over a corpus of
~27 K Wikipedia-style JSON pages, scored by **mean NDCG@10**.

**Current result: mean NDCG@10 = 0.4338** on the 29 public queries
(`AGGREGATE_MODE=zfuse`, the default). Up from 0.2527 under the previous default —
same artifacts, retrieval logic only. See [DIAGNOSIS.md](DIAGNOSIS.md) for the full
empirical record that selected this approach.

## Quick start

```bash
conda activate DataLab-ProjectA-SectionB      # or the project venv
pip install -r requirements.txt

python scripts/build_index.py     # offline, slow GPU job — run manually; commit artifacts/
python scripts/eval_public.py     # self-test on public queries (loads artifacts/, no rebuild)
```

Allowed packages only: `numpy`, `sentence-transformers`, `faiss-cpu`. Embedding model
is fixed: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalized).
Runtime budget: `run(queries)` must finish in **≤ 60 s** (GPU). `eval_public.py`
currently runs in ~28 s.

## How retrieval works (the `zfuse` default)

**Offline build** (`scripts/build_index.py`, untimed): chunk each page → embed with
MiniLM → save `artifacts/index_vectors.npy` + `index_meta.json`; also build a BM25
index → `artifacts/bm25.json.gz`. Per page the index stores: chunk −1 (title-only,
for `ZFUSE_TITLE_W` L6 lever), chunk 0 (lead = `entry_text`, title+content truncated
at 256 tokens), and body chunks 1..5 (150-word windows of content). By default `zfuse`
uses only the lead chunk — the others are inert unless their env flags are set.

**Runtime** (`retrieve.search_batch`, mode `zfuse`), per query over the batch:

1. **BM25 candidate generation** (un-gated, all query tokens): take each query's top
   `ZFUSE_CAND_N=300` pages; union across the batch → candidate pool.
2. **Dense score** per page = its lead-chunk (chunk 0) cosine, minus a **length prior**:
   `dense_raw = cos − ZFUSE_BETA·log(content_word_count)`, `ZFUSE_BETA=0.15`.
   Relevant pages are short (~50–280 words); distractors are long (median ~1218), so
   demoting long pages is the single biggest lever.
3. **z-score normalize** dense and BM25 over the candidate pool, then **fuse**:
   `score = ZFUSE_DENSE_W·dense_z + (1−ZFUSE_DENSE_W)·bm25_z`, `ZFUSE_DENSE_W=0.8`.
4. Return top-10 page_ids by fused score.

Why all three pieces are needed: BM25 candidate restriction removes the mass of global
distractors that pure dense ranks high; the length prior demotes long pages; BM25 in
the fusion re-anchors exact matches (rare entity names, numbers like `1,456,779`) that
a strong length prior would otherwise bury. The prior *alone* peaks at β≈0.05 then
collapses — it only tolerates β=0.15 because BM25 holds up the exact matches.

## File roles

| File | Role |
|------|------|
| `main.py` | `run(queries)` entry point called by the autograder |
| `retrieve.py` | `search_batch()` — **`zfuse` default** + legacy modes behind env vars |
| `bm25.py` | BM25 build (`build_bm25`) and query scoring (`bm25_score_query`) |
| `chunk.py` | `Chunk` dataclass; chunking (chunk 0 = `entry_text`, body windows unused by zfuse) |
| `embed.py` | Lazy-loads MiniLM; `embed_texts()` / `embed_queries()` |
| `index.py` | `build_index()` (offline) / `load_index()` (runtime, returns vectors+page_ids+chunk_ids) |
| `utils.py` | Shared paths, constants, `entry_text()` |
| `eval.py`, `scripts/build_index.py`, `scripts/eval_public.py` | **READ-ONLY** (graded harness) |
| `diagnose_*.py` | Standalone analysis scripts — see "Reproducing the analysis" |

## Results (29 public queries)

| Approach | NDCG@10 |
|---|---|
| Dense baseline (single vector, max-pool) | 0.224* |
| Pure BM25 | 0.319 |
| Dense over BM25 candidate pool (no prior, no fusion) | 0.343 |
| Previous default (length_prior β=0.05 + gated-RRF BM25) | 0.253 |
| z-score fusion, no length prior (dense_w=0.95) | 0.391 |
| **`zfuse` — dense_w=0.8, β=0.15 (current default)** | **0.434** |

\* baseline figure is from the earlier 50-query set; all other rows are the corrected 29-query set.

## What works / dead ends (do not re-try these)

**Works:** BM25 candidate generation + lead-chunk dense + length prior + z-score
weighted fusion (the `zfuse` recipe above).

**Refuted — measured worse, don't revisit:**
- **Overlapping body chunking** (sliding window): 0.10–0.12. Inflates the index to
  ~970 K vectors and floods top-10 with "lottery-ticket" false positives from long
  pages. Answers live in the lead, which chunk 0 already captures.
- **Sentence-granularity matching** (`sent_max`): 0.268, *worse* than full-doc dense
  (0.343). Queries are holistic paraphrases of short pages; no single sentence beats
  the whole lead vector. Do **not** re-embed at sentence level.
- **`lead_anchored`, `mean_top2`** aggregation: degrade at every setting (adding any
  non-lead-chunk signal adds noise).
- **Gated BM25 + RRF fusion** (old default): the IDF gate (`BM25_MIN_IDF=7.0`) fires on
  almost nothing and RRF k=60 flattens exact-match advantage → barely above dense.
  Replaced by un-gated z-score weighted fusion.
- **Decade expansion** (`1820s` → 1820–1829 in BM25): no effect (0.319 == pure BM25).

## Tuning knobs (env vars)

**Core `zfuse` params** (tuned): `ZFUSE_DENSE_W` (0.8), `ZFUSE_BETA` (0.15),
`ZFUSE_CAND_N` (300).  Switch modes with `AGGREGATE_MODE` (e.g. `chunk_0_only`).

**Toggleable levers** — all default OFF; output byte-identical to 0.4338 baseline.
All were A/B-tested on 2026-06-16; **none beat baseline**, so all stay OFF (full
table in [DIAGNOSIS.md](DIAGNOSIS.md)):

| Env var | Default | What it does | Measured | vs 0.4338 |
|---|---|---|---|---|
| `ZFUSE_CHUNK_AGG` | `lead` | `max` = score page by max cosine over all content chunks (lead+body) | 0.4115 (`max`) | ❌ −0.022 |
| `ZFUSE_TITLE_W` | `0.0` | L6 — blend title-only embedding (float weight, e.g. 0.2) | 0.4044 (`0.2`) | ❌ −0.029 |
| `BM25_TITLE_BOOST` | `1.0` | L7 — multiply title-term TF in BM25 (e.g. 2 or 3) | 0.4338 (`3`) | ➖ ±0.000 |
| `BM25_K1` | stored | L7 — override BM25 k1 at query time | — | untested |
| `BM25_B` | stored | L7 — override BM25 b at query time | — | untested |
| `BM25_TEMPORAL` | `0` | L9 — decade→year prefix: "1820s" matches pages with "1826" | 0.4322 (`1`) | ➖ −0.002 |

**Legacy RRF path**: `USE_BM25`, `BM25_MIN_IDF`, `BM25_WEIGHT`, `RRF_K`, `COUNT_BETA`.

## Reproducing the analysis

```bash
python diagnose_retrieval.py   # numpy/stdlib only: pure-BM25 score + recall@depth
python diagnose_hybrid.py      # needs the env: dense/BM25 fusion + length-prior sweeps
python diagnose_rerank.py      # needs the env: sentence-granularity test (refuted)
```

Neither hybrid script rebuilds the index — they embed only the BM25 candidate pool
(~few thousand docs, ~1 min). Full sweep numbers are logged in [DIAGNOSIS.md](DIAGNOSIS.md).

## Remaining headroom (vs peers' ~0.45)

1. **Widen recall**: BM25 recall@500 = 1.00 vs @100 = 0.90. Raise `ZFUSE_CAND_N` to
   500–1000 (or union with global dense top-K) → est. +0.02–0.03.
2. **Cross-validate β before locking**: β=0.15 is a strong short-doc prior fit on 29
   queries. Confirm it generalizes (and won't hurt if the hidden set has any long
   answer pages) with k-fold CV.
3. **Query-side rewriting** (last resort): the residual gap is the multi-hop paraphrase
   queries (synonyms, decade↔year) that neither modality bridges. Only after 1–2.

## Submit

Public GitHub repo with this code, the **required** `artifacts/` (committed; missing
artifacts score 0 functional), and this README. See the assignment PDF for video and
grading details.
