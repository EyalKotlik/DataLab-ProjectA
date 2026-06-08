# Retrieval Pipeline — Improvements Log

This document records what was tried, what worked, and why. It supersedes the original
speculative roadmap; see `CHUNKING_PROGRESS.md` for the detailed results table.

---

## What worked

### Dense aggregation fix — count_corrected β=0.05 (no rebuild)

**Result:** 0.2241 → **0.2578** (+15%)

The original hypothesis in this file ("overlapping chunking is critical because MiniLM
truncates at 256 tokens") turned out to be **wrong for this dataset**. Running the experiment
showed that adding chunks *hurt* NDCG rather than helping it.

**Root cause discovered empirically:**
- Relevant pages: median **133** content words → ~2 chunks/page
- Corpus pages: median **1218** content words → ~36 chunks/page
- Under max-pooling, each chunk is a "lottery ticket." Long irrelevant pages get 36 tickets
  vs 2 for the short relevant pages — false positives flood the top-10.
- Queries are needle-in-haystack (each maps to ~1 page); answers sit in the article lead
  (~first 170 words), which the baseline's 256-token truncation already captured perfectly.

**Fix:** penalise pages with many chunks before ranking.
`page_score = max_chunk_score − 0.05 × log(n_chunks_per_page)`

Implemented in `retrieve.py` as `AGGREGATE_MODE=count_corrected` (default).

---

### BM25 lexical hybrid — Stage 1 rebuild

**Status:** implemented, awaiting rebuild result

Many queries contain exact numbers ("1,456,779 residents") and rare entity names that dense
embeddings blur. BM25 IDF strongly weights singleton terms (those appearing in exactly one
document), making it ideal for these queries.

**Key implementation decision:** `_MIN_DF = 1` in `bm25.py` — all terms are kept, including
hapax legomena. A term like `"1,456,779"` that appears in one document has IDF ≈ 9.8 (the
highest possible) and is the primary BM25 signal for the corresponding query. Setting
`_MIN_DF = 2` would silently drop these and eliminate BM25's main benefit.

Fused with dense via Reciprocal Rank Fusion (RRF, k=60) in `retrieve.py`.

---

### Chunking redesign — Stage 1 rebuild

**Status:** implemented, awaiting rebuild result

Given the lottery-ticket finding, the redesign prioritises keeping the good signal while
limiting chunk-count inflation:

- `chunk_id=0` = `entry_text(record)` for every page — identical to the baseline single vector;
  MiniLM truncates naturally at 256 tokens. This preserves the lead-section signal that
  gave 0.2241.
- Extra chunks: non-overlapping 150-word windows, capped at `MAX_EXTRA_CHUNKS=5` (6 total per
  page). Covers content past the model's truncation point without recreating the 36-ticket
  lottery-ticket problem.
- Estimated index: ~130K vectors (vs 970K with the original overlapping design).
- `count_corrected` penalty still applies; with fewer chunks the penalty is smaller, which is
  correct since the bias is also smaller.

---

## What did NOT work

### Overlapping chunk splitting (original plan — items 1+2 above)

**Result:** 0.2241 → 0.1042 (−53%), then 0.1203 after bug fix

The rationale was sound in theory but wrong for this dataset. The overlapping sliding window
(CHUNK_WORDS=180, STRIDE=60) inflated the index from 27K → 970K vectors and triggered the
lottery-ticket false-positive problem described above.

**Do not revert to this design.** See `CHUNKING_PROGRESS.md` for full evidence.

### Lead-anchored aggregation

`page_score = score(chunk_0) + λ × max(other_chunks)` — degrades at every tested λ > 0.
Any weight on non-lead chunks adds noise. The non-lead chunks are irrelevant to the queries
in this dataset.

### Mean-of-top-2 aggregation

0.1148 — averaging noise with signal is worse than max.

---

## Remaining options (not yet tried)

### FAISS index

Speed-only benefit. Brute-force numpy matmul over ~130K vectors takes < 1s per 50-query
batch. Only relevant if the index grows beyond ~500K vectors or if heavier reranking is added.
Not a priority given the 60s runtime budget.

### Score aggregation with BM25

Once Stage 1 rebuild results are in: if dense-only underperforms, try tuning the RRF k
parameter (currently 60) or the BM25 weight (currently equal to dense).
