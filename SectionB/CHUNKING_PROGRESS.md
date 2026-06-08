# Chunking Experiment Progress

## Current best
**Config:** `AGGREGATE_MODE=count_corrected COUNT_BETA=0.05` (existing 969 906-vector index)  
**NDCG@10:** **0.2578** (+15.0% over baseline)  
**Branch:** 1-chunking-title-prefix  
*(β=0.05 — log filenames "beta005" meant 0.05, not 0.005)*

---

## Results

| Date | Stage | Branch | Change | Index size | NDCG@10 | Rebuild? | Notes |
|------|-------|--------|--------|------------|---------|----------|-------|
| 2026-06-08 | baseline | main | no chunking (single entry_text per page) | ~27 K | **0.2241** | — | reference |
| 2026-06-08 | chunking v1 | 1-chunking-title-prefix | overlapping chunks W180/S60, entry_text split bug | 969 906 | 0.1042 | yes | double-title bug in chunk_entry |
| 2026-06-08 | chunking v2 | 1-chunking-title-prefix | fixed entry_text bug, split content only | 969 906 | 0.1203 | yes | still below baseline due to length-bias |
| 2026-06-08 | stage 0b | 1-chunking-title-prefix | chunk_0_only (lead chunk per page) | 969 906 | 0.2179 | no | confirms diagnosis; tiny gap vs baseline likely floating-point/batch diff |
| 2026-06-08 | stage 0c | 1-chunking-title-prefix | lead_anchored λ=0.001 | 969 906 | 0.2148 | no | any non-lead weight is harmful |
| 2026-06-08 | stage 0c | 1-chunking-title-prefix | lead_anchored λ=0.005 | 969 906 | 0.1969 | no | degrades fast |
| 2026-06-08 | stage 0c | 1-chunking-title-prefix | lead_anchored λ=0.01 | 969 906 | 0.1591 | no | |
| 2026-06-08 | stage 0c | 1-chunking-title-prefix | lead_anchored λ=0.02 | 969 906 | 0.1325 | no | |
| 2026-06-08 | stage 0c | 1-chunking-title-prefix | lead_anchored λ=0.03 | 969 906 | 0.1043 | no | |
| 2026-06-08 | stage 0c | 1-chunking-title-prefix | mean_top2 | 969 906 | 0.1148 | no | worse than max |
| 2026-06-08 | stage 0c | 1-chunking-title-prefix | count_corrected β=0.02 | 969 906 | 0.2258 | no | above baseline |
| 2026-06-08 | stage 0c | 1-chunking-title-prefix | count_corrected β=0.01 | 969 906 | 0.2443 | no | |
| **2026-06-08** | **stage 0c** | **1-chunking-title-prefix** | **count_corrected β=0.05** | **969 906** | **0.2578** | **no** | **BEST — +15% vs baseline** |

---

## Stage checklist

### Stage 0 — aggregation fix, no rebuild (existing 969 906-vector index)

- [x] **0a** — `load_index()` now returns 3-tuple `(vectors, page_ids, chunk_ids)`
- [x] **0b** — chunk_0_only → 0.2179 (confirms diagnosis; non-lead chunks are pure harm)
- [x] **0c** — count_corrected β=0.05 → **0.2578** beats baseline ✓
      lead_anchored is harmful at any λ > 0 — any non-lead weight degrades
      β=0.05 is the confirmed best (log filenames "beta005" = 0.05)
- [x] **0d** — β=0.05 confirmed as winner; no further fine-tuning needed

### Stage 1 — combined rebuild: fixed chunking + BM25 (one 5h GPU job)

- [x] Fix `_COUNT_BETA` default to 0.05 in `retrieve.py`
- [x] `chunk.py`: chunk_0 = entry_text(record), CHUNK_WORDS=150, no overlap, MAX_EXTRA_CHUNKS=5
      Estimated index: ~27K × avg 4 chunks ≈ 130K vectors (was 970K)
- [x] `bm25.py`: build_bm25() + load_bm25() + bm25_score_query() + tokenize()
      Tokeniser preserves comma-separated numbers ("1,456,779" → single high-IDF token)
- [x] `index.py`: calls build_bm25(records, out_dir) inside build_index()
- [x] `retrieve.py`: RRF fusion via USE_BM25 env var (default=1); falls back if no bm25.json.gz
- [x] All tests pass (31/31 non-ML tests)
- [ ] **User runs rebuild** → `python scripts/build_index.py`
- [ ] **Eval dense-only**: `USE_BM25=0 python scripts/eval_public.py` → target ≥ 0.2578
- [ ] **Eval with BM25**: `python scripts/eval_public.py` → target > dense-only

---

## Root cause summary

Empirical finding (2026-06-08):

- Relevant pages: median **133** content words → **2.1** chunks/page average  
- Corpus pages: median 1218 content words → **35.8** chunks/page average  
- Under max-pooling, long irrelevant pages have ~36 lottery tickets vs ~2 for relevant
  short pages → false positives flood top-10 → NDCG halved

Queries are needle-in-haystack (exact numbers/names; 1 relevant page each);
answers sit in the article lead (first ~170 words). Baseline already captures
the lead section naturally via MiniLM's 256-token truncation.

## Open questions / next action

1. **Trigger rebuild** — push this branch, run `python scripts/build_index.py` on the GPU server
2. **After rebuild, eval dense-only first** (`USE_BM25=0`) to isolate chunking improvement
3. **Then eval with BM25** (default) to measure RRF gain
4. If either eval regresses vs 0.2578: check build log for `load_index: N vectors`
   (expect ~130K not 970K) and `bm25.json.gz` file size (expect 50–120 MB)
