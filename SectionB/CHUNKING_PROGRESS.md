# Chunking Experiment Progress

## Current best
**Config:** baseline / no chunking  
**NDCG@10:** 0.2241  
**Branch:** main

---

## Results

| Date | Stage | Branch | Change | Index size | NDCG@10 | Rebuild? | Notes |
|------|-------|--------|--------|------------|---------|----------|-------|
| 2026-06-08 | baseline | main | no chunking (single entry_text per page) | ~27 K | **0.2241** | — | reference |
| 2026-06-08 | chunking v1 | 1-chunking-title-prefix | overlapping chunks W180/S60, entry_text split bug | 969 906 | 0.1042 | yes | double-title bug in chunk_entry |
| 2026-06-08 | chunking v2 | 1-chunking-title-prefix | fixed entry_text bug, split content only | 969 906 | 0.1203 | yes | still below baseline due to length-bias |

---

## Stage checklist

### Stage 0 — aggregation fix, no rebuild (existing 969 906-vector index)

- [x] **0a** — `load_index()` now returns 3-tuple `(vectors, page_ids, chunk_ids)`
- [ ] **0b** — Validate: `AGGREGATE_MODE=chunk_0_only python scripts/eval_public.py`
      Expected: ≈ 0.2241.  Proves that extra chunks (not chunk_0) are pure harm.
- [ ] **0c** — Test aggregation modes with the existing chunked index:
  - [ ] `AGGREGATE_MODE=lead_anchored LEAD_LAMBDA=0.2` — primary candidate
  - [ ] `AGGREGATE_MODE=lead_anchored LEAD_LAMBDA=0.1`
  - [ ] `AGGREGATE_MODE=lead_anchored LEAD_LAMBDA=0.3`
  - [ ] `AGGREGATE_MODE=count_corrected COUNT_BETA=0.1`
  - [ ] `AGGREGATE_MODE=mean_top2`
  - Gate: does any mode beat 0.2241?

### Stage 1 — chunking redesign, one rebuild

- [ ] Cap chunks/page ≤ 6–8 (kills 36-vs-2 lottery-ticket asymmetry)
- [ ] Keep chunk_0 = `entry_text(record)` (identical to baseline unit)
- [ ] No overlap (STRIDE ≈ WINDOW); cuts ~970K → ~100–200K vectors
- [ ] Fix token budget: `CHUNK_WORDS ≈ 150` (title + 150w ≈ 230 tokens < 256)
- [ ] Pair with best Stage-0 aggregation

### Stage 2 — BM25 lexical hybrid, one rebuild

- [ ] `bm25.py` — numpy IDF + per-page TF, serialize to `artifacts/bm25.json`
- [ ] Reciprocal Rank Fusion in `retrieve.py`
- [ ] Gate: does fusion beat dense-only?

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

1. Run Stage 0b (`chunk_0_only`) → confirm ≈ 0.2241
2. Run Stage 0c modes to find best λ
3. Decide whether Stage 1 rebuild is worth doing (need to see per-query breakdown
   from Stage 0 to know if there are "deep-content" queries where chunking could win)
