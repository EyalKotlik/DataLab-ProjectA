# Chunking Experiment Progress

## Current best
**Config:** `AGGREGATE_MODE=count_corrected COUNT_BETA=0.005` (existing 969 906-vector index)  
**NDCG@10:** **0.2578** (+15.0% over baseline)  
**Branch:** 1-chunking-title-prefix

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
| **2026-06-08** | **stage 0c** | **1-chunking-title-prefix** | **count_corrected β=0.005** | **969 906** | **0.2578** | **no** | **BEST — +15% vs baseline** |

---

## Stage checklist

### Stage 0 — aggregation fix, no rebuild (existing 969 906-vector index)

- [x] **0a** — `load_index()` now returns 3-tuple `(vectors, page_ids, chunk_ids)`
- [x] **0b** — chunk_0_only → 0.2179 (confirms diagnosis; non-lead chunks are pure harm)
- [x] **0c** — count_corrected β=0.005 → **0.2578** beats baseline ✓
      lead_anchored is harmful at any λ > 0 — any non-lead weight degrades
      count_corrected β=0.005 is the current best
- [ ] **0d** — Fine-tune β around 0.005 (β=0.002, 0.003, 0.004, 0.006, 0.007)
      Curve shape: β=0 → 0.1203 (pure max), β=0.005 → 0.2578 (peak?), β→∞ → hurts
      Need to confirm whether optimum is β=0.005 or somewhere nearby

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

1. **Fine-tune β (no rebuild)** — run β ∈ {0.002, 0.003, 0.004, 0.006, 0.007} to confirm optimum
2. **Change default** — retrieve.py default changed to count_corrected/β=0.005 in latest commit
3. **Stage 1 decision** — the existing chunked index already beats baseline with count_corrected;
   a better chunking redesign (cap chunks/page, no overlap, `CHUNK_WORDS≈150`) could squeeze
   more out, but the rebuild is 5h — worth doing after β is locked
4. **Stage 2 (BM25)** — these queries have exact numbers/names that BM25 handles better;
   this is the highest remaining ceiling
