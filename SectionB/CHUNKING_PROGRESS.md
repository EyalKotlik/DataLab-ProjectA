# Chunking Experiment Progress

## Current best
**Config:** `AGGREGATE_MODE=count_corrected COUNT_BETA=0.05` (existing 969 906-vector index)  
**NDCG@10:** **0.2578** (+15.0% over baseline)  
**Branch:** 1-chunking-title-prefix  
*(β=0.05 — log filenames "beta005" meant 0.05, not 0.005)*

---

## Results

> **How to read this table:** `NDCG@10` is the full-set mean on 50 public queries (optimistic).
> From Stage 2 onward, also report `CV mean` (5-fold) and `CI 95%` from `scripts/cv_eval.py`.
> Only CV gains that exceed the CI width are meaningful.

| Date | Stage | Config | Index | NDCG@10 | CV mean | CI 95% | Rebuild? | Notes |
|------|-------|--------|-------|---------|---------|--------|----------|-------|
| 2026-06-08 | baseline | no chunking | ~27 K | 0.2241 | — | — | — | reference (main branch) |
| 2026-06-08 | chunk v1 | W180/S60, split bug | 969 906 | 0.1042 | — | — | yes | double-title bug |
| 2026-06-08 | chunk v2 | W180/S60, fixed | 969 906 | 0.1203 | — | — | yes | lottery-ticket false positives |
| 2026-06-08 | stage 0b | chunk_0_only | 969 906 | 0.2179 | — | — | no | confirms: lead chunk = baseline |
| 2026-06-08 | stage 0c | lead_anchored λ=0.001 | 969 906 | 0.2148 | — | — | no | any non-lead weight harmful |
| 2026-06-08 | stage 0c | lead_anchored λ=0.01 | 969 906 | 0.1591 | — | — | no | degrades fast |
| 2026-06-08 | stage 0c | mean_top2 | 969 906 | 0.1148 | — | — | no | averaging noise with signal |
| 2026-06-08 | stage 0c | count_corrected β=0.02 | 969 906 | 0.2258 | — | — | no | above baseline |
| 2026-06-08 | stage 0c | count_corrected β=0.04 | 969 906 | 0.2543 | — | — | no | |
| 2026-06-08 | **stage 0c** | **count_corrected β=0.05** | **969 906** | **0.2578** | — | — | no | **prev best — +15% vs baseline** |
| 2026-06-08 | stage 0c | count_corrected β=0.06 | 969 906 | 0.2548 | — | — | no | near-plateau |
| 2026-06-08 | stage 1 | count_corrected β=0.05 + BM25 RRF | 121 140 | 0.2406 | — | — | yes | **REGRESSION** — BM25 + rebuild hurt |

---

## Stage checklist

### Stage 0 — aggregation fix, no rebuild ✓

- [x] **0a** — `load_index()` returns 3-tuple `(vectors, page_ids, chunk_ids)`
- [x] **0b** — chunk_0_only → 0.2179 (confirms: lead chunk = baseline quality)
- [x] **0c** — count_corrected β=0.05 → **0.2578** (+15% vs baseline)
      β=0.05 confirmed winner (log filenames "beta005" = 0.05, not 0.005)

### Stage 1 — chunking redesign + BM25 rebuild ✗ (REGRESSED)

Result: 0.2406 with count_corrected β=0.05 + BM25 RRF on 121K-vector index.
*No dense-only eval was run on the new index — regression cannot be attributed.*

Post-mortem:
- Relevant pages: median 133 words (51% ≤ 1 chunk; 79% < 256 tokens).
  Chunking never helped; BM25 helped ≤1 of 50 queries (only `1,456,779` is a rare singleton).
  Years (1820s, 1987) have low IDF; equal-weight RRF diluted confident dense results.
- `count_corrected` β penalty is a length proxy in disguise — use real word counts instead.
- **Lesson:** Never co-mingle a rebuild with a scoring change. Isolate one variable per eval.

### Stage 2 — no rebuild; real word count prior + gated BM25 (in progress)

Goal: match or beat 0.2578 with no index rebuild, proper CV evaluation, and reproducible logs.

- [x] `retrieve.py`: add `length_prior` mode (default) — lead-chunk only +
      β·log(real_word_count) penalty; lazy-cached corpus scan
- [x] `retrieve.py`: IDF-gated BM25 — only fires when query contains tokens with IDF ≥ BM25_MIN_IDF
- [x] `scripts/run_eval.sh`: reproducible wrapper — log name/header encodes git SHA, all env vars,
      and index fingerprint; appends row to `logs/results.csv`
- [x] `scripts/run_build.sh`: reproducible build wrapper (run manually on GPU server)
- [x] `scripts/cv_eval.py`: 5-fold CV + bootstrap 95% CI; headline metric is CV mean
- [ ] **Eval baseline control**: `AGGREGATE_MODE=chunk_0_only USE_BM25=0 bash scripts/run_eval.sh`
      → expect ~0.2179–0.2241
- [ ] **Eval length_prior β sweep**: run `cv_eval.py` for β ∈ {0.03,0.04,0.05,0.06,0.07,0.08};
      pick plateau-center β by CV mean; target CV mean ≥ 0.2578
- [ ] **Eval gated BM25**: `USE_BM25=1 BM25_MIN_IDF=7.0 bash scripts/run_eval.sh`;
      must not regress vs dense-only (gating makes it non-harmful)

---

## Root cause summary

- **Answers are short stubs**: relevant pages median 133 words, 51% ≤ 150 words (one chunk),
  79% < 256 tokens (MiniLM truncation point). Corpus median: 1218 words.
- **Chunking can only hurt**: it adds zero signal to short answer pages and multiplies
  long distractor pages into many lottery-ticket vectors. Max-pooling then floods top-10
  with false positives from long, irrelevant pages.
- **`count_corrected` was a length prior in disguise**: `n_chunks ∝ length` → the 0.2578 gain
  came from demoting long pages, not from chunking. Use real word count directly.
- **BM25 only helps ~1/50 queries**: `1,456,779` is the single genuinely rare singleton.
  Years, common words have low IDF. Equal-weight RRF diluted dense results → regression.
- **Methodology**: never co-mingle rebuild + scoring change; isolate one variable per eval;
  use CV mean (not full-set peak) to guard against overfitting 50 labeled examples.

## Next action (Stage 2)

All no-rebuild. Run via `bash scripts/run_eval.sh` and `python scripts/cv_eval.py`.

1. `AGGREGATE_MODE=chunk_0_only USE_BM25=0` — baseline control
2. `AGGREGATE_MODE=length_prior USE_BM25=0` — β sweep (cv_eval.py); pick plateau-center
3. `AGGREGATE_MODE=length_prior USE_BM25=1 BM25_MIN_IDF=7.0` — gated BM25 (must not regress)
