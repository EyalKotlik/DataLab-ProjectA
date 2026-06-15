# Why Section B underperforms (≈0.26 vs peers' ≈0.45)

Date: 2026-06-15. Reproduce with `python diagnose_retrieval.py` (numpy+stdlib only)
and `python diagnose_hybrid.py` (needs the project env). Raw run:
`logs/diagnosis-bm25-20260615.log`.

## TL;DR

We are not stuck on a tuning knob — we have been micro-optimizing the wrong stage.
Both of our signals are individually weak **and** our fusion is built so that the
strong one (lexical) is throttled. The real lever is **ranking precision over a
high-recall candidate pool**, which we have never actually exercised.

| Approach | mean NDCG@10 |
|---|---|
| Dense baseline (single chunk, max-pool) | 0.2241 |
| **Pure BM25 (full corpus)** | **0.2331** |
| Our best to date (count_corrected β=0.05 + gated BM25 RRF) | 0.2578 |
| Peer teams | ~0.45 |

## The decisive measurement

Pure BM25 recall of *any* relevant page, by depth (50 public queries):

```
recall@1   = 0.12     recall@50  = 0.84
recall@3   = 0.22     recall@100 = 0.84
recall@10  = 0.52     recall@500 = 0.94
```

Rank of the best relevant doc per query (sorted):
`[1,1,1,1,1,1,2,2,2,2,2,4,5,5,6,7,7,7,7,8,8,8,8,9,10,10, 11,13,14,16,17,17,19,20,25,27,29,30,33,50,50, 106,300,307,330,487,565,2891,24124]`

**Interpretation:** the answer is almost always *retrievable* (94% by rank 500,
84% by rank 50) but badly *ranked* (only 52% inside top-10, 12% at rank 1). About a
dozen queries have the true doc sitting at ranks 11–50 — recoverable points we are
currently throwing away. This is the classic "recall is fine, precision@10 is the
problem" regime, where fusion + reranking move the needle and single-model tuning
does not.

## Root causes

### 1. We treated dense vs lexical as either/or, and broke the fusion that should combine them
The dataset is adversarial paraphrase: queries reuse the *discriminating* tokens of
the answer page — exact numbers (`1,456,779`, `1987`, `1965`, `September 1958`) and
rare entities (`radiometry`, `thermal imaging pipelines`, `subsurface acoustic
tomography`). Dense MiniLM blurs those; BM25 nails them. They have **complementary
errors**, so a real hybrid should clear both ~0.23 plateaus. Instead `retrieve.py`:

- **Gates BM25 behind `BM25_MIN_IDF=7.0`** (fires only for tokens in ≲25 docs) and
  only when "needle tokens" exist — so most queries run dense-only.
- Fuses with **RRF at k=60**, which flattens rank differences (1/60 vs 1/70): a BM25
  rank-1 exact match barely outweighs a dense rank-30. The exact-match advantage is
  erased by the fusion constant.
- Is **dense-anchored** by design (`BM25_WEIGHT≤1`), capping lexical influence.

Net effect: best hybrid (0.2578) is barely above dense-only — the logs show we
spent dozens of runs tuning β, λ, IDF gate, RRF k around a fusion that can't help.

### 2. We over-invested in chunking/aggregation, which this dataset doesn't reward
Relevant pages are short (answer in the lead, ~50–280 words); bulk corpus pages are
long (median ~1218 words). `CHUNKING_PROGRESS.md`/`IMPROVEMENTS.md` correctly found
that body chunks only add false positives, and `length_prior`/`count_corrected`
penalties claw a little back. But that whole line of work is a ±0.03 side-quest. The
lead-only signal tops out ~0.25 no matter how we pool it, because the ceiling is set
by **bi-encoder ranking quality**, not by which chunk we read.

### 3. No reranking stage exists
We retrieve and rank in one shot. With recall@50 = 0.84, a second-pass reranker over
the top ~50–100 candidates is the standard way to convert that recall into top-10
precision — and we have never tried it.

## How I would fix it (highest leverage first)

1. **Rebuild the hybrid as a true fusion, not a gated add-on.**
   - Drop the IDF gate (or lower to ~3–4); let BM25 score every query.
   - Replace RRF-k=60 with either RRF k≈5–10 *or* z-score-normalized weighted score
     fusion (`α·dense_z + (1−α)·bm25_z`), sweeping α. Low k / score fusion lets a
     confident exact match win, which is exactly what the number/entity queries need.
   - `diagnose_hybrid.py` measures this ceiling directly over the BM25 candidate pool
     (cheap: embeds only ~few-thousand candidates, no full rebuild). **Run it next** —
     it tells us how much headroom fusion alone buys before we commit to a rebuild.

2. **Add a MiniLM rerank pass over the fused top-~100.** We are restricted to the
   fixed bi-encoder, but we can still rerank candidates by embedding the *lead
   sentence(s)* of each candidate (tighter, less-diluted unit than full content) and
   re-scoring against the query. Cheap at runtime (≤100 docs/query) and directly
   targets the rank-11–50 misses.

3. **Strengthen BM25 itself:** title-field boost (answers are entity pages; the
   title token is highly discriminative) and verify the number tokenizer survives
   into the index (`1,456,779` must stay one high-IDF token).

4. **Stop tuning β/λ/chunking.** Freeze `length_prior` as-is; it's a rounding error
   next to fixing fusion + reranking.

## Open question worth one experiment
Run `diagnose_hybrid.py` in the project env and paste results below. If well-fused
dense+BM25 over the candidate pool lands ≥0.40, the fix is purely in `retrieve.py`
(no re-embed needed). If it lands ~0.30, we additionally need the lead-sentence
rerank (#2), which requires re-embedding the corpus at lead granularity.

<!-- diagnose_hybrid.py results (fill in):
candidate docs to embed: ____
dense (over bm25 cand) NDCG@10 = ____
RRF k=5/10/20/60 = ____
zscore fuse dense_w=0.3/0.5/0.7 = ____
-->
