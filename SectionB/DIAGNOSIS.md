# Section B diagnosis

> **IMPORTANT (2026-06-15, late):** the public query set the TA first gave us was
> **wrong**. With the corrected set (29 queries) every number jumps and several
> conclusions flip. The corrected analysis is below; everything under
> "── HISTORICAL (corrupted 50-query set) ──" is kept only as a record of what was
> tried and why the conclusions changed. **Read the corrected section.**

Reproduce: `python diagnose_retrieval.py` (numpy/stdlib), `python diagnose_hybrid.py`
and `python diagnose_rerank.py` (need the project env).

## Corrected query set (29 queries) — current picture

| Approach | corrected | (corrupted) | source |
|---|---|---|---|
| **Production `eval_public.py` (shipping today)** | **0.2527** | — | eval_public.py |
| Pure BM25 (full corpus) | 0.3191 | 0.2331 | diagnose_retrieval.py |
| Dense over BM25 candidate pool | 0.3425 | 0.2339 | diagnose_hybrid.py |
| RRF fusion, best k (=20) | 0.3612 | 0.2542 | diagnose_hybrid.py |
| z-score fusion, dense_w=0.95 (no prior) | 0.3911 | — | diagnose_hybrid.py |
| **z-fusion dense_w=0.85 + length prior β=0.1** | **0.4237** | — | diagnose_hybrid.py |
| sent_max (best-sentence matching) | 0.2678 | — | diagnose_rerank.py |
| BM25 + decade-expansion | 0.3191 | — | diagnose_rerank.py |
| Peer teams | ~0.45 | ~0.45 | reported |

```
recall@1=0.21  @3=0.38  @10=0.66  @50=0.90  @100=0.90  @500=1.00   (BM25, 29q)
```

## The headline: production is mis-configured, not under-powered

Production ships **0.2527**, yet the *same artifacts* reach **0.42** under a better
`retrieve.py`. We are leaving ~0.17 NDCG on the table with **no re-embed**. Why the
gap:

- Production ranks dense **globally** (over all 27k vectors) then RRF-merges a
  **gated** BM25 (`BM25_MIN_IDF=7.0` → fires on almost nothing) with **k=60** flatten.
- The winning recipe instead: **BM25 generates a candidate pool**, dense is scored
  **within that pool**, both are **z-score normalized and weighted-fused**, and a
  **length prior** demotes long pages. The BM25 candidate restriction alone removes a
  huge mass of global distractors that pure global-dense ranks high.

### Two findings that drive the recipe
1. **Length prior is the biggest single lever.** Relevant pages are short (~50–280
   words); distractors are long (median ~1218). Penalizing `−β·log(len)` lifts
   dense_w=0.85 from 0.3815 → **0.4237** at β=0.1, and β was **still climbing** — the
   extended sweep (β up to 0.3) in `diagnose_hybrid.py` will find the peak.
2. **Fusion + candidate restriction matter, sentence-granularity does not.**
   z-fusion (dense_w≈0.85–0.95) over BM25 candidates beats global dense; `sent_max`
   (0.2678) is *worse* than full-doc dense — do **not** re-embed at sentence level.
   Decade expansion is a no-op (0.3191 == pure BM25).

## Revised plan (priority order, all no-re-embed unless noted)

1. **Rewrite `retrieve.py` retrieval to the winning recipe.** Pipeline per query:
   BM25 top-N (N≈300) → candidate pool → score dense within pool → apply
   `−β·log(word_count)` length prior on dense → z-score normalize dense & BM25 →
   rank by `dense_w·dz + (1−dense_w)·bz`. Start at **dense_w=0.85, β=0.1**
   (measured 0.4237). Keep old modes behind env vars. **This is the action item.**
2. **Pin β (and re-confirm dense_w).** β still rising at 0.1; the extended
   `diagnose_hybrid.py` sweep (β=0.1/0.15/0.2/0.3, plus length-prior-only isolation)
   locates the peak and tells us how much BM25 actually adds over length-penalized
   dense alone. Run next.
3. **Widen recall ceiling.** recall@100 = 0.90, recall@500 = 1.00. Enlarge the BM25
   candidate pool (top-500/1000) or union with global dense top-K so the ~10% of
   answers below BM25 rank 300 become reachable. Likely +0.02–0.03.
4. **Guard against overfitting the 29-query set.** β≈0.1+ is a strong short-doc prior
   tuned on 29 queries; it should generalize (hidden set shares the synthetic
   short-answer structure) but confirm with k-fold CV over the public queries before
   locking β. Re-baseline `eval_public.py` after the rewrite for an honest before/after.

## If this plateaus (~0.42)
Residual gap to peers is then **query-side**: rewriting/expanding the multi-hop
paraphrase queries before encoding. Pursue only after items 1–3.

──────────────────────────────────────────────────────────────────────────────
## ── HISTORICAL (corrupted 50-query set) ── superseded; kept for the trail

The sections below were written before we learned the query set was wrong. The
conclusions ("fusion doesn't help", "try sentence granularity") were correct *for the
corrupted data* but are superseded by the corrected analysis above.

### (historical) diagnose_hybrid.py — corrupted set
```
dense (over bm25 cand)        = 0.2339
RRF k=5 / 10 / 20 / 60        = 0.2409 / 0.2470 / 0.2542 / 0.2503
zscore fuse dense_w=0.3/.5/.7 = 0.2297 / 0.2437 / 0.2632
```

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

### 1. The matching signal is weak, and dense+BM25 fail *together* (not complementarily)
I initially blamed the broken/over-gated fusion in `retrieve.py` (IDF gate 7.0,
RRF k=60, dense-anchored weight). Those are still poor defaults, but **un-gating and
re-fusing does not move the score** (0.2632 best, table above). Dense and BM25 are
fooled by the *same* distractor pages, so fusion has nothing to recover. The fusion
config is a red herring; the matching itself is the wall.

Token analysis shows why the signal is weak — two recurring failure modes:
- **Temporal-reasoning gaps:** `"1820s"` (→ token `1820`) vs doc `1826`;
  `"decades after the company was founded"` vs a specific year. No lexical or
  embedding bridge between a decade/relative phrase and a concrete year.
- **Synonym paraphrase:** distinctive query words (`captained`, `modernize`,
  `redesign`, `negotiated`) are absent from the answer page, which uses synonyms.
  MiniLM-L6 (a small bi-encoder) only partly bridges these.

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

## Revised plan (highest leverage first)

Fusion is settled: ~+0.005, not worth a rebuild on its own. The only lever that
attacks the actual bottleneck (precision of matching) is changing the **unit** we
match on, plus closing the temporal gap.

1. **Sentence-granularity matching (the main bet).** We score the whole truncated
   doc today. Answers sit in one short lead sentence; distractors are long
   multi-topic pages whose averaged/truncated vector drifts toward generic topic
   similarity. Score each candidate by its *best-matching sentence* instead.
   `diagnose_rerank.py` measures `lead_sentence`, `sent_max`, and `sent_max+BM25`
   fusion over the candidate pool — **run this next.** If `sent_max` clears ~0.35+,
   the production fix is a re-embed at sentence granularity + max-pool in
   `retrieve.py` (the multi-chunk plumbing already exists; this is the "right" use
   of chunking, vs the body-window chunking that failed).

2. **Decade / temporal expansion in BM25.** `"1820s"` should also match years
   1820–1829; relative phrases ("decades after") need the founding year resolved.
   `diagnose_rerank.py` includes a `BM25 + decade-expansion` test. Cheap, build-time,
   targets a concrete cluster of failing queries.

3. **Strengthen BM25 fields:** title-field boost (answers are entity pages; the
   title token is highly discriminative) and confirm `1,456,779` survives as one
   high-IDF token in the built index.

4. **Stop tuning β/λ/chunking-aggregation and fusion knobs.** All measured at
   ±0.005. Freeze current defaults; spend effort only on items 1–3.

## If sentence-matching also plateaus (~0.26)
Then the ceiling is MiniLM-L6's semantic resolution on these adversarial
paraphrases, and the gap to peers is likely *query-side*: query rewriting/expansion
(decade→years, synonym expansion, decomposing multi-hop questions) before encoding.
That's the next branch to explore — but only after diagnose_rerank.py rules out the
cheaper unit-change fix.

## Results log (corrected query set, authoritative)

eval_public.py (production default) = **0.2527** (29q, 29.4s)

diagnose_rerank.py:
```
lead_sentence = 0.0152   (title-only artifact — ignore)
sent_max      = 0.2678   (WORSE than full-doc dense; sentence granularity rejected)
sent_max+BM25 dense_w=0.4/0.6/0.8 = 0.2932 / 0.3045 / 0.3081
BM25 + decade-expansion = 0.3191  (== pure BM25; no effect)
```

diagnose_hybrid.py (4706 cands):
```
dense (over bm25 cand) = 0.3425
RRF k=5/10/20/60       = 0.3518 / 0.3548 / 0.3612 / 0.3534
z-fuse dense_w 0.6/0.7/0.8/0.85/0.9/0.95 = 0.3741/0.3739/0.3722/0.3815/0.3660/0.3911
+ length prior:
  dense_w=0.8  β=0.02/0.05/0.1 = 0.3991 / 0.4160 / 0.4231
  dense_w=0.85 β=0.02/0.05/0.1 = 0.4063 / 0.4035 / 0.4237   <- best
  dense_w=0.9  β=0.02/0.05/0.1 = 0.4163 / 0.4143 / 0.4202
```

diagnose_hybrid.py — extended sweep (final):
```
dense + length prior ONLY (no BM25): β=0/.05/.1/.15/.2/.3 =
  0.3425 / 0.4053 / 0.3652 / 0.3272 / 0.3122 / 0.2694   (peaks β=0.05, then collapses)
fusion + length prior:
  dense_w=0.8  β=0.1/.15/.2/.3 = 0.4231 / 0.4332 / 0.4176 / 0.4114   <- BEST β=0.15
  dense_w=0.85 β=0.1/.15/.2/.3 = 0.4237 / 0.4135 / 0.4331 / 0.3911
  dense_w=0.9  β=0.1/.15/.2/.3 = 0.4202 / 0.4081 / 0.3596 / 0.3263
```
**Chosen operating point: dense_w=0.8, β=0.15 → 0.4332.**
Key reading: the length prior *alone* over-penalizes past β=0.05 (0.4053 → collapse),
but **fused with BM25 it tolerates β=0.15** — BM25 re-anchors exact matches that the
strong length prior would otherwise bury. Fusion and length prior are complementary,
not redundant. Implemented as the new default `zfuse` mode in `retrieve.py`.

