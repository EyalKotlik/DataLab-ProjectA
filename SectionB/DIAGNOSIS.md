# Why Section B underperforms (≈0.26 vs peers' ≈0.45)

Date: 2026-06-15. Reproduce with `python diagnose_retrieval.py` (numpy+stdlib only)
and `python diagnose_hybrid.py` (needs the project env). Raw run:
`logs/diagnosis-bm25-20260615.log`.

## TL;DR (revised 2026-06-15 after running diagnose_hybrid.py)

We are not stuck on a tuning knob, but my first hypothesis was also wrong: **fixing
the fusion does not help.** A properly built hybrid (un-gated BM25, swept RRF k,
z-score weighted fusion) tops out at **0.2632** — essentially our current 0.2578.
Dense and BM25 do **not** make complementary errors here; they rank the *same*
distractors high, so combining two weak full-document signals stays weak.

The real bottleneck is **ranking precision of the matching signal itself**, not
recall (recall@50 = 0.84) and not fusion. The remaining untested lever is the
matching **unit** — see "Revised plan" below.

| Approach | mean NDCG@10 | source |
|---|---|---|
| Dense baseline (single chunk, max-pool) | 0.2241 | logs |
| Pure BM25 (full corpus) | 0.2331 | diagnose_retrieval.py |
| Dense over BM25 candidate pool | 0.2339 | diagnose_hybrid.py |
| Our best to date (count_corrected β=0.05 + gated BM25 RRF) | 0.2578 | logs |
| RRF fusion, best k (=20) | 0.2542 | diagnose_hybrid.py |
| **z-score fusion, best (dense_w=0.7)** | **0.2632** | diagnose_hybrid.py |
| Peer teams | ~0.45 | reported |

### diagnose_hybrid.py results (2026-06-15, 4706 candidates)
```
dense (over bm25 cand)        = 0.2339
RRF k=5 / 10 / 20 / 60        = 0.2409 / 0.2470 / 0.2542 / 0.2503
zscore fuse dense_w=0.3/.5/.7 = 0.2297 / 0.2437 / 0.2632
```
Better fusion buys ~+0.005 over our current pipeline. Not the gap to 0.45.

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

## Results log

diagnose_hybrid.py — done (see table above): fusion ceiling 0.2632. Hypothesis
"fusion is the lever" REFUTED.

diagnose_rerank.py — **TODO, run on GPU box, paste below:**
<!--
candidate docs / sentences: ____
lead_sentence  = ____
sent_max       = ____
sent_max+BM25 dense_w=0.4/0.6/0.8 = ____
BM25 + decade-expansion = ____
-->

