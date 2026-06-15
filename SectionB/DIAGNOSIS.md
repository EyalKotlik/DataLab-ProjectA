# Section B — retrieval diagnosis & evidence log

Detailed empirical record behind the `zfuse` pipeline (see [README.md](README.md) for
the pipeline itself). All numbers are mean NDCG@10 on the **corrected 29 public
queries** unless noted. Reproduce with `diagnose_retrieval.py` (numpy/stdlib),
`diagnose_hybrid.py`, `diagnose_rerank.py` (need the project env).

> Note: an earlier investigation used a **corrupted 50-query set** the TA later fixed.
> Its conclusions (notably "fusion doesn't help", "try sentence granularity") did not
> survive the correction and are not reproduced here. Only corrected-set results below.

## Dataset shape (what makes this hard)

- Corpus: 27 074 pages. Synthetic, deliberately fictionalized entities ("Lkers", "BBA",
  years shifted ~160 yrs). Answer pages are **short** (~50–280 content words);
  distractor pages are **long** (median ~1218 words).
- Queries (29): adversarial paraphrases, ~1–4 relevant pages each (mean 2.0). Two
  recurring matching gaps:
  - **Temporal**: query `1820s` (token `1820`) vs page `1826`; "decades after founded"
    vs a concrete year — no lexical or embedding bridge.
  - **Synonym paraphrase**: distinctive query words (`captained`, `modernize`,
    `redesign`, `negotiated`) absent from the answer page, which uses synonyms.

## Core measurements

```
PURE BM25 (full corpus):           0.3191
  recall@1=0.21 @3=0.38 @10=0.66 @50=0.90 @100=0.90 @500=1.00
```
**Recall is not the bottleneck — ranking precision is.** The answer is almost always
retrievable (90% by rank 50, 100% by rank 500) but only 66% land in top-10. This is the
regime where candidate-restriction + a good prior + fusion pay off, and where tuning a
single global model does not.

## The winning recipe (`zfuse`)

`diagnose_hybrid.py` (4706 candidates = union of per-query BM25 top-300):

```
dense over BM25 candidate pool                         = 0.3425
RRF fusion           k=5/10/20/60                       = 0.3518/0.3548/0.3612/0.3534
z-score fusion (no prior) dense_w=0.6/.7/.8/.85/.9/.95  = 0.3741/0.3739/0.3722/0.3815/0.3660/0.3911

dense + length prior ONLY (no BM25) β=0/.05/.1/.15/.2/.3
                                    = 0.3425/0.4053/0.3652/0.3272/0.3122/0.2694

z-fusion + length prior:
  dense_w=0.8  β=0.1/.15/.2/.3 = 0.4231/0.4332/0.4176/0.4114   <- BEST β=0.15
  dense_w=0.85 β=0.1/.15/.2/.3 = 0.4237/0.4135/0.4331/0.3911
  dense_w=0.9  β=0.1/.15/.2/.3 = 0.4202/0.4081/0.3596/0.3263
```

**Operating point: dense_w=0.8, β=0.15 → 0.4332** (confirmed in production:
`eval_public.py` = **0.4338**, 28 s).

Key reading: the length prior *alone* over-penalizes past β≈0.05 (0.4053 → collapse),
but **fused with BM25 it tolerates β=0.15** — BM25 re-anchors the exact matches (rare
names, numbers) that a strong length prior would otherwise bury. Fusion and length
prior are complementary, not redundant; candidate restriction removes global distractors
that pure dense ranks high (0.343 over candidates vs ~0.22 global baseline).

## Refuted ideas (measured worse — do not retry)

| Idea | Result | Why it fails |
|---|---|---|
| Overlapping body chunking (W180/S60) | 0.10–0.12 | 970 K vectors; long pages win the max-pool "lottery", flooding top-10 |
| Sentence-granularity (`sent_max`) | 0.2678 | worse than full-doc dense (0.343); query matches whole short page, not one sentence |
| `lead_anchored`, `mean_top2` | ≤0.21 | any non-lead-chunk weight adds noise |
| Gated BM25 + RRF k=60 (old default) | 0.2527 | IDF gate fires on ~nothing; RRF k=60 flattens exact-match advantage |
| Decade expansion in BM25 | 0.3191 | identical to pure BM25 — no effect |

`lead_sentence` in `diagnose_rerank.py` reads 0.0152 — that metric matches the *title*
only (script artifact), ignore it.

## Methodology notes (lessons that cost us time)

- **Isolate one variable per eval.** A past regression co-mingled an index rebuild with
  a scoring change and could not be attributed; always change one thing.
- **Candidate-pool z-scoring is scale-sensitive.** Fusion weight `dense_w` is only
  meaningful relative to the pool the z-scores are computed over; `zfuse` z-scores over
  the BM25 candidate union, matching the diagnostic that tuned it.
- **The length signal is robust to its exact definition.** The sweep used
  `log(token_count of title+content)`; production uses `log(content_word_count)`. The
  short-vs-long log *gap* (~3.2) is near-identical and z-scoring removes scale offsets —
  confirmed by the 0.4332 (diagnostic) ≈ 0.4338 (production) match.

## Open items (headroom toward peers' ~0.45)

1. **Widen recall**: `ZFUSE_CAND_N` 300 → 500/1000 (recall@500=1.00 vs @100=0.90); est. +0.02–0.03.
2. **k-fold CV on β/dense_w** before locking — guard against overfitting 29 labeled queries
   and against any long answer pages in the hidden set (the length prior would hurt those).
3. **Query-side rewriting** for the temporal/synonym gaps — last resort, after 1–2.
