# Feasibility of further improvement (post-lever-sweep, 2026-06-16)

This memo re-assesses the headroom toward higher NDCG@10 **after** the toggleable-lever
A/B sweep landed. It updates the judgments in [PLAN_NDCG_0.5.md](PLAN_NDCG_0.5.md), which
was written *before* those levers were measured. Read it as the current verdict on
"is more improvement realistic, and if so how."

Baseline: **0.4338** (`zfuse`: `ZFUSE_DENSE_W=0.8`, `ZFUSE_BETA=0.15`, `ZFUSE_CAND_N=300`),
29 public queries, ~26 s.

---

## 1. What the new results change

The 2026-06-16 sweep (full table in [DIAGNOSIS.md](DIAGNOSIS.md)) measured every baked-in
lever. **None beat baseline:**

| Lever | Predicted (old plan) | Measured | Reality |
|---|---|---|---|
| `ZFUSE_CHUNK_AGG=max` (body chunks) | unknown | 0.4115 | −0.022 — body chunks hurt, as on the corrupted set |
| `ZFUSE_TITLE_W=0.2` (L6 title vector) | **+0.01–0.03** | 0.4044 | **−0.029 — title blend dilutes the lead vector** |
| `BM25_TITLE_BOOST=3` (L7) | **+0.01–0.03** | 0.4338 | ±0.000 — inert |
| `BM25_TEMPORAL=1` (L9) | +0.00–0.02 | 0.4322 | −0.002 — inert/noise |

This is the important signal: **the two levers the old plan rated "med confidence,
+0.01–0.03" (L6 title-aware dense, L7 title boost) returned flat-to-negative.** The
title is already inside the lead chunk's text on both the dense and BM25 sides; exposing
it as a separate field adds no new information and, blended, only adds noise. The
pre-baked alternative *representations* of each page are exhausted — the lead-chunk +
length-prior + BM25-fusion recipe has already extracted essentially all the signal this
embedding model and this corpus carry at the page level.

**Consequence: Phase 2 of the old plan (offline-rebuild levers L6/L7/L9) is dead.** Do
not ask the user for a GPU rebuild to chase them — the rebuild cost buys a measured loss.

The one *good* news: `artifacts/` is now committed via Git LFS (`index_vectors.npy`,
`bm25.json.gz`, `index_meta.json` all tracked), so the old plan's **blocking** "0
functional" risk is resolved. The submission floor is safe.

## 2. The honest size of the problem

- Recall is **not** the bottleneck — ranking precision is. Pure BM25 already puts the
  answer in the top 50 for ~90% of queries; only 66% land in top-10 (DIAGNOSIS §Core).
  So levers that widen recall (L1 below) can only help the ~10% genuinely clipped, and
  only if those clipped answers then *rank* well — which the length prior + fusion must
  earn.
- 29 queries, mean 2.0 relevant pages each. Each query is worth ~3.4 NDCG points. A
  measured "+0.01" is **a third of one query** — indistinguishable from noise without
  cross-validation. The −0.02/−0.03 lever losses are real (multiple queries); plausible
  *gains* from what's left are at the noise floor.
- Fixed `all-MiniLM-L6-v2`, `numpy`/`sentence-transformers`/`faiss-cpu` only. **No
  cross-encoder, no reranker, no better embedding model** — the standard play that takes
  retrieval from 0.43 to 0.55+ is off the table by rule.

## 3. What is still genuinely live

Only **no-rebuild** levers remain credible. Ranked by expected value, lowered from the
old plan's estimates given the L6/L7 disappointment:

### L2 — global-dense union into the candidate pool *(the only ceiling-raiser)*
- **Why it's the one worth trying:** every other lever re-ranks pages already in the
  BM25 pool. A relevant page with **zero query-token overlap** (the documented synonym
  gap: `captained`, `modernize`, `negotiated` absent from the answer) never enters the
  pool and is **unreachable at any fusion setting**. Adding per-query global-dense top-`M`
  to the union is the only move that can recover those queries — it raises the ceiling,
  not just the sort order.
- **Honest caveat:** dense-alone over the full corpus is weak here (~0.22 global) because
  long distractors score high; the length prior is the only guard. If the union floods
  the pool with long-page dense false positives, β has to be re-tuned and may not hold.
  This is the lever most likely to *help* and also the one most able to *backfire*.
- **Cost:** no GPU rebuild; needs embedding a wider candidate set in the diagnostic (or a
  targeted check on just the out-of-pool queries). ~1–2 CPU-min per setting.
- **Realistic gain:** +0.00 to +0.03, medium-low confidence. Verdict: **worth one bounded
  experiment**, gated by per-query error analysis (which queries are actually out-of-pool).

### L1 — widen `ZFUSE_CAND_N` (300 → 500/750/1000)
- Pure env-var sweep, no rebuild, minutes to run. But recall is not the bottleneck, so
  expect small. **Realistic gain: +0.00–0.01.** Cheap enough to run as a sanity check
  alongside L2; do not expect it to carry.

### L3/L4 — fusion refinement + re-tune `(dense_w, β)`
- Only meaningful *after* L1/L2 change the pool composition. Re-running the `dense_w × β`
  grid is cheap but, on 29 queries, any new peak is overfit until CV says otherwise. β=0.15
  is already a strong short-doc fit; pushing it further risks the hidden set if it contains
  any long answer pages. **Realistic gain: +0.00–0.01, and a regression risk.**

### `BM25_K1`/`BM25_B` — the only untested baked-in lever
- Predicted ≤+0.01; given that every other BM25-side lever was inert, expect ~0. Run it
  once for completeness while the harness is warm; don't invest in a sweep.

## 4. Verdict: is 0.50 realistic?

**No — not with the allowed model and packages.** The old plan's path to 0.50
("L1+L2+L4 → 0.46–0.48, then L6/L7 rebuild → 0.50+") is broken: its rebuild half is now
measured negative. What remains (L1+L2+fusion re-tune) is all no-rebuild and, on the
evidence, more likely to land **0.43–0.46** than 0.50. The "peers' ~0.45" target is
plausibly reachable; **0.50 is not, short of a query-specific hack that won't survive CV
or generalize to the hidden set.**

## 5. Realistic plan (bounded, ~half a day)

Do this much and then stop — the marginal NDCG is no longer the highest-value use of time
(functional score is 50% of the grade; repo quality + video are the other 50%).

1. **Per-query error analysis first.** Write `diagnose_errors.py`: for each public query,
   print NDCG@10 and the rank of every relevant page in (a) global dense, (b) BM25,
   (c) the candidate pool, (d) the final list; flag pages **not in the pool at all**.
   This is the gate — it tells us whether L2 even has anything to recover. *No tuning
   before this exists.* Save the table to DIAGNOSIS.md.
2. **If — and only if — step 1 shows out-of-pool relevant pages:** run **L2** (global-dense
   union, `M ∈ {25,50,100}`) in the diagnostic on those specific queries. Accept only if it
   recovers ≥1 query without flooding others. Re-tune β if long distractors leak in.
3. **L1** `ZFUSE_CAND_N` sweep (env var) as a cheap recall sanity check, same session.
4. **CV-gate anything that looks like a win.** Write `diagnose_cv.py` (5-fold over 29
   queries). A change that doesn't win on held-out folds is reverted — a hidden-set
   regression costs more than a marginal public gain. Track runtime (must stay well
   under 60 s).
5. **Stop-loss:** if after steps 1–4 no CV-stable gain ≥ +0.01 exists, **lock 0.4338 as
   the submission** and stop tuning. It is already a robust, well-documented config.
6. **Reallocate remaining effort to the 50% non-functional grade:** the repo is already
   in good shape (README, DIAGNOSIS, committed artifacts) — polish it and invest in the
   video, where marginal return is far higher than chasing +0.01 NDCG.

**Do not:** request a GPU rebuild for any lever (all rebuild levers are measured ≤0);
re-try chunking/sentence/RRF/title-blend/temporal (all refuted); or ship a 0.50-on-public
config that CV calls fragile.

## 6. One-line bottom line

The cheap and medium levers are spent; the lead-chunk + length-prior + BM25-fusion recipe
is at or near the practical ceiling for this model. **One bounded, error-analysis-driven
L2 experiment is the only move with real upside; everything else is noise-floor tuning.**
Budget half a day, CV-gate it, then lock the config and spend the rest of the time on the
repo and video.

---

## 7. Outcome (plan executed, 2026-06-16)

The plan was executed in full (harness + scripts under `experiments/`, log in
`experiments/PROGRESS.md`). The harness reproduces production 0.4338 exactly.

- **Step 1 error analysis:** recall is *not* the bottleneck — 93/100 relevant pages are
  already in the candidate pool; only 7 (4 queries) are out-of-pool, and those have deep
  global-dense ranks. The dominant failure is in-pool pages the fusion demotes out of
  top-10 (e.g. q18's relevant pages sit at dense ranks 8–26 but β=0.15 buries them).
- **L1 (widen cand_n):** monotonically *worse* (0.434 → 0.403). 300 is the peak.
- **L2 (global-dense union):** *catastrophic* (0.19–0.33) — long-page false positives.
- **L4 ((dense_w, β) grid):** current (0.8, 0.15) is the **global maximum**.

**No config beat 0.4338 on the full set**, so there was nothing for CV to gate — the
stop-loss (§5.5) triggered. The residual failures need a stronger ranker (cross-encoder /
better embeddings), which the package + fixed-model constraints forbid. **0.4338 is locked
as the submission; effort moves to repo quality + video (the other 50% of the grade).**
