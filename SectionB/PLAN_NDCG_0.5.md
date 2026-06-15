# Plan — surpass mean NDCG@10 = 0.50

Target: **mean NDCG@10 > 0.50** on the hidden grading queries.
Baseline: **0.4338** (`zfuse` default: `ZFUSE_DENSE_W=0.8`, `ZFUSE_BETA=0.15`,
`ZFUSE_CAND_N=300`), measured on the 29 public queries in ~28 s.

This document is the execution plan, not a results log. Record outcomes in
`DIAGNOSIS.md` as each experiment lands.

---

## 1. The size of the gap (read this first)

The gap is **+0.066 absolute (~15% relative)** over the current default, and the
documented headroom toward peers (~0.45) is only +0.02–0.03. **0.50 will not come
from one knob.** It requires stacking 2–4 independent wins, and the search is over
**29 labeled queries** — each query is worth ~3.4 NDCG points, so 0.434→0.50 is
"fully fix ~2 more queries (or partially fix ~4) without breaking any."

Two consequences:

- **Error analysis drives everything.** Before tuning, we must know *which* queries
  currently score 0 or near-0 and *why* (out-of-pool? ranked just below 10? long
  distractor on top?). The path to +0.066 is a named list of queries, not a sweep.
- **Overfitting risk is severe.** Any gain found by sweeping on 29 queries can be
  noise. Every accepted change must survive **k-fold cross-validation** (Phase 4)
  before it is locked, or we risk a hidden-set regression.

### What is already refuted — do NOT revisit (from DIAGNOSIS.md)
Overlapping body chunking (0.10–0.12); sentence-granularity `sent_max` (0.268);
`lead_anchored` / `mean_top2` (≤0.21); gated BM25 + RRF k=60 (0.253); decade
expansion in BM25 (no effect). These are measured dead-ends; re-trying them is
wasted budget.

### Hard constraints that bound the solution space
- Packages: **`numpy`, `sentence-transformers`, `faiss-cpu` only.** No cross-encoder,
  no new model. (This rules out the usual "add a reranker" play.)
- Embedding model fixed: `all-MiniLM-L6-v2`, 384-dim, L2-normalized, 256-token trunc.
- `run(queries)` ≤ **60 s** wall-clock at grading (GPU). Current 29-query run ≈ 28 s;
  hidden set is larger (~50), so budget headroom must be tracked on every change.
- **`artifacts/` must be committed** (missing → 0 functional). See Phase 0 — it is
  *not currently committed*.

---

## 2. Principles (lessons already paid for)

1. **One variable per evaluation.** A past regression co-mingled an index rebuild
   with a scoring change and was unattributable. Never bundle.
2. **Prefer no-rebuild experiments.** `diagnose_hybrid.py` embeds only the BM25
   candidate union (~few thousand docs, ~1–2 min CPU) and reproduces the production
   number (0.4332 diag ≈ 0.4338 prod). Use it as the fast loop. A full
   `build_index.py` is a slow GPU job the **user runs manually** — gate every rebuild
   behind a diagnostic preview that already shows the win.
3. **Accept a change only if it (a) beats baseline on the full 29 set AND (b) does not
   regress in k-fold CV.** Single-number improvements on 29 queries are hypotheses,
   not results.
4. **Track runtime on every accepted change.** A win that blows the 60 s budget is a
   loss.

---

## 3. Phase 0 — Reproduce + build the safety net (BLOCKING)

Nothing below is trustworthy until this phase is done. **No `artifacts/` directory
exists locally and none is tracked in git or LFS** (confirmed: `git ls-files` shows
no artifacts; only SectionA data is in LFS). The repo would currently score **0
functional**.

| # | Task | Why | Risk if skipped |
|---|------|-----|-----------------|
| 0.1 | **User rebuilds the index** (`python scripts/build_index.py`) and we confirm `eval_public.py` reproduces **0.4338 ± noise**. | Can't measure anything without artifacts; pins the baseline. | All later numbers float. |
| 0.2 | **Set up Git LFS for `artifacts/`** (`git lfs track "SectionB/artifacts/*.npy" "SectionB/artifacts/*.json.gz"`, commit `.gitattributes`), commit the artifacts. Verify a fresh clone loads them. | Missing/oversized artifacts = 0 functional; `index_vectors.npy` is large. | Submission scores 0 regardless of NDCG. |
| 0.3 | **Per-query error-analysis script** (`diagnose_errors.py`): for each public query print NDCG@10, rank of each relevant page in (a) global dense, (b) BM25, (c) candidate pool, (d) final fused list; flag pages **not in the candidate pool at all**. | This is the map. Every later lever is justified against a named query failure. | We sweep blindly and overfit. |
| 0.4 | **Cross-validation harness** (`diagnose_cv.py`): 5-fold (or leave-one-out) over the 29 queries; reports mean and per-fold NDCG for a given parameter set. | Distinguishes real gains from 29-query noise. | We lock overfit params and regress on hidden set. |

**Exit criterion:** baseline reproduced at 0.4338, artifacts committed via LFS and
clone-verified, and a per-query failure table exists. Save the failure table to
`DIAGNOSIS.md`.

---

## 4. Phase 1 — No-rebuild levers (fast loop, highest EV first)

All of these reuse existing artifacts and are tunable via env vars / small edits to
`retrieve.py` (`_zfuse_batch`). Validate each with `diagnose_hybrid.py` first, then
confirm in `eval_public.py`. Apply the one-variable rule.

### L1 — Widen the candidate pool (`ZFUSE_CAND_N`) — *est. +0.02–0.03*
- **Hypothesis:** BM25 recall@500 = 1.00 vs @100 = 0.90; the answer is often in the
  pool but `cand_n=300` may still clip some. Sweep `cand_n ∈ {300, 500, 750, 1000}`.
- **Measure:** `diagnose_hybrid.py` with `CAND` parameterized; then prod env var.
- **Accept if:** monotone or peaked gain that survives CV. **Watch:** larger pool
  dilutes z-scores and grows runtime (union size × matmul). Record run time.
- **Fallback:** if no gain, the pool is not the limiter → L2 is the real recall fix.

### L2 — Add global dense top-K to the candidate union — *attacks the synonym gap*
- **Key insight:** the fused system can only rank pages that enter the pool. A
  relevant page with **zero query-token overlap** (the documented synonym-paraphrase
  gap: `captained`, `modernize`, `redesign`, `negotiated` absent from the answer page)
  **never enters the BM25 union and is unreachable** — no amount of fusion tuning
  recovers it. README headroom item #1 explicitly suggests "union with global dense
  top-K."
- **Change:** in `_zfuse_batch`, union the per-query BM25 top-`cand_n` **with** the
  per-query global dense (lead-cosine) top-`M` (try `M ∈ {25, 50, 100}`). These pages
  get a real `dense_z` and `bz=0` (neutral lexical), so the length prior + dense
  decide their rank.
- **Measure first in `diagnose_hybrid.py`** by extending `cand` with global dense
  top-M (note: the diagnostic currently only embeds the BM25 pool, so this needs
  embedding a wider set — or, cheaper, run it as a targeted check on the specific
  out-of-pool queries flagged in 0.3).
- **Accept if:** recovers ≥1 currently-unreachable query without flooding others with
  long-page dense false positives (the length prior is the guard). **Watch:**
  re-tune β if long distractors leak in.
- **Why this is the marquee lever:** it raises the *ceiling*, not just the ranking.
  Every other Phase-1 lever only re-sorts pages already in the pool.

### L3 — Fusion refinements (z-score modeling) — *est. +0.00–0.02*
Current `_zfuse_batch` sets unmatched-BM25 candidates to `bz=0` (the pool mean).
Test, one at a time:
- **BM25 imputation:** unmatched → pool **minimum** instead of 0 (push pure-dense
  candidates below lexical matches), vs current (mean). 
- **Rank fusion vs z fusion:** the diagnostic already shows z > RRF; do not re-try RRF,
  but test **min-max** normalization vs z as a robustness alternative.
- **Tie-break / clipping:** clip dense_z and bz to e.g. [−3, 3] so a single outlier
  page can't dominate.
- **Accept if:** CV-stable gain. **Fallback:** keep current fusion; it's already tuned.

### L4 — Re-tune `(dense_w, β)` at the new operating point — *est. +0.01*
Any change in L1/L2 (pool composition) shifts the optimal `(dense_w, β)`. After L1+L2
land, re-run the `dense_w × β` grid from `diagnose_hybrid.py` around the current
`(0.8, 0.15)`. **Do not lock the result without CV** — β=0.15 is already a strong
short-doc fit on 29 queries and the hidden set may contain long answer pages.

### L5 — β-ensemble (optional) — *est. +0.00–0.01*
Average the fused ranks from two operating points (e.g. β=0.10 and β=0.20) to hedge
the length-prior aggressiveness. Cheap; only adopt if CV shows it's more *stable*, not
just marginally higher on 29 queries.

**Phase 1 exit criterion:** a locked, CV-validated config that ideally reaches
~0.46–0.48 using existing artifacts. Record the new defaults in `retrieve.py` and
`CLAUDE.md`/`README.md`.

---

## 5. Phase 2 — Offline-rebuild levers (gate each behind a diagnostic preview)

These change `artifacts/` and require a manual GPU rebuild by the user. **Preview the
expected win in `diagnose_hybrid.py` (which re-embeds candidates on the fly) before
asking for any rebuild.** BM25-only rebuilds are cheap CPU and can be done more freely.

### L6 — Field-weighted / title-aware dense embedding
- **Hypothesis:** queries are entity-centric ("Los Angeles basketball franchise",
  "river delta municipality"); the title is high-signal but currently diluted inside
  the 256-token title+content lead. Store a **second vector per page = title-only
  embedding**, and at query time score `cos = max(cos_lead, γ·cos_title)` or a weighted
  sum.
- **Preview:** in the diagnostic, embed candidate titles separately and test the
  title/lead combination over the BM25 pool — no full rebuild needed to estimate the win.
- **Rebuild cost:** +1 vector row per page (~27 K extra rows; index roughly doubles —
  still far below the refuted 970 K body-chunk blowup). Check 60 s budget.
- **Accept if:** preview shows gain; then user rebuilds. **Fallback:** title-only is a
  field, not a new model — low risk, but if it adds noise, drop it.

### L7 — Field-weighted BM25 (title boost) + k1/b tuning — *cheap (CPU rebuild)*
- **Hypothesis:** BM25 currently concatenates title+content with equal weight. Boosting
  title-term frequency (e.g. title tokens counted ×2–3) sharpens entity matching and
  improves both candidate generation (L2's pool) and the fusion anchor.
- Also sweep `k1 ∈ {1.2, 1.5, 2.0}`, `b ∈ {0.5, 0.75, 1.0}` — `b` interacts with the
  short-vs-long length signal that is central here.
- **Cost:** BM25 build is fast CPU (`build_bm25`), so this iterates quickly *without*
  the GPU rebuild — high EV per unit time.
- **Accept if:** improves candidate recall (0.3 table) and/or fused NDCG under CV.

**Phase 2 exit criterion:** a rebuilt, committed artifact set that beats Phase 1 under
CV, with runtime re-verified ≤ 60 s.

---

## 6. Phase 3 — Residual-gap attacks (last resort, only after 1–2)

Reserve for the queries still failing after Phases 1–2 — the **temporal** (`1820s` vs
`1826`) and **synonym-paraphrase** gaps. These are higher-variance; guard with CV.

### L8 — Pseudo-relevance feedback / RM3 in BM25 (numpy only)
Expand the query with top terms from the top-k BM25 docs and re-score. Directly targets
the synonym gap (bridges query→answer vocabulary). **Risk:** PRF can drift; gate per-query
(only expand when top-k agreement is high) and CV-test hard — it can easily overfit 29
queries.

### L9 — Temporal normalization in tokenization
`1820s` → expose tokens `{1820, 182}` so a query decade can match a page year prefix
(`1826` shares prefix `182`). Note **plain decade expansion (`1820`→1820–1829) was
already refuted** (no effect) — this is the *prefix/normalization* variant, not the
expansion variant. Test only if 0.3's table shows temporal queries still out-of-pool.

### L10 — Query rewriting (heaviest, lowest priority)
Only if a specific named query needs it; bounded, hand-checked. Do not generalize from
one query.

---

## 7. Phase 4 — Robustness lock-in (do before declaring success)

1. **k-fold CV on the final config** (`diagnose_cv.py`): the accepted parameters must
   win on held-out folds, not just the full 29. If a gain is fold-fragile, **revert it**
   — a hidden-set regression costs more than the marginal public gain.
2. **Long-answer-page stress test:** β penalizes long pages; the hidden set may include
   long answer pages. Confirm the chosen β doesn't catastrophically demote a
   hypothetical long answer (sanity check by inspecting the β response curve, not just
   the peak).
3. **Runtime budget check** at hidden-set scale: time `run()` on ~50 queries with the
   widened pool / extra vectors. Must stay ≤ 60 s with margin (target ≤ 45 s).
4. **Determinism / dedup:** confirm `run()` returns deduped top-10 int lists, no NaN
   ordering surprises from the new fusion (ties, empty BM25, empty dense pool fallbacks
   all exercised).
5. **Re-commit artifacts via LFS**, fresh-clone smoke test, update `README.md`,
   `DIAGNOSIS.md`, `CLAUDE.md` with the new default and the new dead-ends.

---

## 8. Expected-value summary

| Lever | Rebuild? | Est. gain | Confidence | Attacks |
|-------|----------|-----------|------------|---------|
| L1 widen `cand_n` | no | +0.02–0.03 | med | recall clipping |
| **L2 dense-union recall** | no | **+0.02–0.05** | **med-high** | out-of-pool synonym gap (raises ceiling) |
| L3 fusion refinement | no | +0.00–0.02 | low-med | ranking precision |
| L4 re-tune (dense_w,β) | no | +0.01 | med | operating point |
| L5 β-ensemble | no | +0.00–0.01 | low | stability |
| L6 title-aware dense | yes (GPU) | +0.01–0.03 | med | entity queries |
| L7 field-weighted BM25 + k1/b | yes (CPU, cheap) | +0.01–0.03 | med | pool recall + anchor |
| L8 PRF/RM3 | no/CPU | +0.00–0.03 | low (high var) | synonym gap |
| L9 temporal prefix tokens | yes (CPU) | +0.00–0.02 | low | temporal gap |

Reaching 0.50 most plausibly = **L1 + L2 + L4** (no rebuild, ~0.46–0.48) **+ L6 and/or
L7** (rebuild, push to 0.50+). L2 is the linchpin because it raises the recall ceiling;
the rest only re-rank within the pool.

---

## 9. Risk register & contingencies

| Risk | Likelihood | Mitigation / contingency |
|------|-----------|--------------------------|
| **Artifacts not committed → 0 functional** | certain if ignored | Phase 0.2 LFS setup is BLOCKING; fresh-clone verify before submit. |
| Overfitting 29 queries | high | Every accept gated by k-fold CV (Phase 4); revert fold-fragile gains. |
| GPU rebuild unavailable / slow | med | Front-load all no-rebuild levers (Phase 1); batch Phase-2 rebuilds into one user run with a written diff; BM25-only changes need no GPU. |
| Runtime blows 60 s (wider pool, extra vectors) | med | Time every change at ~50-query scale; cap `cand_n`/`M`; matmul stays vectorized; target ≤45 s. |
| LFS artifact too large / push fails | med | Check `index_vectors.npy` size early; if title-vector doubling is too big, store title vectors in a separate file loaded only when L6 is enabled. |
| 0.50 unreachable with allowed packages | possible | Stop-loss (§10): bank the best CV-validated config; partial gains still help the relative-ranking bonus and 25% repo-quality score. |
| A lever recovers public queries but the hidden set differs in shape | med | CV + the long-answer stress test; avoid query-specific hacks (L10) that don't generalize. |
| Re-introducing a refuted idea by accident | low | §1 dead-end list is the gate; any chunk/sentence/RRF idea is auto-rejected. |

---

## 10. Decision gates & stop-loss

- **After Phase 1:** if CV-validated NDCG ≥ 0.47, proceed to Phase 2 rebuilds. If
  < 0.45, return to error analysis (0.3) — the failing queries are mis-diagnosed.
- **Each Phase-2 rebuild:** only requested after a `diagnose_hybrid.py` preview shows
  ≥ +0.01. No speculative rebuilds.
- **Stop-loss:** if after L1–L7 the CV-validated mean is < 0.50, **lock the best
  CV-validated config as the submission** (do not ship an overfit 0.50-on-public
  config that CV says is fragile). Document the ceiling honestly in `DIAGNOSIS.md`.
  A robust 0.48 beats a fragile, un-reproducible 0.50.

---

## 11. Concrete first actions (in order)

1. **(User)** rebuild index → confirm `eval_public.py` = 0.4338.
2. Set up LFS, commit artifacts, fresh-clone verify (Phase 0.2).
3. Write `diagnose_errors.py` → per-query failure table → `DIAGNOSIS.md` (0.3).
4. Write `diagnose_cv.py` (0.4).
5. L1 `cand_n` sweep (diagnostic → prod env var).
6. L2 dense-union recall on the out-of-pool queries from step 3.
7. L4 re-tune `(dense_w, β)`; CV-validate the Phase-1 stack.
8. Preview L7 (cheap BM25 rebuild) and L6 (title vector) in the diagnostic; if they
   clear +0.01, request the user rebuild.
9. Phase 4 robustness lock-in; submit.
