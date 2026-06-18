# Experiment log — executing the FEASIBILITY plan

Tracks the bounded improvement attempt described in [../docs/FEASIBILITY.md](../docs/FEASIBILITY.md).
Baseline to beat: **mean NDCG@10 = 0.4338** (`zfuse`, dense_w=0.8, β=0.15, cand_n=300).
Stop-loss: if no CV-stable gain ≥ +0.01 after the steps below, lock 0.4338 and move to
repo quality + video.

## Setup notes
- Local conda env `DataLab-ProjectA` had only numpy. Installed CPU `torch` +
  `sentence-transformers==3.1.1` + `faiss-cpu` so the 29 public queries can be embedded
  locally (no GPU needed for 29 short queries). The graded `artifacts/` are unchanged —
  they were built on the server and are committed via LFS.
- Query vectors for the 29 public queries are embedded once and cached to
  `experiments/query_vecs.npy`; all sweeps are then pure-numpy over the committed
  `artifacts/` (corpus vectors + BM25), so they reproduce production numbers exactly and
  run in seconds.
- Shared loader/scorer: `experiments/_lab.py` (faithful reimplementation of
  `retrieve._zfuse_batch`, lead mode, parametrized by dense_w/β/cand_n + an L2
  `dense_union_m` knob). Sanity gate: it must reproduce **0.4338** at the defaults before
  any sweep number is trusted.

## Steps (from FEASIBILITY §5)
1. [done] Per-query error analysis (`diagnose_errors.py`) — gate for L2.
2. [done] L2 global-dense union (`diagnose_sweep.py`).
3. [done] L1 `ZFUSE_CAND_N` sweep (`diagnose_sweep.py`).
4. [done] L4 (dense_w, β) grid (`diagnose_sweep.py`).
5. [done] **Stop-loss triggered — no gain to CV-gate. Locking 0.4338.**

## Results

### Step 1 — error analysis (`diagnose_errors.py`) [done]
- Harness reproduces production **0.4338 exactly** → sweeps are trustworthy.
- **Recall is not the bottleneck:** 93/100 relevant pages are in the candidate pool;
  only **7 pages OUT of pool, across 4 queries (17, 18, 22, 27)**. And the OUT pages
  mostly have poor global-dense ranks too (g=159/947/3153/8404 …), so a dense-union
  top-M would have to be huge to even add them — and they'd still rank nowhere.
- **The dominant failure is ranking precision inside the pool** (`p=IN f=—`): e.g. q5,
  q7, q10, q22, q28 have relevant pages in the pool that fusion ranks out of top-10.
- **Smoking gun for the length prior:** q18 (9 relevant, NDCG 0.124) has relevant pages
  at global-dense ranks 8/11/13/16/20/22/26 — most are *in the pool with strong dense
  similarity* but fusion + β=0.15 length prior demotes them out of top-10. β tuned to the
  average is hurting the multi-relevant queries.
- **Verdict:** L2 has little plausible upside (4 queries, deep dense ranks). The real
  lever, if any, is re-tuning (dense_w, β) — L4 — but that is the highest overfitting
  risk on 29 queries, so it *must* clear CV. Proceeding to test L1/L2/L4 empirically
  anyway (cheap), then CV-gate.

### Steps 2-4 — L1 / L2 / L4 sweeps (`diagnose_sweep.py`) [done]
All over the committed artifacts; harness reproduces 0.4338 at the defaults.

**L1 — widen `cand_n` (dense_w=0.8, β=0.15):** monotonically *worse*.
| cand_n | 300 | 500 | 750 | 1000 | 1500 |
|---|---|---|---|---|---|
| NDCG@10 | **0.4338** | 0.4188 | 0.4166 | 0.4154 | 0.4034 |
A bigger pool dilutes the z-score normalization and pulls in distractors. 300 is best.

**L2 — global-dense union top-M (cand_n=300):** *catastrophic*.
| M | 0 | 25 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|
| NDCG@10 | **0.4338** | 0.2396 | 0.1941 | 0.2517 | 0.2942 | 0.3327 |
Adding global-dense pages floods the pool with long-page false positives — the exact
failure the length prior was built to suppress. The 4 out-of-pool queries don't recover
(their OUT pages have deep dense ranks); meanwhile every other query is wrecked.

**L4 — (dense_w, β) grid (cand_n=300):** the current **(0.8, 0.15) = 0.4338 is the global
maximum of the entire grid.** No cell beats it (best neighbors: dw0.7/β0.15=0.4266,
dw0.85/β0.2=0.4313). β=0.15 is a clean local peak.

### Step 5 — decision: STOP. Lock 0.4338.
No config beat baseline on the **full** 29-query set, so there is nothing for CV to gate
(CV only ever lowers a fragile full-set gain; it cannot create one). Every live lever from
FEASIBILITY is now measured ≤ baseline:

| Lever | Result | Verdict |
|---|---|---|
| L1 widen cand_n | ≤ 0.4338, worse as it grows | rejected |
| L2 dense-union | 0.19–0.33, catastrophic | rejected |
| L4 re-tune (dense_w, β) | (0.8,0.15) is the grid max | no gain |

**Conclusion:** the lead-chunk + length-prior + BM25-fusion recipe is at its practical
ceiling for this fixed model. FEASIBILITY's prediction holds. Moving to repo quality +
video (the other 50% of the grade), per the user's instruction.

The 7 out-of-pool / many in-pool-but-demoted relevant pages are real failures, but
recovering them needs a stronger ranker (cross-encoder / better embeddings) — disallowed
by the package constraints. No further NDCG work is worthwhile.
