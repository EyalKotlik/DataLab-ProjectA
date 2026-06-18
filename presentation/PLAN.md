# Video presentation plan — DataLab Section B

> **Local-only.** This directory lives on the `presentation` git branch, which is **never
> pushed or merged** into `section-b`/`main`. It is working material for recording the
> video, not part of the graded repo.

## The brief (from `Project A.md`, §"Video presentation")

| Constraint | Limit |
|---|---|
| Length | **≤ 3:00** (−0.1 pt / second over) |
| Slides | **≤ 10** (−0.1 pt / slide over) |
| Speakers | **both** members must speak meaningfully |
| Content | for **each** stage — `chunk`, `embed`, `index`, `retrieve` — explain the method **and** show the process + empirical results |
| Forbidden | scrolling code, reading a manuscript, pasted code on slides |
| Link | put the video URL in `README.md` |

**Rubric (0–100):** 30 clarity · 20 empirical (shown visually) · 20 concise slides ·
20 both participate · 10 time management. Creativity bonus judged mainly from the video.

**How this deck hits each rubric line:** clear single thesis ("ranking, not recall") carried
through all four stages; every claim has a chart from our own logs; 10 slides exactly;
speakers strictly alternate A/B (5 each); script timed to ~2:51 with buffer.

## How to drive the deck

```bash
bash fetch_vendor.sh          # vendor reveal.js + chart.js (once; CDN fallback if offline)
python build_plots.py         # optional: static PNG backups -> assets/img/
python -m http.server 8000    # serve (charts read data.json via fetch — needs http, not file://)
# open http://localhost:8000/  ·  press S for presenter view (script as speaker notes)
#                                ·  arrow keys to advance  ·  charts animate on slide-enter
```

To export a PDF backup of the slides: open `http://localhost:8000/?print-pdf` and print to PDF.

## Slide map (10 slides · ~180 s · A/B alternate)

| # | Speaker | Budget | Slide title / content | Visual |
|---|---------|--------|-----------------------|--------|
| 1 | **A** | 0:00–0:14 | Task + metric + final result (problem statement) | result line |
| 2 | **B** | 0:14–0:30 | Corpus and query characteristics | length-contrast bars |
| 3 | **A** | 0:30–0:50 | Diagnosis: recall is sufficient, ranking is not | recall@depth line + figures |
| 4 | **B** | 0:50–1:12 | Offline stages: `chunk` · `embed` · `index` | stage cards + lead-vs-body bars |
| 5 | **A** | 1:12–1:34 | Retrieval: three-stage scoring pipeline (`zfuse`) | 3-stage pipeline diagram |
| 6 | **B** | 1:34–1:58 | Component ablation: incremental contribution | progression bar chart |
| 7 | **A** | 1:58–2:18 | Length-prior sensitivity (β) | two-line β sweep |
| 8 | **B** | 2:18–2:34 | Rejected alternatives (≤ baseline) | refuted/ablation bars |
| 9 | **A** | 2:34–2:48 | Parameter grid and ceiling analysis | (dense_w × β) grid heatmap |
| 10 | **B** | 2:48–3:00 | Summary (method · result · validation) | summary list |

Each speaker owns 5 slides → satisfies the "both participate" 20%. The four required stages
are all explicitly on-screen: `chunk`/`embed`/`index` on slide 4, `retrieve` on slide 5,
each backed by a results chart (slides 3–9).

---

## Full narration script

Replace `[Speaker A]` / `[Speaker B]` with the real names on slide 1. Lines are written to
the per-slide budget; the same text is embedded in each slide's reveal speaker notes (press
`S`). Total ≈ 428 words ≈ **2:51** at a calm 150 wpm — ~9 s of buffer under the 3:00 cap.

### Slide 1 — Problem statement (A, 0:00–0:14)
> **[A]:** This is our Section B submission: a retrieval system over a synthetic Wikipedia
> corpus. Given a query, it returns a ranked list of page IDs from twenty-seven thousand
> entries, evaluated by mean NDCG at ten. The final system reaches 0.4338, within the
> sixty-second budget, using the fixed MiniLM encoder.

### Slide 2 — Corpus and query characteristics (B, 0:14–0:30)
> **[B]:** The corpus is deliberately hard: fictionalized entities, years shifted by a
> century. Relevant pages are short — fifty to two-eighty words — while distractors exceed
> twelve hundred. There are twenty-nine queries, about two relevant pages each, with temporal
> and synonym-paraphrase gaps.

### Slide 3 — Diagnosis (A, 0:30–0:50)
> **[A]:** We start with a diagnosis. Pure BM25 scores 0.32, and its recall is high: the
> answer is always in the top five hundred, and in the top fifty ninety percent of the time.
> But only sixty-six percent reach the top ten. So the bottleneck is reranking within a
> candidate set, not retrieving more candidates.

### Slide 4 — Offline stages (B, 0:50–1:12)
> **[B]:** The offline pipeline has three stages. **Chunk:** one lead unit per page — title
> plus content; body windows are indexed but excluded from scoring. **Embed:** the fixed
> MiniLM-L6 model, 384-dimensional, L2-normalized. **Index:** roughly a hundred-fifty-thousand
> vectors plus a BM25 index over four-hundred-eighty-five-thousand terms.

### Slide 5 — Retrieval pipeline (A, 1:12–1:34)
> **[A]:** Retrieval has three stages. First, BM25 shortlists the three hundred best-matching
> pages per query. Second, we re-score that shortlist by semantic similarity minus a penalty
> for page length. Third, we put the dense and BM25 scores on a common scale and blend them,
> weighted toward dense. The two signals are complementary — the penalty demotes long
> distractors, while BM25 re-anchors rare exact matches.

### Slide 6 — Component ablation (B, 1:34–1:58)
> **[B]:** This ablation isolates each component. The previous default scored 0.25; pure
> BM25, 0.32; restricting dense scoring to the BM25 pool, 0.34; adding z-score fusion, 0.39;
> and the length prior gives the final 0.4338 — same artifacts, retrieval logic only.

### Slide 7 — Length-prior sensitivity (A, 1:58–2:18)
> **[A]:** The prior is sensitive to beta. Applied alone, it peaks near beta 0.05 then
> declines, over-penalizing pages with rare exact matches. Fused with BM25, the optimum shifts
> to 0.15, because BM25 preserves the exact-match signal — the two components are complementary.

### Slide 8 — Rejected alternatives (B, 2:18–2:34)
> **[B]:** We also rejected several alternatives, each at or below baseline: body-chunk
> pooling, a title-blended vector, sentence-level matching, and injecting global dense
> neighbors — the worst of the set. Widening the candidate pool also degraded results.

### Slide 9 — Grid and ceiling analysis (A, 2:34–2:48)
> **[A]:** On the ceiling: error analysis shows ninety-three of a hundred relevant pages are
> already in the pool, so failures are mostly in-pool ranking errors. Our operating point is
> the global maximum of the dense-weight–beta grid. Closing the remaining gap would require a
> cross-encoder, which the fixed-model constraint disallows.

### Slide 10 — Summary (B, 2:48–3:00)
> **[B]:** In summary: BM25 candidate generation, lead-chunk dense scoring with a log-length
> prior, and z-score fusion give mean NDCG at ten of 0.4338 — about 0.18 above the previous
> default, within the time budget. Every component is justified by ablation, and the ceiling
> is bounded by the fixed encoder. Full details are in the repository.

---

## Data provenance (every number is from our logs / docs)

All figures are the **corrected 29-query** public set. Sources are recorded per-series in
`data.json` (`_source` fields): `docs/DIAGNOSIS.md`, `experiments/PROGRESS.md`, and the
`logs/eval-public-*-1781593521.log` lever batch. `build_plots.py` and `index.html` read the
same `data.json`, so the static PNGs and the live charts are guaranteed identical.

**Accuracy guardrails (do not overstate on camera):**
- No per-query NDCG table and no cross-validation/CI exist in the repo — don't claim them.
- The body-chunk number is **0.4115** (corrected-set re-test); the older 0.10–0.12 figure
  was on the corrupted 50-query set — do not cite it.
- 0.2527 is the *previous default*; 0.3191 is *pure BM25* — keep them distinct.

## Record / rehearsal checklist

- [ ] Names filled in on slide 1 (replace `[Speaker A]` / `[Speaker B]`).
- [ ] `bash fetch_vendor.sh` succeeded (or confirmed online for CDN fallback).
- [ ] `python build_plots.py` ran; `assets/img/` has 7 PNGs.
- [ ] Deck served over http; all **10** slides render; charts animate; press `S` shows notes.
- [ ] Full read-through timed against a clock → **≤ 3:00** (target 2:51).
- [ ] Both speakers audible; A and B each cover their 5 slides.
- [ ] Recorded at ≥ 1080p; slides legible; no code scrolled / read verbatim.
- [ ] Uploaded; **video URL pasted into the repo `README.md`** (on `section-b`, the pushed branch).
- [ ] (Optional) `?print-pdf` PDF backup exported.
