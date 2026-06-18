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

| # | Speaker | Budget | Stage / beat | Visual |
|---|---------|--------|--------------|--------|
| 1 | **A** | 0:00–0:14 | Task + headline 0.4338 | glowing hero number |
| 2 | **B** | 0:14–0:30 | Why it's hard (data shape) | length-contrast bars |
| 3 | **A** | 0:30–0:50 | **Diagnosis:** ranking, not recall | recall@depth line (top-10 marked) |
| 4 | **B** | 0:50–1:12 | `chunk` · `embed` · `index` | stage cards + lead-vs-body bars |
| 5 | **A** | 1:12–1:34 | `retrieve` — the `zfuse` recipe | 3-step flow diagram |
| 6 | **B** | 1:34–1:58 | Why each piece (build-up) | progression bar chart |
| 7 | **A** | 1:58–2:18 | Tuning the length prior | two-line β sweep |
| 8 | **B** | 2:18–2:34 | What failed (honest) | refuted/ablation bars |
| 9 | **A** | 2:34–2:48 | The ceiling | (dense_w × β) grid heatmap |
| 10 | **B** | 2:48–3:00 | Result + wrap | hero number + stat strip |

Each speaker owns 5 slides → satisfies the "both participate" 20%. The four required stages
are all explicitly on-screen: `chunk`/`embed`/`index` on slide 4, `retrieve` on slide 5,
each backed by a results chart (slides 3–9).

---

## Full narration script

Replace `[Speaker A]` / `[Speaker B]` with the real names on slide 1. Lines are written to
the per-slide budget; the same text is embedded in each slide's reveal speaker notes (press
`S`). Total ≈ 428 words ≈ **2:51** at a calm 150 wpm — ~9 s of buffer under the 3:00 cap.

### Slide 1 — Title (A, 0:00–0:14)
> **[A]:** Hi — we're [Speaker A] and [Speaker B]. Our task: given a query, return the most
> relevant pages from twenty-seven thousand synthetic Wikipedia entries, scored by NDCG at
> ten. Our final system reaches **0.4338**.

### Slide 2 — Why it's hard (B, 0:14–0:30)
> **[B]:** What makes it hard? The corpus is adversarial — fictional entities, dates shifted
> by a century. The answer pages are short, the distractors are long, and queries paraphrase
> the answer or refer to a decade instead of a concrete year.

### Slide 3 — Diagnosis (A, 0:30–0:50)
> **[A]:** First we diagnosed the problem. With plain BM25 the right page is retrievable — a
> hundred percent of the time by rank five hundred. But only sixty-six percent land in the
> top ten. So recall is **not** the bottleneck — ranking precision is. That insight drove
> every later decision.

### Slide 4 — chunk · embed · index (B, 0:50–1:12)
> **[B]:** Three offline stages. **Chunk:** we embed only the lead chunk — title plus
> content; adding body chunks measured worse, because long pages win a max-pooling lottery.
> **Embed:** the fixed MiniLM model, 384-dimensional, L2-normalized. **Index:** a
> hundred-forty-eight-thousand-vector store plus a BM25 index over the corpus.

### Slide 5 — the zfuse recipe (A, 1:12–1:34)
> **[A]:** Retrieval — our "zfuse" recipe, three steps. One: BM25 generates three hundred
> candidates per query. Two: we score each by dense cosine minus a length prior — beta times
> log word-count. Three: we z-score-normalize dense and BM25 and fuse them, dense weight
> point-eight, then take the top ten.

### Slide 6 — build-up (B, 1:34–1:58)
> **[B]:** Why all three pieces? This chart builds them up. The old default scored 0.25.
> Pure BM25, 0.32. Restricting dense scoring to the BM25 pool, 0.34. Adding z-score fusion,
> 0.39. And finally the length prior brings us to **0.4338** — a seventy-two percent relative
> gain, same artifacts, retrieval logic only.

### Slide 7 — tuning the prior (A, 1:58–2:18)
> **[A]:** The length prior is delicate. On its own it peaks at beta 0.05 and then collapses
> — it buries pages with rare exact matches. But fused with BM25, BM25 re-anchors those
> matches, so the prior tolerates beta 0.15. The two signals are complementary, not redundant.

### Slide 8 — what failed (B, 2:18–2:34)
> **[B]:** We were disciplined about dead ends. Body chunks, a title-blend vector,
> sentence-level matching, and injecting global dense hits — every one measured at or below
> baseline; the dense union was catastrophic. Widening the candidate pool only hurt too.

### Slide 9 — the ceiling (A, 2:34–2:48)
> **[A]:** Is 0.4338 the ceiling? Ninety-three of a hundred relevant pages are already in the
> pool, and our operating point is the maximum of the whole parameter grid. Going higher
> needs a cross-encoder — which the fixed-model rule forbids.

### Slide 10 — wrap (B, 2:48–3:00)
> **[B]:** So: diagnosed precisely, justified with experiments, and proven at its ceiling —
> 0.4338 on a frozen encoder, in about twenty-six seconds. Code and write-up are in the
> README. Thanks for watching.

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
