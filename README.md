# Section B — Dense–Sparse Retrieval Pipeline

A retrieval system over a corpus of ~27 K Wikipedia-style JSON pages. For each
natural-language query it returns a ranked list of relevant `page_id`s, evaluated by
**mean NDCG@10**.

> ### 🎥 [Watch the video presentation](https://technionmail-my.sharepoint.com/:v:/g/personal/eyal_kotlik_campus_technion_ac_il/IQCIQtM0yl11SrhNct5uDW2uAU_7Gj79PuKz74tQdI-6_Zo?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=TL4JFd)

**Result: mean NDCG@10 = 0.4338** on the 29 public queries, with `run()` completing in
~28 s against the 60 s budget. Authors: Eyal Kotlik & Anastasia Ravits.

---

## Task and constraints

| Requirement | This submission |
|---|---|
| Grading call | `run(queries: list[str]) -> list[list[int]]` in `main.py`, top-10 `page_id`s per query |
| Metric | mean NDCG@10 over the hidden queries |
| Embedding model (fixed) | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalized) |
| Allowed packages | `numpy`, `sentence-transformers`, `faiss-cpu` only |
| Runtime budget | `run()` ≤ 60 s on the grading GPU |
| Artifacts | prebuilt index committed to the repo; the autograder loads it without rebuilding |

## Installation

Requires **Python 3.10** and **Git LFS** — the prebuilt `artifacts/` (index vectors and
the BM25 index) are stored via LFS and must be materialized as real files, since the
autograder loads them directly.

```bash
# 1. Fetch the repository and its LFS artifacts
git lfs install
git clone https://github.com/EyalKotlik/DataLab-ProjectA.git
cd DataLab-ProjectA
git lfs pull                                  # materializes artifacts/*.npy and *.json.gz

# 2. Create the environment and install dependencies
conda create -n DataLab-ProjectA-SectionB python=3.10.11
conda activate DataLab-ProjectA-SectionB
pip install -r requirements.txt

# 3. Self-evaluate on the public queries (loads artifacts/, no rebuild)
python scripts/eval_public.py
```

`requirements.txt` pins a CUDA build of PyTorch to match the grading server; on a
CPU-only machine, install the CPU build of `torch` instead (the pipeline runs on CPU,
only slower). Rebuilding the index is an offline GPU job, needed only if chunking or
embedding changes:

```bash
python scripts/build_index.py    # writes artifacts/index_vectors.npy, index_meta.json, bm25.json.gz
```

## Method

The pipeline separates an untimed offline build from the timed runtime call.

**Offline build** (`scripts/build_index.py`): chunk each page → embed with MiniLM → save
`artifacts/index_vectors.npy` + `index_meta.json`, and build a BM25 index →
`artifacts/bm25.json.gz`. Each page contributes a lead chunk (`entry_text` =
title + content, truncated at 256 tokens), plus title-only and body-window chunks that
support optional levers. Retrieval scores on the lead chunk.

**Runtime** (`retrieve.search_batch`), per query:

1. **Candidate generation.** BM25 over all query tokens selects the top
   `ZFUSE_CAND_N = 300` pages; the union across the batch forms the candidate pool. This
   removes the mass of global distractors that pure dense ranking surfaces.
2. **Dense scoring with a length prior.** Each candidate is scored by its lead-chunk
   cosine similarity minus a length penalty:
   `dense_raw = cos − ZFUSE_BETA · log(content_word_count)`, with `ZFUSE_BETA = 0.15`.
   Relevant pages are short (~50–280 words) while distractors are long (median ~1218), so
   demoting long pages is the single largest contributor.
3. **z-score fusion.** Dense and BM25 scores are standardized over the pool and blended:
   `score = ZFUSE_DENSE_W · dense_z + (1 − ZFUSE_DENSE_W) · bm25_z`, with
   `ZFUSE_DENSE_W = 0.8`. BM25 in the fusion re-anchors exact matches (rare entity names,
   numbers such as `1,456,779`) that a strong length prior would otherwise bury.
4. Return the top-10 `page_id`s by fused score.

The three signals are complementary: candidate restriction bounds the problem, the length
prior demotes long distractors, and the BM25 fusion term preserves exact matches. The
prior alone peaks at β≈0.05 and then collapses; it tolerates β=0.15 only because BM25
holds up the exact-match signal.

## Results (29 public queries)

| Approach | NDCG@10 |
|---|---|
| Dense baseline (single vector, max-pool) | 0.224\* |
| Pure BM25 | 0.319 |
| Dense over BM25 candidate pool (no prior, no fusion) | 0.343 |
| z-score fusion, no length prior (dense_w=0.95) | 0.391 |
| **`zfuse` — dense_w=0.8, β=0.15** | **0.4338** |

\* from the earlier 50-query set; all other rows are the corrected 29-query set.

The (dense_w × β) grid attains its global maximum at the operating point (0.8, 0.15).
Error analysis shows 93/100 relevant pages already lie in the candidate pool, so the
remaining failures are predominantly in-pool ranking errors. Closing that gap would
require a stronger ranker (e.g. a cross-encoder), which the fixed-model and
allowed-packages constraints preclude.

## Design decisions

Each component is justified by ablation; the alternatives below were measured and
performed at or below the baseline. Full numbers are in [docs/DIAGNOSIS.md](docs/DIAGNOSIS.md),
[docs/FEASIBILITY.md](docs/FEASIBILITY.md), and [experiments/PROGRESS.md](experiments/PROGRESS.md).

- **Lead chunk over body chunking.** Overlapping body windows (0.10–0.12) inflate the
  index and flood the top-10 with false positives from long pages; answers live in the
  lead, which the lead chunk already captures.
- **Document-level over sentence-granularity.** Sentence matching (0.268) underperforms
  full-doc dense (0.343); queries are holistic paraphrases of short pages.
- **z-score weighted fusion over gated RRF.** The IDF gate fires on almost nothing and
  RRF flattens the exact-match advantage.
- **Candidate pool of 300.** Widening it (500–1500) is monotonically worse; recall is not
  the limiter. Injecting global dense neighbors is catastrophic (0.19–0.33).
- **Decade expansion** (`1820s` → 1820–1829 in BM25) had no measurable effect.

## Repository layout

| File | Role |
|------|------|
| `main.py` | `run(queries)` entry point called by the autograder |
| `retrieve.py` | `search_batch()` — the `zfuse` pipeline (plus optional levers behind env vars) |
| `bm25.py` | BM25 build (`build_bm25`) and query scoring (`bm25_score_query`) |
| `chunk.py` | `Chunk` dataclass and chunking (chunk 0 = `entry_text`) |
| `embed.py` | Lazy-loads MiniLM; `embed_texts()` / `embed_queries()` |
| `index.py` | `build_index()` (offline) / `load_index()` (runtime) |
| `utils.py` | Shared paths, constants, `entry_text()` |
| `eval.py`, `scripts/build_index.py`, `scripts/eval_public.py` | Graded harness (read-only) |
| `experiments/diagnose_*.py` | Analysis scripts (see below) |

### Configuration

Core parameters: `ZFUSE_DENSE_W` (0.8), `ZFUSE_BETA` (0.15), `ZFUSE_CAND_N` (300).
Additional optional levers (`ZFUSE_CHUNK_AGG`, `ZFUSE_TITLE_W`, `BM25_TITLE_BOOST`,
`BM25_TEMPORAL`, and the legacy RRF path) default OFF and reproduce the 0.4338 result
byte-for-byte; their measured effects are tabulated in [docs/DIAGNOSIS.md](docs/DIAGNOSIS.md).

## Reproducing the analysis

```bash
python experiments/diagnose_retrieval.py   # pure-BM25 score + recall@depth (numpy/stdlib only)
python experiments/diagnose_hybrid.py      # dense/BM25 fusion + length-prior sweeps
python experiments/diagnose_rerank.py      # sentence-granularity comparison
python experiments/diagnose_errors.py      # per-query failure analysis over committed artifacts
python experiments/diagnose_sweep.py       # cand_n / dense-union / (dense_w, β) sweeps
```

These scripts run over the committed artifacts (reproducing the production 0.4338 exactly)
without rebuilding the index; the dense sweeps embed only the BM25 candidate pool and
finish in about a minute.
