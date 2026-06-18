# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

This is a university course project (DataLab, Technion). It covers **Section B only**: an end-to-end retrieval pipeline over a corpus of Wikipedia-style JSON entries. The goal is to retrieve relevant `page_id` values for a batch of queries, scored by mean NDCG@10.

All code lives at the **repo root** (the repo was restructured so Section B is the root; Section A is not part of this work and has been removed).

## Current state (read this first)

- **Default retrieval mode is `zfuse`** (in `retrieve.py`): BM25 candidate generation →
  lead-chunk dense score with a `−β·log(word_count)` length prior → z-score weighted
  fusion of dense + BM25. Params: `ZFUSE_DENSE_W=0.8`, `ZFUSE_BETA=0.15`, `ZFUSE_CAND_N=300`.
- **Result: mean NDCG@10 = 0.4338** on the public queries (was 0.2527 under the old
  `length_prior` + gated-RRF default — same artifacts, retrieval logic only).
- **Full rationale, results, and refuted dead-ends live in `README.md` and
  `docs/DIAGNOSIS.md`.** Read them before changing retrieval — several intuitive
  ideas (body chunking, sentence-granularity matching, decade expansion, gated-RRF
  fusion) are already measured *worse* and must not be re-tried.
- **0.4338 is confirmed as the practical ceiling for this fixed model** — every
  toggleable lever and the no-rebuild headroom (wider candidate pool, global-dense
  union, (dense_w, β) re-tune) were measured ≤ baseline. See `docs/FEASIBILITY.md` and
  `experiments/PROGRESS.md`; do not re-open these.
- Diagnostic scripts all live in `experiments/`: `diagnose_{retrieval,hybrid,rerank}.py`
  and `diagnose_{errors,sweep}.py` reproduce the numbers without an index rebuild.

## Commands

All commands run from the **repo root**. Use the `DataLab-ProjectA-SectionB` conda environment locally to match the server.

```bash
# Create the environment (once)
conda create -n DataLab-ProjectA-SectionB python=3.10.11
conda activate DataLab-ProjectA-SectionB

# Install dependencies
pip install -r requirements.txt

# Build the index offline (run once locally; artifacts/ must be committed to the repo)
python scripts/build_index.py

# Self-evaluate on the public queries
python scripts/eval_public.py
```

Always activate the environment before running any project code:
```bash
conda activate DataLab-ProjectA-SectionB
```

> **IMPORTANT (Claude Code):** Never run `scripts/build_index.py` autonomously. Index builds are slow GPU jobs — the user runs them manually. Only suggest the command; do not execute it.

## Pipeline Architecture

The retrieval pipeline has four stages executed in sequence:

**Offline (untimed — your machine):**
```
scripts/build_index.py → main.build_offline_index()
  → index.build_index()
    → chunk.chunk_corpus()   # splits pages into retrieval units
    → embed.embed_texts()    # encodes chunks with MiniLM
    → saves to artifacts/    # index_vectors.npy + index_meta.json
```

**Runtime (timed — grading, ≤60s on GPU):**
```
autograder calls main.run(queries)
  → retrieve.search_batch()
    → index.load_index()     # loads prebuilt artifacts/
    → embed.embed_queries()  # encodes queries
    → ranked page_id lists (deduped, top-10 per query)
```

### File roles

| File | Role | Editable |
|------|------|----------|
| `main.py` | Entry point: `run(queries)` called by autograder | Yes |
| `chunk.py` | `Chunk` dataclass; `chunk_entry()` / `chunk_corpus()` | Yes |
| `embed.py` | Lazy-loads MiniLM; `embed_texts()` / `embed_queries()` | Yes |
| `index.py` | `build_index()` (offline) and `load_index()` (runtime) | Yes |
| `retrieve.py` | `search_batch()` — `zfuse` default + legacy modes (env-gated) | Yes |
| `bm25.py` | BM25 build (`build_bm25`) and query scoring (`bm25_score_query`) | Yes |
| `utils.py` | Shared paths, constants (`ARTIFACTS_DIR`, `K_EVAL=10`) | Yes |
| `eval.py` | NDCG@10 utilities | **READ-ONLY** |
| `scripts/build_index.py` | Runs `build_offline_index()` | **READ-ONLY** |
| `scripts/eval_public.py` | Evaluates on public queries | **READ-ONLY** |

## Key Constraints

- **Embedding model is fixed**: must use `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalized)
- **Allowed packages only**: `numpy`, `sentence-transformers`, `faiss-cpu` (or `faiss`) — no additional imports
- **Artifacts must be committed**: the autograder never rebuilds the index; it only calls `run()` and loads from `artifacts/`. Use Git LFS for large files
- **Grading call**: `run(queries: list[str]) -> list[list[int]]` — return one ranked list of `page_id` (int) per query, most relevant first; only top 10 are scored
- **60-second wall-clock limit** for the full `run()` call at grading time (GPU available)

## Data

- Corpus: `data/Wikipedia Entries/*.json` — each file has `page_id`, `title`, `content`
- Public queries: `data/public_queries.json` — **29 queries** with `relevant_page_ids`
  (binary relevance, a query may match multiple pages). An earlier 50-query set was
  corrupted; the TA corrected it — all current numbers are on the 29-query set.
- `utils.entry_text(record)` concatenates title + content as `"{title}\n\n{content}"`

## Multi-chunk Pipelines

`index.py` stores one vector row per chunk (with `chunk_id` in `index_meta.json`).
Chunk 0 of each page is `entry_text` (title + content); body-window chunks (id > 0) also
exist in the index but the **`zfuse` default uses only the lead chunk** — body chunks
were measured to hurt (lottery-ticket false positives; see docs/DIAGNOSIS.md). The legacy
modes' dedup loop (`seen` set / `np.maximum.at`) aggregates chunks to page level.

## Scoring

Section B grade: 50% functional (NDCG@10 on hidden queries) + 25% GitHub repo quality + 25% video presentation. Relative ranking bonus applies for top-3 functional scores. The `artifacts/` directory missing from the repo yields 0 for the functional component.
