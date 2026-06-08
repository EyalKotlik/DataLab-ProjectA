# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

This is a university course project (DataLab, Technion). We work exclusively on **Section B**: an end-to-end retrieval pipeline over a corpus of Wikipedia-style JSON entries. The goal is to retrieve relevant `page_id` values for a batch of queries, scored by mean NDCG@10. Submission deadline: 2026-06-12.

All working code lives under `SectionB/`. Section A is not part of this work.

## Commands

All commands run from `SectionB/`. Use the `DataLab-ProjectA-SectionB` conda environment locally to match the server.

```bash
# Create the environment (once)
conda create -n DataLab-ProjectA-SectionB python=3.10.11
conda activate DataLab-ProjectA-SectionB

# Install dependencies
pip install -r requirements.txt

# Build the index offline (run once locally; artifacts/ must be committed to the repo)
python scripts/build_index.py

# Self-evaluate on the 50 public queries
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
| `retrieve.py` | `search_batch()` — brute-force dot product baseline | Yes |
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
- Public queries: `data/public_queries.json` — 50 queries with `relevant_page_ids` (binary relevance, a query may match multiple pages)
- `utils.entry_text(record)` concatenates title + content as `"{title}\n\n{content}"`

## Multi-chunk Pipelines

When `chunk_entry()` returns multiple `Chunk` objects per page, `index.py` stores one vector row per chunk (with `chunk_id` in `index_meta.json`). `retrieve.py`'s deduplication loop (`seen` set) already handles aggregation to `page_id` level — the highest-scoring chunk wins for each page.

## Scoring

Section B grade: 50% functional (NDCG@10 on 50 hidden queries) + 25% GitHub repo quality + 25% video presentation. Relative ranking bonus applies for top-3 functional scores. The `artifacts/` directory missing from the repo yields 0 for the functional component.
