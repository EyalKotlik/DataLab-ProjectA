# Repo cleanup for final submission

> Scratch notes — **not committed**. Delete once acted on.

Repo-quality is 25% of the grade and is reviewed by a human browsing the repo on
GitHub. The functional grade depends **only** on `artifacts/` being present and
`run()` working — nothing below touches that. Every change here is therefore
safe for the functional score.

---

## 1. The `data/` directory — DONE (staged, not committed)

Corpus collapsed from 27,074 tracked files into one LFS object
`data/corpus.tar.gz` (134 MB). `data/Wikipedia Entries/` is now gitignored and
auto-extracted on demand by `utils.ensure_corpus_extracted()` (called from
`iter_entries`), so `build_index.py` (read-only) and the diagnostics work
unchanged. Staged: the archive (LFS), `utils.py`, `.gitignore`, `.gitattributes`,
and the 27,074 deletions. Review then commit.

<details><summary>original analysis</summary>

### 1. The `data/` directory — the headline issue

**State:** `data/Wikipedia Entries/` holds **27,074 raw JSON files (427 MB), all
tracked in git.**

**Who actually reads it:**
- `scripts/build_index.py` (offline index build — run manually, locally).
- `diagnose_*.py` (local diagnostics).
- **NOT** `main.run()` at grading time — runtime loads only `artifacts/`
  (`index_vectors.npy`, `bm25.json.gz`, `index_meta.json`, all committed via LFS).

**Why this is a problem:**
- 27k tiny files is the single biggest repo-quality smell — the GitHub file
  browser is unusable, and it dwarfs the ~12 source files that are the actual work.
- The corpus is **course-provided and identical for every student and the
  grader** — committing it adds nothing a reviewer needs.
- It is the bulk of the non-LFS git history (packs ~150 MB).

**Recommendation (ranked):**

**A. (Recommended) Replace the 27k files with one LFS-tracked archive.**
Keeps full reproducibility; turns 27,074 tracked files into 1.
```bash
tar czf data/corpus.tar.gz -C data "Wikipedia Entries"   # ~135 MB
git rm -r --cached "data/Wikipedia Entries"
echo 'data/Wikipedia Entries/' >> .gitignore
git lfs track "data/corpus.tar.gz"        # *.gz is already LFS-tracked
git add .gitattributes data/corpus.tar.gz .gitignore
```
Then add a one-liner to `build_index.py` (or document it) that extracts
`data/corpus.tar.gz` into `data/Wikipedia Entries/` if the dir is absent, and
note the rebuild step in the README.

**B. (Lighter) Untrack the corpus, don't ship it at all.**
The grader doesn't need it (artifacts are committed). Document in the README:
"to rebuild the index, place the course-provided corpus under
`data/Wikipedia Entries/`."
```bash
git rm -r --cached "data/Wikipedia Entries"
echo 'data/Wikipedia Entries/' >> .gitignore
```

**Avoid:** LFS-tracking the 27k files individually (still 27k pointer files in
the tree, and 27k LFS objects burns quota), or leaving it as-is.

`data/public_queries.json` (7.6 KB) **stays tracked** — it's small and part of
the verification harness.

</details>

---

## 2. `.git` LFS bloat — keep history, no rewrite (decided)

Decision: **do not rewrite history** (TA wants history preserved to show work).
That's fine — here's why it's not the blocker it looks like, plus the
non-destructive mitigations.

**The key distinction:** the TA wants the *commit history* — messages, code
evolution, experiment progression. That is fully preserved no matter what; it is
unaffected by LFS storage. The 2.0 GB is **38 binary versions of
`index_vectors.npy` / `bm25.json.gz`** — regenerable build outputs, not "work
shown." Removing them is the *only* thing a rewrite would do, and we're not doing it.

**Why a clone is still cheap (~415 MB, not 2 GB):** git-lfs only fetches the LFS
objects for the *checked-out* commit. A grader cloning your default branch
downloads the current artifacts + corpus (~415 MB) — old versions are fetched
lazily only if someone checks out those old commits. So clone speed and
bandwidth are fine; the 2 GB is *storage* only.

**Non-rewrite mitigations:**
- **Stop re-committing artifacts every iteration.** The 38 versions came from
  re-committing `index_vectors.npy` on each experiment. Going forward, re-commit
  artifacts only at real milestones (or not during experiments) to cap growth.
- **`git lfs prune`** reclaims *local* disk (your 2 GB `.git/lfs`) by dropping old
  cached blobs you can re-fetch — does not touch history or the remote. Safe.
- **Confirm the submission remote's LFS quota.** GitHub free = 1 GB storage +
  1 GB bandwidth/month; 2 GB stored may be rejected. If so, without rewriting:
  use a remote with generous LFS (the Technion GitLab, if that's the submission
  target, typically is), or a GitHub LFS data pack. Check *where* you submit first.

Bottom line: history stays intact; just don't add more artifact versions, prune
locally, and verify the remote's quota before the final push.

---

## 3. `logs/` — untrack (build noise)

57 tracked files, including multi-MB build logs (`build_*.log` up to 2.4 MB).
Not something a reviewer should see.
```bash
git rm -r --cached logs
echo 'logs/' >> .gitignore
```
(Keep `experiments/PROGRESS.md` — that's the curated results log, not raw logs.)

---

## 4. `SectionB/` — delete leftover directory

Leftover from the "make Section B the repo root" restructure. 0 tracked files;
contains only `__pycache__/` and `.pytest_cache/`. Remove from the working tree:
```bash
rm -rf SectionB
```

---

## 5. Root tidy-up (optional, polish)

- **Internal planning docs at root:** `PLAN_NDCG_0.5.md`, `DIAGNOSIS.md`,
  `FEASIBILITY.md` are valuable but clutter the root. Consider a `docs/` folder,
  leaving `README.md` (+ `CLAUDE.md`) at the top level.
- **Assignment spec:** `Project A.pdf` / `Project A.md` have spaces in the
  filename and are the given handout. Move under `docs/` or drop — your call on
  whether the grader expects the spec echoed back.
- **Root diagnostic scripts:** `diagnose_retrieval.py`, `diagnose_hybrid.py`,
  `diagnose_rerank.py`, `cuda_diagnostics.py` sit next to the core pipeline.
  Consider moving to `diagnostics/` (or alongside `experiments/`) so the root is
  just the pipeline: `main / chunk / embed / index / retrieve / bm25 / utils / eval`.
  Update the path refs in those scripts if moved.

---

## Suggested order

1. `data/` (item 1) — biggest win.
2. `logs/` + `SectionB/` (items 3, 4) — quick.
3. Decide the LFS/push story (item 2) — do this *before* the final push.
4. Root tidy-up (item 5) — last, optional.
