# Retrieval Pipeline — Improvements Roadmap

Ranked roughly by expected NDCG@10 uplift vs. implementation cost.

---

## 1. Overlapping Chunk Splitting (Critical)

**Why it matters**
`all-MiniLM-L6-v2` has a hard 256-token context window. Most Wikipedia articles are thousands of tokens long. Everything past token ~256 is silently truncated — so the current "single chunk per page" strategy embeds only the lead paragraph of each article. This is almost certainly the single largest source of missed relevance.

**Implementation sketch**
In `chunk.py`, replace the trivial `chunk_entry()` with a word-level sliding window:

```python
CHUNK_WORDS = 180     # fits comfortably inside 256 tokens
STRIDE_WORDS = 60     # overlap keeps context across boundaries

def chunk_entry(record):
    page_id = int(record["page_id"])
    title   = record.get("title", "")
    words   = entry_text(record).split()
    chunks  = []
    start   = 0
    chunk_id = 0
    while start < len(words):
        window = words[start : start + CHUNK_WORDS]
        # prepend title so every chunk is self-contextualised (see item 2)
        text = f"{title}\n\n" + " ".join(window)
        chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id, text=text))
        chunk_id += 1
        if start + CHUNK_WORDS >= len(words):
            break
        start += STRIDE_WORDS
    return chunks
```

After this change, rebuild the index and commit the new artifacts. `retrieve.py`'s deduplication loop already handles multiple chunks per page.

---

## 2. Title Prefix on Every Chunk (Free Quality Gain)

**Why it matters**
When a chunk covers the middle of an article, there is no signal about the page topic. Prepending the title ensures that every chunk vector is anchored to the page subject. Gains are especially large for queries that name an entity directly (the title) but whose relevant content appears mid-article.

**Implementation sketch**
Already folded into item 1 above (`f"{title}\n\n" + chunk_text`). If you keep single-chunk mode, the change is one line in `chunk_entry()`.

---

## 3. FAISS Flat Index (Speed / Scale Insurance)

**Why it matters**
After chunking, the corpus grows from ~27K vectors to potentially 150K–300K. Brute-force NumPy matmul at 300K × 384 still only takes ~0.1 s per query batch, so this is not a blocker today. But it becomes critical if you add more pages or heavier reranking in the 60-second window.

**Implementation sketch**
In `index.py`:

```python
import faiss

def build_index(...):
    ...
    # after embed_texts, also save a FAISS flat index
    dim = vectors.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)   # inner product on L2-normalised = cosine
    faiss_index.add(vectors)
    faiss.write_index(faiss_index, str(out_dir / "index.faiss"))
```

In `retrieve.py`, replace the matmul loop with:

```python
faiss_index = faiss.read_index(str(artifacts_dir / "index.faiss"))
scores, indices = faiss_index.search(query_vectors, top_k * 5)  # over-fetch for dedup
```

---

## 4. BM25 Hybrid Retrieval (High Ceiling, More Work)

**Why it matters**
Dense retrieval misses exact keyword matches; BM25 catches them. Fusing BM25 and dense scores (reciprocal rank fusion or linear interpolation) consistently beats either alone on Wikipedia-style corpora. The gain is usually 5–15 NDCG points.

**Implementation sketch** (numpy only — no new packages)

```python
# In a new file bm25.py (pure numpy):
# 1. At index-build time: compute IDF for every token, store per-page TF
#    as a sparse dict → serialize to artifacts/bm25.json
# 2. At query time: tokenise query, compute BM25 score vs every page (O(|vocab_hit| * N))
# 3. Fuse with dense score:
#    combined = alpha * dense_score + (1 - alpha) * bm25_score_normalised
#    or use Reciprocal Rank Fusion: 1/(k+rank_dense) + 1/(k+rank_bm25)
```

The BM25 index is just term frequencies; numpy sparse dicts are enough. The main implementation cost is the tokeniser and RRF merge in `retrieve.py`.

---

## 5. Score Aggregation Tuning (Low-Hanging Fruit)

**Why it matters**
`retrieve.py` currently takes the single highest-scoring chunk as the page score. For long articles with multiple relevant sections, averaging the top-2 chunk scores per page or summing all chunks can improve ranking.

**Implementation sketch**
In the dedup loop inside `search_batch()`, instead of stopping at the first chunk hit per page, collect all chunk scores per page_id, then sort by `max`, `mean_of_top2`, or `sum`:

```python
from collections import defaultdict
page_scores = defaultdict(list)
for idx, score in zip(np.argsort(-row), row[np.argsort(-row)]):
    page_scores[page_ids[idx]].append(float(score))

# aggregate: try max, then mean of top-2
ranked_pages = sorted(page_scores, key=lambda p: page_scores[p][0], reverse=True)
```

This is a zero-cost experiment (no rebuild needed) — just swap the aggregation function and re-eval.

---

## Priority Order

| # | Change | Rebuild needed | Expected gain |
|---|--------|---------------|---------------|
| 1+2 | Chunking + title prefix | Yes | Very high |
| 5 | Score aggregation tuning | No | Low–medium |
| 3 | FAISS index | Yes | Speed only |
| 4 | BM25 hybrid | Yes | High, most work |

Start with items 1+2 (one rebuild, likely biggest NDCG jump), run `scripts/eval_public.py` to confirm, then tune aggregation (item 5) for free, then invest in BM25 if time allows.
