"""Optional preprocessing and chunking."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from utils import entry_text

logger = logging.getLogger(__name__)

CHUNK_WORDS = 150       # safe under 256 tokens with title prefix (150w × 1.4t/w + title ≈ 230t)
MAX_EXTRA_CHUNKS = 5    # at most 5 additional chunks beyond the lead; 6 total per page

_PROGRESS_INTERVAL = 500


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """Split one corpus entry into retrieval chunks.

    chunk_id=0 is always entry_text(record) — identical to the single-chunk
    baseline, letting MiniLM truncate naturally at 256 tokens.  Additional
    non-overlapping chunks (ids 1..MAX_EXTRA_CHUNKS) cover content beyond the
    model's truncation point for pages with more than CHUNK_WORDS content words.
    """
    page_id = int(record["page_id"])
    title = record.get("title", "")
    words = record.get("content", "").split()

    chunks = [Chunk(page_id=page_id, chunk_id=0, text=entry_text(record))]

    # L6: title-only chunk (chunk_id=-1) — baked into the index so that the runtime
    # ZFUSE_TITLE_W env flag can A/B-test a title-embedding signal without a rebuild.
    # Inert by default (ZFUSE_TITLE_W=0.0 means retrieve.py never reads these rows).
    if title.strip():
        chunks.append(Chunk(page_id=page_id, chunk_id=-1, text=title.strip()))

    for i in range(MAX_EXTRA_CHUNKS):
        start = (i + 1) * CHUNK_WORDS
        if start >= len(words):
            break
        window = words[start : start + CHUNK_WORDS]
        text = f"{title}\n\n" + " ".join(window)
        chunks.append(Chunk(page_id=page_id, chunk_id=i + 1, text=text))

    logger.debug(
        "chunk_entry: page_id=%d  words=%d → %d chunks",
        page_id,
        len(words),
        len(chunks),
    )
    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    total = len(records)
    logger.info("chunk_corpus: chunking %d entries …", total)
    chunks: List[Chunk] = []
    for i, record in enumerate(records, start=1):
        chunks.extend(chunk_entry(record))
        if i % _PROGRESS_INTERVAL == 0:
            logger.debug("chunk_corpus: %d/%d entries processed", i, total)
    logger.info("chunk_corpus: produced %d chunks from %d entries", len(chunks), total)
    return chunks
