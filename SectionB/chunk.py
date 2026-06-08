"""Optional preprocessing and chunking."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from utils import entry_text

logger = logging.getLogger(__name__)

CHUNK_WORDS = 180   # fits inside MiniLM's 256-token limit with title headroom
STRIDE_WORDS = 60   # overlap so context isn't lost at boundaries

_PROGRESS_INTERVAL = 500


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """Split one corpus entry into overlapping word-window chunks."""
    page_id = int(record["page_id"])
    title = record.get("title", "")
    words = record.get("content", "").split()
    if len(words) <= CHUNK_WORDS:
        return [Chunk(page_id=page_id, chunk_id=0, text=entry_text(record))]
    chunks: List[Chunk] = []
    start = 0
    chunk_id = 0
    while start < len(words):
        window = words[start : start + CHUNK_WORDS]
        text = f"{title}\n\n" + " ".join(window)
        chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id, text=text))
        chunk_id += 1
        if start + CHUNK_WORDS >= len(words):
            break
        start += STRIDE_WORDS
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
