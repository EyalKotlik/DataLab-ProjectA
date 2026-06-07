"""Optional preprocessing and chunking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from utils import entry_text

CHUNK_WORDS = 180   # fits inside MiniLM's 256-token limit with title headroom
STRIDE_WORDS = 60   # overlap so context isn't lost at boundaries


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """Split one corpus entry into overlapping word-window chunks."""
    page_id = int(record["page_id"])
    title = record.get("title", "")
    words = entry_text(record).split()
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
    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
