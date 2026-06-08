"""Unit tests for chunk.py — no ML model required."""
from __future__ import annotations

import pytest

from chunk import CHUNK_WORDS, MAX_EXTRA_CHUNKS, Chunk, chunk_corpus, chunk_entry
from utils import entry_text


def _record(page_id: int, content: str, title: str = "Test Title") -> dict:
    return {"page_id": page_id, "title": title, "content": content}


def _long_content(n_words: int) -> str:
    return " ".join(f"word{i}" for i in range(n_words))


class TestChunkEntry:
    def test_single_chunk_short_entry(self):
        record = _record(1, "short content here")
        chunks = chunk_entry(record)
        assert len(chunks) == 1
        assert chunks[0].chunk_id == 0
        assert chunks[0].page_id == 1
        assert "short content here" in chunks[0].text

    def test_single_chunk_exactly_at_limit(self):
        record = _record(2, _long_content(CHUNK_WORDS))
        chunks = chunk_entry(record)
        assert len(chunks) == 1

    def test_chunk_0_is_entry_text(self):
        """chunk_id=0 must equal entry_text(record) for all page lengths."""
        for n in (5, CHUNK_WORDS, CHUNK_WORDS + 1, CHUNK_WORDS * 3):
            record = _record(10, _long_content(n))
            chunks = chunk_entry(record)
            assert chunks[0].text == entry_text(record), f"failed for n={n}"

    def test_multi_chunk_long_entry(self):
        record = _record(3, _long_content(CHUNK_WORDS + 1))
        chunks = chunk_entry(record)
        assert len(chunks) == 2
        for i, c in enumerate(chunks):
            assert c.chunk_id == i
            assert c.page_id == 3
            assert c.text.strip()

    def test_multi_chunk_consecutive_ids(self):
        record = _record(4, _long_content(CHUNK_WORDS * 3))
        chunks = chunk_entry(record)
        ids = [c.chunk_id for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_extra_chunks_no_overlap(self):
        """Extra chunks (id ≥ 1) must cover non-overlapping content windows."""
        record = _record(5, _long_content(CHUNK_WORDS * 3))
        chunks = chunk_entry(record)
        assert len(chunks) >= 3
        # chunk_id=1 covers words[CHUNK_WORDS:2*CHUNK_WORDS]
        # chunk_id=2 covers words[2*CHUNK_WORDS:3*CHUNK_WORDS]
        # No shared content words between chunks 1 and 2
        words1 = set(chunks[1].text.split()[2:])  # skip "Title\n\n"
        words2 = set(chunks[2].text.split()[2:])
        assert words1.isdisjoint(words2)

    def test_max_extra_chunks_capped(self):
        """Very long content must not exceed MAX_EXTRA_CHUNKS + 1 total chunks."""
        record = _record(6, _long_content(CHUNK_WORDS * (MAX_EXTRA_CHUNKS + 10)))
        chunks = chunk_entry(record)
        assert len(chunks) == MAX_EXTRA_CHUNKS + 1

    def test_empty_content(self):
        record = _record(7, "", title="Only Title")
        chunks = chunk_entry(record)
        assert len(chunks) == 1
        assert "Only Title" in chunks[0].text

    def test_no_title(self):
        record = {"page_id": 8, "title": "", "content": "some content words here"}
        chunks = chunk_entry(record)
        assert len(chunks) == 1
        assert chunks[0].page_id == 8

    def test_page_id_coerced_to_int(self):
        record = {"page_id": "99", "title": "T", "content": "c"}
        chunks = chunk_entry(record)
        assert isinstance(chunks[0].page_id, int)
        assert chunks[0].page_id == 99


class TestChunkCorpus:
    def test_returns_at_least_one_chunk_per_record(self, synthetic_records):
        chunks = chunk_corpus(synthetic_records)
        assert len(chunks) >= len(synthetic_records)

    def test_all_page_ids_present(self, synthetic_records):
        chunks = chunk_corpus(synthetic_records)
        chunk_page_ids = {c.page_id for c in chunks}
        expected_ids = {int(r["page_id"]) for r in synthetic_records}
        assert expected_ids == chunk_page_ids

    def test_chunk_dataclass_fields(self, synthetic_records):
        chunks = chunk_corpus(synthetic_records)
        for c in chunks:
            assert isinstance(c, Chunk)
            assert isinstance(c.page_id, int)
            assert isinstance(c.chunk_id, int)
            assert isinstance(c.text, str)
            assert c.text.strip()
