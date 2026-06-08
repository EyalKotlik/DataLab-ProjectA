"""Unit tests for chunk.py — no ML model required."""
from __future__ import annotations

import pytest

from chunk import CHUNK_WORDS, STRIDE_WORDS, Chunk, chunk_corpus, chunk_entry


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

    def test_multi_chunk_long_entry(self):
        record = _record(3, _long_content(CHUNK_WORDS + 1))
        chunks = chunk_entry(record)
        assert len(chunks) > 1
        for i, c in enumerate(chunks):
            assert c.chunk_id == i
            assert c.page_id == 3
            assert c.text.strip()

    def test_multi_chunk_consecutive_ids(self):
        record = _record(4, _long_content(300))
        chunks = chunk_entry(record)
        ids = [c.chunk_id for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_multi_chunk_overlap(self):
        """Adjacent chunks must share words due to stride < window."""
        record = _record(5, _long_content(300))
        chunks = chunk_entry(record)
        assert len(chunks) >= 2
        words0 = chunks[0].text.split()
        words1 = chunks[1].text.split()
        # After the title prefix, the overlapping words are the last
        # (CHUNK_WORDS - STRIDE_WORDS) words of chunk 0.
        overlap_expected = CHUNK_WORDS - STRIDE_WORDS
        shared = set(words0) & set(words1)
        assert len(shared) >= overlap_expected

    def test_empty_content(self):
        record = _record(6, "", title="Only Title")
        chunks = chunk_entry(record)
        assert len(chunks) == 1
        assert "Only Title" in chunks[0].text

    def test_no_title(self):
        record = {"page_id": 7, "title": "", "content": "some content words here"}
        chunks = chunk_entry(record)
        assert len(chunks) == 1
        assert chunks[0].page_id == 7

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
