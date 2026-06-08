"""Integration tests for retrieve.py — end-to-end pipeline (slow)."""
from __future__ import annotations

import pytest

from index import build_index
from retrieve import search_batch
from utils import K_EVAL


@pytest.fixture(scope="module")
def built_index(tmp_path_factory, synthetic_records):
    """Build a tiny index once per module — shared across all retrieve tests."""
    import json
    entries_dir = tmp_path_factory.mktemp("entries")
    artifacts_dir = tmp_path_factory.mktemp("artifacts")
    for record in synthetic_records:
        pid = record["page_id"]
        (entries_dir / f"{pid}.json").write_text(json.dumps(record), encoding="utf-8")
    build_index(entries_dir=entries_dir, artifacts_dir=artifacts_dir)
    return artifacts_dir


@pytest.mark.slow
class TestSearchBatch:
    def test_returns_one_list_per_query(self, built_index):
        queries = ["What is this about?", "another query"]
        results = search_batch(queries, artifacts_dir=built_index)
        assert len(results) == len(queries)

    def test_each_list_is_list_of_ints(self, built_index):
        results = search_batch(["test"], artifacts_dir=built_index)
        assert isinstance(results[0], list)
        for pid in results[0]:
            assert isinstance(pid, int)

    def test_top_k_respected(self, built_index):
        results = search_batch(["test"] * 3, top_k=3, artifacts_dir=built_index)
        for r in results:
            assert len(r) <= 3

    def test_no_duplicates_within_result(self, built_index):
        results = search_batch(["query about long pages"], artifacts_dir=built_index)
        for r in results:
            assert len(r) == len(set(r))

    def test_empty_query_list(self, built_index):
        results = search_batch([], artifacts_dir=built_index)
        assert results == []

    def test_relevant_page_appears_in_results(self, built_index, synthetic_records):
        """A query containing a page's exact title should retrieve that page."""
        long_page = next(r for r in synthetic_records if r["title"] == "Long Page Alpha")
        query = long_page["title"]
        results = search_batch([query], top_k=K_EVAL, artifacts_dir=built_index)
        assert int(long_page["page_id"]) in results[0]
