"""Integration tests for index.py — loads MiniLM model (slow)."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

import numpy as np

from index import INDEX_META_NAME, INDEX_VECTORS_NAME, build_index, load_index


@pytest.mark.slow
class TestBuildIndex:
    def test_creates_artifact_files(self, tmp_entries_dir, tmp_artifacts_dir):
        build_index(entries_dir=tmp_entries_dir, artifacts_dir=tmp_artifacts_dir)
        assert (tmp_artifacts_dir / INDEX_VECTORS_NAME).exists()
        assert (tmp_artifacts_dir / INDEX_META_NAME).exists()

    def test_vector_dim(self, tmp_entries_dir, tmp_artifacts_dir):
        vectors, _ = build_index(entries_dir=tmp_entries_dir, artifacts_dir=tmp_artifacts_dir)
        assert vectors.shape[1] == 384

    def test_page_ids_length_matches_vectors(self, tmp_entries_dir, tmp_artifacts_dir):
        vectors, page_ids = build_index(entries_dir=tmp_entries_dir, artifacts_dir=tmp_artifacts_dir)
        assert vectors.shape[0] == len(page_ids)

    def test_page_ids_are_ints(self, tmp_entries_dir, tmp_artifacts_dir):
        _, page_ids = build_index(entries_dir=tmp_entries_dir, artifacts_dir=tmp_artifacts_dir)
        for pid in page_ids:
            assert isinstance(pid, int)

    def test_all_source_page_ids_represented(self, tmp_entries_dir, tmp_artifacts_dir, synthetic_records):
        _, page_ids = build_index(entries_dir=tmp_entries_dir, artifacts_dir=tmp_artifacts_dir)
        built_ids = set(page_ids)
        expected_ids = {int(r["page_id"]) for r in synthetic_records}
        assert expected_ids == built_ids

    def test_vectors_are_l2_normalized(self, tmp_entries_dir, tmp_artifacts_dir):
        vectors, _ = build_index(entries_dir=tmp_entries_dir, artifacts_dir=tmp_artifacts_dir)
        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


@pytest.mark.slow
class TestLoadIndex:
    def test_roundtrip_vectors(self, tmp_entries_dir, tmp_artifacts_dir):
        built_vectors, built_ids = build_index(
            entries_dir=tmp_entries_dir, artifacts_dir=tmp_artifacts_dir
        )
        loaded_vectors, loaded_ids = load_index(artifacts_dir=tmp_artifacts_dir)
        np.testing.assert_allclose(built_vectors, loaded_vectors, atol=1e-6)
        assert built_ids == loaded_ids

    def test_loaded_page_ids_are_ints(self, tmp_entries_dir, tmp_artifacts_dir):
        build_index(entries_dir=tmp_entries_dir, artifacts_dir=tmp_artifacts_dir)
        _, page_ids = load_index(artifacts_dir=tmp_artifacts_dir)
        for pid in page_ids:
            assert isinstance(pid, int)

    def test_meta_contains_expected_keys(self, tmp_entries_dir, tmp_artifacts_dir):
        build_index(entries_dir=tmp_entries_dir, artifacts_dir=tmp_artifacts_dir)
        meta = json.loads((tmp_artifacts_dir / INDEX_META_NAME).read_text())
        for key in ("page_ids", "chunk_ids", "model", "num_vectors"):
            assert key in meta
