"""Unit tests for utils.py — no ML model required."""
from __future__ import annotations

import json
import logging

import pytest

from utils import entry_text, iter_entries, normalize_page_id, setup_logging


class TestNormalizePageId:
    def test_int_passthrough(self):
        assert normalize_page_id(42) == 42

    def test_numeric_string(self):
        assert normalize_page_id("42") == 42

    def test_numeric_string_with_whitespace(self):
        assert normalize_page_id("  7  ") == 7

    def test_zero(self):
        assert normalize_page_id(0) == 0

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            normalize_page_id("abc")

    def test_float_raises(self):
        with pytest.raises((ValueError, AttributeError)):
            normalize_page_id(3.14)


class TestEntryText:
    def test_with_title_and_content(self):
        record = {"title": "My Title", "content": "Some content."}
        assert entry_text(record) == "My Title\n\nSome content."

    def test_empty_title_returns_content_only(self):
        record = {"title": "", "content": "Just content."}
        result = entry_text(record)
        assert result == "Just content."
        assert not result.startswith("\n")

    def test_missing_title_key(self):
        record = {"content": "Only content."}
        result = entry_text(record)
        assert "Only content." in result

    def test_missing_content_key(self):
        record = {"title": "Title Only"}
        result = entry_text(record)
        assert "Title Only" in result

    def test_strips_surrounding_whitespace(self):
        record = {"title": "  T  ", "content": "  C  "}
        result = entry_text(record)
        assert not result.startswith(" ")
        assert not result.endswith(" ")


class TestIterEntries:
    def test_yields_all_records(self, tmp_entries_dir, synthetic_records):
        results = list(iter_entries(tmp_entries_dir))
        assert len(results) == len(synthetic_records)

    def test_page_ids_are_ints(self, tmp_entries_dir):
        for record in iter_entries(tmp_entries_dir):
            assert isinstance(record["page_id"], int)

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(iter_entries(tmp_path / "nonexistent"))

    def test_records_have_expected_keys(self, tmp_entries_dir):
        for record in iter_entries(tmp_entries_dir):
            assert "page_id" in record


class TestSetupLogging:
    def teardown_method(self):
        """Reset root logger handlers after each test."""
        root = logging.getLogger()
        root.handlers.clear()

    def test_adds_handler(self):
        setup_logging()
        assert len(logging.getLogger().handlers) >= 1

    def test_idempotent(self):
        setup_logging()
        count_after_first = len(logging.getLogger().handlers)
        setup_logging()
        assert len(logging.getLogger().handlers) == count_after_first

    def test_file_handler_created(self, tmp_path):
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file)
        logging.getLogger().info("test message")
        # Flush handlers
        for h in logging.getLogger().handlers:
            h.flush()
        assert log_file.exists()
        assert "test message" in log_file.read_text()
