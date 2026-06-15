"""Shared fixtures and pytest configuration."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure SectionB/ is importable when running pytest from inside tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (loads ML model or builds index)")


def _long_content(n_words: int = 250) -> str:
    """Return a deterministic string of n_words distinct words."""
    return " ".join(f"word{i}" for i in range(n_words))


@pytest.fixture
def synthetic_records():
    return [
        {"page_id": 1, "title": "Short Page", "content": "This is a short entry."},
        {"page_id": 2, "title": "Another Short", "content": "Another brief entry here."},
        {"page_id": 3, "title": "Long Page Alpha", "content": _long_content(250)},
        {"page_id": 4, "title": "Long Page Beta", "content": _long_content(300)},
        {"page_id": "5", "title": "", "content": "Entry with no title and string page_id."},
    ]


@pytest.fixture
def tmp_entries_dir(tmp_path, synthetic_records):
    entries_dir = tmp_path / "entries"
    entries_dir.mkdir()
    for record in synthetic_records:
        pid = record["page_id"]
        (entries_dir / f"{pid}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
    return entries_dir


@pytest.fixture
def tmp_artifacts_dir(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    return artifacts
