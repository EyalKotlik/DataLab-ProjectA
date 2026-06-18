"""Shared paths and helpers for Section B."""
from __future__ import annotations

import json
import logging
import logging.handlers
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

STUDENT_ROOT = Path(__file__).resolve().parent
DATA_DIR = STUDENT_ROOT / "data"
ENTRIES_DIR = DATA_DIR / "Wikipedia Entries"
# The raw corpus (~27k JSON files) is shipped as a single LFS-tracked archive
# instead of thousands of tracked files; extract it on first use.
CORPUS_ARCHIVE = DATA_DIR / "corpus.tar.gz"
PUBLIC_QUERIES_PATH = DATA_DIR / "public_queries.json"
ARTIFACTS_DIR = STUDENT_ROOT / "artifacts"
LOG_DIR = STUDENT_ROOT / "logs"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K_EVAL = 10

_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"


def setup_logging(
    level: int = logging.DEBUG,
    log_file: Optional[Path] = None,
) -> None:
    """Configure root logger with a stderr handler and optional file handler."""
    root = logging.getLogger()
    root.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT)

    has_stderr = any(
        type(h) is logging.StreamHandler for h in root.handlers
    )
    if not has_stderr:
        stderr_handler = logging.StreamHandler()
        stderr_handler.setFormatter(formatter)
        root.addHandler(stderr_handler)

    if log_file is not None:
        target = str(log_file)
        has_file = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == target
            for h in root.handlers
        )
        if not has_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)


def normalize_page_id(value: Any) -> int:
    """Coerce page_id from JSON (int or numeric string) to int."""
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"Invalid page_id: {value!r}")


def load_public_queries(path: Path | None = None) -> List[Dict[str, Any]]:
    path = path or PUBLIC_QUERIES_PATH
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        row["relevant_page_ids"] = [
            normalize_page_id(pid) for pid in row["relevant_page_ids"]
        ]
    return rows


def ensure_corpus_extracted() -> Path:
    """Extract the corpus archive into ENTRIES_DIR if not already present.

    The raw corpus ships as ``data/corpus.tar.gz`` (one LFS object) rather than
    ~27k tracked files. Offline index builds and diagnostics call this to make
    the JSON files available; the timed runtime never needs the corpus.
    """
    if ENTRIES_DIR.is_dir() and any(ENTRIES_DIR.glob("*.json")):
        return ENTRIES_DIR
    if CORPUS_ARCHIVE.is_file():
        with tarfile.open(CORPUS_ARCHIVE, "r:gz") as tar:
            tar.extractall(DATA_DIR)
    return ENTRIES_DIR


def iter_entries(entries_dir: Path | None = None) -> Iterator[Dict[str, Any]]:
    """Yield one record per JSON file in the corpus directory."""
    if entries_dir is None:
        ensure_corpus_extracted()
    root = entries_dir or ENTRIES_DIR
    if not root.is_dir():
        raise FileNotFoundError(
            f"Corpus directory not found: {root}. "
            "Expected data/Wikipedia Entries/ (or data/corpus.tar.gz to extract)."
        )
    for path in sorted(root.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["page_id"] = normalize_page_id(data.get("page_id", path.stem))
        yield data


def entry_text(record: Dict[str, Any]) -> str:
    title = record.get("title", "")
    content = record.get("content", "")
    if title:
        return f"{title}\n\n{content}".strip()
    return str(content).strip()


def ensure_artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR
