"""
history.py — persist completed jobs across container restarts.

Stores a list of minimal job records as JSON at /mnt/data/jobs_history.json
(configurable via HISTORY_FILE env var). Atomic write via os.replace.
Cap at MAX_HISTORY entries (FIFO trim) to keep the file bounded.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

HISTORY_FILE = Path(os.environ.get(
    "HISTORY_FILE",
    str(Path(os.environ.get("WORK_DIR", "/tmp/jobs")).parent / "jobs_history.json"),
))
MAX_HISTORY = 200

PERSISTED_FIELDS = (
    "id", "match", "filename", "status", "stage", "result_url", "blob_name",
    "goals_found", "endgame_clips", "error", "created_at", "finished_at",
    "size_bytes",
)


def _project(job: dict) -> dict:
    return {k: job[k] for k in PERSISTED_FIELDS if k in job}


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_history(jobs: list[dict]) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    trimmed = jobs[-MAX_HISTORY:]
    fd, tmp_path = tempfile.mkstemp(
        prefix=".jobs_history.", suffix=".json", dir=str(HISTORY_FILE.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(trimmed, f, indent=2)
        os.replace(tmp_path, HISTORY_FILE)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def append_job(job: dict) -> list[dict]:
    """Load history, append a projected copy of `job`, save, return new list."""
    history = load_history()
    history.append(_project(job))
    save_history(history)
    return history


def clear_history() -> list[dict]:
    """Return the list as it was before clearing (so caller can delete blobs)."""
    history = load_history()
    save_history([])
    return history
