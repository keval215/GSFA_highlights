"""
service/config.py — every environment variable in one place.

All settings come from the environment (populated on the VM from
/etc/gsfa-highlights.env via docker-compose env_file). Nothing is
hardcoded; missing required vars fail fast at startup with a clear
message instead of deep inside a request.
"""

from __future__ import annotations

import os
from pathlib import Path


def _required(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise RuntimeError(
            f"Required environment variable {name} is not set. "
            f"Add it to /etc/gsfa-highlights.env (see NEEDED_FROM_YOU.md)."
        )
    return val


# --- Azure resources -------------------------------------------------------

def storage_conn_str() -> str:
    """Connection string for the gsfastorage account (Blob + Queue)."""
    return _required("AZURE_STORAGE_CONNECTION_STRING")


def sql_conn_str() -> str:
    """pyodbc connection string for Azure SQL gsfa_stats."""
    return _required("SQL_CONN_STR")


def callback_url() -> str:
    """Main-app callback URL. Empty string disables sending (outbox rows
    stay pending so nothing is lost while the URL is not yet provided)."""
    return os.environ.get("CALLBACK_URL", "").strip()


CLIPS_CONTAINER   = os.environ.get("CLIPS_CONTAINER", "clips")
QUEUE_NAME        = os.environ.get("QUEUE_NAME", "clips")
POISON_QUEUE_NAME = os.environ.get("POISON_QUEUE_NAME", "clips-poison")

# --- Model weights (VM disk paths) ----------------------------------------

def player_weights() -> str:
    return _required("PLAYER_WEIGHTS")


def ball_weights() -> str:
    return _required("BALL_WEIGHTS")


# --- Local state directories ------------------------------------------------

MATCH_STATE_DIR = Path(os.environ.get("MATCH_STATE_DIR", "/mnt/data/match_state"))
JOBS_DIR        = Path(os.environ.get("JOBS_DIR", "/mnt/data/jobs"))
HEARTBEAT_FILE  = MATCH_STATE_DIR / "worker_heartbeat.json"

# --- Tunables ----------------------------------------------------------------

TARGET_PROCESS_FPS    = float(os.environ.get("TARGET_PROCESS_FPS", "15"))
DEVICE                = os.environ.get("DEVICE", "cuda")          # detectors + classifier
FIT_SAMPLE_EVERY      = int(os.environ.get("FIT_SAMPLE_EVERY", "5"))
FIT_SILHOUETTE_MIN    = float(os.environ.get("FIT_SILHOUETTE_MIN", "0.20"))
QUEUE_VISIBILITY_SEC  = int(os.environ.get("QUEUE_VISIBILITY_SEC", "90"))
MAX_DEQUEUE_COUNT     = int(os.environ.get("MAX_DEQUEUE_COUNT", "3"))
ORDERING_RETRIES      = int(os.environ.get("ORDERING_RETRIES", "3"))
ORDERING_RETRY_DELAY  = int(os.environ.get("ORDERING_RETRY_DELAY", "10"))   # seconds
CALLBACK_RETRIES      = int(os.environ.get("CALLBACK_RETRIES", "3"))
CALLBACK_BACKOFF_BASE = float(os.environ.get("CALLBACK_BACKOFF_BASE", "1.0"))  # 1s, 2s, 4s
SESSION_IDLE_EVICT_S  = int(os.environ.get("SESSION_IDLE_EVICT_S", str(30 * 60)))
MAX_UPLOAD_GB         = float(os.environ.get("MAX_UPLOAD_GB", "2"))
API_PORT              = int(os.environ.get("API_PORT", "8000"))
