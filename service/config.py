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
    """Base origin of the main app (e.g. https://dev-server.clubduelz.in), no
    path. Empty string disables sending (outbox rows stay pending so nothing is
    lost while the URL is not yet provided)."""
    return os.environ.get("CALLBACK_URL", "").strip()


def super_admin_key() -> str:
    """X-Super-Admin-Key for the advance-stats endpoint. Empty disables sending."""
    return os.environ.get("SUPER_ADMIN_KEY", "").strip()


def advance_stats_url(duel_id: str) -> str:
    """Full advance-stats URL for a tournament duel. Empty base disables sending."""
    base = callback_url().rstrip("/")
    return f"{base}/v1/pvt/tournament-duelz/{duel_id}/advance-stats" if base else ""


CLIPS_CONTAINER   = os.environ.get("CLIPS_CONTAINER", "clips")
QUEUE_NAME        = os.environ.get("QUEUE_NAME", "clips")
POISON_QUEUE_NAME = os.environ.get("POISON_QUEUE_NAME", "clips-poison")

# --- Model weights (VM disk paths) ----------------------------------------

def player_weights(ruleset_name: str) -> str:
    """Unified YOLOv11m weights path for one ruleset: players + ball + refs +
    posts. 'futsal' keeps today's PLAYER_WEIGHTS env var name (backward
    compatible with existing deployments); other rulesets use
    <RULESET>_PLAYER_WEIGHTS, required only when that ruleset is actually
    used (a deployment that never serves classic matches doesn't need
    CLASSIC_PLAYER_WEIGHTS set)."""
    if ruleset_name == "futsal":
        return _required("PLAYER_WEIGHTS")
    return _required(f"{ruleset_name.upper()}_PLAYER_WEIGHTS")


# --- Local state directories ------------------------------------------------

MATCH_STATE_DIR = Path(os.environ.get("MATCH_STATE_DIR", "/mnt/data/match_state"))
JOBS_DIR        = Path(os.environ.get("JOBS_DIR", "/mnt/data/jobs"))
HEARTBEAT_FILE  = MATCH_STATE_DIR / "worker_heartbeat.json"

# --- Tunables ----------------------------------------------------------------

TARGET_PROCESS_FPS    = float(os.environ.get("TARGET_PROCESS_FPS", "15"))
# BoT-SORT camera-motion compensation: ecc (default, accurate, CPU-heavy) |
# sof (sparse optical flow, cheaper) | orb | sift | none. sof is a speed/
# accuracy trade — validate events/minute_stats before switching off ecc.
CMC_METHOD            = os.environ.get("CMC_METHOD", "ecc")
# Frames per GPU batch in clip_processor's two-pass loop (Pass 1 inference).
CLIP_BATCH_WINDOW     = int(os.environ.get("CLIP_BATCH_WINDOW", "16"))
# Overlap clip decode + Pass 1 (GPU) on a producer thread with the sequential
# Pass 2 (tracker/FSM) on the main thread, via a bounded queue. Off by default:
# when False, clip_processor runs the unchanged serial path with no thread
# created. Kill switch for the worker's first threaded code path — flip off to
# re-serialise without a redeploy.
CLIP_PIPELINE_THREADED = os.environ.get("CLIP_PIPELINE_THREADED", "false").strip().lower() in (
    "1", "true", "yes", "on",
)
# Ball is now produced by the unified detection model every processed frame
# (no separate stride). BallTracker still coasts via Kalman on gaps.
DEVICE                = os.environ.get("DEVICE", "cuda")          # detectors + classifier
# Team-fit torso-crop sampling stride (every Nth raw frame of whatever video
# collect_crops() is given). Bounds crop volume for whole-match /post-processing
# uploads (which hand collect_crops the entire match, not just a ~60s clip) while
# still sampling frames spread across the full file.
FIT_SAMPLE_EVERY      = int(os.environ.get("FIT_SAMPLE_EVERY", "30"))
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
