"""
service/api.py — FastAPI ingestion endpoint (no GPU).

POST /api/clips  multipart/form-data:
    file (60 s mp4), match_id
        [+ clip_duration_seconds, half, minute, team0_name, team1_name,
             team0_colour, team1_colour]
  → upload blob clips/<match_id>/<half>_<minute>.mp4
    → enqueue {"match_id","half","minute","blob_path","clip_duration_seconds"}
  → 202 in ~1–2 s. Processing is never inline.

POST /post-processing  multipart/form-data:
    file (whole-match mp4), match_id
        [+ team0_name, team1_name, team0_colour, team1_colour]
    → upload blob clips/<match_id>/post_processing.mp4
        → enqueue {"kind":"post_processing", ...}
    → 200 once the upload is fully received and queued.

half and minute are optional (default 0). Duplicate (match_id, half, minute)
⇒ 202 with "duplicate": true, clip skipped.

GET /health  — api liveness + worker heartbeat + GPU visibility.
GET /metrics — queue depths, last clip seconds, seconds-behind-live, GPU mem.

Run:  uvicorn service.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from service import config, db, logging_setup
from service.blob import ClipBlobStore, blob_name, post_processing_blob_name
from service.queueing import ClipQueue

# Under `uvicorn service.api:app` the root logger has no handler, so our gsfa.api
# lines would be swallowed. Configure it ourselves (IST timestamps, azure HTTP
# logging silenced — shared with the worker process).
logging_setup.configure("api")

log = logging.getLogger("gsfa.api")

app = FastAPI(title="GSFA Highlights ingestion", version="1.0")

_blob:  Optional[ClipBlobStore] = None
_queue: Optional[ClipQueue]     = None


class _AccessLogFilter(logging.Filter):
    """Drop uvicorn access-log lines for paths we don't serve — internet
    scanners hammering /, /favicon.ico, /mcp, etc. spam 404s otherwise."""

    _KEEP = ("/api/", "/post-processing", "/health", "/metrics")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return any(p in msg for p in self._KEEP)


@app.exception_handler(RequestValidationError)
async def _on_validation_error(request: Request, exc: RequestValidationError):
    """FastAPI's built-in validation (missing match_id/file, non-int half/minute,
    …) otherwise returns a bare 422 with no reason in the log. Log the precise
    field+reason and still return the detailed body to the caller."""
    log.warning("invalid POST %s — %s", request.url.path, exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(HTTPException)
async def _on_http_error(request: Request, exc: HTTPException):
    """Surface the *reason* for our explicit 4xx rejections (415/413/422) in the
    gsfa.api log, not just the uvicorn status line. 404 etc. stay quiet."""
    if exc.status_code >= 400 and exc.status_code != 404:
        log.warning("rejected %s — HTTP %d: %s",
                    request.url.path, exc.status_code, exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail},
                        headers=getattr(exc, "headers", None))


@app.on_event("startup")
def _startup() -> None:
    global _blob, _queue
    logging.getLogger("uvicorn.access").addFilter(_AccessLogFilter())
    _blob  = ClipBlobStore()
    _blob.ensure_container()
    _queue = ClipQueue()
    _queue.ensure_queues()


@app.post("/api/clips", status_code=202)
async def post_clip(
    file: UploadFile = File(...),
    match_id: str = Form(...),
    clip_duration_seconds: float = Form(...),
    half: Optional[int] = Form(None),
    minute: Optional[int] = Form(None),
    team0_name: Optional[str] = Form(None),
    team1_name: Optional[str] = Form(None),
    team0_colour: Optional[str] = Form(None),   # hex "#FF6600" or CSS name "orange"
    team1_colour: Optional[str] = Form(None),
):
    # Basic hygiene only (no auth in v1 — NSG restricts port 8000).
    if file.content_type not in ("video/mp4", "application/octet-stream", None):
        raise HTTPException(415, f"unsupported content type: {file.content_type}")
    if file.size is not None and file.size > config.MAX_UPLOAD_GB * 1024**3:
        raise HTTPException(413, f"clip exceeds {config.MAX_UPLOAD_GB} GB cap")
    if clip_duration_seconds <= 0:
        raise HTTPException(422, f"clip_duration_seconds must be > 0, got {clip_duration_seconds}")
    if half is not None and half < 1:
        raise HTTPException(422, f"half must be >= 1, got {half}")
    if minute is not None and minute < 1:
        raise HTTPException(422, f"minute must be >= 1, got {minute}")

    # First clip auto-creates the match; later clips fill missing metadata.
    conn = db.get_conn()
    try:
        db.ensure_match(conn, match_id, team0_name, team1_name,
                        team0_colour, team1_colour)
        resolved_half   = half   if half   is not None else 1
        resolved_minute = minute if minute is not None else db.claim_next_minute(conn, match_id, resolved_half)
        already_processed = db.minute_exists(conn, match_id, resolved_half, resolved_minute)
    finally:
        conn.close()
    half, minute = resolved_half, resolved_minute

    name = blob_name(match_id, half, minute)
    if already_processed or _blob.exists(name):
        log.debug("duplicate clip %s h%d m%d — skipped", match_id, half, minute)
        return {"accepted": True, "duplicate": True,
                "match_id": match_id, "half": half, "minute": minute}

    _blob.upload_stream(name, file.file)
    _queue.enqueue(match_id, half, minute, name, clip_duration_seconds)
    size_mb = (file.size / 1024**2) if file.size else 0.0
    log.info("received clip %s h%d m%d (%.1f MB) — queued", match_id, half, minute, size_mb)
    return {"accepted": True, "match_id": match_id, "half": half, "minute": minute}


@app.post("/post-processing", status_code=200)
async def post_processing(
    file: UploadFile = File(...),
    match_id: str = Form(...),
    team0_name: Optional[str] = Form(None),
    team1_name: Optional[str] = Form(None),
    team0_colour: Optional[str] = Form(None),
    team1_colour: Optional[str] = Form(None),
):
    if file.content_type not in ("video/mp4", "application/octet-stream", None):
        raise HTTPException(415, f"unsupported content type: {file.content_type}")
    if file.size is not None and file.size > config.MAX_UPLOAD_GB * 1024**3:
        raise HTTPException(413, f"video exceeds {config.MAX_UPLOAD_GB} GB cap")

    conn = db.get_conn()
    try:
        db.ensure_match(conn, match_id, team0_name, team1_name, team0_colour, team1_colour)
        already_processed = db.post_processing_exists(conn, match_id)
    finally:
        conn.close()

    name = post_processing_blob_name(match_id)
    if already_processed or _blob.exists(name):
        log.debug("duplicate post-processing upload %s — skipped", match_id)
        return {"received": True, "duplicate": True, "match_id": match_id}

    _blob.upload_stream(name, file.file)
    _queue.enqueue_post_processing(
        match_id=match_id,
        blob_path=name,
        team0_name=team0_name,
        team1_name=team1_name,
        team0_colour=team0_colour,
        team1_colour=team1_colour,
    )
    size_mb = (file.size / 1024**2) if file.size else 0.0
    log.info("received post-processing video %s (%.1f MB) — queued", match_id, size_mb)
    return {"received": True, "match_id": match_id}


@app.get("/health")
def health():
    hb = _read_heartbeat()
    worker_alive = hb is not None and (time.time() - hb.get("ts", 0)) < 300
    body = {
        "api": "ok",
        "worker_alive": worker_alive,
        "gpu_visible": bool(hb and hb.get("gpu_visible")),
        "worker_heartbeat_age_s": round(time.time() - hb["ts"], 1) if hb else None,
    }
    return JSONResponse(body, status_code=200 if worker_alive else 503)


@app.get("/metrics")
def metrics():
    main_depth, poison_depth = _queue.depths()
    hb = _read_heartbeat() or {}
    return {
        "queue_depth": main_depth,
        "poison_depth": poison_depth,
        "last_clip_processing_seconds": hb.get("last_clip_seconds"),
        "seconds_behind_live": hb.get("seconds_behind_live"),
        "active_matches": hb.get("active_matches", []),
        "gpu_memory_allocated_mb": hb.get("gpu_mem_mb"),
        "last_dequeue_count": hb.get("last_dequeue_count"),
        "last_post_processing_error": hb.get("last_post_processing_error"),
    }


def _read_heartbeat() -> Optional[dict]:
    try:
        return json.loads(config.HEARTBEAT_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
