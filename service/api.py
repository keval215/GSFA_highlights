"""
service/api.py — FastAPI ingestion endpoint (no GPU).

POST /api/clips  multipart/form-data:
    file (60 s mp4), match_id
    [+ half, minute, team0_name, team1_name, team0_colour, team1_colour]
  → upload blob clips/<match_id>/<half>_<minute>.mp4
  → enqueue {"match_id","half","minute","blob_path"}
  → 202 in ~1–2 s. Processing is never inline.

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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from service import config, db
from service.blob import ClipBlobStore, blob_name
from service.queueing import ClipQueue

log = logging.getLogger("gsfa.api")

app = FastAPI(title="GSFA Highlights ingestion", version="1.0")

_blob:  Optional[ClipBlobStore] = None
_queue: Optional[ClipQueue]     = None


@app.on_event("startup")
def _startup() -> None:
    global _blob, _queue
    _blob  = ClipBlobStore()
    _blob.ensure_container()
    _queue = ClipQueue()
    _queue.ensure_queues()


@app.post("/api/clips", status_code=202)
async def post_clip(
    file: UploadFile = File(...),
    match_id: str = Form(...),
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

    # First clip auto-creates the match; later clips fill missing metadata.
    conn = db.get_conn()
    try:
        db.ensure_match(conn, match_id, team0_name, team1_name,
                        team0_colour, team1_colour)
        resolved_half   = half   if half   is not None else 1
        resolved_minute = minute if minute is not None else db.next_minute(conn, match_id, resolved_half)
        already_processed = db.minute_exists(conn, match_id, resolved_half, resolved_minute)
    finally:
        conn.close()
    half, minute = resolved_half, resolved_minute

    name = blob_name(match_id, half, minute)
    if already_processed or _blob.exists(name):
        log.info("duplicate clip %s h%d m%d — skipped", match_id, half, minute)
        return {"accepted": True, "duplicate": True,
                "match_id": match_id, "half": half, "minute": minute}

    _blob.upload_stream(name, file.file)
    _queue.enqueue(match_id, half, minute, name)
    return {"accepted": True, "match_id": match_id, "half": half, "minute": minute}


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
    }


def _read_heartbeat() -> Optional[dict]:
    try:
        return json.loads(config.HEARTBEAT_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
