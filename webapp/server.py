"""
server.py — FastAPI app that wraps the highlight pipeline as a web service.

Endpoints:
    GET  /                  -> index.html (file-upload form + history)
    GET  /health            -> liveness check
    POST /jobs              -> multipart upload (file + match name) -> {"job_id"}
    GET  /jobs              -> list of jobs (running + history)
    GET  /jobs/{id}         -> single job status JSON
    POST /jobs/{id}/cancel  -> request cooperative cancel
    DELETE /jobs            -> clear history + delete blobs

Job lifecycle (single-process, single-VM v1):
    queued -> running (stages: scanning -> extracting -> uploading) -> done|failed|cancelled
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
from pathlib import Path
from uuid import uuid4

import json

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from video_highlight.pipeline import run_pipeline, PipelineCancelled  # noqa: E402
from webapp.blob import upload_and_sas, delete_blob  # noqa: E402
from webapp import history  # noqa: E402


app = FastAPI(title="GSFA Highlights", version="0.3.0")

# In-memory state for *active* jobs. Completed jobs are persisted via webapp/history.
JOBS: dict[str, dict] = {}
_cancel_flags: dict[str, bool] = {}

_job_sem = asyncio.Semaphore(1)

WORK_DIR = Path(os.environ.get("WORK_DIR", "/tmp/jobs"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

USE_GPU = os.environ.get("USE_GPU", "0") == "1"
MAX_UPLOAD_GB = float(os.environ.get("MAX_UPLOAD_GB", "15"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_GB * 1024 * 1024 * 1024)

ALLOWED_EXTS = {".mp4", ".mkv", ".mov", ".webm"}
TERMINAL_STATUSES = {"done", "failed", "cancelled"}


def _is_terminal(status: str) -> bool:
    return status in TERMINAL_STATUSES


def _persist(job: dict) -> None:
    try:
        history.append_job(job)
    except Exception as e:
        print(f"[history] save failed: {e}")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "use_gpu": USE_GPU,
        "jobs_in_memory": len(JOBS),
        "max_upload_gb": MAX_UPLOAD_GB,
    }


@app.post("/jobs")
async def create_job(
    request: Request,
    bg: BackgroundTasks,
    file: UploadFile = File(...),
    match: str = Form("match"),
) -> dict:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"File exceeds limit of {MAX_UPLOAD_GB:g} GB")
        except ValueError:
            pass

    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            400,
            f"Unsupported file type {ext!r}. Allowed: {', '.join(sorted(ALLOWED_EXTS))}",
        )

    safe_match = (
        "".join(c if (c.isalnum() or c in "-_") else "_" for c in match.strip())[:64]
        or "match"
    )

    job_id = uuid4().hex[:12]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    dest_path = job_dir / f"input{ext}"

    JOBS[job_id] = {
        "id": job_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0.0,
        "match": safe_match,
        "filename": filename,
        "created_at": time.time(),
    }

    bytes_written = 0
    try:
        with dest_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    out.close()
                    shutil.rmtree(job_dir, ignore_errors=True)
                    JOBS.pop(job_id, None)
                    raise HTTPException(413, f"File exceeds limit of {MAX_UPLOAD_GB:g} GB")
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        JOBS.pop(job_id, None)
        raise HTTPException(500, f"Upload failed: {e}") from e
    finally:
        await file.close()

    if bytes_written == 0:
        shutil.rmtree(job_dir, ignore_errors=True)
        JOBS.pop(job_id, None)
        raise HTTPException(400, "Uploaded file is empty")

    JOBS[job_id]["size_bytes"] = bytes_written
    bg.add_task(_run_job, job_id, dest_path, safe_match)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if job:
        return JSONResponse(job)
    for entry in history.load_history():
        if entry.get("id") == job_id:
            return JSONResponse(entry)
    raise HTTPException(404, "Unknown job_id")


@app.get("/jobs")
def list_jobs() -> dict:
    active = list(JOBS.values())
    historical = history.load_history()
    seen = {j["id"] for j in active}
    merged = active + [h for h in historical if h.get("id") not in seen]
    merged.sort(key=lambda j: j.get("finished_at") or j.get("created_at") or 0, reverse=True)
    return {"jobs": merged[:100]}


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job_id")
    if _is_terminal(job.get("status", "")):
        raise HTTPException(409, f"Job already {job['status']}")
    _cancel_flags[job_id] = True
    job["stage"] = "cancelling"
    return {"ok": True, "status": "cancelling"}


@app.delete("/jobs/{job_id}/upload")
def delete_upload(job_id: str) -> dict:
    """Best-effort cleanup of a partial upload when the client aborts before processing starts."""
    job = JOBS.get(job_id)
    if job and job.get("status") not in ("queued", None):
        raise HTTPException(409, f"Job already {job.get('status')}")
    job_dir = WORK_DIR / job_id
    try:
        shutil.rmtree(job_dir, ignore_errors=True)
    except Exception:
        pass
    JOBS.pop(job_id, None)
    _cancel_flags.pop(job_id, None)
    return {"ok": True}


@app.get("/jobs/{job_id}/events")
async def stream_job_events(job_id: str) -> StreamingResponse:
    """SSE stream of job state updates. Closes when the job reaches a terminal status."""

    async def gen():
        last_snapshot: str | None = None
        # Brief grace period: job may be created just after upload completes.
        for _ in range(20):
            if job_id in JOBS:
                break
            for entry in history.load_history():
                if entry.get("id") == job_id:
                    yield f"data: {json.dumps(entry)}\n\n"
                    return
            await asyncio.sleep(0.1)
        else:
            yield f"data: {json.dumps({'error': 'unknown job_id', 'id': job_id})}\n\n"
            return

        while True:
            job = JOBS.get(job_id)
            if job is None:
                for entry in history.load_history():
                    if entry.get("id") == job_id:
                        yield f"data: {json.dumps(entry)}\n\n"
                        return
                return
            snap = json.dumps(job, sort_keys=True, default=str)
            if snap != last_snapshot:
                last_snapshot = snap
                yield f"data: {snap}\n\n"
            if _is_terminal(job.get("status", "")):
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/jobs")
def delete_all_jobs() -> dict:
    historical = history.clear_history()
    failures: list[str] = []
    for entry in historical:
        blob_name = entry.get("blob_name")
        if not blob_name:
            continue
        if not delete_blob(blob_name):
            failures.append(blob_name)
    return {"deleted": len(historical), "blob_failures": failures}


async def _run_job(job_id: str, video_path: Path, match: str) -> None:
    job = JOBS[job_id]
    tmp_dir = video_path.parent
    persisted = False
    async with _job_sem:
        try:
            if _cancel_flags.get(job_id):
                raise PipelineCancelled()

            job.update({"status": "running", "stage": "scanning", "progress": 0.0})

            def cb(stage: str, frac: float) -> None:
                job["stage"] = stage
                job["progress"] = float(frac)

            out_reel = tmp_dir / "highlights.mp4"
            result = await asyncio.to_thread(
                run_pipeline,
                input_video=video_path,
                out_dir=tmp_dir / "clips",
                output_path=out_reel,
                match=match,
                use_gpu=USE_GPU,
                progress_cb=cb,
                cancel_check=lambda: _cancel_flags.get(job_id, False),
            )

            if result.get("output_path") is None or not out_reel.exists():
                job.update({
                    "status": "failed",
                    "stage": "failed",
                    "error": "No highlights detected in this video (no goals, halftime or penalties found).",
                    "finished_at": time.time(),
                })
                return

            job.update({"stage": "uploading", "progress": 0.0})
            blob_name = f"{match}_{job_id}.mp4"
            job["blob_name"] = blob_name
            sas_url = await asyncio.to_thread(upload_and_sas, out_reel, blob_name)

            job.update({
                "status": "done",
                "stage": "done",
                "progress": 1.0,
                "result_url": sas_url,
                "goals_found": len(result["goals"]),
                "endgame_clips": len(result["endgame_clips"]),
                "finished_at": time.time(),
            })

        except PipelineCancelled:
            job.update({
                "status": "cancelled",
                "stage": "cancelled",
                "error": "Cancelled by user",
                "finished_at": time.time(),
            })
        except Exception as e:
            job.update({
                "status": "failed",
                "stage": "failed",
                "error": f"{type(e).__name__}: {e}",
                "finished_at": time.time(),
            })
        finally:
            _persist(job)
            persisted = True
            shutil.rmtree(tmp_dir, ignore_errors=True)
            _cancel_flags.pop(job_id, None)
            JOBS.pop(job_id, None)
    if not persisted:
        _persist(job)


@app.on_event("startup")
def _sweep_stale_work_dirs() -> None:
    if not WORK_DIR.exists():
        return
    cutoff = time.time() - 3600
    for child in WORK_DIR.iterdir():
        try:
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
        except Exception:
            pass
