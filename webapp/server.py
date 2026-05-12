"""
server.py — FastAPI app that wraps the highlight pipeline as a web service.

Endpoints:
    GET  /              -> index.html (file-upload form)
    POST /jobs          -> multipart upload (file + match name) -> {"job_id"}
    GET  /jobs/{id}     -> job status JSON
    GET  /health        -> liveness check

Job lifecycle (single-process, single-VM v1):
    queued -> running (stages: scanning -> extracting -> uploading) -> done|failed
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

# Make the project root importable so we can do `from video_highlight.pipeline import run_pipeline`
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from video_highlight.pipeline import run_pipeline  # noqa: E402
from webapp.blob import upload_and_sas  # noqa: E402


app = FastAPI(title="GSFA Highlights", version="0.2.0")

JOBS: dict[str, dict] = {}

_job_sem = asyncio.Semaphore(1)

WORK_DIR = Path(os.environ.get("WORK_DIR", "/tmp/jobs"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

USE_GPU = os.environ.get("USE_GPU", "0") == "1"
MAX_UPLOAD_GB = float(os.environ.get("MAX_UPLOAD_GB", "15"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_GB * 1024 * 1024 * 1024)

ALLOWED_EXTS = {".mp4", ".mkv", ".mov", ".webm"}


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
    # Fast-reject oversize bodies before we stream a single byte to disk.
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    413, f"File exceeds limit of {MAX_UPLOAD_GB:g} GB"
                )
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

    # Stream the upload to disk in chunks. Never .read() the whole file —
    # a 10 GB match would OOM the container.
    bytes_written = 0
    try:
        with dest_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    out.close()
                    shutil.rmtree(job_dir, ignore_errors=True)
                    JOBS.pop(job_id, None)
                    raise HTTPException(
                        413, f"File exceeds limit of {MAX_UPLOAD_GB:g} GB"
                    )
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
    if not job:
        raise HTTPException(404, "Unknown job_id")
    return JSONResponse(job)


async def _run_job(job_id: str, video_path: Path, match: str) -> None:
    job = JOBS[job_id]
    tmp_dir = video_path.parent
    async with _job_sem:
        try:
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
            )

            if result.get("output_path") is None or not out_reel.exists():
                job.update({
                    "status": "failed",
                    "error": "No highlights detected in this video (no goals, halftime or penalties found).",
                })
                return

            job.update({"stage": "uploading", "progress": 0.0})
            blob_name = f"{match}_{job_id}.mp4"
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

        except Exception as e:
            job.update({"status": "failed", "error": f"{type(e).__name__}: {e}"})
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


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
