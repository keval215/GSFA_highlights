"""
server.py — FastAPI app that wraps the highlight pipeline as a web service.

Endpoints:
    GET  /              -> index.html (URL submission form)
    POST /jobs          -> {"youtube_url": "...", "match": "..."} -> {"job_id"}
    GET  /jobs/{id}     -> job status JSON
    GET  /health        -> liveness check

Job lifecycle (single-process, single-VM v1):
    queued -> running (stages: downloading -> scanning -> extracting -> uploading) -> done|failed
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

# Make the project root importable so we can do `from video_highlight.pipeline import run_pipeline`
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from video_highlight.pipeline import run_pipeline  # noqa: E402
from webapp.blob import upload_and_sas  # noqa: E402
from webapp.ytdlp_fetch import download_video, probe_duration_min, YtDlpError  # noqa: E402


app = FastAPI(title="GSFA Highlights", version="0.1.0")

# In-memory job state. Acceptable for single-instance v1; on restart, in-flight
# jobs are lost (documented limitation).
JOBS: dict[str, dict] = {}

# Serialize GPU-bound work — only one pipeline run at a time.
_job_sem = asyncio.Semaphore(1)

# Working directory for downloads + intermediate clips.
WORK_DIR = Path(os.environ.get("WORK_DIR", "/tmp/jobs"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

USE_GPU = os.environ.get("USE_GPU", "0") == "1"
MAX_VIDEO_MIN = float(os.environ.get("MAX_VIDEO_DURATION_MIN", "120"))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict:
    return {"ok": True, "use_gpu": USE_GPU, "jobs_in_memory": len(JOBS)}


@app.post("/jobs")
async def create_job(payload: dict, bg: BackgroundTasks) -> dict:
    youtube_url = (payload.get("youtube_url") or "").strip()
    match = (payload.get("match") or "").strip() or "match"
    if not youtube_url:
        raise HTTPException(400, "youtube_url is required")
    if "youtube.com" not in youtube_url and "youtu.be" not in youtube_url:
        raise HTTPException(400, "Only YouTube URLs are supported")

    # Sanitize match name for filenames
    safe_match = "".join(c if (c.isalnum() or c in "-_") else "_" for c in match)[:64] or "match"

    job_id = uuid4().hex[:12]
    JOBS[job_id] = {
        "id": job_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0.0,
        "match": safe_match,
        "youtube_url": youtube_url,
        "created_at": time.time(),
    }
    bg.add_task(_run_job, job_id, youtube_url, safe_match)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job_id")
    return JSONResponse(job)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

async def _run_job(job_id: str, youtube_url: str, match: str) -> None:
    job = JOBS[job_id]

    # Pre-flight duration check (no download yet)
    try:
        duration_min = await asyncio.to_thread(probe_duration_min, youtube_url)
    except Exception:
        duration_min = None
    if duration_min is not None and duration_min > MAX_VIDEO_MIN:
        job.update({
            "status": "failed",
            "error": f"Video is {duration_min:.1f} min — exceeds limit of {MAX_VIDEO_MIN} min",
        })
        return
    if duration_min is not None:
        job["duration_min"] = round(duration_min, 1)

    async with _job_sem:
        tmp_dir = WORK_DIR / job_id
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            job.update({"status": "running", "stage": "downloading", "progress": 0.0})

            video_path = await asyncio.to_thread(
                download_video, youtube_url, tmp_dir / "input.mp4"
            )
            job.update({"stage": "scanning", "progress": 0.0})

            def cb(stage: str, frac: float) -> None:
                # Called from worker thread; dict mutation is safe enough for
                # this purpose (CPython dict updates are atomic per-key).
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

        except YtDlpError as e:
            job.update({"status": "failed", "error": f"YouTube download failed: {e}"})
        except Exception as e:  # last-resort catch — surface message to user
            job.update({"status": "failed", "error": f"{type(e).__name__}: {e}"})
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Startup hygiene: clean stale working dirs from previous crashes (>1 hr old).
# ---------------------------------------------------------------------------

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
