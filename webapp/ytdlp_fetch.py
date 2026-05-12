"""
ytdlp_fetch.py — Wrap yt-dlp into a callable so the webapp can download a
YouTube video to a known path. Honours an optional cookies file
(`YOUTUBE_COOKIES_FILE` env var) needed when downloading from cloud IPs.
"""

from __future__ import annotations

import os
from pathlib import Path

import yt_dlp


class YtDlpError(Exception):
    pass


def download_video(youtube_url: str, dest: Path) -> Path:
    """
    Download `youtube_url` to `dest` (mp4). Returns the final path.

    The merge step may rename the file (e.g. .webm -> .mp4). The function
    returns the actual produced file, which is guaranteed to live in
    `dest.parent` and start with `dest.stem`.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out_template = str(dest.with_suffix("")) + ".%(ext)s"

    ydl_opts: dict = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": out_template,
        "merge_output_format": "mp4",
        "socket_timeout": 30,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
    }

    cookies_file = os.environ.get("YOUTUBE_COOKIES_FILE", "").strip()
    if cookies_file and Path(cookies_file).exists():
        ydl_opts["cookiefile"] = cookies_file

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
    except yt_dlp.utils.DownloadError as e:
        raise YtDlpError(str(e)) from e

    # Locate the produced file (extension may have changed during merge)
    for candidate in dest.parent.glob(dest.stem + ".*"):
        if candidate.is_file() and candidate.suffix.lower() in {".mp4", ".mkv", ".webm"}:
            return candidate
    raise YtDlpError(f"yt-dlp produced no output under {dest.parent} for stem {dest.stem!r}")


def probe_duration_min(youtube_url: str) -> float | None:
    """
    Cheap pre-flight: ask yt-dlp for the video's duration in minutes without
    downloading. Returns None if not available.
    """
    cookies_file = os.environ.get("YOUTUBE_COOKIES_FILE", "").strip()
    ydl_opts: dict = {"quiet": True, "no_warnings": True, "skip_download": True}
    if cookies_file and Path(cookies_file).exists():
        ydl_opts["cookiefile"] = cookies_file
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
    except Exception:
        return None
    dur = info.get("duration") if isinstance(info, dict) else None
    return (dur / 60.0) if dur else None
