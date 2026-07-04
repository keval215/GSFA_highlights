"""service/post_processing/post_processing.py — whole-match processing.

This module reuses the existing team-fit + clip-processing pipeline for the
post-match upload path. The only difference from live clips is that the entire
video is processed as one unit and the resulting counters are written to the
post_processing table instead of minute_stats.
"""

from __future__ import annotations

from service.clip_processor import process_clip
from service.session import MatchSession


def process_match_video(
    session: MatchSession,
    video_path: str,
    blob_path: str | None = None,
):
    """Run the existing pipeline over the entire uploaded match video."""
    if session.fit_status != "ok":
        session.ensure_fit(video_path)
    return process_clip(
        session,
        video_path,
        half=1,
        minute=1,
        clip_duration_seconds=0.0,
        clip_blob_path=blob_path,
    )