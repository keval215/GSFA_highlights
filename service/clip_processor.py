"""
service/clip_processor.py — drives ONE 60 s clip through the existing
pipeline classes (video_analysis/, detectors/, team_classifier/,
tracking/ — no CV logic rewritten, no rendering).

Frames (15 fps stride) are processed in windows of CLIP_BATCH_WINDOW:
    Pass 1 (batched GPU, stateless): unified detect (players + ball) +
        SigLIP team embed for the whole window in one call each.
    Pass 2 (strictly sequential, stateful): track → ball Kalman →
        CarrierEngine → PassEventTracker → bucket the possession label and
        any retroactive adjustments into this minute's counters.
The two passes are equivalent to the old per-frame loop (same frames, same
models, same order into the stateful stages) — only the GPU work is batched.

Adjustments that resolve in this clip but whose travel frames started in
the previous clip are split via session.split_adjustment(): the
prior-minute share becomes a PriorCorrection (one UPDATE to the previous
row), the rest is applied to the in-memory counters before the row is
written.
"""

from __future__ import annotations

import logging
import time

import cv2

from service import config
from service.session import MatchSession
from video_analysis.possession import best_ball
from service.stats import (
    KIND_MAP,
    ClipResult,
    EventRow,
    MinuteCounters,
    PriorCorrection,
)

log = logging.getLogger("gsfa.clip")


def process_clip(
    session: MatchSession,
    clip_path: str,
    half: int,
    minute: int,
    clip_duration_seconds: float,
    clip_blob_path: str | None = None,
) -> ClipResult:
    if session.team_clf is None:
        raise RuntimeError(
            f"process_clip called before team fit for match {session.match_id}"
        )

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise IOError(f"process_clip: cannot open {clip_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, round(fps / config.TARGET_PROCESS_FPS))
    window_k   = max(1, config.CLIP_BATCH_WINDOW)

    counters   = MinuteCounters()
    correction: PriorCorrection | None = None

    # Per-stage timers (ms, accumulated over the whole clip).
    t = {"decode": 0.0, "player_det": 0.0, "team_clf": 0.0, "ball_det": 0.0,
         "tracker": 0.0, "ball_kalman": 0.0, "carrier": 0.0, "pass": 0.0}
    n_processed = 0

    # Processed frames buffered until a window is full, then run as one batch.
    win_frames: list = []
    win_fidx:   list[int] = []

    def flush_window() -> None:
        """Run one window: batched GPU inference (Pass 1, stateless) then the
        sequential stateful logic (Pass 2) in strict frame order — identical
        semantics to the old per-frame loop, just reordered for batching."""
        nonlocal correction, n_processed
        if not win_frames:
            return

        # --- Pass 1: batched, stateless GPU inference -------------------
        p0 = time.perf_counter()
        dets_list = session.models.player_det.detect_batch(win_frames, win_fidx, fps)
        p1 = time.perf_counter(); t["player_det"] += (p1 - p0) * 1000

        session.team_clf.classify_batch(win_frames, dets_list)
        p2 = time.perf_counter(); t["team_clf"] += (p2 - p1) * 1000

        # Ball now comes from the same unified detection pass: pick the best ball
        # per frame from dets_list. No separate model and no stride — the ball is
        # available every processed frame, which improves Kalman continuity.
        balls = [best_ball(d) for d in dets_list]
        p3 = time.perf_counter(); t["ball_det"] += (p3 - p2) * 1000

        # --- Pass 2: strictly sequential, stateful logic ----------------
        for frame, player_dets, raw_ball in zip(win_frames, dets_list, balls):
            s0 = time.perf_counter()
            session.tracker.update(frame, player_dets.players)
            s1 = time.perf_counter(); t["tracker"] += (s1 - s0) * 1000

            ball_state, ball = session.ball_tracker.update(raw_ball)
            s2 = time.perf_counter(); t["ball_kalman"] += (s2 - s1) * 1000

            carrier = session.carrier_eng.update(player_dets.players, ball_state, ball)
            s3 = time.perf_counter(); t["carrier"] += (s3 - s2) * 1000

            label, adjustments = session.pass_track.update(carrier, session.proc_idx)
            session.proc_idx += 1
            t["pass"] += (time.perf_counter() - s3) * 1000

            counters.add_label(label)

            for kind, team_id, n in adjustments:
                prior_n, current_n = session.split_adjustment(n)
                counters.apply_adjustment(kind, team_id, current_n)
                if prior_n > 0 and session.last_written is not None:
                    prev_half, prev_minute = session.last_written
                    if correction is not None:
                        # At most one in-flight pass can span the boundary, so a
                        # second boundary correction should be impossible.
                        log.error("[%s] second boundary correction in one clip — "
                                  "dropping it (h%d m%d)", session.match_id, half, minute)
                    else:
                        correction = PriorCorrection(
                            half=prev_half, minute=prev_minute,
                            kind=kind, team_id=team_id, frames=prior_n,
                        )
            n_processed += 1

        win_frames.clear()
        win_fidx.clear()

    fidx = 0
    try:
        while True:
            d0 = time.perf_counter()
            ret, frame = cap.read()
            t["decode"] += (time.perf_counter() - d0) * 1000
            if not ret:
                break
            if fidx % frame_step != 0:
                fidx += 1
                continue

            win_frames.append(frame)
            win_fidx.append(fidx)
            if len(win_frames) >= window_k:
                flush_window()

            fidx += 1

        flush_window()   # process the trailing partial window
    finally:
        cap.release()

    # --- events resolved during this clip → rows for the events table
    event_rows = []
    for evt in session.new_events():
        event_rows.append(EventRow(
            half=half, minute=minute,
            frame_idx=evt.end_frame,
            kind=KIND_MAP[evt.kind],
            from_team=evt.from_team_id, to_team=evt.to_team_id,
            from_track=evt.from_track_id, to_track=evt.to_track_id,
            travel_frames=evt.travel_frames,
        ))
        counters.count_event(evt.kind, evt.from_team_id)

    session.finish_clip(half, minute)

    if n_processed:
        total_s = sum(t.values()) / 1000.0
        log.debug(
            "[%s] h%d m%d timing: %d frames | total %.1fs | per-frame ms: "
            "decode=%.1f player_det=%.1f team_clf=%.1f ball_det=%.1f "
            "tracker=%.1f ball_kalman=%.1f carrier=%.1f pass=%.1f",
            session.match_id, half, minute, n_processed, total_s,
            t["decode"]      / n_processed, t["player_det"] / n_processed,
            t["team_clf"]    / n_processed, t["ball_det"]   / n_processed,
            t["tracker"]     / n_processed, t["ball_kalman"]/ n_processed,
            t["carrier"]     / n_processed, t["pass"]       / n_processed,
        )

    return ClipResult(
        minute_row=counters.to_minute_row(
            session.match_id,
            half,
            minute,
            clip_blob_path,
            clip_duration_seconds,
        ),
        correction=correction,
        events=event_rows,
    )
