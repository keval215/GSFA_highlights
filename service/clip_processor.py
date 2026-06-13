"""
service/clip_processor.py — drives ONE 60 s clip through the existing
pipeline classes (video_analysis/, detectors/, team_classifier/,
tracking/ — no CV logic rewritten, no rendering).

Per processed frame (15 fps stride):
    detect → team classify → track → ball Kalman → CarrierEngine
    → PassEventTracker → bucket the possession label and any
    retroactive adjustments into this minute's counters.

Adjustments that resolve in this clip but whose travel frames started in
the previous clip are split via session.split_adjustment(): the
prior-minute share becomes a PriorCorrection (one UPDATE to the previous
row), the rest is applied to the in-memory counters before the row is
written.
"""

from __future__ import annotations

import logging

import cv2

from service import config
from service.session import MatchSession
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

    counters   = MinuteCounters()
    correction: PriorCorrection | None = None

    fidx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fidx % frame_step != 0:
                fidx += 1
                continue

            # --- detection + attribution (state continues from previous clip)
            player_dets = session.models.player_det.detect(frame, fidx, fps)
            session.team_clf.classify(frame, player_dets)
            session.tracker.update(frame, player_dets.players)

            raw_ball         = session.models.ball_det.detect(frame)
            ball_state, ball = session.ball_tracker.update(raw_ball)

            carrier = session.carrier_eng.update(player_dets.players, ball_state, ball)

            label, adjustments = session.pass_track.update(carrier, session.proc_idx)
            session.proc_idx += 1

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

            fidx += 1
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

    return ClipResult(
        minute_row=counters.to_minute_row(session.match_id, half, minute, clip_blob_path),
        correction=correction,
        events=event_rows,
    )
