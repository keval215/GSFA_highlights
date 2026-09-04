"""
service/clip_processor.py — drives ONE 60 s clip through the existing
pipeline classes (modules/detectors, modules/team_classifier,
modules/tracking, modules/possession — no CV logic rewritten, no rendering).

Frames (15 fps stride) are processed in windows of CLIP_BATCH_WINDOW:
    Pass 1 (batched GPU, stateless): unified detect (players + ball) +
        SigLIP team embed for the whole window in one call each.
    Pass 2 (strictly sequential, stateful): track → ball Kalman →
        CarrierEngine → PassEventTracker → bucket the possession label and
        any retroactive adjustments into this minute's counters.
The two passes are equivalent to the old per-frame loop (same frames, same
models, same order into the stateful stages) — only the GPU work is batched.

When config.CLIP_PIPELINE_THREADED is set, decode + Pass 1 run on a single
daemon producer thread feeding a bounded queue while the main thread consumes
and runs Pass 2 — a pure reordering of *when* work happens, not *what*. The
env var defaults off; the serial path above is byte-for-byte unchanged.

Adjustments that resolve in this clip but whose travel frames started in
the previous clip are split via session.split_adjustment(): the
prior-minute share becomes a PriorCorrection (one UPDATE to the previous
row), the rest is applied to the in-memory counters before the row is
written.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import cv2

from service import config
from service.session import MatchSession
from modules.possession import best_ball
from service.stats import (
    KIND_MAP,
    ClipResult,
    EventRow,
    MinuteCounters,
    PriorCorrection,
    orient_for_team_a,
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
    session.ensure_gk_ready()

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise IOError(f"process_clip: cannot open {clip_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, round(fps / config.TARGET_PROCESS_FPS))
    window_k   = max(1, config.CLIP_BATCH_WINDOW)

    counters   = MinuteCounters()
    correction: PriorCorrection | None = None
    # The orientation (session.team_a_cluster_id) that was active when the
    # row `correction` targets (session.last_written) was itself written —
    # captured at the moment `correction` is built, before session.finish_clip()
    # below overwrites session.last_written_team_a_cluster_id with THIS
    # clip's own value. See stats.orient_for_team_a's docstring.
    correction_team_a_cluster_id: int | None = None

    # Per-stage timers (ms, accumulated over the whole clip).
    t = {"decode": 0.0, "player_det": 0.0, "team_clf": 0.0, "ball_det": 0.0,
         "tracker": 0.0, "ball_kalman": 0.0, "carrier": 0.0, "pass": 0.0}
    n_processed = 0

    # Processed frames buffered until a window is full, then run as one batch.
    win_frames: list = []
    win_fidx:   list[int] = []

    def _pass1(frames: list, fidxs: list[int]) -> tuple[list, list]:
        """Pass 1 — batched, stateless GPU inference for one window. Returns
        (dets_list, balls). Touches only the stateless model objects
        (player_det / team_clf / gk_det) and the per-stage timer dict, so it
        is safe to run on the producer thread when config.CLIP_PIPELINE_THREADED
        is on (its outputs are plain dataclasses / numpy by the time they
        return — no live GPU tensor crosses the thread boundary)."""
        p0 = time.perf_counter()
        dets_list = session.player_det.detect_batch(frames, fidxs, fps)
        p1 = time.perf_counter(); t["player_det"] += (p1 - p0) * 1000

        session.team_clf.classify_batch(frames, dets_list)
        if session.gk_det is not None:
            for frame, dets in zip(frames, dets_list):
                session.gk_det.classify(frame, dets)
        p2 = time.perf_counter(); t["team_clf"] += (p2 - p1) * 1000

        # Ball now comes from the same unified detection pass: pick the best ball
        # per frame from dets_list. No separate model and no stride — the ball is
        # available every processed frame, which improves Kalman continuity.
        balls = [best_ball(d) for d in dets_list]
        p3 = time.perf_counter(); t["ball_det"] += (p3 - p2) * 1000
        return dets_list, balls

    def _pass2(frames: list, dets_list: list, balls: list) -> None:
        """Pass 2 — strictly sequential, stateful logic in strict frame order,
        identical semantics to the old per-frame loop. Always runs on the main
        thread: single writer for tracker / ball_tracker / carrier_eng /
        pass_track / proc_idx / counters / the boundary-correction bookkeeping."""
        nonlocal correction, correction_team_a_cluster_id, n_processed
        for frame, player_dets, raw_ball in zip(frames, dets_list, balls):
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
                        correction_team_a_cluster_id = session.last_written_team_a_cluster_id
            n_processed += 1

    def flush_window() -> None:
        """One window, serially: Pass 1 (batched GPU) then Pass 2 (sequential
        stateful) — the non-threaded path, unchanged from the original loop."""
        if not win_frames:
            return
        dets_list, balls = _pass1(win_frames, win_fidx)
        _pass2(win_frames, dets_list, balls)
        win_frames.clear()
        win_fidx.clear()

    if not config.CLIP_PIPELINE_THREADED:
        # --- Serial path: decode + Pass 1 + Pass 2 all on this thread. No
        # thread is created; behaviour is byte-for-byte identical to the
        # pre-threading version.
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
    else:
        # --- Threaded path: one daemon producer runs the decode loop + Pass 1
        # (both GIL-releasing) and feeds windows through a small bounded queue;
        # this (main) thread consumes and runs Pass 2. maxsize=2 bounds RAM to
        # ~2 windows in flight. Gated by config.CLIP_PIPELINE_THREADED so it can
        # be switched off in prod without a redeploy.
        q: queue.Queue = queue.Queue(maxsize=2)
        stop_event = threading.Event()
        producer_error: list[BaseException] = []

        def _producer() -> None:
            frames: list = []
            fidxs: list[int] = []
            fidx = 0
            try:
                while not stop_event.is_set():
                    d0 = time.perf_counter()
                    ret, frame = cap.read()
                    t["decode"] += (time.perf_counter() - d0) * 1000
                    if not ret:
                        if frames:                       # one trailing partial window
                            dets_list, balls = _pass1(frames, fidxs)
                            q.put((frames, fidxs, dets_list, balls))
                        break
                    if fidx % frame_step == 0:           # exact original stride semantics
                        frames.append(frame)
                        fidxs.append(fidx)
                        if len(frames) >= window_k:
                            dets_list, balls = _pass1(frames, fidxs)
                            q.put((frames, fidxs, dets_list, balls))
                            frames, fidxs = [], []
                    fidx += 1
            except BaseException as exc:   # noqa: BLE001 — captured, re-raised on main thread
                producer_error.append(exc)
            finally:
                # cap is owned end-to-end by this thread now; release it here
                # and nowhere else on the threaded path. The sentinel is ALWAYS
                # sent (clean end, ret False, or exception) so the consumer
                # never blocks on an empty queue forever.
                cap.release()
                q.put(None)

        producer = threading.Thread(
            target=_producer, name=f"clip-producer-{session.match_id}", daemon=True
        )
        try:
            producer.start()
        except Exception:
            # The producer's finally (which releases cap) never runs if the
            # thread never started — this is the one threaded path where the
            # main thread still owns cap.
            cap.release()
            raise

        producer_wedged = False
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                win_frames_i, _win_fidx_i, dets_list, balls = item
                _pass2(win_frames_i, dets_list, balls)
        finally:
            # Shut the producer down on EVERY exit path (clean break, a Pass 2
            # exception, KeyboardInterrupt). Keep draining while it is alive so a
            # producer blocked on a full q.put() — or one still mid-Pass-1 that
            # refills the queue after a single drain pass — always unblocks,
            # reaches its stop_event check and exits. Bounded to ~30s total.
            # MatchSessionManager reuses this session for the match's next clip,
            # so a leaked producer still holding session.player_det /
            # session.team_clf would race that clip's main-thread work.
            stop_event.set()
            deadline = time.monotonic() + 30.0
            while True:
                try:
                    while True:
                        q.get_nowait()
                except queue.Empty:
                    pass
                producer.join(timeout=0.5)
                if not producer.is_alive():
                    break
                if time.monotonic() >= deadline:
                    log.error(
                        "[%s] clip-producer thread still alive 30s after stop "
                        "signal — leaking; session models may be unsafe to reuse "
                        "on this match's next clip", session.match_id,
                    )
                    producer_wedged = True
                    break

        if producer_error:
            # Re-raise the producer's original exception (with its traceback)
            # now that the thread is joined, so worker.py's catch-all sees it
            # exactly as it would on the serial path.
            raise producer_error[0]
        if producer_wedged:
            # Only reached on the clean consumer path — a Pass 2 exception is
            # already propagating and takes precedence. Fail the clip rather
            # than return a ClipResult while a leaked producer thread may still
            # be touching this session's shared model objects.
            raise RuntimeError(
                f"[{session.match_id}] clip-producer thread did not stop after "
                f"30s — aborting clip to avoid a cross-clip model race"
            )

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
        # NB: with config.CLIP_PIPELINE_THREADED, decode/player_det/team_clf/
        # ball_det accrue on the producer thread and tracker/ball_kalman/carrier/
        # pass on this one; they ran concurrently, so `total` is the sum of both
        # threads' stage time, not wall-clock (which is closer to max(producer,
        # consumer) + queue wait). The per-stage figures are still valid.
        total_s = sum(t.values()) / 1000.0
        log.info(
            "[%s] h%d m%d timing: %d frames | total %.1fs | per-frame ms: "
            "decode=%.1f player_det=%.1f team_clf=%.1f ball_det=%.1f "
            "tracker=%.1f ball_kalman=%.1f carrier=%.1f pass=%.1f",
            session.match_id, half, minute, n_processed, total_s,
            t["decode"]      / n_processed, t["player_det"] / n_processed,
            t["team_clf"]    / n_processed, t["ball_det"]   / n_processed,
            t["tracker"]     / n_processed, t["ball_kalman"]/ n_processed,
            t["carrier"]     / n_processed, t["pass"]       / n_processed,
        )

    result = ClipResult(
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
    # Colour-anchored orientation, applied once at write time so a mid-match
    # re-fit's cluster-order swap never has to be undone at read time. The
    # correction (if any) targets a DIFFERENT, already-written row, so it is
    # reoriented against that row's own orientation, not this clip's.
    return orient_for_team_a(result, session.team_a_cluster_id, correction_team_a_cluster_id)
