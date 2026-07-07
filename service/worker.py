"""
service/worker.py — GPU worker: queue poll loop, one clip at a time.

Flow per clip (implementation.md §6):
  1. Dequeue (visibility 90 s; dequeue_count > 3 ⇒ poison).
  2. Ordering guard: expected = next (half, minute) after the match's
     last processed clip. Mismatch ⇒ defer with a short delay, max 3
     tries, then process anyway and log the gap.
  3. MatchSession get-or-create; clip 1 ⇒ team fit (dense sampling,
     silhouette guard; below threshold ⇒ refit on clip 2 with combined
     samples).
  4. Download blob, process at 15 fps stride continuing cross-clip state.
  5/6. One SQL transaction: minute row + prior-minute correction +
     events + match progress + outbox row.
  7. Notifier sends pending outbox rows in order (3× backoff,
     exhausted ⇒ failed + continue).
  8. Cleanup: delete clip blob + local job dir; evict idle sessions.

Run:  python -m service.worker
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import warnings

from service import config, db, logging_setup, notifier
from service.blob import ClipBlobStore
from service.clip_processor import process_clip
from service.post_processing.post_processing import process_match_video
from service.queueing import ClipMessage, ClipQueue
from service.session import MatchSession, MatchSessionManager, ModelBundle
from service.stats import is_expected

log = logging.getLogger("gsfa.worker")

_IDLE_SLEEP_S = 2.0


class Worker:
    def __init__(self) -> None:
        config.MATCH_STATE_DIR.mkdir(parents=True, exist_ok=True)
        config.JOBS_DIR.mkdir(parents=True, exist_ok=True)
        self.queue   = ClipQueue()
        self.queue.ensure_queues()
        self.blob    = ClipBlobStore()
        self.models  = ModelBundle.load()
        self.manager = MatchSessionManager(self.models)
        self.conn    = db.get_conn()
        self._last_clip_seconds: float | None = None
        self._last_dequeue_count = 0
        self._last_post_processing_error: dict | None = None

    # ------------------------------------------------------------------

    def run_forever(self) -> None:
        log.info("worker started (device=%s, fps=%s)", config.DEVICE, config.TARGET_PROCESS_FPS)
        while True:
            try:
                self.manager.evict_idle()
                msg = self.queue.dequeue()
                if msg is None:
                    self._heartbeat()
                    time.sleep(_IDLE_SLEEP_S)
                    continue
                self._handle(msg)
            except Exception:
                log.exception("worker loop error — reconnecting SQL and continuing")
                self._reconnect()
                time.sleep(_IDLE_SLEEP_S)
            self._heartbeat()

    # ------------------------------------------------------------------

    def _handle(self, msg: ClipMessage) -> None:
        self._last_dequeue_count = msg.dequeue_count

        if msg.kind == "post_processing":
            self._handle_post_processing(msg)
            return

        if msg.dequeue_count > config.MAX_DEQUEUE_COUNT:
            log.error("POISON clip %s h%d m%d (dequeue_count=%d)",
                      msg.match_id, msg.half, msg.minute, msg.dequeue_count)
            self.queue.move_to_poison(msg)
            return

        # Idempotency: a replayed message for an already-written minute is dropped.
        if db.minute_exists(self.conn, msg.match_id, msg.half, msg.minute):
            log.info("clip %s h%d m%d already processed — dropping message",
                     msg.match_id, msg.half, msg.minute)
            self.queue.delete(msg)
            self.blob.delete(msg.blob_path)
            return

        progress = db.get_match_progress(self.conn, msg.match_id)
        if progress is None:
            # api normally creates the match row; cover the gap anyway.
            db.ensure_match(self.conn, msg.match_id)
            progress = (1, 0)
        last_half, last_minute = progress

        # --- Ordering guard (Azure Queue Storage is only approximately FIFO)
        if not is_expected(last_half, last_minute, msg.half, msg.minute):
            if msg.ordering_retries < config.ORDERING_RETRIES:
                log.warning("clip %s h%d m%d out of order (expected after h%d m%d) — "
                            "deferring (%d/%d)", msg.match_id, msg.half, msg.minute,
                            last_half, last_minute,
                            msg.ordering_retries + 1, config.ORDERING_RETRIES)
                self.queue.defer(msg, config.ORDERING_RETRY_DELAY)
                return
            log.error("clip %s h%d m%d still out of order after %d retries — "
                      "processing anyway (GAP after h%d m%d)", msg.match_id, msg.half,
                      msg.minute, config.ORDERING_RETRIES, last_half, last_minute)

        session  = self.manager.get_or_create(msg.match_id)
        job_dir  = config.JOBS_DIR / msg.match_id
        clip_path = job_dir / f"{msg.half}_{msg.minute}.mp4"
        self.blob.download_to(msg.blob_path, clip_path)

        t_start = time.monotonic()

        # --- Team fit (clip 1, or quality-guard refit on clip 2)
        if session.fit_status != "ok":
            session.ensure_fit(str(clip_path))

        # --- Process the clip with cross-clip state
        result = process_clip(session, str(clip_path), msg.half, msg.minute,
                              msg.clip_duration_seconds,
                              clip_blob_path=msg.blob_path)

        # --- One SQL transaction (minute row + correction + events + outbox)
        team_names = session.team_clf.team_id_to_name if session.team_clf else None
        db.write_clip_result(self.conn, result.minute_row, result.correction,
                             result.events, team_names)

        self._last_clip_seconds = round(time.monotonic() - t_start, 1)
        log.info("clip %s h%d m%d processed in %.1fs (events=%d, correction=%s)",
                 msg.match_id, msg.half, msg.minute, self._last_clip_seconds,
                 len(result.events), "yes" if result.correction else "no")

        # --- Callbacks (after commit, strict order, best-effort)
        notifier.send_pending_for_match(self.conn, msg.match_id)

        # --- Cleanup
        self.queue.delete(msg)
        self.blob.delete(msg.blob_path)
        shutil.rmtree(job_dir, ignore_errors=True)

    def _handle_post_processing(self, msg: ClipMessage) -> None:
        # Delete immediately — no lease renewal, no automatic retry. A whole-match
        # job that dies partway through (in-process exception, container OOM-kill,
        # VM shutdown) must never be silently redelivered and reprocessed on top
        # of leftover state; recovery for a failed job is a manual re-upload.
        self.queue.delete(msg)

        if db.post_processing_exists(self.conn, msg.match_id):
            log.info("post-processing %s already processed — dropping message", msg.match_id)
            self.blob.delete(msg.blob_path)
            return

        db.ensure_match(self.conn, msg.match_id,
                        msg.team0_name, msg.team1_name,
                        msg.team0_colour, msg.team1_colour)

        # Own session, constructed directly (never through MatchSessionManager,
        # never stored there) — isolated from the live-clip path for this match
        # and never reused across attempts, so every run starts from clean
        # tracker/ball/carrier/pass-FSM state.
        session = MatchSession(msg.match_id, self.models)
        job_dir = config.JOBS_DIR / msg.match_id
        video_path = job_dir / "post_processing.mp4"

        t_start = time.monotonic()
        try:
            self.blob.download_to(msg.blob_path, video_path)
            result = process_match_video(session, str(video_path), blob_path=msg.blob_path)

            db.write_post_processing_result(
                self.conn,
                result.minute_row,
                team0_name=msg.team0_name,
                team1_name=msg.team1_name,
                team0_colour=msg.team0_colour,
                team1_colour=msg.team1_colour,
                video_blob_path=msg.blob_path,
            )

            self._last_clip_seconds = round(time.monotonic() - t_start, 1)
            log.info("post-processing %s completed in %.1fs (events=%d)",
                     msg.match_id, self._last_clip_seconds, len(result.events))
        except Exception as exc:
            elapsed = round(time.monotonic() - t_start, 1)
            log.exception(
                "post-processing FAILED match=%s blob=%s elapsed=%.1fs — %s",
                msg.match_id, msg.blob_path, elapsed, exc,
            )
            self._last_post_processing_error = {
                "match_id": msg.match_id,
                "error": str(exc),
                "at": time.time(),
            }
        finally:
            self.blob.delete(msg.blob_path)
            shutil.rmtree(job_dir, ignore_errors=True)

    # ------------------------------------------------------------------

    def _reconnect(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
        try:
            self.conn = db.get_conn()
        except Exception:
            log.exception("SQL reconnect failed — will retry next loop")

    def _heartbeat(self) -> None:
        gpu_visible = False
        gpu_mem_mb  = None
        try:
            import torch
            gpu_visible = torch.cuda.is_available()
            if gpu_visible:
                gpu_mem_mb = round(torch.cuda.memory_allocated() / 1024**2)
        except Exception:
            pass
        try:
            depth, _ = self.queue.depths()
        except Exception:
            depth = None
        hb = {
            "ts": time.time(),
            "gpu_visible": gpu_visible,
            "gpu_mem_mb": gpu_mem_mb,
            "last_clip_seconds": self._last_clip_seconds,
            "last_dequeue_count": self._last_dequeue_count,
            "active_matches": self.manager.active_matches,
            # 1 clip ≈ 1 minute of match time; queue depth approximates lag.
            "seconds_behind_live": depth * 60 if depth is not None else None,
            "last_post_processing_error": self._last_post_processing_error,
        }
        try:
            config.HEARTBEAT_FILE.write_text(json.dumps(hb), encoding="utf-8")
        except OSError:
            pass


def main() -> None:
    # IST timestamps + azure HTTP-logging silenced, shared with the api process.
    logging_setup.configure("worker")
    # Benign: boxmot cosine-distance on zero-vector embeddings (players without
    # a SigLIP embedding) yields NaN, which boxmot masks. Don't spam the log.
    warnings.filterwarnings("ignore", category=RuntimeWarning,
                            message="invalid value encountered in divide")
    Worker().run_forever()


if __name__ == "__main__":
    main()
