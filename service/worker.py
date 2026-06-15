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

from service import config, db, notifier
from service.blob import ClipBlobStore
from service.clip_processor import process_clip
from service.queueing import ClipMessage, ClipQueue
from service.session import MatchSessionManager, ModelBundle
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
        }
        try:
            config.HEARTBEAT_FILE.write_text(json.dumps(hb), encoding="utf-8")
        except OSError:
            pass


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # The Azure SDK's HTTP logging policy logs every request/response at INFO,
    # which floods the log on each 2 s queue poll and buries our own logs.
    logging.getLogger("azure").setLevel(logging.WARNING)
    Worker().run_forever()


if __name__ == "__main__":
    main()
