# Code Review — `08a2639..HEAD`

Scope: whole-match `/post-processing` endpoint, `clip_duration_seconds` field, `scripts/get_csv.py`, and the detector/tracker `smoothed_bbox`/`classes` changes.

## Root cause behind most high-severity findings

The whole-match `/post-processing` path was bolted onto infrastructure built for per-minute live clips (`process_clip` / `MatchSession` / `ClipMessage`) using hardcoded `half=1, minute=1` filler and a shared session manager, instead of getting its own isolated session and dequeue-guard path. This is the root cause of findings 1–4 and 6 below — treat it as one design gap, not five independent bugs.

## Findings (most severe first)

### 1. `service/worker.py:80` — Post-processing messages bypass the poison-queue guard
Messages are dispatched and returned before the `dequeue_count > MAX_DEQUEUE_COUNT` check ever runs, so they can never be poisoned.
**Failure scenario:** A corrupt/oversized whole-match upload makes `_handle_post_processing` raise every time; `run_forever`'s blanket except logs and reconnects but never deletes the message, so Azure redelivers it forever with no dequeue-count ceiling — unlike the clip path, which quarantines after 3 tries.

### 2. `service/worker.py:163` — Shared mutable session state between live and post-processing paths
`_handle_post_processing` fetches the `MatchSession` from the same `MatchSessionManager.get_or_create(match_id)` used by the live per-clip path.
**Failure scenario:** Live clips for a match advance `session.last_written` past h1/m1; the same match later goes through `/post-processing`, which calls `process_clip(..., half=1, minute=1, ...)` and `session.finish_clip(1,1)` — overwriting `last_written`/`carryover_travel_frames` with fabricated coordinates and feeding the whole video through tracker/FSM state built from unrelated live frames. Corrupts both paths' counts with no error raised.

### 3. `service/worker.py:171` — Events and correction data discarded for whole-match uploads
`process_match_video` returns a full `ClipResult` (minute_row, correction, events), but `_handle_post_processing` only forwards `result.minute_row` to `db.write_post_processing_result`. `result.events` is used only for a log `len()` at line 183.
**Failure scenario:** Every pass/interception/ball-lost event detected across a whole-match upload never reaches the `events` table, even though the identical pipeline computes them for live clips. Anyone building highlights or event review off `events` for a `/post-processing`-ingested match finds it empty.

### 4. `service/worker.py:179` — No advance-stats callback fires for whole-match uploads
`_handle_post_processing` never calls `notifier.send_pending_for_match`, and `write_post_processing_result` never inserts into `callback_outbox`.
**Failure scenario:** A downstream consumer subscribed to the advance-stats callback (fired for every live-clip minute) receives nothing when a match is ingested via `/post-processing`; the aggregate is stored in SQL but silently never pushed.

### 5. `service/queueing.py:129` — No lease renewal for long-running post-processing jobs
The queue message's visibility timeout defaults to `QUEUE_VISIBILITY_SEC` (90s) for every message kind, including `post_processing`, and nothing renews the lease while a whole-match job runs far longer than that.
**Failure scenario:** If the worker restarts or crashes mid-job (e.g. a deploy via `deploy.yml`'s `docker rm -f`/rebuild) after the 90s lease expires, the message becomes re-claimable and the entire whole-match video is re-downloaded and reprocessed from frame 0, discarding all prior GPU work with no checkpointing.

### 6. `service/worker.py:68` — Whole-match processing blocks the single-threaded worker loop
`_handle_post_processing` runs the full whole-match CV pipeline synchronously inside the same single-threaded `run_forever` loop that also serves live 60s clips for every other in-flight match.
**Failure scenario:** A 90-minute match uploaded via `/post-processing` blocks the one worker for the entire processing duration (tens of minutes of GPU time); any live clips queued for other matches during that window sit unprocessed, growing `seconds_behind_live` for those matches even though nothing is wrong with them.

### 7. `service/session.py:209` — Team-fit quality gate never gets its second chance for whole-match uploads
`ensure_fit` is only ever called once by the whole-match `process_match_video` path (`post_processing.py:21-22`); if the silhouette score is below `FIT_SILHOUETTE_MIN` on that single call, `fit_status` is set to `"refit"` and stays there permanently — the "committing anyway" branch (line 218-227) only fires on a *second* call, which never happens for whole-match uploads.
**Failure scenario:** A whole-match video whose sampled crops give a weak team-cluster separation gets processed start-to-finish with an unvetted low-quality team split, with no error-level log (`"refit still below threshold — committing anyway"` never fires) to flag the quality problem, unlike the live-clip path which gets a second, combined-sample attempt on clip 2.

### 8. `video_analysis/possession.py:740` — Smoothed bbox computed but never used for rendering
`_to_sv` (feeding `draw_frame`'s renderer) builds detection boxes from `p.bbox`, the raw per-frame box — it never reads the new `Detection.smoothed_bbox` that `PlayerTracker` now computes specifically for jitter-free drawing.
**Failure scenario:** The Kalman-smoothed box is computed every frame at some CPU cost but has zero effect on the rendered output video — boxes still wobble exactly as before the change, contradicting the stated purpose of the field.

### 9. `detectors/player_detector.py:109` — Misleading comment about referee filtering
The comment claims "Local possession passes [0, 1, 2] to hard-filter referees," but `video_analysis/possession.py`'s actual `PlayerDetector(...)` call (line 829, the real local-run entry point) passes no `classes=` argument — only the manual debug script `team_classifier/test_team_classifier.py` uses the filter.
**Failure scenario:** A reader trusts the comment and assumes referees are already hard-filtered out of local/production possession runs; they aren't — `classes` defaults to `None` there, so referee boxes are still produced by the model call (routed to `fd.referees` by the existing confidence gate, so the practical cost today is just wasted referee-class inference, not wrong stats — but the next person extending this code will be misled).

### 10. `service/db.py:292` — Redundant existence check / unreachable UPDATE branch
`write_post_processing_result` re-runs `SELECT 1 FROM post_processing WHERE match_id = ?` to decide INSERT vs UPDATE, even though its only caller (`worker.py:153`, `db.post_processing_exists`) already confirmed the row doesn't exist a few lines earlier and returns early if it does — making the UPDATE branch (lines 297-313) unreachable in normal single-worker operation and the second query redundant.
**Failure scenario:** A wasted DB round-trip on every post-processing write, and dead code that looks load-bearing: a future maintainer may assume the UPDATE path is exercised (e.g. for retries) when it structurally never is under the current caller, making a latent bug in that branch invisible until someone calls this function from a new context.

## Lower-priority items (confirmed, not ranked above)

- `ClipMessage`'s dual-purpose dataclass carries both clip and post-processing fields.
- `defer`/`move_to_poison` hand-roll message-body dicts that can drift in shape from `enqueue`.
- `scripts/get_csv.py` double-walks `h_rows` for both minute-rows and 5-min blocks.
- `scripts/get_csv.py`'s "export all matches" mode has an N+1 query-per-match pattern.

## Process note

A background agent session in this review flagged that some of its tool results had anomalous extra content appended (fake memory/observation tool hints and an unrelated agent-type list) — a likely prompt-injection attempt. It reported not acting on any of it. Worth checking tool/MCP configuration if this recurs.
