# `service/`

The production cloud service. It wraps the shared CV pipeline (from `detectors/`,
`team_classifier/`, `tracking/`, `video_analysis/`) into a real-time, clip-by-clip system
backed by Azure Blob + Queue + SQL, with an HTTP callback to the main app.

**Two processes, one Docker image:**
- **API** (`api.py`, no GPU) — ingests clips, enqueues work, serves health/metrics.
- **Worker** (`worker.py`, GPU) — pulls the queue, runs the pipeline, writes SQL, calls back.

See also: [docs/API.md](../../API.md) (HTTP contract + env vars) and
[ARCHITECTURE.md §3](../ARCHITECTURE.md).

| File | Role | GPU? |
|---|---|---|
| `config.py` | All environment variables in one place; fail-fast on missing required vars | — |
| `api.py` | FastAPI ingest (`POST /api/clips`, `POST /post-processing`, `GET /health`, `GET /metrics`) | no |
| `worker.py` | Queue poll loop; orchestrates one clip end-to-end | yes |
| `session.py` | `MatchSession` (cross-clip state) + team-fit logic + session manager | yes |
| `clip_processor.py` | Drives ONE clip through the pipeline (batched two-pass) | yes |
| `stats.py` | Dependency-free data shapes + per-minute counting/correction logic | — |
| `db.py` | pyodbc layer for Azure SQL (the per-clip transaction) | — |
| `blob.py` | Azure Blob wrapper (upload/download/delete clips) | — |
| `queueing.py` | Azure Queue wrapper (enqueue/dequeue/defer/poison) | — |
| `notifier.py` | Outbox sender — POSTs cumulative stats to the advance-stats endpoint | — |
| `__init__.py` | Package marker | — |

---

## Per-clip life of a request (the orchestration)

`worker.py::Worker._handle(msg)` is the spine. Order matters:

1. **Poison guard** — `dequeue_count > MAX_DEQUEUE_COUNT` ⇒ `queue.move_to_poison`.
2. **Idempotency** — `db.minute_exists(...)` ⇒ drop message + delete blob (replay-safe).
3. **Progress + ordering** — `db.get_match_progress`; `stats.is_expected(...)`. Out of
   order ⇒ `queue.defer(...)` up to `ORDERING_RETRIES`, then process-anyway with a gap log.
4. **Session** — `manager.get_or_create(match_id)`; download blob to a job dir.
5. **Team fit** — if `session.fit_status != "ok"`, `session.ensure_fit(clip_path)`.
6. **Process** — `clip_processor.process_clip(session, clip_path, half, minute, blob_path)`.
7. **Persist** — `db.write_clip_result(conn, minute_row, correction, events, team_names)`
   — a single transaction.
8. **Callback** — `notifier.send_pending_for_match(conn, match_id)`.
9. **Cleanup** — delete queue message + blob + job dir; `manager.evict_idle()`.

Between iterations the worker writes a heartbeat JSON (`_heartbeat`) that `api.py`'s
`/health` and `/metrics` read.

---

## `config.py`
- Centralises every env var. `_required(name)` raises a clear error at startup if a
  required var is missing (so failures aren't buried mid-request).
- Required: `AZURE_STORAGE_CONNECTION_STRING`, `SQL_CONN_STR`, `PLAYER_WEIGHTS`.
- Optional tunables with defaults: `TARGET_PROCESS_FPS` (15), `CMC_METHOD` (ecc),
  `CLIP_BATCH_WINDOW` (16), `FIT_SAMPLE_EVERY` (5), `FIT_SILHOUETTE_MIN` (0.20),
  queue/ordering/callback retry knobs, `SESSION_IDLE_EVICT_S`, paths, etc.
- **Does NOT** hardcode secrets; everything comes from `/etc/gsfa-highlights.env` on the
  VM (loaded by docker-compose `env_file`). `PLAYER_WEIGHTS` must be the **bare path** to
  the weights — a value like `PLAYER_WEIGHTS=/path/...` (name duplicated) is the classic
  misconfig that crashes the worker at model load.

## `api.py` (FastAPI, no GPU)
- `POST /api/clips` (multipart): hygiene checks (content type, size cap), `db.ensure_match`
  (first clip creates the match; later clips fill missing team metadata via COALESCE),
  resolve/claim the minute (`db.claim_next_minute` if not supplied), dedupe
  (`minute_exists` or blob already exists ⇒ `202 duplicate`), else `blob.upload_stream` +
  `queue.enqueue` ⇒ `202`.
- `POST /post-processing` (multipart): accepts a whole-match video, stores it at
  `clips/<match_id>/post_processing.mp4`, enqueues a background job, and returns `200`
  once the upload is fully received. The worker deletes the blob after processing.
- `GET /health` — 200 if the worker heartbeat is < 300 s old, else 503.
- `GET /metrics` — queue depths, last-clip seconds, seconds-behind-live, GPU mem, etc.
- **Does NOT** process clips inline, touch the GPU, or render — it returns in ~1–2 s.
- An access-log filter drops 404 spam from internet scanners.

## `worker.py` (GPU)
- `Worker.__init__` loads everything once: queue, blob, `ModelBundle.load()` (the unified
  detector in VRAM), `MatchSessionManager`, a SQL connection.
- `run_forever()` polls every ~2 s; on any loop exception it logs, **reconnects SQL**, and
  continues (resilient to transient DB drops).
- Writes a heartbeat each loop (GPU visibility/mem, last clip seconds, queue depth →
  seconds-behind-live, active matches).
- **Does NOT** parallelise across clips — one clip at a time (simple, ~1 concurrent match
  expected). **Does NOT** render video.
- Routes `kind = post_processing` queue messages through the whole-match path and writes
  the new `post_processing` SQL table.

## `session.py`
- **`ModelBundle`** — process-wide models loaded once (currently just the unified
  `PlayerDetector`). The team classifier is per-match, not here.
- **`MatchSession`** — all cross-clip state for one match: `tracker`, `ball_tracker`,
  `carrier_eng`, `pass_track`, `proc_idx`, `team_clf`, `fit_status`, the carryover for a
  pass spanning a clip boundary, and `last_written`. On construction it reloads the team
  fit pkl from `MATCH_STATE_DIR/<match_id>/team_siglip.pkl` if present (survives restarts).
  - `ensure_fit(clip)` — clip-1 dense fit with a **silhouette quality guard**
    (`fit_and_score`): below `FIT_SILHOUETTE_MIN` ⇒ keep clip-1 crops and **refit on clip
    2** with combined samples; never refit after committing (re-running KMeans could swap
    the 0/1 labels mid-match). Commits the pkl + resolves team names.
  - `split_adjustment(n)` / `finish_clip(half, minute)` / `new_events()` — the boundary
    bookkeeping used by `clip_processor`.
- **`MatchSessionManager`** — `dict[match_id → MatchSession]` with idle eviction.
- Uses the team classifier's **static** crop helpers (`_torso_crop`, `_is_sharp`) in its
  own `collect_crops`; it does **NOT** call `GSFATeamClassifier.fit_from_video`.
- **Asserts at import** that `stats.py`'s label/event strings equal
  `video_analysis.possession`'s — the cross-module contract guard.

## `clip_processor.py`
- `process_clip(session, clip_path, half, minute, clip_duration_seconds, clip_blob_path=None)`
  decodes the clip, strides to `TARGET_PROCESS_FPS`, and processes in
  windows of `CLIP_BATCH_WINDOW` frames:
  - **Pass 1 (batched, stateless GPU):** `player_det.detect_batch` + `team_clf.classify_batch`
    + `best_ball` per frame.
  - **Pass 2 (strictly sequential, stateful):** `tracker.update` → `ball_tracker.update`
    → `carrier_eng.update` → `pass_track.update`; bucket the label into `MinuteCounters`
    and apply adjustments. **Semantically identical** to the old per-frame loop — only the
    GPU work is batched.
- Splits boundary-spanning adjustments via `session.split_adjustment`: the prior-minute
  share becomes a `PriorCorrection` (one UPDATE to the previous row), the rest hits this
  minute's counters.
- Builds `EventRow`s from `session.new_events()`; returns a `ClipResult`
  (`minute_row`, `correction`, `events`). `clip_duration_seconds` (the client-supplied
  clip length) is threaded straight through into `MinuteCounters.to_minute_row(...)` and
  stored on `MinuteRow.clip_duration_seconds` — used only for the CSV export
  (`scripts/get_csv.py`), not for any pipeline math.
- **Does NOT** render, write SQL, or send callbacks — it returns plain data the worker
  persists. Per-stage timing is logged at DEBUG.

## `post_processing/post_processing.py`
- Thin whole-match wrapper around the existing pipeline. It reuses `process_clip(...)`
  over the full uploaded video, so the same detector, team classifier, tracker, ball
  Kalman, carrier, and pass FSM logic is used without duplicating the CV path.
- The worker writes the resulting aggregate counters to the `post_processing` table and
  deletes the blob after the SQL commit.

## `stats.py` (dependency-free, unit-tested)
- Pure stdlib data shapes: `MinuteRow`, `EventRow`, `PriorCorrection`, `MinuteCounters`,
  `ClipResult`; the label/event constants (mirroring `possession.py`); `KIND_MAP`.
- `MinuteCounters` does the per-minute bucketing (`add_label`), correction
  (`apply_adjustment` — `flip_to`/`drop`, clamped), and event tallying (`count_event`).
- Helpers: `split_adjustment(n, carryover)`, `is_expected(...)` (ordering guard),
  `build_payload(...)` (the flat advance-stats body; **positional** team mapping 0→a, 1→b).
- Deliberately torch/boxmot/pyodbc-free so the correctness-critical math is testable
  (`tests/test_stats.py`).

## `db.py` (Azure SQL via pyodbc)
- **Cumulative-on-read:** `minute_stats` stores raw per-minute counters; `cumulative_read`
  `SUM()`s over rows ≤ the current minute. Never stores cumulative numbers.
- **`write_clip_result(...)` = one transaction:** apply prior correction (revision += 1) →
  upsert this minute row (PK = `(match_id, half, minute)` ⇒ replay overwrites, now also
  storing `clip_duration_seconds`) → insert events → advance match progress → insert the
  **outbox** row with a fresh cumulative payload. Rolls back on any error.
- Match upsert (`ensure_match`, COALESCE so later clips can't overwrite), atomic minute
  claim (`claim_next_minute` via `UPDATE...OUTPUT`), `minute_exists`, `get_match_progress`,
  `get_team_specs` (only if both teams have name **and** colour). Outbox ops
  (`fetch_pending`, `mark_sent`, `mark_failed`) for the notifier.
- `post_processing_exists(conn, match_id)` / `write_post_processing_result(...)` — dedupe
  check and upsert for the new whole-match `post_processing` table, keyed by `match_id`;
  stores the team metadata snapshot alongside the raw counters. Unlike `write_clip_result`,
  this does **not** touch `callback_outbox` — no advance-stats callback is sent for
  whole-match uploads.

## `blob.py` / `queueing.py` (Azure adapters)
- `blob.py` — `ClipBlobStore`: container ensure/exists/upload/download/delete. Blob layout
  `<container>/<match_id>/<half>_<minute>.mp4`, plus `post_processing_blob_name(match_id)`
  → `<container>/<match_id>/post_processing.mp4` for the whole-match upload path. Delete
  is best-effort (orphan is harmless).
- `queueing.py` — `ClipQueue`: enqueue/dequeue/`delete`/`defer`/`move_to_poison`/`depths`.
  `ClipMessage.kind` is `"clip"` (default) or `"post_processing"`; `enqueue(...)` takes
  `clip_duration_seconds` and an optional `kind` + team metadata, and
  `enqueue_post_processing(...)` is a thin wrapper that always sends `kind=post_processing`
  with just `match_id`/`blob_path`/team fields (no half/minute). Carries `ordering_retries`
  in the message body across re-sends; malformed messages go straight to poison. Azure
  Queue is only **approximately** FIFO — ordering is enforced in `worker.py`, not here.

## `notifier.py` (outbox sender)
- After commit, POSTs each pending `callback_outbox` row (strict `(half, minute)` order)
  to `{CALLBACK_URL}/v1/pvt/tournament-duelz/{match_id}/advance-stats` with the
  `X-Super-Admin-Key` header. Each POST is a **full overwrite**; server replies 204.
- Retries `CALLBACK_RETRIES` times with exponential backoff; `401` fails fast.
  Exhaustion ⇒ mark `failed`, log loudly, **continue** (SQL stays source of truth).
- If `CALLBACK_URL`/`SUPER_ADMIN_KEY` are unset, rows stay `pending` (nothing lost).
- **Does NOT** compute stats — it just ships the payload `db.py` already built.

---

## What the service as a whole does NOT do
- No authentication on the API in v1 — access is restricted at the network level (Azure
  NSG on port 8000).
- No video rendering / highlight cutting — outputs are SQL rows + the events log + the
  callback. The `events` table is the future highlight-reel source.
- No multi-worker scaling / distributed locking — single worker, one clip at a time.
- No contested-possession or dribble events (matches the pipeline's design).
