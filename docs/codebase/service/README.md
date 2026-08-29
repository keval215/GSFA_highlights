# `service/`

The production cloud service. It wraps the shared CV pipeline (from `modules/detectors/`,
`modules/team_classifier/`, `modules/tracking/`, `modules/possession/`), parametrized per
match by a `RulesetConfig` (`rulesets/`), into a real-time, clip-by-clip system backed by
Azure Blob + Queue + SQL, with an HTTP callback to the main app.

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
   If the match row is somehow missing at this point (the API normally creates it), the
   worker creates it itself defaulting to `ruleset="futsal"` and logs a loud warning — the
   actual requested ruleset isn't carried in the clip queue message, so a genuinely
   classic match hitting this path ends up silently (but loudly-logged) wrong.
4. **Ruleset + session** — `ruleset = get_ruleset(db.get_match_ruleset(conn, match_id))`;
   `manager.get_or_create(match_id, ruleset)`; download blob to a job dir. `ruleset` is
   only used if a new session is created — an existing session keeps whatever ruleset it
   was built with.
5. **Team fit** — if `session.fit_status != "ok"`, `session.ensure_fit(clip_path)`.
6. **Process** — `clip_processor.process_clip(session, clip_path, half, minute,
   clip_duration_seconds, blob_path)` — internally calls `session.ensure_gk_ready()` first
   (lazily builds `GoalkeeperDetector` once the DB has both GK reference colours; a no-op
   once built) and runs `gk_det.classify(...)` per frame if ready.
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
- `player_weights(ruleset_name)` (was a no-arg function): `"futsal"` still reads
  `PLAYER_WEIGHTS` (backward compatible with existing deployments); any other ruleset
  name reads `<RULESET>_PLAYER_WEIGHTS` (e.g. `CLASSIC_PLAYER_WEIGHTS`), required only
  when that ruleset is actually used. `CLASSIC_PLAYER_WEIGHTS` is **not set on any
  deployment yet** — selecting `ruleset=classic` today fails fast at model load.
- Optional tunables with defaults: `TARGET_PROCESS_FPS` (15), `CMC_METHOD` (ecc),
  `CLIP_BATCH_WINDOW` (16), `FIT_SAMPLE_EVERY` (**30**, was 5), `FIT_SILHOUETTE_MIN`
  (0.20), queue/ordering/callback retry knobs, `SESSION_IDLE_EVICT_S`, paths, etc.
  `FIT_SAMPLE_EVERY` was raised from 5 to 30 in the "added fixes to stop the oom error"
  commit (a separate, closely-following commit from the docker-compose memory limits in
  [infra/README.md](../infra/README.md)) to bound torso-crop volume for whole-match
  `/post-processing` uploads — `collect_crops` there samples the entire match file, not
  just a ~60 s clip, so a small stride generates far more crops than fitting needs —
  while still sampling frames spread across the full file.
- **Does NOT** hardcode secrets; everything comes from `/etc/gsfa-highlights.env` on the
  VM (loaded by docker-compose `env_file`). `PLAYER_WEIGHTS` must be the **bare path** to
  the weights — a value like `PLAYER_WEIGHTS=/path/...` (name duplicated) is the classic
  misconfig that crashes the worker at model load.

## `api.py` (FastAPI, no GPU)
- `POST /api/clips` (multipart): hygiene checks (content type, size cap), all six
  `team_*` fields required and non-blank (`_require_nonblank` ⇒ `422`), validates
  `ruleset` against the `rulesets` registry (`422` on unknown), `db.ensure_match`
  (first clip creates the match — with its `ruleset`, fixed for the match's lifetime —
  and stores the six `team_a_*`/`team_b_*` fields; later clips must still send them but
  they are ignored past the first, and any `ruleset` they send is ignored),
  resolve/claim the minute (`db.claim_next_minute`
  if not supplied), dedupe (`minute_exists` or blob already exists ⇒ `202 duplicate`),
  else `blob.upload_stream` + `queue.enqueue` ⇒ `202`.
- `POST /post-processing` (multipart): same `ruleset` validation + the same six required
  `team_*` fields as above; accepts a whole-match video, stores it at
  `clips/<match_id>/post_processing.mp4`, enqueues a background job, and returns `200`
  once the upload is fully received. The worker deletes the blob after processing.
- `GET /health` — 200 if the worker heartbeat is < 300 s old, else 503.
- `GET /metrics` — queue depths, last-clip seconds, seconds-behind-live, GPU mem,
  `last_post_processing_error` (the most recent whole-match job's failure, if any — not
  retried automatically, see `worker.py` below), etc.
- **Does NOT** process clips inline, touch the GPU, or render — it returns in ~1–2 s.
- An access-log filter drops 404 spam from internet scanners.

## `worker.py` (GPU)
- `Worker.__init__` sets up queue, blob, `ModelBundle()` (empty — no longer loads a model
  eagerly at startup; see `session.py` below), `MatchSessionManager`, a SQL connection.
- `run_forever()` polls every ~2 s; on any loop exception it logs, **reconnects SQL**, and
  continues (resilient to transient DB drops).
- Writes a heartbeat each loop (GPU visibility/mem, last clip seconds, queue depth →
  seconds-behind-live, active matches, `last_post_processing_error`).
- **Does NOT** parallelise across clips — one clip at a time (simple, ~1 concurrent match
  expected). **Does NOT** render video.
- Routes `kind = post_processing` queue messages through a separate whole-match path
  (`_handle_post_processing`) and writes the `post_processing` SQL table:
  - The queue message is **deleted before processing starts** — no lease renewal, no
    automatic retry. A job that dies partway through (in-process exception, container
    OOM-kill, VM shutdown) must never be silently redelivered and reprocessed on top of
    leftover state; recovery is a **manual re-upload**.
  - The `MatchSession` is constructed **directly** (`MatchSession(match_id, self.models,
    ruleset)`), never through `MatchSessionManager` and never stored/reused — every
    attempt starts from clean tracker/ball/carrier/pass-FSM state, isolated from the
    live-clip path for the same match.
  - `db.ensure_match(...)` is called first (it only sets `ruleset` on the row's initial
    `INSERT`, so a match created by a prior `/api/clips` upload keeps that ruleset), then
    `ruleset = get_ruleset(db.get_match_ruleset(conn, match_id))` reads it back.
  - The blob download, `process_match_video(...)` call, and `db.write_post_processing_result`
    write are wrapped in a `try/except`: on failure the exception is logged and recorded
    in-memory as `self._last_post_processing_error` (surfaced via `GET /metrics`), and a
    `finally` block still deletes the blob and job dir — but the queue message is already
    gone, so **no retry happens**.

## `session.py`
- **`ModelBundle`** — was a `@dataclass` eagerly loading one global `PlayerDetector` at
  worker startup (`ModelBundle.load()`); now a plain class holding
  `dict[ruleset_name, PlayerDetector]`, populated **lazily**: `player_detector(ruleset)`
  loads and caches a `PlayerDetector` for that ruleset's weights/conf on first use, so a
  deployment that only ever serves one ruleset never loads VRAM for the other. The team
  classifier is per-match, not here.
- **`MatchSession`** — `__init__(match_id, models, ruleset: RulesetConfig)` (gained the
  `ruleset` param). All cross-clip state for one match: `tracker`, `ball_tracker`,
  `carrier_eng`, `pass_track`, `proc_idx`, `team_clf`, `fit_status`, `gk_det` (see below),
  the carryover for a pass spanning a clip boundary, and `last_written`. `self.player_det
  = models.player_detector(ruleset)` resolves this match's detector. Every CV class
  (`PlayerTracker`, `BallTracker`, `CarrierEngine`, `PassEventTracker`,
  `GSFATeamClassifier`) is now constructed **from `ruleset`'s fields** rather than
  hardcoded defaults (values are unchanged for `futsal`). On construction it reloads the
  team fit pkl from `MATCH_STATE_DIR/<match_id>/team_siglip.pkl` if present (survives
  restarts).
  - `ensure_fit(clip)` — clip-1 dense fit with a **silhouette quality guard**
    (`fit_and_score`): below `FIT_SILHOUETTE_MIN` ⇒ keep clip-1 crops and **refit on clip
    2** with combined samples; never refit after committing (re-running KMeans could swap
    the 0/1 labels mid-match). Commits the pkl + resolves team names.
  - `ensure_gk_ready()` — constructs `self.gk_det` (`GoalkeeperDetector`) once
    `db.get_gk_colours(match_id)` returns both reference colours; no-op if already
    constructed or if construction previously failed (`ValueError` on an unparseable
    colour, cached as `_gk_colour_invalid` so it isn't retried every clip). Never raises —
    GK classification is an overlay on top of the core possession stats, called at the
    top of every `clip_processor.process_clip(...)`.
  - `split_adjustment(n)` / `finish_clip(half, minute)` / `new_events()` — the boundary
    bookkeeping used by `clip_processor`.
- **`MatchSessionManager.get_or_create(match_id, ruleset)`** (gained the `ruleset` param)
  — `dict[match_id → MatchSession]` with idle eviction. `ruleset` is only used when
  creating a new session; an existing session keeps whatever ruleset it was created with
  (a match's ruleset is fixed for its lifetime, enforced by `db.ensure_match` only setting
  it on `INSERT`).
- `collect_crops(clip_path, player_det, sample_every, ruleset)` (gained the `ruleset`
  param) uses the team classifier's **static** crop helpers (`_torso_crop`, `_is_sharp`),
  now parametrized by `ruleset.torso_ratio`/`min_crop_px`/`blur_threshold`; it does
  **NOT** call `GSFATeamClassifier.fit_from_video`.
- **Asserts at import** that `stats.py`'s label/event strings equal
  `modules.possession.labels`'s — the cross-module contract guard.

## `clip_processor.py`
- `process_clip(session, clip_path, half, minute, clip_duration_seconds, clip_blob_path=None)`
  first calls `session.ensure_gk_ready()` (see `session.py` above), then decodes the
  clip, strides to `TARGET_PROCESS_FPS`, and processes in windows of `CLIP_BATCH_WINDOW`
  frames:
  - **Pass 1 (batched, stateless GPU):** `player_det.detect_batch` + `team_clf.classify_batch`
    + (if `session.gk_det` is ready) `gk_det.classify(frame, dets)` per frame + `best_ball`
    per frame.
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
  `build_payload(...)` (the flat advance-stats body; maps internal `team_a`/`team_b`
  counters onto the unchanged wire keys `*_a`/`*_b` — cluster id 0 → team_a → a).
- Deliberately torch/boxmot/pyodbc-free so the correctness-critical math is testable
  (`tests/test_stats.py`).

## `db.py` (Azure SQL via pyodbc)
- **Cumulative-on-read:** `minute_stats` stores raw per-minute counters; `cumulative_read`
  `SUM()`s over rows ≤ the current minute. Never stores cumulative numbers.
- **`write_clip_result(...)` = one transaction:** apply prior correction (revision += 1) →
  upsert this minute row (PK = `(match_id, half, minute)` ⇒ replay overwrites, now also
  storing `clip_duration_seconds`) → insert events → advance match progress → insert the
  **outbox** row with a fresh cumulative payload. Rolls back on any error.
- Match upsert (`ensure_match` — takes `team_a/b_name`, `team_a/b_colour`,
  `team_a/b_gk_colour` and `ruleset`;
  `ruleset` is written only on the `INSERT` branch, never the `UPDATE` branch, so it's
  fixed at match creation; COALESCE so later clips can't overwrite name/colour fields),
  atomic minute claim (`claim_next_minute` via `UPDATE...OUTPUT`), `minute_exists`,
  `get_match_progress`, `get_match_ruleset` (returns `"classic"` if the row is somehow
  missing — should not happen, callers `ensure_match()` first), `get_team_specs` (only if
  both teams have name **and** colour), `get_gk_colours` (only if **both** GK colours are
  set — independent of team names). Outbox ops (`fetch_pending`, `mark_sent`,
  `mark_failed`) for the notifier.
- `post_processing_exists(conn, match_id)` / `write_post_processing_result(...)` — dedupe
  check and upsert for the whole-match `post_processing` table, keyed by `match_id`;
  stores the team metadata snapshot (including `team_a/b_gk_colour`) alongside the raw
  counters. Does **not** currently take/store a `ruleset` value. Unlike
  `write_clip_result`, this does **not** touch `callback_outbox` — no advance-stats
  callback is sent for whole-match uploads.

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

## Why clips stay ~60 s (and why longer doesn't help detection)

"60 s" is a **client-side convention, not an enforced limit**. `scripts/upload_clips.ps1`
slices the source with `ffmpeg -t 60`; the API only checks `clip_duration_seconds > 0` and
the `MAX_UPLOAD_GB` byte cap (`service/api.py`). The pipeline treats a clip as an opaque
video of arbitrary length.

**Detection / tracking quality is independent of clip length.** `MatchSession`
(`session.py`) constructs `tracker` / `ball_tracker` / `carrier_eng` / `pass_track` and the
team fit **once** and reuses them for every clip of the match (the session is evicted only
after `SESSION_IDLE_EVICT_S` = 30 min idle). A clip boundary is therefore **not** a cold
start — the trackers, ball Kalman, carrier hysteresis and pass FSM all carry their state
across it, and a pass that spans the boundary resolves normally via
`session.carryover_travel_frames` + a `PriorCorrection`. The only genuine cold start is
match start / worker restart / session eviction. Making clips longer just spreads that one
discontinuity over more frames — a marginal robustness gain, not an accuracy gain.

**Longer clips are actively worse in production:**

- **`QUEUE_VISIBILITY_SEC` = 90 s and the worker never renews the lease during
  processing.** A 60 s clip already takes ~85–90 s to process (see `docs/API.md`'s
  `last_clip_processing_seconds: 87.4`). If processing exceeds 90 s the queue message
  becomes visible again, gets re-dequeued, and after `MAX_DEQUEUE_COUNT` (3) it is moved to
  the poison queue and the clip is lost.
- **"1 clip = 1 minute" is baked in.** `db.claim_next_minute`, `stats.is_expected` (the
  ordering guard), one `minute_stats` row per minute, and `db._apply_prior_correction`
  (which corrects only the *immediately* prior minute row) all assume it.
- **Live latency.** Per-minute clips keep the advance-stats callback near-real-time; longer
  clips push the stats feed further behind live.

**If you need maximum temporal context** (e.g. a one-off full re-analysis), use the
whole-match `POST /api/post-processing` path — it runs the entire match through a single
`process_clip` call with the trackers/FSM live for the whole file. It writes one aggregate
row to `post_processing` and does **not** emit the live advance-stats callback.

---

## What the service as a whole does NOT do
- No authentication on the API in v1 — access is restricted at the network level (Azure
  NSG on port 8000).
- No video rendering / highlight cutting — outputs are SQL rows + the events log + the
  callback. The `events` table is the future highlight-reel source.
- No multi-worker scaling / distributed locking — single worker, one clip at a time.
- No contested-possession or dribble events (matches the pipeline's design).
