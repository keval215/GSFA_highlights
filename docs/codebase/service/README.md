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
4. **Ruleset + session + generation check** — `ruleset = get_ruleset(db.get_match_ruleset(conn,
   match_id))`; `want_gen = db.get_fit_generation(conn, match_id)`; `session, just_reset =
   manager.get_or_create(match_id, ruleset, want_gen)`; download blob to a job dir. `ruleset`
   is only used if a new session is created — an existing session keeps whatever ruleset it
   was built with. If `just_reset` (the DB's `fit_generation` had advanced past this
   session's own — a mid-match team/GK colour change, auto-detected or via
   `POST /api/matches/{id}/reset-fit`), `db.mark_minutes_superseded(conn, match_id,
   last_half, last_minute)` flags every `minute_stats`/`events` row up to the last
   processed minute so `cumulative_read` excludes them going forward, and a WARNING is
   logged with the row count. The ordering pointer (`last_half_processed`/
   `last_minute_processed`) is untouched.
5. **Team fit** — if `session.fit_status != "ok"`, `session.ensure_fit(clip_path)`. After a
   generation reset this is "pending" again, so the very next clip re-fits from scratch.
6. **Process** — `clip_processor.process_clip(session, clip_path, half, minute,
   clip_duration_seconds, blob_path)` — internally calls `session.ensure_gk_ready()` first
   (lazily builds `GoalkeeperDetector` once the DB has both GK reference colours; a no-op
   once built), runs `gk_det.classify(...)` per frame if ready, and reorients the returned
   `ClipResult` via `stats.orient_for_team_a` before returning it (see `clip_processor.py`
   below).
7. **Persist** — `db.write_clip_result(conn, minute_row, correction, events)` — a single
   transaction. No team-name/colour argument any more; orientation was already applied in
   step 6.
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
  (first clip creates the match — with its `ruleset`, fixed for the match's lifetime.
  Names stay fill-in-only (COALESCE, first value wins); the four **colour** fields are
  overwritten on every request and bump `matches.fit_generation` when a supplied colour
  genuinely differs from the stored one — the mid-match colour-change path, see `db.py`),
  resolve/claim the minute (`db.claim_next_minute`
  if not supplied), dedupe (`minute_exists` or blob already exists ⇒ `202 duplicate`),
  else `blob.upload_stream` + `queue.enqueue` ⇒ `202`.
- `POST /post-processing` (multipart): same `ruleset` validation + the same six required
  `team_*` fields as above; accepts a whole-match video, stores it at
  `clips/<match_id>/post_processing.mp4`, enqueues a background job, and returns `200`
  once the upload is fully received. The worker deletes the blob after processing.
- `POST /api/matches/{match_id}/reset-fit` (JSON, body optional — `ResetFitBody`, all 6
  name/colour fields default `None`): manual override that unconditionally bumps
  `matches.fit_generation` (`db.bump_fit_generation`), forcing a re-fit on the match's
  next clip even with no real colour change. Any supplied field overwrites the stored
  value (not COALESCE). `404` if the match row is absent; `_require_nonblank_if_present`
  ⇒ `422` if a field is present but blank (an omitted field is fine — only an explicit
  `""`/whitespace is rejected, since a blanked colour would silently disable colour→team
  resolution and GK matching). Returns `{match_id, fit_generation}`. See [docs/API.md](../../API.md).
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
- **Mid-match fit-generation reset:** `_handle` reads `db.get_fit_generation` and passes it
  as `want_generation` into `manager.get_or_create`. If that triggers a per-match session
  reset (`just_reset`), it calls `db.mark_minutes_superseded(match_id, last_half,
  last_minute)` and logs a WARNING with the superseded row count. Only that one match's
  session is affected — no worker restart, other matches untouched.
- Routes `kind = post_processing` queue messages through a separate whole-match path
  (`_handle_post_processing`) and writes the `post_processing` SQL table:
  - The queue message is **deleted before processing starts** — no lease renewal, no
    automatic retry. A job that dies partway through (in-process exception, container
    OOM-kill, VM shutdown) must never be silently redelivered and reprocessed on top of
    leftover state; recovery is a **manual re-upload**.
  - The `MatchSession` is constructed **directly** (`MatchSession(match_id, self.models,
    ruleset)`), never through `MatchSessionManager` and never stored/reused — every
    attempt starts from clean tracker/ball/carrier/pass-FSM state, isolated from the
    live-clip path for the same match. It always fits fresh (`fit_status` starts
    `"pending"`), so it never consults `fit_generation` or the reset check — a colour
    change detected here by `ensure_match` has no separate effect on this one-shot run.
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
  `fit_generation`, `team_a_cluster_id`, the carryover for a pass spanning a clip
  boundary, `last_written`, and `last_written_team_a_cluster_id`. `self.player_det =
  models.player_detector(ruleset)` resolves this match's detector. Every CV class
  (`PlayerTracker`, `BallTracker`, `CarrierEngine`, `PassEventTracker`,
  `GSFATeamClassifier`) is now constructed **from `ruleset`'s fields** rather than
  hardcoded defaults (values are unchanged for `futsal`). On construction it reloads the
  team fit pkl from `MATCH_STATE_DIR/<match_id>/team_siglip.pkl` if present (survives
  restarts), and restores `fit_generation`/`team_a_cluster_id` from the `fit_meta.json`
  sidecar next to it (see below).
  - `ensure_fit(clip)` — clip-1 dense fit with a **silhouette quality guard**
    (`fit_and_score`): below `FIT_SILHOUETTE_MIN` ⇒ keep clip-1 crops and **refit on clip
    2** with combined samples; never refits **the same committed fit** afterwards
    (re-running KMeans mid-match on unchanged jerseys could swap the 0/1 labels and
    corrupt stats). A genuinely new `fit_generation` (mid-match colour change) is the one
    case where a full refit is wanted — that goes through `reset_for_new_generation`
    below, not `ensure_fit`'s own refit-once logic. Commits the pkl, writes the
    `fit_meta.json` sidecar (`_write_fit_meta`), and resolves team names
    (`_resolve_team_names`, which also sets `team_a_cluster_id`).
  - `_resolve_team_names(clf, crops)` — maps the two KMeans clusters to the caller's team
    names/colours (`clf.resolve_team_names`) and records which cluster resolved to
    `team_a_name` as `self.team_a_cluster_id` (`None` if resolution wasn't attempted or
    failed — falls back to raw cluster order). On a re-fit (`fit_generation > 1`) a
    resolution failure is logged at ERROR, since it silently leaves `team_a`/`team_b`
    orientation stale until the next successful re-fit — a documented v1 limitation.
  - `_write_fit_meta()` / `_load_fit_meta()` — persist/restore `{fit_generation,
    team_a_cluster_id}` as `fit_meta.json` next to `team_siglip.pkl`, so a disk-reloaded
    session (worker restart, or a late clip after idle eviction) recovers them instead of
    reverting to generation 1 / unresolved orientation. A missing/corrupt sidecar degrades
    to that same fallback rather than raising.
  - `reset_for_new_generation(n)` — the mid-match colour-change reset: deletes
    `team_siglip.pkl` + `fit_meta.json`, clears `team_clf`/sets `fit_status = "pending"`/
    clears `_fit_crops_clip1`, clears `gk_det`/`_gk_colour_invalid`, clears
    `team_a_cluster_id`, sets `fit_generation = n` — i.e. starts the team/GK fit over as
    if this were clip 1 again, under the new generation. `tracker`/`ball_tracker`/
    `carrier_eng`/`pass_track`/`proc_idx`/`n_events_seen`/`last_written`/
    `carryover_travel_frames` are **left untouched** so an in-flight pass isn't lost.
  - `ensure_gk_ready()` — constructs `self.gk_det` (`GoalkeeperDetector`) once
    `db.get_gk_colours(match_id)` returns both reference colours; no-op if already
    constructed or if construction previously failed (`ValueError` on an unparseable
    colour, cached as `_gk_colour_invalid` so it isn't retried every clip). Never raises —
    GK classification is an overlay on top of the core possession stats, called at the
    top of every `clip_processor.process_clip(...)`.
  - `split_adjustment(n)` / `finish_clip(half, minute)` / `new_events()` — the boundary
    bookkeeping used by `clip_processor`. `finish_clip` also snapshots the current
    `team_a_cluster_id` into `last_written_team_a_cluster_id` — "the orientation the just-
    written row was written under" — for a future clip's boundary correction to reorient
    against (see `clip_processor.py` / `stats.orient_for_team_a` below).
- **`MatchSessionManager.get_or_create(match_id, ruleset, want_generation=None)`** —
  `dict[match_id → MatchSession]` with idle eviction, now returns `(session, just_reset)`.
  `ruleset` is only used when creating a new session; an existing session keeps whatever
  ruleset it was created with (a match's ruleset is fixed for its lifetime, enforced by
  `db.ensure_match` only setting it on `INSERT`). `want_generation` (the worker passes
  `db.get_fit_generation`'s current value) is compared against the session's own
  `fit_generation` — new, cached, and disk-reloaded sessions alike; a stale
  `fit_meta.json` sidecar next to an up-to-date pkl is caught here too. If
  `want_generation` is higher, `reset_for_new_generation` runs before the session is
  returned and `just_reset=True`.
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
- Builds `EventRow`s from `session.new_events()`; assembles a `ClipResult`
  (`minute_row`, `correction`, `events`) and then returns
  `stats.orient_for_team_a(result, session.team_a_cluster_id,
  correction_team_a_cluster_id)` — colour-anchored orientation applied once here at write
  time. `session.team_a_cluster_id` orients this clip's own row + events;
  `correction_team_a_cluster_id` (captured as `session.last_written_team_a_cluster_id` at
  the moment the `PriorCorrection` is built, before `finish_clip` overwrites it) orients
  the boundary correction, which targets a *different*, already-written row that may have
  been written under a different orientation.
- `clip_duration_seconds` (the client-supplied clip length) is threaded straight through
  into `MinuteCounters.to_minute_row(...)` and stored on `MinuteRow.clip_duration_seconds`
  — used only for the CSV export (`scripts/get_csv.py`), not for any pipeline math.
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
  `build_payload(match_id, half, minute, revision, sums)` (the flat advance-stats body;
  maps internal `team_a`/`team_b` counters onto the unchanged wire keys `*_a`/`*_b`). The
  old `team_id_to_name` param was **removed** — orientation is now resolved upstream at
  write time, so `build_payload` just sums already-oriented rows.
- `orient_for_team_a(result, team_a_cluster_id, correction_team_a_cluster_id=<same as
  row>)` — reorders a `ClipResult`'s `minute_row` counters, `events` (`from_team`/
  `to_team`), and `correction` (`team_id`) so `team_a`/`0` always means the cluster whose
  resolved jersey colour matches `team_a_colour`. Each part is a **no-op** unless its
  cluster id is `1` (only happens when a re-fit's KMeans landed the clusters in the
  opposite order) — a cluster id of `None` (resolution not run / failed) is also a no-op,
  falling back to raw cluster order. The `correction` is reoriented against
  `correction_team_a_cluster_id` (the orientation the *target* row was written under), not
  the current clip's, because the two can differ across a mid-match re-fit. Must be
  applied at write time (from `clip_processor`), never at `cumulative_read`/`build_payload`
  time — a mid-match swap must not retroactively reorder already-summed prior minutes.
- Deliberately torch/boxmot/pyodbc-free so the correctness-critical math is testable
  (`tests/test_stats.py`).

## `db.py` (Azure SQL via pyodbc)
- **Cumulative-on-read:** `minute_stats` stores raw per-minute counters; `cumulative_read`
  `SUM()`s over rows ≤ the current minute **and `superseded = 0`**. Never stores
  cumulative numbers.
- **`write_clip_result(conn, row, correction, events)` = one transaction:** apply prior
  correction (revision += 1) → upsert this minute row (PK = `(match_id, half, minute)` ⇒
  replay overwrites, storing `clip_duration_seconds`) → insert events → advance match
  progress → insert the **outbox** row with a fresh cumulative payload. Rolls back on any
  error. The `team_id_to_name` parameter was **removed** — team orientation is applied
  upstream by `clip_processor` (`stats.orient_for_team_a`) before rows reach this function.
- Match upsert (`ensure_match` — takes `team_a/b_name`, `team_a/b_colour`,
  `team_a/b_gk_colour` and `ruleset`). `ruleset` is written only on the `INSERT` branch,
  never `UPDATE`, so it's fixed at match creation. On `UPDATE`: names stay COALESCE
  (fill-in-only); the four **colour** columns are now `COALESCE(?, existing)` — a supplied
  value **overwrites**. When a supplied colour genuinely differs from the stored one
  (compared after `_normalise_colour`: lowercase, `#`-stripped, trimmed — so `#FF6600` ==
  `ff6600`), `fit_generation` is bumped by 1 (once per call; NULL→value first-fill is not
  a "change").
- `get_fit_generation(conn, match_id)` — current `matches.fit_generation` (`1` for a
  never-bumped or missing row). The worker reads this each clip and passes it as
  `want_generation` to `MatchSessionManager.get_or_create`.
- `mark_minutes_superseded(conn, match_id, upto_half, upto_minute)` — sets `superseded = 1`
  on every `minute_stats` **and** `events` row through `(upto_half, upto_minute)`
  inclusive; idempotent (only touches `superseded = 0` rows); returns total rowcount;
  commits. Called by the worker right after a generation-triggered session reset. (Only
  `minute_stats.superseded` is currently consulted on read — `cumulative_read`;
  `events.superseded` is set for consistency / future use.)
- `bump_fit_generation(conn, match_id, **name_colour_fields)` — backs
  `POST /api/matches/{id}/reset-fit`: unconditionally `fit_generation += 1` and overwrites
  any of the 6 name/colour fields supplied (plain `COALESCE(?, existing)`, not the
  auto-detect split `ensure_match` uses). Returns the new generation, or `None` if the
  match row is absent (→ `404`). Commits.
- Also: atomic minute claim (`claim_next_minute` via `UPDATE...OUTPUT`), `minute_exists`,
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
- No automatic recovery if jersey-colour resolution fails on a mid-match re-fit
  (`fit_generation > 1`): the failure is logged at ERROR and processing continues with raw
  cluster order, so `team_a`/`team_b` orientation can be wrong for those minutes until the
  next successful re-fit — a documented v1 limitation.
