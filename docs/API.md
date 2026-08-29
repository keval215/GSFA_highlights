# GSFA Highlights Service — API Reference

## Running the service

There are two separate processes; both must be running for end-to-end operation.

**API server** (ingestion, no GPU required):

```
uvicorn service.api:app --host 0.0.0.0 --port 8000
```

**GPU worker** (clip processing, needs CUDA):

```
python -m service.worker
```

**Base URL:** `http://<host>:8000`  
**Auth:** None in v1 — access is restricted at the network level (Azure NSG on port 8000).

**Production ingest:** clips are produced by an **RTMP relay running on the same VM** that
segments the live broadcast into ~60 s clips and `POST`s each to `http://localhost:8000/api/clips`
over loopback. The relay is responsible for supplying every form field below. See
[`azure_deploy.md` → "RTMP live ingest"](azure_deploy.md#rtmp-live-ingest).

---

## Required environment variables

Set in `/etc/gsfa-highlights.env` (loaded via docker-compose `env_file`).

| Variable | Required | Default | Description |
|---|---|---|---|
| `AZURE_STORAGE_CONNECTION_STRING` | yes | — | Blob + Queue storage connection string |
| `SQL_CONN_STR` | yes | — | pyodbc connection string for Azure SQL `gsfa_stats` |
| `PLAYER_WEIGHTS` | only if `futsal` matches are served | — | Absolute VM path to the unified YOLOv11m weights for the `futsal` ruleset (players + ball + refs + posts) |
| `<RULESET>_PLAYER_WEIGHTS` | required for the default ruleset (`classic`) | — | e.g. `CLASSIC_PLAYER_WEIGHTS` — weights path for a non-`futsal` ruleset (see `rulesets/`). `classic` is now the default ruleset, so `CLASSIC_PLAYER_WEIGHTS` must point at a real classic-trained checkpoint on the VM or the worker fails fast at model load on the first clip |
| `CALLBACK_URL` | no | `""` | Base origin of the main app (e.g. `https://dev-server.clubduelz.in`), no path. The worker appends `/v1/pvt/tournament-duelz/{match_id}/advance-stats`. Empty = disable |
| `SUPER_ADMIN_KEY` | no | `""` | `X-Super-Admin-Key` sent with each advance-stats POST. Empty = disable callback |
| `CALLBACK_RETRIES` | no | `3` | Max delivery attempts per outbox row before it is marked `failed` |
| `CALLBACK_BACKOFF_BASE` | no | `1.0` | Base seconds for exponential callback backoff (`base × 2^n` → 1s, 2s, 4s) |
| `CLIPS_CONTAINER` | no | `clips` | Azure Blob container for uploaded clips |
| `QUEUE_NAME` | no | `clips` | Azure Queue name for the processing queue |
| `POISON_QUEUE_NAME` | no | `clips-poison` | Queue for clips that exceed `MAX_DEQUEUE_COUNT` |
| `MATCH_STATE_DIR` | no | `/mnt/data/match_state` | Worker state / heartbeat directory |
| `JOBS_DIR` | no | `/mnt/data/jobs` | Scratch directory for in-flight clip downloads |
| `DEVICE` | no | `cuda` | Torch device for detectors + classifier (`cuda` or `cpu`) |
| `TARGET_PROCESS_FPS` | no | `15` | Frames per second sampled from each clip |
| `CMC_METHOD` | no | `ecc` | BoT-SORT camera-motion compensation: `ecc` / `sof` / `orb` / `sift` / `none` |
| `CLIP_BATCH_WINDOW` | no | `16` | Frames per GPU batch in the two-pass processor |
| `FIT_SAMPLE_EVERY` | no | `30` | Torso-crop sampling stride for team fitting (every Nth raw frame). Raised from 5 to bound crop volume on whole-match `/post-processing` uploads |
| `FIT_SILHOUETTE_MIN` | no | `0.20` | Minimum KMeans silhouette score on clip-1 crops before the fit is accepted; below this, clip 2's crops are combined in and refit (once) |
| `MAX_UPLOAD_GB` | no | `2` | Max clip file size accepted by POST /api/clips |
| `API_PORT` | no | `8000` | API listen port (for documentation only; pass to uvicorn separately) |
| `SESSION_IDLE_EVICT_S` | no | `1800` | Seconds of inactivity before a MatchSession is evicted (30 min) |

---

## Endpoints

### POST /api/clips

Upload a 60-second clip for processing. Returns immediately (~1–2 s); processing is asynchronous via the GPU worker.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | binary (mp4) | yes | The clip file. `video/mp4` or `application/octet-stream`. Max `MAX_UPLOAD_GB` GB. |
| `match_id` | string | yes | Unique match identifier. Auto-creates the match row on first clip. |
| `clip_duration_seconds` | number | yes | Duration supplied by the client for this clip, stored directly in SQL. Must be `> 0`; no other constraint — clips are not required to be exactly 60s. ~60 s is nonetheless recommended: the worker's 90 s queue-visibility lease and the 1-clip-per-minute SQL mapping make longer clips risky, and longer clips do **not** improve detection (tracker state already persists across clips). See [service/README.md](codebase/service/README.md) "Why clips stay ~60 s". |
| `half` | integer | no | Match half (>= 1 if given). Defaults to `1` when omitted. |
| `minute` | integer | no | Minute within the half (>= 1 if given). When omitted, the server auto-assigns the next sequential minute for `(match_id, half)` via `db.claim_next_minute()`. |
| `team_a_name` | string | **yes** | Display name for team A (CV cluster id 0), e.g. `"FCA"`. Stored in `matches`; does not change the callback body keys. |
| `team_b_name` | string | **yes** | Display name for team B (CV cluster id 1), e.g. `"Rovers"`. |
| `team_a_colour` | string | **yes** | Outfield jersey colour for team A — hex (`"#FF6600"` / `"FF6600"`) or CSS name (`"orange"`). Drives cluster-to-team mapping. |
| `team_b_colour` | string | **yes** | Outfield jersey colour for team B. |
| `team_a_gk_colour` | string | **yes** | Goalkeeper jersey colour for team A, same format. Drives direct colour-match GK classification (`modules/detectors/goalkeeper_detector.py`); `session.ensure_gk_ready()` builds the GK detector from clip 1. A malformed colour disables GK classification for that match but never fails the clip. |
| `team_b_gk_colour` | string | **yes** | Goalkeeper jersey colour for team B. |
| `ruleset` | string | no | `"classic"` (default) or `"futsal"` — selects the sport-tuning profile (`rulesets/`). Only used on the **first** request for a `match_id` (creates the match row); later requests ignore it — a match's ruleset is fixed for its lifetime. Unknown value ⇒ `422`. Omit the field (or send `classic`) to get classic; senders that still pass `ruleset=futsal` explicitly keep getting futsal. |

> **v6 field rename + required fields:** the inbound form fields were `team0_*` / `team1_*` before v6 and are now `team_a_*` / `team_b_*` (convention: CV cluster id 0 → team_a, 1 → team_b). Hard cutover — the old names are no longer accepted. **All six `team_*` fields (names, outfield colours, GK colours) are now required** — a missing *or blank* one is a `422`. The **outbound** advance-stats callback body is unchanged (still `frames_a` / `frames_b` / …).

All six `team_*` fields are required on **every** request, not just the first clip. The
first request's values initialise the `matches` row; on later requests they are still
required but ignored (`db.ensure_match` only fills `NULL`s and never overwrites, and cannot
change the `ruleset`). The server stores `clip_duration_seconds` exactly as sent by the client.

**202 Accepted — new clip enqueued:**

```json
{
  "accepted": true,
  "match_id": "match_abc123",
  "half": 1,
  "minute": 5
}
```

**202 Accepted — duplicate (already processed or blob already exists):**

```json
{
  "accepted": true,
  "duplicate": true,
  "match_id": "match_abc123",
  "half": 1,
  "minute": 5
}
```

**Error responses:**

| Status | Condition |
|---|---|
| 413 | File exceeds `MAX_UPLOAD_GB` |
| 415 | Unsupported content type |
| 422 | a required `team_*` field missing or blank, `half`/`minute` given but < 1, `clip_duration_seconds` <= 0, or unknown `ruleset` |

---

### POST /post-processing

Upload a whole-match video for post-match analysis. The API returns `200` as soon as the file has been fully received, stored in blob storage, and queued for background processing.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | binary (mp4) | yes | Whole-match video. `video/mp4` or `application/octet-stream`. Max `MAX_UPLOAD_GB` GB. |
| `match_id` | string | yes | Unique match identifier. Used as the SQL primary key in `post_processing`. |
| `team_a_name` | string | **yes** | Display name for team A (CV cluster id 0). Stored in `matches` and `post_processing`. |
| `team_b_name` | string | **yes** | Display name for team B (CV cluster id 1). |
| `team_a_colour` | string | **yes** | Outfield jersey colour for team A. |
| `team_b_colour` | string | **yes** | Outfield jersey colour for team B. |
| `team_a_gk_colour` | string | **yes** | Goalkeeper jersey colour for team A. Same format / semantics as on `POST /api/clips` above. |
| `team_b_gk_colour` | string | **yes** | Goalkeeper jersey colour for team B. |
| `ruleset` | string | no | `"classic"` (default) or `"futsal"`. Same semantics as on `POST /api/clips` — first-request-wins, unknown value ⇒ `422`. |

All six `team_*` fields are required (missing or blank ⇒ `422`), same as `POST /api/clips`.

The server uploads the file to blob storage at `clips/<match_id>/post_processing.mp4`,
enqueues a background job, and deletes the blob after processing completes — whether it
succeeds or fails. A failed whole-match job is **not retried**; the failure is recorded
and surfaced via `GET /metrics`' `last_post_processing_error`, and recovery is a manual
re-upload (see `service/README.md`'s worker section).

**200 OK — received and queued:**

```json
{
  "received": true,
  "match_id": "match_abc123"
}
```

**200 OK — duplicate upload already processed:**

```json
{
  "received": true,
  "duplicate": true,
  "match_id": "match_abc123"
}
```

---

### GET /health

Liveness check. Returns 200 when the worker is alive, 503 when it has not written a heartbeat within the last 300 s.

**Response (200 — healthy):**

```json
{
  "api": "ok",
  "worker_alive": true,
  "gpu_visible": true,
  "worker_heartbeat_age_s": 1.4
}
```

**Response (503 — worker dead or not started):**

```json
{
  "api": "ok",
  "worker_alive": false,
  "gpu_visible": false,
  "worker_heartbeat_age_s": null
}
```

---

### GET /metrics

Operational metrics. Always 200; fields are `null` when the worker has not run yet.

**Response:**

```json
{
  "queue_depth": 3,
  "poison_depth": 0,
  "last_clip_processing_seconds": 87.4,
  "seconds_behind_live": 180,
  "active_matches": ["match_abc123"],
  "gpu_memory_allocated_mb": 2048,
  "last_dequeue_count": 1,
  "last_post_processing_error": null
}
```

| Field | Description |
|---|---|
| `queue_depth` | Clips waiting to be processed |
| `poison_depth` | Clips that failed > `MAX_DEQUEUE_COUNT` times (manual inspection needed) |
| `last_clip_processing_seconds` | Wall time for the most recent clip |
| `seconds_behind_live` | `queue_depth × 60` — approximate lag behind live match time |
| `active_matches` | Match IDs with live MatchSession state in the worker |
| `gpu_memory_allocated_mb` | `torch.cuda.memory_allocated()` in MB |
| `last_dequeue_count` | Azure dequeue count of the last message (> 1 means it was retried) |
| `last_post_processing_error` | `null`, or `{"match_id", "error", "at"}` for the most recent whole-match `/post-processing` job that failed. Whole-match jobs are never auto-retried (see `POST /post-processing` above), so this is the only signal that one needs a manual re-upload. |

---

## Callback (outbox) — POST to the advance-stats endpoint

After each clip is processed and its SQL transaction commits, the worker POSTs the cumulative stats to the tournament-duelz advance-stats endpoint:

```
POST {CALLBACK_URL}/v1/pvt/tournament-duelz/{match_id}/advance-stats
X-Super-Admin-Key: {SUPER_ADMIN_KEY}
```

`CALLBACK_URL` is the **base origin only** (e.g. `https://dev-server.clubduelz.in`); `match_id` is the tournament-duel ObjectID and fills the `{id}` path segment. Each POST is a **full overwrite** of the duel's `advance_stats` subdocument and the server replies `204 No Content`.

Deliveries are strictly ordered by `(half, minute)`. Retry: up to `CALLBACK_RETRIES` attempts with exponential backoff (`CALLBACK_BACKOFF_BASE × 2^n` seconds); a `401` (bad/missing super-admin key) fails fast without retrying. On permanent failure the row is marked `failed` in SQL but processing continues.

Note: the whole-match `POST /post-processing` path does **not** go through this callback — `write_post_processing_result` upserts `post_processing` directly and no outbox row is created for it.

If `CALLBACK_URL` or `SUPER_ADMIN_KEY` is unset, rows stay `pending` in `callback_outbox` — no stats are lost.

**Body shape (flat, all integers ≥ 0):**

```json
{
  "frames_a": 720,
  "frames_b": 450,
  "frames_loose": 80,
  "frames_oof": 10,
  "passes_completed_a": 14,
  "passes_completed_b": 9,
  "interceptions_a": 2,
  "interceptions_b": 1,
  "ball_lost_a": 3,
  "ball_lost_b": 4
}
```

- All counters are **running totals from minute 1 up to and including the current minute** — not deltas for this clip alone. Because each call overwrites, the duel's `advance_stats` always reflects the latest cumulative state.
- Team mapping is **positional**: CV cluster id `0 → team_a → a`, cluster id `1 → team_b → b` (the same 0/1 the KMeans team fit assigns). No jersey-name resolution is applied to the body. The wire keys keep their historical `_a` / `_b` suffixes even though the internal/SQL columns are now `team_a` / `team_b` (`service/stats.py::build_payload` does the mapping).
- A retroactive correction to a prior minute simply produces a fresh cumulative body on the next POST, which overwrites with the corrected totals.
- Possession percentage is derived by the receiver: `frames_a / (frames_a + frames_b + frames_loose)`.
