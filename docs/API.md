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

---

## Required environment variables

Set in `/etc/gsfa-highlights.env` (loaded via docker-compose `env_file`).

| Variable | Required | Default | Description |
|---|---|---|---|
| `AZURE_STORAGE_CONNECTION_STRING` | yes | — | Blob + Queue storage connection string |
| `SQL_CONN_STR` | yes | — | pyodbc connection string for Azure SQL `gsfa_stats` |
| `PLAYER_WEIGHTS` | yes | — | Absolute VM path to YOLO player-detector weights |
| `BALL_WEIGHTS` | yes | — | Absolute VM path to RF-DETR ball-detector weights |
| `CALLBACK_URL` | no | `""` | HTTP POST endpoint to receive per-minute stats (empty = disable) |
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
| `half` | integer | yes | Match half (>= 1). |
| `minute` | integer | yes | Minute within the half (>= 1). |
| `team0_name` | string | no | Display name for team 0 (e.g. `"FCA"`). Used in callback payload keys. |
| `team1_name` | string | no | Display name for team 1 (e.g. `"Rovers"`). |
| `team0_colour` | string | no | Jersey colour for team 0 — hex (`"#FF6600"` / `"FF6600"`) or CSS name (`"orange"`). Used for cluster-to-team mapping. |
| `team1_colour` | string | no | Jersey colour for team 1. |

Team name/colour fields on the first clip initialise the match; subsequent clips only fill in values that are still `NULL` (later calls cannot overwrite).

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
| 422 | `half` or `minute` < 1 |

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
  "last_dequeue_count": 1
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

---

## Callback (outbox) — POST to CALLBACK_URL

After each clip is processed and its SQL transaction commits, the worker POSTs the cumulative stats payload to `CALLBACK_URL`. Deliveries are strictly ordered by `(half, minute)`. Retry: up to `CALLBACK_RETRIES` attempts with exponential backoff (`CALLBACK_BACKOFF_BASE × 2^n` seconds). On permanent failure the row is marked `failed` in SQL but processing continues.

If `CALLBACK_URL` is unset, rows stay `pending` in `callback_outbox` — no stats are lost.

**Payload shape:**

```json
{
  "match_id": "match_abc123",
  "half": 1,
  "minute": 5,
  "revision": 0,
  "teams": { "0": "FCA", "1": "Rovers" },
  "cumulative": {
    "frames_FCA":              720,
    "frames_Rovers":           450,
    "frames_loose":             80,
    "frames_oof":               10,
    "passes_completed_FCA":     14,
    "passes_completed_Rovers":   9,
    "interceptions_FCA":         2,
    "interceptions_Rovers":      1,
    "ball_lost_FCA":             3,
    "ball_lost_Rovers":          4
  }
}
```

- All `cumulative` counters are **running totals from minute 1 up to and including the current minute** — not deltas for this clip alone.
- Per-team keys use the resolved team names from `teams`. If team names were not supplied, the keys fall back to `team0` / `team1`.
- `revision` increments when a retroactive correction was applied to this minute row (e.g. a cross-clip pass resolved as an interception).
- Possession percentage is derived by the receiver: `frames_FCA / (frames_FCA + frames_Rovers + frames_loose)`.
