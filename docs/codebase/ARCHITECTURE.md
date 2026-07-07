# Architecture

This document explains how the whole system fits together: the two execution modes,
the per-frame data flow, the cross-clip state machine in the service, and a
module-to-module connection map.

---

## 1. What the system computes

From match video it derives, per team (team 0 and team 1):

- **Possession %** — share of "in-play" frames where a team's player is the ball carrier.
- **Passes completed**, **interceptions**, **ball-lost** events.
- An append-only **event log** (each pass/interception/ball-lost with from/to track IDs).

It does this by chaining single-purpose stages, each of which mutates or wraps the
previous stage's output. The chain is:

```
        ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐
frame → │ PlayerDetector│ →│GSFATeamClassif│ →│GoalkeeperDetector│  (per-frame perception)
        │  (YOLOv11m)  │   │ (SigLIP team) │   │  (GK + team fix) │
        └─────────────┘   └──────────────┘   └──────────────────┘
                │                                      │
                ▼                                      ▼
        ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
        │ PlayerTracker│ → │  BallTracker │ → │ CarrierEngine│        (who has the ball)
        │  (BoT-SORT)  │   │   (Kalman)   │   │ (foot zone)  │
        └──────────────┘   └──────────────┘   └──────────────┘
                                                      │
                                                      ▼
                                          ┌────────────────────┐
                                          │  PassEventTracker   │       (pass FSM)
                                          │ release/travel/recv │
                                          └────────────────────┘
                                                      │
                                ┌─────────────────────┴───────────────────┐
                                ▼                                          ▼
                    ┌────────────────────┐                    ┌────────────────────┐
                    │  PossessionStats    │ (local mode)       │  MinuteCounters     │ (service mode)
                    │  in-memory % + bars │                    │  → SQL minute_stats │
                    └────────────────────┘                    └────────────────────┘
```

The dividing line is important:

- **Perception + "who has the ball" + the pass FSM** are written **once** in
  `video_analysis/possession.py` and `detectors/`, `team_classifier/`, `tracking/`.
- **Where the numbers go** differs by mode: local mode renders an overlay and prints a
  summary; service mode buckets the same labels into per-minute SQL rows.

---

## 2. Mode A — Local script (`video_analysis/possession.py`)

**Entry:** `python video_analysis/possession.py` → `run()`.

1. Build `PlayerDetector` (unified YOLOv11m), `GSFATeamClassifier`, `GoalkeeperDetector`.
2. `team_clf.fit_from_video_or_load(...)` — fit SigLIP+UMAP+KMeans once (cached pkl).
3. `gk_det.fit_from_video_or_load(...)` — learn the two GK zones (cached pkl).
4. Loop frames up to `PROCESS_DURATION_SEC` (default 60 s):
   detect → classify → GK → track → ball Kalman → carrier → pass FSM → stats → draw → write.
5. Save `data/output/possession_output.mp4`, print possession + pass summaries.

Mode A **renders** video. It is the reference implementation / debugging harness.

> Tuning note: the event/Kalman thresholds in `possession.py` are calibrated for
> 15 fps; `run()` rescales them by `eff_fps/15` so behaviour holds when processing at
> native fps. See [video_analysis/README.md](video_analysis/README.md).

---

## 3. Mode B — Production service (`service/`)

Two processes from one Docker image:

### API process (`service/api.py`, no GPU)
- `POST /api/clips` accepts a 60 s mp4 + `match_id` (+ half/minute/team metadata).
- Uploads the clip to Azure Blob, enqueues an Azure Queue message, returns `202`.
- Never processes inline. Also serves `GET /health` and `GET /metrics`.

### Worker process (`service/worker.py`, GPU)
Polls the queue. Per clip (`Worker._handle`):

1. **Poison guard** — `dequeue_count > MAX_DEQUEUE_COUNT` → poison queue.
2. **Idempotency** — if the `(match_id, half, minute)` minute row already exists, drop.
3. **Ordering guard** — `is_expected(...)`; out-of-order clips are deferred (re-queued
   hidden) up to `ORDERING_RETRIES`, then processed anyway with a logged gap.
4. **Session** — `MatchSessionManager.get_or_create(match_id)` returns the cross-clip
   `MatchSession`. On clip 1 (or a quality-guard refit on clip 2) it runs the team fit.
5. **Process** — `clip_processor.process_clip(...)` runs the shared CV pipeline over the
   clip at `TARGET_PROCESS_FPS`, continuing the session's tracker/ball/carrier/pass state.
6. **One SQL transaction** — `db.write_clip_result(...)` writes the minute row, any
   prior-minute correction, the events, the match progress, and the outbox row.
7. **Callback** — `notifier.send_pending_for_match(...)` POSTs cumulative stats in order.
8. **Cleanup** — delete queue message + blob + local job dir; evict idle sessions.

For `kind = post_processing` queue messages, the worker takes a separate but still
shared path: it ensures the match metadata exists, fits/loads the team classifier from
the whole uploaded video if needed, reuses the same shared CV pipeline over the entire
match file, writes one aggregate row to the `post_processing` table, and then deletes
the blob.

Mode B **does not render** anything. Its outputs are SQL rows and HTTP callbacks.

See [service/README.md](service/README.md) and [docs/API.md](../API.md).

---

## 4. The shared data types (the "wire format" between stages)

Defined in `detectors/player_detector.py`:

- **`Detection`** — one detected object: `bbox`, `confidence`, `foot_point`,
  `centre_point`, and the mutable fields later stages fill in:
  `team_id` (TeamClassifier), `is_goalkeeper` (GoalkeeperDetector),
  `track_id` (PlayerTracker), `embedding` (SigLIP, set during `classify`), `smoothed_bbox`
  (PlayerTracker's Kalman-smoothed box, drawing-only — `bbox` itself is untouched).
- **`FrameDetections`** — all detections for one frame, split into
  `players`, `referees`, `goal_posts`, `balls`, and `all`.

Every stage takes `FrameDetections` (or its `.players`) and **mutates Detections
in place** rather than returning new objects. This is the key to understanding the
pipeline: the same Detection object accumulates `team_id`, then `is_goalkeeper`, then
`track_id` as it flows through.

`video_analysis/possession.py` adds two more:
- **`BallDetection`** — the ball's `bbox`/`centre`/`confidence` (what `BallTracker` and
  `CarrierEngine` consume). `best_ball(fd)` picks the top-confidence ball from `fd.balls`.
- **`CarrierState`** — the per-frame answer to "who has the ball" (`carrier`/`loose`/`oof`).

---

## 5. Module connection map

Arrows mean "calls / imports / feeds data to".

```
detectors/player_detector.py ─┬─> team_classifier/team_classifier.py
  (Detection, FrameDetections)│      (reads .players, sets .team_id + .embedding)
                              ├─> detectors/goalkeeper_detector.py
                              │      (reads .goal_posts + .players, sets .is_goalkeeper/.team_id)
                              ├─> tracking/player_tracker.py
                              │      (reads .players + .embedding, sets .track_id)
                              └─> video_analysis/possession.py::best_ball
                                     (reads .balls -> BallDetection)

video_analysis/possession.py  (BallTracker, CarrierEngine, PassEventTracker,
  PossessionStats, POSSESS_*/EVT_*/PHASE_* constants, best_ball)
        ▲
        │ imported by
        │
service/session.py ───────────> wraps the pipeline classes into a MatchSession
service/clip_processor.py ────> drives one clip through them (batched two-pass)
service/stats.py ─────────────> mirrors POSSESS_*/EVT_* as dependency-free constants
                                 (session.py asserts the strings match at import)

service/worker.py  → orchestrates: queueing + blob + session + clip_processor + db + notifier
service/api.py     → ingest: blob + queueing + db
service/db.py      → Azure SQL (matches, minute_stats, events, callback_outbox)
service/notifier.py→ HTTP callback from callback_outbox rows
detectors/cache.py → pkl path helper used by both team_classifier + goalkeeper_detector
```

### Critical contract: the constant-string mirror

`service/stats.py` re-declares the possession labels (`team0/team1/loose/oof`) and event
kinds (`completed/interception/ball_lost`) as plain strings so the counting logic stays
torch-free and unit-testable. `service/session.py` **asserts at import** that these match
`video_analysis.possession`'s constants. If you rename a constant in `possession.py`,
the service will fail fast at startup until `stats.py` is updated too.

---

## 6. The unified detection model (recent change)

The pipeline used to run **two** detection models per frame (YOLOv11 for players + RF-DETR
for the ball). It now runs **one** unified YOLOv11m model that emits all four classes
(`active_player`, `ball`, `goal_post`, `referee`). The ball comes free from the same
forward pass via `best_ball()`. The old separate RF-DETR ball model and its weights were
removed entirely; the write-up that tracked this migration (`yolo_change.md`) has since
been deleted from the repo. Out of scope for that change (still use older detectors):
`heatmap.py`, `shots_on_t.py`.

---

## 7. Correctness machinery worth knowing about

These are the non-obvious invariants that keep the stats trustworthy:

- **Provisional credit + retroactive adjustment.** During a pass's "travel" phase the
  passer's team is credited provisionally. If the pass turns out to be an interception
  (`flip_to`) or the ball is lost (`drop`), the frames are corrected afterward — in
  local mode via `PossessionStats.apply_adjustments`, in service mode via
  `MinuteCounters.apply_adjustment` and, when the pass spans a clip boundary, a
  `PriorCorrection` UPDATE to the previous minute row.
- **Cumulative-on-read.** `minute_stats` stores **raw per-minute** counters only.
  Cumulative totals are `SUM()`-ed at read time (`db.cumulative_read`); percentages are
  derived *after* summing, never by averaging per-minute percentages.
- **Transactional outbox.** The callback payload row is written in the *same* SQL
  transaction as the minute row, so a stat is never sent without being persisted, and a
  persisted stat is always eventually sent (or marked `failed`).
- **Idempotency + ordering.** Replayed queue messages are dropped via the minute-row
  primary key; out-of-order clips are deferred then processed-with-a-gap.

Full detail in [service/README.md](service/README.md).
