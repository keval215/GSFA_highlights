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
previous stage's output. `GoalkeeperDetector` runs **independently of** (not sequentially
after) `GSFATeamClassifier` — both read the same frame + detections and write to the same
`Detection` objects, but neither depends on the other's output. The chain is:

```
        ┌─────────────┐   ┌──────────────┐
frame → │ PlayerDetector│ →│GSFATeamClassif│──┐                       (per-frame perception)
        │  (YOLOv11m)  │   │ (SigLIP team) │  │
        └─────────────┘   └──────────────┘   │
                │           ┌──────────────────┐
                └──────────>│GoalkeeperDetector│  (runs independently — GK colour match)
                            │  (GK by colour)  │
                            └──────────────────┘
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

Every numeric parameter these stages are tuned with (model weights, confidence gates,
team-classifier crop/blur tuning, GK colour-match distance, tracker thresholds, Kalman
params, foot-zone sizing, pass-FSM timing) now comes from a **`RulesetConfig`**
(`rulesets/` package) selected per match — see [§8](#8-rulesets-per-sport-tuning) and
[rulesets/README.md](rulesets/README.md). `futsal` (the only production-ready ruleset)
carries the exact pre-refactor constants; `classic` (11-a-side) is a structurally-complete
but not-yet-calibrated profile.

The dividing line is important:

- **Perception + "who has the ball" + the pass FSM** are written **once**, in
  `modules/possession/` and `modules/detectors/`, `modules/team_classifier/`,
  `modules/tracking/` — parametrized per-call by a `RulesetConfig`.
- **Where the numbers go** differs by mode: local mode renders an overlay and prints a
  summary; service mode buckets the same labels into per-minute SQL rows.

---

## 2. Mode A — Local script (`video_analysis/run.py`)

**Entry:** `python video_analysis/run.py [--ruleset futsal|classic]` → `run()`.
(`video_analysis/possession.py`, the old monolithic module, was deleted — its reusable
classes moved to `modules/possession/*.py` and its `run()` became this file.)

1. Resolve the `RulesetConfig` (`rulesets.get_ruleset(args.ruleset)`, default `classic`).
2. Build `PlayerDetector` (unified YOLOv11m, ruleset weights/conf), `GSFATeamClassifier`
   (ruleset crop/blur tuning). `GoalkeeperDetector` is only constructed if both
   `--team-a-gk-colour`/`--team-b-gk-colour` are supplied (direct colour match, no fit step).
3. `team_clf.fit_from_video_or_load(...)` — fit SigLIP+UMAP+KMeans once (cached pkl).
4. Loop frames up to `PROCESS_DURATION_SEC` (default 480 s):
   detect → classify → GK (if enabled) → track → ball Kalman → carrier → pass FSM → stats
   → draw → write.
5. Save `data/output/possession_output.mp4`, print possession + pass summaries.

Mode A **renders** video. It is the reference implementation / debugging harness.

> Tuning note: the event/Kalman thresholds come from the selected `RulesetConfig`,
> calibrated at `reference_fps` (15 for futsal); `run()` rescales them by
> `eff_fps/reference_fps` so behaviour holds when processing at native fps. See
> [video_analysis/README.md](video_analysis/README.md).

---

## 3. Mode B — Production service (`service/`)

Two processes from one Docker image:

### API process (`service/api.py`, no GPU)
- `POST /api/clips` accepts a clip mp4 (≈60 s by convention — not enforced; see
  [service/README.md](service/README.md) "Why clips stay ~60 s") + `match_id` + team/GK colours (+ half/minute/team
  name/`ruleset` metadata).
- Uploads the clip to Azure Blob, enqueues an Azure Queue message, returns `202`.
- Never processes inline. Also serves `GET /health` and `GET /metrics`.
- `ruleset` (`"futsal"` | `"classic"`, default `"classic"`) is validated against the
  `rulesets` registry (unknown value → `422`) and only takes effect on the **first**
  request for a `match_id` — it creates the `matches` row; a match's ruleset is fixed for
  its lifetime and later requests for the same `match_id` ignore the field.

### Worker process (`service/worker.py`, GPU)
Polls the queue. Per clip (`Worker._handle`):

1. **Poison guard** — `dequeue_count > MAX_DEQUEUE_COUNT` → poison queue.
2. **Idempotency** — if the `(match_id, half, minute)` minute row already exists, drop.
3. **Ordering guard** — `is_expected(...)`; out-of-order clips are deferred (re-queued
   hidden) up to `ORDERING_RETRIES`, then processed anyway with a logged gap.
4. **Ruleset + session** — `get_ruleset(db.get_match_ruleset(match_id))` resolves the
   match's fixed ruleset, then `MatchSessionManager.get_or_create(match_id, ruleset)`
   returns the cross-clip `MatchSession` (built from that ruleset on first creation only).
   On clip 1 (or a quality-guard refit on clip 2) it runs the team fit;
   `session.ensure_gk_ready()` lazily constructs the `GoalkeeperDetector` once the DB has
   both GK reference colours (cheap, retried every clip — no fit step needed).
5. **Process** — `clip_processor.process_clip(...)` runs the shared CV pipeline over the
   clip at `TARGET_PROCESS_FPS`, continuing the session's tracker/ball/carrier/pass state,
   including a per-frame `gk_det.classify(...)` call if the GK classifier is ready.
6. **One SQL transaction** — `db.write_clip_result(...)` writes the minute row, any
   prior-minute correction, the events, the match progress, and the outbox row.
7. **Callback** — `notifier.send_pending_for_match(...)` POSTs cumulative stats in order.
8. **Cleanup** — delete queue message + blob + local job dir; evict idle sessions.

For `kind = post_processing` queue messages, the worker takes a separate but still
shared path: it resolves the match's ruleset the same way, builds its own `MatchSession`
**directly** (never through `MatchSessionManager`, never reused across attempts — every
run starts from clean tracker/ball/carrier/pass-FSM state), reuses the same shared CV
pipeline over the entire match file, writes one aggregate row to the `post_processing`
table, and deletes the blob. The queue message is deleted **before** processing starts
(no lease renewal, no automatic retry) — a job that dies partway through (exception,
container OOM-kill, VM shutdown) is never silently redelivered on top of leftover state;
recovery is a manual re-upload. A failure is recorded in-memory and surfaced via
`GET /metrics`' `last_post_processing_error`, not retried.

Mode B **does not render** anything. Its outputs are SQL rows and HTTP callbacks.

See [service/README.md](service/README.md) and [docs/API.md](../API.md).

---

## 4. The shared data types (the "wire format" between stages)

Defined in `modules/detectors/player_detector.py`:

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

`modules/possession/` adds two more:
- **`BallDetection`** — the ball's `bbox`/`centre`/`confidence` (what `BallTracker` and
  `CarrierEngine` consume). `best_ball(fd)` picks the top-confidence ball from `fd.balls`.
- **`CarrierState`** — the per-frame answer to "who has the ball" (`carrier`/`loose`/`oof`).

---

## 5. Module connection map

Arrows mean "calls / imports / feeds data to".

```
modules/detectors/player_detector.py ─┬─> modules/team_classifier/team_classifier.py
  (Detection, FrameDetections)        │      (reads .players, sets .team_id + .embedding)
                                       ├─> modules/detectors/goalkeeper_detector.py
                                       │      (reads .players, GK colour match; independent
                                       │       of team_classifier — sets .is_goalkeeper/.team_id)
                                       ├─> modules/tracking/player_tracker.py
                                       │      (reads .players + .embedding, sets .track_id)
                                       └─> modules/possession/ball_tracker.py::best_ball
                                              (reads .balls -> BallDetection)

modules/possession/  (BallTracker, CarrierEngine, PassEventTracker,
  PossessionStats, POSSESS_*/EVT_*/PHASE_* constants, best_ball — split out of the old
  monolithic video_analysis/possession.py; video_analysis/run.py is now the local CLI)
        ▲
        │ imported by
        │
rulesets/ ─────────────────────> RulesetConfig selects the tuning every modules/* class
                                  above and service/session.py construct from
service/session.py ───────────> wraps the pipeline classes into a MatchSession
service/clip_processor.py ────> drives one clip through them (batched two-pass)
service/stats.py ─────────────> mirrors POSSESS_*/EVT_* as dependency-free constants
                                 (session.py asserts the strings match at import)

service/worker.py  → orchestrates: queueing + blob + session + clip_processor + db + notifier
service/api.py     → ingest: blob + queueing + db
service/db.py      → Azure SQL (matches, minute_stats, events, callback_outbox)
service/notifier.py→ HTTP callback from callback_outbox rows
modules/detectors/cache.py → pkl path helper used by both team_classifier + goalkeeper_detector
```

### Critical contract: the constant-string mirror

`service/stats.py` re-declares the possession labels (`team_a/team_b/loose/oof`) and event
kinds (`completed/interception/ball_lost`) as plain strings so the counting logic stays
torch-free and unit-testable. `service/session.py` **asserts at import** that these match
`modules.possession.labels`'s constants. If you rename a constant in `labels.py`,
the service will fail fast at startup until `stats.py` is updated too.

---

## 6. The unified detection model (recent change)

The pipeline used to run **two** detection models per frame (YOLOv11 for players + RF-DETR
for the ball). It now runs **one** unified YOLOv11m model that emits all four classes
(`active_player`, `ball`, `goal_post`, `referee`). The ball comes free from the same
forward pass via `best_ball()`. The old separate RF-DETR ball model and its weights were
removed entirely; the write-up that tracked this migration (`yolo_change.md`) has since
been deleted from the repo. The root-level `heatmap.py` and `shots_on_t.py` standalone
tools, which were out of scope for that change and kept the older separate detectors,
have themselves since been deleted from the repo — see
[scripts/README.md](scripts/README.md).

## 6a. The goalkeeper detector rewrite (recent change)

`GoalkeeperDetector` used to be a **fit-then-classify** module: scan the video once,
find the player nearest each detected goal post, average those positions into two "GK
zone" centroids, and assign each zone to whichever team's outfield centroid was closer.
Per frame it re-derived "nearest player to each post" and overrode that player's team.

It is now **fit-free**: the caller supplies two reference jersey colours
(`team_a_gk_colour`, `team_b_gk_colour`, same hex/CSS-name format as outfield team
colours), and `classify(frame, detections)` finds, independently for each colour, the
single player in the frame whose jersey colour is closest to it (reusing
`GSFATeamClassifier`'s HSV colour-vector helpers) — a match under `max_gk_colour_dist`
marks that player `is_goalkeeper=True`. No goal-post dependency, no tracking, no cache
pkl, and no ordering requirement relative to `GSFATeamClassifier` (see the diagram in
§1). In the service, `team_a_gk_colour`/`team_b_gk_colour` are **required** fields on both
upload endpoints (see [docs/API.md](../API.md)); `MatchSession.ensure_gk_ready()`
constructs the detector from clip 1 (a malformed colour disables GK for that match but
never fails the clip).

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

---

## 8. Rulesets — per-sport tuning

Every sport-tunable CV parameter (detection weights/conf, team-classifier crop/blur
tuning, GK colour-match distance, BoT-SORT thresholds, Kalman coast/gate, foot-zone
sizing, pass-FSM timing) lives in one `RulesetConfig` dataclass (`rulesets/base.py`).
`rulesets/futsal.py` (`FUTSAL`) carries today's exact pre-refactor constants — verified
byte-identical to the values that used to be scattered module-level globals.
`rulesets/classic.py` (`CLASSIC`, 11-a-side football) is structurally complete but has
several fields explicitly marked `PLACEHOLDER` pending real footage and a trained
classic-model checkpoint — **not yet production-usable**. `rulesets/registry.py`
(`get_ruleset(name)` / `available_rulesets()` / `DEFAULT_RULESET = "futsal"`) is the
single validated lookup point, used by both `video_analysis/run.py --ruleset` and
`service/`.

A match's ruleset is selected once (the `ruleset` form field on the **first** upload
request for a `match_id`) and fixed for that match's lifetime — `matches.ruleset` in SQL
(migration v4, not yet applied to the live DB), never updated after `INSERT`. The
`futsal` ruleset keeps reading the existing `PLAYER_WEIGHTS` env var (backward
compatible); other rulesets read `<RULESET>_PLAYER_WEIGHTS` (e.g.
`CLASSIC_PLAYER_WEIGHTS` — not yet set anywhere, since no classic-trained checkpoint
exists). `service/session.py::ModelBundle` loads one `PlayerDetector` per ruleset
lazily, on first match of that ruleset, rather than eagerly at worker startup.

See [rulesets/README.md](rulesets/README.md) for the full field list.
