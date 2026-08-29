# `modules/detectors/`

Per-frame perception: turn a video frame into structured `Detection` objects, plus the
goalkeeper logic and the shared cache-path helper. (Moved here from top-level
`detectors/` — same files, package reorganised under `modules/` alongside
`team_classifier/`, `tracking/`, and the new `possession/`.)

This package defines the **shared data types** (`Detection`, `FrameDetections`) that flow
through the entire pipeline — see [ARCHITECTURE.md §4](../ARCHITECTURE.md).

| File | One-line role |
|---|---|
| `player_detector.py` | Unified YOLOv11m detector; defines `Detection` + `FrameDetections` |
| `goalkeeper_detector.py` | Classifies the two goalkeepers per frame by jersey-colour match |
| `cache.py` | Derives a per-match pkl path from the video filename |
| `__init__.py` | Package marker (empty) |
| `test_goalkeeper.py` | Manual/visual test harness for GK detection |

---

## `player_detector.py`

### What it does
- Wraps the **unified YOLOv11m** model (trained at `imgsz=960`, 4 classes):
  `0=active_player, 1=ball, 2=goal_post, 3=referee` (see `PlayerDetector.CLASS_NAMES`).
- Defines the two core data types the whole codebase passes around:
  - **`Detection`** — `class_id`, `class_name`, `bbox`, `confidence`, `foot_point`,
    `centre_point`, plus mutable fields filled in by later stages: `team_id`,
    `is_goalkeeper`, `track_id`, `embedding`, `smoothed_bbox` (BoT-SORT's Kalman-smoothed
    box, set by `PlayerTracker` — used for jitter-free drawing; `bbox` itself is left
    untouched so foot-zone/carrier geometry is unaffected).
  - **`FrameDetections`** — `frame_idx`, `timestamp_s`, and the lists
    `players` / `referees` / `goal_posts` / `balls` / `all`.
- Inference entry points:
  - `detect(frame, frame_idx, fps)` — one frame → `FrameDetections`.
  - `detect_batch(frames, frame_indices, fps)` — K frames in one GPU call (used by the
    service's batched pass-1); output identical to calling `detect` per frame.
  - `process_video(video_path, sample_every)` — convenience full-video loop.
- **Per-class confidence gating** (`_parse`): the model runs at a low floor
  (`conf=0.20`), then each box is dropped unless it clears its class threshold —
  `player_conf`/`ball_conf` constructor params, defaulting to `0.50`/`0.25` (the
  small/fast ball gets a lower bar than players). A `RulesetConfig` supplies these two
  per match/run (`rulesets/base.py`); both rulesets currently use the same defaults.
- `PlayerDetector.__init__` also takes an optional `classes: list[int] | None` — a class
  allow-list passed straight through to the Ultralytics call (`conf`, `imgsz`, `half`,
  and now `classes`). `None` (default) keeps all four classes, unchanged for the
  service/VM and `video_analysis/run.py` paths — only the manual debug harness
  `modules/team_classifier/test_team_classifier.py` actually passes `classes=[0,1,2]` to
  hard-filter referees.
- fp16 on CUDA (`half=True`) for throughput; fp32 on CPU.
- `draw()` renders boxes + foot points for debugging.

### What it does NOT do
- Does **not** pick "the" ball — it returns *all* ball boxes in `fd.balls`. Selecting the
  single best ball is `modules/possession/ball_tracker.py::best_ball`.
- Does **not** assign teams, goalkeepers, or track IDs — those fields start `None`/`False`
  and are filled by later stages.
- Does **not** know about RF-DETR any more (the old separate ball model was removed
  entirely — the ball is now `class_id=1` in the unified model's own output).
- The module-level `MODEL_PATH` default is a local Windows path; the service overrides it
  via `config.player_weights(ruleset_name)` (per-ruleset weights — see
  [rulesets/README.md](../rulesets/README.md)).

### Connections
- **Consumed by** every downstream stage: team classifier, goalkeeper detector, tracker,
  and `best_ball`. They all read/mutate the same `Detection` objects in place.
- `detect_batch` is the GPU hot path called by `service/clip_processor.py`.

---

## `goalkeeper_detector.py`

### What it does
**Rewritten** — no longer a fit-then-classify module. There is **no fit stage, no
goal-post dependency, and no tracking**: a player IS the goalkeeper because their jersey
colour matches a known reference colour supplied by the caller.

- `__init__(team_a_gk_colour, team_b_gk_colour, max_colour_dist=60.0, *, torso_ratio=0.55,
  centre_crop_ratio=0.50)` — the two reference colours (hex or CSS name, same format as
  outfield team colours) are converted once to `GSFATeamClassifier`'s cylindrical HSV
  colour-vector space (`_hsv_vec(_colour_to_hsv(...))`) and stored. `torso_ratio` /
  `centre_crop_ratio` mirror `GSFATeamClassifier`'s camera-framing tuning — kept as
  independent parameters (not read from a `GSFATeamClassifier` instance) since this
  detector deliberately runs without depending on one; a `RulesetConfig` supplies
  matching values to both.
- `classify(frame, detections)`: for each reference colour, computes a colour vector for
  every player crop (`GSFATeamClassifier._torso_crop` + `_mean_colour_vec`) and picks the
  single closest-matching player. A match under `max_colour_dist` sets
  `is_goalkeeper=True` and `team_id=<that team>` on that player. No match under threshold
  ⇒ no GK flagged for that team this frame (a bad match is never forced onto the
  "least bad" player).
- Runs **independently of** `GSFATeamClassifier` ("in parallel") — both read the same
  frame + detections and write to the same `Detection` objects, but neither depends on
  the other having already run; call order doesn't matter.

### What it does NOT do
- Does **not** fit, cache a pkl, or require goal posts to be detected — construction is
  cheap enough to retry every clip until reference colours are known (see
  `service/session.py::MatchSession.ensure_gk_ready`).
- Does **not** track goalkeepers over time — it re-derives the closest colour match every
  frame independently.
- Does **not** assign a GK if the best-matching player's colour distance is `>=
  max_colour_dist` — a placeholder value (`MAX_GK_COLOUR_DIST = 60.0`) that needs
  calibration against real footage, same caveat as `BLUR_THRESHOLD` in
  `team_classifier.py`.

### Connections
- **Reads** `FrameDetections.players` from `PlayerDetector`; reuses
  `GSFATeamClassifier`'s static colour-vector helpers (`_hsv_vec`, `_colour_to_hsv`,
  `_torso_crop`, `_mean_colour_vec`) rather than depending on a classifier instance.
- Used in local mode by `video_analysis/run.py::run` (only constructed if both
  `--team-a-gk-colour`/`--team-b-gk-colour` are supplied) and in service mode by
  `service/session.py::MatchSession` / `service/clip_processor.py`.

---

## `cache.py`

### What it does
- `cache_path(video_path, suffix) -> Path` derives a deterministic, match-specific pkl
  path under `CACHE_DIR` (`data/cache/`): `Path(video).stem.lower().replace(" ","_") + "_" + suffix + ".pkl"`.
- Creates the cache directory if needed.

### What it does NOT do
- Does **not** read/write the pkl itself (callers use `joblib`).
- `CACHE_DIR` is a hard-coded absolute Windows path (`D:\GSFA_highlights\data\cache`) —
  this is a local-dev helper. The **service** does not use it; it stores fits under
  `MATCH_STATE_DIR/<match_id>/team_siglip.pkl` (see `service/session.py`).
- `GoalkeeperDetector` no longer uses it (no fit step, no cache) — only
  `team_classifier.py` and `colour_histogram.py` still do.

### Connections
- Used by `modules/team_classifier/team_classifier.py` and
  `modules/team_classifier/colour_histogram.py` for their default save paths.

---

## `test_goalkeeper.py`
Manual harness that wires `PlayerDetector` + a team classifier + `GoalkeeperDetector` over
a sample video to eyeball GK detection. Not part of the automated test suite in `tests/`.
