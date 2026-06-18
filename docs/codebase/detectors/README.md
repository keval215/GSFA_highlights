# `detectors/`

Per-frame perception: turn a video frame into structured `Detection` objects, plus the
goalkeeper logic and the shared cache-path helper.

This package defines the **shared data types** (`Detection`, `FrameDetections`) that flow
through the entire pipeline — see [ARCHITECTURE.md §4](../ARCHITECTURE.md).

| File | One-line role |
|---|---|
| `player_detector.py` | Unified YOLOv11m detector; defines `Detection` + `FrameDetections` |
| `goalkeeper_detector.py` | Learns GK zones, marks the two goalkeepers per frame |
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
    `is_goalkeeper`, `track_id`, `embedding`.
  - **`FrameDetections`** — `frame_idx`, `timestamp_s`, and the lists
    `players` / `referees` / `goal_posts` / `balls` / `all`.
- Inference entry points:
  - `detect(frame, frame_idx, fps)` — one frame → `FrameDetections`.
  - `detect_batch(frames, frame_indices, fps)` — K frames in one GPU call (used by the
    service's batched pass-1); output identical to calling `detect` per frame.
  - `process_video(video_path, sample_every)` — convenience full-video loop.
- **Per-class confidence gating** (`_parse`): the model runs at a low floor
  (`conf=0.20`), then each box is dropped unless it clears its class threshold —
  `ball_conf=0.25`, everything else `player_conf=0.50`. The small/fast ball gets a
  lower bar than players.
- fp16 on CUDA (`half=True`) for throughput; fp32 on CPU.
- `draw()` renders boxes + foot points for debugging.

### What it does NOT do
- Does **not** pick "the" ball — it returns *all* ball boxes in `fd.balls`. Selecting the
  single best ball is `video_analysis/possession.py::best_ball`.
- Does **not** assign teams, goalkeepers, or track IDs — those fields start `None`/`False`
  and are filled by later stages.
- Does **not** know about RF-DETR any more (the old separate ball model was removed; see
  [`yolo_change.md`](../../../yolo_change.md)).
- The module-level `MODEL_PATH` default is a local Windows path; the service overrides it
  via `config.player_weights()`.

### Connections
- **Consumed by** every downstream stage: team classifier, goalkeeper detector, tracker,
  and `best_ball`. They all read/mutate the same `Detection` objects in place.
- `detect_batch` is the GPU hot path called by `service/clip_processor.py`.

---

## `goalkeeper_detector.py`

### What it does
Two-stage, **does not require tracking**:

1. **Fit** (`fit_from_video` / `fit_from_video_or_load`): scan the video ~1 fps. For each
   frame, sort goal posts left→right, find the player nearest each post, and accumulate
   those positions. Also accumulate per-team outfield centroids. The averaged
   nearest-player position per post = a **GK zone** centroid; each zone is assigned to
   whichever team centroid is closer (`_gk_teams`).
2. **Classify** (`classify(detections)`): per frame, for each goal post find the nearest
   player, set `is_goalkeeper = True`, and **override** its `team_id` with the fitted
   assignment for that post.

- Caches one pkl per match (`data/cache/<stem>_goalkeeper.pkl`) via `cache.cache_path`.

### What it does NOT do
- Does **not** track goalkeepers over time — it re-derives "nearest player to each post"
  every frame independently.
- Does **not** run the team classifier itself; `classify()` assumes the team classifier
  has already run on the same `FrameDetections` (it then corrects the GK's team).
- Handles at most **2** posts/frame (`posts[:2]`); extra posts are ignored.
- Requires goal posts to be detected during fit, or it raises.

### Connections
- **Reads** `FrameDetections.goal_posts` + `.players` from `PlayerDetector`.
- **Depends on** a team classifier having set `team_id` first (any of the
  `team_classifier/` variants — its `fit_from_video` takes a `team_clf` argument).
- Used in local mode by `possession.py::run`. **Not** used by the service's
  `clip_processor` per-frame loop (the service path focuses on team possession).

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

### Connections
- Used by `team_classifier/team_classifier.py`, `team_classifier/colour_histogram.py`,
  and `goalkeeper_detector.py` for their default save paths.

---

## `test_goalkeeper.py`
Manual harness that wires `PlayerDetector` + a team classifier + `GoalkeeperDetector` over
a sample video to eyeball GK detection. Not part of the automated test suite in `tests/`.
