# YOLO Change — Unify detection into a single YOLOv11 model

Replace the two detection models (YOLOv11 players + RF-DETR ball) with **one YOLOv11 model**
that detects all four classes: `ball`, `active_player`, `goal_post`, `referee`.

## Why
Today the pipeline runs **two** detection models per frame:
- `PlayerDetector` — YOLOv11 (`GSFA_PLAYER_DETECTION.pt`): `0=active_player, 1=goal_post, 2=referee`, conf 0.50.
- `BallDetector` — RF-DETR Medium (`gsfa_ball_detection.pth`): `ball_class_id=1`, conf 0.25, returns one best `BallDetection`.

A single YOLOv11 model removes the RF-DETR dependency, halves model loads/VRAM, and makes ball
detection free per frame (it comes out of the same forward pass). The model will be uploaded to the
VM disk; the path comes later.

## Decisions
- **Per-class confidence gating**: run YOLO at the lowest threshold (ball ~0.25) and filter each class
  to its own minimum in the parse step (active_player/referee/goal_post >= 0.50, ball >= 0.25).
- **Class mapping read dynamically from `model.names`** via an alias table, so we don't depend on the
  trained index order. Exact alias strings to be confirmed against the uploaded model's `names`.
- **Scope**: production service path + `video_analysis/possession.py run()`.
  **Out of scope:** `heatmap.py` and `shots_on_t.py` (leave their detectors untouched).

## Current class mapping (for reference)
- Players/posts/referees: YOLOv11 `GSFA_PLAYER_DETECTION.pt` — `0=active_player, 1=goal_post, 2=referee` (conf 0.50).
- Ball: RF-DETR Medium `gsfa_ball_detection.pth` — `num_classes=2`, `ball_class_id=1`, conf 0.25.

## Changes

### 1. `detectors/player_detector.py` (core)
- `CLASS_NAMES`: build the index→role map from `self.model.names` at construction, normalized through an
  alias dict (defaults: `active_player`, `goal_post`, `referee`, `ball`; plus common synonyms like
  `player`→`active_player`). Keep a hardcoded fallback if a name is unrecognized.
- `FrameDetections`: add `balls: list[Detection]` (mirrors `players/referees/goal_posts`).
- Add per-class conf: a `CLASS_CONF` map (`active_player/referee/goal_post: 0.50`, `ball: 0.25`). Run
  `self.model(..., conf=min(CLASS_CONF.values()))` and in `_parse` drop any box below its class minimum.
- `_parse`: route ball-class boxes into `fd.balls` (and `fd.all`); everything else unchanged.
- `draw()`: add a colour for `ball`.
- `MODEL_PATH` default left as-is (overridden by config in the service; local default updated to the new
  unified weights path once known).

### 2. `video_analysis/possession.py`
- Keep `BallDetection`, `BallTracker`, `CarrierEngine`, `PassEventTracker`, `PossessionStats` unchanged.
- **Remove** the RF-DETR `BallDetector` class and the `from rfdetr import RFDETRMedium` usage.
- Add a small adapter `best_ball(fd: FrameDetections) -> Optional[BallDetection]` that picks the
  highest-confidence `ball` Detection from `fd.balls` and wraps it into a `BallDetection`
  (bbox, centre, confidence) — the bridge into `BallTracker`. No second model call.
- `run()`: drop `ball_det_model = BallDetector(...)`; replace `raw_ball = ball_det_model.detect(frame)`
  with `raw_ball = best_ball(player_dets)`.

### 3. `service/session.py`
- `ModelBundle`: remove the `ball_det: BallDetector` field and its `load()` wiring; keep only the unified
  `player_det`. Drop the `BallDetector` import and the `config.ball_weights()` reference in the log line.

### 4. `service/clip_processor.py`
- Pass 1: ball now comes from the same `player_det.detect_batch(...)` result. Replace the separate
  `session.models.ball_det.detect_batch(...)` block (and the `BALL_DETECT_EVERY` stride logic) with
  `balls = [best_ball(d) for d in dets_list]` — ball detected every processed frame (improves Kalman
  continuity vs the old stride).
- Fold the `ball_det` timing bucket into `player_det` (or drop it); update the timing log line.

### 5. `service/config.py`
- Remove `ball_weights()` / `BALL_WEIGHTS` and `BALL_DETECT_EVERY`.
- Add per-class conf tunables (`PLAYER_CONF` default 0.50, `BALL_CONF` default 0.25) consumed by
  `PlayerDetector`. `player_weights()` stays as the single detection-model path (the new unified weights).

### 6. Dependencies / deploy
- `requirements-service.txt`: remove `rfdetr`.
- `Dockerfile`: drop the rfdetr-driven rationale comment. **Keep deadsnakes stable Python 3.11** — it is
  the safe interpreter regardless; only the *justification* (rfdetr `@torch.jit.script` segfault on
  Ubuntu's 3.11.0rc1) is removed. Do not change the base Python in this change.
- VM env (`/etc/gsfa-highlights.env`): `BALL_WEIGHTS` no longer needed; `PLAYER_WEIGHTS` → new model path.

### 7. Out of scope (do not modify)
- `heatmap.py` (root) and `shots_on_t.py` — keep their existing detectors/RF-DETR as-is.

## Open item (needs the model)
- Confirm the uploaded model's `model.names` strings so the alias table maps them correctly, and confirm
  ball is detected at the imgsz we run (ball is small; RF-DETR used resolution 576 — validate YOLO recall
  at imgsz 640, raise imgsz if ball recall drops).

## Verification
- Unit: `PlayerDetector` on a frame returns balls in `fd.balls`, players/refs/posts gated at their
  thresholds; `best_ball()` returns the top-conf ball or None.
- Compile: `python -m py_compile` on the four edited modules.
- End-to-end (service): run the worker on a known clip; confirm one model loads (no RF-DETR log), ball
  appears in `CarrierEngine`/possession output, and `events`/`minute_stats` are comparable to the
  two-model baseline on the same clip.
- Local: `python video_analysis/possession.py` renders the overlay with ball triangle + possession bar.
- Regression: `detectors/test_goalkeeper.py` and `team_classifier/test_team_classifier.py` still pass
  (they use `dets.players/.goal_posts/.referees`, unaffected).
