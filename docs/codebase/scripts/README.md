# Standalone scripts & experiments

Code that is **not** part of the production service or the core possession pipeline:
one-off experiments, alternate analytics, and scratch files. Most have hard-coded local
paths and a `CONFIG` block to edit before running.

| File | Role | Status |
|---|---|---|
| `heatmap.py` (repo root) | Player position heatmaps via homography | Working dev tool |
| `shots_on_t.py` (repo root) | Shots-on-target detector (Colab/T4) | Experimental |
| `test.py` (repo root) | Scratch / throwaway | Ignore |
| `scripts/01_ocr_score_overlay.py` | OCR the scoreboard ROI (EasyOCR) | Experiment |
| `scripts/02_team_classification.py` | Player detect + HSV-histogram team clustering probe | Experiment |
| `scripts/crop_scoreboard.py` | Crop the scoreboard region from a frame | Utility |
| `scripts/diagnose_teams.py` | Diagnose team-classification quality | Utility |
| `scripts/test_endgame_ocr.py` | OCR probe on end-of-game overlays | Experiment |

---

## `heatmap.py`
Runs `PlayerDetector` + `ColourHistogramTeamClassifier` + `GoalkeeperDetector` over every
frame, projects each player's `foot_point` to pitch metres via homography, and writes two
team heatmap PNGs over a top-down pitch diagram (`data/output/team{0,1}_heatmap.png`).

- **Uses the older detector stack** (colour-histogram classifier) — intentionally left
  untouched by the unified-model change (see `yolo_change.md` §7).
- **Does NOT** compute possession/passes or touch the service/SQL.

## `shots_on_t.py`
A self-contained, paste-into-Colab shots-on-target detector. Gated ball-tracking cascade
(court mask → size → distance gate → physics gate), a track lifecycle state machine,
optical-flow camera-motion handling, and detection-tied goal planes (SAM2 from goalpost
bboxes). See its long module docstring for the full architecture.

- **Experimental / out of scope** for the unified-model change; keeps its own detectors.
- **Does NOT** integrate with `video_analysis/possession.py` or the service.

## `scripts/` (probes & utilities)
Small investigative scripts used while building the pipeline — OCR of the scoreboard
(EasyOCR, because PaddleOCR has a Windows runtime bug), an HSV-histogram team-clustering
experiment (the precursor reasoning behind torso-cropping in the real classifier),
scoreboard cropping, and team-quality diagnostics. They are **not imported** by any
pipeline module and are safe to ignore for understanding the production flow.

## `test.py`
Root-level scratch file. Not a real test (the maintained tests live in `tests/`). Ignore.
