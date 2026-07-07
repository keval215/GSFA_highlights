# Standalone scripts & experiments

Code that is **not** part of the production service or the core possession pipeline:
one-off experiments, alternate analytics, and scratch files. Most have hard-coded local
paths and a `CONFIG` block to edit before running.

| File | Role | Status |
|---|---|---|
| `heatmap.py` (repo root) | Player position heatmaps via homography | Working dev tool |
| `shots_on_t.py` (repo root) | Shots-on-target detector (Colab/T4) | Experimental |
| `scripts/diagnose_teams.py` | Diagnose team-classification quality | Utility |
| `scripts/get_csv.py` | Export per-match stats from Azure SQL to CSV | Working ops tool |
| `scripts/upload_clips.ps1` | Slice a local video into 60 s clips and POST them to `/api/clips` | Working ops tool |

The old OCR/team-classification probes (`scripts/01_ocr_score_overlay.py`,
`scripts/02_team_classification.py`, `scripts/crop_scoreboard.py`,
`scripts/test_endgame_ocr.py`) and the root-level `test.py` scratch file have been
deleted from the repo — they are no longer present, remove any mental model that
depends on them.

---

## `heatmap.py`
Runs `PlayerDetector` + `ColourHistogramTeamClassifier` + `GoalkeeperDetector` over every
frame, projects each player's `foot_point` to pitch metres via homography, and writes two
team heatmap PNGs over a top-down pitch diagram (`data/output/team{0,1}_heatmap.png`).

- **Uses the older detector stack** (colour-histogram classifier) — intentionally left
  untouched by the unified-model change (see [ARCHITECTURE.md §6](../ARCHITECTURE.md)).
- **Does NOT** compute possession/passes or touch the service/SQL.

## `shots_on_t.py`
A self-contained, paste-into-Colab shots-on-target detector. Gated ball-tracking cascade
(court mask → size → distance gate → physics gate), a track lifecycle state machine,
optical-flow camera-motion handling, and detection-tied goal planes (SAM2 from goalpost
bboxes). See its long module docstring for the full architecture.

- **Experimental / out of scope** for the unified-model change; keeps its own detectors.
- **Does NOT** integrate with `video_analysis/possession.py` or the service.

## `scripts/diagnose_teams.py`
Team-quality diagnostics probe. **Not imported** by any pipeline module — safe to ignore
for understanding the production flow.

## `scripts/get_csv.py`
CLI/ops tool: reads `minute_stats` from Azure SQL (`gsfa_stats`) via `pyodbc` and writes
one CSV per match with three row types:
- `minute` — one row per `minute_stats` row (raw frames + pass counts).
- `block_5min` — frames/pass counts summed into ~300 s (±20 s) blocks.
- `match_total` — a single summary row for the whole match.

It does **not** compute possession %, pass accuracy, or pass density itself — the module
docstring gives the exact formulas (`frames_team0 / (frames_team0+frames_team1+frames_loose)`
etc.) for the consuming analytics tool to derive. Config (`SQL_CONN_STR` or the individual
`GSFA_SQL_*` env vars, `GSFA_EXPORT_DIR`) is read from env vars with an editable `CONFIG`
dict fallback. Usage: no args = export all matches in the DB; positional args or
`GSFA_MATCH_IDS` (comma-separated) = export specific match IDs (CLI args win). Depends on
`minute_stats.clip_duration_seconds` (see [sql/README.md](../sql/README.md)) for the
`actual_duration_sec` / pass-density column — rows written before that column existed
will have it `NULL`.

## `scripts/upload_clips.ps1`
PowerShell ops script: takes one local video file, uses `ffprobe`/`ffmpeg` to slice it
into sequential 60 s clips (re-encoded `mpeg4`/`aac`, faststart), and `curl.exe`-POSTs each
clip to `POST /api/clips` in order (`half`, incrementing `minute`, `clip_duration_seconds`
computed per clip, plus the four team name/colour params). Parameterised via
`-InputVideoPath`/`-ServerUrl`/`-MatchId`/`-Half`/`-Team0Name`/etc.; defaults in the file
are placeholder values for one specific match upload, not a template to run as-is.
