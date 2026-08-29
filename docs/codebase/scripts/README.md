# Standalone scripts & experiments

Code that is **not** part of the production service or the core possession pipeline:
one-off experiments, alternate analytics, and scratch files. Most have hard-coded local
paths and a `CONFIG` block to edit before running.

| File | Role | Status |
|---|---|---|
| `scripts/diagnose_teams.py` | Diagnose team-classification quality | Utility |
| `scripts/get_csv.py` | Export per-match stats from Azure SQL to CSV | Working ops tool |
| `scripts/upload_clips.ps1` | Slice a local video into 60 s clips and POST them to `/api/clips` | Working ops tool |
| `scripts/colab_sam2_player_tracking.py` | Individual player tracking via SAM2 video predictor (Colab) | Research/experiment |
| `scripts/dam4sam_player_tracking.py` | Individual player tracking via DAM4SAM (Colab) | Research/experiment |
| `scripts/samurai_player_tracking.py` | Individual player tracking via SAMURAI (Colab) | Research/experiment |
| `scripts/colab_botsort_reid_tracking.py` | Production BoT-SORT + long-term re-ID overlay (Colab) | Research/experiment |
| `scripts/colab_mcbyte_tracking.py` | Individual player tracking via McByte, full mask mode (Colab) | Research/experiment |
| `scripts/mcbyte_tracking_test.py` | McByte feasibility test — GPU OOM check + re-ID gap analysis (local) | Research/experiment |

The root-level `heatmap.py`, `shots_on_t.py`, and `test.py`, and the old OCR/
team-classification probes (`scripts/01_ocr_score_overlay.py`,
`scripts/02_team_classification.py`, `scripts/crop_scoreboard.py`,
`scripts/test_endgame_ocr.py`) have all been deleted from the repo — they are no longer
present, remove any mental model that depends on them.

---

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
docstring gives the exact formulas (`frames_team_a / (frames_team_a+frames_team_b+frames_loose)`
etc.) for the consuming analytics tool to derive. CSV header columns are `team_a`/`team_b`
(and `passes_completed_team_a` …) since v6 — downstream consumers of the export must update. Config (`SQL_CONN_STR` or the individual
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
computed per clip, plus the four `team_a_*`/`team_b_*` name/colour params and `ruleset`,
default `classic`). Parameterised via
`-InputVideoPath`/`-ServerUrl`/`-MatchId`/`-Half`/`-TeamAName`/`-TeamBName`/`-TeamAColour`/`-TeamBColour`/`-Ruleset`;
defaults in the file are placeholder values for one specific match upload, not a template
to run as-is.

## Tracker research scripts (`scripts/colab_*`, `scripts/*mcbyte*`)

A batch of standalone, paste-into-Colab (mostly) scripts exploring alternative
individual-player tracking approaches — **not** integrated into `modules/tracking/` or
the production pipeline. Written to inform `docs/codebase/tracking/` decisions, not to
replace `PlayerTracker` (BoT-SORT) today. Each shares the same first two stages
(YOLO detection with the repo's model + SigLIP→UMAP→KMeans team classify, "same recipe as
`GSFATeamClassifier`") and differs only in the tracking stage:

- **`colab_sam2_player_tracking.py`** — feeds YOLO boxes as box prompts into SAM2's video
  predictor; SAM2's memory bank tracks identity through occlusion. Multi-object,
  one pass. Known limitation: a new player can take up to `REDETECT_STRIDE` frames to
  pick up an id.
- **`dam4sam_player_tracking.py`** — DAM4SAM (CVPR 2025) is single-object upstream, so
  this runs one DAM4SAM tracker per selected player and composites the masks. Players are
  selected via a numbered first-frame snapshot (no interactive GUI in Colab).
- **`samurai_player_tracking.py`** — SAMURAI (motion-aware SAM2) is also single-object
  upstream; same per-player selection UX as the DAM4SAM script, but runs one **offline**
  SAM2 sweep (`init_state`/`propagate_in_video`) per player rather than one online pass.
- **`colab_botsort_reid_tracking.py`** — runs production's exact BoT-SORT config
  (matching `modules/tracking/player_tracker.py`: `reid_model=None`, external SigLIP
  embeddings), then layers a **long-term re-ID** remap on top: every new BoT-SORT internal
  id is checked against a capped gallery of recently-lost stable ids (kept alive up to
  `LOST_GALLERY_TTL_SECONDS`), with a small local VLM (Qwen2-VL-2B-Instruct, 4-bit) called
  only to break ties in an ambiguous cosine-similarity band, plus a jersey-number OCR
  veto. Exists because BoT-SORT's own memory only covers ~2 s
  (`track_buffer_frames_at_30fps=60` scaled by frame rate) — anyone gone longer gets a
  brand-new id with nothing checking whether it's someone already seen.
- **`colab_mcbyte_tracking.py`** — feeds YOLO boxes into `McByteTracker.update()` every
  frame; full mode adds an internal SAM(vit_b) + Cutie mask-propagation cue on top of
  McByte's IoU/Hungarian matching + camera-motion compensation. Reported HOTA (paper):
  SoccerNet 85.0, SportsMOT 76.5.
- **`mcbyte_tracking_test.py`** — local-GPU (not Colab) feasibility test for McByte:
  checks it runs without CUDA OOM in both lightweight and full-mask modes, and empirically
  demonstrates (via `--gap-report`) that McByte's `lost_track_buffer` is a short-horizon
  **continuity** mechanism (IoU re-association against a Kalman-predicted position), **not**
  appearance-based re-identification — a player who leaves frame and reappears elsewhere
  on the pitch will not be re-matched even within the buffer window.

### What these do NOT do
- None of them are imported by `modules/`, `service/`, or `video_analysis/run.py` —
  purely standalone investigation.
- None have been adopted as a `PlayerTracker` replacement; `modules/tracking/` still uses
  BoT-SORT alone (no re-ID model, no SAM/Cutie mask propagation) in production.

### Connections
- Reference the same detection weights and `GSFATeamClassifier` recipe as production, but
  do not import `modules/` code directly — they are self-contained Colab-paste scripts
  (except `mcbyte_tracking_test.py`, which runs locally).
