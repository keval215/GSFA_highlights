# `rulesets/`

Per-sport parameter profiles for the `modules/` CV pipeline. Every value that used to be
a scattered module-level constant (in the old `video_analysis/possession.py`,
`detectors/goalkeeper_detector.py`, `team_classifier/team_classifier.py`,
`tracking/player_tracker.py`) is now one field on a `RulesetConfig` dataclass, so futsal
and classic (11-a-side) football can have different tuning without editing source.

| File | One-line role |
|---|---|
| `base.py` | `RulesetConfig` — every sport-tunable CV parameter, in one dataclass |
| `futsal.py` | `FUTSAL` — the original pre-refactor constants, byte-identical |
| `classic.py` | `CLASSIC` — 11-a-side profile; **the default ruleset** (`DEFAULT_RULESET`). Needs `CLASSIC_PLAYER_WEIGHTS` set on the deployment |
| `registry.py` | `get_ruleset(name)` / `available_rulesets()` / `DEFAULT_RULESET` |
| `__init__.py` | Re-exports the above |

---

## `base.py` — `RulesetConfig`

### What it does
- One frozen dataclass, grouped by the `modules/` class each field feeds:
  - **Detection** (`modules/detectors/player_detector.py`): `player_model_weights`,
    `player_conf` (0.50), `ball_conf` (0.25), `class_names` (per-ruleset detector
    id→name map; default = the 4-class futsal schema
    `{0: active_player, 1: ball, 2: goal_post, 3: referee}`).
  - **Team classifier** (`modules/team_classifier/team_classifier.py`): `torso_ratio`
    (0.55), `blur_threshold` (80.0), `min_crop_px` (32), `centre_crop_ratio` (0.50).
  - **Goalkeeper** (`modules/detectors/goalkeeper_detector.py`): `max_gk_colour_dist`
    (60.0).
  - **Tracking** (`modules/tracking/player_tracker.py`): BoT-SORT thresholds
    (`track_high_thresh`, `track_low_thresh`, `new_track_thresh`, `match_thresh`,
    `proximity_thresh`, `appearance_thresh`, `track_buffer_frames_at_30fps`).
  - **Ball tracking** (`modules/possession/ball_tracker.py`): `kalman_coast_frames`,
    `kalman_gate_sigma`.
  - **Carrier engine** (`modules/possession/carrier_engine.py`): `foot_zone_ratio`,
    `foot_zone_min_px`, `foot_zone_max_px`, `carrier_hysteresis_n`.
  - **Pass FSM** (`modules/possession/pass_event_tracker.py`): `release_sustain`,
    `reception_settle`, `travel_min_gap`, `travel_timeout_frames`, `reference_fps` (15.0 —
    the fps these frame-count thresholds were calibrated at; callers rescale by
    `eff_fps/reference_fps`).
- Deliberately **excludes** infra/operational settings that apply uniformly regardless of
  sport (device, `TARGET_PROCESS_FPS`, CMC method, batch window, DB/blob/queue config) —
  those stay in `service/config.py`, unchanged by this package's existence.

### Connections
- Consumed by `video_analysis/run.py` (local dev CLI, `--ruleset` flag) and
  `service/session.py` (`MatchSession.__init__`, `ModelBundle.player_detector`,
  `collect_crops`, `ensure_gk_ready`) — every `modules/*` class is constructed from one of
  these instances rather than reading module-level constants directly.

---

## `futsal.py` — `FUTSAL`

Every field is the original pre-refactor constant — verified byte-identical to the values
that used to be hardcoded, so this refactor changed **no** futsal behaviour. No longer the
default (see `classic.py`); still selectable with an explicit `ruleset=futsal` and
`PLAYER_WEIGHTS` set.

## `classic.py` — `CLASSIC`

11-a-side football profile and **the default ruleset** (`DEFAULT_RULESET`). Every
`RulesetConfig` field is set with a committed value tuned for classic football's wider
broadcast framing, longer pitch, and larger roster. The values are reasoned starting
points, not frame-by-frame calibrated numbers — `travel_timeout_frames`, the Kalman
coast/gate, the crop/blur tuning and the foot-zone px bounds are the fields most likely to
move once real match telemetry is in (noted inline in the file).

Fields that differ from `futsal`: `player_model_weights`, `class_names`
(`{0: active_player, 1: ball, 2: referee}` — a 3-class model, no `goal_post`),
`torso_ratio` (0.65), `blur_threshold` (55.0), `min_crop_px` (24),
`track_buffer_frames_at_30fps` (90), `kalman_coast_frames` (20), `kalman_gate_sigma`
(8.0), `foot_zone_min_px` (14), `foot_zone_max_px` (100), `travel_timeout_frames` (45).
Everything else matches `futsal`.

### Deployment requirement
- `player_model_weights` in the file is only the local-dev default for
  `video_analysis/run.py`. The service resolves classic weights from the
  `CLASSIC_PLAYER_WEIGHTS` env var (`service/config.py::player_weights`). Because classic
  is now the default, **`CLASSIC_PLAYER_WEIGHTS` must be set to a real classic-trained
  YOLOv11m checkpoint** on the VM, or the worker fails fast at model load on the first
  clip of every new match.
- That checkpoint is a **3-class** model (`active_players`, `ball`, `refree` — no
  `goal_post`). `classic.py::class_names` must match its id order; a count mismatch is
  **logged, not fatal**, by `PlayerDetector.__init__` (`"player model class count … !=
  configured map …"`). Verify with
  `docker compose exec worker python3.11 -c "import os; from ultralytics import YOLO; print(YOLO(os.environ['CLASSIC_PLAYER_WEIGHTS']).names)"`.

## `registry.py`

- `get_ruleset(name) -> RulesetConfig` — raises `ValueError` (with the valid-name list)
  on an unknown name. `service/api.py` catches this and returns HTTP `422`.
- `available_rulesets() -> list[str]` — `["classic", "futsal"]`, sorted.
- `DEFAULT_RULESET = "classic"` — used as the default form value on both upload endpoints
  and the default CLI value in `video_analysis/run.py`. A caller that omits `ruleset` gets
  classic; a caller that still sends `ruleset=futsal` explicitly is unaffected.

### Connections
- Single validated lookup point shared by `video_analysis/run.py`, `service/api.py`
  (request validation), and `service/worker.py` (resolves a match's fixed ruleset from
  `db.get_match_ruleset(...)` before creating/reusing a session, for both the live-clip
  and post-processing paths).

---

## What the package as a whole does NOT do
- Does not itself construct `modules/*` objects — it only supplies the parameters;
  `service/session.py` and `video_analysis/run.py` do the constructing.
- Does not validate that a ruleset's weights file exists — that's `config.player_weights`
  raising at load time via `_required(...)`.
- A match's ruleset, once created, cannot be changed — `db.ensure_match(..., ruleset=...)`
  only sets it on `INSERT`; there is no "change ruleset mid-match" path, by design (team
  fit, tracker, and pass-FSM state all assume one ruleset for the life of the match).
