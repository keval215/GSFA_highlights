# `rulesets/`

Per-sport parameter profiles for the `modules/` CV pipeline. Every value that used to be
a scattered module-level constant (in the old `video_analysis/possession.py`,
`detectors/goalkeeper_detector.py`, `team_classifier/team_classifier.py`,
`tracking/player_tracker.py`) is now one field on a `RulesetConfig` dataclass, so futsal
and classic (11-a-side) football can have different tuning without editing source.

| File | One-line role |
|---|---|
| `base.py` | `RulesetConfig` — every sport-tunable CV parameter, in one dataclass |
| `futsal.py` | `FUTSAL` — today's pre-refactor constants, byte-identical (production default) |
| `classic.py` | `CLASSIC` — 11-a-side profile, several fields `PLACEHOLDER` (not production-ready) |
| `registry.py` | `get_ruleset(name)` / `available_rulesets()` / `DEFAULT_RULESET` |
| `__init__.py` | Re-exports the above |

---

## `base.py` — `RulesetConfig`

### What it does
- One frozen dataclass, grouped by the `modules/` class each field feeds:
  - **Detection** (`modules/detectors/player_detector.py`): `player_model_weights`,
    `player_conf` (0.50), `ball_conf` (0.25).
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

Production default (`DEFAULT_RULESET`). Every field is today's exact pre-refactor
constant — verified byte-identical to the values that used to be hardcoded, so this
refactor changes **no** futsal behaviour.

## `classic.py` — `CLASSIC`

11-a-side football profile. Structurally complete (every `RulesetConfig` field is set)
but **not production-usable yet**:
- `player_model_weights` points at a checkpoint that does not exist — there is no
  classic-trained YOLOv11m model. The service requires the `CLASSIC_PLAYER_WEIGHTS` env
  var to be set (see `service/config.py::player_weights`) before this ruleset can be
  selected in production; it is not set anywhere yet.
- Several numeric fields are explicitly commented `PLACEHOLDER` (crop/blur tuning,
  `track_buffer_frames_at_30fps`, Kalman coast/gate, foot-zone px bounds,
  `travel_timeout_frames`) — reasoned starting points based on classic football's wider
  broadcast framing, longer pitch, and larger roster, not numbers calibrated against real
  11-a-side footage.

### What it does NOT do
- Does **not** get exercised by any existing test or production match yet — selecting
  `ruleset=classic` today will fail fast at model load (`CLASSIC_PLAYER_WEIGHTS` unset)
  unless that env var and a real checkpoint are provided first.

## `registry.py`

- `get_ruleset(name) -> RulesetConfig` — raises `ValueError` (with the valid-name list)
  on an unknown name. `service/api.py` catches this and returns HTTP `422`.
- `available_rulesets() -> list[str]` — `["classic", "futsal"]`, sorted.
- `DEFAULT_RULESET = "futsal"` — used as the default form value on both upload endpoints
  and the default CLI value in `video_analysis/run.py`.

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
