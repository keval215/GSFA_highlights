# `video_analysis/` + `modules/possession/`

The heart of the project. `modules/possession/` defines the "who has the ball" logic, the
pass state machine, and the possession bookkeeping that **both** execution modes reuse.
`video_analysis/` holds the local dev entrypoint (`run.py`) plus homography/keypoint
tooling and log inspectors.

> **Recent change:** the old monolithic `video_analysis/possession.py` (classes +
> `run()` in one file) was deleted and split: the reusable classes moved to
> `modules/possession/{ball_tracker,carrier_engine,pass_event_tracker,possession_stats,labels}.py`
> (so they live alongside `modules/detectors/`, `modules/team_classifier/`,
> `modules/tracking/`), and the local-script `run()` became `video_analysis/run.py`. The
> logic itself is unchanged — this was a pure relocation plus ruleset-parametrization
> (see below), not a rewrite of the possession/pass math.

| File | One-line role |
|---|---|
| `run.py` | **Local dev entrypoint**: `python video_analysis/run.py [--ruleset futsal\|classic]` — process one whole video file, render an annotated `.mp4`, print a possession/pass summary |
| `simple_homography.py` | Keypoint (pose) detection → per-frame homography → pitch overlay + coverage heatmap |
| `inspect_log.py`, `inspect_log2.py` | Throwaway helpers to inspect `keypoint_log.json` output |
| `__init__.py` | Package marker |

`modules/possession/`:

| File | One-line role |
|---|---|
| `labels.py` | `POSSESS_*` / `EVT_*` string constants — split out so `pass_event_tracker.py` and `possession_stats.py` don't depend on each other |
| `ball_tracker.py` | `BallDetection`, `BallTracker` (Kalman), `best_ball(fd)` |
| `carrier_engine.py` | `CarrierState`, `CarrierEngine` (foot-zone + hysteresis) |
| `pass_event_tracker.py` | `PassEvent`, `PassEventTracker` (the release/travel/reception FSM), `PHASE_*` constants |
| `possession_stats.py` | `PossessionStats` (local-mode strict-denominator counter) |
| `__init__.py` | Re-exports all of the above as `modules.possession.*` |

---

## `modules/possession/` — the pipeline core

Imported by both `video_analysis/run.py` and the production service
(`service/session.py`, `service/clip_processor.py`, `service/stats.py`).

### Reusable building blocks

| Symbol | Role |
|---|---|
| `BallDetection` | Dataclass: ball `bbox` / `centre` / `confidence`. |
| `best_ball(fd)` | Pick the highest-confidence ball from `FrameDetections.balls` → `BallDetection` (or `None`). The bridge from the unified detector to the ball tracker. |
| `BallTracker` | Constant-velocity **Kalman** filter around the ball centre. Smooths position, **coasts** through detection gaps up to `coast_frames`, and rejects impossible jumps via a Mahalanobis gate (`gate_sigma`). States: `DETECTED` / `COASTING` / `LOST`. `coast_frames`/`gate_sigma` are now constructor params (`RulesetConfig.kalman_coast_frames`/`kalman_gate_sigma`), defaults unchanged. |
| `CarrierState` | Per-frame "who has the ball": `kind ∈ {carrier, loose, oof}` (+ track/team/player). |
| `CarrierEngine` | Computes the carrier from ball position + player **foot zones**, with an N-frame **hysteresis** buffer to debounce one-frame ID flickers. `hysteresis_n`/`foot_zone_ratio`/`foot_zone_min_px`/`foot_zone_max_px` are now constructor params (`RulesetConfig` fields), defaults unchanged. |
| `PassEventTracker` | The **pass FSM** (idle → in_possession → candidate_release → travel → candidate_reception). Emits `completed` / `interception` / `ball_lost` events and per-frame possession labels, plus retroactive **adjustments**. `release_sustain`/`reception_settle`/`travel_min_gap`/`travel_timeout` are now constructor params (`RulesetConfig` fields), defaults unchanged. |
| `PossessionStats` | Counts frames per label and computes possession % over the strict denominator (team_a+team_b only). Applies adjustments (`flip_to` / `drop`). Used only by `run.py`'s printed summary — the service reimplements equivalent counting independently in `service/stats.py::MinuteCounters`. |
| Constants | `POSSESS_TEAM_A/TEAM_B/LOOSE/OOF`, `EVT_COMPLETED/INTERCEPTION/BALL_LOST` (`labels.py`), `PHASE_*` (`pass_event_tracker.py`). **Mirrored** (as strings) in `service/stats.py`; `service/session.py` asserts they match. |

### How the carrier is decided (`CarrierEngine`)
- Foot zone radius = `foot_zone_ratio × bbox_height`, clamped to `[foot_zone_min_px,
  foot_zone_max_px]` (all `RulesetConfig` fields; futsal defaults `0.45`, `20`, `140`).
- A player is a candidate if the ball centre is within their foot zone. The nearest
  candidate becomes the carrier — **unless** players from both teams are in the zone, in
  which case the frame is `loose` (contested). Ball lost/untracked ⇒ `oof`.
- Raw per-frame state is pushed through a hysteresis deque; a change of `(kind, track_id)`
  commits only after `carrier_hysteresis_n` (default 3) consecutive agreeing samples.

### How passes are resolved (`PassEventTracker`)
- A confirmed carrier enters `in_possession`. When the ball leaves their foot zone,
  `candidate_release` waits `release_sustain` frames to confirm a real release (vs a
  flicker), then `travel`.
- In `travel`, a new player's foot zone holding the ball opens `candidate_reception`,
  which settles after `reception_settle` frames into a **completed** pass (same team) or
  **interception** (other team). No reception within `travel_timeout` (frames) ⇒
  **ball_lost**.
- During travel the passer's team gets **provisional** possession credit. On interception
  the credit is flipped (`flip_to` the receiver's team); on ball-lost it is dropped
  (`drop` → OOF). These corrections are the `adjustments` returned each frame.
- These four frame-count thresholds are calibrated at `reference_fps` (15.0 for futsal);
  callers rescale by `eff_fps/reference_fps` when processing at native fps.

### What `modules/possession/` does NOT do
- The reusable classes do not open files, hit the network, or render — they are pure
  per-frame logic (easy to drive from the service).
- It does **not** detect contested-possession or dribbles as events: two teams in the
  zone ⇒ `loose`; a passer reclaiming their own ball produces **no** event (by design —
  see `sql/schema.sql` header).
- Velocity from `BallTracker` includes camera motion and must **not** be used for
  kick/acceleration detection (the FSM is purely geometric, so this is fine).

### Connections
- **Imported by** `service/session.py` (the building blocks, constructed per-match from a
  `RulesetConfig`), `service/clip_processor.py` (`best_ball`), and `service/stats.py`
  mirrors its label/event constants.
- All ruleset-tunable parameters flow from `rulesets/base.py::RulesetConfig` — see
  [rulesets/README.md](../rulesets/README.md).

---

## `run.py` — the local dev entrypoint

### What it does
- `python video_analysis/run.py [--ruleset futsal|classic] [--video ...] [--out ...]
  [--team-a-gk-colour ...] [--team-b-gk-colour ...]`.
- Resolves the `RulesetConfig` (`rulesets.get_ruleset(args.ruleset)`, default `classic`),
  builds `PlayerDetector`/`GSFATeamClassifier` from it, and — only if both GK colour
  flags are supplied — a `GoalkeeperDetector` (no fit step; skipped entirely otherwise).
- Fits the team classifier (cached), then loops frames up to `PROCESS_DURATION_SEC`
  (default **480 s**), rendering every native frame.
- Rescales the FSM/Kalman thresholds from `reference_fps` to the actual fps
  (`fps_scale = eff_fps/reference_fps`, `_sc(...)`), so timing behaviour is
  fps-independent.
- Draws ellipses/labels (supervision), the ball triangle, a possession bar, and a pass
  overlay; writes `data/output/possession_output.mp4`; prints possession + pass summaries.
- Prints rich phase-transition and resolved-event logs (`NEW CARRIER`, `RELEASE
  CONFIRMED`, `PASS COMPLETED`, `INTERCEPTED`, `BALL LOST`, …).

### What it does NOT do
- Does **not** write to SQL or send callbacks — that's the service path.
- The config block at the top (`VIDEO_PATH`, `OUTPUT_PATH`, `DEVICE`, …) uses local
  Windows paths — this is a dev harness, not the production entry point.
- Does **not** pass `classes=` to `PlayerDetector` — every class the ruleset's model emits
  is detected (futsal: 4, incl. `goal_post`; classic: 3, no `goal_post`), same as the
  service path. The id→name map comes from `RulesetConfig.class_names`; `--model-classes`
  still overrides it for a one-off test model whose schema differs.

### Connections
- **Imports** `PlayerDetector`/`Detection`/`FrameDetections` (`modules/detectors/`),
  `GSFATeamClassifier` (`modules/team_classifier/`), `GoalkeeperDetector`
  (`modules/detectors/`), `PlayerTracker` (`modules/tracking/`), the `modules/possession/`
  building blocks, and `rulesets` for `RulesetConfig`/`get_ruleset`.

---

## `simple_homography.py`

### What it does
- Runs a **YOLO11m-pose** model per frame to detect pitch **keypoints**, filters them by
  confidence (corners get a lower threshold), and computes a per-frame **homography** (H)
  via `findHomography` (RANSAC). Frames without a valid H interpolate from neighbours.
- Reprojects a pitch wireframe (boundary, halfway line, goal mouths) back onto the image,
  writes an annotated video + debug frames + `keypoint_log.json`, and a top-down **pitch
  coverage heatmap** PNG. The module docstring documents the 13-keypoint layout.

### What it does NOT do
- Unrelated to possession/pass stats — it's pitch-geometry tooling.
- Not used by the service. Standalone dev script with hard-coded paths.
- **Out of scope for the `modules/` reorganisation** — it still imports the pre-refactor
  `detectors.cache` path (`from detectors.cache import cache_path as _cache_path` inside
  `build_h_list`), which no longer exists now that `detectors/` moved to
  `modules/detectors/`. This import will fail if `build_h_list` is actually called; not
  fixed here since it's flagged out of scope by the refactor itself (see
  [ARCHITECTURE.md §6](../ARCHITECTURE.md)) — surfacing it, not silently patching it.

### Connections
- Shares the homography concept with the now-deleted root-level `heatmap.py` (same pitch
  dimensions; see [scripts/README.md](../scripts/README.md)). Output JSON is what
  `inspect_log*.py` poke at.

---

## `inspect_log.py` / `inspect_log2.py`
Small throwaway scripts to print/inspect the `keypoint_log.json` produced by
`simple_homography.py`. Not part of any pipeline.
