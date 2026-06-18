# `video_analysis/`

The heart of the project. `possession.py` defines the "who has the ball" logic, the pass
state machine, and the possession bookkeeping that **both** execution modes reuse. The
other files here are homography/keypoint tooling and log inspectors.

| File | One-line role |
|---|---|
| `possession.py` | **Core**: ball tracking → carrier → pass FSM → possession stats (+ local renderer) |
| `simple_homography.py` | Keypoint (pose) detection → per-frame homography → pitch overlay + coverage heatmap |
| `inspect_log.py`, `inspect_log2.py` | Throwaway helpers to inspect `keypoint_log.json` output |
| `__init__.py` | Package marker |

---

## `possession.py` — the pipeline core

Despite the name, this single module holds most of the possession/pass intelligence and
is imported by the production service. It has two layers: **reusable classes** (used
everywhere) and a **local `run()`** (renders an annotated video).

### Reusable building blocks (imported by the service)

| Symbol | Role |
|---|---|
| `BallDetection` | Dataclass: ball `bbox` / `centre` / `confidence`. |
| `best_ball(fd)` | Pick the highest-confidence ball from `FrameDetections.balls` → `BallDetection` (or `None`). The bridge from the unified detector to the ball tracker. |
| `BallTracker` | Constant-velocity **Kalman** filter around the ball centre. Smooths position, **coasts** through detection gaps up to `KALMAN_COAST_FRAMES`, and rejects impossible jumps via a Mahalanobis gate. States: `DETECTED` / `COASTING` / `LOST`. |
| `CarrierState` | Per-frame "who has the ball": `kind ∈ {carrier, loose, oof}` (+ track/team/player). |
| `CarrierEngine` | Computes the carrier from ball position + player **foot zones**, with an N-frame **hysteresis** buffer to debounce one-frame ID flickers. |
| `PassEventTracker` | The **pass FSM** (idle → in_possession → candidate_release → travel → candidate_reception). Emits `completed` / `interception` / `ball_lost` events and per-frame possession labels, plus retroactive **adjustments**. |
| `PossessionStats` | Counts frames per label and computes possession % over the strict denominator (team0+team1 only). Applies adjustments (`flip_to` / `drop`). |
| Constants | `POSSESS_TEAM0/TEAM1/LOOSE/OOF`, `EVT_COMPLETED/INTERCEPTION/BALL_LOST`, `PHASE_*`. **Mirrored** (as strings) in `service/stats.py`; `service/session.py` asserts they match. |

### How the carrier is decided (`CarrierEngine`)
- Foot zone radius = `FOOT_ZONE_RATIO × bbox_height`, clamped to `[FOOT_ZONE_MIN_PX, FOOT_ZONE_MAX_PX]`.
- A player is a candidate if the ball centre is within their foot zone. The nearest
  candidate becomes the carrier — **unless** players from both teams are in the zone, in
  which case the frame is `loose` (contested). Ball lost/untracked ⇒ `oof`.
- Raw per-frame state is pushed through a hysteresis deque; a change of `(kind, track_id)`
  commits only after `CARRIER_HYSTERESIS_N` consecutive agreeing samples.

### How passes are resolved (`PassEventTracker`)
- A confirmed carrier enters `in_possession`. When the ball leaves their foot zone,
  `candidate_release` waits `RELEASE_SUSTAIN_R` frames to confirm a real release (vs a
  flicker), then `travel`.
- In `travel`, a new player's foot zone holding the ball opens `candidate_reception`,
  which settles after `RECEPTION_SETTLE_C` frames into a **completed** pass (same team) or
  **interception** (other team). No reception within `TRAVEL_TIMEOUT_FRAMES` ⇒ **ball_lost**.
- During travel the passer's team gets **provisional** possession credit. On interception
  the credit is flipped (`flip_to` the receiver's team); on ball-lost it is dropped
  (`drop` → OOF). These corrections are the `adjustments` returned each frame.

### The local `run()` (renderer)
- Builds the detectors/classifier/GK, fits team + GK (cached), then loops frames up to
  `PROCESS_DURATION_SEC` (default **60 s**), rendering every native frame.
- Rescales the FSM/Kalman thresholds from their 15 fps calibration to the actual fps
  (`fps_scale = eff_fps/15`, `_sc(...)`), so timing behaviour is fps-independent.
- Draws ellipses/labels (supervision), the ball triangle, a possession bar, and a pass
  overlay; writes `data/output/possession_output.mp4`; prints possession + pass summaries.
- Prints rich phase-transition and resolved-event logs (`NEW CARRIER`, `RELEASE
  CONFIRMED`, `PASS COMPLETED`, `INTERCEPTED`, `BALL LOST`, …).

### What `possession.py` does NOT do
- The **reusable classes** do not open files, hit the network, or render — they are pure
  per-frame logic (easy to drive from the service).
- It does **not** detect contested-possession or dribbles as events: two teams in the
  zone ⇒ `loose`; a passer reclaiming their own ball produces **no** event (by design —
  see `sql/schema.sql` header).
- `run()` does **not** write to SQL or send callbacks — that's the service path.
- Velocity from `BallTracker` includes camera motion and must **not** be used for
  kick/acceleration detection (the FSM is purely geometric, so this is fine).
- The config block at the top (`PLAYER_MODEL_WEIGHTS`, `VIDEO_PATH`, …) uses local
  Windows paths — `run()` is a dev harness, not the production entry point.

### Connections
- **Imports** `PlayerDetector`/`Detection`/`FrameDetections`, `GSFATeamClassifier`,
  `GoalkeeperDetector`, `PlayerTracker`.
- **Imported by** `service/session.py` (the building blocks), `service/clip_processor.py`
  (`best_ball`), and `service/stats.py` mirrors its constants.

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

### Connections
- Shares the homography concept with `heatmap.py` (same pitch dimensions). Output JSON is
  what `inspect_log*.py` poke at.

---

## `inspect_log.py` / `inspect_log2.py`
Small throwaway scripts to print/inspect the `keypoint_log.json` produced by
`simple_homography.py`. Not part of any pipeline.
