# `modules/tracking/`

Assign each player a **stable `track_id`** across frames so the pass FSM can say "player
#7 passed to player #9". (Moved here from top-level `tracking/` — same file, package
reorganised under `modules/`.)

| File | One-line role |
|---|---|
| `player_tracker.py` | `PlayerTracker` — BoT-SORT wrapper using external SigLIP embeddings |
| `__init__.py` | Package marker |

---

## `player_tracker.py` — `PlayerTracker`

### What it does
- `__init__(fps, cmc_method="ecc", *, track_high_thresh=0.5, track_low_thresh=0.1,
  new_track_thresh=0.6, match_thresh=0.8, proximity_thresh=0.5, appearance_thresh=0.25,
  track_buffer_frames_at_30fps=60)` — the six BoT-SORT association thresholds and the
  track-buffer length were previously hardcoded inside `__init__`; they are now
  constructor parameters (defaults unchanged) so a `RulesetConfig` can override them per
  sport — 11-a-side football has more simultaneous tracks/occlusion than 5-a-side futsal,
  a prime candidate to actually need different values (see
  [rulesets/README.md](../rulesets/README.md)).
- Wraps **BoT-SORT** from `boxmot` (`BotSort`) with:
  - Built-in **ECC global motion compensation** (`cmc_method="ecc"` by default) so IDs
    survive fast camera pans.
  - **External SigLIP appearance embeddings** — `with_reid=True` but `reid_model=None`;
    instead of running boxmot's own ReID network it feeds the 768-D SigLIP embeddings that
    `GSFATeamClassifier.classify` already computed.
  - Two-stage ByteTrack association (high-conf then low-conf thresholds).
- `update(frame, players)`:
  1. Build an `N×6` dets array (`x1,y1,x2,y2,conf,cls`).
  2. Build an `N×emb_dim` embeddings matrix from each `Detection.embedding`; detections
     without an embedding get an all-zeros row.
  3. Call `tracker.update(...)` and write the returned `track_id` **and**
     `smoothed_bbox` (BoT-SORT's Kalman-smoothed box, from the same output row) back onto
     each `Detection` **in place** (matched by boxmot's `det_ind` column). The raw
     `.bbox` is left untouched so foot-zone/carrier geometry doesn't shift; only drawing
     uses the smoothed box, to avoid per-frame jitter.
- `track_buffer` keeps lost tracks alive ~2 s (scaled by frame rate).

### What it does NOT do
- Does **not** detect or classify anything — it consumes `Detection`s that already have
  bboxes (and ideally embeddings).
- Does **not** run its own appearance model. Detections **without** a SigLIP embedding are
  still tracked, but only by motion (weaker recovery across pans).
- Does **not** track the ball (that's `BallTracker`'s Kalman filter) or goal posts/refs.

### Important robustness detail
If **no** detection in a frame carries an embedding, the code still passes an empty/zeros
embeddings matrix (never `None`). Passing `embs=None` would make boxmot reach for its
internal ReID model — which is `None` here — and crash with
`'NoneType' object has no attribute 'get_features'`. The all-zeros fallback degrades that
frame to motion-only association, which is the documented behaviour. (The resulting
cosine-distance-on-zero-vector `RuntimeWarning` is filtered in `service/worker.py`.)

### Connections
- **Reads** `Detection.bbox`, `.confidence`, `.embedding` (set by `GSFATeamClassifier`).
- **Writes** `Detection.track_id`, consumed by `CarrierEngine` and `PassEventTracker` in
  `modules/possession/`; also writes `Detection.smoothed_bbox`, consumed only by drawing
  code (not by carrier/pass geometry).
- Instantiated in local mode by `video_analysis/run.py::run` and in service mode held on
  the `MatchSession` (`service/session.py`), so track IDs persist across clips within a
  match. In both cases the six BoT-SORT thresholds + track-buffer length come from the
  match/run's `RulesetConfig`.
