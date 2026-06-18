# `tracking/`

Assign each player a **stable `track_id`** across frames so the pass FSM can say "player
#7 passed to player #9".

| File | One-line role |
|---|---|
| `player_tracker.py` | `PlayerTracker` — BoT-SORT wrapper using external SigLIP embeddings |
| `__init__.py` | Package marker |

---

## `player_tracker.py` — `PlayerTracker`

### What it does
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
  3. Call `tracker.update(...)` and write the returned `track_id` back onto each
     `Detection` **in place** (matched by boxmot's `det_ind` column).
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
  `video_analysis/possession.py`.
- Instantiated in local mode by `possession.py::run` and in service mode held on the
  `MatchSession` (`service/session.py`), so track IDs persist across clips within a match.
