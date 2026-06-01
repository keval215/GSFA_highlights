# Player Tracking — Status

## Architecture (3 layers, planned)

1. **DeepSORT** — frame-to-frame association. IoU first, SigLIP cosine second. Stable track IDs through normal play and short occlusions.
2. **Per-track SigLIP gallery** — rolling last K=10–20 embeddings per `track_id`, averaged into a representative vector. Updated while alive; frozen for ~3–5 s after a track dies.
3. **Re-identification module** — when DeepSORT yields an unmatched detection, compare its SigLIP embedding to every frozen gallery within spatial plausibility. If `cosine > threshold` and distance is reasonable, revive the old `track_id`.

## Key insight — no extra SigLIP cost

`GSFATeamClassifier.classify()` already runs SigLIP per frame for team labelling, then throws the 768-D embedding away after UMAP. We capture that embedding and feed it to the tracker. **Zero added GPU inference.**

## Phases

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 1** | DeepSORT tracker + capture SigLIP embeddings onto `Detection` | **Done (wired, untested live)** |
| Phase 2 | Per-track gallery + frozen-track bank + cosine re-ID with spatial gate | Not started |
| Phase 3 (optional) | Dual crop (torso + full body) for stronger re-ID; per-player stats | Not started |

## Phase 1 — what changed

- `detectors/player_detector.py` — `Detection` gains `track_id` and `embedding` fields.
- `team_classifier/team_classifier.py` — `classify()` split into `extract_features → reducer.transform → cluster_model.predict`; writes 768-D embedding onto each `Detection`.
- `tracking/player_tracker.py` (new) — `PlayerTracker` wraps `deep-sort-realtime` with `embedder=None`, consumes external SigLIP embeds, writes `track_id` back onto detections via IoU matching.
- `video_analysis/possession.py` — tracker initialised once, updated each frame after GK classification. Track IDs are appended to the on-screen label as `T0#7`.
- Dependency: `deep-sort-realtime 1.3.2` (installed in `highlights/` venv).

Call order in the main loop:
```
detect → team_clf.classify (embed) → gk_det.classify → tracker.update → possession → draw
```

## Where we are right now

Phase 1 is complete and imports cleanly. Not yet run on real video.

## What you run

```powershell
D:/GSFA_highlights/highlights/Scripts/python.exe video_analysis/possession.py
```

This processes `C:\Users\Admin\Downloads\Video Project 8.mp4` and writes `data/output/possession_output.mp4`. To test on something shorter, edit `VIDEO_PATH` at `video_analysis/possession.py:39`.

## What to check in the output

1. **Pipeline completes** without errors.
2. **Track IDs visible** — ellipse labels read `T0#7`, `GK-T1#3`, etc.
3. **IDs stick to players** through 1–2 s occlusions during normal play.
4. **ID count is sane** — over 30 s of continuous play, unique `track_id`s should stay under ~14 (10 outfield + 2 GKs + slack). Hundreds = config is too loose.

If IDs explode or flicker badly, tell me and we'll tune `max_cosine_distance` / `max_age` in `tracking/player_tracker.py`. If stable, we move to Phase 2.
