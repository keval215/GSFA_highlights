# `team_classifier/`

Assign each detected player a `team_id` of `0` or `1`. There are **two** independent
classifier implementations with the same interface; the SigLIP one is the production
default.

> Project rule: **always use `GSFATeamClassifier` (SigLIP), never the colour-histogram
> variant** as the default. The colour-histogram class is kept for comparison/fallback.

| File | One-line role |
|---|---|
| `team_classifier.py` | **`GSFATeamClassifier`** — SigLIP → UMAP → KMeans (production) |
| `colour_histogram.py` | `ColourHistogramTeamClassifier` — HSV histogram → KMeans (alt) |
| `test_team_classifier.py` | Manual/visual test harness |
| `__init__.py` | Package marker |

Both classifiers share the same lifecycle:
`fit_from_video_or_load()` once per match (fit + cache pkl, or reload), then
`classify(frame, detections)` per frame which **mutates `detections.players[i].team_id`
in place**. Referees and goal posts are left `None`.

---

## `team_classifier.py` — `GSFATeamClassifier` (production)

### What it does
- Wraps `sports.common.team.TeamClassifier` (SigLIP embeddings → UMAP(→3D) → KMeans(k=2)).
- **Crop strategy:** uses the **torso** region only (`_torso_crop`, top `TORSO_RATIO=0.55`
  of the bbox) so it keys on jersey colour, not legs/court/background. During *fitting* it
  additionally drops blurry crops (`_is_sharp`, Laplacian variance) and tiny crops.
- **Fit** (`fit_from_video` / `fit_from_video_or_load`): sample ~1 fps, collect torso
  crops, fit SigLIP+UMAP+KMeans, pickle to disk (`cache_path(..., "team_siglip")`).
  Warm start reloads the pkl instantly.
- **`classify(frame, detections)`:** crop all players, embed with SigLIP (fp16 forward via
  autocast — `_embed`), project through the fitted UMAP reducer, predict KMeans cluster →
  write `team_id`. It also stores the 768-D SigLIP `embedding` on each Detection (reused
  by the tracker — no separate ReID model needed).
- **`classify_batch(frames, detections_list)`:** same result as per-frame `classify`, but
  embeds crops from many frames in one SigLIP pass (the service's batched path).
- **Team-name resolution** (`resolve_team_names`): optionally maps the two clusters to
  caller-supplied team names by comparing each cluster's saturation-weighted mean jersey
  colour (HSV) against the caller's colours. Runs once at fit time; result pickled.
- **Speed details:** swaps in the torchvision "fast" SigLIP processor on every
  construction *and* warm-load (`_use_fast_processor`) — same normalization, faster CPU
  preprocessing. The pkl bakes in the slow processor, so this is re-applied after load.
- Save/load via joblib (`save`, `load`).

### What it does NOT do
- Does **not** assign a *meaningful* identity to cluster 0 vs 1 by itself — the mapping is
  arbitrary-but-fixed per match unless `resolve_team_names` is given names + colours.
- Does **not** classify referees or goal posts (left `None`).
- Does **not** handle goalkeepers specially — the `GoalkeeperDetector` overrides GK teams
  afterward.
- Does **not** re-fit per frame; fitting is a one-time per-match step.
- Crops below `MIN_CROP_PX` (32) are skipped (no `team_id` set).

### Connections
- **Reads/writes** `Detection` objects from `PlayerDetector`.
- The `embedding` it sets is **consumed by** `tracking/player_tracker.py`.
- **Service wrapper:** `service/session.py` does *not* call `fit_from_video`; it reuses
  the static crop helpers (`_torso_crop`, `_is_sharp`) in its own `collect_crops`, then
  calls `clf._classifier.fit(...)` directly with a silhouette-quality guard
  (`fit_and_score`). It uses `classify_batch` per clip and `resolve_team_names` for naming.
- `SIGLIP_MODEL_PATH` / `TeamClassifier` / `create_batches` come from the external
  `roboflow/sports` package.

---

## `colour_histogram.py` — `ColourHistogramTeamClassifier` (alternative)

### What it does
- Torso crop → HSV histogram (`H:64 bins + S:32 bins` → 96-D) → KMeans(k=2) directly on
  the raw histograms. PCA(2D) is computed for a debug scatter plot only.
- Same `fit_from_video_or_load` / `classify` interface and pkl caching
  (`cache_path(..., "team_colour")`).

### What it does NOT do
- Does **not** use SigLIP, UMAP, or any neural embedding — purely colour statistics.
- Does **not** produce a 768-D `embedding`, so a tracker relying on appearance features
  would fall back to motion-only association if this were used.
- Per project rule, **not** the default — kept for comparison/fallback.

### Connections
- Same `Detection`/`FrameDetections` contract. Historically used by `heatmap.py`.

---

## `test_team_classifier.py`
Manual harness: fit on a sample video and render team-coloured boxes to eyeball cluster
quality. Not in the `tests/` automated suite.
