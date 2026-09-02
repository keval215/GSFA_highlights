# `modules/team_classifier/`

Assign each detected player a `team_id` of `0` or `1`. There are **two** independent
classifier implementations with the same interface; the SigLIP one is the production
default. (Moved here from top-level `team_classifier/` — same files, package
reorganised under `modules/`.)

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
- `__init__(device="cuda", batch_size=32, *, torso_ratio=0.55, blur_threshold=80.0,
  min_crop_px=32, centre_crop_ratio=0.50)` — **CUDA-only, fail-fast:** asserts
  `device.startswith("cuda") and torch.cuda.is_available()` at construction; there is
  **no CPU fallback** any more (previously defaulted to `device="cpu"` with no check,
  risking a silent slow CPU run or a confusing third-party stack trace instead of a clear
  error). The four crop/blur/quality knobs were previously module-level constants
  (`TORSO_RATIO`, `BLUR_THRESHOLD`, `MIN_CROP_PX`, `CENTRE_CROP_RATIO`); they are now
  constructor parameters defaulting to the same values, so a `RulesetConfig` can override
  them per sport (see [rulesets/README.md](../rulesets/README.md)).
- **Crop strategy:** uses the **torso** region only (`_torso_crop`, top `torso_ratio`
  of the bbox, default `0.55`) so it keys on jersey colour, not legs/court/background.
  During *fitting* it additionally drops blurry crops (`_is_sharp`, Laplacian variance
  vs. `blur_threshold`) and tiny crops (`< min_crop_px`).
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
  colour (HSV) against the caller's colours. Runs once at fit time; the `{cluster_id →
  name}` mapping is pickled. In the service this mapping is also what pins **which cluster
  is `team_a`**: `service/session.py` extracts the cluster id that resolved to
  `team_a_name` as `MatchSession.team_a_cluster_id`, and `service/stats.py::orient_for_team_a`
  reorders each minute's stored counters so `team_a`/`team_b` always follow jersey colour,
  not raw KMeans label order.
- **Speed details:** swaps in the torchvision "fast" SigLIP processor on every
  construction *and* warm-load (`_use_fast_processor`) — same normalization, faster CPU
  preprocessing. The pkl bakes in the slow processor, so this is re-applied after load.
- Save/load via joblib (`save`, `load`).

### What it does NOT do
- Does **not** assign a *meaningful* identity to cluster 0 vs 1 by itself — the mapping is
  arbitrary unless `resolve_team_names` is given names + colours. In the service it is
  fixed for the life of a *fit*, but a fit is no longer necessarily permanent: a mid-match
  team/GK colour change bumps `matches.fit_generation`, and the worker then discards the
  committed fit and re-fits from the next clip (see
  [service/README.md](../service/README.md) — `MatchSession.reset_for_new_generation`). A
  re-fit's KMeans can land the clusters in the opposite 0/1 order; `orient_for_team_a`
  (write-time, per minute) is what keeps stored `team_a`/`team_b` consistent across that
  swap, so the previous "never refit after committing" guard is now "never refit *the same
  generation's* fit".
- Does **not** classify referees or goal posts (left `None`).
- Does **not** handle goalkeepers specially — `GoalkeeperDetector` now classifies GKs by
  direct colour match, running **independently** of (not sequentially after) this class;
  see `modules/detectors/goalkeeper_detector.py`.
- Does **not** re-fit per frame; fitting is a one-time per-match step.
- Crops below `min_crop_px` (default 32) are skipped (no `team_id` set).

### Connections
- **Reads/writes** `Detection` objects from `PlayerDetector`.
- The `embedding` it sets is **consumed by** `modules/tracking/player_tracker.py`.
- `GoalkeeperDetector` reuses this class's **static** colour-vector helpers (`_hsv_vec`,
  `_colour_to_hsv`, `_torso_crop`, `_mean_colour_vec`) without depending on a
  `GSFATeamClassifier` instance.
- **Service wrapper:** `service/session.py` does *not* call `fit_from_video`; it reuses
  the static crop helpers (`_torso_crop`, `_is_sharp`) in its own `collect_crops` (now
  ruleset-parametrized — takes a `RulesetConfig` and passes its `torso_ratio`/
  `min_crop_px`/`blur_threshold` through), then calls `clf._classifier.fit(...)` directly
  with a silhouette-quality guard (`fit_and_score`). It uses `classify_batch` per clip and
  `resolve_team_names` for naming (storing the resolved `team_a` cluster id as
  `MatchSession.team_a_cluster_id`, persisted in a `fit_meta.json` sidecar alongside the
  pkl so it survives a disk reload). The `GSFATeamClassifier(device=config.DEVICE, ...)` it
  constructs also passes the match's ruleset's `torso_ratio`/`blur_threshold`/
  `min_crop_px`/`centre_crop_ratio`.
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
- Same `Detection`/`FrameDetections` contract. Historically used by the now-deleted
  root-level `heatmap.py`.

---

## `test_team_classifier.py`
Manual harness: fit on a sample video and render team-coloured boxes to eyeball cluster
quality. Not in the `tests/` automated suite.
