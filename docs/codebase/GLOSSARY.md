# Glossary

Domain and codebase terms, in the sense this project uses them.

| Term | Meaning |
|---|---|
| **Detection** | One detected object in one frame (`modules/detectors/player_detector.py::Detection`). Carries bbox + the mutable fields later stages fill in (`team_id`, `is_goalkeeper`, `track_id`, `embedding`, `smoothed_bbox`). |
| **FrameDetections** | All detections for a single frame, split into `players` / `referees` / `goal_posts` / `balls` / `all`. Which buckets fill depends on the ruleset's `class_names`: under `classic` the model has no `goal_post` class, so `goal_posts` is always empty. |
| **foot_point** | Bottom-centre of a bbox `((x1+x2)//2, y2)`. Used for ground-plane reasoning (homography, foot-zone carrier test). |
| **centre_point** | Geometric centre of a bbox. Used for the ball's tracked position. |
| **team_id** | `0` or `1` — which team a player belongs to. `None` = unclassified (referees, posts, low-quality crops). Set by the team classifier; the cluster→team mapping is arbitrary-but-fixed per match. |
| **embedding** | 768-D SigLIP feature vector for a player crop. Produced by the team classifier and reused by the tracker as appearance features (no separate ReID model). |
| **track_id** | Stable per-player identity across frames, assigned by BoT-SORT (`modules/tracking/player_tracker.py`). |
| **Team fit** | The one-time per-match step that learns the two team clusters (SigLIP→UMAP→KMeans). Cached to a pkl; reloaded on later runs/clips. |
| **GK colour match** | How `GoalkeeperDetector` identifies goalkeepers (replaced the old fit-based "GK zone" approach): the caller supplies two reference jersey colours; each frame, the single player whose crop colour is closest to a reference (within `max_gk_colour_dist`) is flagged `is_goalkeeper=True`. No fit step, no goal-post dependency, no tracking. |
| **RulesetConfig** | A dataclass (`rulesets/base.py`) bundling every sport-tunable CV parameter (weights, conf thresholds, crop/blur tuning, GK colour distance, tracker thresholds, Kalman params, foot-zone sizing, pass-FSM timing). One instance per sport — `FUTSAL` (production default) and `CLASSIC` (11-a-side, not yet production-ready). Selected per match via the `ruleset` field/flag and fixed for that match's lifetime. |
| **best_ball** | Adapter (`modules/possession/ball_tracker.py`) that picks the single highest-confidence ball from `FrameDetections.balls` and wraps it as a `BallDetection`. |
| **Coasting** | When the ball isn't detected, `BallTracker` emits the Kalman-predicted position for up to `KALMAN_COAST_FRAMES` frames before declaring it `LOST`. |
| **Carrier** | The player currently judged to "have" the ball — nearest player whose **foot zone** contains the ball centre. |
| **Foot zone** | A radius around a player's `foot_point` (`FOOT_ZONE_RATIO × bbox_height`, clamped). Ball inside it ⇒ that player is a carrier candidate. |
| **CarrierState** | Per-frame result of the carrier test: `kind ∈ {carrier, loose, oof}` plus the track/team if a carrier. |
| **loose** | Ball is in play but no single team's player owns it (e.g. two teams' players both in the zone, or none are). |
| **OOF** ("out of frame/play") | The catch-all state: ball lost/untracked, or no carrier and not loose. Excluded from the possession % denominator. |
| **Pass FSM** | The release→travel→reception state machine (`PassEventTracker`) that turns carrier transitions into pass / interception / ball-lost events. |
| **Release** | The passer's foot zone stops containing the ball, sustained for `RELEASE_SUSTAIN_R` frames. |
| **Travel** | Ball is between players after a confirmed release; waiting for a reception or a timeout. |
| **Reception** | A new player's foot zone holds the ball for `RECEPTION_SETTLE_C` frames — resolves the pass (same team = completed, other team = interception). |
| **Provisional credit** | During travel, possession frames are credited to the passer's team optimistically. Corrected later if the pass was actually an interception or a loss. |
| **Adjustment** | A `(kind, team_id, n)` correction emitted by the pass FSM. `flip_to` = move n frames to the other team; `drop` = reclassify n frames as OOF. |
| **PriorCorrection** | A service-mode adjustment whose frames started in the *previous* clip/minute — applied as an UPDATE to that prior `minute_stats` row (`revision += 1`). |
| **Possession denominator** | Only `team_a + team_b` frames. `loose` and `oof` are intentionally excluded, so possession % is "of the time the ball was clearly owned". |
| **team_a / team_b** | The two teams. Baseline convention is KMeans cluster 0 = `team_a`, cluster 1 = `team_b`, but in the service the stored assignment is **colour-anchored at write time** (`stats.orient_for_team_a`): `team_a` follows the cluster whose resolved jersey colour matches `team_a_colour`, so a mid-match re-fit that swaps the cluster order doesn't swap which team's stats land in which column. One spelling end-to-end (HTTP fields, SQL columns, internal labels) since v6. The advance-stats callback body keeps the older `_a`/`_b` suffixes on the wire. |
| **fit_generation** | `matches.fit_generation` (starts at 1). Bumped when a `POST /api/clips` request carries a team/GK colour that genuinely differs from the stored one, or unconditionally by `POST /api/matches/{id}/reset-fit`. Signals the worker to reset that match's team classifier + GK detector and re-fit from the next clip (mid-match jersey change). Persisted per session in a `fit_meta.json` sidecar next to the team-fit pkl. |
| **superseded** | `BIT` flag on `minute_stats` / `events`, set by `db.mark_minutes_superseded` on all rows up to the last processed minute when `fit_generation` bumps. `db.cumulative_read` excludes `superseded` rows, so old-colour and new-colour stats are never summed together. |
| **team_a_cluster_id** | `MatchSession` field: which KMeans cluster id (0/1) resolved to `team_a_name`/`team_a_colour` under the current fit (from `resolve_team_names`). Drives `orient_for_team_a`. `None` if colour resolution wasn't attempted or failed (→ fall back to raw cluster order). |
| **MatchSession** | Service-mode object holding all cross-clip state for one match (tracker, ball Kalman, carrier engine, pass FSM, team fit, `fit_generation`, `team_a_cluster_id`, carryover). |
| **minute row** | One `minute_stats` row = one processed 60 s clip = ~one match-minute. Stores **raw** counters only. |
| **Cumulative-on-read** | Totals are `SUM()`-ed across minute rows at read time, never stored. Percentages derived after summing. |
| **Transactional outbox** | The `callback_outbox` row is written in the same transaction as the stats, guaranteeing every persisted stat is eventually delivered. |
| **Ordering guard** | Worker logic that only processes the next expected `(half, minute)`; out-of-order clips are deferred then processed-with-a-gap. |
| **Poison queue** | Where messages that fail more than `MAX_DEQUEUE_COUNT` times are parked for manual inspection. |
| **CMC** | Camera-motion compensation in BoT-SORT (`ecc` by default) — keeps track IDs stable across fast camera pans. |
| **Homography (H)** | The 3×3 transform mapping image pixels ↔ top-down pitch metres. Used by `simple_homography.py` (the now-deleted `heatmap.py` also used it), **not** by the possession pipeline. |
