---
name: docs_last_synced
description: Commit SHA + date of the last successful /update-doc sync — diff from here next time.
type: project
---

Last synced HEAD: `c7ef0b6245310cfc783835ca081a7624219711fc` on 2026-08-24.

**Important caveat:** the largest part of what this sync reconciled was **NOT yet
committed** — a large working-tree refactor (git status showed it as uncommitted at sync
time): `detectors/`, `team_classifier/`, `tracking/` git-mv'd into `modules/`;
`video_analysis/possession.py` deleted and split into `modules/possession/*.py` +
`video_analysis/run.py`; a new `rulesets/` package (`RulesetConfig`, `futsal.py`,
`classic.py`, `registry.py`) parametrizing every CV constant; `service/session.py`,
`service/worker.py`, `service/api.py`, `service/db.py`, `service/config.py`, and
`sql/schema.sql` all updated to thread a per-match `ruleset` through. Also uncommitted:
deletion of root-level `heatmap.py`/`shots_on_t.py`, and three new
`scripts/colab_botsort_reid_tracking.py` / `colab_mcbyte_tracking.py` /
`mcbyte_tracking_test.py` research scripts.

**How to apply next time:** do NOT assume `git diff c7ef0b6..HEAD` alone captures
everything already reconciled — start with `git status --short` + `git diff`/`git diff
--staged` (uncommitted work) exactly as the standard workflow says, *in addition to*
diffing from this SHA. If the `modules:`/`rulesets:` refactor has been committed by the
time of the next sync, treat this whole SHA range as already covered by this sync
(2026-08-24) and don't re-review it — only diff what's genuinely new since. If it is
**still** uncommitted at the next sync, `git status --short` will show the same
`modules/`, `rulesets/`, `video_analysis/run.py` files again — don't be alarmed, that
just means it hasn't been committed yet; re-check for *further* changes on top of what's
already documented rather than re-documenting from scratch.

**Also reconciled this sync (already committed, landed between the prior sync SHA
`2ecc1ee` and this one via commits `c2c113f`/`42954c1`/`c7ef0b6`, but had not actually
been synced to docs yet despite `docs/*` diffs appearing in that range — see below):**
- `docker-compose.yml` memory limits (OOM-freeze fix, `c2c113f`).
- `service/worker.py` / `service/api.py` / `service/config.py` post-processing
  robustness fixes (`42954c1`): delete-before-process with no retry, error tracking via
  heartbeat, `FIT_SAMPLE_EVERY` 5→30.
- `GoalkeeperDetector` full rewrite from fit-based GK-zone-centroid detection to
  fit-free direct jersey-colour matching, plus CUDA-only `GSFATeamClassifier` (`c7ef0b6`).
- **Note on the `2ecc1ee..HEAD` doc diff seen at sync start:** a large chunk of doc
  content (GET /metrics restoration, smoothed_bbox, classes param, CI branch retarget,
  scripts reorg, clip_duration_seconds, post_processing dedup, blob/queueing kind field)
  had already been written to `docs/` by a *prior* sync's edits that sat uncommitted and
  were later swept into unrelated commits (`c2c113f`/`42954c1`) by the user's own
  `git add`. That content was verified accurate and left as-is; only the genuinely new
  GK-rewrite / OOM-fix / ruleset material required new edits this round.

**Why the marker is a commit SHA when most of the work was uncommitted:** the standing
convention (see `.claude/commands/update-doc.md`) is SHA + date; a working-tree diff has
no SHA to anchor to. The caveat above is the actual source of truth for "what's really
been reconciled" — trust it over the SHA alone.
