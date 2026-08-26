---
name: recurring_discrepancies
description: Known code/doc discrepancies found during syncs that were deliberately not silently fixed — surface these again if still true.
type: project
---

Open items as of the 2026-08-24 sync (commit `c7ef0b6` + uncommitted `modules/`/`rulesets/`
refactor):

- **`PlayerDetector.classes` allow-list is still unused in production.** (Reconfirmed
  this sync.) `modules/detectors/player_detector.py`'s `classes: list[int] | None`
  constructor param — the docstring/comment says "Local possession passes `[0, 1, 2]` to
  hard-filter referees" — but `video_analysis/run.py`'s `run()` does **not** pass
  `classes=` to `PlayerDetector(...)` (verified via grep). The only caller that actually
  uses `classes=[0,1,2]` is the manual debug harness
  `modules/team_classifier/test_team_classifier.py`. Docs describe the param neutrally —
  don't add a claim that the local/service path filters referees until `run.py` actually
  passes it.
- **`sql/schema.sql` still has no inline migration note for two older additions.** The
  v2/v3/v4 migrations (`next_clip_seq_h1/h2`; `team0/1_gk_colour`; `matches.ruleset`) all
  have inline `ALTER TABLE` comments, but the older `minute_stats.clip_duration_seconds`
  column and the whole `post_processing` table still do not — anyone deploying against a
  pre-existing DB from before those were added has to hand-write those ALTER/CREATE
  statements. Flagged, not fixed (editing `schema.sql` is out of docs-maintainer scope).
- **`video_analysis/simple_homography.py` has a broken import as of the `modules/`
  refactor (new this sync, uncommitted).** `build_h_list` still does `from detectors.cache
  import cache_path as _cache_path` — the pre-refactor path. `detectors/` no longer
  exists at the repo root (git-mv'd to `modules/detectors/`), so this import will raise
  `ModuleNotFoundError` if `build_h_list` is ever actually called. The refactor's own
  scope notes call `simple_homography.py` deliberately out of scope (still uses the older
  detector stack conceptually), but the specific import breakage looks like an
  oversight rather than an intentional decision. Documented (not silently patched) in
  `docs/codebase/video_analysis/README.md`'s `simple_homography.py` section — this is a
  **code** bug, out of docs-maintainer's remit to fix.
- **`docs/azure_deploy.md` is significantly stale relative to the current `service/`
  architecture** (predates the tracked delta for this sync — not something that changed
  in the diffs reviewed, so left untouched rather than rewritten). It describes a
  different app shape entirely: a single `server.py` + `webapp/blob.py` FastAPI app with
  direct browser video uploads, a `_job_sem`/`_sweep_stale_work_dirs` job model, and one
  `highlights` Docker container — none of which matches the actual current
  `service/api.py` (ingest) + `service/worker.py` (GPU) two-process/two-container
  architecture, the clip-based upload flow, or Azure Blob/Queue/SQL backing described
  everywhere else in `docs/`. This looks like an early-draft runbook that was never
  updated after the real service architecture was built. Worth a deliberate rewrite pass
  (out of scope for a surgical `/update-doc` sync — flagging for the user rather than
  silently rewriting a whole document's architecture description).
- **Previous manual doc edits (not via `/update-doc`) broke `docs/API.md`.** (Historical,
  from the 2026-07-06 sync — kept for the lesson.) Commit `a04fea8` ("feat - addes post
  processing logic") hand-edited `docs/API.md` to add `POST /post-processing` but in
  doing so deleted the entire `GET /metrics` section and the `## Callback (outbox)`
  header/intro — leaving orphaned body-shape text under a `GET /health` section.
  Restored in the 2026-07-06 sync. Lesson: don't assume commits that touch doc files by
  the same author who touches code are safe — always diff the actual doc content, not
  just trust that "docs were already updated." (Confirmed this sync that the restored
  content survived intact through the intervening commits.)
