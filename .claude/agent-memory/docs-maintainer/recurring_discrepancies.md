---
name: recurring_discrepancies
description: Known code/doc discrepancies found during syncs that were deliberately not silently fixed — surface these again if still true.
type: project
---

**2026-08-26 discrepancy-fix pass (targeted, not a full `/update-doc` delta sync):**
fixed the two doc-only items below — `docs/azure_deploy.md` was rewritten in full to
match the actual `service/` (api+worker) architecture, and `docs/codebase/sql/README.md`'s
Migrations section now documents the `clip_duration_seconds`/`post_processing` gap. Both
Supermemory rows updated (see `supermemory_index.md`). The other two items (unused
`classes` allow-list, broken `simple_homography.py` import) were re-confirmed still
accurate and deliberately left untouched — they're code bugs, out of docs-maintainer
scope. Also confirmed during this pass: no nginx/TLS/certbot/basic-auth config exists
anywhere in the repo — the "port 8000 bound to 127.0.0.1 behind nginx" VM-topology
detail floated as something to cross-check could **not** be verified against source
(`docker-compose.yml`'s `"8000:8000"` binds all interfaces) or against any accessible
project memory (no `deployment_setup.md`-type file exists). Treat that detail as
unconfirmed live-VM state, not something this repo's config establishes.

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
- **`sql/schema.sql` still has no inline migration note for two older additions —
  FIXED in docs on 2026-08-26.** The v2/v3/v4 migrations (`next_clip_seq_h1/h2`;
  `team0/1_gk_colour`; `matches.ruleset`) all have inline `ALTER TABLE` comments in
  `schema.sql` itself; the older `minute_stats.clip_duration_seconds` column and the
  whole `post_processing` table still don't (and `schema.sql` was correctly left
  unedited — that's a code file, out of docs-maintainer scope). What changed:
  `docs/codebase/sql/README.md`'s Migrations section now documents this gap explicitly
  — the exact `ALTER TABLE` a pre-existing-DB migration needs for
  `clip_duration_seconds`, and the fact that `post_processing` needs its full
  `CREATE TABLE` copied by hand. A fresh DB from current `schema.sql` is unaffected.
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
- **`docs/azure_deploy.md` was significantly stale relative to the current `service/`
  architecture — FIXED (full rewrite) on 2026-08-26.** It described a different app
  shape entirely: a single `server.py` + `webapp/blob.py` FastAPI app with direct
  browser video uploads, a `_job_sem`/`_sweep_stale_work_dirs` job model, and one
  `highlights` Docker container. Rewritten to describe the actual current
  `service/api.py` (ingest, CPU) + `service/worker.py` (GPU) two-container architecture:
  the real `/mnt/data` subpath layout (`JOBS_DIR`/`MATCH_STATE_DIR`/`HF_HOME`), the
  actual required Azure resources (Blob **and** Queue **and** SQL — the old doc only
  covered Blob), the real CI/CD flow (`.github/workflows/deploy.yml`, auto-deploy on
  push to `main`), and the real env vars from `service/config.py`/`docs/API.md`. VM
  identity facts at the top (name, region, IP, GPU SKU) were left as-is — unverifiable
  against repo source, and not what was flagged as stale.
- **Previous manual doc edits (not via `/update-doc`) broke `docs/API.md`.** (Historical,
  from the 2026-07-06 sync — kept for the lesson.) Commit `a04fea8` ("feat - addes post
  processing logic") hand-edited `docs/API.md` to add `POST /post-processing` but in
  doing so deleted the entire `GET /metrics` section and the `## Callback (outbox)`
  header/intro — leaving orphaned body-shape text under a `GET /health` section.
  Restored in the 2026-07-06 sync. Lesson: don't assume commits that touch doc files by
  the same author who touches code are safe — always diff the actual doc content, not
  just trust that "docs were already updated." (Confirmed this sync that the restored
  content survived intact through the intervening commits.)
