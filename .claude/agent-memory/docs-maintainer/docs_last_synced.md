---
name: docs_last_synced
description: Commit SHA + date of the last successful /update-doc sync — diff from here next time.
type: project
---

Last synced HEAD: `3844daa2c7596d7762c6b7f6a21eecafa3257adf` on 2026-08-26.

**What this sync covered:** diffed `471d643..3844daa` (i.e. everything after the
"Merge CV pipeline into modules/, add config-driven futsal/classic rulesets" commit,
which the prior 2026-08-24 sync already fully reconciled — see that sync's note below).
The only change in this range is commit `3844daa` ("updated docker for refactor code"):
`Dockerfile` swapped `COPY detectors/` / `COPY team_classifier/` / `COPY tracking/` for
`COPY modules/` / `COPY rulesets/`, i.e. the Docker build catching up to the refactor
that was already committed in `471d643`. Working tree was clean (`git status --short`
empty, no staged/unstaged diff) — no uncommitted work to fold in this time.

**No doc edits were needed.** `docs/codebase/infra/README.md`'s `## Dockerfile` section
already describes the build generically ("... then copies the source packages") and
never enumerated the old `detectors/`/`team_classifier/`/`tracking/` COPY paths, so
there was no stale text to fix. Grepped all of `docs/` for `COPY detectors`/`COPY
team_classifier`/`COPY tracking` — no matches. No Supermemory rows were touched this
sync (step 4 only fires on doc edits made in step 3; there were none).

**Prior sync note (2026-08-24, preserved for context):** the largest part of that sync
reconciled a then-uncommitted working-tree refactor — `detectors/`, `team_classifier/`,
`tracking/` git-mv'd into `modules/`; `video_analysis/possession.py` split into
`modules/possession/*.py` + `video_analysis/run.py`; new `rulesets/` package; `service/*`
and `sql/schema.sql` threading a per-match ruleset — which was subsequently committed for
real in `471d643`. That whole SHA range (`c7ef0b6..471d643`) is fully covered; do not
re-review it. See git history / the 2026-08-24 supermemory_index.md rows for the detailed
per-section breakdown of what was documented for that refactor.

**How to apply next time:** diff `3844daa..HEAD` plus `git status --short` / working+
staged diff for anything new. No known uncommitted refactor pending as of this sync.

**Addendum (2026-08-26, discrepancy-fix pass — not a new code-delta sync, HEAD
unchanged):** ran a targeted follow-up fixing two doc-only discrepancies that were
flagged (not fixed) by the sync above: `docs/azure_deploy.md` was rewritten in full to
match the real `service/` (api+worker, Azure Blob+Queue+SQL) architecture instead of a
stale pre-service-era draft, and `docs/codebase/sql/README.md`'s Migrations section now
documents the previously-undocumented `clip_duration_seconds`/`post_processing` gap. No
source code changed in this pass, so the `3844daa` HEAD marker above is still correct —
this addendum just records that those two flagged items are no longer open. See
`recurring_discrepancies.md` for the full before/after and `supermemory_index.md` for
the two new Supermemory rows saved. The other two items flagged in that sync (unused
`PlayerDetector.classes`, broken `simple_homography.py` import) are code bugs and remain
open/out of scope.
