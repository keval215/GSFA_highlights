---
name: docs_last_synced
description: Commit SHA + date of the last successful /update-doc sync — diff from here next time.
type: project
---

Last synced HEAD: `5cf10e3d6b9e09e0a0f3e1cc503947a77fd48b36` on 2026-09-01
(committed portion) — PLUS an uncommitted working-tree diff documented ahead of commit;
see the 2026-09-01 addendum at the bottom. Previous marker was
`3844daa2c7596d7762c6b7f6a21eecafa3257adf` (2026-08-26).

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

**Addendum (2026-09-01, thor Mode-3 partial sync — "Approach A" mid-match colour change):**
Triggered by ironman/captain (shared TASKS.md task 2), not a full `/update-doc` sweep.

- **Committed range `3844daa..5cf10e3`** (commits `3697434` team0/1→team_a/b rename +
  classic-as-default-ruleset, `5cf10e3` per-ruleset detector class map / classic 3-class):
  spot-checked `docs/` — already fully reconciled (the 2026-08-24 uncommitted-refactor
  sync, later committed, plus CLAUDE.md, already carry `team_a`/`team_b` naming, classic
  default, and the futsal-4-class / classic-3-class split). **No doc edits needed** for
  these two commits. Marker advanced to `5cf10e3` to reflect that.
- **Uncommitted working-tree diff** (captain's task-1 "Approach A" implementation across
  `service/api.py`, `clip_processor.py`, `db.py`, `session.py`, `stats.py`, `worker.py`,
  `sql/schema.sql` + `tests/`): **documented ahead of commit** this pass. Nothing is
  committed — do not treat `5cf10e3` as covering it. Edits made:
  - `docs/API.md` — new "Mid-match colour changes (`fit_generation`)" subsection; colour
    fields now mutable/overwrite + generation bump; `superseded` exclusion noted in the
    callback contract; new `POST /api/matches/{match_id}/reset-fit` endpoint section
    (body, 200 shape, 404/422); "Team mapping is positional" bullet rewritten as
    colour-anchored via `orient_for_team_a`; `build_payload` `team_id_to_name` removal.
  - `docs/codebase/service/README.md` — per-clip flow steps 4/5/7 (generation check,
    `get_or_create` tuple return, `mark_minutes_superseded`, `write_clip_result` arg drop);
    `api.py` reset-fit bullet; `worker.py` generation-reset bullet + post_processing
    isolation note; `session.py` section largely rewritten (`fit_generation`,
    `team_a_cluster_id`, `fit_meta.json`, `_resolve_team_names`, `reset_for_new_generation`,
    `finish_clip` snapshot, `get_or_create(want_generation=)`); `clip_processor.py`
    `orient_for_team_a` call; `stats.py` new `orient_for_team_a` + `build_payload` sig;
    `db.py` new helpers + `ensure_match` colour-overwrite + `cumulative_read` filter;
    added a v1-limitation bullet to "does NOT do".
  - `docs/codebase/team_classifier/README.md` — `resolve_team_names` now also pins
    `team_a_cluster_id`; "arbitrary-but-fixed per match" nuanced (refits happen, gated by
    `fit_generation`); service-wrapper bullet mentions the sidecar.
  - `docs/codebase/sql/README.md` — key-column entries for `matches.fit_generation` and
    `minute_stats.superseded`/`events.superseded`; colour-anchored note on the
    `team_a/b_name`+`colour` entry; new "v7" migration bullet.
  - `docs/codebase/ARCHITECTURE.md` — worker flow step 4 generation check; new §7 bullet
    "Mid-match colour changes (`fit_generation`)".
  - `docs/codebase/GLOSSARY.md` — new rows `fit_generation`, `superseded`,
    `team_a_cluster_id`; `team_a / team_b` and `MatchSession` rows updated.
- **Supermemory:** 2 new rows added to `sm_project_gsfa` (ids `tL2ynvWQQyPEo1z9s4V23A`,
  `2J5K3Bz5TMDk6AEbg96ybj`), recorded in `supermemory_index.md`. Additive — no `forget`
  attempted (no exact stale duplicate; older overlapping rows stay valid).
- **CONTEXT.md** refreshed (Mode 2) — sections 1–6 populated from the stub, "Recent
  changes" flags the Approach A diff as "uncommitted, pending user review/commit".

**How to apply next time:** once Approach A is committed, diff `5cf10e3..HEAD`; the
service/sql changes above will appear as newly-committed — they are **already documented**,
so verify the docs still match rather than re-documenting. Then re-verify the 2 Supermemory
rows and advance this marker + their "Last-synced SHA" cells to the real commit.

**Addendum (2026-09-03, thor Mode-3 scoped reconciliation — `CLIP_PIPELINE_THREADED`).**
Triggered by ironman/captain, NOT a full `/update-doc` sweep — marker deliberately left at
`5cf10e3`. Captain landed (uncommitted working tree) an opt-in threaded producer/consumer
split in `service/clip_processor.py::process_clip()` behind a new `CLIP_PIPELINE_THREADED`
bool in `service/config.py` (`# --- Tunables ---`, default False). Pass 1/Pass 2 extracted
to `_pass1`/`_pass2` closures; default serial path byte-for-byte unchanged, no thread
created. New `tests/test_clip_processor_threaded.py`. No API contract / SQL / threshold /
`team_a`-`team_b` change.
- Docs edited this pass: `docs/API.md` (one env-var row added to the inventory table —
  captain expected no API change, but that table is a full `service/config.py` mirror so
  omitting the knob would be drift; no contract/endpoint change), `docs/codebase/service/
  README.md` (config tunables line, file-role table cell, new `clip_processor.py` bullet
  on the threaded split, timing-log `total` caveat, worker "does NOT parallelise" note),
  `docs/codebase/ARCHITECTURE.md` (worker flow step 5 + connection-map line).
- `docs/azure_deploy.md` and `NEEDED_FROM_YOU.md` — **do not exist** (azure_deploy.md
  deleted in `6c128c2`, NEEDED_FROM_YOU.md untracked since `2303936`); the only env-var
  inventory in `docs/` is `docs/API.md`'s table, now updated.
- Supermemory: **not touched** this pass (single opt-in toggle, minor; change still
  uncommitted). Fold into the next full sweep.
- CONTEXT.md refreshed (Mode 2): header working-tree note, §4 new uncommitted bullet, §5
  decode-loop rewrite + tunables list.

**A proper `/update-doc` (`5cf10e3..HEAD`) is still warranted** — to (a) verify the
already-documented Approach A diff still matches now that it's committed as `6c128c2` and
advance the 2 Supermemory rows' SHA cells, (b) reconcile `cf9d23e` (per-clip timing log
DEBUG→INFO — service README already says "INFO", worth a spot-check), and (c) commit-anchor
this threaded split once it lands and add its Supermemory row.
