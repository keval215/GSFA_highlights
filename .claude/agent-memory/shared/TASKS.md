# TASKS — mid-match team/GK colour-change reset (Approach A) — 2026-09-01

Decisions locked:
- D1: jerseys genuinely changed on video → re-fit is the correct fix.
- D2: automatic detection at ingest (colour diff → `fit_generation` bump) + manual
  `POST /api/matches/{id}/reset-fit` override endpoint.
- D3: exclude pre-reset minutes from cumulative going forward via a `superseded` flag.

Constraints passed to every agent: do NOT run the worker / possession.py / any clip
processing. Do NOT `git push`, commit, or deploy. Edit local files only.

- [x] 1. Implement Approach A end-to-end — owner: captain — deps: none
      result: done. 8 files changed (service/api,clip_processor,db,session,stats,worker.py;
      sql/schema.sql; tests/test_stats.py) + 3 new test files + conftest.py, 591+/45- lines.
      75 passed/1 skipped (pre-existing SQL_CONN_STR skip) verified independently by
      ironman; py_compile clean; diff scoped to service/+sql/+tests/ only; /code-review
      run, 2 real findings fixed (correction-orientation independence via
      last_written_team_a_cluster_id snapshot; reset-fit blank-field validation), 1 finding
      (docs) correctly deferred to task 2. Nothing run/committed/pushed.
      Parts:
        A. schema + colour persistence: `matches.fit_generation INT NOT NULL DEFAULT 1`;
           `minute_stats.superseded BIT NOT NULL DEFAULT 0` (+ `events.superseded` for
           consistency); migration blocks in `sql/schema.sql` matching the existing
           "Migration v#" convention. `db.ensure_match` UPDATE branch: normalise
           (lowercase / strip '#' / strip ws) incoming vs stored colours; on any real
           change overwrite the 4 colour columns + `fit_generation = fit_generation + 1`
           (bump once per request); names stay COALESCE; `ruleset` unchanged. New
           `db.get_fit_generation`, `db.mark_minutes_superseded(match_id, upto_half,
           upto_minute)` (minute_stats + events, only rows <= given half/minute, returns
           rowcount), `db.bump_fit_generation(match_id, optional colours/names)`. Add
           `AND superseded = 0` to `db.cumulative_read`.
        B. session reset: `MatchSession` tracks `fit_generation`, persisted in a
           `fit_meta.json` sidecar in the match-state dir, written whenever a fit is
           committed (alongside the joblib.dump). `MatchSession.reset_for_new_generation(n)`
           deletes `team_siglip.pkl` + `fit_meta.json`, clears `team_clf`/`fit_status`
           →"pending"/`_fit_crops_clip1`, clears `gk_det`/`_gk_colour_invalid`, sets
           `fit_generation = n`; leaves tracker / ball_tracker / carrier_eng / pass_track /
           proc_idx / counters untouched. `MatchSessionManager.get_or_create(match_id,
           ruleset, want_generation=None)`: after obtaining the session (new / cached /
           disk-reloaded), if `want_generation` > `session.fit_generation` call
           `reset_for_new_generation` and surface that a reset happened (return flag or
           queryable state). Covers the disk-reload path (stale sidecar + existing pkl).
        C. colour-anchored team orientation: at the point each minute's counters are
           written to `minute_stats`/`events`, order them so the `team_a` columns always
           hold the cluster whose resolved jersey colour matches `team_a_colour`, using the
           current fit's `team_clf.team_id_to_name` (already computed by
           `_resolve_team_names`/`resolve_team_names`, currently dropped). Must be applied
           at write time (per-minute), NOT at `cumulative_read`/`build_payload` time, so
           sums stay consistent across a mid-match re-fit where the KMeans label order
           flips. If colour resolution fails on a re-fit (generation > 1), log ERROR
           prominently and proceed (documented v1 limitation).
        D. `POST /api/matches/{match_id}/reset-fit` — optional body: 4 colours + 2 names;
           writes any provided values, `fit_generation += 1` unconditionally; 404 if the
           match row is absent; returns `{match_id, fit_generation}`.
        E. `service/worker.py::_handle`: fetch `db.get_fit_generation` and pass as
           `want_generation` into `get_or_create` (before the `ensure_fit` gate). If a
           reset fired, call `mark_minutes_superseded(match_id, last_half_processed,
           last_minute_processed)` and log a WARNING with the counts. Do NOT touch the
           `last_half_processed`/`last_minute_processed` ordering pointer. Confirm the
           `ensure_match` change does not disturb the `post_processing` path
           (`worker.py` ~:180-183); post path builds its own session directly — leave as is
           but note behaviour.
      accept:
        - `python -m py_compile` clean on every changed .py
        - existing `service/` pytest suite green + new unit tests green covering:
          ensure_match generation bump (only on real colour change; names still COALESCE);
          get_fit_generation; mark_minutes_superseded scope + cumulative_read exclusion;
          get_or_create reset on stale generation (team_clf/gk_det cleared, fit_status not
          "ok", fit_generation updated, tracker/pass_track object identity unchanged);
          sidecar written/removed on fit/reset (tmp MATCH_STATE_DIR); colour-anchored
          write orientation (cluster1→team_a mapping ⇒ team_a counters come from cluster 1);
          reset-fit endpoint bumps generation + 404 on unknown match
        - `/code-review` on captain's own diff, clean
        - `git diff --stat` scoped to `service/`, `sql/`, tests only — no docs/, no other dirs
        - captain reports exact commands run + output; no green-when-red
        - NOT run: worker.py, possession.py, any clip processing; NOT committed / pushed

- [x] 2. Reconcile docs + refresh shared context — owner: thor — deps: 1
      result: done. Edited docs/API.md, docs/codebase/{service,team_classifier,sql}/README.md,
      docs/codebase/ARCHITECTURE.md, docs/codebase/GLOSSARY.md. CONTEXT.md seed stub filled
      (sections 1-6, HEAD 5cf10e3, Approach A listed as "uncommitted, pending user
      review/commit"). Supermemory: 2 new rows added + verified. thor/docs_last_synced.md
      marker 3844daa→5cf10e3 with a 2026-09-01 addendum noting Approach A documented
      ahead-of-commit. Noted (not a bug): events.superseded is written but not yet read
      (cumulative_read filters minute_stats.superseded only) — "for consistency / future use".
      accept:
        - `docs/API.md`: colours now mutable mid-match; `fit_generation` semantics;
          `POST /api/matches/{id}/reset-fit` documented
        - `docs/codebase/service/README.md`: generation-triggered per-match session reset
          path; `superseded` flag + cumulative_read change
        - `docs/codebase/team_classifier/README.md`: `team_id_to_name` now drives stored
          team_a/team_b orientation
        - `.claude/agent-memory/shared/CONTEXT.md` refreshed against HEAD (sections 1–6)
        - Supermemory reconciled per the /update-doc convention
        - thor reports which files changed + the sync SHA
