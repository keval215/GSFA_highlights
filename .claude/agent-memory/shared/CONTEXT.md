# Shared CONTEXT — GSFA_highlights

> Refreshed: 2026-09-01 · HEAD: `5cf10e3` (+ uncommitted Approach-A diff across `service/` + `sql/` + `tests/`)
> Owner: thor. Every agent reads this file first. Keep it to ~one screen.

## 1. What this repo is
Futsal/football match-video analysis. From video it derives, per team (`team_a`/`team_b`):
possession %, completed passes, interceptions, ball-lost events → append-only `events` +
`minute_stats` tables. Does **not** yet cut highlight reels — those tables are raw material.

Two execution modes, one shared CV pipeline:
- **Local script:** `python video_analysis/run.py [--ruleset futsal|classic]` — one whole
  video file, renders an annotated `.mp4` + printed summary.
- **Production service:** `service/api.py` (ingest, no GPU) + `service/worker.py` (GPU) —
  60 s clips over HTTP, one per match-minute (+ a whole-match `/post-processing` path).
  Output is Azure SQL rows + an HTTP advance-stats callback. **No video render.**

## 2. Key modules / perception chain (per frame)
`PlayerDetector` (`modules/detectors/player_detector.py`, unified YOLOv11m) →
`GSFATeamClassifier` (`modules/team_classifier/team_classifier.py`, SigLIP→UMAP→KMeans, CUDA-only) →
`GoalkeeperDetector` (`modules/detectors/goalkeeper_detector.py`, fit-free colour match, runs
independently of the team classifier) →
`PlayerTracker` (`modules/tracking/player_tracker.py`, BoT-SORT, consumes SigLIP embeddings) →
`BallTracker` (`modules/possession/ball_tracker.py`, Kalman + coast) →
`CarrierEngine` (`modules/possession/carrier_engine.py`, foot-zone) →
`PassEventTracker` (`modules/possession/pass_event_tracker.py`, release/travel/receive FSM).
Labels/constants: `modules/possession/labels.py` (mirrored torch-free in `service/stats.py`;
`service/session.py` asserts the two match at import). Tuning: one `RulesetConfig`
(`rulesets/base.py`), selected per match/run.

## 3. Load-bearing invariants
- Team classifier is always `GSFATeamClassifier` (SigLIP). Never the colour-histogram variant.
- Unified YOLOv11m; class set is per-ruleset via `RulesetConfig.class_names`
  (futsal 4-class incl. `goal_post`, classic separate 3-class checkpoint, no `goal_post`).
- Docker uses deadsnakes Python 3.11 (Ubuntu default 3.11.0rc1 segfaults `torch.jit.script`).
- Naming is `team_a` / `team_b` end-to-end (since v6); classic is the default ruleset.
- A match's `ruleset` is fixed at creation. `matches` names are fill-in-only (COALESCE).
- **NEW (uncommitted):** the four team/GK **colour** columns are now mutable mid-match —
  every request overwrites them and a genuine change bumps `matches.fit_generation`,
  which drives a per-match team/GK re-fit. `team_a`/`team_b` column assignment is
  colour-anchored at write time (`stats.orient_for_team_a`), not raw KMeans label order.
- Never run the pipeline / worker / clip processing without explicit user permission.
- Never `git push` or deploy to the production VM.

## 4. Recent changes
- **uncommitted, pending user review/commit** — "Approach A": automatic mid-match team/GK
  colour-change handling. Diff across `service/api.py`, `clip_processor.py`, `db.py`,
  `session.py`, `stats.py`, `worker.py`, `sql/schema.sql`, + new/extended tests
  (`tests/conftest.py`, `test_api_reset_fit.py`, `test_db_generation.py`,
  `test_session_generation.py`, `test_stats.py`). Key points:
  - `matches.fit_generation INT NOT NULL DEFAULT 1`; `minute_stats.superseded` +
    `events.superseded` `BIT NOT NULL DEFAULT 0` (schema "Migration v7").
  - `db.ensure_match`: colour columns become `COALESCE(?, existing)` (overwrite); bumps
    `fit_generation` when a normalised incoming colour differs from stored. Names still
    COALESCE. New `db.get_fit_generation`, `mark_minutes_superseded` (minute_stats+events
    ≤ given half/minute), `bump_fit_generation`. `cumulative_read` gains `AND superseded = 0`.
  - `MatchSession`: `fit_generation` + `team_a_cluster_id` (backed by `fit_meta.json`
    sidecar next to `team_siglip.pkl`), `reset_for_new_generation()`,
    `last_written_team_a_cluster_id` snapshot in `finish_clip()`.
    `MatchSessionManager.get_or_create(..., want_generation=)` → `(session, just_reset)`;
    resets that match's team classifier + GK detector when DB generation is ahead,
    preserving tracker / ball / carrier / pass-FSM state. No worker restart.
  - `worker._handle`: reads `get_fit_generation`, passes as `want_generation`; on reset,
    calls `mark_minutes_superseded(match_id, last_half, last_minute)`. `post_processing`
    path builds its own session directly — unaffected.
  - `service/stats.py::orient_for_team_a(result, team_a_cluster_id,
    correction_team_a_cluster_id=…)` — reorders a clip's minute_row/events/correction so
    `team_a`/`0` == the cluster matching `team_a_colour`. Applied in `clip_processor` at
    write time, never at read time. `build_payload` lost its `team_id_to_name` param;
    `db.write_clip_result` lost its `team_names` param.
  - New endpoint `POST /api/matches/{match_id}/reset-fit` (optional JSON body of the 6
    name/colour fields): unconditionally bumps `fit_generation`; 404 unknown match;
    422 on a field present-but-blank. Returns `{match_id, fit_generation}`.
- `5cf10e3` — detector per-ruleset class map (classic 3-class, no `goal_post`).
- `3697434` — rename `team0`/`team1` → `team_a`/`team_b`; classic is now the default ruleset.
- `3844daa` — Dockerfile updated for the `modules/` + `rulesets/` refactor.
- `471d643` — CV pipeline merged into `modules/`, config-driven futsal/classic rulesets.

## 5. Open research findings
- None. `.claude/agent-memory/shared/research/` holds only `README.md` (no briefs yet).

## 6. Pointers
- Big picture: `docs/codebase/ARCHITECTURE.md` (mid-match colour-change flow is in §3 + §7)
- Terms: `docs/codebase/GLOSSARY.md` (`fit_generation`, `superseded`, `team_a_cluster_id`)
- Service detail: `docs/codebase/service/README.md`; HTTP contract: `docs/API.md`
- Schema: `docs/codebase/sql/README.md` (migration v7)
- Per-package detail: `docs/codebase/<package>/README.md`
- Last docs sync SHA/date: `.claude/agent-memory/thor/docs_last_synced.md`
