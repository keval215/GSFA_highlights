# Implementation: rename `team0`/`team1` → `team_a`/`team_b` (one uniform convention, end-to-end)

> Status: **code complete, deploy pending** — captured 2026-08-28, implemented 2026-08-29.
>
> **Scope decision (user, 2026-08-29):** the **outbound** advance-stats callback body is
> left UNCHANGED (still `frames_a`/`frames_b`/`passes_completed_a`/…). Only the codebase's
> internal convention changed to `team_id 0 ↔ team_a`, `1 ↔ team_b`.
> `service/stats.py::build_payload` maps the new internal names onto the old wire keys, so
> the tournament-duelz consumer needs **no** change (the §6 fallback was taken).
>
> **Done in code:** `modules/possession/{labels,__init__,pass_event_tracker,possession_stats}.py`,
> `modules/detectors/goalkeeper_detector.py` (ctor param names), `modules/team_classifier`
> comment, `service/{stats,db,api,queueing,worker,session}.py`, `video_analysis/run.py`
> (`--team-a-gk-colour`/`--team-b-gk-colour`), `scripts/get_csv.py` (CSV headers) +
> `scripts/upload_clips.ps1` (`-TeamAName`… + `ruleset`), `sql/schema.sql` (v5+v6 notes)
> + new `sql/migrations/v6_team_a_b_rename.sql` (forward + rollback), `tests/test_db.py`,
> `tests/test_stats.py`, and the docs listed in §11. `pytest` green (19 passed, 1 skipped
> — the DB integration test needs `SQL_CONN_STR`); `import service.session` passes the
> lockstep asserts. Also folded in: `classic` is now `DEFAULT_RULESET` (separate ask).
>
> **Still to do (operational — user runs these):** ① announce/freeze uploads ② drain the
> Azure Queue (no old-format `team0_*` message in flight) ③ run the v5 + v6 SQL
> migrations against the live DB ④ deploy the v6 api + worker containers ⑤ switch the
> external clip uploader to the `team_a_*` form fields. See "Deploy sequencing" below.

## Context

Team identity is spelled **three** different ways across the stack, and one of
those is a hidden remap that has burned this project's kind of drift before:

| Layer | Today | Where |
|---|---|---|
| Inbound HTTP form fields + `matches` columns | `team0_*` / `team1_*` | `service/api.py`, `sql/schema.sql` |
| Frame counters | `frames_team0` / `frames_team1` | `sql/schema.sql`, `service/stats.py` |
| Pass / turnover counters | `passes_completed_t0/t1`, `interceptions_t0/t1`, `ball_lost_t0/t1` | same |
| **Outbound callback body** | `frames_a` / `frames_b`, `passes_completed_a/b`, … | `service/stats.py::build_payload` |
| CV pipeline runtime | integer `team_id` `0` / `1` / `None` | `modules/**` |

**The only "conversion" in the system** is one-directional and outbound-only:
`service/stats.py::build_payload()` (lines 215-226) re-keys the cumulative
`sums` dict — `frames_team0 → frames_a`, `frames_team1 → frames_b`,
`*_t0 → *_a`, `*_t1 → *_b`. Team id `0 → a`, id `1 → b`, positional. There is
**no reverse** (`a → 0`) anywhere, and **no `teamA`/`teamB` token exists** in
the repo at all. The inbound API does **no** translation — it receives fields
literally named `team0_name`, `team1_colour`, `team0_gk_colour`, …

### Decisions (confirmed with user)

- **Scope: everything** — DB columns, inbound HTTP fields, outbound callback
  keys, Azure queue JSON, all internal Python, CV label constants, docs, tests.
- **Target spelling: `team_a` / `team_b`** (snake_case, matches repo style).
- **Include the `_t0` / `_t1` counter columns** too.
- Consequence: `build_payload` becomes an **identity pass-through** (input keys
  == output keys) and can be simplified to `return dict(sums)` shape.

### NOT in scope (boundary — leave as-is)

- `events.from_team` / `events.to_team` (`TINYINT`, hold the integer `0`/`1`)
  and the runtime `team_id` int throughout `modules/`. KMeans produces cluster
  ids `0`/`1`; that stays the positional key. The convention becomes simply
  **`team_id 0 ↔ team_a`, `team_id 1 ↔ team_b`** (today it is `0 ↔ a`). No
  `TINYINT → CHAR` column change, no rewrite of `if team_id == 0` branch logic.
- Overlay label text `f"T{p.team_id}"` / `f"GK-T{p.team_id}"` in the offline
  renderer — cosmetic HUD strings, not a contract. Optional to touch.

## Target naming scheme

| Old | New |
|---|---|
| `team0_name` / `team1_name` | `team_a_name` / `team_b_name` |
| `team0_colour` / `team1_colour` | `team_a_colour` / `team_b_colour` |
| `team0_gk_colour` / `team1_gk_colour` | `team_a_gk_colour` / `team_b_gk_colour` |
| `frames_team0` / `frames_team1` | `frames_team_a` / `frames_team_b` |
| `passes_completed_t0` / `_t1` | `passes_completed_team_a` / `_team_b` |
| `interceptions_t0` / `_t1` | `interceptions_team_a` / `_team_b` |
| `ball_lost_t0` / `_t1` | `ball_lost_team_a` / `_team_b` |
| callback `frames_a` / `frames_b` / `passes_completed_a` … | `frames_team_a` / `frames_team_b` / `passes_completed_team_a` … |
| `POSSESS_TEAM0 = "team0"` / `LBL_TEAM0` | `POSSESS_TEAM_A = "team_a"` / `LBL_TEAM_A` (name **and** value) |
| `--team0-gk-colour` / `--team1-gk-colour` CLI | `--team-a-gk-colour` / `--team-b-gk-colour` |

## Change set — files, by area, with criticality

### 1. Azure SQL — 28 physical columns · CRITICAL · needs a live migration

No migration framework exists (only `sql/schema.sql` with hand-written
commented `ALTER`s). Write an explicit `sp_rename` script:

- `matches` (6): `team0_name, team1_name, team0_colour, team1_colour, team0_gk_colour, team1_gk_colour`
- `minute_stats` (8): `frames_team0, frames_team1, passes_completed_t0/t1, interceptions_t0/t1, ball_lost_t0/t1`
- `post_processing` (14): the six team-meta columns + `frames_team0/1` + the six `_t0/_t1` counters

```sql
-- sql/migrations/vN_team_a_b_rename.sql  (new file)
EXEC sp_rename 'matches.team0_name',              'team_a_name',             'COLUMN';
EXEC sp_rename 'matches.team1_name',              'team_b_name',             'COLUMN';
-- … one line per column, 28 total …
EXEC sp_rename 'minute_stats.frames_team0',       'frames_team_a',           'COLUMN';
EXEC sp_rename 'minute_stats.passes_completed_t0','passes_completed_team_a', 'COLUMN';
-- … etc …
```

Indexes/PK unaffected (`PK_minute_stats`, `IX_events_match`, `IX_outbox_pending`
are not column-name-derived). Also rewrite `sql/schema.sql` to the new names and
append a `-- vN` history line. Rollback = the inverse `sp_rename` script (write
it in the same commit).

### 2. `service/db.py` — CRITICAL (SQL strings + `_SUM_COLS`)

- `_SUM_COLS` tuple (L192-197) — **find-replace blind spot**: the SQL in
  `cumulative_read` (L204) and the `sums` dict keys (L211) are generated from
  this tuple. Edit the tuple; the f-string lines follow automatically.
- Every SQL literal in `ensure_match`, `get_team_specs`, `get_gk_colours`,
  `write_clip_result` (INSERT + UPDATE `minute_stats`), `write_post_processing_result`
  (INSERT + UPDATE), `_apply_prior_correction` (SELECT + UPDATE).
- Param names on `ensure_match` / `write_post_processing_result`.
- Return dict keys of `write_post_processing_result` (L371-390) — this is the
  `POST /post-processing` HTTP response body.
- Local `t0, t1, oof` in `_apply_prior_correction` (L405-428) — pure locals,
  rename for readability only (optional), branch logic unchanged.

### 3. `service/stats.py` — CRITICAL (label constants + dataclasses + the remap)

- `LBL_TEAM0/1` → `LBL_TEAM_A/B` (value `"team_a"`/`"team_b"`).
- `MinuteRow`, `MinuteCounters` fields → `frames_team_a`, `passes_completed_team_a`, …
- `add_label`, `apply_adjustment`, `count_event`, `to_minute_row` bodies.
- `build_payload` (L215-226): after the rename its input and output keys are
  identical → collapse to a pass-through (keep the function + `team_id_to_name`
  arg for call-site compatibility; drop the literal remap dict).

### 4. `modules/possession/` — MED (lockstep with stats.py)

- `labels.py` `POSSESS_TEAM0/1` (name + value) and `__init__.py` re-exports.
- `pass_event_tracker.py` L293/299-301 `return POSSESS_TEAM0 if team == 0 …`.
- `possession_stats.py` L25-57 (dict keys, denominator, `src_label`/`other_label`
  branch selectors) — **blind spot**: value chosen by `team_id == 0`, not a
  literal.
- `service/session.py:51-52` asserts `stats.LBL_TEAM_A == vp_labels.POSSESS_TEAM_A`
  — must move in the same commit or the **worker won't import**.

### 5. `service/api.py` — CRITICAL (public HTTP contract)

- Module docstring, `Form(...)` field names on both endpoints
  (`team0_name … team1_gk_colour`), call-site kwargs into `db.ensure_match` and
  `_queue.enqueue_post_processing`.
- Reconcile with the **uncommitted WIP** already in `service/api.py` (shows
  modified in `git status`).
- Optional grace period: accept both old and new field names for one release
  (pre-parse the multipart form, or dual `Form` aliases). User chose a hard
  break — note as fallback only.

### 6. `service/queueing.py` + `service/worker.py` — CRITICAL (queue wire format)

- `ClipMessage` fields, `enqueue*` params, all `body["team0_*"]` dict keys in
  `enqueue`, `enqueue_post_processing`, `dequeue` (`body.get`), `defer`,
  `move_to_poison`.
- `worker.py` `msg.team0_* → msg.team_a_*` passthrough (L180-209).
- **Deploy hazard**: `dequeue` uses `body.get(...)`, so any message enqueued by
  the old API still sitting in the queue during rollout silently yields `None`
  for the team fields → GK colours (and post-processing team meta) lost for
  those jobs. **The queue must be fully drained before the new build goes in.**

### 7. `service/session.py` — LOW

- L51-52 asserts (see §4), GK-colour locals L323-332, doc/log strings L285-301
  ("falls back to team0/team1 labels").

### 8. `video_analysis/run.py` — MED (offline CLI)

- `--team0-gk-colour` / `--team1-gk-colour` argparse flags + `run()` signature
  + plumbing (L190-233, L401-424). This is a user-facing CLI flag rename.
- `f"T{p.team_id}"` overlay strings — optional cosmetic.

### 9. `scripts/` — MED (downstream consumers)

- `scripts/get_csv.py`: `CSV_FIELDS` + `STAT_COLS` lists (**blind spot**, list-
  driven), the two raw `SELECT`s (L212, L226-229), `make_row`/`build_5min_blocks`
  params. CSV **output headers change** — warn anyone consuming that export.
- `scripts/upload_clips.ps1`: `$Team0Name`/etc. params + the `curl -F
  team0_name=…` fields (L71-74). Must ship **with** the API change.

### 10. Tests — MED

- `tests/test_db.py`: kwargs, raw SQL, `sums["frames_team0"]` asserts (L79-80),
  and the callback asserts `payload["frames_a"] == 170` / `passes_completed_a`
  (L82-83, L102) → now `payload["frames_team_a"]`.
- `tests/test_stats.py`: `LBL_TEAM0/1` import, `MinuteCounters.frames_team0`
  asserts throughout.
- `tests/test_notifier.py`: no change (stubbed payloads).
- Ad-hoc vis scripts (`modules/**/test_*.py`): cosmetic prints only — optional.

### 11. Docs — CRITICAL per CLAUDE.md ("update docs alongside the code change")

- `docs/API.md`: both request-field tables + the callback body section
  (L242-258 — `frames_a` → `frames_team_a`, and the "positional 0→a/1→b" note
  becomes "0→team_a / 1→team_b"). Reconcile with the uncommitted WIP in this
  file.
- `docs/codebase/sql/README.md`, `service/README.md`, `ARCHITECTURE.md`,
  `GLOSSARY.md`, `scripts/README.md`, `video_analysis/README.md`,
  `detectors/README.md` — all carry `team0/1_*` column/flag references.
- `CLAUDE.md` L3 ("per team (team0/team1)").
- Run `/update-doc` after, and let it reconcile Supermemory (`sm_project_gsfa`).

## Deploy sequencing (this is where it breaks if rushed)

1. **Announce/freeze uploads** for the maintenance window; stop the external
   uploader.
2. **Drain the Azure queue** — let the worker process every pending clip and
   `post_processing` message; confirm `callback_outbox` has no `pending` rows
   (all `sent`). Now no old-format message or payload is in flight.
3. **Apply the SQL `sp_rename` migration** (28 columns). Verify with a
   `SELECT TOP 1 *` on each of the 3 tables.
4. **Deploy new `api` + `worker`** containers (new field names, new column
   names, new queue keys, identity `build_payload`).
5. **Deploy the updated uploader** (`upload_clips.ps1` / the real client) with
   `team_a_*` form fields.
6. **Coordinate the `advance-stats` consumer** (tournament-duelz) to accept
   `frames_team_a` / `passes_completed_team_a` / … Ideally it accepts old+new
   for a transition window; otherwise strict cutover in this same window.
   *(Fallback if the third party can't move: keep `build_payload` mapping
   `frames_team_a → frames_a` on the wire and leave the callback contract
   alone — shrinks the blast radius to our side only.)*
7. **Resume uploads.** Watch `GET /metrics` (`queue_depth`,
   `last_post_processing_error`) and the worker log for the first few clips.

Rollback: inverse `sp_rename` script + redeploy previous container tags.

## Find-replace blind spots — must be hand-edited (token never appears literally)

- `service/db.py:204,211` — SQL + dict keys generated from `_SUM_COLS`.
- `scripts/get_csv.py` — `STAT_COLS` / `CSV_FIELDS` list-driven sum + headers.
- `service/stats.py:100-129`, `service/db.py:408-420`,
  `modules/possession/possession_stats.py:37-39`,
  `modules/possession/pass_event_tracker.py:293` — team chosen by
  `if team_id == 0` branch, not a literal.
- `build_payload` — two-sided: the left side (`sums["frames_team0"]`) and the
  wire side (`frames_a`) are different roots; both must change.
- `f"T{p.team_id}"` / `f"GK-T{p.team_id}"` — `run.py`, `team_classifier.py`,
  `goalkeeper_detector.py`, `colour_histogram.py` (cosmetic; skip unless asked).

## Verification

1. `pytest tests/` — `test_db.py`, `test_stats.py`, `test_notifier.py` green
   after their assertions are updated.
2. `python -c "import service.session"` — the import-time asserts (§4) pass,
   proving `stats` and `modules.possession.labels` are in lockstep.
3. `grep -rniE "team0|team1|_t0\b|_t1\b|frames_a|passes_completed_a" service/ modules/ video_analysis/ scripts/ sql/ docs/`
   returns only the intentionally-kept integer `team_id`/`from_team` references.
4. On a scratch DB: run `sql/schema.sql`, then the migration on a copy holding
   old-named columns, confirm both converge to identical column sets.
5. End-to-end dry run (needs explicit user go-ahead to run the worker):
   `upload_clips.ps1` (new fields) → one clip → `minute_stats` row written with
   `frames_team_a` populated → `callback_outbox` payload has `frames_team_a`
   keys → notifier POST body inspected in the log.
6. `python video_analysis/run.py --team-a-gk-colour … --team-b-gk-colour …`
   on a short clip — CLI flags parse, run completes.

## Open items to confirm before executing

- Who owns the `advance-stats` (tournament-duelz) consumer, and can it take the
  new `frames_team_a` keys (or old+new during transition)? If not → use the
  §6 fallback.
- There are **uncommitted changes** to `service/api.py` and several `docs/`
  files on `main` right now — rebase this rename on top of a clean tree, or
  fold that WIP in deliberately.
- Confirm `events.from_team` / runtime `team_id` staying integer `0`/`1` is
  acceptable (plan assumes yes).
- Note: this repo already references a design doc named `implementation.md`
  (see `service/db.py`, `service/worker.py` docstrings) that does not currently
  exist as a file. If that doc resurfaces, move this content to a distinct
  filename to avoid clobbering it.
