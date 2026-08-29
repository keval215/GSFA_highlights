# `sql/`

The Azure SQL (`gsfa_stats`) schema the service writes to. One file: `schema.sql`.

| Table | Purpose |
|---|---|
| `matches` | One row per match: team names/colours, processing progress, per-half upload counters |
| `minute_stats` | One row per processed 60 s clip — **raw** counters only |
| `post_processing` | One row per whole-match upload — raw counters plus team metadata snapshot |
| `events` | Append-only log of pass / interception / ball_lost events |
| `callback_outbox` | Transactional outbox: cumulative payloads the notifier delivers |

---

## Design principles baked into the schema

These explain *why* the tables look the way they do (see also `service/db.py`):

- **Raw, not cumulative.** `minute_stats` stores only this-minute counters. Cumulative
  totals are computed on read (`SUM()` in `db.cumulative_read`); percentages are derived
  **after** summing, never by averaging per-minute percentages.
- **Idempotency key.** `PK_minute_stats (match_id, half, minute)` makes a replayed clip
  overwrite its own row instead of double-counting.
- **Retroactive corrections.** A pass that spans a clip boundary and resolves as
  interception/ball-lost UPDATEs the prior minute row and bumps `revision`. This happens
  in the **same transaction** as the current minute insert, so the cumulative read already
  includes it.
- **Transactional outbox.** The `callback_outbox` row is inserted in that same
  transaction, so a stat is never delivered without being persisted, and a persisted stat
  is always eventually delivered (or marked `failed`).
- **Atomic minute claim.** `next_clip_seq_h1` / `next_clip_seq_h2` are bumped with
  `UPDATE...OUTPUT` so concurrent uploads get distinct minute numbers.
- **Whole-match upsert.** `post_processing` is keyed by `match_id` and stores the raw
  whole-match counters and the team metadata snapshot used for that upload.

## Key columns

- `matches.last_half_processed` / `last_minute_processed` — drive the worker's ordering
  guard (`is_expected`).
- `matches.team_a/b_name` + `team_a/b_colour` — used for cluster→team-name resolution
  (`db.get_team_specs` returns them only if **all four** are present). Convention: CV
  cluster id 0 → `team_a`, cluster id 1 → `team_b` (renamed from `team0`/`team1` in v6).
- `matches.team_a/b_gk_colour` — goalkeeper reference jersey colours, same hex/CSS-name
  format as `team_a/b_colour`. Columns are `NULL`-able (rows from before these fields
  existed), but the API now **requires** both on every request, so any match created after
  that change has both set and `MatchSession.ensure_gk_ready()` builds `GoalkeeperDetector`
  from clip 1. `db.get_gk_colours` still guards on **both** present (for the old rows).
  No fit step — see [detectors/README.md](../detectors/README.md). Mirrored on
  `post_processing` too.
- `matches.ruleset` — `'classic'` (default, since v5/v6) or `'futsal'`, selecting the `RulesetConfig`
  (see [rulesets/README.md](../rulesets/README.md)) this match's pipeline is tuned with.
  Set only on `INSERT` (first upload for a `match_id`) and never updated afterward — a
  match's ruleset is fixed for its lifetime. `db.get_match_ruleset` reads it back; the
  worker resolves it before creating or reusing a `MatchSession`.
- `minute_stats.frames_team_a/team_b/loose/oof` — possession denominator is `team_a+team_b`.
  The pass/turnover counters are `passes_completed_team_a/team_b`, `interceptions_team_a/team_b`,
  `ball_lost_team_a/team_b` (all renamed from `_t0`/`_t1` in v6).
- `minute_stats.clip_duration_seconds` — the client-supplied clip length from
  `POST /api/clips`, stored verbatim (no derivation). Nullable — rows written before this
  column existed are `NULL`. Used only by the CSV export (`scripts/get_csv.py`) for
  duration/pass-density math, not by the pipeline.
- `minute_stats.revision` — incremented by retroactive corrections.
- `post_processing.match_id` — the whole-match primary key; there is no half/minute or
  revision column because the table represents one processed match file.
- `events.kind` — `pass | interception | ball_lost` (mapped from FSM kinds via
  `stats.KIND_MAP`). `from_track`/`to_track` are the BoT-SORT track IDs.
- `callback_outbox.status` — `pending | sent | failed`.

## What it does NOT model
- No contested-possession / dribble events (deliberate — see the file header).
- No player-level stats tables; `events` carries track IDs but there's no players table.
- No stored cumulative or percentage columns anywhere (computed on read by design).

## Migrations
`schema.sql` includes inline `ALTER TABLE` notes for the migrations against an
existing DB, each to be run once before deploying the corresponding service version:
- **v2** — adds `next_clip_seq_h1`/`next_clip_seq_h2` to `matches`.
- **v3** — adds `team0_gk_colour`/`team1_gk_colour` to both `matches` and
  `post_processing` (later renamed by v6).
- **v4** — adds `matches.ruleset NVARCHAR(16) NOT NULL DEFAULT 'futsal'`. The default
  backfills existing rows so every match processed before this migration is treated as
  futsal, matching its actual (pre-ruleset) processing.
  `post_processing` does **not** get a `ruleset` column (it is not currently threaded
  through `write_post_processing_result`).
- **v5** — flips the `matches.ruleset` column DEFAULT to `'classic'` (classic is now
  the default ruleset for new matches). Drop + re-add the named default constraint;
  existing rows are untouched.
- **v6** — renames every `team0`/`team1` (and `_t0`/`_t1`) column to `team_a`/`team_b`
  across `matches`, `minute_stats`, and `post_processing` (28 `sp_rename` calls; full
  script + rollback in `sql/migrations/v6_team_a_b_rename.sql`). Pure renames, no data
  moves; PK/indexes unaffected. Run with the Azure Queue drained, before deploying the
  v6 api + worker. The outbound advance-stats callback body is **not** changed.

Two older additions predate this inline-migration-note convention and have **no**
`ALTER`/`CREATE` snippet in `schema.sql`, unlike v2–v4 above:
- **`minute_stats.clip_duration_seconds`** (`DECIMAL(6,2) NULL`) — a DB created from an
  older copy of `schema.sql` needs `ALTER TABLE minute_stats ADD clip_duration_seconds
  DECIMAL(6,2) NULL;` run by hand before deploying a service version that writes it
  (rows written before the column existed read back as `NULL`, which `db.py` and the CSV
  export already handle — see "Key columns" above).
- **The `post_processing` table itself** — added whole, not as an `ALTER`. A DB created
  before `POST /post-processing` existed needs the full `CREATE TABLE post_processing
  (...)` statement from `schema.sql` run by hand; there's no incremental diff to apply.

Anyone provisioning a **fresh** database from the current `schema.sql` gets both for free
(they're in the base `CREATE TABLE` statements) — this gap only matters when migrating an
existing pre-existing DB that predates these additions.
