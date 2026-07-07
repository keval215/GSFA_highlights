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
- `matches.team0/1_name` + `team0/1_colour` — used for cluster→team-name resolution
  (`db.get_team_specs` returns them only if **all four** are present).
- `minute_stats.frames_team0/1/loose/oof` — possession denominator is `team0+team1`.
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
`schema.sql` includes an inline note for the v2 migration that added the
`next_clip_seq_h1/h2` columns to an existing DB. Run it once before deploying the updated
service against a pre-existing database.
