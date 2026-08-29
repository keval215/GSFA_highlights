-- GSFA Highlights — Real-Time Clip Pipeline v1
-- Azure SQL (gsfa_stats) schema.
-- Contested-possession and dribble tracking are intentionally absent:
-- multiple teams in the foot zone count as "loose", and a passer
-- reclaiming the ball produces no event.

CREATE TABLE matches (
  match_id              NVARCHAR(64)  NOT NULL PRIMARY KEY,
  team_a_name           NVARCHAR(128) NULL,   -- team_a = CV cluster id 0
  team_b_name           NVARCHAR(128) NULL,   -- team_b = CV cluster id 1
  team_a_colour         NVARCHAR(32)  NULL,   -- jersey colour: hex (#FF6600) or CSS name (orange)
  team_b_colour         NVARCHAR(32)  NULL,
  team_a_gk_colour      NVARCHAR(32)  NULL,   -- goalkeeper jersey colour, same format
  team_b_gk_colour      NVARCHAR(32)  NULL,
  ruleset               NVARCHAR(16)  NOT NULL DEFAULT 'classic',  -- 'futsal' | 'classic'; fixed at match creation, never changes
  last_half_processed   TINYINT       NOT NULL DEFAULT 1,
  last_minute_processed INT           NOT NULL DEFAULT 0,
  next_clip_seq_h1      INT           NOT NULL DEFAULT 0,  -- atomic per-half upload counters
  next_clip_seq_h2      INT           NOT NULL DEFAULT 0,
  created_at            DATETIME2     NOT NULL DEFAULT SYSUTCDATETIME(),
  updated_at            DATETIME2     NOT NULL DEFAULT SYSUTCDATETIME()
);

-- Migration v2 (run once against existing DB before deploying updated service):
-- ALTER TABLE matches ADD next_clip_seq_h1 INT NOT NULL DEFAULT 0, next_clip_seq_h2 INT NOT NULL DEFAULT 0;

-- Migration v3 (run once against existing DB before deploying updated service):
-- ALTER TABLE matches ADD team0_gk_colour NVARCHAR(32) NULL, team1_gk_colour NVARCHAR(32) NULL;   -- (v6 later renamed these to team_a_/team_b_)
-- ALTER TABLE post_processing ADD team0_gk_colour NVARCHAR(32) NULL, team1_gk_colour NVARCHAR(32) NULL;

-- Migration v4 (run once against existing DB before deploying updated service):
-- Adds futsal/classic ruleset selection (rulesets/ package). DEFAULT 'futsal'
-- backfills existing rows so every match processed before this migration is
-- treated as futsal, matching its actual (pre-ruleset) processing.
-- ALTER TABLE matches ADD ruleset NVARCHAR(16) NOT NULL DEFAULT 'futsal';

-- Migration v5 (run once against existing DB before deploying updated service):
-- Flips the column DEFAULT to 'classic' (classic is now the default ruleset for
-- new matches). Existing rows keep whatever ruleset they were created with — the
-- DEFAULT only applies to future INSERTs that omit the column (which the service
-- never does; db.ensure_match always passes it explicitly). SQL Server needs the
-- named default constraint dropped and re-added:
--   DECLARE @c SYSNAME = (SELECT dc.name FROM sys.default_constraints dc
--                         JOIN sys.columns col ON col.object_id = dc.parent_object_id
--                                              AND col.column_id = dc.parent_column_id
--                         WHERE dc.parent_object_id = OBJECT_ID('matches')
--                           AND col.name = 'ruleset');
--   EXEC('ALTER TABLE matches DROP CONSTRAINT ' + @c);
--   ALTER TABLE matches ADD CONSTRAINT DF_matches_ruleset DEFAULT 'classic' FOR ruleset;

-- Migration v6 (run once against existing DB before deploying updated service):
-- Renames every team0/team1 (and _t0/_t1) column to team_a/team_b so the whole
-- stack uses one spelling (convention: CV cluster id 0 = team_a, 1 = team_b).
-- The outbound advance-stats callback body is UNCHANGED (still frames_a/frames_b/
-- passes_completed_a/…) — build_payload maps the new internal names onto the old
-- wire keys. Pure renames: no data moves, PK/indexes are not column-name-derived.
-- Full script + rollback: sql/migrations/v6_team_a_b_rename.sql
--   EXEC sp_rename 'matches.team0_name',              'team_a_name',             'COLUMN';
--   EXEC sp_rename 'matches.team1_name',              'team_b_name',             'COLUMN';
--   EXEC sp_rename 'matches.team0_colour',            'team_a_colour',           'COLUMN';
--   EXEC sp_rename 'matches.team1_colour',            'team_b_colour',           'COLUMN';
--   EXEC sp_rename 'matches.team0_gk_colour',         'team_a_gk_colour',        'COLUMN';
--   EXEC sp_rename 'matches.team1_gk_colour',         'team_b_gk_colour',        'COLUMN';
--   EXEC sp_rename 'minute_stats.frames_team0',       'frames_team_a',           'COLUMN';
--   EXEC sp_rename 'minute_stats.frames_team1',       'frames_team_b',           'COLUMN';
--   EXEC sp_rename 'minute_stats.passes_completed_t0','passes_completed_team_a', 'COLUMN';
--   EXEC sp_rename 'minute_stats.passes_completed_t1','passes_completed_team_b', 'COLUMN';
--   EXEC sp_rename 'minute_stats.interceptions_t0',   'interceptions_team_a',    'COLUMN';
--   EXEC sp_rename 'minute_stats.interceptions_t1',   'interceptions_team_b',    'COLUMN';
--   EXEC sp_rename 'minute_stats.ball_lost_t0',       'ball_lost_team_a',        'COLUMN';
--   EXEC sp_rename 'minute_stats.ball_lost_t1',       'ball_lost_team_b',        'COLUMN';
--   EXEC sp_rename 'post_processing.team0_name',      'team_a_name',             'COLUMN';
--   EXEC sp_rename 'post_processing.team1_name',      'team_b_name',             'COLUMN';
--   EXEC sp_rename 'post_processing.team0_colour',    'team_a_colour',           'COLUMN';
--   EXEC sp_rename 'post_processing.team1_colour',    'team_b_colour',           'COLUMN';
--   EXEC sp_rename 'post_processing.team0_gk_colour', 'team_a_gk_colour',        'COLUMN';
--   EXEC sp_rename 'post_processing.team1_gk_colour', 'team_b_gk_colour',        'COLUMN';
--   EXEC sp_rename 'post_processing.frames_team0',        'frames_team_a',           'COLUMN';
--   EXEC sp_rename 'post_processing.frames_team1',        'frames_team_b',           'COLUMN';
--   EXEC sp_rename 'post_processing.passes_completed_t0', 'passes_completed_team_a', 'COLUMN';
--   EXEC sp_rename 'post_processing.passes_completed_t1', 'passes_completed_team_b', 'COLUMN';
--   EXEC sp_rename 'post_processing.interceptions_t0',    'interceptions_team_a',    'COLUMN';
--   EXEC sp_rename 'post_processing.interceptions_t1',    'interceptions_team_b',    'COLUMN';
--   EXEC sp_rename 'post_processing.ball_lost_t0',        'ball_lost_team_a',        'COLUMN';
--   EXEC sp_rename 'post_processing.ball_lost_t1',        'ball_lost_team_b',        'COLUMN';

-- RAW counters only; one row per processed 60 s clip.
-- Cumulative numbers are always computed on read (SUM over rows),
-- never stored — see db.cumulative_read().
CREATE TABLE minute_stats (
  match_id             NVARCHAR(64) NOT NULL,
  half                 TINYINT      NOT NULL,
  minute               INT          NOT NULL,
  clip_duration_seconds DECIMAL(6,2) NULL,      -- seconds reported by the incoming clip payload
  frames_team_a           INT NOT NULL DEFAULT 0,   -- team_a = CV cluster id 0
  frames_team_b           INT NOT NULL DEFAULT 0,   -- team_b = CV cluster id 1
  frames_loose            INT NOT NULL DEFAULT 0,
  frames_oof              INT NOT NULL DEFAULT 0,
  passes_completed_team_a INT NOT NULL DEFAULT 0,
  passes_completed_team_b INT NOT NULL DEFAULT 0,
  interceptions_team_a    INT NOT NULL DEFAULT 0,   -- passes BY team_a that were intercepted
  interceptions_team_b    INT NOT NULL DEFAULT 0,
  ball_lost_team_a        INT NOT NULL DEFAULT 0,
  ball_lost_team_b        INT NOT NULL DEFAULT 0,
  revision             INT NOT NULL DEFAULT 0,   -- bumped by retroactive corrections
  clip_blob_path       NVARCHAR(400) NULL,
  processed_at         DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
  CONSTRAINT PK_minute_stats PRIMARY KEY (match_id, half, minute)   -- idempotency key
);

-- Whole-match aggregate written by POST /post-processing.
CREATE TABLE post_processing (
  match_id              NVARCHAR(64)  NOT NULL PRIMARY KEY,
  team_a_name           NVARCHAR(128) NULL,   -- team_a = CV cluster id 0
  team_b_name           NVARCHAR(128) NULL,   -- team_b = CV cluster id 1
  team_a_colour         NVARCHAR(32)  NULL,
  team_b_colour         NVARCHAR(32)  NULL,
  team_a_gk_colour      NVARCHAR(32)  NULL,
  team_b_gk_colour      NVARCHAR(32)  NULL,
  frames_team_a           INT NOT NULL DEFAULT 0,
  frames_team_b           INT NOT NULL DEFAULT 0,
  frames_loose            INT NOT NULL DEFAULT 0,
  frames_oof              INT NOT NULL DEFAULT 0,
  passes_completed_team_a INT NOT NULL DEFAULT 0,
  passes_completed_team_b INT NOT NULL DEFAULT 0,
  interceptions_team_a    INT NOT NULL DEFAULT 0,
  interceptions_team_b    INT NOT NULL DEFAULT 0,
  ball_lost_team_a        INT NOT NULL DEFAULT 0,
  ball_lost_team_b        INT NOT NULL DEFAULT 0,
  video_blob_path       NVARCHAR(400) NULL,
  processed_at          DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
);

-- Append-only event log; future highlight-reel source.
CREATE TABLE events (
  event_id      BIGINT IDENTITY PRIMARY KEY,
  match_id      NVARCHAR(64) NOT NULL,
  half          TINYINT      NOT NULL,
  minute        INT          NOT NULL,
  frame_idx     INT          NOT NULL,           -- processed-frame index within the match session
  kind          NVARCHAR(24) NOT NULL,           -- pass|interception|ball_lost
  from_team     TINYINT NULL,
  to_team       TINYINT NULL,
  from_track    INT NULL,
  to_track      INT NULL,
  travel_frames INT NULL,
  created_at    DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
);
CREATE INDEX IX_events_match ON events (match_id, half, minute);

-- Transactional outbox: the worker inserts a row in the same transaction
-- as the minute_stats write; the notifier sends pending rows strictly in
-- (half, minute) order per match.
CREATE TABLE callback_outbox (
  outbox_id       BIGINT IDENTITY PRIMARY KEY,
  match_id        NVARCHAR(64) NOT NULL,
  half            TINYINT      NOT NULL,
  minute          INT          NOT NULL,
  payload         NVARCHAR(MAX) NOT NULL,
  status          NVARCHAR(12) NOT NULL DEFAULT 'pending',  -- pending|sent|failed
  attempts        TINYINT      NOT NULL DEFAULT 0,
  last_attempt_at DATETIME2 NULL
);
CREATE INDEX IX_outbox_pending ON callback_outbox (status, match_id, half, minute);
