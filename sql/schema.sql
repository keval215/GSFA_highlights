-- GSFA Highlights — Real-Time Clip Pipeline v1
-- Azure SQL (gsfa_stats) schema.
-- Contested-possession and dribble tracking are intentionally absent:
-- multiple teams in the foot zone count as "loose", and a passer
-- reclaiming the ball produces no event.

CREATE TABLE matches (
  match_id              NVARCHAR(64)  NOT NULL PRIMARY KEY,
  team0_name            NVARCHAR(128) NULL,
  team1_name            NVARCHAR(128) NULL,
  team0_colour          NVARCHAR(32)  NULL,   -- jersey colour: hex (#FF6600) or CSS name (orange)
  team1_colour          NVARCHAR(32)  NULL,
  team0_gk_colour       NVARCHAR(32)  NULL,   -- goalkeeper jersey colour, same format
  team1_gk_colour       NVARCHAR(32)  NULL,
  ruleset               NVARCHAR(16)  NOT NULL DEFAULT 'futsal',  -- 'futsal' | 'classic'; fixed at match creation, never changes
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
-- ALTER TABLE matches ADD team0_gk_colour NVARCHAR(32) NULL, team1_gk_colour NVARCHAR(32) NULL;
-- ALTER TABLE post_processing ADD team0_gk_colour NVARCHAR(32) NULL, team1_gk_colour NVARCHAR(32) NULL;

-- Migration v4 (run once against existing DB before deploying updated service):
-- Adds futsal/classic ruleset selection (rulesets/ package). DEFAULT 'futsal'
-- backfills existing rows so every match processed before this migration is
-- treated as futsal, matching its actual (pre-ruleset) processing.
-- ALTER TABLE matches ADD ruleset NVARCHAR(16) NOT NULL DEFAULT 'futsal';

-- RAW counters only; one row per processed 60 s clip.
-- Cumulative numbers are always computed on read (SUM over rows),
-- never stored — see db.cumulative_read().
CREATE TABLE minute_stats (
  match_id             NVARCHAR(64) NOT NULL,
  half                 TINYINT      NOT NULL,
  minute               INT          NOT NULL,
  clip_duration_seconds DECIMAL(6,2) NULL,      -- seconds reported by the incoming clip payload
  frames_team0         INT NOT NULL DEFAULT 0,
  frames_team1         INT NOT NULL DEFAULT 0,
  frames_loose         INT NOT NULL DEFAULT 0,
  frames_oof           INT NOT NULL DEFAULT 0,
  passes_completed_t0  INT NOT NULL DEFAULT 0,
  passes_completed_t1  INT NOT NULL DEFAULT 0,
  interceptions_t0     INT NOT NULL DEFAULT 0,   -- passes BY t0 that were intercepted
  interceptions_t1     INT NOT NULL DEFAULT 0,
  ball_lost_t0         INT NOT NULL DEFAULT 0,
  ball_lost_t1         INT NOT NULL DEFAULT 0,
  revision             INT NOT NULL DEFAULT 0,   -- bumped by retroactive corrections
  clip_blob_path       NVARCHAR(400) NULL,
  processed_at         DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
  CONSTRAINT PK_minute_stats PRIMARY KEY (match_id, half, minute)   -- idempotency key
);

-- Whole-match aggregate written by POST /post-processing.
CREATE TABLE post_processing (
  match_id              NVARCHAR(64)  NOT NULL PRIMARY KEY,
  team0_name            NVARCHAR(128) NULL,
  team1_name            NVARCHAR(128) NULL,
  team0_colour          NVARCHAR(32)  NULL,
  team1_colour          NVARCHAR(32)  NULL,
  team0_gk_colour       NVARCHAR(32)  NULL,
  team1_gk_colour       NVARCHAR(32)  NULL,
  frames_team0          INT NOT NULL DEFAULT 0,
  frames_team1          INT NOT NULL DEFAULT 0,
  frames_loose          INT NOT NULL DEFAULT 0,
  frames_oof            INT NOT NULL DEFAULT 0,
  passes_completed_t0   INT NOT NULL DEFAULT 0,
  passes_completed_t1   INT NOT NULL DEFAULT 0,
  interceptions_t0      INT NOT NULL DEFAULT 0,
  interceptions_t1      INT NOT NULL DEFAULT 0,
  ball_lost_t0          INT NOT NULL DEFAULT 0,
  ball_lost_t1          INT NOT NULL DEFAULT 0,
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
