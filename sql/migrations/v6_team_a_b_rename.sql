-- Migration v6 — rename team0/team1 (and _t0/_t1) columns to team_a/team_b.
--
-- Run ONCE against the existing gsfa_stats DB, in the maintenance window,
-- AFTER the Azure Queue is fully drained and BEFORE deploying the updated
-- api + worker containers.
--
-- Convention going forward: CV cluster id 0 = team_a, cluster id 1 = team_b.
-- These are pure metadata renames — no rows are rewritten, and the PK /
-- indexes (PK_minute_stats, PK on post_processing, IX_events_match,
-- IX_outbox_pending) are not column-name-derived, so they are unaffected.
--
-- The OUTBOUND advance-stats callback body is deliberately NOT changed by this
-- migration or the code deploy — it still sends frames_a / frames_b /
-- passes_completed_a / … ; service/stats.py::build_payload maps the new
-- internal names onto those wire keys. tournament-duelz needs no change.

SET XACT_ABORT ON;
BEGIN TRANSACTION;

-- matches (6)
EXEC sp_rename 'matches.team0_name',              'team_a_name',             'COLUMN';
EXEC sp_rename 'matches.team1_name',              'team_b_name',             'COLUMN';
EXEC sp_rename 'matches.team0_colour',            'team_a_colour',           'COLUMN';
EXEC sp_rename 'matches.team1_colour',            'team_b_colour',           'COLUMN';
EXEC sp_rename 'matches.team0_gk_colour',         'team_a_gk_colour',        'COLUMN';
EXEC sp_rename 'matches.team1_gk_colour',         'team_b_gk_colour',        'COLUMN';

-- minute_stats (8)
EXEC sp_rename 'minute_stats.frames_team0',        'frames_team_a',           'COLUMN';
EXEC sp_rename 'minute_stats.frames_team1',        'frames_team_b',           'COLUMN';
EXEC sp_rename 'minute_stats.passes_completed_t0', 'passes_completed_team_a', 'COLUMN';
EXEC sp_rename 'minute_stats.passes_completed_t1', 'passes_completed_team_b', 'COLUMN';
EXEC sp_rename 'minute_stats.interceptions_t0',    'interceptions_team_a',    'COLUMN';
EXEC sp_rename 'minute_stats.interceptions_t1',    'interceptions_team_b',    'COLUMN';
EXEC sp_rename 'minute_stats.ball_lost_t0',        'ball_lost_team_a',        'COLUMN';
EXEC sp_rename 'minute_stats.ball_lost_t1',        'ball_lost_team_b',        'COLUMN';

-- post_processing (14)
EXEC sp_rename 'post_processing.team0_name',           'team_a_name',             'COLUMN';
EXEC sp_rename 'post_processing.team1_name',           'team_b_name',             'COLUMN';
EXEC sp_rename 'post_processing.team0_colour',         'team_a_colour',           'COLUMN';
EXEC sp_rename 'post_processing.team1_colour',         'team_b_colour',           'COLUMN';
EXEC sp_rename 'post_processing.team0_gk_colour',      'team_a_gk_colour',        'COLUMN';
EXEC sp_rename 'post_processing.team1_gk_colour',      'team_b_gk_colour',        'COLUMN';
EXEC sp_rename 'post_processing.frames_team0',         'frames_team_a',           'COLUMN';
EXEC sp_rename 'post_processing.frames_team1',         'frames_team_b',           'COLUMN';
EXEC sp_rename 'post_processing.passes_completed_t0',  'passes_completed_team_a', 'COLUMN';
EXEC sp_rename 'post_processing.passes_completed_t1',  'passes_completed_team_b', 'COLUMN';
EXEC sp_rename 'post_processing.interceptions_t0',     'interceptions_team_a',    'COLUMN';
EXEC sp_rename 'post_processing.interceptions_t1',     'interceptions_team_b',    'COLUMN';
EXEC sp_rename 'post_processing.ball_lost_t0',         'ball_lost_team_a',        'COLUMN';
EXEC sp_rename 'post_processing.ball_lost_t1',         'ball_lost_team_b',        'COLUMN';

COMMIT TRANSACTION;

-- Verify: SELECT TOP 1 * FROM matches;  SELECT TOP 1 * FROM minute_stats;  SELECT TOP 1 * FROM post_processing;


-- =====================================================================
-- ROLLBACK (v6 -> v5): run only if the new build is being reverted.
-- =====================================================================
-- SET XACT_ABORT ON;
-- BEGIN TRANSACTION;
-- EXEC sp_rename 'matches.team_a_name',              'team0_name',              'COLUMN';
-- EXEC sp_rename 'matches.team_b_name',              'team1_name',              'COLUMN';
-- EXEC sp_rename 'matches.team_a_colour',            'team0_colour',            'COLUMN';
-- EXEC sp_rename 'matches.team_b_colour',            'team1_colour',            'COLUMN';
-- EXEC sp_rename 'matches.team_a_gk_colour',         'team0_gk_colour',         'COLUMN';
-- EXEC sp_rename 'matches.team_b_gk_colour',         'team1_gk_colour',         'COLUMN';
-- EXEC sp_rename 'minute_stats.frames_team_a',           'frames_team0',            'COLUMN';
-- EXEC sp_rename 'minute_stats.frames_team_b',           'frames_team1',            'COLUMN';
-- EXEC sp_rename 'minute_stats.passes_completed_team_a', 'passes_completed_t0',     'COLUMN';
-- EXEC sp_rename 'minute_stats.passes_completed_team_b', 'passes_completed_t1',     'COLUMN';
-- EXEC sp_rename 'minute_stats.interceptions_team_a',    'interceptions_t0',        'COLUMN';
-- EXEC sp_rename 'minute_stats.interceptions_team_b',    'interceptions_t1',        'COLUMN';
-- EXEC sp_rename 'minute_stats.ball_lost_team_a',        'ball_lost_t0',            'COLUMN';
-- EXEC sp_rename 'minute_stats.ball_lost_team_b',        'ball_lost_t1',            'COLUMN';
-- EXEC sp_rename 'post_processing.team_a_name',              'team0_name',           'COLUMN';
-- EXEC sp_rename 'post_processing.team_b_name',              'team1_name',           'COLUMN';
-- EXEC sp_rename 'post_processing.team_a_colour',            'team0_colour',         'COLUMN';
-- EXEC sp_rename 'post_processing.team_b_colour',            'team1_colour',         'COLUMN';
-- EXEC sp_rename 'post_processing.team_a_gk_colour',         'team0_gk_colour',      'COLUMN';
-- EXEC sp_rename 'post_processing.team_b_gk_colour',         'team1_gk_colour',      'COLUMN';
-- EXEC sp_rename 'post_processing.frames_team_a',            'frames_team0',         'COLUMN';
-- EXEC sp_rename 'post_processing.frames_team_b',            'frames_team1',         'COLUMN';
-- EXEC sp_rename 'post_processing.passes_completed_team_a',  'passes_completed_t0',  'COLUMN';
-- EXEC sp_rename 'post_processing.passes_completed_team_b',  'passes_completed_t1',  'COLUMN';
-- EXEC sp_rename 'post_processing.interceptions_team_a',     'interceptions_t0',     'COLUMN';
-- EXEC sp_rename 'post_processing.interceptions_team_b',     'interceptions_t1',     'COLUMN';
-- EXEC sp_rename 'post_processing.ball_lost_team_a',         'ball_lost_t0',         'COLUMN';
-- EXEC sp_rename 'post_processing.ball_lost_team_b',         'ball_lost_t1',         'COLUMN';
-- COMMIT TRANSACTION;
