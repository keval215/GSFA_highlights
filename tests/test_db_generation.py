"""Unit tests for the mid-match team/GK colour-change reset logic added to
service/db.py: ensure_match's fit_generation bump, get_fit_generation,
mark_minutes_superseded, cumulative_read's superseded exclusion, and the
manual bump_fit_generation override.

Runs against the in-memory sqlite stand-in (tests/conftest.py) rather than
tests/test_db.py's real-SQL-Server integration tests, which are skipped
without SQL_CONN_STR (true in most dev/CI environments) — these must run
unconditionally.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from service import db
from service.stats import EventRow, MinuteRow


def _row(mid, half, minute, **over):
    return MinuteRow(match_id=mid, half=half, minute=minute, **over)


# ---------------------------------------------------------------------------
# ensure_match: fit_generation bump
# ---------------------------------------------------------------------------

def test_first_insert_does_not_bump_generation(sqlite_conn):
    db.ensure_match(sqlite_conn, "m1", team_a_colour="#FF0000", team_b_colour="#0000FF")
    assert db.get_fit_generation(sqlite_conn, "m1") == 1


def test_first_fill_in_of_a_previously_null_colour_does_not_bump(sqlite_conn):
    # Match created without colours (e.g. worker's no-matches-row fallback).
    db.ensure_match(sqlite_conn, "m1")
    assert db.get_fit_generation(sqlite_conn, "m1") == 1
    # First time colours arrive — a fill-in, not a change.
    db.ensure_match(sqlite_conn, "m1", team_a_colour="#FF0000", team_b_colour="#0000FF")
    assert db.get_fit_generation(sqlite_conn, "m1") == 1

    cur = sqlite_conn.cursor()
    cur.execute("SELECT team_a_colour, team_b_colour FROM matches WHERE match_id = ?", "m1")
    assert tuple(cur.fetchone()) == ("#FF0000", "#0000FF")


def test_real_colour_change_bumps_generation_once_and_overwrites(sqlite_conn):
    db.ensure_match(sqlite_conn, "m1", team_a_colour="#FF0000", team_b_colour="#0000FF")
    db.ensure_match(sqlite_conn, "m1", team_a_colour="#00FF00", team_b_colour="#0000FF")
    assert db.get_fit_generation(sqlite_conn, "m1") == 2

    cur = sqlite_conn.cursor()
    cur.execute("SELECT team_a_colour, team_b_colour FROM matches WHERE match_id = ?", "m1")
    assert tuple(cur.fetchone()) == ("#00FF00", "#0000FF")


def test_colour_change_detection_is_case_and_hash_insensitive(sqlite_conn):
    db.ensure_match(sqlite_conn, "m1", team_a_colour="#FF6600")
    # Same colour, different casing/leading '#' — not a real change.
    db.ensure_match(sqlite_conn, "m1", team_a_colour="ff6600")
    assert db.get_fit_generation(sqlite_conn, "m1") == 1


def test_gk_colour_change_also_bumps_generation(sqlite_conn):
    db.ensure_match(sqlite_conn, "m1", team_a_gk_colour="yellow", team_b_gk_colour="black")
    db.ensure_match(sqlite_conn, "m1", team_a_gk_colour="green", team_b_gk_colour="black")
    assert db.get_fit_generation(sqlite_conn, "m1") == 2


def test_unrelated_field_update_does_not_bump_generation(sqlite_conn):
    db.ensure_match(sqlite_conn, "m1", team_a_name="Alpha", team_a_colour="#FF0000")
    db.ensure_match(sqlite_conn, "m1", team_b_name="Bravo")
    assert db.get_fit_generation(sqlite_conn, "m1") == 1


def test_names_stay_coalesced_never_overwritten(sqlite_conn):
    db.ensure_match(sqlite_conn, "m1", team_a_name="Alpha", team_b_name="Bravo")
    db.ensure_match(sqlite_conn, "m1", team_a_name="Charlie", team_b_name="Delta",
                    team_a_colour="#FF0000", team_b_colour="#00FF00")
    cur = sqlite_conn.cursor()
    cur.execute("SELECT team_a_name, team_b_name FROM matches WHERE match_id = ?", "m1")
    assert tuple(cur.fetchone()) == ("Alpha", "Bravo")


def test_ruleset_unchanged_by_later_calls(sqlite_conn):
    db.ensure_match(sqlite_conn, "m1", ruleset="futsal")
    db.ensure_match(sqlite_conn, "m1", ruleset="classic", team_a_colour="#FF0000")
    assert db.get_match_ruleset(sqlite_conn, "m1") == "futsal"


# ---------------------------------------------------------------------------
# get_fit_generation
# ---------------------------------------------------------------------------

def test_get_fit_generation_unknown_match_defaults_to_1(sqlite_conn):
    assert db.get_fit_generation(sqlite_conn, "does-not-exist") == 1


# ---------------------------------------------------------------------------
# mark_minutes_superseded + cumulative_read exclusion
# ---------------------------------------------------------------------------

def _seed_minutes(conn, mid):
    db.ensure_match(conn, mid)
    db.write_clip_result(conn, _row(mid, 1, 1, frames_team_a=100, frames_team_b=0), None,
                         [EventRow(half=1, minute=1, frame_idx=1, kind="pass", from_team=0)])
    db.write_clip_result(conn, _row(mid, 1, 2, frames_team_a=100, frames_team_b=0), None, [])
    db.write_clip_result(conn, _row(mid, 1, 3, frames_team_a=100, frames_team_b=0), None, [])


def test_mark_minutes_superseded_scopes_to_upto_half_minute(sqlite_conn):
    _seed_minutes(sqlite_conn, "m1")
    n = db.mark_minutes_superseded(sqlite_conn, "m1", upto_half=1, upto_minute=2)
    # minute 1 (with its one event) + minute 2 rows flagged; minute 3 untouched.
    assert n == 3  # 2 minute_stats rows + 1 events row

    cur = sqlite_conn.cursor()
    cur.execute(
        "SELECT minute, superseded FROM minute_stats WHERE match_id = ? ORDER BY minute", "m1",
    )
    rows = cur.fetchall()
    assert [(r[0], r[1]) for r in rows] == [(1, 1), (2, 1), (3, 0)]

    cur.execute("SELECT superseded FROM events WHERE match_id = ?", "m1")
    assert cur.fetchone()[0] == 1


def test_mark_minutes_superseded_on_fresh_match_is_noop(sqlite_conn):
    db.ensure_match(sqlite_conn, "m1")
    n = db.mark_minutes_superseded(sqlite_conn, "m1", upto_half=1, upto_minute=0)
    assert n == 0


def test_mark_minutes_superseded_is_idempotent(sqlite_conn):
    _seed_minutes(sqlite_conn, "m1")
    db.mark_minutes_superseded(sqlite_conn, "m1", upto_half=1, upto_minute=2)
    n_second = db.mark_minutes_superseded(sqlite_conn, "m1", upto_half=1, upto_minute=2)
    assert n_second == 0


def test_cumulative_read_excludes_superseded_minutes(sqlite_conn):
    _seed_minutes(sqlite_conn, "m1")
    db.mark_minutes_superseded(sqlite_conn, "m1", upto_half=1, upto_minute=2)
    sums = db.cumulative_read(sqlite_conn, "m1", 1, 3)
    # Only minute 3's 100 frames count — minutes 1-2 (200 frames) are superseded.
    assert sums["frames_team_a"] == 100


def test_cumulative_read_includes_everything_when_nothing_superseded(sqlite_conn):
    _seed_minutes(sqlite_conn, "m1")
    sums = db.cumulative_read(sqlite_conn, "m1", 1, 3)
    assert sums["frames_team_a"] == 300


# ---------------------------------------------------------------------------
# bump_fit_generation (manual reset-fit override)
# ---------------------------------------------------------------------------

def test_bump_fit_generation_unconditional_and_overwrites_supplied_fields(sqlite_conn):
    db.ensure_match(sqlite_conn, "m1", team_a_name="Alpha", team_a_colour="#FF0000")
    new_gen = db.bump_fit_generation(sqlite_conn, "m1", team_a_colour="#FF0000")  # same colour!
    assert new_gen == 2   # unconditional — bumps even with no real change

    cur = sqlite_conn.cursor()
    cur.execute("SELECT team_a_name, team_a_colour FROM matches WHERE match_id = ?", "m1")
    assert tuple(cur.fetchone()) == ("Alpha", "#FF0000")


def test_bump_fit_generation_overwrites_names_too(sqlite_conn):
    db.ensure_match(sqlite_conn, "m1", team_a_name="Alpha")
    db.bump_fit_generation(sqlite_conn, "m1", team_a_name="Charlie")
    cur = sqlite_conn.cursor()
    cur.execute("SELECT team_a_name FROM matches WHERE match_id = ?", "m1")
    assert cur.fetchone()[0] == "Charlie"


def test_bump_fit_generation_unknown_match_returns_none(sqlite_conn):
    assert db.bump_fit_generation(sqlite_conn, "does-not-exist") is None


def test_bump_fit_generation_repeated_calls_keep_incrementing(sqlite_conn):
    db.ensure_match(sqlite_conn, "m1")
    assert db.bump_fit_generation(sqlite_conn, "m1") == 2
    assert db.bump_fit_generation(sqlite_conn, "m1") == 3
