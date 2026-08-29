"""Integration tests for service/db.py against a real Azure SQL database.

Skipped unless SQL_CONN_STR is set (and pyodbc installed). These are the
step-1 acceptance tests from implementation.md §8:
  • upsert idempotency (same key twice ⇒ one row)
  • correction bumps revision
  • cumulative query matches hand-computed sums

Uses a throwaway match_id and cleans up after itself.
"""

import os
import sys
import threading
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("pyodbc")
if not os.environ.get("SQL_CONN_STR"):
    pytest.skip("SQL_CONN_STR not set — skipping DB integration tests",
                allow_module_level=True)

from service import db
from service.stats import EventRow, MinuteRow, PriorCorrection


@pytest.fixture
def conn():
    c = db.get_conn()
    yield c
    c.close()


@pytest.fixture
def match_id(conn):
    mid = f"test_{uuid.uuid4().hex[:12]}"
    db.ensure_match(conn, mid, team_a_name="A", team_b_name="B")
    yield mid
    cur = conn.cursor()
    for table in ("callback_outbox", "events", "minute_stats", "matches"):
        cur.execute(f"DELETE FROM {table} WHERE match_id = ?", mid)
    conn.commit()


def _row(mid, half, minute, **over):
    return MinuteRow(match_id=mid, half=half, minute=minute, **over)


def test_upsert_idempotency(conn, match_id):
    row = _row(match_id, 1, 1, frames_team_a=100, frames_team_b=50)
    db.write_clip_result(conn, row, None, [])
    db.write_clip_result(conn, row, None, [])     # replay
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM minute_stats WHERE match_id = ?", match_id)
    assert cur.fetchone()[0] == 1


def test_correction_bumps_revision_and_cumulative_is_correct(conn, match_id):
    db.write_clip_result(conn, _row(match_id, 1, 1, frames_team_a=100, frames_team_b=50), None, [])
    correction = PriorCorrection(half=1, minute=1, kind="flip_to", team_id=1, frames=10)
    payload = db.write_clip_result(
        conn, _row(match_id, 1, 2, frames_team_a=80, frames_team_b=70), correction, [],
    )

    cur = conn.cursor()
    cur.execute(
        "SELECT frames_team_a, frames_team_b, revision FROM minute_stats "
        "WHERE match_id = ? AND half = 1 AND minute = 1", match_id,
    )
    t0, t1, rev = cur.fetchone()
    assert (t0, t1, rev) == (90, 60, 1)

    # Cumulative payload reflects the corrected minute 1: t0=170, t1=130
    sums = db.cumulative_read(conn, match_id, 1, 2)
    assert sums["frames_team_a"] == 170
    assert sums["frames_team_b"] == 130
    # Flat advance-stats body: team 0 → a, team 1 → b.
    assert payload["frames_a"] == 170
    assert payload["frames_b"] == 130


def test_events_and_outbox_written(conn, match_id):
    events = [EventRow(half=1, minute=1, frame_idx=42, kind="pass",
                       from_team=0, to_team=0, from_track=3, to_track=7, travel_frames=5)]
    db.write_clip_result(
        conn,
        _row(match_id, 1, 1, passes_completed_team_a=1, clip_duration_seconds=60.0),
        None,
        events,
    )

    cur = conn.cursor()
    cur.execute("SELECT kind FROM events WHERE match_id = ?", match_id)
    assert cur.fetchone()[0] == "pass"

    pending = db.fetch_pending(conn, match_id)
    assert len(pending) == 1
    assert pending[0].payload["passes_completed_a"] == 1


def test_clip_duration_seconds_is_persisted(conn, match_id):
    db.write_clip_result(
        conn,
        _row(match_id, 1, 1, clip_duration_seconds=63.25),
        None,
        [],
    )
    cur = conn.cursor()
    cur.execute(
        "SELECT clip_duration_seconds FROM minute_stats WHERE match_id = ? AND half = 1 AND minute = 1",
        match_id,
    )
    assert float(cur.fetchone()[0]) == 63.25


def test_post_processing_upsert(conn, match_id):
    row = _row(
        match_id, 1, 1,
        frames_team_a=120, frames_team_b=80,
        frames_loose=12, frames_oof=3,
        passes_completed_team_a=4, passes_completed_team_b=5,
        interceptions_team_a=1, interceptions_team_b=2,
        ball_lost_team_a=7, ball_lost_team_b=8,
    )
    db.write_post_processing_result(
        conn,
        row,
        team_a_name="A",
        team_b_name="B",
        team_a_colour="#FF6600",
        team_b_colour="#0033FF",
        video_blob_path="clips/test/post_processing.mp4",
    )
    db.write_post_processing_result(
        conn,
        row,
        team_a_name="A",
        team_b_name="B",
        team_a_colour="#FF6600",
        team_b_colour="#0033FF",
        video_blob_path="clips/test/post_processing.mp4",
    )

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM post_processing WHERE match_id = ?", match_id)
    assert cur.fetchone()[0] == 1


def test_progress_advances(conn, match_id):
    db.write_clip_result(conn, _row(match_id, 1, 1), None, [])
    db.write_clip_result(conn, _row(match_id, 1, 2), None, [])
    half, minute = db.get_match_progress(conn, match_id)
    assert (half, minute) == (1, 2)


def test_claim_next_minute_is_atomic(match_id):
    """Two threads racing claim_next_minute must receive distinct sequential values."""
    results, errors = [], []

    def claim():
        try:
            c = db.get_conn()
            try:
                results.append(db.claim_next_minute(c, match_id, 1))
            finally:
                c.close()
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=claim)
    t2 = threading.Thread(target=claim)
    t1.start(); t2.start()
    t1.join();  t2.join()

    assert not errors, f"errors during concurrent claim: {errors}"
    assert sorted(results) == [1, 2], f"expected [1, 2], got {sorted(results)}"
