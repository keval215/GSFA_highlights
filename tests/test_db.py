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
    db.ensure_match(conn, mid, team0_name="A", team1_name="B")
    yield mid
    cur = conn.cursor()
    for table in ("callback_outbox", "events", "minute_stats", "matches"):
        cur.execute(f"DELETE FROM {table} WHERE match_id = ?", mid)
    conn.commit()


def _row(mid, half, minute, **over):
    return MinuteRow(match_id=mid, half=half, minute=minute, **over)


def test_upsert_idempotency(conn, match_id):
    row = _row(match_id, 1, 1, frames_team0=100, frames_team1=50)
    db.write_clip_result(conn, row, None, [])
    db.write_clip_result(conn, row, None, [])     # replay
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM minute_stats WHERE match_id = ?", match_id)
    assert cur.fetchone()[0] == 1


def test_correction_bumps_revision_and_cumulative_is_correct(conn, match_id):
    db.write_clip_result(conn, _row(match_id, 1, 1, frames_team0=100, frames_team1=50), None, [])
    correction = PriorCorrection(half=1, minute=1, kind="flip_to", team_id=1, frames=10)
    payload = db.write_clip_result(
        conn, _row(match_id, 1, 2, frames_team0=80, frames_team1=70), correction, [],
    )

    cur = conn.cursor()
    cur.execute(
        "SELECT frames_team0, frames_team1, revision FROM minute_stats "
        "WHERE match_id = ? AND half = 1 AND minute = 1", match_id,
    )
    t0, t1, rev = cur.fetchone()
    assert (t0, t1, rev) == (90, 60, 1)

    # Cumulative payload reflects the corrected minute 1: t0=170, t1=130
    sums = db.cumulative_read(conn, match_id, 1, 2)
    assert sums["frames_team0"] == 170
    assert sums["frames_team1"] == 130
    # Flat advance-stats body: team 0 → a, team 1 → b.
    assert payload["frames_a"] == 170
    assert payload["frames_b"] == 130


def test_events_and_outbox_written(conn, match_id):
    events = [EventRow(half=1, minute=1, frame_idx=42, kind="pass",
                       from_team=0, to_team=0, from_track=3, to_track=7, travel_frames=5)]
    db.write_clip_result(conn, _row(match_id, 1, 1, passes_completed_t0=1), None, events)

    cur = conn.cursor()
    cur.execute("SELECT kind FROM events WHERE match_id = ?", match_id)
    assert cur.fetchone()[0] == "pass"

    pending = db.fetch_pending(conn, match_id)
    assert len(pending) == 1
    assert pending[0].payload["passes_completed_a"] == 1


def test_progress_advances(conn, match_id):
    db.write_clip_result(conn, _row(match_id, 1, 1), None, [])
    db.write_clip_result(conn, _row(match_id, 1, 2), None, [])
    half, minute, _ = db.get_match_progress(conn, match_id)
    assert (half, minute) == (1, 2)
