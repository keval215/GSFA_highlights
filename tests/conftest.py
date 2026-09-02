"""tests/conftest.py — shared pytest fixtures.

`sqlite_conn` / `sqlite_conn_factory`: a pyodbc-compatible stand-in over an
in-memory SQLite database, built from a SQLite-friendly copy of the
matches/minute_stats/post_processing/events/callback_outbox tables in
sql/schema.sql (`?` placeholders and COALESCE/SUM are sqlite-native;
SYSUTCDATETIME() is registered as a custom SQL function since the T-SQL
builtin isn't available). This lets the fit_generation/superseded DB-logic
tests run unconditionally and fast — unlike tests/test_db.py, which is
skipped whenever SQL_CONN_STR isn't set (true in most dev/CI environments).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

_SCHEMA = """
CREATE TABLE matches (
  match_id              TEXT PRIMARY KEY,
  team_a_name           TEXT,
  team_b_name           TEXT,
  team_a_colour         TEXT,
  team_b_colour         TEXT,
  team_a_gk_colour      TEXT,
  team_b_gk_colour      TEXT,
  ruleset               TEXT NOT NULL DEFAULT 'classic',
  fit_generation        INTEGER NOT NULL DEFAULT 1,
  last_half_processed   INTEGER NOT NULL DEFAULT 1,
  last_minute_processed INTEGER NOT NULL DEFAULT 0,
  next_clip_seq_h1      INTEGER NOT NULL DEFAULT 0,
  next_clip_seq_h2      INTEGER NOT NULL DEFAULT 0,
  created_at            TEXT,
  updated_at            TEXT
);

CREATE TABLE minute_stats (
  match_id                 TEXT NOT NULL,
  half                     INTEGER NOT NULL,
  minute                   INTEGER NOT NULL,
  clip_duration_seconds    REAL,
  frames_team_a            INTEGER NOT NULL DEFAULT 0,
  frames_team_b            INTEGER NOT NULL DEFAULT 0,
  frames_loose             INTEGER NOT NULL DEFAULT 0,
  frames_oof               INTEGER NOT NULL DEFAULT 0,
  passes_completed_team_a  INTEGER NOT NULL DEFAULT 0,
  passes_completed_team_b  INTEGER NOT NULL DEFAULT 0,
  interceptions_team_a     INTEGER NOT NULL DEFAULT 0,
  interceptions_team_b     INTEGER NOT NULL DEFAULT 0,
  ball_lost_team_a         INTEGER NOT NULL DEFAULT 0,
  ball_lost_team_b         INTEGER NOT NULL DEFAULT 0,
  revision                 INTEGER NOT NULL DEFAULT 0,
  superseded               INTEGER NOT NULL DEFAULT 0,
  clip_blob_path           TEXT,
  processed_at             TEXT,
  PRIMARY KEY (match_id, half, minute)
);

CREATE TABLE post_processing (
  match_id                 TEXT PRIMARY KEY,
  team_a_name              TEXT,
  team_b_name              TEXT,
  team_a_colour            TEXT,
  team_b_colour            TEXT,
  team_a_gk_colour         TEXT,
  team_b_gk_colour         TEXT,
  frames_team_a            INTEGER NOT NULL DEFAULT 0,
  frames_team_b            INTEGER NOT NULL DEFAULT 0,
  frames_loose             INTEGER NOT NULL DEFAULT 0,
  frames_oof               INTEGER NOT NULL DEFAULT 0,
  passes_completed_team_a  INTEGER NOT NULL DEFAULT 0,
  passes_completed_team_b  INTEGER NOT NULL DEFAULT 0,
  interceptions_team_a     INTEGER NOT NULL DEFAULT 0,
  interceptions_team_b     INTEGER NOT NULL DEFAULT 0,
  ball_lost_team_a         INTEGER NOT NULL DEFAULT 0,
  ball_lost_team_b         INTEGER NOT NULL DEFAULT 0,
  video_blob_path          TEXT,
  processed_at             TEXT
);

CREATE TABLE events (
  event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  match_id      TEXT NOT NULL,
  half          INTEGER NOT NULL,
  minute        INTEGER NOT NULL,
  frame_idx     INTEGER NOT NULL,
  kind          TEXT NOT NULL,
  from_team     INTEGER,
  to_team       INTEGER,
  from_track    INTEGER,
  to_track      INTEGER,
  travel_frames INTEGER,
  superseded    INTEGER NOT NULL DEFAULT 0,
  created_at    TEXT
);

CREATE TABLE callback_outbox (
  outbox_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  match_id        TEXT NOT NULL,
  half            INTEGER NOT NULL,
  minute          INTEGER NOT NULL,
  payload         TEXT NOT NULL,
  status          TEXT NOT NULL DEFAULT 'pending',
  attempts        INTEGER NOT NULL DEFAULT 0,
  last_attempt_at TEXT
);
"""


class _CursorAdapter:
    """Wraps sqlite3.Cursor so service/db.py's pyodbc-style calling
    convention — `cur.execute(sql, p1, p2, ...)`, params given as separate
    positional args, never pre-packed into a tuple — works unmodified
    against sqlite3, whose `execute()` wants exactly one sequence
    argument."""

    def __init__(self, cur: sqlite3.Cursor) -> None:
        self._cur = cur

    def execute(self, sql: str, *params):
        self._cur.execute(sql, params)
        return self

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    @property
    def rowcount(self) -> int:
        return self._cur.rowcount


class _ConnAdapter:
    """Wraps sqlite3.Connection to expose the same surface service/db.py
    uses on a pyodbc connection (.cursor()/.commit()/.rollback()/.close())."""

    def __init__(self, raw: sqlite3.Connection, close_raw: bool = True) -> None:
        self._raw = raw
        self._close_raw = close_raw

    def cursor(self) -> _CursorAdapter:
        return _CursorAdapter(self._raw.cursor())

    def commit(self) -> None:
        self._raw.commit()

    def rollback(self) -> None:
        self._raw.rollback()

    def close(self) -> None:
        # A real pyodbc connection is opened/closed per call (service/db.py's
        # normal pattern); an in-memory sqlite DB would lose all its data if
        # we did the same here. Non-factory use (the `sqlite_conn` fixture)
        # closes for real; the `sqlite_conn_factory` fixture's wrappers no-op
        # here and let the raw connection close once, at fixture teardown.
        if self._close_raw:
            self._raw.close()


def _make_raw_conn() -> sqlite3.Connection:
    # check_same_thread=False: a real pyodbc connection has no such
    # restriction, and FastAPI's TestClient (used by test_api_reset_fit.py)
    # runs the request in a worker thread via anyio — the same in-memory
    # connection must be usable from there too. Test-only DB, single-
    # threaded access in practice (no concurrent requests in these tests).
    raw = sqlite3.connect(":memory:", check_same_thread=False)
    raw.create_function("SYSUTCDATETIME", 0, lambda: datetime.now(timezone.utc).isoformat())
    raw.executescript(_SCHEMA)
    return raw


@pytest.fixture
def sqlite_conn():
    """A single pyodbc-shaped connection over a fresh in-memory SQLite DB,
    for tests that only ever hold one `conn` (mirrors tests/test_db.py's own
    `conn` fixture)."""
    raw = _make_raw_conn()
    conn = _ConnAdapter(raw)
    yield conn
    conn.close()


@pytest.fixture
def sqlite_conn_factory():
    """A zero-arg callable shaped like `db.get_conn` — every call returns a
    fresh wrapper around the SAME underlying in-memory sqlite3 connection
    (so data persists across the open/close-per-call pattern service code
    uses, e.g. session.py's `conn = db.get_conn(); ...; conn.close()`).
    Use this (instead of `sqlite_conn`) when monkeypatching `db.get_conn`
    for code that opens its own connection internally."""
    raw = _make_raw_conn()
    yield lambda: _ConnAdapter(raw, close_raw=False)
    raw.close()
