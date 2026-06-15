"""
service/db.py — pyodbc layer for Azure SQL gsfa_stats.

Design rules (implementation.md, Conflict 3):
  • minute_stats holds RAW per-minute counters only — one row per clip.
  • Cumulative numbers are computed on read with SUM(); never stored,
    and percentages are derived AFTER the sums (never averaged).
  • Retroactive corrections are an UPDATE to the affected prior-minute
    row with revision += 1, applied in the SAME transaction as the
    current minute's insert, so the cumulative read in that transaction
    already includes them.
  • The callback_outbox row is written in the same transaction
    (transactional outbox pattern).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import pyodbc

from service import config
from service.stats import EventRow, MinuteRow, PriorCorrection, build_payload

__all__ = [
    "EventRow", "MinuteRow", "PriorCorrection", "OutboxRow", "build_payload",
    "get_conn", "ensure_match", "get_team_specs", "get_match_progress",
    "minute_exists", "cumulative_read", "write_clip_result",
    "fetch_pending", "mark_sent", "mark_failed",
]


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_conn() -> pyodbc.Connection:
    conn = pyodbc.connect(config.sql_conn_str(), autocommit=False)
    return conn


# ---------------------------------------------------------------------------
# Match upsert / reads
# ---------------------------------------------------------------------------

def ensure_match(
    conn: pyodbc.Connection,
    match_id: str,
    venue_id: Optional[str] = None,
    team0_name: Optional[str] = None,
    team1_name: Optional[str] = None,
    team0_colour: Optional[str] = None,
    team1_colour: Optional[str] = None,
) -> None:
    """First clip auto-creates the match; later calls only fill in missing
    metadata. Commits."""
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM matches WHERE match_id = ?", match_id)
    if cur.fetchone() is None:
        cur.execute(
            "INSERT INTO matches "
            "  (match_id, venue_id, team0_name, team1_name, team0_colour, team1_colour) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            match_id, venue_id, team0_name, team1_name, team0_colour, team1_colour,
        )
    else:
        cur.execute(
            "UPDATE matches SET "
            "  venue_id     = COALESCE(venue_id, ?), "
            "  team0_name   = COALESCE(team0_name, ?), "
            "  team1_name   = COALESCE(team1_name, ?), "
            "  team0_colour = COALESCE(team0_colour, ?), "
            "  team1_colour = COALESCE(team1_colour, ?), "
            "  updated_at = SYSUTCDATETIME() "
            "WHERE match_id = ?",
            venue_id, team0_name, team1_name, team0_colour, team1_colour, match_id,
        )
    conn.commit()


def get_team_specs(
    conn: pyodbc.Connection, match_id: str,
) -> Optional[list[tuple[str, str]]]:
    """[(team0_name, team0_colour), (team1_name, team1_colour)] for colour→name
    resolution, or None unless BOTH teams have a name AND a colour. Cluster
    matching needs all four values, so a partial set is treated as 'not set'."""
    cur = conn.cursor()
    cur.execute(
        "SELECT team0_name, team0_colour, team1_name, team1_colour "
        "FROM matches WHERE match_id = ?",
        match_id,
    )
    row = cur.fetchone()
    if row is None or not all(row):
        return None
    return [(str(row[0]), str(row[1])), (str(row[2]), str(row[3]))]


def get_match_progress(conn: pyodbc.Connection, match_id: str) -> Optional[tuple[int, int]]:
    """(last_half_processed, last_minute_processed) or None."""
    cur = conn.cursor()
    cur.execute(
        "SELECT last_half_processed, last_minute_processed "
        "FROM matches WHERE match_id = ?",
        match_id,
    )
    row = cur.fetchone()
    return (int(row[0]), int(row[1])) if row else None


def minute_exists(conn: pyodbc.Connection, match_id: str, half: int, minute: int) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM minute_stats WHERE match_id = ? AND half = ? AND minute = ?",
        match_id, half, minute,
    )
    return cur.fetchone() is not None


# ---------------------------------------------------------------------------
# Cumulative read  (the "minute N returns totals 1..N" requirement)
# ---------------------------------------------------------------------------

_SUM_COLS = (
    "frames_team0", "frames_team1", "frames_loose", "frames_oof",
    "passes_completed_t0", "passes_completed_t1",
    "interceptions_t0", "interceptions_t1",
    "ball_lost_t0", "ball_lost_t1",
)


def cumulative_read(
    conn: pyodbc.Connection, match_id: str, half: int, minute: int,
) -> dict[str, int]:
    cur = conn.cursor()
    cols = ", ".join(f"COALESCE(SUM({c}), 0)" for c in _SUM_COLS)
    cur.execute(
        f"SELECT {cols} FROM minute_stats "
        "WHERE match_id = ? AND (half < ? OR (half = ? AND minute <= ?))",
        match_id, half, half, minute,
    )
    row = cur.fetchone()
    return {c: int(v) for c, v in zip(_SUM_COLS, row)}


# ---------------------------------------------------------------------------
# The per-clip transaction
# ---------------------------------------------------------------------------

def write_clip_result(
    conn: pyodbc.Connection,
    row: MinuteRow,
    correction: Optional[PriorCorrection],
    events: list[EventRow],
    team_id_to_name: Optional[dict[int, str]] = None,
) -> dict:
    """One SQL transaction per processed clip:
        upsert minute row + UPDATE prior minute (revision += 1) if corrected
        + insert events + update matches progress + insert outbox row
        (payload = fresh cumulative read, post-correction).
    Returns the payload that was placed in the outbox."""
    cur = conn.cursor()
    try:
        # 1. Apply the retroactive correction to the prior minute first so
        #    the cumulative read below already includes it.
        if correction is not None and correction.frames > 0:
            _apply_prior_correction(cur, row.match_id, correction)

        # 2. Upsert this minute's row (replay of the same clip overwrites —
        #    PK (match_id, half, minute) keeps it a single row).
        cur.execute(
            "SELECT revision FROM minute_stats "
            "WHERE match_id = ? AND half = ? AND minute = ?",
            row.match_id, row.half, row.minute,
        )
        existing = cur.fetchone()
        if existing is None:
            revision = 0
            cur.execute(
                "INSERT INTO minute_stats (match_id, half, minute, "
                "  frames_team0, frames_team1, frames_loose, frames_oof, "
                "  passes_completed_t0, passes_completed_t1, "
                "  interceptions_t0, interceptions_t1, "
                "  ball_lost_t0, ball_lost_t1, revision, clip_blob_path) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
                row.match_id, row.half, row.minute,
                row.frames_team0, row.frames_team1, row.frames_loose, row.frames_oof,
                row.passes_completed_t0, row.passes_completed_t1,
                row.interceptions_t0, row.interceptions_t1,
                row.ball_lost_t0, row.ball_lost_t1,
                row.clip_blob_path,
            )
        else:
            revision = int(existing[0])
            cur.execute(
                "UPDATE minute_stats SET "
                "  frames_team0 = ?, frames_team1 = ?, frames_loose = ?, frames_oof = ?, "
                "  passes_completed_t0 = ?, passes_completed_t1 = ?, "
                "  interceptions_t0 = ?, interceptions_t1 = ?, "
                "  ball_lost_t0 = ?, ball_lost_t1 = ?, "
                "  clip_blob_path = ?, processed_at = SYSUTCDATETIME() "
                "WHERE match_id = ? AND half = ? AND minute = ?",
                row.frames_team0, row.frames_team1, row.frames_loose, row.frames_oof,
                row.passes_completed_t0, row.passes_completed_t1,
                row.interceptions_t0, row.interceptions_t1,
                row.ball_lost_t0, row.ball_lost_t1,
                row.clip_blob_path,
                row.match_id, row.half, row.minute,
            )

        # 3. Insert events.
        for e in events:
            cur.execute(
                "INSERT INTO events (match_id, half, minute, frame_idx, kind, "
                "  from_team, to_team, from_track, to_track, travel_frames) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row.match_id, e.half, e.minute, e.frame_idx, e.kind,
                e.from_team, e.to_team, e.from_track, e.to_track, e.travel_frames,
            )

        # 4. Advance match progress.
        cur.execute(
            "UPDATE matches SET last_half_processed = ?, last_minute_processed = ?, "
            "  updated_at = SYSUTCDATETIME() WHERE match_id = ?",
            row.half, row.minute, row.match_id,
        )

        # 5. Outbox row with a fresh cumulative payload (corrections included).
        sums    = cumulative_read(conn, row.match_id, row.half, row.minute)
        payload = build_payload(row.match_id, row.half, row.minute, revision, sums,
                                team_id_to_name)
        cur.execute(
            "INSERT INTO callback_outbox (match_id, half, minute, payload) "
            "VALUES (?, ?, ?, ?)",
            row.match_id, row.half, row.minute, json.dumps(payload),
        )

        conn.commit()
        return payload
    except Exception:
        conn.rollback()
        raise


def _apply_prior_correction(cur, match_id: str, c: PriorCorrection) -> None:
    """Read-modify-write with clamping (single worker ⇒ no contention).
    flip_to: move frames from the other team's count to c.team_id.
    drop:    move frames from c.team_id's count to frames_oof."""
    cur.execute(
        "SELECT frames_team0, frames_team1, frames_oof FROM minute_stats "
        "WHERE match_id = ? AND half = ? AND minute = ?",
        match_id, c.half, c.minute,
    )
    prior = cur.fetchone()
    if prior is None:
        return  # prior minute row missing (e.g. logged gap) — nothing to correct
    t0, t1, oof = int(prior[0]), int(prior[1]), int(prior[2])

    if c.kind == "flip_to":
        if c.team_id == 0:
            move = min(c.frames, t1)
            t1 -= move; t0 += move
        else:
            move = min(c.frames, t0)
            t0 -= move; t1 += move
    elif c.kind == "drop":
        if c.team_id == 0:
            move = min(c.frames, t0)
            t0 -= move; oof += move
        else:
            move = min(c.frames, t1)
            t1 -= move; oof += move
    else:
        return

    cur.execute(
        "UPDATE minute_stats SET frames_team0 = ?, frames_team1 = ?, frames_oof = ?, "
        "  revision = revision + 1 "
        "WHERE match_id = ? AND half = ? AND minute = ?",
        t0, t1, oof, match_id, c.half, c.minute,
    )


# ---------------------------------------------------------------------------
# Outbox operations (consumed by notifier)
# ---------------------------------------------------------------------------

@dataclass
class OutboxRow:
    outbox_id: int
    match_id:  str
    half:      int
    minute:    int
    payload:   dict
    attempts:  int


def fetch_pending(conn: pyodbc.Connection, match_id: str) -> list[OutboxRow]:
    """Pending outbox rows for one match in strict (half, minute) order."""
    cur = conn.cursor()
    cur.execute(
        "SELECT outbox_id, match_id, half, minute, payload, attempts "
        "FROM callback_outbox WHERE status = 'pending' AND match_id = ? "
        "ORDER BY half, minute",
        match_id,
    )
    return [
        OutboxRow(int(r[0]), str(r[1]), int(r[2]), int(r[3]), json.loads(r[4]), int(r[5]))
        for r in cur.fetchall()
    ]


def mark_sent(conn: pyodbc.Connection, outbox_id: int, attempts: int) -> None:
    cur = conn.cursor()
    cur.execute(
        "UPDATE callback_outbox SET status = 'sent', attempts = ?, "
        "  last_attempt_at = SYSUTCDATETIME() WHERE outbox_id = ?",
        attempts, outbox_id,
    )
    conn.commit()


def mark_failed(conn: pyodbc.Connection, outbox_id: int, attempts: int) -> None:
    cur = conn.cursor()
    cur.execute(
        "UPDATE callback_outbox SET status = 'failed', attempts = ?, "
        "  last_attempt_at = SYSUTCDATETIME() WHERE outbox_id = ?",
        attempts, outbox_id,
    )
    conn.commit()
