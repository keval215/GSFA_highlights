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
    "get_conn", "ensure_match", "get_team_specs", "get_gk_colours", "get_match_progress",
    "get_match_ruleset", "claim_next_minute", "minute_exists", "cumulative_read",
    "write_clip_result", "post_processing_exists", "write_post_processing_result",
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
    team_a_name: Optional[str] = None,
    team_b_name: Optional[str] = None,
    team_a_colour: Optional[str] = None,
    team_b_colour: Optional[str] = None,
    team_a_gk_colour: Optional[str] = None,
    team_b_gk_colour: Optional[str] = None,
    ruleset: str = "classic",
) -> None:
    """First clip auto-creates the match; later calls only fill in missing
    metadata. `ruleset` is only used on creation — it is fixed for the life
    of the match (team fit, tracker, and pass-FSM state all assume one
    ruleset), so later calls never change it. team_a = CV cluster id 0,
    team_b = cluster id 1. Commits."""
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM matches WHERE match_id = ?", match_id)
    if cur.fetchone() is None:
        cur.execute(
            "INSERT INTO matches "
            "  (match_id, team_a_name, team_b_name, team_a_colour, team_b_colour, "
            "   team_a_gk_colour, team_b_gk_colour, ruleset) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            match_id, team_a_name, team_b_name, team_a_colour, team_b_colour,
            team_a_gk_colour, team_b_gk_colour, ruleset,
        )
    else:
        cur.execute(
            "UPDATE matches SET "
            "  team_a_name      = COALESCE(team_a_name, ?), "
            "  team_b_name      = COALESCE(team_b_name, ?), "
            "  team_a_colour    = COALESCE(team_a_colour, ?), "
            "  team_b_colour    = COALESCE(team_b_colour, ?), "
            "  team_a_gk_colour = COALESCE(team_a_gk_colour, ?), "
            "  team_b_gk_colour = COALESCE(team_b_gk_colour, ?), "
            "  updated_at = SYSUTCDATETIME() "
            "WHERE match_id = ?",
            team_a_name, team_b_name, team_a_colour, team_b_colour,
            team_a_gk_colour, team_b_gk_colour, match_id,
        )
    conn.commit()


def get_team_specs(
    conn: pyodbc.Connection, match_id: str,
) -> Optional[list[tuple[str, str]]]:
    """[(team_a_name, team_a_colour), (team_b_name, team_b_colour)] for colour→name
    resolution, or None unless BOTH teams have a name AND a colour. Cluster
    matching needs all four values, so a partial set is treated as 'not set'."""
    cur = conn.cursor()
    cur.execute(
        "SELECT team_a_name, team_a_colour, team_b_name, team_b_colour "
        "FROM matches WHERE match_id = ?",
        match_id,
    )
    row = cur.fetchone()
    if row is None or not all(row):
        return None
    return [(str(row[0]), str(row[1])), (str(row[2]), str(row[3]))]


def get_gk_colours(
    conn: pyodbc.Connection, match_id: str,
) -> Optional[tuple[str, str]]:
    """(team_a_gk_colour, team_b_gk_colour) for goalkeeper colour-matching, or
    None unless BOTH are set. Independent of team_a_name/team_b_name — GK
    classification only needs the two reference colours, not display names."""
    cur = conn.cursor()
    cur.execute(
        "SELECT team_a_gk_colour, team_b_gk_colour FROM matches WHERE match_id = ?",
        match_id,
    )
    row = cur.fetchone()
    if row is None or not all(row):
        return None
    return (str(row[0]), str(row[1]))


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


def get_match_ruleset(conn: pyodbc.Connection, match_id: str) -> str:
    """The ruleset ('futsal' | 'classic') this match was created with.
    Defaults to 'classic' if the match row is somehow missing (should not
    happen — callers ensure_match() before reaching this point)."""
    cur = conn.cursor()
    cur.execute("SELECT ruleset FROM matches WHERE match_id = ?", match_id)
    row = cur.fetchone()
    return str(row[0]) if row and row[0] else "classic"


def claim_next_minute(conn: pyodbc.Connection, match_id: str, half: int) -> int:
    """Atomically claim the next sequential minute number for this match+half (1-based).

    Uses UPDATE...OUTPUT so two concurrent API uploads serialize on the row
    X-lock and always receive distinct values — safe for concurrent clip uploads.
    """
    if half not in (1, 2):
        raise ValueError(f"half must be 1 or 2, got {half!r}")
    col = "next_clip_seq_h1" if half == 1 else "next_clip_seq_h2"
    cur = conn.cursor()
    cur.execute(
        f"UPDATE matches SET {col} = {col} + 1 OUTPUT INSERTED.{col} WHERE match_id = ?",
        match_id,
    )
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"match_id {match_id!r} not found in matches table")
    conn.commit()
    return int(row[0])


def minute_exists(conn: pyodbc.Connection, match_id: str, half: int, minute: int) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM minute_stats WHERE match_id = ? AND half = ? AND minute = ?",
        match_id, half, minute,
    )
    return cur.fetchone() is not None


def post_processing_exists(conn: pyodbc.Connection, match_id: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM post_processing WHERE match_id = ?",
        match_id,
    )
    return cur.fetchone() is not None


# ---------------------------------------------------------------------------
# Cumulative read  (the "minute N returns totals 1..N" requirement)
# ---------------------------------------------------------------------------

_SUM_COLS = (
    "frames_team_a", "frames_team_b", "frames_loose", "frames_oof",
    "passes_completed_team_a", "passes_completed_team_b",
    "interceptions_team_a", "interceptions_team_b",
    "ball_lost_team_a", "ball_lost_team_b",
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
                "  clip_duration_seconds, "
                "  frames_team_a, frames_team_b, frames_loose, frames_oof, "
                "  passes_completed_team_a, passes_completed_team_b, "
                "  interceptions_team_a, interceptions_team_b, "
                "  ball_lost_team_a, ball_lost_team_b, revision, clip_blob_path) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
                row.match_id, row.half, row.minute,
                row.clip_duration_seconds,
                row.frames_team_a, row.frames_team_b, row.frames_loose, row.frames_oof,
                row.passes_completed_team_a, row.passes_completed_team_b,
                row.interceptions_team_a, row.interceptions_team_b,
                row.ball_lost_team_a, row.ball_lost_team_b,
                row.clip_blob_path,
            )
        else:
            revision = int(existing[0])
            cur.execute(
                "UPDATE minute_stats SET "
                "  clip_duration_seconds = ?, "
                "  frames_team_a = ?, frames_team_b = ?, frames_loose = ?, frames_oof = ?, "
                "  passes_completed_team_a = ?, passes_completed_team_b = ?, "
                "  interceptions_team_a = ?, interceptions_team_b = ?, "
                "  ball_lost_team_a = ?, ball_lost_team_b = ?, "
                "  clip_blob_path = ?, processed_at = SYSUTCDATETIME() "
                "WHERE match_id = ? AND half = ? AND minute = ?",
                row.clip_duration_seconds,
                row.frames_team_a, row.frames_team_b, row.frames_loose, row.frames_oof,
                row.passes_completed_team_a, row.passes_completed_team_b,
                row.interceptions_team_a, row.interceptions_team_b,
                row.ball_lost_team_a, row.ball_lost_team_b,
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


def write_post_processing_result(
    conn: pyodbc.Connection,
    row: MinuteRow,
    team_a_name: Optional[str] = None,
    team_b_name: Optional[str] = None,
    team_a_colour: Optional[str] = None,
    team_b_colour: Optional[str] = None,
    team_a_gk_colour: Optional[str] = None,
    team_b_gk_colour: Optional[str] = None,
    video_blob_path: Optional[str] = None,
) -> dict:
    """Upsert the whole-match aggregate row into post_processing.
    team_a = CV cluster id 0, team_b = cluster id 1. The returned dict is the
    HTTP response body of POST /api/post-processing."""
    cur = conn.cursor()
    blob_path = video_blob_path or row.clip_blob_path
    cur.execute(
        "SELECT 1 FROM post_processing WHERE match_id = ?",
        row.match_id,
    )
    exists = cur.fetchone() is not None
    if exists:
        cur.execute(
            "UPDATE post_processing SET "
            "  team_a_name = ?, team_b_name = ?, team_a_colour = ?, team_b_colour = ?, "
            "  team_a_gk_colour = ?, team_b_gk_colour = ?, "
            "  frames_team_a = ?, frames_team_b = ?, frames_loose = ?, frames_oof = ?, "
            "  passes_completed_team_a = ?, passes_completed_team_b = ?, "
            "  interceptions_team_a = ?, interceptions_team_b = ?, "
            "  ball_lost_team_a = ?, ball_lost_team_b = ?, video_blob_path = ?, "
            "  processed_at = SYSUTCDATETIME() "
            "WHERE match_id = ?",
            team_a_name, team_b_name, team_a_colour, team_b_colour,
            team_a_gk_colour, team_b_gk_colour,
            row.frames_team_a, row.frames_team_b, row.frames_loose, row.frames_oof,
            row.passes_completed_team_a, row.passes_completed_team_b,
            row.interceptions_team_a, row.interceptions_team_b,
            row.ball_lost_team_a, row.ball_lost_team_b, blob_path,
            row.match_id,
        )
    else:
        cur.execute(
            "INSERT INTO post_processing (match_id, team_a_name, team_b_name, team_a_colour, team_b_colour, "
            "  team_a_gk_colour, team_b_gk_colour, "
            "  frames_team_a, frames_team_b, frames_loose, frames_oof, "
            "  passes_completed_team_a, passes_completed_team_b, interceptions_team_a, interceptions_team_b, "
            "  ball_lost_team_a, ball_lost_team_b, video_blob_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            row.match_id, team_a_name, team_b_name, team_a_colour, team_b_colour,
            team_a_gk_colour, team_b_gk_colour,
            row.frames_team_a, row.frames_team_b, row.frames_loose, row.frames_oof,
            row.passes_completed_team_a, row.passes_completed_team_b,
            row.interceptions_team_a, row.interceptions_team_b,
            row.ball_lost_team_a, row.ball_lost_team_b, blob_path,
        )
    conn.commit()
    return {
        "match_id": row.match_id,
        "team_a_name": team_a_name,
        "team_b_name": team_b_name,
        "team_a_colour": team_a_colour,
        "team_b_colour": team_b_colour,
        "team_a_gk_colour": team_a_gk_colour,
        "team_b_gk_colour": team_b_gk_colour,
        "frames_team_a": row.frames_team_a,
        "frames_team_b": row.frames_team_b,
        "frames_loose": row.frames_loose,
        "frames_oof": row.frames_oof,
        "passes_completed_team_a": row.passes_completed_team_a,
        "passes_completed_team_b": row.passes_completed_team_b,
        "interceptions_team_a": row.interceptions_team_a,
        "interceptions_team_b": row.interceptions_team_b,
        "ball_lost_team_a": row.ball_lost_team_a,
        "ball_lost_team_b": row.ball_lost_team_b,
        "video_blob_path": blob_path,
    }


def _apply_prior_correction(cur, match_id: str, c: PriorCorrection) -> None:
    """Read-modify-write with clamping (single worker ⇒ no contention).
    flip_to: move frames from the other team's count to c.team_id.
    drop:    move frames from c.team_id's count to frames_oof."""
    cur.execute(
        "SELECT frames_team_a, frames_team_b, frames_oof FROM minute_stats "
        "WHERE match_id = ? AND half = ? AND minute = ?",
        match_id, c.half, c.minute,
    )
    prior = cur.fetchone()
    if prior is None:
        return  # prior minute row missing (e.g. logged gap) — nothing to correct
    # t0/t1 locals: t0 = team_a count, t1 = team_b count (c.team_id 0 → team_a)
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
        "UPDATE minute_stats SET frames_team_a = ?, frames_team_b = ?, frames_oof = ?, "
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
