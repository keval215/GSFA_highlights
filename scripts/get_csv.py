"""
GSFA Highlights — Match Statistics CSV Exporter
================================================
Reads minute_stats from Azure SQL (gsfa_stats) and writes one CSV per match.

CSV row types
─────────────
  minute       one row per minute_stats row — raw frames + pass counts
  block_5min   frames/pass counts summed into ~300 s (±20 s) blocks
  match_total  single summary row for the whole match

Nothing is calculated here except the block accumulation.
Possession %, Pass Accuracy %, Pass Density → derive in your analytics tool:
  Possession %  = frames_team0 / (frames_team0 + frames_team1 + frames_loose)
  Pass accuracy = passes_completed_t0 / (passes_completed_t0 + interceptions_t0 + ball_lost_t0)
  Pass density  = passes_completed_t0 / actual_duration_sec * 60

NOTE: interceptions_t0 = passes BY t0 that were intercepted (by t1), per schema.

Usage
─────
  Set env vars or edit CONFIG, then:
      pip install pyodbc

  Export ALL matches in the DB:
      python export_match_stats.py

  Export specific match IDs (any number):
      python export_match_stats.py match_001 match_002 match_003

  Export via env var (comma-separated, no spaces):
      GSFA_MATCH_IDS=match_001,match_002,match_003 python export_match_stats.py

  Priority: CLI args > GSFA_MATCH_IDS env var > all matches in DB
"""

import os
import sys
import csv
from pathlib import Path

try:
    import pyodbc
except ImportError:
    raise SystemExit("pyodbc not found — run: pip install pyodbc")


# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "sql_conn_str": os.getenv("SQL_CONN_STR", "").strip(),
    "server":    os.getenv("GSFA_SQL_SERVER",   "your-server.database.windows.net"),
    "database":  os.getenv("GSFA_SQL_DB",        "gsfa_stats"),
    "username":  os.getenv("GSFA_SQL_USER",      "your-user"),
    "password":  os.getenv("GSFA_SQL_PASSWORD",  "your-password"),
    "driver":    "{ODBC Driver 18 for SQL Server}",
    "output_dir": Path(os.getenv("GSFA_EXPORT_DIR", "/opt/csv_export")),
}

# 5-min block window: accumulate until duration is in [BLOCK_MIN, BLOCK_MAX]
BLOCK_MIN = 280   # seconds  (300 - 20)
BLOCK_MAX = 320   # seconds  (300 + 20)


# ── CSV columns ───────────────────────────────────────────────────────────────
CSV_FIELDS = [
    # identity / grouping
    "row_type",
    "match_id",
    "team0_name",
    "team1_name",
    "half",              # 1 | 2 | "" for match_total
    "period_label",      # H1_M01 / H1_B01 / MATCH_TOTAL
    # timing (cumulative within half; 0..total for match_total)
    "period_start_sec",
    "period_end_sec",
    "actual_duration_sec",
    "clip_count",
    # raw frame counters — no calculation
    "frames_team0",
    "frames_team1",
    "frames_loose",      # ball in foot-zone, multiple teams present
    "frames_oof",        # ball out of frame / out of play
    # raw pass / turnover counters — no calculation
    "passes_completed_t0",
    "passes_completed_t1",
    "interceptions_t0",  # passes BY t0 that were intercepted
    "interceptions_t1",  # passes BY t1 that were intercepted
    "ball_lost_t0",
    "ball_lost_t1",
]

# Columns that are summed when building blocks / totals
STAT_COLS = [
    "frames_team0", "frames_team1", "frames_loose", "frames_oof",
    "passes_completed_t0", "passes_completed_t1",
    "interceptions_t0",   "interceptions_t1",
    "ball_lost_t0",       "ball_lost_t1",
]


# ── Helpers ──────────────────────────────────────────────────────────────────
def get_conn():
    c = CONFIG
    if c["sql_conn_str"]:
        return pyodbc.connect(c["sql_conn_str"])
    cs = (
        f"DRIVER={c['driver']};"
        f"SERVER={c['server']};"
        f"DATABASE={c['database']};"
        f"UID={c['username']};"
        f"PWD={c['password']};"
        "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
    )
    return pyodbc.connect(cs)


def zero_stats():
    return {col: 0 for col in STAT_COLS}


def add_stats(acc, src):
    """Add stat columns from src dict into acc dict."""
    for col in STAT_COLS:
        acc[col] += int(src.get(col) or 0)


def make_row(row_type, match_id, team0, team1, half,
             label, start_sec, end_sec, actual_dur, clip_count, stats):
    return {
        "row_type":           row_type,
        "match_id":           match_id,
        "team0_name":         team0,
        "team1_name":         team1,
        "half":               half,
        "period_label":       label,
        "period_start_sec":   round(float(start_sec),   2),
        "period_end_sec":     round(float(end_sec),     2),
        "actual_duration_sec": round(float(actual_dur), 2),
        "clip_count":         clip_count,
        **stats,
    }


# ── 5-min block builder ───────────────────────────────────────────────────────
def build_5min_blocks(minute_rows, match_id, team0, team1, half):
    """
    Greedy accumulation across minute rows (sorted by minute).

    Rules
    ─────
    1. If adding the next clip would push the block past BLOCK_MAX
       AND the block is already >= BLOCK_MIN  →  close BEFORE adding.
    2. After adding, if block_duration >= BLOCK_MIN  →  close naturally.
    3. Edge case: a single very long clip (>320 s) forms its own block
       even though it exceeds BLOCK_MAX — we can't split a single clip.
    4. Remaining clips at end of half flush as a partial block (no minimum).

    This guarantees every block except the last is 280–320 s, and the
    last block is whatever's left (label shows actual_duration_sec).
    """
    blocks  = []
    b_stats = zero_stats()
    b_start = 0.0    # cumulative-within-half seconds at block start
    b_dur   = 0.0    # duration accumulated into current block
    b_clips = 0
    b_num   = 0

    def emit():
        nonlocal b_stats, b_start, b_dur, b_clips, b_num
        if b_clips == 0:
            return
        b_num += 1
        blocks.append(make_row(
            "block_5min", match_id, team0, team1, half,
            f"H{half}_B{b_num:02d}",
            b_start, b_start + b_dur, b_dur, b_clips, dict(b_stats),
        ))
        b_start += b_dur
        b_stats  = zero_stats()
        b_dur    = 0.0
        b_clips  = 0

    for m in minute_rows:
        clip_dur  = float(m.get("clip_duration_seconds") or 0)
        projected = b_dur + clip_dur

        # Rule 1: close before adding if block is valid and clip would blow cap
        if b_dur >= BLOCK_MIN and projected > BLOCK_MAX:
            emit()

        # Add this minute into the current block
        add_stats(b_stats, m)
        b_dur   += clip_dur
        b_clips += 1

        # Rule 2: close after adding once we've hit the natural target
        if b_dur >= BLOCK_MIN:
            emit()

    # Rule 4: flush remaining partial block
    emit()

    return blocks


# ── Per-match processing ─────────────────────────────────────────────────────
def process_match(conn, match_id):
    cur = conn.cursor()

    # Match metadata
    cur.execute(
        "SELECT team0_name, team1_name FROM matches WHERE match_id = ?",
        match_id,
    )
    m = cur.fetchone()
    if m is None:
        raise ValueError(f"match_id '{match_id}' not found in matches table")
    team0 = m.team0_name or "Team 0"
    team1 = m.team1_name or "Team 1"

    # All minute stats for this match
    cur.execute(
        """
        SELECT half, minute,
               clip_duration_seconds,
               frames_team0, frames_team1, frames_loose, frames_oof,
               passes_completed_t0, passes_completed_t1,
               interceptions_t0,    interceptions_t1,
               ball_lost_t0,        ball_lost_t1
        FROM   minute_stats
        WHERE  match_id = ?
        ORDER  BY half, minute
        """,
        match_id,
    )
    cols    = [d[0] for d in cur.description]
    db_rows = [dict(zip(cols, row)) for row in cur.fetchall()]

    if not db_rows:
        print(f"    (no minute_stats rows for {match_id}, skipping)")
        return []

    # Group by half preserving insertion order
    halves = {}
    for r in db_rows:
        halves.setdefault(r["half"], []).append(r)

    out_rows    = []
    total_stats = zero_stats()
    total_dur   = 0.0
    total_clips = 0

    for half in sorted(halves):
        h_rows    = halves[half]
        cur_sec   = 0.0          # cumulative seconds within this half
        m_rows    = []           # minute-level rows for this half

        for m in h_rows:
            dur   = float(m.get("clip_duration_seconds") or 0)
            stats = {col: int(m.get(col) or 0) for col in STAT_COLS}

            m_rows.append(make_row(
                "minute", match_id, team0, team1, half,
                f"H{half}_M{m['minute']:02d}",
                cur_sec, cur_sec + dur, dur, 1, stats,
            ))
            cur_sec     += dur
            add_stats(total_stats, m)
            total_dur   += dur
            total_clips += 1

        b_rows = build_5min_blocks(h_rows, match_id, team0, team1, half)

        # Ordering: all raw minutes for H1, then H1 blocks, then H2, then H2 blocks
        out_rows.extend(m_rows)
        out_rows.extend(b_rows)

    # Match total summary
    out_rows.append(make_row(
        "match_total", match_id, team0, team1, "",
        "MATCH_TOTAL", 0, total_dur, total_dur, total_clips, total_stats,
    ))

    return out_rows


# ── Filename sanitiser ────────────────────────────────────────────────────────
def safe_filename(match_id):
    return "".join(
        c if (c.isalnum() or c in "-_") else "_"
        for c in match_id
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    out_dir = CONFIG["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = get_conn()
    cur  = conn.cursor()

    # Priority 1: match IDs passed as CLI arguments
    if len(sys.argv) > 1:
        match_ids = sys.argv[1:]
        source = "CLI args"

    # Priority 2: comma-separated env var
    elif os.getenv("GSFA_MATCH_IDS", "").strip():
        match_ids = [m.strip() for m in os.getenv("GSFA_MATCH_IDS").split(",") if m.strip()]
        source = "GSFA_MATCH_IDS env var"

    # Priority 3: fetch every match from the DB
    else:
        cur.execute("SELECT match_id FROM matches ORDER BY created_at")
        match_ids = [r[0] for r in cur.fetchall()]
        source = "all matches in DB"

    print(f"Source  : {source}")
    print(f"Matches : {len(match_ids)}")
    print(f"Output  : {out_dir}/")
    print()

    ok = err = 0
    for match_id in match_ids:
        try:
            rows     = process_match(conn, match_id)
            if not rows:
                continue

            out_path = out_dir / f"{safe_filename(match_id)}.csv"
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)

            n_min   = sum(1 for r in rows if r["row_type"] == "minute")
            n_blk   = sum(1 for r in rows if r["row_type"] == "block_5min")
            ok += 1
            print(f"  ✓  {match_id}")
            print(f"       {n_min} minute rows  |  {n_blk} × 5-min blocks  →  {out_path.name}")

        except Exception as exc:
            err += 1
            print(f"  ✗  {match_id}  →  {exc}")

    conn.close()
    print()
    print(f"Done.  {ok} exported, {err} failed.")


if __name__ == "__main__":
    main()