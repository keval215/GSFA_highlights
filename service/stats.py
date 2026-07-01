"""
service/stats.py — pure data shapes + per-minute counting logic.

Deliberately dependency-free (stdlib only) so the correctness-critical
bucketing/correction logic is unit-testable without torch/boxmot/pyodbc.
The label and event constants are string-identical to
video_analysis/possession.py; service/session.py asserts that at import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Possession labels (must match video_analysis.possession.POSSESS_*)
LBL_TEAM0 = "team0"
LBL_TEAM1 = "team1"
LBL_LOOSE = "loose"
LBL_OOF   = "oof"

# Pass-event kinds (must match video_analysis.possession.EVT_*)
EVT_COMPLETED    = "completed"
EVT_INTERCEPTION = "interception"
EVT_BALL_LOST    = "ball_lost"

# events.kind values stored in SQL
KIND_MAP = {
    EVT_COMPLETED:    "pass",
    EVT_INTERCEPTION: "interception",
    EVT_BALL_LOST:    "ball_lost",
}


# ---------------------------------------------------------------------------
# Row shapes written by db.py
# ---------------------------------------------------------------------------

@dataclass
class MinuteRow:
    match_id: str
    half:     int
    minute:   int
    clip_duration_seconds: Optional[float] = None
    frames_team0: int = 0
    frames_team1: int = 0
    frames_loose: int = 0
    frames_oof:   int = 0
    passes_completed_t0: int = 0
    passes_completed_t1: int = 0
    interceptions_t0:    int = 0
    interceptions_t1:    int = 0
    ball_lost_t0:        int = 0
    ball_lost_t1:        int = 0
    clip_blob_path: Optional[str] = None


@dataclass
class EventRow:
    half:       int
    minute:     int
    frame_idx:  int
    kind:       str                      # pass | interception | ball_lost
    from_team:  Optional[int] = None
    to_team:    Optional[int] = None
    from_track: Optional[int] = None
    to_track:   Optional[int] = None
    travel_frames: Optional[int] = None


@dataclass
class PriorCorrection:
    """Adjustment to the immediately-prior minute row, produced when a pass
    spanning the clip boundary resolves as interception (flip_to) or
    ball_lost (drop)."""
    half:    int
    minute:  int
    kind:    str    # "flip_to" | "drop"
    team_id: int    # flip_to: team that should own the frames; drop: team losing them
    frames:  int


# ---------------------------------------------------------------------------
# Per-minute raw counters
# ---------------------------------------------------------------------------

@dataclass
class MinuteCounters:
    frames_team0: int = 0
    frames_team1: int = 0
    frames_loose: int = 0
    frames_oof:   int = 0
    passes_completed_t0: int = 0
    passes_completed_t1: int = 0
    interceptions_t0:    int = 0
    interceptions_t1:    int = 0
    ball_lost_t0:        int = 0
    ball_lost_t1:        int = 0

    def add_label(self, label: str) -> None:
        if label == LBL_TEAM0:
            self.frames_team0 += 1
        elif label == LBL_TEAM1:
            self.frames_team1 += 1
        elif label == LBL_LOOSE:
            self.frames_loose += 1
        else:
            self.frames_oof += 1

    def apply_adjustment(self, kind: str, team_id: int, n: int) -> None:
        """Same semantics as PossessionStats.apply_adjustments, applied to
        this minute's counters only (n = the current-minute portion)."""
        if n <= 0:
            return
        if kind == "flip_to":
            if team_id == 0:
                move = min(n, self.frames_team1)
                self.frames_team1 -= move
                self.frames_team0 += move
            else:
                move = min(n, self.frames_team0)
                self.frames_team0 -= move
                self.frames_team1 += move
        elif kind == "drop":
            if team_id == 0:
                move = min(n, self.frames_team0)
                self.frames_team0 -= move
            else:
                move = min(n, self.frames_team1)
                self.frames_team1 -= move
            self.frames_oof += move

    def count_event(self, kind: str, from_team: Optional[int]) -> None:
        """kind is the internal FSM kind (completed/interception/ball_lost)."""
        if from_team not in (0, 1):
            return
        if kind == EVT_COMPLETED:
            if from_team == 0:
                self.passes_completed_t0 += 1
            else:
                self.passes_completed_t1 += 1
        elif kind == EVT_INTERCEPTION:
            if from_team == 0:
                self.interceptions_t0 += 1
            else:
                self.interceptions_t1 += 1
        elif kind == EVT_BALL_LOST:
            if from_team == 0:
                self.ball_lost_t0 += 1
            else:
                self.ball_lost_t1 += 1

    def to_minute_row(
        self,
        match_id: str,
        half: int,
        minute: int,
        clip_blob_path: Optional[str],
        clip_duration_seconds: Optional[float] = None,
    ) -> MinuteRow:
        return MinuteRow(
            match_id=match_id, half=half, minute=minute,
            clip_duration_seconds=clip_duration_seconds,
            frames_team0=self.frames_team0, frames_team1=self.frames_team1,
            frames_loose=self.frames_loose, frames_oof=self.frames_oof,
            passes_completed_t0=self.passes_completed_t0,
            passes_completed_t1=self.passes_completed_t1,
            interceptions_t0=self.interceptions_t0,
            interceptions_t1=self.interceptions_t1,
            ball_lost_t0=self.ball_lost_t0, ball_lost_t1=self.ball_lost_t1,
            clip_blob_path=clip_blob_path,
        )


@dataclass
class ClipResult:
    minute_row: MinuteRow
    correction: Optional[PriorCorrection]
    events:     list[EventRow]


# ---------------------------------------------------------------------------
# Pure helpers shared by worker / session
# ---------------------------------------------------------------------------

def split_adjustment(n: int, carryover: int) -> tuple[int, int, int]:
    """Split an adjustment of n provisional frames into
    (prior_minute_frames, current_minute_frames, remaining_carryover)."""
    prior = min(n, carryover)
    return prior, n - prior, carryover - prior


def is_expected(last_half: int, last_minute: int, half: int, minute: int) -> bool:
    """Ordering guard: the next clip is either the next minute of the same
    half or minute 1 of the next half. A fresh match (last_minute == 0)
    expects (1, 1)."""
    if last_minute == 0:
        return (half, minute) == (1, 1)
    return (half, minute) in ((last_half, last_minute + 1), (last_half + 1, 1))


def build_payload(
    match_id: str, half: int, minute: int, revision: int, sums: dict[str, int],
    team_id_to_name: Optional[dict[int, str]] = None,
) -> dict:
    """advance-stats body for POST /v1/pvt/tournament-duelz/{id}/advance-stats.

    Raw cumulative counters as a flat, all-integer body — the server does a full
    overwrite of the duel's advance_stats subdocument and derives percentages /
    accuracy itself. Team mapping is positional: team id 0 → a, team id 1 → b
    (so no jersey-colour name resolution is needed for the body).

    match_id / half / minute / revision are not part of the body — they live on
    the outbox columns and drive ordering and the per-duel URL. team_id_to_name
    is accepted for call-site compatibility but is unused here."""
    return {
        "frames_a":           sums["frames_team0"],
        "frames_b":           sums["frames_team1"],
        "frames_loose":       sums["frames_loose"],
        "frames_oof":         sums["frames_oof"],
        "passes_completed_a": sums["passes_completed_t0"],
        "passes_completed_b": sums["passes_completed_t1"],
        "interceptions_a":    sums["interceptions_t0"],
        "interceptions_b":    sums["interceptions_t1"],
        "ball_lost_a":        sums["ball_lost_t0"],
        "ball_lost_b":        sums["ball_lost_t1"],
    }
