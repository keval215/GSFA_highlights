"""
service/stats.py — pure data shapes + per-minute counting logic.

Deliberately dependency-free (stdlib only) so the correctness-critical
bucketing/correction logic is unit-testable without torch/boxmot/pyodbc.
The label and event constants are string-identical to
modules/possession/labels.py; service/session.py asserts that at import.

Team convention: CV cluster id 0 → team_a, cluster id 1 → team_b. The
outbound advance-stats callback body keeps its historical keys
(frames_a / frames_b / passes_completed_a / …); build_payload maps the
internal team_a/team_b names onto those wire keys.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

# Possession labels (must match modules.possession.labels.POSSESS_*)
LBL_TEAM_A = "team_a"
LBL_TEAM_B = "team_b"
LBL_LOOSE  = "loose"
LBL_OOF    = "oof"

# Pass-event kinds (must match modules.possession.labels.EVT_*)
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
    frames_team_a: int = 0
    frames_team_b: int = 0
    frames_loose:  int = 0
    frames_oof:    int = 0
    passes_completed_team_a: int = 0
    passes_completed_team_b: int = 0
    interceptions_team_a:    int = 0
    interceptions_team_b:    int = 0
    ball_lost_team_a:        int = 0
    ball_lost_team_b:        int = 0
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
    frames_team_a: int = 0
    frames_team_b: int = 0
    frames_loose:  int = 0
    frames_oof:    int = 0
    passes_completed_team_a: int = 0
    passes_completed_team_b: int = 0
    interceptions_team_a:    int = 0
    interceptions_team_b:    int = 0
    ball_lost_team_a:        int = 0
    ball_lost_team_b:        int = 0

    def add_label(self, label: str) -> None:
        if label == LBL_TEAM_A:
            self.frames_team_a += 1
        elif label == LBL_TEAM_B:
            self.frames_team_b += 1
        elif label == LBL_LOOSE:
            self.frames_loose += 1
        else:
            self.frames_oof += 1

    def apply_adjustment(self, kind: str, team_id: int, n: int) -> None:
        """Same semantics as PossessionStats.apply_adjustments, applied to
        this minute's counters only (n = the current-minute portion).
        team_id 0 → team_a, 1 → team_b."""
        if n <= 0:
            return
        if kind == "flip_to":
            if team_id == 0:
                move = min(n, self.frames_team_b)
                self.frames_team_b -= move
                self.frames_team_a += move
            else:
                move = min(n, self.frames_team_a)
                self.frames_team_a -= move
                self.frames_team_b += move
        elif kind == "drop":
            if team_id == 0:
                move = min(n, self.frames_team_a)
                self.frames_team_a -= move
            else:
                move = min(n, self.frames_team_b)
                self.frames_team_b -= move
            self.frames_oof += move

    def count_event(self, kind: str, from_team: Optional[int]) -> None:
        """kind is the internal FSM kind (completed/interception/ball_lost).
        from_team 0 → team_a, 1 → team_b."""
        if from_team not in (0, 1):
            return
        if kind == EVT_COMPLETED:
            if from_team == 0:
                self.passes_completed_team_a += 1
            else:
                self.passes_completed_team_b += 1
        elif kind == EVT_INTERCEPTION:
            if from_team == 0:
                self.interceptions_team_a += 1
            else:
                self.interceptions_team_b += 1
        elif kind == EVT_BALL_LOST:
            if from_team == 0:
                self.ball_lost_team_a += 1
            else:
                self.ball_lost_team_b += 1

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
            frames_team_a=self.frames_team_a, frames_team_b=self.frames_team_b,
            frames_loose=self.frames_loose, frames_oof=self.frames_oof,
            passes_completed_team_a=self.passes_completed_team_a,
            passes_completed_team_b=self.passes_completed_team_b,
            interceptions_team_a=self.interceptions_team_a,
            interceptions_team_b=self.interceptions_team_b,
            ball_lost_team_a=self.ball_lost_team_a,
            ball_lost_team_b=self.ball_lost_team_b,
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


def _flip_team_id(team_id: Optional[int]) -> Optional[int]:
    return 1 - team_id if team_id in (0, 1) else team_id


_SAME_AS_ROW = object()   # sentinel: correction_team_a_cluster_id defaults to team_a_cluster_id


def orient_for_team_a(
    result: ClipResult,
    team_a_cluster_id: Optional[int],
    correction_team_a_cluster_id: object = _SAME_AS_ROW,
) -> ClipResult:
    """Reorient one clip's raw minute_row / events / correction so the
    `team_a_*` fields (and from_team/to_team/correction.team_id == 0) always
    mean the CV cluster whose resolved jersey colour matches team_a_colour
    (service/session.py's `_resolve_team_names`, exposed as
    `MatchSession.team_a_cluster_id`).

    CV cluster id 0 IS team_a by convention, so each part is a no-op unless
    its cluster id == 1 — which only happens after a re-fit whose KMeans
    happened to land the clusters in the other order. A cluster id of None
    (colour resolution hasn't run or failed) is also treated as no-op — fall
    back to raw cluster order, a documented v1 limitation (see
    MatchSession._resolve_team_names).

    `team_a_cluster_id` orients minute_row + events (THIS clip's own row).
    `result.correction`, when present, targets a DIFFERENT row — the
    previous minute, already written under whatever orientation was active
    at THAT time — so it must be reoriented with
    `correction_team_a_cluster_id` (MatchSession.last_written_team_a_cluster_id
    at the moment the correction was built), not `team_a_cluster_id`. The two
    can differ, e.g. clip 1 commits before colour resolution succeeds
    (team_a_cluster_id is still None there) and clip 2's combined refit
    resolves it, possibly to the other cluster — reorienting a clip-2
    boundary correction against clip 1's (unreoriented) row with clip 2's
    cluster id would flip the wrong team_a/team_b columns on that row.
    Omitting this argument defaults it to `team_a_cluster_id`, for callers
    that know orientation hasn't changed since the target row was written.

    Must be applied once, at write time (here, from clip_processor before
    the row reaches db.write_clip_result), NOT at cumulative_read/
    build_payload time — a mid-match swap must not retroactively reorder
    already-summed prior minutes."""
    if correction_team_a_cluster_id is _SAME_AS_ROW:
        correction_team_a_cluster_id = team_a_cluster_id

    flip_row  = team_a_cluster_id not in (None, 0)
    flip_corr = correction_team_a_cluster_id not in (None, 0)
    if not flip_row and not flip_corr:
        return result

    oriented_row    = result.minute_row
    oriented_events = result.events
    if flip_row:
        row = result.minute_row
        oriented_row = dataclasses.replace(
            row,
            frames_team_a=row.frames_team_b,
            frames_team_b=row.frames_team_a,
            passes_completed_team_a=row.passes_completed_team_b,
            passes_completed_team_b=row.passes_completed_team_a,
            interceptions_team_a=row.interceptions_team_b,
            interceptions_team_b=row.interceptions_team_a,
            ball_lost_team_a=row.ball_lost_team_b,
            ball_lost_team_b=row.ball_lost_team_a,
        )
        oriented_events = [
            dataclasses.replace(e, from_team=_flip_team_id(e.from_team), to_team=_flip_team_id(e.to_team))
            for e in result.events
        ]

    oriented_correction = result.correction
    if flip_corr and result.correction is not None:
        oriented_correction = dataclasses.replace(
            result.correction, team_id=_flip_team_id(result.correction.team_id)
        )

    return ClipResult(minute_row=oriented_row, correction=oriented_correction, events=oriented_events)


def build_payload(
    match_id: str, half: int, minute: int, revision: int, sums: dict[str, int],
) -> dict:
    """advance-stats body for POST /v1/pvt/tournament-duelz/{id}/advance-stats.

    Raw cumulative counters as a flat, all-integer body — the server does a full
    overwrite of the duel's advance_stats subdocument and derives percentages /
    accuracy itself.

    The internal counters are keyed team_a / team_b (cluster id 0 → team_a,
    1 → team_b, or whatever cluster orient_for_team_a() aligned to team_a at
    write time — see that function); the wire body keeps its historical
    `_a` / `_b` suffixes so the tournament-duelz consumer is unchanged.

    match_id / half / minute / revision are not part of the body — they live
    on the outbox columns and drive ordering and the per-duel URL."""
    return {
        "frames_a":           sums["frames_team_a"],
        "frames_b":           sums["frames_team_b"],
        "frames_loose":       sums["frames_loose"],
        "frames_oof":         sums["frames_oof"],
        "passes_completed_a": sums["passes_completed_team_a"],
        "passes_completed_b": sums["passes_completed_team_b"],
        "interceptions_a":    sums["interceptions_team_a"],
        "interceptions_b":    sums["interceptions_team_b"],
        "ball_lost_a":        sums["ball_lost_team_a"],
        "ball_lost_b":        sums["ball_lost_team_b"],
    }
