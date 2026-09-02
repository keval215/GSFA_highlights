"""Unit tests for the dependency-free stats layer (service/stats.py):
minute bucketing, retroactive-correction splitting and the ordering guard.
(Payload shaping lives in build_payload and is exercised end-to-end; the
raw cumulative counters it emits are asserted there, not here.)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from service.stats import (
    EVT_BALL_LOST,
    EVT_COMPLETED,
    EVT_INTERCEPTION,
    LBL_LOOSE,
    LBL_OOF,
    LBL_TEAM_A,
    LBL_TEAM_B,
    ClipResult,
    EventRow,
    MinuteCounters,
    MinuteRow,
    PriorCorrection,
    build_payload,
    is_expected,
    orient_for_team_a,
    split_adjustment,
)


# ---------------------------------------------------------------------------
# Label bucketing
# ---------------------------------------------------------------------------

def test_add_label_buckets():
    c = MinuteCounters()
    for lbl in [LBL_TEAM_A] * 3 + [LBL_TEAM_B] * 2 + [LBL_LOOSE] + [LBL_OOF] * 4:
        c.add_label(lbl)
    assert (c.frames_team_a, c.frames_team_b, c.frames_loose, c.frames_oof) == (3, 2, 1, 4)


def test_unknown_label_counts_as_oof():
    c = MinuteCounters()
    c.add_label("something_else")
    assert c.frames_oof == 1


# ---------------------------------------------------------------------------
# Adjustments (current-minute portion)
# ---------------------------------------------------------------------------

def test_flip_to_moves_frames_between_teams():
    c = MinuteCounters(frames_team_a=10, frames_team_b=5)
    c.apply_adjustment("flip_to", 1, 4)     # 4 frames belong to team_b, not team_a
    assert (c.frames_team_a, c.frames_team_b) == (6, 9)


def test_flip_to_clamps_at_available():
    c = MinuteCounters(frames_team_a=2, frames_team_b=0)
    c.apply_adjustment("flip_to", 1, 10)
    assert (c.frames_team_a, c.frames_team_b) == (0, 2)


def test_drop_moves_frames_to_oof():
    c = MinuteCounters(frames_team_a=10, frames_oof=1)
    c.apply_adjustment("drop", 0, 3)
    assert (c.frames_team_a, c.frames_oof) == (7, 4)


def test_drop_clamps_at_available():
    c = MinuteCounters(frames_team_b=2, frames_oof=0)
    c.apply_adjustment("drop", 1, 10)
    assert (c.frames_team_b, c.frames_oof) == (0, 2)


def test_zero_or_negative_adjustment_is_noop():
    c = MinuteCounters(frames_team_a=5)
    c.apply_adjustment("flip_to", 1, 0)
    c.apply_adjustment("drop", 0, -3)
    assert c.frames_team_a == 5


# ---------------------------------------------------------------------------
# Event counting
# ---------------------------------------------------------------------------

def test_count_events_per_team():
    c = MinuteCounters()
    c.count_event(EVT_COMPLETED, 0)
    c.count_event(EVT_COMPLETED, 0)
    c.count_event(EVT_COMPLETED, 1)
    c.count_event(EVT_INTERCEPTION, 0)
    c.count_event(EVT_BALL_LOST, 1)
    c.count_event(EVT_COMPLETED, None)   # no attributable team — ignored
    assert c.passes_completed_team_a == 2
    assert c.passes_completed_team_b == 1
    assert c.interceptions_team_a == 1
    assert c.ball_lost_team_b == 1


# ---------------------------------------------------------------------------
# Boundary split (Conflict 3)
# ---------------------------------------------------------------------------

def test_split_fully_within_current_minute():
    assert split_adjustment(5, 0) == (0, 5, 0)


def test_split_spanning_boundary():
    # 8 travel frames pending at boundary; adjustment of 10 ⇒ 8 prior, 2 current.
    assert split_adjustment(10, 8) == (8, 2, 0)


def test_split_smaller_than_carryover():
    # Adjustment smaller than the carryover consumes only part of it.
    assert split_adjustment(3, 8) == (3, 0, 5)


def test_second_adjustment_gets_no_prior_share():
    prior1, cur1, rem = split_adjustment(10, 6)
    assert (prior1, cur1, rem) == (6, 4, 0)
    prior2, cur2, rem = split_adjustment(7, rem)
    assert (prior2, cur2, rem) == (0, 7, 0)


# ---------------------------------------------------------------------------
# Ordering guard
# ---------------------------------------------------------------------------

def test_fresh_match_expects_h1_m1():
    assert is_expected(1, 0, 1, 1)
    assert not is_expected(1, 0, 1, 2)
    assert not is_expected(1, 0, 2, 1)


def test_next_minute_same_half():
    assert is_expected(1, 7, 1, 8)
    assert not is_expected(1, 7, 1, 9)
    assert not is_expected(1, 7, 1, 7)


def test_halftime_rollover():
    assert is_expected(1, 20, 2, 1)
    assert not is_expected(1, 20, 2, 2)


# ---------------------------------------------------------------------------
# build_payload — no orientation logic, no team_id_to_name param (dead code
# removed; orientation is applied once, at write time, by orient_for_team_a)
# ---------------------------------------------------------------------------

def test_build_payload_maps_internal_names_to_wire_keys():
    sums = {
        "frames_team_a": 10, "frames_team_b": 20, "frames_loose": 1, "frames_oof": 2,
        "passes_completed_team_a": 3, "passes_completed_team_b": 4,
        "interceptions_team_a": 5, "interceptions_team_b": 6,
        "ball_lost_team_a": 7, "ball_lost_team_b": 8,
    }
    payload = build_payload("m1", 1, 1, 0, sums)
    assert payload == {
        "frames_a": 10, "frames_b": 20, "frames_loose": 1, "frames_oof": 2,
        "passes_completed_a": 3, "passes_completed_b": 4,
        "interceptions_a": 5, "interceptions_b": 6,
        "ball_lost_a": 7, "ball_lost_b": 8,
    }


# ---------------------------------------------------------------------------
# orient_for_team_a — colour-anchored reorientation at write time
# ---------------------------------------------------------------------------

def _clip_result(**row_over):
    row = MinuteRow(
        match_id="m1", half=1, minute=1,
        frames_team_a=10, frames_team_b=20,
        passes_completed_team_a=1, passes_completed_team_b=2,
        interceptions_team_a=3, interceptions_team_b=4,
        ball_lost_team_a=5, ball_lost_team_b=6,
        **row_over,
    )
    correction = PriorCorrection(half=1, minute=0, kind="flip_to", team_id=0, frames=2)
    events = [EventRow(half=1, minute=1, frame_idx=9, kind="pass", from_team=0, to_team=None)]
    return ClipResult(minute_row=row, correction=correction, events=events)


def test_orient_noop_when_cluster_0_is_team_a():
    result = _clip_result()
    oriented = orient_for_team_a(result, team_a_cluster_id=0)
    assert oriented is result   # true no-op, not just equal


def test_orient_noop_when_cluster_id_unresolved():
    result = _clip_result()
    oriented = orient_for_team_a(result, team_a_cluster_id=None)
    assert oriented is result


def test_orient_swaps_minute_row_counters_when_cluster_1_is_team_a():
    result = _clip_result()
    oriented = orient_for_team_a(result, team_a_cluster_id=1)
    row = oriented.minute_row
    assert (row.frames_team_a, row.frames_team_b) == (20, 10)
    assert (row.passes_completed_team_a, row.passes_completed_team_b) == (2, 1)
    assert (row.interceptions_team_a, row.interceptions_team_b) == (4, 3)
    assert (row.ball_lost_team_a, row.ball_lost_team_b) == (6, 5)
    # Original result is untouched (a new ClipResult is returned).
    assert result.minute_row.frames_team_a == 10


def test_orient_swaps_event_team_ids():
    result = _clip_result()
    oriented = orient_for_team_a(result, team_a_cluster_id=1)
    evt = oriented.events[0]
    assert evt.from_team == 1     # was 0
    assert evt.to_team is None    # non-0/1 values pass through unchanged


def test_orient_swaps_correction_team_id():
    result = _clip_result()
    oriented = orient_for_team_a(result, team_a_cluster_id=1)
    assert oriented.correction.team_id == 1   # was 0


def test_orient_handles_no_correction():
    result = ClipResult(minute_row=_clip_result().minute_row, correction=None, events=[])
    oriented = orient_for_team_a(result, team_a_cluster_id=1)
    assert oriented.correction is None


# ---------------------------------------------------------------------------
# orient_for_team_a — correction gets its OWN orientation, independent of
# this clip's team_a_cluster_id (correction.py review finding: a
# PriorCorrection targets the PREVIOUS minute's row, which was written under
# whatever orientation was active back then, not necessarily today's)
# ---------------------------------------------------------------------------

def test_correction_orientation_defaults_to_row_orientation_when_omitted():
    # Backward-compatible default: caller not tracking the two separately.
    result = _clip_result()
    oriented = orient_for_team_a(result, team_a_cluster_id=1)
    assert oriented.correction.team_id == 1
    assert oriented.minute_row.frames_team_a == 20


def test_correction_orientation_independent_of_row_when_row_unresolved():
    # This clip's own row is unresolved (team_a_cluster_id=None → not
    # flipped) but the correction targets a PRIOR row that WAS written under
    # cluster 1 == team_a — the correction alone must flip.
    result = _clip_result()
    oriented = orient_for_team_a(result, team_a_cluster_id=None,
                                 correction_team_a_cluster_id=1)
    assert oriented.minute_row.frames_team_a == 10   # row itself untouched
    assert oriented.correction.team_id == 1           # was 0 — correction flipped


def test_correction_orientation_independent_of_row_when_correction_unresolved():
    # Mirror case: this clip's row IS reoriented (cluster 1 == team_a today)
    # but the prior row the correction targets was written before colour
    # resolution succeeded (correction_team_a_cluster_id=None) — the
    # correction must stay in raw cluster order.
    result = _clip_result()
    oriented = orient_for_team_a(result, team_a_cluster_id=1,
                                 correction_team_a_cluster_id=None)
    assert oriented.minute_row.frames_team_a == 20   # row flipped
    assert oriented.correction.team_id == 0           # correction NOT flipped


def test_correction_orientation_both_flip_when_both_resolved_to_cluster_1():
    result = _clip_result()
    oriented = orient_for_team_a(result, team_a_cluster_id=1,
                                 correction_team_a_cluster_id=1)
    assert oriented.minute_row.frames_team_a == 20
    assert oriented.correction.team_id == 1
