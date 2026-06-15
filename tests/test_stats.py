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
    LBL_TEAM0,
    LBL_TEAM1,
    MinuteCounters,
    is_expected,
    split_adjustment,
)


# ---------------------------------------------------------------------------
# Label bucketing
# ---------------------------------------------------------------------------

def test_add_label_buckets():
    c = MinuteCounters()
    for lbl in [LBL_TEAM0] * 3 + [LBL_TEAM1] * 2 + [LBL_LOOSE] + [LBL_OOF] * 4:
        c.add_label(lbl)
    assert (c.frames_team0, c.frames_team1, c.frames_loose, c.frames_oof) == (3, 2, 1, 4)


def test_unknown_label_counts_as_oof():
    c = MinuteCounters()
    c.add_label("something_else")
    assert c.frames_oof == 1


# ---------------------------------------------------------------------------
# Adjustments (current-minute portion)
# ---------------------------------------------------------------------------

def test_flip_to_moves_frames_between_teams():
    c = MinuteCounters(frames_team0=10, frames_team1=5)
    c.apply_adjustment("flip_to", 1, 4)     # 4 frames belong to team1, not team0
    assert (c.frames_team0, c.frames_team1) == (6, 9)


def test_flip_to_clamps_at_available():
    c = MinuteCounters(frames_team0=2, frames_team1=0)
    c.apply_adjustment("flip_to", 1, 10)
    assert (c.frames_team0, c.frames_team1) == (0, 2)


def test_drop_moves_frames_to_oof():
    c = MinuteCounters(frames_team0=10, frames_oof=1)
    c.apply_adjustment("drop", 0, 3)
    assert (c.frames_team0, c.frames_oof) == (7, 4)


def test_drop_clamps_at_available():
    c = MinuteCounters(frames_team1=2, frames_oof=0)
    c.apply_adjustment("drop", 1, 10)
    assert (c.frames_team1, c.frames_oof) == (0, 2)


def test_zero_or_negative_adjustment_is_noop():
    c = MinuteCounters(frames_team0=5)
    c.apply_adjustment("flip_to", 1, 0)
    c.apply_adjustment("drop", 0, -3)
    assert c.frames_team0 == 5


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
    assert c.passes_completed_t0 == 2
    assert c.passes_completed_t1 == 1
    assert c.interceptions_t0 == 1
    assert c.ball_lost_t1 == 1


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
