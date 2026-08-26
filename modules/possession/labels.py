"""Shared possession-label and pass-event-kind string constants.

Split out on their own so both `pass_event_tracker.py` and `possession_stats.py`
can depend on the label vocabulary without depending on each other.
"""

from __future__ import annotations

# Possession labels surfaced to stats
POSSESS_TEAM0 = "team0"
POSSESS_TEAM1 = "team1"
POSSESS_LOOSE = "loose"
POSSESS_OOF   = "oof"

# Pass event kinds
EVT_COMPLETED    = "completed"
EVT_INTERCEPTION = "interception"
EVT_BALL_LOST    = "ball_lost"
# EVT_SHOT placeholder — no shot detector wired in yet.
