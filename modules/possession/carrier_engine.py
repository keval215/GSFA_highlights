"""Per-frame "who has the ball" with hysteresis debouncing.

Split out of the old video_analysis/possession.py. The foot-zone sizing
(`foot_zone_ratio`/`foot_zone_min_px`/`foot_zone_max_px`) used to be module-
level globals read directly inside a free function; they're now real
constructor parameters so a ruleset config can hand each sport its own
values (e.g. classic football's wider camera framing) instead of editing
this file.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

from modules.detectors.player_detector import Detection
from modules.possession.ball_tracker import BallDetection, BallTracker


@dataclass
class CarrierState:
    kind:     str                    # "carrier" | "loose" | "oof"
    track_id: Optional[int] = None
    team_id:  Optional[int] = None
    player:   Optional[Detection] = None


class CarrierEngine:
    """Computes the per-frame carrier from ball position + player bboxes.

    Foot zone = foot_zone_ratio * bbox_height (clamped to
    [foot_zone_min_px, foot_zone_max_px]). The zone is measured from the ball
    centre to the player's foot_point in image pixels, but because both
    quantities translate together when the camera pans, the test is
    pan-invariant within a single frame.

    Raw per-frame label can be noisy across one-frame ID flickers, so we
    debounce with a small hysteresis buffer. A change of (kind, track_id)
    is committed only after N consecutive matching raw samples.

    Defaults below match today's futsal-tuned values; a ruleset config
    supplies its own values explicitly at construction time.
    """

    OOF_STATE   = CarrierState(kind="oof")
    LOOSE_STATE = CarrierState(kind="loose")

    def __init__(
        self,
        hysteresis_n:    int   = 3,
        foot_zone_ratio: float = 0.45,
        foot_zone_min_px: int  = 20,
        foot_zone_max_px: int  = 140,
    ) -> None:
        self.hysteresis_n    = hysteresis_n
        self.foot_zone_ratio = foot_zone_ratio
        self.foot_zone_min_px = foot_zone_min_px
        self.foot_zone_max_px = foot_zone_max_px
        self._buffer: deque[CarrierState] = deque(maxlen=hysteresis_n)
        self._committed: CarrierState = self.OOF_STATE

    def get_state(self) -> CarrierState:
        return self._committed

    def update(
        self,
        players:    list[Detection],
        ball_state: str,
        ball:       Optional[BallDetection],
    ) -> CarrierState:
        raw = self._raw(players, ball_state, ball)
        self._buffer.append(raw)

        if len(self._buffer) < self.hysteresis_n:
            return self._committed

        keys = {(s.kind, s.track_id) for s in self._buffer}
        if len(keys) == 1:
            # Use the latest sample so player Detection reference is fresh.
            self._committed = self._buffer[-1]
        return self._committed

    def _foot_zone_radius(self, p: Detection) -> float:
        h = max(1, p.bbox[3] - p.bbox[1])
        r = self.foot_zone_ratio * h
        return max(self.foot_zone_min_px, min(self.foot_zone_max_px, r))

    def _raw(
        self,
        players: list[Detection],
        ball_state: str,
        ball: Optional[BallDetection],
    ) -> CarrierState:
        if ball_state == BallTracker.LOST or ball is None:
            return self.OOF_STATE

        bx, by = ball.centre

        in_zone: list[tuple[float, Detection]] = []
        for p in players:
            if p.team_id not in (0, 1):
                continue
            if p.track_id is None:
                continue
            d = math.dist(p.foot_point, (bx, by))
            if d <= self._foot_zone_radius(p):
                in_zone.append((d, p))

        if not in_zone:
            return self.LOOSE_STATE

        in_zone.sort(key=lambda x: x[0])
        teams = {p.team_id for _, p in in_zone}
        if len(teams) > 1:
            return self.LOOSE_STATE

        d, p = in_zone[0]
        return CarrierState(
            kind     = "carrier",
            track_id = p.track_id,
            team_id  = p.team_id,
            player   = p,
        )
