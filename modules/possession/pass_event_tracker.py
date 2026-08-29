"""Release / travel / reception pass-event FSM.

Split out of the old video_analysis/possession.py. All four timing
parameters were already constructor args backed by module-level constants
tuned for 15 fps futsal footage; they're unchanged here except the defaults
are now literal (a ruleset config supplies its own values explicitly).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from modules.possession.carrier_engine import CarrierState
from modules.possession.labels import (
    EVT_BALL_LOST,
    EVT_COMPLETED,
    EVT_INTERCEPTION,
    POSSESS_LOOSE,
    POSSESS_OOF,
    POSSESS_TEAM_A,
    POSSESS_TEAM_B,
)

PHASE_IDLE       = "idle"
PHASE_POSS       = "in_possession"
PHASE_CAND_REL   = "candidate_release"
PHASE_TRAVEL     = "travel"
PHASE_CAND_RCV   = "candidate_reception"


@dataclass
class PassEvent:
    kind:           str
    from_track_id:  Optional[int]
    from_team_id:   Optional[int]
    to_track_id:    Optional[int]
    to_team_id:     Optional[int]
    release_frame:  int
    end_frame:      int
    travel_frames:  int


class PassEventTracker:
    """3-phase pass FSM driven by CarrierState transitions.

    Phases:
      IDLE                — no confirmed carrier yet.
      IN_POSSESSION(A)    — carrier A holds the ball.
      CAND_RELEASE(A)     — carrier signal left A; waiting R frames to confirm.
      TRAVEL(A)           — release confirmed; waiting for a reception or timeout.
      CAND_RECEPTION(A,B) — ball entered B's zone; waiting C frames to settle.

    Outputs each frame: possession_label (provisional or committed) and a
    list of resolved events. PossessionStats applies retroactive adjustments
    on interception/ball_lost so the provisional credit during travel is
    flipped or dropped to match reality.

    Defaults below match today's futsal tuning (calibrated for 15 fps); a
    ruleset config supplies its own values explicitly at construction time.
    """

    def __init__(
        self,
        release_sustain: int = 1,
        reception_settle: int = 2,
        travel_min_gap:   int = 1,
        travel_timeout:   int = 22,
    ) -> None:
        self.release_sustain  = release_sustain
        self.reception_settle = reception_settle
        self.travel_min_gap   = travel_min_gap
        self.travel_timeout   = travel_timeout

        self.phase: str = PHASE_IDLE
        self._passer:    Optional[tuple[int, int]] = None  # (track_id, team_id)
        self._receiver:  Optional[tuple[int, int]] = None
        self._cand_release_at:   int = -1
        self._release_at:        int = -1
        self._cand_reception_at: int = -1
        self._travel_credit_team: Optional[int] = None
        self._travel_frames_so_far: int = 0   # provisional frames credited to passer

        self.events: list[PassEvent] = []
        self.stats_internal: dict[int, dict[str, int]] = {
            0: {EVT_COMPLETED: 0, EVT_INTERCEPTION: 0, EVT_BALL_LOST: 0},
            1: {EVT_COMPLETED: 0, EVT_INTERCEPTION: 0, EVT_BALL_LOST: 0},
        }

    # ------------------------------------------------------------------
    # Public per-frame step
    # ------------------------------------------------------------------

    def update(
        self,
        carrier: CarrierState,
        frame_idx: int,
    ) -> tuple[str, list[tuple[str, int, int]]]:
        """Advance the FSM by one frame.

        Returns:
          (possession_label, adjustments)
            possession_label: one of POSSESS_TEAM_A/B | POSSESS_LOOSE
                              | POSSESS_OOF
            adjustments     : list of (kind, team_id, frame_count) that
                              PossessionStats should retroactively apply.
                              kind ∈ {"flip_to", "drop"}.
        """
        adjustments: list[tuple[str, int, int]] = []

        prev_phase = self.phase
        self._step(carrier, frame_idx, adjustments)

        # Provisional-credit accumulation during travel-like phases.
        if self.phase in (PHASE_CAND_REL, PHASE_TRAVEL, PHASE_CAND_RCV):
            self._travel_frames_so_far += 1

        # Current possession label for this frame.
        label = self._label_for_current_state(carrier)
        return label, adjustments

    # ------------------------------------------------------------------
    # FSM step
    # ------------------------------------------------------------------

    def _step(
        self,
        carrier: CarrierState,
        f: int,
        adjustments: list[tuple[str, int, int]],
    ) -> None:
        if self.phase == PHASE_IDLE:
            if carrier.kind == "carrier":
                self._enter_possession(carrier)
            return

        if self.phase == PHASE_POSS:
            assert self._passer is not None
            if carrier.kind == "carrier" and carrier.track_id == self._passer[0]:
                return  # still holding
            self._enter_cand_release(f)
            return

        if self.phase == PHASE_CAND_REL:
            assert self._passer is not None
            dt = f - self._cand_release_at
            if carrier.kind == "carrier" and carrier.track_id == self._passer[0]:
                # Touch came straight back — passer keeps the ball.
                self.phase     = PHASE_POSS
                self._receiver = None
                self._reset_travel()
                return
            if dt >= self.release_sustain:
                # Release confirmed.
                self._release_at = f
                self.phase = PHASE_TRAVEL
            return

        if self.phase == PHASE_TRAVEL:
            assert self._passer is not None
            travel_dt = f - self._release_at

            if travel_dt > self.travel_timeout:
                self._resolve_ball_lost(f, adjustments)
                return

            if carrier.kind == "carrier":
                if carrier.track_id == self._passer[0]:
                    # Ball came back to passer — keeps possession.
                    self.phase     = PHASE_POSS
                    self._receiver = None
                    self._reset_travel()
                    return
                if travel_dt >= self.travel_min_gap:
                    # Candidate reception by a new player.
                    self._enter_cand_reception(carrier, f)
                return
            # carrier.kind in (loose, oof) — stay in travel.
            return

        if self.phase == PHASE_CAND_RCV:
            assert self._passer is not None and self._receiver is not None
            dt = f - self._cand_reception_at

            if carrier.kind == "carrier" and carrier.track_id == self._receiver[0]:
                if dt >= self.reception_settle:
                    self._resolve_reception(f, adjustments)
                return
            # Receiver no longer the carrier — revert. If a different new
            # candidate is present, immediately re-open candidacy on them.
            self._receiver = None
            self.phase     = PHASE_TRAVEL
            if (
                carrier.kind == "carrier"
                and carrier.track_id != self._passer[0]
                and (f - self._release_at) >= self.travel_min_gap
            ):
                self._enter_cand_reception(carrier, f)
            return

    # ------------------------------------------------------------------
    # Phase entry / resolution helpers
    # ------------------------------------------------------------------

    def _enter_possession(self, carrier: CarrierState) -> None:
        assert carrier.track_id is not None and carrier.team_id is not None
        self.phase    = PHASE_POSS
        self._passer  = (carrier.track_id, carrier.team_id)
        self._receiver = None
        self._reset_travel()

    def _enter_cand_release(self, f: int) -> None:
        self.phase = PHASE_CAND_REL
        self._cand_release_at = f
        self._travel_credit_team = self._passer[1] if self._passer else None
        self._travel_frames_so_far = 0

    def _enter_cand_reception(self, carrier: CarrierState, f: int) -> None:
        assert carrier.track_id is not None and carrier.team_id is not None
        self.phase = PHASE_CAND_RCV
        self._receiver = (carrier.track_id, carrier.team_id)
        self._cand_reception_at = f

    def _resolve_ball_lost(
        self, f: int, adjustments: list[tuple[str, int, int]],
    ) -> None:
        assert self._passer is not None
        self.events.append(PassEvent(
            kind          = EVT_BALL_LOST,
            from_track_id = self._passer[0],
            from_team_id  = self._passer[1],
            to_track_id   = None,
            to_team_id    = None,
            release_frame = self._release_at,
            end_frame     = f,
            travel_frames = self._travel_frames_so_far,
        ))
        self.stats_internal[self._passer[1]][EVT_BALL_LOST] += 1
        # Provisional credit during travel was given to passer's team — drop
        # it (treated as OOF for possession %).
        if self._travel_credit_team is not None and self._travel_frames_so_far:
            adjustments.append(("drop", self._travel_credit_team, self._travel_frames_so_far))
        self.phase = PHASE_IDLE
        self._passer = None
        self._receiver = None
        self._reset_travel()

    def _resolve_reception(
        self, f: int, adjustments: list[tuple[str, int, int]],
    ) -> None:
        assert self._passer is not None and self._receiver is not None
        from_tid, from_tid_team = self._passer
        to_tid, to_team = self._receiver
        same_team = (from_tid_team == to_team)
        kind = EVT_COMPLETED if same_team else EVT_INTERCEPTION
        self.events.append(PassEvent(
            kind          = kind,
            from_track_id = from_tid,
            from_team_id  = from_tid_team,
            to_track_id   = to_tid,
            to_team_id    = to_team,
            release_frame = self._release_at,
            end_frame     = f,
            travel_frames = self._travel_frames_so_far,
        ))
        self.stats_internal[from_tid_team][kind] += 1

        if not same_team and self._travel_credit_team is not None and self._travel_frames_so_far:
            # Provisional credit went to passer; reality is the other team
            # actually owned the ball during travel. Flip retroactively.
            adjustments.append(("flip_to", to_team, self._travel_frames_so_far))

        # Receiver becomes the new carrier.
        self.phase    = PHASE_POSS
        self._passer  = (to_tid, to_team)
        self._receiver = None
        self._reset_travel()

    def _reset_travel(self) -> None:
        self._cand_release_at = -1
        self._release_at      = -1
        self._cand_reception_at = -1
        self._travel_credit_team = None
        self._travel_frames_so_far = 0

    # ------------------------------------------------------------------
    # Per-frame possession label
    # ------------------------------------------------------------------

    def _label_for_current_state(self, carrier: CarrierState) -> str:
        if self.phase == PHASE_POSS:
            team = self._passer[1] if self._passer else None
            return POSSESS_TEAM_A if team == 0 else POSSESS_TEAM_B
        if self.phase in (PHASE_CAND_REL, PHASE_TRAVEL, PHASE_CAND_RCV):
            # Provisional — credit the passer's team. Adjusted later on
            # interception / ball_lost. (team_id 0 → team_a, 1 → team_b)
            team = self._passer[1] if self._passer else None
            if team == 0:
                return POSSESS_TEAM_A
            if team == 1:
                return POSSESS_TEAM_B
            # Falls through if passer somehow None.
        if carrier.kind == "loose":
            return POSSESS_LOOSE
        return POSSESS_OOF

    # ------------------------------------------------------------------
    # External summary in the legacy successful / inaccurate schema
    # ------------------------------------------------------------------

    def summary_for_overlay(self) -> dict[int, dict[str, int]]:
        """Maps the rich internal schema to the legacy schema used by overlays:
            completed   -> successful
            interception-> inaccurate (counted on the passer's team)
            ball_lost   -> ignored
        """
        out: dict[int, dict[str, int]] = {
            0: {"successful": 0, "inaccurate": 0},
            1: {"successful": 0, "inaccurate": 0},
        }
        for tid in (0, 1):
            out[tid]["successful"] = self.stats_internal[tid][EVT_COMPLETED]
            out[tid]["inaccurate"] = self.stats_internal[tid][EVT_INTERCEPTION]
        return out

    def summary(self) -> str:
        s = self.summary_for_overlay()
        lines = ["--- Pass Summary ---"]
        for tid in (0, 1):
            ok = s[tid]["successful"]
            bad = s[tid]["inaccurate"]
            acc = f"{100 * ok / max(1, ok + bad):.0f}%"
            internal = self.stats_internal[tid]
            lines.append(
                f"  Team {tid}: {ok} successful  {bad} inaccurate  ({acc} accuracy)"
            )
            lines.append(
                f"    [internal: completed={internal[EVT_COMPLETED]} "
                f"intercepted={internal[EVT_INTERCEPTION]} "
                f"lost={internal[EVT_BALL_LOST]}]"
            )
        return "\n".join(lines)
