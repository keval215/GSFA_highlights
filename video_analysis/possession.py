"""
video_analysis/possession.py — Ball possession analysis for GSFA football matches.

Pipeline:
  Detect (RF-DETR ball + YOLOv11 players) -> SigLIP team -> GK -> BoT-SORT (+GMC)
  -> BallTracker (Kalman smoothing) -> CarrierEngine (bbox-relative foot-zone)
  -> PassEventTracker (release / travel / reception phases)
  -> PossessionStats (strict denominator)

Run:
    python video_analysis/possession.py
"""

from __future__ import annotations

import math
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import supervision as sv
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).parent.parent))

from detectors.goalkeeper_detector import GoalkeeperDetector
from detectors.player_detector import Detection, FrameDetections, PlayerDetector
from team_classifier.team_classifier import GSFATeamClassifier
from tracking.player_tracker import PlayerTracker

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

BALL_MODEL_WEIGHTS = r"C:\Users\Admin\OneDrive\Desktop\CZ\gsfa_ball_detection.pth"
VIDEO_PATH         = r"C:\Users\Admin\Downloads\Video Project_1min.mp4"
OUTPUT_PATH        = r"data/output/possession_output.mp4"

BALL_CLASS_ID  = 1
BALL_CONF      = 0.25      # lowered from 0.35; Kalman gates false positives

# BallTracker
KALMAN_COAST_FRAMES   = 12         # emit predicted position for up to N frames
KALMAN_GATE_SIGMA     = 6.0        # reject measurements outside this many sigma

# CarrierEngine
FOOT_ZONE_RATIO       = 0.45       # foot_zone_radius = ratio * bbox_height
FOOT_ZONE_MIN_PX      = 20
FOOT_ZONE_MAX_PX      = 140
CARRIER_HYSTERESIS_N  = 3          # frames of agreement to commit a state change

# PassEventTracker — tuned for TARGET_PROCESS_FPS = 15 fps
RELEASE_SUSTAIN_R     = 1   # processed frames (~67 ms)
RECEPTION_SETTLE_C    = 2   # processed frames (~133 ms)
TRAVEL_MIN_GAP        = 1   # processed frames (~67 ms)
TRAVEL_TIMEOUT_FRAMES = 22  # processed frames (~1.47 s)

# Debug / speed run
TARGET_PROCESS_FPS   = 15.0        # analyse every Nth frame
PROCESS_DURATION_SEC = 60       # stop after this many seconds

# Visuals
POSSESS_BAR_H   = 42
TEAM_BGR  = {0: (255, 80, 0), 1: (0, 80, 255), None: (160, 160, 160)}
GK_BGR    = (0, 215, 255)
BALL_BGR  = (0, 255, 255)
REF_BGR   = (80, 220, 80)

# Possession labels surfaced to stats
POSSESS_TEAM0     = "team0"
POSSESS_TEAM1     = "team1"
POSSESS_LOOSE     = "loose"
POSSESS_OOF       = "oof"

# ---------------------------------------------------------------------------
# BALL DETECTION + KALMAN TRACKING
# ---------------------------------------------------------------------------

@dataclass
class BallDetection:
    bbox:       tuple[int, int, int, int]
    centre:     tuple[int, int]
    confidence: float


class BallDetector:
    def __init__(
        self,
        weights:       str   = BALL_MODEL_WEIGHTS,
        ball_class_id: int   = BALL_CLASS_ID,
        conf:          float = BALL_CONF,
    ) -> None:
        from rfdetr import RFDETRMedium
        print(f"[BallDetector] Loading: {weights}")
        self.model = RFDETRMedium.from_checkpoint(weights, num_classes=2, resolution=576)
        self.ball_class_id = ball_class_id
        self.conf          = conf

    def detect(self, frame: np.ndarray) -> Optional[BallDetection]:
        pil  = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        dets = self.model.predict(pil, threshold=self.conf)
        if dets is None or len(dets) == 0:
            return None
        mask = dets.class_id == self.ball_class_id
        if not np.any(mask):
            return None
        confs = dets.confidence[mask]
        boxes = dets.xyxy[mask]
        best  = int(np.argmax(confs))
        x1, y1, x2, y2 = (int(v) for v in boxes[best])
        return BallDetection(
            bbox       = (x1, y1, x2, y2),
            centre     = ((x1 + x2) // 2, (y1 + y2) // 2),
            confidence = float(confs[best]),
        )


class BallTracker:
    """Constant-velocity Kalman around ball centre.

    Responsibilities:
      • Smooth the ball position frame to frame.
      • Coast through detection gaps for up to KALMAN_COAST_FRAMES frames.
      • Reject impossible-velocity false positives via a Mahalanobis-style gate.

    Notes:
      • Velocity is reported in *image pixels per frame*. It includes camera
        motion. Do NOT use it for kick / acceleration detection.
      • Pass detection in the FSM is purely geometric (ball-in/out of a
        bbox-relative foot zone), so the camera motion mixed into v_ball
        does not affect pass counts.
    """

    DETECTED = "detected"
    COASTING = "coasting"
    LOST     = "lost"

    def __init__(
        self,
        coast_frames: int   = KALMAN_COAST_FRAMES,
        gate_sigma:   float = KALMAN_GATE_SIGMA,
    ) -> None:
        self.coast_frames = coast_frames
        self.gate_sigma   = gate_sigma
        self.state        = self.LOST
        self.missing_for  = 0
        self.last_obs:    Optional[BallDetection] = None
        self._initialised = False
        self._kf          = self._make_kf()

    @staticmethod
    def _make_kf() -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        kf.processNoiseCov     = np.diag([4.0, 4.0, 25.0, 25.0]).astype(np.float32)
        kf.measurementNoiseCov = np.diag([9.0, 9.0]).astype(np.float32)
        kf.errorCovPost        = np.eye(4, dtype=np.float32) * 1000.0
        return kf

    def _seed(self, det: BallDetection) -> None:
        x, y = det.centre
        self._kf.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
        self._kf.errorCovPost = np.eye(4, dtype=np.float32) * 100.0
        self._initialised = True

    def update(self, det: Optional[BallDetection]) -> tuple[str, Optional[BallDetection]]:
        """Advance the filter by one frame and return (state, BallDetection|None).

        The returned BallDetection's `centre` is the *filtered* position.
        `bbox` is the most recent observed bbox (held constant while coasting).
        """
        if not self._initialised:
            if det is None:
                self.state = self.LOST
                return self.LOST, None
            self._seed(det)
            self.state       = self.DETECTED
            self.missing_for = 0
            self.last_obs    = det
            return self.DETECTED, det

        pred = self._kf.predict()
        # cv2.KalmanFilter returns an (N, 1) column vector; .item() pulls the
        # scalar out (float() on a size-1 ndarray raises under numpy 2.x).
        px, py = float(pred[0].item()), float(pred[1].item())

        if det is not None and self._accept(det, px, py):
            meas = np.array([[np.float32(det.centre[0])], [np.float32(det.centre[1])]])
            est  = self._kf.correct(meas)
            ex, ey = int(est[0].item()), int(est[1].item())
            self.state       = self.DETECTED
            self.missing_for = 0
            self.last_obs    = det
            out = BallDetection(
                bbox       = det.bbox,
                centre     = (ex, ey),
                confidence = det.confidence,
            )
            return self.DETECTED, out

        # No accepted measurement this frame — coast.
        self.missing_for += 1
        if self.missing_for <= self.coast_frames and self.last_obs is not None:
            self.state = self.COASTING
            out = BallDetection(
                bbox       = self.last_obs.bbox,
                centre     = (int(px), int(py)),
                confidence = 0.0,
            )
            return self.COASTING, out

        self.state    = self.LOST
        self.last_obs = None
        return self.LOST, None

    def _accept(self, det: BallDetection, px: float, py: float) -> bool:
        S = self._kf.measurementMatrix @ self._kf.errorCovPre @ self._kf.measurementMatrix.T \
            + self._kf.measurementNoiseCov
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return True
        innov = np.array([det.centre[0] - px, det.centre[1] - py], dtype=np.float64)
        m2 = float(innov @ S_inv @ innov)
        return m2 <= (self.gate_sigma ** 2)


# ---------------------------------------------------------------------------
# CARRIER ENGINE — per-frame "who has the ball" with hysteresis
# ---------------------------------------------------------------------------

@dataclass
class CarrierState:
    kind:     str                    # "carrier" | "loose" | "oof"
    track_id: Optional[int] = None
    team_id:  Optional[int] = None
    player:   Optional[Detection] = None


def _foot_zone_radius(p: Detection) -> float:
    h = max(1, p.bbox[3] - p.bbox[1])
    r = FOOT_ZONE_RATIO * h
    return max(FOOT_ZONE_MIN_PX, min(FOOT_ZONE_MAX_PX, r))


class CarrierEngine:
    """Computes the per-frame carrier from ball position + player bboxes.

    Foot zone = FOOT_ZONE_RATIO * bbox_height (clamped). The zone is
    measured from the ball centre to the player's foot_point in image
    pixels, but because both quantities translate together when the
    camera pans, the test is pan-invariant within a single frame.

    Raw per-frame label can be noisy across one-frame ID flickers, so we
    debounce with a small hysteresis buffer. A change of (kind, track_id)
    is committed only after N consecutive matching raw samples.
    """

    OOF_STATE   = CarrierState(kind="oof")
    LOOSE_STATE = CarrierState(kind="loose")

    def __init__(self, hysteresis_n: int = CARRIER_HYSTERESIS_N) -> None:
        self.hysteresis_n = hysteresis_n
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
            if d <= _foot_zone_radius(p):
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


# ---------------------------------------------------------------------------
# PASS EVENT TRACKER — release / travel / reception phases
# ---------------------------------------------------------------------------

EVT_COMPLETED      = "completed"
EVT_INTERCEPTION   = "interception"
EVT_BALL_LOST      = "ball_lost"
# EVT_SHOT placeholder — no shot detector wired in yet.

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
    """

    def __init__(
        self,
        release_sustain: int = RELEASE_SUSTAIN_R,
        reception_settle: int = RECEPTION_SETTLE_C,
        travel_min_gap:   int = TRAVEL_MIN_GAP,
        travel_timeout:   int = TRAVEL_TIMEOUT_FRAMES,
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
            possession_label: one of POSSESS_TEAM{0,1} | POSSESS_LOOSE
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
            return POSSESS_TEAM0 if team == 0 else POSSESS_TEAM1
        if self.phase in (PHASE_CAND_REL, PHASE_TRAVEL, PHASE_CAND_RCV):
            # Provisional — credit the passer's team. Adjusted later on
            # interception / ball_lost.
            team = self._passer[1] if self._passer else None
            if team == 0:
                return POSSESS_TEAM0
            if team == 1:
                return POSSESS_TEAM1
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


# ---------------------------------------------------------------------------
# POSSESSION STATS — strict denominator with retroactive adjustments
# ---------------------------------------------------------------------------

class PossessionStats:
    """Counts in-play frames per outcome and exposes possession % over only
    the confirmed-team frames (loose / OOF are excluded).

    Provisional team credit accumulated during travel is corrected via
    `apply_adjustments` when the pass FSM resolves the event:
        ("flip_to", team_id, n)  → move n frames from the other team to team_id
        ("drop",    team_id, n)  → remove n frames from team_id, give them to OOF
    """

    def __init__(self) -> None:
        self.frame_counts: dict[str, int] = {
            POSSESS_TEAM0: 0,
            POSSESS_TEAM1: 0,
            POSSESS_LOOSE: 0,
            POSSESS_OOF:   0,
        }
        self.total = 0

    def update(self, label: str) -> None:
        self.frame_counts[label] = self.frame_counts.get(label, 0) + 1
        self.total += 1

    def apply_adjustments(self, adjustments: list[tuple[str, int, int]]) -> None:
        for kind, team_id, n in adjustments:
            src_label = POSSESS_TEAM0 if team_id == 0 else POSSESS_TEAM1
            other_label = POSSESS_TEAM1 if team_id == 0 else POSSESS_TEAM0
            if kind == "flip_to":
                # We credited `other` provisionally; move n frames over to team_id.
                move = min(n, self.frame_counts[other_label])
                self.frame_counts[other_label] -= move
                self.frame_counts[src_label]   += move
            elif kind == "drop":
                # We credited team_id provisionally; reclassify n as OOF.
                move = min(n, self.frame_counts[src_label])
                self.frame_counts[src_label] -= move
                self.frame_counts[POSSESS_OOF] += move

    def percentages(self) -> tuple[float, float]:
        denom = self.frame_counts[POSSESS_TEAM0] + self.frame_counts[POSSESS_TEAM1]
        if denom == 0:
            return 0.0, 0.0
        return (
            100.0 * self.frame_counts[POSSESS_TEAM0] / denom,
            100.0 * self.frame_counts[POSSESS_TEAM1] / denom,
        )

    def summary(self) -> str:
        t0, t1 = self.percentages()
        d = max(1, self.total)
        return "\n".join([
            "--- Possession Summary (denominator excludes loose/OOF) ---",
            f"  Team 0: {t0:.1f}%",
            f"  Team 1: {t1:.1f}%",
            f"  Loose : {100*self.frame_counts[POSSESS_LOOSE]/d:.1f}%",
            f"  OOF   : {100*self.frame_counts[POSSESS_OOF]/d:.1f}%",
        ])


# ---------------------------------------------------------------------------
# SUPERVISION ANNOTATORS
# ---------------------------------------------------------------------------
# Index 0 = team0 (blue), 1 = team1 (red), 2 = unclassified (grey)
_PALETTE = sv.ColorPalette.from_hex(["#0050FF", "#FF5000", "#A0A0A0"])

_ellipse_ann  = sv.EllipseAnnotator(color=_PALETTE, thickness=2)
_triangle_ann = sv.TriangleAnnotator(
    color=sv.Color.from_hex("#00FFFF"), base=16, height=16,
    color_lookup=sv.ColorLookup.INDEX,
)
_label_ann = sv.LabelAnnotator(
    color          = _PALETTE,
    text_color     = sv.Color.WHITE,
    text_scale     = 0.38,
    text_thickness = 1,
    text_padding   = 3,
)


def _to_sv(players: list[Detection]) -> tuple[sv.Detections, list[str]]:
    if not players:
        return sv.Detections(xyxy=np.empty((0, 4))), []
    xyxy     = np.array([[*p.bbox] for p in players], dtype=float)
    class_id = np.array(
        [p.team_id if p.team_id is not None else 2 for p in players], dtype=int
    )
    labels = []
    for p in players:
        base = (f"GK-T{p.team_id}" if p.is_goalkeeper
                else (f"T{p.team_id}" if p.team_id is not None else "?"))
        labels.append(f"{base}#{p.track_id}" if p.track_id is not None else base)
    return sv.Detections(xyxy=xyxy, class_id=class_id), labels


# ---------------------------------------------------------------------------
# FRAME RENDERING
# ---------------------------------------------------------------------------

def draw_frame(
    frame:        np.ndarray,
    player_dets:  FrameDetections,
    ball_det:     Optional[BallDetection],
    carrier:      CarrierState,
    poss_stats:   PossessionStats,
    pass_tracker: PassEventTracker,
) -> np.ndarray:
    out = frame.copy()
    H_vid, W_vid = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    sv_dets, labels = _to_sv(player_dets.players)
    if len(sv_dets) > 0:
        out = _ellipse_ann.annotate(scene=out, detections=sv_dets)
        out = _label_ann.annotate(scene=out, detections=sv_dets, labels=labels)

    for p in player_dets.players:
        if p.is_goalkeeper:
            x1, y1, x2, y2 = p.bbox
            cx = (x1 + x2) // 2
            rx = max((x2 - x1) // 2, 10)
            cv2.ellipse(out, (cx, y2), (rx, 6), 0, -45, 225, GK_BGR, 2)

    for gp in player_dets.goal_posts:
        x1, y1, x2, y2 = gp.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), 2)

    if ball_det is not None:
        ball_sv = sv.Detections(xyxy=np.array([[*ball_det.bbox]], dtype=float))
        out = _triangle_ann.annotate(scene=out, detections=ball_sv)

    if carrier.kind == "carrier" and carrier.player is not None:
        cp = carrier.player.foot_point
        cv2.circle(out, cp, 14, (255, 255, 0), 2)

    # Possession bar
    t0_pct, t1_pct = poss_stats.percentages()
    bx1, bx2 = 10, W_vid - 10
    by1, by2 = 8, 8 + POSSESS_BAR_H
    bar_w = bx2 - bx1
    cv2.rectangle(out, (bx1, by1), (bx2, by2), (25, 25, 25), -1)
    t0_w = int(bar_w * t0_pct / 100)
    if t0_w > 0:
        cv2.rectangle(out, (bx1, by1), (bx1 + t0_w, by2), TEAM_BGR[0], -1)
    t1_w = int(bar_w * t1_pct / 100)
    if t1_w > 0:
        cv2.rectangle(out, (bx2 - t1_w, by1), (bx2, by2), TEAM_BGR[1], -1)
    cv2.putText(out, f"T0  {t0_pct:.1f}%",
                (bx1 + 8, by2 - 10), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"{t1_pct:.1f}%  T1",
                (bx2 - 130, by2 - 10), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Pass overlay
    pass_summary = pass_tracker.summary_for_overlay()
    y0 = by2 + 24
    for tid in (0, 1):
        s = pass_summary[tid]
        txt = f"T{tid}  passes: {s['successful']}   inaccurate: {s['inaccurate']}"
        cv2.putText(out, txt, (12, y0 + tid * 22),
                    font, 0.55, TEAM_BGR[tid], 2, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run(video_path: str = VIDEO_PATH, out_path: str = OUTPUT_PATH) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print("[Possession] Initialising models …")
    player_det     = PlayerDetector()
    ball_det_model = BallDetector(BALL_MODEL_WEIGHTS)

    team_clf = GSFATeamClassifier()
    team_clf.fit_from_video_or_load(video_path, player_det)

    gk_det = GoalkeeperDetector()
    gk_det.fit_from_video_or_load(video_path, player_det, team_clf)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W_vid   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_vid   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    import warnings, os
    warnings.filterwarnings("ignore")
    os.environ["TQDM_DISABLE"] = "1"

    frame_step = max(1, round(fps / TARGET_PROCESS_FPS))
    max_frame  = int(fps * PROCESS_DURATION_SEC)
    eff_fps    = fps / frame_step

    tracker      = PlayerTracker(fps=fps)
    ball_tracker = BallTracker()
    carrier_eng  = CarrierEngine()
    pass_track   = PassEventTracker()   # constants are in 15fps processed-frame units
    stats        = PossessionStats()

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        TARGET_PROCESS_FPS,
        (W_vid, H_vid),
    )

    print(
        f"[Possession] first {PROCESS_DURATION_SEC:.0f}s | "
        f"native {fps:.0f}fps → {eff_fps:.0f}fps (step={frame_step}) | "
        f"{W_vid}x{H_vid}"
    )
    print("-" * 60)

    fidx     = 0
    proc_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or fidx >= max_frame:
            break

        if fidx % frame_step != 0:
            fidx += 1
            continue

        t_sec = fidx / fps

        player_dets = player_det.detect(frame, fidx, fps)
        team_clf.classify(frame, player_dets)
        gk_det.classify(player_dets)
        tracker.update(frame, player_dets.players)

        raw_ball         = ball_det_model.detect(frame)
        ball_state, ball = ball_tracker.update(raw_ball)

        carrier = carrier_eng.update(player_dets.players, ball_state, ball)

        prev_phase  = pass_track.phase
        prev_passer = pass_track._passer
        prev_n_evts = len(pass_track.events)

        label, adjustments = pass_track.update(carrier, proc_idx)

        # --- Phase-transition logs ---
        cur_phase = pass_track.phase
        if cur_phase != prev_phase:
            p = pass_track._passer
            r = pass_track._receiver
            if cur_phase == PHASE_POSS and prev_phase == PHASE_IDLE:
                print(f"[{t_sec:.2f}s] NEW CARRIER       team={p[1]} track={p[0]}")
            elif cur_phase == PHASE_CAND_REL:
                src = prev_passer or p
                if src:
                    print(f"[{t_sec:.2f}s] CAND RELEASE      team={src[1]} track={src[0]}  ball left foot-zone")
            elif cur_phase == PHASE_TRAVEL and prev_phase == PHASE_CAND_REL:
                if p:
                    print(f"[{t_sec:.2f}s] RELEASE CONFIRMED team={p[1]} track={p[0]}  ball in travel")
            elif cur_phase == PHASE_CAND_RCV:
                if r and p:
                    print(f"[{t_sec:.2f}s] CAND RECEPTION    receiver team={r[1]} track={r[0]}  from team={p[1]} track={p[0]}")
            elif cur_phase == PHASE_TRAVEL and prev_phase == PHASE_CAND_RCV:
                print(f"[{t_sec:.2f}s] RECEPTION DROPPED receiver lost ball, back to travel")

        proc_idx += 1

        # --- Resolved-event logs ---
        for evt in pass_track.events[prev_n_evts:]:
            t_evt = evt.end_frame / TARGET_PROCESS_FPS
            if evt.kind == EVT_COMPLETED:
                print(
                    f"[{t_evt:.2f}s] PASS COMPLETED    "
                    f"T{evt.from_team_id}#{evt.from_track_id} → T{evt.to_team_id}#{evt.to_track_id}  "
                    f"travel={evt.travel_frames}f"
                )
            elif evt.kind == EVT_INTERCEPTION:
                print(
                    f"[{t_evt:.2f}s] INTERCEPTED       "
                    f"T{evt.from_team_id}#{evt.from_track_id} → T{evt.to_team_id}#{evt.to_track_id}  "
                    f"travel={evt.travel_frames}f"
                )
            elif evt.kind == EVT_BALL_LOST:
                print(
                    f"[{t_evt:.2f}s] BALL LOST         "
                    f"from T{evt.from_team_id}#{evt.from_track_id}  "
                    f"travel={evt.travel_frames}f"
                )

        stats.update(label)
        if adjustments:
            stats.apply_adjustments(adjustments)

        out_frame = draw_frame(frame, player_dets, ball, carrier, stats, pass_track)
        writer.write(out_frame)

        fidx += 1

    cap.release()
    writer.release()
    print("-" * 60)
    print(f"\n[Possession] Saved → {out_path}")
    print(stats.summary())
    print(pass_track.summary())


if __name__ == "__main__":
    run()
