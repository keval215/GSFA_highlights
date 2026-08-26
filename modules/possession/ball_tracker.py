"""Ball detection adapter + Kalman tracker.

Split out of the old video_analysis/possession.py so it can be reused,
unchanged, by both the local dev script and the Azure service — parametrized
per sport via a ruleset config rather than module-level constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from modules.detectors.player_detector import FrameDetections


@dataclass
class BallDetection:
    bbox:       tuple[int, int, int, int]
    centre:     tuple[int, int]
    confidence: float


def best_ball(fd: FrameDetections) -> Optional[BallDetection]:
    """Pick the single highest-confidence ball from a frame's detections.

    The unified YOLOv11m model emits ball detections as `Detection` objects in
    `fd.balls` (already gated by PlayerDetector.ball_conf). This adapter selects
    the best one and converts it to the `BallDetection` type the BallTracker /
    CarrierEngine consume. Returns None when no ball was detected this frame.
    """
    if not fd.balls:
        return None
    d = max(fd.balls, key=lambda x: x.confidence)
    return BallDetection(
        bbox       = d.bbox,
        centre     = d.centre_point,
        confidence = d.confidence,
    )


class BallTracker:
    """Constant-velocity Kalman around ball centre.

    Responsibilities:
      • Smooth the ball position frame to frame.
      • Coast through detection gaps for up to `coast_frames` frames.
      • Reject impossible-velocity false positives via a Mahalanobis-style gate.

    Notes:
      • Velocity is reported in *image pixels per frame*. It includes camera
        motion. Do NOT use it for kick / acceleration detection.
      • Pass detection in the FSM is purely geometric (ball-in/out of a
        bbox-relative foot zone), so the camera motion mixed into v_ball
        does not affect pass counts.

    `coast_frames`/`gate_sigma` default to today's futsal-tuned values; a
    ruleset config supplies its own values explicitly at construction time.
    """

    DETECTED = "detected"
    COASTING = "coasting"
    LOST     = "lost"

    def __init__(
        self,
        coast_frames: int   = 12,
        gate_sigma:   float = 6.0,
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
