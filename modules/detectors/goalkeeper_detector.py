"""
modules/detectors/goalkeeper_detector.py — Goalkeeper classification by jersey colour

No fit stage, no goal-post dependency, no tracking. A player IS the
goalkeeper because their jersey colour matches team_a_gk_colour or
team_b_gk_colour — the reference colours are known constants (supplied by the
caller), so there is nothing to learn from video.

Runs independently of GSFATeamClassifier ("in parallel"): both classifiers
read the same frame + detections and write to the same Detection objects,
but neither depends on the other's output.

Per-frame classify():
    For each of the two reference colours, find the single player in the
    frame whose jersey colour is closest to it. If that distance is under
    max_colour_dist, mark that player is_goalkeeper=True, team_id=<that
    team>. No match under threshold ⇒ no GK flagged for that team this
    frame (a bad/no match is not forced onto the "least bad" player).

Import:
    from modules.detectors.goalkeeper_detector import GoalkeeperDetector

Usage:
    from modules.detectors.player_detector import PlayerDetector
    from modules.team_classifier.team_classifier import GSFATeamClassifier
    from modules.detectors.goalkeeper_detector import GoalkeeperDetector

    player_det = PlayerDetector()
    team_clf   = GSFATeamClassifier()
    team_clf.fit_from_video_or_load(video_path, player_det)

    gk_det = GoalkeeperDetector(team_a_gk_colour="#00FF00", team_b_gk_colour="black")

    # Per-frame:
    dets = player_det.detect(frame, frame_idx=fidx, fps=fps)
    team_clf.classify(frame, dets)   # independent of gk_det
    gk_det.classify(frame, dets)     # independent of team_clf; marks is_goalkeeper + team_id

    for p in dets.players:
        if p.is_goalkeeper:
            print(f"GK — team {p.team_id}")
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from modules.team_classifier.team_classifier import (
    CENTRE_CROP_RATIO,
    TORSO_RATIO,
    GSFATeamClassifier,
)

if TYPE_CHECKING:
    from modules.detectors.player_detector import FrameDetections


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Max Euclidean distance, in GSFATeamClassifier's cylindrical HSV colour
# space (_hsv_vec: [sat*cos(hue), sat*sin(hue), val], sat/val in 0-255), for
# a player's jersey colour to count as a match to a GK reference colour.
# First-pass placeholder — needs calibration against real footage, same as
# BLUR_THRESHOLD in team_classifier/team_classifier.py.
MAX_GK_COLOUR_DIST = 60.0


# ---------------------------------------------------------------------------
# GOALKEEPER DETECTOR
# ---------------------------------------------------------------------------

class GoalkeeperDetector:
    """
    Classifies goalkeepers by direct jersey-colour match against two known
    reference colours. No fit step — construct once per match and classify
    from frame 1.
    """

    def __init__(
        self,
        team_a_gk_colour: str,
        team_b_gk_colour: str,
        max_colour_dist: float = MAX_GK_COLOUR_DIST,
        *,
        torso_ratio: float = TORSO_RATIO,
        centre_crop_ratio: float = CENTRE_CROP_RATIO,
    ) -> None:
        self.max_colour_dist = max_colour_dist
        # Same camera-framing tuning as GSFATeamClassifier — kept independent
        # (not read from a GSFATeamClassifier instance) since this detector
        # deliberately runs without depending on one; a ruleset config
        # supplies matching values to both.
        self.torso_ratio = torso_ratio
        self.centre_crop_ratio = centre_crop_ratio
        self._ref_vec: list[np.ndarray] = [
            GSFATeamClassifier._hsv_vec(*GSFATeamClassifier._colour_to_hsv(team_a_gk_colour)),
            GSFATeamClassifier._hsv_vec(*GSFATeamClassifier._colour_to_hsv(team_b_gk_colour)),
        ]

    # ------------------------------------------------------------------
    # Per-frame classify
    # ------------------------------------------------------------------

    def classify(self, frame: np.ndarray, detections: "FrameDetections") -> None:
        """
        For each reference colour (team 0, team 1), find the single closest-
        matching player in this frame and — if within max_colour_dist — mark
        it is_goalkeeper=True, team_id=<that team>.

        Modifies detections.players in-place. Independent of GSFATeamClassifier
        (does not read or require p.team_id / p.embedding from an outfield
        classifier having already run).
        """
        for p in detections.players:
            p.is_goalkeeper = False

        if not detections.players:
            return

        # Colour vector per player (None if crop is empty/unusable).
        player_vecs: list[np.ndarray | None] = []
        for p in detections.players:
            crop = GSFATeamClassifier._torso_crop(frame, p.bbox, self.torso_ratio)
            if crop.shape[0] > 0 and crop.shape[1] > 0:
                player_vecs.append(
                    GSFATeamClassifier._mean_colour_vec([crop], self.centre_crop_ratio)
                )
            else:
                player_vecs.append(None)

        for team_id in (0, 1):
            ref = self._ref_vec[team_id]
            best_idx: int | None = None
            best_dist = math.inf
            for i, vec in enumerate(player_vecs):
                if vec is None:
                    continue
                dist = float(np.linalg.norm(vec - ref))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx is not None and best_dist < self.max_colour_dist:
                detections.players[best_idx].is_goalkeeper = True
                detections.players[best_idx].team_id = team_id

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    @staticmethod
    def draw(
        frame:      np.ndarray,
        detections: "FrameDetections",
    ) -> np.ndarray:
        """
        Draw players with GK highlighted in yellow.
        Non-GK players use team colours (blue=0, red=1).
        """
        import cv2

        TEAM_COLOURS = {
            0:    (255, 80,   0),
            1:    (0,   80, 255),
            None: (160, 160, 160),
        }
        GK_COLOUR = (0, 215, 255)   # yellow

        out = frame.copy()
        for p in detections.players:
            x1, y1, x2, y2 = p.bbox
            if p.is_goalkeeper:
                colour = GK_COLOUR
                label  = f"GK-T{p.team_id}"
                cv2.rectangle(out, (x1, y1), (x2, y2), colour, 3)
            else:
                colour = TEAM_COLOURS[p.team_id]
                label  = f"T{p.team_id}" if p.team_id is not None else "?"
                cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(out, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
        return out
