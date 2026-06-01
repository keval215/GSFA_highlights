"""
detectors/goalkeeper_detector.py — Goalkeeper detection and team assignment

Two-stage pipeline:
  Stage 1 — Fit (accumulated over video):
    Sample frames → for each goal post, find the closest active_player.
    Average those positions → 2 GK zone centroids (one per goal).

  Stage 2 — Team assignment (during fit):
    Compute centroid of team 0 and team 1 outfield players across frames.
    Assign each GK zone to the team whose centroid is closer.

  Per-frame classify:
    For each detected goal post → find closest active_player →
    mark is_goalkeeper=True, set team_id from fitted assignment.

Cache: one pkl per match (data/cache/<video_stem>_goalkeeper.pkl).

Import:
    from detectors.goalkeeper_detector import GoalkeeperDetector

Usage:
    from detectors.player_detector import PlayerDetector
    from detectors.goalkeeper_detector import GoalkeeperDetector
    from team_classifier.colour_histogram import ColourHistogramTeamClassifier

    player_det = PlayerDetector()
    team_clf   = ColourHistogramTeamClassifier()
    team_clf.fit_from_video_or_load(video_path, player_det)

    gk_det = GoalkeeperDetector()
    gk_det.fit_from_video_or_load(video_path, player_det, team_clf)

    # Per-frame:
    dets = player_det.detect(frame, frame_idx=fidx, fps=fps)
    team_clf.classify(frame, dets)
    gk_det.classify(dets)           # marks is_goalkeeper + corrects team_id

    for p in dets.players:
        if p.is_goalkeeper:
            print(f"GK — team {p.team_id}")
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import joblib
import numpy as np

from detectors.cache import cache_path

if TYPE_CHECKING:
    from detectors.player_detector import Detection, FrameDetections, PlayerDetector


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _dist(a: tuple, b: tuple) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _centroid(points: list[tuple]) -> tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    return (
        float(np.mean([p[0] for p in points])),
        float(np.mean([p[1] for p in points])),
    )


# ---------------------------------------------------------------------------
# GOALKEEPER DETECTOR
# ---------------------------------------------------------------------------

class GoalkeeperDetector:
    """
    Identifies the two goalkeepers and assigns each to a team.

    Fit once per match (or load from cache). Classify per frame.
    Does NOT require tracking — works on raw per-frame detections.
    """

    def __init__(self) -> None:
        self._gk_zones: list[tuple[float, float]] | None = None
        # avg pixel position (cx, cy) of GK near each goal post
        # index 0 = left goal post, index 1 = right goal post (sorted by x)

        self._gk_teams: list[int] | None = None
        # team_id assigned to each GK zone [team_for_left_post, team_for_right_post]

        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit_from_video(
        self,
        video_path:   str,
        player_det:   "PlayerDetector",
        team_clf,                          # GSFATeamClassifier or ColourHistogramTeamClassifier
        sample_every: int       = 30,
        save_path:    Path|None = None,
        progress:     bool      = True,
    ) -> None:
        """
        Scan video at 1fps, accumulate closest-player-to-each-post positions
        and team centroids, compute GK zones and team assignments, save pkl.
        """
        if save_path is None:
            save_path = cache_path(video_path, "goalkeeper")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"GoalkeeperDetector: cannot open {video_path}")

        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Accumulators
        post_nearest: list[list[tuple]] = [[], []]  # [left_post_positions, right_post_positions]
        team_positions: dict[int, list[tuple]] = {0: [], 1: []}
        fidx = 0

        if progress:
            print("[GoalkeeperDetector] Scanning video for GK zones …")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if fidx % sample_every == 0:
                dets = player_det.detect(frame, frame_idx=fidx, fps=fps)
                team_clf.classify(frame, dets)

                # Need at least 1 goal post and 2 players
                if dets.goal_posts and len(dets.players) >= 2:

                    # Sort posts left → right by x pixel position
                    posts = sorted(dets.goal_posts, key=lambda p: p.foot_point[0])

                    for post_idx, post in enumerate(posts[:2]):
                        # Player closest to this goal post
                        closest = min(
                            dets.players,
                            key=lambda p: _dist(p.foot_point, post.foot_point),
                        )
                        post_nearest[post_idx].append(closest.foot_point)

                    # Accumulate outfield team positions (all classified players)
                    for p in dets.players:
                        if p.team_id in (0, 1):
                            team_positions[p.team_id].append(p.foot_point)

                if progress and (fidx // sample_every) % 100 == 0:
                    pct = fidx / max(1, total_f) * 100
                    print(f"  frame {fidx:>5}/{total_f} ({pct:.0f}%)  "
                          f"post0={len(post_nearest[0])}  post1={len(post_nearest[1])}")

            fidx += 1

        cap.release()

        if not post_nearest[0] and not post_nearest[1]:
            raise RuntimeError(
                "GoalkeeperDetector: no goal posts detected during fit. "
                "Check PlayerDetector or video path."
            )

        # GK zone centroids (avg pixel position of closest player per post)
        self._gk_zones = [
            _centroid(post_nearest[0]) if post_nearest[0] else (0.0, 0.0),
            _centroid(post_nearest[1]) if post_nearest[1] else (0.0, 0.0),
        ]

        # Team centroids
        t0_centroid = _centroid(team_positions[0])
        t1_centroid = _centroid(team_positions[1])

        # Assign each GK zone to closer team
        self._gk_teams = []
        for zone in self._gk_zones:
            d0 = _dist(zone, t0_centroid)
            d1 = _dist(zone, t1_centroid)
            self._gk_teams.append(0 if d0 <= d1 else 1)

        self._is_fitted = True

        if progress:
            print(f"[GoalkeeperDetector] GK zone 0 (left post)  → "
                  f"pixel ({self._gk_zones[0][0]:.0f}, {self._gk_zones[0][1]:.0f})  "
                  f"→ team {self._gk_teams[0]}")
            print(f"[GoalkeeperDetector] GK zone 1 (right post) → "
                  f"pixel ({self._gk_zones[1][0]:.0f}, {self._gk_zones[1][1]:.0f})  "
                  f"→ team {self._gk_teams[1]}")

        self.save(save_path)
        if progress:
            print(f"[GoalkeeperDetector] Saved → {save_path}")

    def fit_from_video_or_load(
        self,
        video_path:   str,
        player_det:   "PlayerDetector",
        team_clf,
        sample_every: int       = 30,
        save_path:    Path|None = None,
        progress:     bool      = True,
        force_refit:  bool      = False,
    ) -> None:
        """Load from cache if exists; otherwise fit and save. One pkl per match."""
        if save_path is None:
            save_path = cache_path(video_path, "goalkeeper")

        if not force_refit and Path(save_path).exists():
            loaded = GoalkeeperDetector.load(save_path, progress=progress)
            self._gk_zones  = loaded._gk_zones
            self._gk_teams  = loaded._gk_teams
            self._is_fitted = loaded._is_fitted
        else:
            self.fit_from_video(
                video_path   = video_path,
                player_det   = player_det,
                team_clf     = team_clf,
                sample_every = sample_every,
                save_path    = save_path,
                progress     = progress,
            )

    # ------------------------------------------------------------------
    # Per-frame classify
    # ------------------------------------------------------------------

    def classify(self, detections: "FrameDetections") -> None:
        """
        For each detected goal post: find the closest active_player,
        mark is_goalkeeper=True, override team_id with the fitted assignment.

        Modifies detections.players in-place.
        Requires team_clf.classify() to have already run on the same detections.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "GoalkeeperDetector not fitted. "
                "Call fit_from_video_or_load() first."
            )

        # Reset goalkeeper flags
        for p in detections.players:
            p.is_goalkeeper = False

        if not detections.goal_posts or not detections.players:
            return

        # Sort posts left → right (consistent with fit phase)
        posts = sorted(detections.goal_posts, key=lambda p: p.foot_point[0])

        for post_idx, post in enumerate(posts[:2]):
            closest: "Detection" = min(
                detections.players,
                key=lambda p: _dist(p.foot_point, post.foot_point),
            )
            closest.is_goalkeeper = True
            closest.team_id = self._gk_teams[post_idx]

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(
        path:     Path,
        progress: bool = True,
    ) -> "GoalkeeperDetector":
        if progress:
            print(f"[GoalkeeperDetector] Loading from {path} …")
        obj = joblib.load(path)
        if progress:
            print("[GoalkeeperDetector] Loaded.")
        return obj

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
