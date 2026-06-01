"""
team_classifier/team_classifier.py — Unsupervised team classification

Pipeline:
  1. Sample one frame per second from the video
  2. Detect active_players using PlayerDetector
  3. Crop each player bounding box
  4. Pass all crops through SigLIP → high-dimensional embeddings
  5. UMAP: reduce (N, 768) → (N, 3)
  6. KMeans(k=2): partition into two teams
  7. Save fitted model to disk — reload on next run (no re-training)

On subsequent runs: loads from disk instantly, skips steps 1–6.

Import:
    from team_classifier.team_classifier import GSFATeamClassifier

Usage:
    from detectors.player_detector import PlayerDetector
    from team_classifier.team_classifier import GSFATeamClassifier

    player_det = PlayerDetector()
    team_clf   = GSFATeamClassifier()

    # Cold start: fits on video (~30s), saves pkl.
    # Warm start: loads pkl instantly.
    team_clf.fit_from_video_or_load(video_path, player_det)

    # Per-frame (inside any video loop):
    dets = player_det.detect(frame, frame_idx=fidx, fps=fps)
    team_clf.classify(frame, dets)       # mutates dets.players[i].team_id in-place

    for p in dets.players:
        print(p.team_id)   # 0 or 1
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import joblib
import numpy as np

from sports.common.team import TeamClassifier
from detectors.cache import cache_path

if TYPE_CHECKING:
    from detectors.player_detector import FrameDetections, PlayerDetector


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

MIN_CROP_PX   = 32          # crops smaller than this are skipped
TORSO_RATIO   = 0.55        # use top 55% of bbox (jersey region, excludes legs/court)
BLUR_THRESHOLD = 80         # Laplacian variance below this → blurry crop, skip during fitting


# ---------------------------------------------------------------------------
# MAIN CLASS
# ---------------------------------------------------------------------------

class GSFATeamClassifier:
    """
    Wraps sports.common.team.TeamClassifier (SigLIP + UMAP + KMeans).

    Fits once per video, persists to disk, reloads on next run.
    Assigns team_id (0 or 1) to each active_player Detection in-place.
    Referees and goal_posts are left with team_id = None.
    """

    def __init__(
        self,
        device:     str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self._classifier  = TeamClassifier(device=device, batch_size=batch_size)
        self._is_fitted   = False
        self.device       = device
        self.batch_size   = batch_size

    # ------------------------------------------------------------------
    # Internal crop helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _torso_crop(frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """Return top TORSO_RATIO of the bounding box — jersey only, no legs."""
        x1, y1, x2, y2 = bbox
        torso_y2 = y1 + int((y2 - y1) * TORSO_RATIO)
        return frame[y1:torso_y2, x1:x2]

    @staticmethod
    def _is_sharp(crop: np.ndarray, threshold: float = BLUR_THRESHOLD) -> bool:
        """True if crop is sharp enough to contribute useful signal to fitting."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var()) > threshold

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit_from_video(
        self,
        video_path:    str,
        player_det:    "PlayerDetector",
        sample_every:  int = 30,        # 30 → 1 fps at 30 fps video
        save_path:     Path | None = None,
        progress:      bool = True,
    ) -> None:
        """
        Collect player crops from the video (1 fps), fit SigLIP+UMAP+KMeans,
        and save the fitted classifier to disk.

        Args:
            video_path:   Path to input video.
            player_det:   Already-instantiated PlayerDetector (model loaded once).
            sample_every: Run detection every Nth frame (default 30 → 1 fps).
            save_path:    Where to pickle the fitted model.
            progress:     Print progress during crop collection.
        """
        if save_path is None:
            save_path = cache_path(video_path, "team_siglip")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"GSFATeamClassifier: cannot open video: {video_path}")

        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
        crops: list[np.ndarray] = []
        fidx  = 0

        if progress:
            print("[TeamClassifier] Collecting player crops …")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if fidx % sample_every == 0:
                dets = player_det.detect(frame, frame_idx=fidx, fps=fps)
                for p in dets.players:
                    crop = self._torso_crop(frame, p.bbox)
                    if crop.shape[0] >= MIN_CROP_PX and crop.shape[1] >= MIN_CROP_PX:
                        if self._is_sharp(crop):
                            crops.append(crop)

                if progress and (fidx // sample_every) % 100 == 0:
                    pct = fidx / max(1, total_f) * 100
                    print(f"  frame {fidx:>5}/{total_f} ({pct:.0f}%)  "
                          f"crops collected: {len(crops)}")

            fidx += 1

        cap.release()

        if not crops:
            raise RuntimeError(
                "GSFATeamClassifier.fit_from_video: no valid player crops found. "
                "Check PlayerDetector confidence threshold or video path."
            )

        if progress:
            print(f"[TeamClassifier] Fitting on {len(crops)} crops "
                  f"(SigLIP → UMAP → KMeans) …")

        self._classifier.fit(crops)
        self._is_fitted = True

        self.save(save_path)
        if progress:
            print(f"[TeamClassifier] Fitted and saved → {save_path}")

    def fit_from_video_or_load(
        self,
        video_path:   str,
        player_det:   "PlayerDetector",
        save_path:    Path | None = None,
        sample_every: int  = 30,
        progress:     bool = True,
        force_refit:  bool = False,
    ) -> None:
        """
        Load from disk if a saved model exists; otherwise fit from video and save.
        save_path defaults to data/cache/<video_stem>_team_siglip.pkl — one file per match.
        Pass force_refit=True to ignore an existing pkl and refit from scratch.
        """
        if save_path is None:
            save_path = cache_path(video_path, "team_siglip")
        if not force_refit and Path(save_path).exists():
            loaded = GSFATeamClassifier.load(save_path, progress=progress)
            self._classifier = loaded._classifier
            self._is_fitted  = loaded._is_fitted
        else:
            self.fit_from_video(
                video_path   = video_path,
                player_det   = player_det,
                sample_every = sample_every,
                save_path    = save_path,
                progress     = progress,
            )

    # ------------------------------------------------------------------
    # Classify
    # ------------------------------------------------------------------

    def classify(
        self,
        frame:       np.ndarray,
        detections:  "FrameDetections",
    ) -> None:
        """
        Assign team_id (0 or 1) to every active_player Detection in-place.
        Referees and goal_posts are unchanged (team_id stays None).

        Args:
            frame:      BGR frame the detections came from.
            detections: FrameDetections returned by PlayerDetector.detect().
        """
        if not self._is_fitted:
            raise RuntimeError(
                "GSFATeamClassifier is not fitted. "
                "Call fit_from_video_or_load() first."
            )

        if not detections.players:
            return

        # Crop all players in this frame
        crops: list[np.ndarray] = []
        valid_indices: list[int] = []

        for i, p in enumerate(detections.players):
            crop = self._torso_crop(frame, p.bbox)
            if crop.shape[0] >= MIN_CROP_PX and crop.shape[1] >= MIN_CROP_PX:
                crops.append(crop)
                valid_indices.append(i)

        if not crops:
            return

        # Split predict() into its three stages so we keep the 768-D SigLIP
        # features for the player tracker (otherwise discarded after UMAP).
        features    = self._classifier.extract_features(crops)        # (N, 768)
        projections = self._classifier.reducer.transform(features)
        team_ids    = self._classifier.cluster_model.predict(projections)

        for list_pos, det_idx in enumerate(valid_indices):
            detections.players[det_idx].team_id   = int(team_ids[list_pos])
            detections.players[det_idx].embedding = features[list_pos].astype(np.float32)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> None:
        """Serialize the fitted classifier to disk using joblib.

        If path is None, raises ValueError — callers must supply an explicit path
        (fit_from_video and fit_from_video_or_load always resolve it before calling save).
        """
        if path is None:
            raise ValueError(
                "GSFATeamClassifier.save() requires an explicit path. "
                "Use fit_from_video_or_load() which resolves the path automatically."
            )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(
        path:     Path | None = None,
        progress: bool = True,
    ) -> "GSFATeamClassifier":
        """
        Load a previously fitted GSFATeamClassifier from disk.

        Returns the loaded instance and also updates the caller's object
        when used as: clf.load(path) — but the idiomatic use is:
            clf = GSFATeamClassifier.load(path)
        or via fit_from_video_or_load() which calls this automatically.
        """
        if path is None:
            raise ValueError(
                "GSFATeamClassifier.load() requires an explicit path. "
                "Use fit_from_video_or_load() which resolves the path automatically."
            )
        if progress:
            print(f"[TeamClassifier] Loading from {path} …")
        obj = joblib.load(path)
        if progress:
            print("[TeamClassifier] Loaded.")
        return obj

    # ------------------------------------------------------------------
    # Visualisation helper
    # ------------------------------------------------------------------

    @staticmethod
    def draw(
        frame:      np.ndarray,
        detections: "FrameDetections",
    ) -> np.ndarray:
        """
        Draw player bboxes coloured by team_id on a copy of the frame.
            team_id = 0 → blue
            team_id = 1 → red
            team_id = None → grey (unclassified)
        """
        TEAM_COLOURS = {
            0:    (255, 80, 0),    # blue  — team 0
            1:    (0,   80, 255),  # red   — team 1
            None: (160, 160, 160), # grey  — unclassified
        }
        out = frame.copy()
        for p in detections.players:
            x1, y1, x2, y2 = p.bbox
            colour = TEAM_COLOURS[p.team_id]
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            label = f"T{p.team_id}" if p.team_id is not None else "?"
            cv2.putText(out, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
        return out
