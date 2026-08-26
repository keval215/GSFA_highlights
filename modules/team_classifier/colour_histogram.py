"""
modules/team_classifier/colour_histogram.py — Colour histogram team classification

Pipeline:
  torso crops → HSV histogram (H: 64 bins + S: 32 bins) → (N, 96)
  → KMeans(k=2) directly on raw histograms
  → team_id 0 or 1

  PCA(n_components=2) used for reporting only (scatter plot saved to disk).
  No UMAP — KMeans runs directly on histogram features for cleaner separation.

Import:
    from modules.team_classifier.colour_histogram import ColourHistogramTeamClassifier

Usage:
    clf = ColourHistogramTeamClassifier()
    clf.fit_from_video_or_load(video_path, player_det)

    dets = player_det.detect(frame, frame_idx=fidx, fps=fps)
    clf.classify(frame, dets)

    for p in dets.players:
        print(p.team_id)   # 0 or 1
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from modules.detectors.cache import cache_path

if TYPE_CHECKING:
    from modules.detectors.player_detector import FrameDetections, PlayerDetector


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

N_BINS_H    = 64    # hue histogram bins        (H range 0–180 in OpenCV)
N_BINS_S    = 32    # saturation histogram bins  (S range 0–255)
MIN_CROP_PX = 32    # minimum crop dimension in px
TORSO_RATIO = 0.55  # use top 55% of bbox — jersey only, no legs
PCA_PLOT    = Path(r"D:\GSFA_highlights\data\debug\team_classifier_pca.png")


# ---------------------------------------------------------------------------
# MAIN CLASS
# ---------------------------------------------------------------------------

class ColourHistogramTeamClassifier:
    """
    Unsupervised team classifier: HSV histogram → KMeans(k=2).
    Drop-in replacement for GSFATeamClassifier — identical public API.
    """

    def __init__(
        self,
        n_bins_h: int = N_BINS_H,
        n_bins_s: int = N_BINS_S,
    ) -> None:
        self.n_bins_h   = n_bins_h
        self.n_bins_s   = n_bins_s
        self._kmeans    = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _torso_crop(frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """Top TORSO_RATIO of bbox — jersey region, excludes legs/court."""
        x1, y1, x2, y2 = bbox
        torso_y2 = y1 + int((y2 - y1) * TORSO_RATIO)
        return frame[y1:torso_y2, x1:x2]

    def _extract_features(self, crop: np.ndarray) -> np.ndarray:
        """
        BGR torso crop → 96-dim L1-normalised HSV histogram.
        H (64 bins) captures jersey hue.
        S (32 bins) handles white/grey jerseys (low saturation).
        """
        hsv    = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_hist = np.histogram(hsv[:, :, 0], bins=self.n_bins_h, range=(0, 180))[0]
        s_hist = np.histogram(hsv[:, :, 1], bins=self.n_bins_s, range=(0, 256))[0]
        feat   = np.concatenate([h_hist, s_hist]).astype(np.float32)
        total  = feat.sum()
        if total > 0:
            feat /= total   # L1 normalise — brightness/size invariant
        return feat

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit_from_video(
        self,
        video_path:   str,
        player_det:   "PlayerDetector",
        sample_every: int       = 30,
        save_path:    Path|None = None,
        progress:     bool      = True,
    ) -> None:
        """
        Sample 1fps from video, collect torso crops, extract HSV histograms,
        fit KMeans(k=2) directly, save PCA scatter plot + pkl.
        """
        if save_path is None:
            save_path = cache_path(video_path, "team_colour")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"ColourHistogramTeamClassifier: cannot open {video_path}")

        total_f  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        features: list[np.ndarray] = []
        fidx = 0

        if progress:
            print("[ColourHistogram] Collecting player crops …")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fidx % sample_every == 0:
                dets = player_det.detect(frame, frame_idx=fidx, fps=fps)
                for p in dets.players:
                    crop = self._torso_crop(frame, p.bbox)
                    if crop.shape[0] >= MIN_CROP_PX and crop.shape[1] >= MIN_CROP_PX:
                        features.append(self._extract_features(crop))

                if progress and (fidx // sample_every) % 100 == 0:
                    pct = fidx / max(1, total_f) * 100
                    print(f"  frame {fidx:>5}/{total_f} ({pct:.0f}%)  "
                          f"features: {len(features)}")
            fidx += 1

        cap.release()

        if len(features) < 4:
            raise RuntimeError(
                "ColourHistogramTeamClassifier: fewer than 4 crops collected."
            )

        X = np.stack(features)   # (N, 96)

        if progress:
            print(f"[ColourHistogram] Fitting KMeans(k=2) directly on "
                  f"{len(features)} histograms (no UMAP) …")

        self._kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = self._kmeans.fit_predict(X)
        self._is_fitted = True

        self._save_pca_report(X, labels, progress=progress)
        self.save(save_path)

        if progress:
            print(f"[ColourHistogram] Fitted and saved → {save_path}")

    def _dominant_hue_colour(self, team_id: int):
        """
        Return (hue_degrees, matplotlib_hex) for a cluster using its KMeans centroid.
        Hue is in OpenCV range (0-180); converted to a full-saturation HSV colour.
        """
        import colorsys
        centroid   = self._kmeans.cluster_centers_[team_id]
        h_hist     = centroid[:self.n_bins_h]
        peak_bin   = int(np.argmax(h_hist))
        # map bin index → OpenCV hue (0–180)
        hue_ocv    = peak_bin * 180.0 / self.n_bins_h
        # convert to matplotlib colour: colorsys expects hue in 0–1
        r, g, b    = colorsys.hsv_to_rgb(hue_ocv / 180.0, 0.9, 0.95)
        return hue_ocv, "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

    def _save_pca_report(
        self,
        X:        np.ndarray,
        labels:   np.ndarray,
        progress: bool = True,
    ) -> None:
        """
        PCA(n_components=2) on histogram features → scatter plot PNG.
        Dot colour = actual dominant hue detected per cluster from the HSV hue wheel.
        Two distinct blobs = good colour separation.
        Overlapping blobs = jerseys too similar for histogram-based clustering.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            if progress:
                print("[ColourHistogram] matplotlib not found — skipping PCA plot")
            return

        pca  = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        var  = pca.explained_variance_ratio_

        # derive actual hue colours from centroids
        team_hues   = {}
        team_hexes  = {}
        for team in [0, 1]:
            hue_ocv, hex_col    = self._dominant_hue_colour(team)
            team_hues[team]     = hue_ocv
            team_hexes[team]    = hex_col

        if progress:
            for team in [0, 1]:
                print(f"  Cluster {team} dominant hue: {team_hues[team]:.1f}° (OpenCV)  "
                      f"→ colour {team_hexes[team]}")

        fig, ax = plt.subplots(figsize=(8, 6))
        for team in [0, 1]:
            mask = labels == team
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=team_hexes[team],
                label=(f"Team {team}  ({mask.sum()} crops)  "
                       f"hue={team_hues[team]:.0f}°"),
                alpha=0.55, s=18, edgecolors="none",
            )

        ax.set_title(
            f"PCA of HSV histograms — KMeans(k=2)  [no UMAP]\n"
            f"PC1 {var[0]*100:.1f}%  +  PC2 {var[1]*100:.1f}% variance explained\n"
            f"KMeans inertia: {self._kmeans.inertia_:.1f}  |  "
            f"Total crops: {len(labels)}\n"
            f"Dot colour = detected jersey hue from HSV hue wheel"
        )
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        PCA_PLOT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(PCA_PLOT), dpi=120, bbox_inches="tight")
        plt.close(fig)

        n0 = int((labels == 0).sum())
        n1 = int((labels == 1).sum())
        if progress:
            print(f"[ColourHistogram] PCA report → {PCA_PLOT}")
            print(f"  Cluster 0: {n0} crops   Cluster 1: {n1} crops")
            print(f"  KMeans inertia : {self._kmeans.inertia_:.1f}  "
                  f"(lower = tighter clusters)")
            print(f"  PCA variance   : PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%")
            print(f"  → Open {PCA_PLOT.name} — two distinct blobs = good separation")

    def fit_from_video_or_load(
        self,
        video_path:   str,
        player_det:   "PlayerDetector",
        save_path:    Path | None = None,
        sample_every: int  = 30,
        progress:     bool = True,
        force_refit:  bool = False,
    ) -> None:
        """Load from disk if exists; otherwise fit from video and save.
        save_path defaults to data/cache/<video_stem>_team_colour.pkl — one file per match."""
        if save_path is None:
            save_path = cache_path(video_path, "team_colour")
        if not force_refit and Path(save_path).exists():
            loaded = ColourHistogramTeamClassifier.load(save_path, progress=progress)
            self._kmeans    = loaded._kmeans
            self._is_fitted = loaded._is_fitted
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
        frame:      np.ndarray,
        detections: "FrameDetections",
    ) -> None:
        """
        Assign team_id (0 or 1) to each active_player Detection in-place.
        Referees and goal_posts unchanged (team_id stays None).
        """
        if not self._is_fitted:
            raise RuntimeError(
                "ColourHistogramTeamClassifier not fitted. "
                "Call fit_from_video_or_load() first."
            )
        if not detections.players:
            return

        feats:     list[np.ndarray] = []
        valid_idx: list[int]        = []

        for i, p in enumerate(detections.players):
            crop = self._torso_crop(frame, p.bbox)
            if crop.shape[0] >= MIN_CROP_PX and crop.shape[1] >= MIN_CROP_PX:
                feats.append(self._extract_features(crop))
                valid_idx.append(i)

        if not feats:
            return

        X        = np.stack(feats)
        team_ids = self._kmeans.predict(X)   # direct KMeans — no UMAP step

        for list_pos, det_idx in enumerate(valid_idx):
            detections.players[det_idx].team_id = int(team_ids[list_pos])

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> None:
        if path is None:
            raise ValueError("save() requires an explicit path — use fit_from_video_or_load() which sets it automatically.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(
        path:     Path,
        progress: bool = True,
    ) -> "ColourHistogramTeamClassifier":
        if progress:
            print(f"[ColourHistogram] Loading from {path} …")
        obj = joblib.load(path)
        if progress:
            print("[ColourHistogram] Loaded.")
        return obj

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    @staticmethod
    def draw(
        frame:      np.ndarray,
        detections: "FrameDetections",
    ) -> np.ndarray:
        """Blue = team 0, Red = team 1, Grey = unclassified."""
        TEAM_COLOURS = {
            0:    (255, 80,   0),
            1:    (0,   80, 255),
            None: (160, 160, 160),
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
