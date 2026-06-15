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

from dataclasses import dataclass
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

# --- Colour → team-name resolution (runs once per match, at fit time) -------
HUE_SAMPLE_PER_CLUSTER = 30   # crops sampled per cluster to estimate jersey colour
CENTRE_CROP_RATIO      = 0.50 # use the central 50% of each torso crop (jersey core)
RESOLVE_SAMPLE_SEED    = 0    # deterministic sampling so the mapping is reproducible

# A small CSS-style name table so callers may send "orange" instead of "#FF6600".
NAMED_COLOURS: dict[str, str] = {
    "white":   "#FFFFFF", "black":  "#000000", "grey":   "#808080", "gray": "#808080",
    "silver":  "#C0C0C0", "red":    "#FF0000", "maroon": "#800000", "orange": "#FFA500",
    "gold":    "#FFD700", "yellow": "#FFFF00", "lime":   "#00FF00", "green":  "#008000",
    "teal":    "#008080", "cyan":   "#00FFFF", "aqua":   "#00FFFF", "blue":   "#0000FF",
    "navy":    "#000080", "sky":    "#87CEEB", "skyblue":"#87CEEB", "purple": "#800080",
    "magenta": "#FF00FF", "pink":   "#FFC0CB", "brown":  "#A52A2A",
}


@dataclass
class TeamSpec:
    """One team as supplied by the caller: a display name and a jersey colour.
    `colour` is a hex string ("#FF6600" / "FF6600" / "#F60") or a CSS-style
    name from NAMED_COLOURS ("orange")."""
    name:   str
    colour: str


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
        # cluster id (0/1) → caller-supplied team name; set once by
        # resolve_team_names() at fit time, then persisted with the pickle.
        # None ⇒ caller gave no colours; downstream falls back to team0/team1.
        self.team_id_to_name: dict[int, str] | None = None

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

    def classify_batch(
        self,
        frames:          list[np.ndarray],
        detections_list: list["FrameDetections"],
    ) -> None:
        """Batched form of classify(): embed the player crops from MANY frames
        in one SigLIP pass, then scatter team_id + embedding back onto each
        Detection. Result is identical to calling classify() per frame (same
        crops, same reducer, same KMeans) — only fewer, larger GPU calls.

        frames[i] is the BGR frame that detections_list[i] came from.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "GSFATeamClassifier is not fitted. "
                "Call fit_from_video_or_load() first."
            )

        # One flat crop list across all frames, with a parallel index map back
        # to (frame position, detection index). Same crop filters as classify().
        crops: list[np.ndarray] = []
        index_map: list[tuple[int, int]] = []
        for fpos, dets in enumerate(detections_list):
            frame = frames[fpos]
            for det_idx, p in enumerate(dets.players):
                crop = self._torso_crop(frame, p.bbox)
                if crop.shape[0] >= MIN_CROP_PX and crop.shape[1] >= MIN_CROP_PX:
                    crops.append(crop)
                    index_map.append((fpos, det_idx))

        if not crops:
            return

        features    = self._classifier.extract_features(crops)        # (N, 768)
        projections = self._classifier.reducer.transform(features)
        team_ids    = self._classifier.cluster_model.predict(projections)

        for list_pos, (fpos, det_idx) in enumerate(index_map):
            detections_list[fpos].players[det_idx].team_id   = int(team_ids[list_pos])
            detections_list[fpos].players[det_idx].embedding = features[list_pos].astype(np.float32)

    # ------------------------------------------------------------------
    # Colour → team-name resolution (called ONCE per match, at fit time)
    # ------------------------------------------------------------------

    def resolve_team_names(
        self,
        specs: list["TeamSpec"],
        crops: list[np.ndarray],
    ) -> dict[int, str]:
        """Map cluster id (0/1) → caller-supplied team name by jersey colour.

        KMeans is fitted once per match, so cluster identity is fixed for the
        whole match; this resolution therefore runs a single time (clip 1) and
        the result is stored on the instance (pickled with it). Per-clip work
        is then an O(1) dict lookup — no colour computation is ever repeated.

        Algorithm:
          1. Re-predict cluster labels for every fit crop (index-aligned).
          2. Per cluster, sample up to HUE_SAMPLE_PER_CLUSTER crops, take the
             central CENTRE_CROP_RATIO (jersey core, less skin/background) and
             compute a saturation-weighted mean colour in HSV.
          3. Match clusters to specs by minimum total colour distance over the
             two possible 2×2 assignments (guarantees a bijection — no two
             clusters collapse onto the same team).

        Args:
            specs: exactly two TeamSpec (name + colour).
            crops: the torso crops the classifier was fitted on.

        Returns:
            The {cluster_id: name} mapping (also stored as self.team_id_to_name).
        """
        if not self._is_fitted:
            raise RuntimeError("resolve_team_names: classifier is not fitted.")
        if len(specs) != 2:
            raise ValueError(f"resolve_team_names expects exactly 2 specs, got {len(specs)}.")
        if not crops:
            raise ValueError("resolve_team_names: no crops supplied.")

        features    = self._classifier.extract_features(crops)
        projections = self._classifier.reducer.transform(features)
        labels      = self._classifier.cluster_model.predict(projections)

        rng = np.random.default_rng(RESOLVE_SAMPLE_SEED)
        cluster_vec: dict[int, np.ndarray | None] = {}
        for cid in (0, 1):
            idx = np.flatnonzero(labels == cid)
            if idx.size > HUE_SAMPLE_PER_CLUSTER:
                idx = rng.choice(idx, HUE_SAMPLE_PER_CLUSTER, replace=False)
            sampled = [crops[i] for i in idx.tolist()]
            cluster_vec[cid] = self._mean_colour_vec(sampled)

        spec_vec = [self._hsv_vec(*self._colour_to_hsv(s.colour)) for s in specs]

        # Two possible bijections between {cluster 0,1} and {spec 0,1}.
        def total_cost(mapping: dict[int, int]) -> float:
            cost = 0.0
            for cid, sidx in mapping.items():
                cvec = cluster_vec[cid]
                # A cluster with no usable colour signal: large fixed penalty so
                # the other (informative) assignment decides the orientation.
                cost += 1e9 if cvec is None else float(np.linalg.norm(cvec - spec_vec[sidx]))
            return cost

        straight = {0: 0, 1: 1}
        swapped  = {0: 1, 1: 0}
        chosen   = straight if total_cost(straight) <= total_cost(swapped) else swapped

        self.team_id_to_name = {cid: specs[sidx].name for cid, sidx in chosen.items()}
        return self.team_id_to_name

    # ---- colour helpers ------------------------------------------------

    @staticmethod
    def _centre_crop(crop: np.ndarray) -> np.ndarray:
        """Central CENTRE_CROP_RATIO box of a crop — isolates the jersey core."""
        h, w = crop.shape[:2]
        my = int(h * (1.0 - CENTRE_CROP_RATIO) / 2.0)
        mx = int(w * (1.0 - CENTRE_CROP_RATIO) / 2.0)
        return crop[my:h - my, mx:w - mx]

    @classmethod
    def _mean_colour_vec(cls, crops: list[np.ndarray]) -> np.ndarray | None:
        """Saturation-weighted mean HSV of the central jersey region across crops,
        returned as a cylindrical (chroma_a, chroma_b, value) vector. Returns None
        if no usable pixels were found."""
        sin_sum = cos_sum = sat_sum = val_sum = 0.0
        n_px = 0
        for crop in crops:
            centre = cls._centre_crop(crop)
            if centre.size == 0:
                continue
            hsv = cv2.cvtColor(centre, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0].astype(np.float64).ravel()  # 0–180 (OpenCV)
            sat = hsv[:, :, 1].astype(np.float64).ravel()  # 0–255
            val = hsv[:, :, 2].astype(np.float64).ravel()  # 0–255
            ang = hue * (np.pi / 90.0)                      # 0–180 → 0–2π
            sin_sum += float(np.sum(sat * np.sin(ang)))     # hue weighted by saturation
            cos_sum += float(np.sum(sat * np.cos(ang)))
            sat_sum += float(sat.sum())
            val_sum += float(val.sum())
            n_px    += hue.size
        if n_px == 0:
            return None
        mean_hue = (np.arctan2(sin_sum, cos_sum) * 90.0 / np.pi) % 180.0
        return cls._hsv_vec(mean_hue, sat_sum / n_px, val_sum / n_px)

    @staticmethod
    def _hsv_vec(hue: float, sat: float, val: float) -> np.ndarray:
        """Project an HSV colour into a cylindrical (a, b, value) space where
        Euclidean distance behaves sensibly: chromatic colours separate by hue,
        while achromatic colours (low saturation, e.g. white/black/grey) collapse
        toward the value axis so hue noise stops mattering."""
        ang = hue * (np.pi / 90.0)
        return np.array([sat * np.cos(ang), sat * np.sin(ang), val], dtype=np.float64)

    @staticmethod
    def _colour_to_hsv(colour: str) -> tuple[float, float, float]:
        """Resolve a caller colour ('orange' or hex) to OpenCV HSV (h:0–180, s/v:0–255)."""
        s = colour.strip().lower()
        # Resolve a CSS-style name to its hex, then normalise (the table's hex
        # values are upper-case) so the digit check and int() see lower-case.
        s = NAMED_COLOURS.get(s, s).lstrip("#").lower()
        if len(s) == 3:
            s = "".join(c * 2 for c in s)
        if len(s) != 6 or any(c not in "0123456789abcdef" for c in s):
            raise ValueError(f"unrecognised colour: {colour!r}")
        r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
        bgr = np.uint8([[[b, g, r]]])
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
        return float(hsv[0]), float(hsv[1]), float(hsv[2])

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
