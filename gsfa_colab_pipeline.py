"""
gsfa_colab_pipeline.py — GSFA Futsal Possession + Pass Pipeline (single-file Colab edition)

Flattened from:
  detectors/player_detector.py
  detectors/goalkeeper_detector.py
  detectors/cache.py
  team_classifier/team_classifier.py
  tracking/player_tracker.py
  video_analysis/possession.py

Run in Colab:
  !python gsfa_colab_pipeline.py

The CONFIG block below is the only section you need to fill in.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  COLAB INSTALL  (paste into a Colab code cell first)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  !pip install ultralytics supervision rfdetr boxmot joblib
  !pip install torch torchvision   # already present on Colab GPU runtime
  !pip install git+https://github.com/roboflow/sports.git   # sports.common.team (SigLIP)
  # umap-learn is a transitive dep of sports; install explicitly if not pulled in:
  !pip install umap-learn scikit-learn Pillow
"""

from __future__ import annotations

# ============================================================
#  CONFIG  — fill in these paths before running
# ============================================================

# Path to the YOLOv11 player-detection weights
# Classes:  0=active_player  1=goal_post  2=referee
PLAYER_MODEL_PATH   = r"/content/GSFA_PLAYER_DETECTION.pt"

# Path to the RF-DETR ball-detection weights (.pth checkpoint)
# Trained with num_classes=2, resolution=576; class 1 = ball
BALL_MODEL_PATH     = r"/content/gsfa_ball_detection.pth"

# Input video
INPUT_VIDEO_PATH    = r"/content/input_video.mp4"

# Annotated output video
OUTPUT_VIDEO_PATH   = r"/content/possession_output.mp4"

# --- optional pkl caches (set to None to always refit) ---
# If the path does not exist the classifier will fit from scratch and save here.
# Set USE_STUB = False to skip loading any cached pkl and always refit.
USE_STUB              = True
TEAM_SIGLIP_PKL_PATH  = r"/content/team_siglip.pkl"
GOALKEEPER_PKL_PATH   = r"/content/goalkeeper.pkl"

# ============================================================
#  PROCESSING PARAMETERS  (match video_analysis/possession.py)
# ============================================================

BALL_CLASS_ID            = 1
BALL_CONF                = 0.25

KALMAN_COAST_FRAMES      = 12
KALMAN_GATE_SIGMA        = 6.0

FOOT_ZONE_RATIO          = 0.45
FOOT_ZONE_MIN_PX         = 20
FOOT_ZONE_MAX_PX         = 140
CARRIER_HYSTERESIS_N     = 3

# PassEventTracker — tuned for TARGET_PROCESS_FPS = 15 fps
RELEASE_SUSTAIN_R        = 1
RECEPTION_SETTLE_C       = 2
TRAVEL_MIN_GAP           = 1
TRAVEL_TIMEOUT_FRAMES    = 22

TARGET_PROCESS_FPS       = 15.0
PROCESS_DURATION_SEC     = None   # None = process entire video; set e.g. 60.0 for first 60s

POSSESS_BAR_H            = 42
TEAM_BGR  = {0: (255, 80, 0), 1: (0, 80, 255), None: (160, 160, 160)}
GK_BGR    = (0, 215, 255)
BALL_BGR  = (0, 255, 255)

# TeamClassifier (GSFATeamClassifier)
TC_DEVICE     = "cpu"   # "cuda" if GPU is available in Colab
TC_BATCH_SIZE = 32
TC_SAMPLE_EVERY = 30    # sample 1 fps from a 30fps video for fit
MIN_CROP_PX   = 32
TORSO_RATIO   = 0.55
BLUR_THRESHOLD = 80

# PlayerDetector
PD_CONF   = 0.50
PD_DEVICE = "cpu"

# BoT-SORT
TRACK_BUFFER_FRAMES = 60

# Possession labels
POSSESS_TEAM0     = "team0"
POSSESS_TEAM1     = "team1"
POSSESS_LOOSE     = "loose"
POSSESS_OOF       = "oof"

# Pass event kinds
EVT_COMPLETED      = "completed"
EVT_INTERCEPTION   = "interception"
EVT_BALL_LOST      = "ball_lost"

# FSM phases
PHASE_IDLE     = "idle"
PHASE_POSS     = "in_possession"
PHASE_CAND_REL = "candidate_release"
PHASE_TRAVEL   = "travel"
PHASE_CAND_RCV = "candidate_reception"

# ============================================================
#  IMPORTS
# ============================================================

import math
import sys
import warnings
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import joblib
import numpy as np
import supervision as sv
from PIL import Image as PILImage
from ultralytics import YOLO

warnings.filterwarnings("ignore")
os.environ["TQDM_DISABLE"] = "1"

# ============================================================
#  DETECTION DATA CLASSES
#  (from detectors/player_detector.py)
# ============================================================

@dataclass
class Detection:
    """Single detected object in one frame."""
    class_id:     int
    class_name:   str
    bbox:         tuple[int, int, int, int]
    confidence:   float
    foot_point:   tuple[int, int]
    centre_point: tuple[int, int]
    team_id:      Optional[int] = None
    is_goalkeeper: bool         = False
    track_id:     Optional[int] = None
    embedding:    Optional[np.ndarray] = None


@dataclass
class FrameDetections:
    """All detections for one video frame."""
    frame_idx:   int
    timestamp_s: float
    players:     list = field(default_factory=list)
    referees:    list = field(default_factory=list)
    goal_posts:  list = field(default_factory=list)
    all:         list = field(default_factory=list)


# ============================================================
#  PLAYER DETECTOR
#  (from detectors/player_detector.py)
#  Model: YOLOv11, classes 0=active_player 1=goal_post 2=referee
#  Inference: conf=0.50, device cpu
# ============================================================

class PlayerDetector:

    CLASS_NAMES: dict = {
        0: "active_player",
        1: "goal_post",
        2: "referee",
    }

    def __init__(
        self,
        model_path: str = PLAYER_MODEL_PATH,
        conf: float = PD_CONF,
        device: str = PD_DEVICE,
    ) -> None:
        self.conf   = conf
        self.device = device
        self.model  = YOLO(model_path)

    def detect(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        fps: float = 30.0,
    ) -> FrameDetections:
        results = self.model(frame, conf=self.conf, device=self.device, verbose=False)
        return self._parse(results[0], frame_idx, fps)

    def process_video(
        self,
        video_path: str,
        sample_every: int = 1,
        progress: bool = True,
    ) -> list:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"PlayerDetector: cannot open video: {video_path}")
        fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results_out = []
        fidx = 0
        sampled = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fidx % sample_every == 0:
                dets = self.detect(frame, frame_idx=fidx, fps=fps)
                results_out.append(dets)
                sampled += 1
                if progress and sampled % 200 == 0:
                    pct = fidx / max(1, total_f) * 100
                    print(f"  [PlayerDetector] frame {fidx:>5}/{total_f} ({pct:.0f}%)  sampled={sampled}")
            fidx += 1
        cap.release()
        return results_out

    def _parse(self, r, frame_idx: int, fps: float) -> FrameDetections:
        fd = FrameDetections(frame_idx=frame_idx, timestamp_s=frame_idx / max(fps, 1.0))
        if r.boxes is None or len(r.boxes) == 0:
            return fd
        for box in r.boxes:
            cid  = int(box.cls[0].cpu().numpy())
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            det  = Detection(
                class_id     = cid,
                class_name   = self.CLASS_NAMES.get(cid, f"cls_{cid}"),
                bbox         = (x1, y1, x2, y2),
                confidence   = conf,
                foot_point   = ((x1 + x2) // 2, y2),
                centre_point = ((x1 + x2) // 2, (y1 + y2) // 2),
            )
            fd.all.append(det)
            if det.class_name == "active_player":
                fd.players.append(det)
            elif det.class_name == "referee":
                fd.referees.append(det)
            elif det.class_name == "goal_post":
                fd.goal_posts.append(det)
        return fd


# ============================================================
#  GSFA TEAM CLASSIFIER  (SigLIP + UMAP + KMeans)
#  (from team_classifier/team_classifier.py)
#  Uses sports.common.team.TeamClassifier
#  Crops: top TORSO_RATIO (55%) of player bbox
#  Blur filter: Laplacian variance > BLUR_THRESHOLD (80)
#  MIN_CROP_PX = 32
#  Fit: 1 fps sampling, SigLIP -> UMAP -> KMeans(k=2)
#  Output: embedding (768-D float32) + team_id (0 or 1)
# ============================================================

class GSFATeamClassifier:

    def __init__(
        self,
        device:     str = TC_DEVICE,
        batch_size: int = TC_BATCH_SIZE,
    ) -> None:
        from sports.common.team import TeamClassifier
        self._classifier = TeamClassifier(device=device, batch_size=batch_size)
        self._is_fitted  = False
        self.device      = device
        self.batch_size  = batch_size

    # --- internal crop helpers ---

    @staticmethod
    def _torso_crop(frame: np.ndarray, bbox: tuple) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        torso_y2 = y1 + int((y2 - y1) * TORSO_RATIO)
        return frame[y1:torso_y2, x1:x2]

    @staticmethod
    def _is_sharp(crop: np.ndarray) -> bool:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var()) > BLUR_THRESHOLD

    # --- fit ---

    def fit_from_video(
        self,
        video_path:   str,
        player_det:   PlayerDetector,
        sample_every: int = TC_SAMPLE_EVERY,
        save_path:    Optional[Path] = None,
        progress:     bool = True,
    ) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"GSFATeamClassifier: cannot open video: {video_path}")
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
        crops   = []
        fidx    = 0
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
                    print(f"  frame {fidx:>5}/{total_f} ({pct:.0f}%)  crops: {len(crops)}")
            fidx += 1
        cap.release()
        if not crops:
            raise RuntimeError("GSFATeamClassifier.fit_from_video: no valid crops found.")
        if progress:
            print(f"[TeamClassifier] Fitting on {len(crops)} crops (SigLIP → UMAP → KMeans) …")
        self._classifier.fit(crops)
        self._is_fitted = True
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, save_path)
            if progress:
                print(f"[TeamClassifier] Fitted and saved → {save_path}")

    def fit_from_video_or_load(
        self,
        video_path:   str,
        player_det:   PlayerDetector,
        save_path:    Optional[Path] = None,
        sample_every: int  = TC_SAMPLE_EVERY,
        progress:     bool = True,
        force_refit:  bool = False,
    ) -> None:
        if save_path is None:
            save_path = Path(TEAM_SIGLIP_PKL_PATH)
        use_cache = (
            USE_STUB
            and not force_refit
            and Path(save_path).exists()
        )
        if use_cache:
            loaded = GSFATeamClassifier._load(save_path, progress=progress)
            self._classifier = loaded._classifier
            self._is_fitted  = loaded._is_fitted
        else:
            self.fit_from_video(
                video_path   = video_path,
                player_det   = player_det,
                sample_every = sample_every,
                save_path    = save_path if USE_STUB else None,
                progress     = progress,
            )

    # --- classify (per frame) ---

    def classify(
        self,
        frame:      np.ndarray,
        detections: FrameDetections,
    ) -> None:
        """
        Assigns team_id (0 or 1) and 768-D SigLIP embedding to every
        active_player Detection in-place. Referees/goal_posts unchanged.
        """
        if not self._is_fitted:
            raise RuntimeError("GSFATeamClassifier is not fitted.")
        if not detections.players:
            return
        crops        = []
        valid_indices = []
        for i, p in enumerate(detections.players):
            crop = self._torso_crop(frame, p.bbox)
            if crop.shape[0] >= MIN_CROP_PX and crop.shape[1] >= MIN_CROP_PX:
                crops.append(crop)
                valid_indices.append(i)
        if not crops:
            return
        # Three-stage predict — preserves the 768-D SigLIP features
        features    = self._classifier.extract_features(crops)           # (N, 768)
        projections = self._classifier.reducer.transform(features)       # (N, 3)
        team_ids    = self._classifier.cluster_model.predict(projections) # (N,)
        for list_pos, det_idx in enumerate(valid_indices):
            detections.players[det_idx].team_id   = int(team_ids[list_pos])
            detections.players[det_idx].embedding = features[list_pos].astype(np.float32)

    # --- save / load ---

    @staticmethod
    def _load(path: Path, progress: bool = True) -> "GSFATeamClassifier":
        if progress:
            print(f"[TeamClassifier] Loading from {path} …")
        obj = joblib.load(path)
        if progress:
            print("[TeamClassifier] Loaded.")
        return obj


# ============================================================
#  GOALKEEPER DETECTOR
#  (from detectors/goalkeeper_detector.py)
#  Fit: scan at 1fps; for each goal_post find closest player;
#       average positions per post → 2 GK zone centroids;
#       assign each zone to nearer team centroid.
#  Classify: per post, mark nearest player is_goalkeeper=True,
#            override team_id with fitted assignment.
# ============================================================

def _dist(a: tuple, b: tuple) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _centroid(points: list) -> tuple:
    if not points:
        return (0.0, 0.0)
    return (
        float(np.mean([p[0] for p in points])),
        float(np.mean([p[1] for p in points])),
    )


class GoalkeeperDetector:

    def __init__(self) -> None:
        self._gk_zones: Optional[list] = None
        self._gk_teams: Optional[list] = None
        self._is_fitted: bool = False

    def fit_from_video(
        self,
        video_path:   str,
        player_det:   PlayerDetector,
        team_clf:     GSFATeamClassifier,
        sample_every: int = 30,
        save_path:    Optional[Path] = None,
        progress:     bool = True,
    ) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"GoalkeeperDetector: cannot open {video_path}")
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
        post_nearest: list = [[], []]
        team_positions: dict = {0: [], 1: []}
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
                if dets.goal_posts and len(dets.players) >= 2:
                    posts = sorted(dets.goal_posts, key=lambda p: p.foot_point[0])
                    for post_idx, post in enumerate(posts[:2]):
                        closest = min(
                            dets.players,
                            key=lambda p: _dist(p.foot_point, post.foot_point),
                        )
                        post_nearest[post_idx].append(closest.foot_point)
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
            raise RuntimeError("GoalkeeperDetector: no goal posts detected during fit.")
        self._gk_zones = [
            _centroid(post_nearest[0]) if post_nearest[0] else (0.0, 0.0),
            _centroid(post_nearest[1]) if post_nearest[1] else (0.0, 0.0),
        ]
        t0_c = _centroid(team_positions[0])
        t1_c = _centroid(team_positions[1])
        self._gk_teams = []
        for zone in self._gk_zones:
            d0 = _dist(zone, t0_c)
            d1 = _dist(zone, t1_c)
            self._gk_teams.append(0 if d0 <= d1 else 1)
        self._is_fitted = True
        if progress:
            print(f"[GoalkeeperDetector] GK zone 0 → team {self._gk_teams[0]}")
            print(f"[GoalkeeperDetector] GK zone 1 → team {self._gk_teams[1]}")
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, save_path)
            if progress:
                print(f"[GoalkeeperDetector] Saved → {save_path}")

    def fit_from_video_or_load(
        self,
        video_path:   str,
        player_det:   PlayerDetector,
        team_clf:     GSFATeamClassifier,
        sample_every: int = 30,
        save_path:    Optional[Path] = None,
        progress:     bool = True,
        force_refit:  bool = False,
    ) -> None:
        if save_path is None:
            save_path = Path(GOALKEEPER_PKL_PATH)
        use_cache = (
            USE_STUB
            and not force_refit
            and Path(save_path).exists()
        )
        if use_cache:
            loaded = GoalkeeperDetector._load(save_path, progress=progress)
            self._gk_zones  = loaded._gk_zones
            self._gk_teams  = loaded._gk_teams
            self._is_fitted = loaded._is_fitted
        else:
            self.fit_from_video(
                video_path   = video_path,
                player_det   = player_det,
                team_clf     = team_clf,
                sample_every = sample_every,
                save_path    = save_path if USE_STUB else None,
                progress     = progress,
            )

    def classify(self, detections: FrameDetections) -> None:
        """
        Per post: find closest active_player, mark is_goalkeeper=True,
        override team_id with fitted assignment.
        Must be called after team_clf.classify() on the same detections.
        """
        if not self._is_fitted:
            raise RuntimeError("GoalkeeperDetector not fitted.")
        for p in detections.players:
            p.is_goalkeeper = False
        if not detections.goal_posts or not detections.players:
            return
        posts = sorted(detections.goal_posts, key=lambda p: p.foot_point[0])
        for post_idx, post in enumerate(posts[:2]):
            closest = min(
                detections.players,
                key=lambda p: _dist(p.foot_point, post.foot_point),
            )
            closest.is_goalkeeper = True
            closest.team_id = self._gk_teams[post_idx]

    @staticmethod
    def _load(path: Path, progress: bool = True) -> "GoalkeeperDetector":
        if progress:
            print(f"[GoalkeeperDetector] Loading from {path} …")
        obj = joblib.load(path)
        if progress:
            print("[GoalkeeperDetector] Loaded.")
        return obj


# ============================================================
#  PLAYER TRACKER  (BoT-SORT with ECC + SigLIP embeddings)
#  (from tracking/player_tracker.py)
#  track_high_thresh=0.5  track_low_thresh=0.1  new_track_thresh=0.6
#  match_thresh=0.8  proximity_thresh=0.5  appearance_thresh=0.25
#  track_buffer=60 frames at 30fps  cmc_method="ecc"  with_reid=True
# ============================================================

class PlayerTracker:

    def __init__(self, fps: float) -> None:
        from boxmot.trackers.bbox.botsort.botsort import BotSort
        self.tracker = BotSort(
            reid_model          = None,
            with_reid           = True,
            cmc_method          = "ecc",
            track_high_thresh   = 0.5,
            track_low_thresh    = 0.1,
            new_track_thresh    = 0.6,
            match_thresh        = 0.8,
            proximity_thresh    = 0.5,
            appearance_thresh   = 0.25,
            track_buffer        = TRACK_BUFFER_FRAMES,
            frame_rate          = int(round(fps)),
            fuse_first_associate = False,
        )

    def update(self, frame: np.ndarray, players: list) -> None:
        """
        Runs BoT-SORT on one frame's player detections.
        Writes track_id onto each Detection in-place.
        SigLIP embeddings from GSFATeamClassifier.classify() are passed as
        the appearance matrix; detections without embeddings are motion-only.
        """
        if not players:
            self.tracker.update(np.empty((0, 6), dtype=np.float32), frame)
            return
        dets = np.array(
            [[*p.bbox, p.confidence, 0] for p in players],
            dtype=np.float32,
        )
        emb_dim = None
        for p in players:
            if p.embedding is not None:
                emb_dim = int(p.embedding.shape[0])
                break
        if emb_dim is None:
            embs = None
        else:
            embs = np.zeros((len(players), emb_dim), dtype=np.float32)
            for i, p in enumerate(players):
                if p.embedding is not None:
                    embs[i] = p.embedding.astype(np.float32)
        out = self.tracker.update(dets, frame, embs=embs)
        out_arr = np.asarray(out)
        if out_arr.size == 0:
            return
        # boxmot rows: x1,y1,x2,y2, id, conf, cls, det_ind
        for row in out_arr:
            det_ind = int(row[7])
            if 0 <= det_ind < len(players):
                players[det_ind].track_id = int(row[4])


# ============================================================
#  BALL DETECTOR  (RF-DETR)
#  (from video_analysis/possession.py)
#  RFDETRMedium, num_classes=2, resolution=576
#  class_id=1 → ball, conf=0.25
# ============================================================

@dataclass
class BallDetection:
    bbox:       tuple
    centre:     tuple
    confidence: float


class BallDetector:

    def __init__(
        self,
        weights:       str   = BALL_MODEL_PATH,
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


# ============================================================
#  BALL TRACKER  (constant-velocity Kalman filter)
#  (from video_analysis/possession.py)
#  State: [x, y, vx, vy]   Measurement: [x, y]
#  processNoiseCov: diag([4,4,25,25])
#  measurementNoiseCov: diag([9,9])
#  coast_frames=12  gate_sigma=6.0
# ============================================================

class BallTracker:

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
        self._kf.statePost    = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
        self._kf.errorCovPost = np.eye(4, dtype=np.float32) * 100.0
        self._initialised     = True

    def update(self, det: Optional[BallDetection]) -> tuple:
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
        px, py = float(pred[0]), float(pred[1])

        if det is not None and self._accept(det, px, py):
            meas = np.array([[np.float32(det.centre[0])], [np.float32(det.centre[1])]])
            est  = self._kf.correct(meas)
            ex, ey = int(est[0]), int(est[1])
            self.state       = self.DETECTED
            self.missing_for = 0
            self.last_obs    = det
            out = BallDetection(bbox=det.bbox, centre=(ex, ey), confidence=det.confidence)
            return self.DETECTED, out

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
        S = (self._kf.measurementMatrix
             @ self._kf.errorCovPre
             @ self._kf.measurementMatrix.T
             + self._kf.measurementNoiseCov)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return True
        innov = np.array([det.centre[0] - px, det.centre[1] - py], dtype=np.float64)
        m2 = float(innov @ S_inv @ innov)
        return m2 <= (self.gate_sigma ** 2)


# ============================================================
#  CARRIER ENGINE
#  (from video_analysis/possession.py)
#  Ball-to-player assignment: ball centre vs player foot_point.
#  foot_zone_radius = clamp(FOOT_ZONE_RATIO * bbox_height,
#                           FOOT_ZONE_MIN_PX, FOOT_ZONE_MAX_PX)
#                   = clamp(0.45 * h, 20, 140)
#  Only team 0 / team 1 players with a track_id are considered.
#  Multiple players in zone → LOOSE.
#  No players in zone → LOOSE.
#  Ball LOST → OOF.
#  Hysteresis: commit only after CARRIER_HYSTERESIS_N (3) consecutive matching frames.
# ============================================================

@dataclass
class CarrierState:
    kind:     str
    track_id: Optional[int] = None
    team_id:  Optional[int] = None
    player:   Optional[Detection] = None


def _foot_zone_radius(p: Detection) -> float:
    h = max(1, p.bbox[3] - p.bbox[1])
    r = FOOT_ZONE_RATIO * h
    return max(FOOT_ZONE_MIN_PX, min(FOOT_ZONE_MAX_PX, r))


class CarrierEngine:

    OOF_STATE   = CarrierState(kind="oof")
    LOOSE_STATE = CarrierState(kind="loose")

    def __init__(self, hysteresis_n: int = CARRIER_HYSTERESIS_N) -> None:
        self.hysteresis_n = hysteresis_n
        self._buffer: deque = deque(maxlen=hysteresis_n)
        self._committed: CarrierState = self.OOF_STATE

    def get_state(self) -> CarrierState:
        return self._committed

    def update(
        self,
        players:    list,
        ball_state: str,
        ball:       Optional[BallDetection],
    ) -> CarrierState:
        raw = self._raw(players, ball_state, ball)
        self._buffer.append(raw)
        if len(self._buffer) < self.hysteresis_n:
            return self._committed
        keys = {(s.kind, s.track_id) for s in self._buffer}
        if len(keys) == 1:
            self._committed = self._buffer[-1]
        return self._committed

    def _raw(
        self,
        players:    list,
        ball_state: str,
        ball:       Optional[BallDetection],
    ) -> CarrierState:
        if ball_state == BallTracker.LOST or ball is None:
            return self.OOF_STATE
        bx, by = ball.centre
        in_zone = []
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
        return CarrierState(kind="carrier", track_id=p.track_id, team_id=p.team_id, player=p)


# ============================================================
#  PASS EVENT TRACKER  (3-phase FSM)
#  (from video_analysis/possession.py)
#
#  Phases: IDLE -> IN_POSSESSION -> CAND_RELEASE -> TRAVEL -> CAND_RECEPTION -> ...
#
#  Release confirmed after RELEASE_SUSTAIN_R (1) processed frames without
#  the same carrier re-entering zone (i.e. dt >= 1 frame gap).
#
#  Reception confirmed after RECEPTION_SETTLE_C (2) consecutive processed
#  frames with the new carrier holding.
#
#  Travel min gap = TRAVEL_MIN_GAP (1) processed frame before a new candidate
#  receiver is accepted.
#
#  Travel timeout = TRAVEL_TIMEOUT_FRAMES (22) processed frames (~1.47s at 15fps).
#  After timeout → EVT_BALL_LOST.
#
#  Reception by same team = EVT_COMPLETED (counts as "successful" in overlay)
#  Reception by other team = EVT_INTERCEPTION (counts as "inaccurate" on passer's team)
#  Timeout = EVT_BALL_LOST (ignored for pass stats)
#
#  Pass accuracy = successful / (successful + inaccurate) per team
# ============================================================

@dataclass
class PassEvent:
    kind:          str
    from_track_id: Optional[int]
    from_team_id:  Optional[int]
    to_track_id:   Optional[int]
    to_team_id:    Optional[int]
    release_frame: int
    end_frame:     int
    travel_frames: int


class PassEventTracker:

    def __init__(
        self,
        release_sustain:  int = RELEASE_SUSTAIN_R,
        reception_settle: int = RECEPTION_SETTLE_C,
        travel_min_gap:   int = TRAVEL_MIN_GAP,
        travel_timeout:   int = TRAVEL_TIMEOUT_FRAMES,
    ) -> None:
        self.release_sustain  = release_sustain
        self.reception_settle = reception_settle
        self.travel_min_gap   = travel_min_gap
        self.travel_timeout   = travel_timeout

        self.phase: str = PHASE_IDLE
        self._passer:            Optional[tuple] = None
        self._receiver:          Optional[tuple] = None
        self._cand_release_at:   int = -1
        self._release_at:        int = -1
        self._cand_reception_at: int = -1
        self._travel_credit_team: Optional[int] = None
        self._travel_frames_so_far: int = 0

        self.events: list = []
        self.stats_internal: dict = {
            0: {EVT_COMPLETED: 0, EVT_INTERCEPTION: 0, EVT_BALL_LOST: 0},
            1: {EVT_COMPLETED: 0, EVT_INTERCEPTION: 0, EVT_BALL_LOST: 0},
        }

    def update(self, carrier: CarrierState, frame_idx: int) -> tuple:
        adjustments = []
        self._step(carrier, frame_idx, adjustments)
        if self.phase in (PHASE_CAND_REL, PHASE_TRAVEL, PHASE_CAND_RCV):
            self._travel_frames_so_far += 1
        label = self._label_for_current_state(carrier)
        return label, adjustments

    def _step(self, carrier: CarrierState, f: int, adjustments: list) -> None:
        if self.phase == PHASE_IDLE:
            if carrier.kind == "carrier":
                self._enter_possession(carrier)
            return

        if self.phase == PHASE_POSS:
            assert self._passer is not None
            if carrier.kind == "carrier" and carrier.track_id == self._passer[0]:
                return
            self._enter_cand_release(f)
            return

        if self.phase == PHASE_CAND_REL:
            assert self._passer is not None
            dt = f - self._cand_release_at
            if carrier.kind == "carrier" and carrier.track_id == self._passer[0]:
                self.phase     = PHASE_POSS
                self._receiver = None
                self._reset_travel()
                return
            if dt >= self.release_sustain:
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
                    self.phase     = PHASE_POSS
                    self._receiver = None
                    self._reset_travel()
                    return
                if travel_dt >= self.travel_min_gap:
                    self._enter_cand_reception(carrier, f)
                return
            return

        if self.phase == PHASE_CAND_RCV:
            assert self._passer is not None and self._receiver is not None
            dt = f - self._cand_reception_at
            if carrier.kind == "carrier" and carrier.track_id == self._receiver[0]:
                if dt >= self.reception_settle:
                    self._resolve_reception(f, adjustments)
                return
            self._receiver = None
            self.phase     = PHASE_TRAVEL
            if (
                carrier.kind == "carrier"
                and carrier.track_id != self._passer[0]
                and (f - self._release_at) >= self.travel_min_gap
            ):
                self._enter_cand_reception(carrier, f)
            return

    def _enter_possession(self, carrier: CarrierState) -> None:
        assert carrier.track_id is not None and carrier.team_id is not None
        self.phase     = PHASE_POSS
        self._passer   = (carrier.track_id, carrier.team_id)
        self._receiver = None
        self._reset_travel()

    def _enter_cand_release(self, f: int) -> None:
        self.phase                = PHASE_CAND_REL
        self._cand_release_at     = f
        self._travel_credit_team  = self._passer[1] if self._passer else None
        self._travel_frames_so_far = 0

    def _enter_cand_reception(self, carrier: CarrierState, f: int) -> None:
        assert carrier.track_id is not None and carrier.team_id is not None
        self.phase               = PHASE_CAND_RCV
        self._receiver           = (carrier.track_id, carrier.team_id)
        self._cand_reception_at  = f

    def _resolve_ball_lost(self, f: int, adjustments: list) -> None:
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
        if self._travel_credit_team is not None and self._travel_frames_so_far:
            adjustments.append(("drop", self._travel_credit_team, self._travel_frames_so_far))
        self.phase    = PHASE_IDLE
        self._passer  = None
        self._receiver = None
        self._reset_travel()

    def _resolve_reception(self, f: int, adjustments: list) -> None:
        assert self._passer is not None and self._receiver is not None
        from_tid, from_tid_team = self._passer
        to_tid, to_team         = self._receiver
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
            adjustments.append(("flip_to", to_team, self._travel_frames_so_far))
        self.phase    = PHASE_POSS
        self._passer  = (to_tid, to_team)
        self._receiver = None
        self._reset_travel()

    def _reset_travel(self) -> None:
        self._cand_release_at      = -1
        self._release_at           = -1
        self._cand_reception_at    = -1
        self._travel_credit_team   = None
        self._travel_frames_so_far = 0

    def _label_for_current_state(self, carrier: CarrierState) -> str:
        if self.phase == PHASE_POSS:
            team = self._passer[1] if self._passer else None
            return POSSESS_TEAM0 if team == 0 else POSSESS_TEAM1
        if self.phase in (PHASE_CAND_REL, PHASE_TRAVEL, PHASE_CAND_RCV):
            team = self._passer[1] if self._passer else None
            if team == 0:
                return POSSESS_TEAM0
            if team == 1:
                return POSSESS_TEAM1
        if carrier.kind == "loose":
            return POSSESS_LOOSE
        return POSSESS_OOF

    def summary_for_overlay(self) -> dict:
        """
        Maps internal stats to the overlay schema:
            completed    -> successful
            interception -> inaccurate  (on passer's team)
            ball_lost    -> ignored
        Pass accuracy = successful / (successful + inaccurate)
        """
        out = {
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
            ok  = s[tid]["successful"]
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


# ============================================================
#  POSSESSION STATS
#  (from video_analysis/possession.py)
#
#  Denominator = team0_frames + team1_frames (excludes loose/OOF)
#  Retroactive corrections via adjustments from PassEventTracker:
#    ("flip_to", team_id, n) — move n frames from other team to team_id
#    ("drop",    team_id, n) — remove n frames from team_id, give to OOF
# ============================================================

class PossessionStats:

    def __init__(self) -> None:
        self.frame_counts: dict = {
            POSSESS_TEAM0: 0,
            POSSESS_TEAM1: 0,
            POSSESS_LOOSE: 0,
            POSSESS_OOF:   0,
        }
        self.total = 0

    def update(self, label: str) -> None:
        self.frame_counts[label] = self.frame_counts.get(label, 0) + 1
        self.total += 1

    def apply_adjustments(self, adjustments: list) -> None:
        for kind, team_id, n in adjustments:
            src_label   = POSSESS_TEAM0 if team_id == 0 else POSSESS_TEAM1
            other_label = POSSESS_TEAM1 if team_id == 0 else POSSESS_TEAM0
            if kind == "flip_to":
                move = min(n, self.frame_counts[other_label])
                self.frame_counts[other_label] -= move
                self.frame_counts[src_label]   += move
            elif kind == "drop":
                move = min(n, self.frame_counts[src_label])
                self.frame_counts[src_label] -= move
                self.frame_counts[POSSESS_OOF] += move

    def percentages(self) -> tuple:
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


# ============================================================
#  SUPERVISION ANNOTATORS
#  (from video_analysis/possession.py)
#  Palette: team0=blue (#0050FF), team1=red (#FF5000), unclassified=grey (#A0A0A0)
#  EllipseAnnotator (thickness=2)
#  TriangleAnnotator for ball (cyan, base=16, height=16)
#  LabelAnnotator (text_scale=0.38, text_thickness=1, text_padding=3)
# ============================================================

_PALETTE     = sv.ColorPalette.from_hex(["#0050FF", "#FF5000", "#A0A0A0"])
_ellipse_ann = sv.EllipseAnnotator(color=_PALETTE, thickness=2)
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


def _to_sv(players: list) -> tuple:
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


# ============================================================
#  FRAME RENDERER
#  (from video_analysis/possession.py → draw_frame)
#
#  Draws:
#    - Ellipses under players (team colour via supervision EllipseAnnotator)
#    - Team/track labels (supervision LabelAnnotator)
#    - Cyan ellipse under each goalkeeper
#    - Black rectangle for goal_posts
#    - Cyan triangle above ball (supervision TriangleAnnotator)
#    - Yellow circle at carrier foot_point (r=14, thickness=2)
#    - Possession bar (top of frame, 42px high): team0 fills from left, team1 from right
#    - Pass overlay (below possession bar): T0/T1 pass counts
# ============================================================

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


# ============================================================
#  MAIN PIPELINE
# ============================================================

def run(
    video_path:   str = INPUT_VIDEO_PATH,
    out_path:     str = OUTPUT_VIDEO_PATH,
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print("[Pipeline] Initialising models …")
    player_det     = PlayerDetector(model_path=PLAYER_MODEL_PATH)
    ball_det_model = BallDetector(weights=BALL_MODEL_PATH)

    team_clf = GSFATeamClassifier(device=TC_DEVICE, batch_size=TC_BATCH_SIZE)
    team_clf.fit_from_video_or_load(
        video_path  = video_path,
        player_det  = player_det,
        save_path   = Path(TEAM_SIGLIP_PKL_PATH) if USE_STUB else None,
    )

    gk_det = GoalkeeperDetector()
    gk_det.fit_from_video_or_load(
        video_path = video_path,
        player_det = player_det,
        team_clf   = team_clf,
        save_path  = Path(GOALKEEPER_PKL_PATH) if USE_STUB else None,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_step = max(1, round(fps / TARGET_PROCESS_FPS))
    if PROCESS_DURATION_SEC is not None:
        max_frame = int(fps * PROCESS_DURATION_SEC)
    else:
        max_frame = total_frames

    eff_fps = fps / frame_step

    tracker      = PlayerTracker(fps=fps)
    ball_tracker = BallTracker()
    carrier_eng  = CarrierEngine()
    pass_track   = PassEventTracker()
    stats        = PossessionStats()

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        TARGET_PROCESS_FPS,
        (W_vid, H_vid),
    )

    print(
        f"[Pipeline] native {fps:.0f}fps → {eff_fps:.0f}fps (step={frame_step}) | "
        f"{W_vid}x{H_vid} | max_frame={max_frame}"
    )
    print("-" * 60)

    fidx     = 0
    proc_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or fidx >= max_frame:
                break

            if fidx % frame_step != 0:
                fidx += 1
                continue

            t_sec = fidx / fps

            # --- Detection ---
            player_dets = player_det.detect(frame, fidx, fps)
            team_clf.classify(frame, player_dets)    # sets team_id + 768-D embedding
            gk_det.classify(player_dets)              # sets is_goalkeeper + overrides team_id
            tracker.update(frame, player_dets.players) # sets track_id

            # --- Ball ---
            raw_ball         = ball_det_model.detect(frame)
            ball_state, ball = ball_tracker.update(raw_ball)

            # --- Carrier ---
            carrier = carrier_eng.update(player_dets.players, ball_state, ball)

            # --- Pass FSM ---
            prev_phase  = pass_track.phase
            prev_passer = pass_track._passer
            prev_n_evts = len(pass_track.events)

            label, adjustments = pass_track.update(carrier, proc_idx)

            # Phase-transition log
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

            # Resolved-event log
            for evt in pass_track.events[prev_n_evts:]:
                t_evt = evt.end_frame / TARGET_PROCESS_FPS
                if evt.kind == EVT_COMPLETED:
                    print(f"[{t_evt:.2f}s] PASS COMPLETED    T{evt.from_team_id}#{evt.from_track_id} → T{evt.to_team_id}#{evt.to_track_id}  travel={evt.travel_frames}f")
                elif evt.kind == EVT_INTERCEPTION:
                    print(f"[{t_evt:.2f}s] INTERCEPTED       T{evt.from_team_id}#{evt.from_track_id} → T{evt.to_team_id}#{evt.to_track_id}  travel={evt.travel_frames}f")
                elif evt.kind == EVT_BALL_LOST:
                    print(f"[{t_evt:.2f}s] BALL LOST         from T{evt.from_team_id}#{evt.from_track_id}  travel={evt.travel_frames}f")

            # --- Possession stats ---
            stats.update(label)
            if adjustments:
                stats.apply_adjustments(adjustments)

            # --- Render + write ---
            out_frame = draw_frame(frame, player_dets, ball, carrier, stats, pass_track)
            writer.write(out_frame)

            fidx += 1

    finally:
        cap.release()
        writer.release()

    print("-" * 60)
    print(f"\n[Pipeline] Saved → {out_path}")
    print(stats.summary())
    print(pass_track.summary())


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GSFA Possession + Pass Pipeline")
    parser.add_argument("--input",  default=INPUT_VIDEO_PATH,  help="Input video path")
    parser.add_argument("--output", default=OUTPUT_VIDEO_PATH, help="Output video path")
    args = parser.parse_args()
    run(video_path=args.input, out_path=args.output)
