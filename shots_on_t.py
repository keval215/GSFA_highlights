"""
GSFA Futsal Analytics — Shots on Target (Colab / T4, paste-in-one-cell)
========================================================================

Architecture (v2 — gated cascade, event-triggered prediction):

  BALL TRACKING — every candidate detection passes a filter cascade
  before it may touch tracker state. A teleport never accepted is a
  teleport never smoothed away:
    Gate 1  Court mask      — HSV-segmented playable court; candidates
                              in the stands/scoreboard die instantly.
    Gate 2  Size/shape      — 0.5x..2x expected ball diameter.
    Gate 3  Distance gate   — nearest candidate to the Kalman prediction,
                              accepted only within an adaptive radius
                              (R_base + 1.5 * |velocity|). Replaces
                              ByteTrack: IoU association is meaningless
                              for a 18 px object moving > its own
                              diameter per frame.
    Gate 4  Physics gate    — position innovation ceiling. Kicks are
                              huge ACCELERATIONS and pass; teleports are
                              huge POSITION JUMPS and never do.
  Track lifecycle is a state machine (INACTIVE -> TENTATIVE -> CONFIRMED
  -> COASTING), with CONSECUTIVE (not cumulative) hit counting. Coasted
  predictions are used for association only — never drawn, never fed to
  trajectory prediction.

  CAMERA MOTION — median sparse optical flow on off-court features is
  the pan velocity. Track state, trail and goal planes are shifted with
  it; heavy pans raise a flag that suppresses shot detection entirely.

  GOAL PLANES — live and detection-tied: SAM2-SMALL runs ONLY on
  frames where YOLO detects a goalpost (bbox prompt) + white-pixel
  filter -> convex hull -> 4 corners. A goalpost whose bbox centre is
  left of the frame midline is the LEFT goal, else RIGHT. The plane
  EXISTS ONLY WHILE the goalpost bbox is being detected — if YOLO
  stops seeing it for PLANE_TTL_FRAMES the plane is dropped, so it can
  never persist and drift across the court. A new extraction far from
  the old plane REPLACES it (no blending across a pan).

  SHOT DETECTION — once a goal plane exists, every confirmed ball
  frame predicts the trajectory 1.2 seconds ahead (constant velocity x
  friction decay, re-computed each frame from the LATEST state) and
  tests the path against the goal-plane polygon (point-in-polygon —
  on target means INSIDE the trapezoid). A shot is counted when the
  path hits the plane on >= 3 CONSECUTIVE frames while the ball moves
  toward that goal above a minimum speed. Hindsight validation then
  watches the next ~0.8 s of confirmed track: if the ball never
  actually arrived near the plane, the count is revoked. HUD shows
  provisional; the final report shows validated.

Colab usage:
  1. Runtime -> Change runtime type -> T4 GPU.
  2. Paste this entire file into ONE cell and run. Dependencies are
     installed automatically on first run (~2-3 min).

Inputs (fixed paths, edit the CONFIG block):
  Video  : /content/video/SHOTS ON TARGET.mp4
  Ball   : /content/models/gsfa_ball_detection.pth   (RF-DETR Medium)
  Player : /content/models/GSFA_PLAYER_DETECTION.pt  (YOLO11-L,
           classes {0:'active_players', 1:'goal post', 2:'refree'})
"""

# ══════════════════════════════════════════════════════════════════════════════
# 0.  DEPENDENCY BOOTSTRAP (Colab-friendly: installs only what's missing)
# ══════════════════════════════════════════════════════════════════════════════

import importlib
import subprocess
import sys


def _ensure(import_name: str, pip_spec: str) -> None:
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"[SETUP] Installing {pip_spec} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_spec],
                       check=True)


_ensure("rfdetr",      "rfdetr")
_ensure("ultralytics", "ultralytics")
_ensure("filterpy",    "filterpy")
_ensure("tqdm",        "tqdm")
_ensure("sam2",        "git+https://github.com/facebookresearch/sam2.git")

import collections
import math
import os
import urllib.request
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from filterpy.kalman import KalmanFilter
from tqdm import tqdm
from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIG — every tunable in one place
# ══════════════════════════════════════════════════════════════════════════════

# ── Paths ─────────────────────────────────────────────────────────────────────
VIDEO_PATH        = "/content/video/SHOTS ON TARGET.mp4"
BALL_MODEL_PATH   = "/content/models/gsfa_ball_detection.pth"
PLAYER_MODEL_PATH = "/content/models/GSFA_PLAYER_DETECTION.pt"
OUTPUT_PATH       = "/content/output_annotated.mp4"
SAM2_CKPT_DIR     = "/content/checkpoints"
SAM2_CKPT_URL     = ("https://dl.fbaipublicfiles.com/segment_anything_2/"
                     "092824/sam2.1_hiera_small.pt")
SAM2_MODEL_CFG    = "configs/sam2.1/sam2.1_hiera_s.yaml"

# ── Detection ─────────────────────────────────────────────────────────────────
BALL_CONF_THRESHOLD   = 0.15   # low on purpose — the gates do the filtering
PLAYER_CONF           = 0.45
GOALPOST_CONF         = 0.40
PLAYER_DET_STRIDE     = 2      # YOLO every N frames (players are decoration)

# ── Court mask (Gate 1) ───────────────────────────────────────────────────────
COURT_HSV_LOW    = np.array([90, 40, 40])    # futsal court is uniform blue
COURT_HSV_HIGH   = np.array([130, 255, 255])
COURT_DILATE_PX  = 20
COURT_SAMPLES    = 10                         # frames sampled to build mask

# ── Ball geometry (Gate 2) ────────────────────────────────────────────────────
BALL_DIAMETER_PX = 18          # calibrate once for your resolution
BALL_SIZE_MIN    = 0.5         # accept 0.5x .. 2x diameter
BALL_SIZE_MAX    = 2.0         # loose upper bound: motion blur stretches it

# ── Association (Gate 3) + physics (Gate 4) ──────────────────────────────────
GATE_BASE_PX      = 3.0 * BALL_DIAMETER_PX   # ~55 px static gate radius
GATE_VEL_FACTOR   = 1.5                       # gate grows with speed
MAX_INNOVATION_PX = 60.0                      # hard physics ceiling per frame

# ── Track state machine ───────────────────────────────────────────────────────
TENTATIVE_HITS_REQUIRED = 3    # consecutive detections to confirm — 5 made
                               # tracking start visibly late after every reset
COAST_MAX_FRAMES        = 8    # predict-only frames before track drop

# ── Kalman noise ──────────────────────────────────────────────────────────────
KF_PROCESS_NOISE = 10.0
KF_MEAS_NOISE    = 6.0

# ── Camera motion ─────────────────────────────────────────────────────────────
PAN_THRESHOLD_PX  = 2.5        # median flow above this = panning, suppress shots
FLOW_DEADBAND_PX  = 0.5        # ignore sub-pixel flow noise (fixed camera drift)
FLOW_REFRESH      = 30         # re-pick corner features every N frames

# ── Trajectory prediction + shot decision ─────────────────────────────────────
PREDICT_SECONDS  = 1.2         # horizon; converted to frames from video fps
MIN_SHOT_SPEED   = 6.0         # px/frame — slow rolls never count as shots
FRICTION         = 0.985       # floor friction decay per predicted frame
VOTES_REQUIRED   = 3           # consecutive plane-hit frames to count a shot
COOLDOWN_FRAMES  = 45          # ~1.5 s between counted shots per goal

# ── Hindsight validation ──────────────────────────────────────────────────────
VALIDATE_FRAMES      = 24      # ~0.8 s of confirmed track after the count
VALIDATE_DIST_RATIO  = 0.60    # ball must come within 0.6 x plane width

# ── Goal plane (live SAM2-small on YOLO goalpost detections) ──────────────────
SAM2_UPDATE_STRIDE = 10        # run SAM2 on detected goalposts every N frames
PLANE_EMA_ALPHA    = 0.35      # weight of each new extraction (smoothing)
PLANE_TTL_FRAMES   = 12        # drop a plane whose goalpost bbox hasn't been
                               # detected this many frames — the plane exists
                               # ONLY while YOLO actually sees the goalpost
WHITE_HSV_LOW      = np.array([0, 0, 180])         # goalposts are white
WHITE_HSV_HIGH     = np.array([180, 55, 255])

# ── Rendering ─────────────────────────────────────────────────────────────────
TRAIL_MAX_LEN   = 25
TRAIL_EMA_ALPHA = 0.4          # render-only smoothing; never touches the track


# ══════════════════════════════════════════════════════════════════════════════
# 2.  MODEL WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════

class BallDetector:
    """RF-DETR Medium ball detector. Low threshold by design — the gate
    cascade (court mask, size, distance, physics) does the filtering
    that a high confidence threshold would do badly."""

    def __init__(self, weights_path: str):
        from rfdetr import RFDETRMedium
        self.model = RFDETRMedium(pretrain_weights=weights_path)
        try:
            self.model.optimize_for_inference()
            print("[INFO] RF-DETR optimize_for_inference() applied.")
        except Exception as exc:                          # noqa: BLE001
            print(f"[INFO] RF-DETR optimize skipped: {exc}")

    def detect(self, frame_bgr: np.ndarray
               ) -> List[Tuple[float, float, float, float, float]]:
        """Returns candidates as (cx, cy, w, h, conf) in pixel coords."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        dets = self.model.predict(rgb, threshold=BALL_CONF_THRESHOLD)
        out = []
        if dets is not None and len(dets) > 0:
            for (x1, y1, x2, y2), conf in zip(dets.xyxy, dets.confidence):
                out.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0,
                            float(x2 - x1), float(y2 - y1), float(conf)))
        return out


class PlayerDetector:
    """YOLO11-L: {0:'active_players', 1:'goal post', 2:'refree'}.
    fp16 on T4 for ~2x throughput."""

    CLASS_PLAYER, CLASS_GOALPOST, CLASS_REFEREE = 0, 1, 2

    def __init__(self, weights_path: str):
        self.model = YOLO(weights_path)
        self.half = DEVICE == "cuda"

    def detect(self, frame_bgr: np.ndarray) -> Dict[str, list]:
        out = {"players": [], "goalposts": [], "referees": []}
        results = self.model(frame_bgr, imgsz=640, device=DEVICE,
                             half=self.half, verbose=False)[0]
        for box in results.boxes:
            cls  = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if cls == self.CLASS_PLAYER and conf >= PLAYER_CONF:
                out["players"].append((x1, y1, x2, y2, conf))
            elif cls == self.CLASS_GOALPOST and conf >= GOALPOST_CONF:
                out["goalposts"].append((x1, y1, x2, y2, conf))
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 3.  COURT MASK (Gate 1)
# ══════════════════════════════════════════════════════════════════════════════

class CourtMask:
    """Binary mask of the playable court, built once from sampled frames.

    HSV-segment the (uniform blue) court, majority-vote across samples,
    keep the largest connected component, dilate for margin. Candidates
    outside it — scoreboard, stands, ad boards — are discarded before
    any tracker logic runs. The inverse mask doubles as the feature
    region for camera-motion estimation (off-court = static background).
    """

    def __init__(self, video_path: str):
        self.mask = self._build(video_path)
        self.off_court = cv2.bitwise_not(self.mask)

    @staticmethod
    def _build(video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"CourtMask: cannot open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, max(total - 1, 0), COURT_SAMPLES, dtype=int)

        acc = None
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            m = cv2.inRange(hsv, COURT_HSV_LOW, COURT_HSV_HIGH)
            acc = m.astype(np.uint16) if acc is None else acc + m
        cap.release()
        if acc is None:
            raise IOError("CourtMask: could not read any sample frames.")

        # Majority vote across samples
        mask = ((acc / 255) >= (COURT_SAMPLES / 2)).astype(np.uint8) * 255

        # Largest connected component = the court
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = np.where(labels == largest, 255, 0).astype(np.uint8)

        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, np.ones((COURT_DILATE_PX, COURT_DILATE_PX),
                                        np.uint8))

        coverage = mask.mean() / 255.0
        if coverage < 0.15:
            print(f"[WARNING] Court mask covers only {coverage:.0%} of frame — "
                  "HSV range may not match this venue. Falling back to "
                  "full-frame mask (Gate 1 disabled).")
            mask = np.full(mask.shape, 255, np.uint8)
        else:
            print(f"[INFO] Court mask built ({coverage:.0%} of frame).")
        return mask

    def contains(self, x: float, y: float) -> bool:
        h, w = self.mask.shape
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            return False
        return self.mask[yi, xi] > 0


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CAMERA MOTION ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════

class CameraMotion:
    """Per-frame global (pan) motion via median sparse optical flow on
    off-court features. Returns (dx, dy, panning). dx/dy below the
    deadband are zeroed so a fixed camera never accumulates drift."""

    def __init__(self, off_court_mask: np.ndarray):
        self.feature_mask = off_court_mask
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts:  Optional[np.ndarray] = None
        self.frames_since_refresh = 0

    def _pick_features(self, gray: np.ndarray) -> Optional[np.ndarray]:
        return cv2.goodFeaturesToTrack(
            gray, maxCorners=300, qualityLevel=0.01, minDistance=12,
            mask=self.feature_mask)

    def update(self, gray: np.ndarray) -> Tuple[float, float, bool]:
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = self._pick_features(gray)
            return 0.0, 0.0, False

        dx = dy = 0.0
        if self.prev_pts is not None and len(self.prev_pts) >= 10:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_pts, None,
                winSize=(21, 21), maxLevel=3)
            ok = status.reshape(-1) == 1
            if ok.sum() >= 10:
                flow = (new_pts[ok] - self.prev_pts[ok]).reshape(-1, 2)
                dx = float(np.median(flow[:, 0]))
                dy = float(np.median(flow[:, 1]))

        panning = math.hypot(dx, dy) > PAN_THRESHOLD_PX
        if math.hypot(dx, dy) < FLOW_DEADBAND_PX:
            dx = dy = 0.0

        self.prev_gray = gray
        self.frames_since_refresh += 1
        if self.frames_since_refresh >= FLOW_REFRESH or self.prev_pts is None:
            self.prev_pts = self._pick_features(gray)
            self.frames_since_refresh = 0
        elif dx or dy:
            self.prev_pts = self._pick_features(gray)

        return dx, dy, panning


# ══════════════════════════════════════════════════════════════════════════════
# 5.  GOAL PLANE — SAM2 calibration, per-side averaging, disk cache
# ══════════════════════════════════════════════════════════════════════════════

class GoalPlane:
    """A locked goal plane: 4 corners (TL, TR, BR, BL) in pixel coords.
    Shifted along with camera motion so it stays glued to the goal."""

    def __init__(self, side: str, corners: np.ndarray):
        self.side = side                                   # 'left' | 'right'
        self.polygon = corners.astype(np.float32)          # (4, 2)

    @property
    def width(self) -> float:
        return float(np.linalg.norm(self.polygon[1] - self.polygon[0]))

    @property
    def centroid(self) -> Tuple[float, float]:
        c = self.polygon.mean(axis=0)
        return float(c[0]), float(c[1])

    def shift(self, dx: float, dy: float) -> None:
        self.polygon[:, 0] += dx
        self.polygon[:, 1] += dy

    def contains(self, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(
            self.polygon.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0

    def distance(self, x: float, y: float) -> float:
        """Absolute distance to the polygon (0 if inside)."""
        d = cv2.pointPolygonTest(
            self.polygon.reshape(-1, 1, 2), (float(x), float(y)), True)
        return abs(float(d)) if d < 0 else 0.0

    def edges(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        p = self.polygon
        return [(p[i], p[(i + 1) % 4]) for i in range(4)]

    def draw(self, frame: np.ndarray, color=(0, 255, 150)) -> None:
        pts = self.polygon.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(frame, [pts], True, color, 2)
        cv2.putText(frame, f"Goal ({self.side})",
                    tuple(self.polygon[0].astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


class GoalPlaneEstimator:
    """Live SAM2-small goal-plane extraction — no locking, no disk cache.

    Whenever YOLO reports a goalpost (on the SAM2 stride), SAM2 segments
    it with the bbox as prompt; a white-pixel HSV filter keeps only post
    pixels SAM2 agreed on, convex hull + polygon approximation yields 4
    diagonal-extreme corners. side = left/right by bbox centre vs the
    frame midline. The plane is usable from the FIRST successful
    extraction onward, refines with a light EMA on every new one, and
    persists between extractions (shifted along with camera motion).
    """

    def __init__(self, frame_w: int):
        self.frame_w = frame_w
        self.planes: Dict[str, Optional[GoalPlane]] = {"left": None,
                                                       "right": None}
        self.last_seen: Dict[str, int] = {"left": -10**9, "right": -10**9}
        self._predictor = None

    # ── Detection-tied lifecycle ────────────────────────────────────────────
    def mark_seen(self, goalpost_bboxes: List[Tuple], frame_idx: int) -> None:
        """Record which sides have a goalpost bbox THIS detection frame."""
        for (x1, _y1, x2, _y2, _conf) in goalpost_bboxes:
            side = "left" if (x1 + x2) / 2 < self.frame_w / 2 else "right"
            self.last_seen[side] = frame_idx

    def expire(self, frame_idx: int) -> None:
        """Drop any plane whose goalpost bbox has not been detected for
        PLANE_TTL_FRAMES. This is what stops the plane travelling across
        the video: when the camera pans away and YOLO stops seeing the
        goal, the plane dies instead of persisting and accumulating
        camera-motion drift."""
        for side in ("left", "right"):
            if (self.planes[side] is not None
                    and frame_idx - self.last_seen[side] > PLANE_TTL_FRAMES):
                self.planes[side] = None
                print(f"[PLANE] Goal plane ({side}) dropped "
                      f"(goalpost not detected).")

    # ── SAM2-small (lazy-loaded on first goalpost detection) ───────────────
    def _get_predictor(self):
        if self._predictor is None:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            os.makedirs(SAM2_CKPT_DIR, exist_ok=True)
            ckpt = os.path.join(SAM2_CKPT_DIR, "sam2.1_hiera_small.pt")
            if not os.path.exists(ckpt):
                print("[INFO] Downloading SAM2-small checkpoint ...")
                urllib.request.urlretrieve(SAM2_CKPT_URL, ckpt)
            model = build_sam2(config_file=SAM2_MODEL_CFG, ckpt_path=ckpt,
                               device=DEVICE)
            self._predictor = SAM2ImagePredictor(model)
            print("[INFO] SAM2-small loaded for goal-plane estimation.")
        return self._predictor

    # ── Corner extraction (white filter + hull, per side) ──────────────────
    @staticmethod
    def _extract_corners(mask: np.ndarray, bbox: np.ndarray,
                         frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv_roi, WHITE_HSV_LOW, WHITE_HSV_HIGH)

        sam_roi = (mask[y1:y2, x1:x2] * 255).astype(np.uint8)
        combined = cv2.bitwise_and(sam_roi, white)
        kernel = np.ones((5, 5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        all_pts = np.vstack([c.reshape(-1, 2) for c in contours])
        all_pts[:, 0] += x1
        all_pts[:, 1] += y1
        if len(all_pts) < 4:
            return None

        hull = cv2.convexHull(all_pts.reshape(-1, 1, 2).astype(np.int32))
        approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
        pts = approx.reshape(-1, 2)
        if len(pts) < 4:
            pts = all_pts

        tl = pts[np.argmin(pts[:, 0] + pts[:, 1])]
        tr = pts[np.argmin(pts[:, 1] - pts[:, 0])]
        br = pts[np.argmax(pts[:, 0] + pts[:, 1])]
        bl = pts[np.argmax(pts[:, 1] - pts[:, 0])]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    # ── Public API ─────────────────────────────────────────────────────────
    @property
    def any_plane(self) -> bool:
        return any(p is not None for p in self.planes.values())

    def feed(self, frame_bgr: np.ndarray,
             goalpost_bboxes: List[Tuple]) -> None:
        """One SAM2 extraction per detected goalpost; EMA-refines the
        side's plane (or creates it on the first success)."""
        if not goalpost_bboxes:
            return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        predictor = self._get_predictor()
        predictor.set_image(rgb)

        for (x1, y1, x2, y2, _conf) in goalpost_bboxes:
            side = "left" if (x1 + x2) / 2 < self.frame_w / 2 else "right"
            bbox = np.array([int(x1), int(y1), int(x2), int(y2)])
            masks, scores, _ = predictor.predict(box=bbox,
                                                 multimask_output=True)
            mask = masks[int(np.argmax(scores))]
            corners = self._extract_corners(mask, bbox, frame_bgr)
            if corners is None:
                continue
            if self.planes[side] is None:
                self.planes[side] = GoalPlane(side, corners)
                print(f"[PLANE] Goal plane ({side}) acquired (SAM2-small).")
            else:
                old = self.planes[side].polygon
                jump = float(np.linalg.norm(corners.mean(axis=0)
                                            - old.mean(axis=0)))
                if jump > self.planes[side].width:
                    # New extraction is far from the old plane (pan,
                    # drift, or a previous false positive): REPLACE —
                    # blending would smear the plane across the court.
                    self.planes[side].polygon = corners.astype(np.float32)
                else:
                    self.planes[side].polygon = (
                        PLANE_EMA_ALPHA * corners
                        + (1 - PLANE_EMA_ALPHA) * old).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  BALL TRACK — gated cascade + state machine
# ══════════════════════════════════════════════════════════════════════════════

class TrackState(Enum):
    INACTIVE  = auto()   # no track; collecting candidates
    TENTATIVE = auto()   # provisional; needs consecutive consistent hits
    CONFIRMED = auto()   # live; Kalman updates, trail renders
    COASTING  = auto()   # miss; predict internally only, never drawn


class BallTrack:
    """Distance-gated, physics-validated single-ball tracker.

    Association is nearest-candidate-within-adaptive-gate — IoU is
    meaningless for an 18 px object that moves more than its own
    diameter per frame. The state machine counts CONSECUTIVE hits, so
    the system always knows how trustworthy its current state is:
    a single-frame false positive can never start a real track, and a
    track that just coasted through an occlusion knows its velocity
    needs re-earning before the kick detector may trust it.
    """

    def __init__(self):
        self.state = TrackState.INACTIVE
        self.kf: Optional[KalmanFilter] = None
        self.consecutive_hits = 0
        self.coast_frames = 0
        self.last_size: Optional[float] = None
        self.tentative_buf: List[Tuple[float, float]] = []
        self.velocity: Tuple[float, float] = (0.0, 0.0)

    # ── Kalman helpers ─────────────────────────────────────────────────────
    def _init_kf(self, cx: float, cy: float, vx: float, vy: float) -> None:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                         [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        kf.R *= KF_MEAS_NOISE ** 2
        kf.Q *= KF_PROCESS_NOISE
        kf.P *= 100
        kf.x = np.array([[cx], [cy], [vx], [vy]])
        self.kf = kf

    def _kf_pos(self) -> Tuple[float, float]:
        return float(self.kf.x[0]), float(self.kf.x[1])

    # ── Camera-motion compensation ─────────────────────────────────────────
    def apply_camera_shift(self, dx: float, dy: float) -> None:
        """Shift track state with the camera so association happens in
        current-frame pixel coords (translation-only GMC)."""
        if self.kf is not None:
            self.kf.x[0] += dx
            self.kf.x[1] += dy
        self.tentative_buf = [(x + dx, y + dy) for x, y in self.tentative_buf]

    # ── Gates ──────────────────────────────────────────────────────────────
    @staticmethod
    def _size_ok(w: float, h: float) -> bool:
        """Gate 2: 0.5x..2x expected diameter; loose upper bound because
        motion blur stretches a fast ball into an ellipse."""
        longest, shortest = max(w, h), min(w, h)
        return (shortest >= BALL_SIZE_MIN * BALL_DIAMETER_PX and
                longest <= BALL_SIZE_MAX * BALL_DIAMETER_PX * 1.25)

    def _gate_radius(self) -> float:
        """Gate 3: adaptive — tight when slow (nothing far away can
        hijack the track), wide when fast (keeps up with shots)."""
        speed = math.hypot(*self.velocity)
        return GATE_BASE_PX + GATE_VEL_FACTOR * speed

    def _select_candidate(self, candidates: List[Tuple],
                          anchor: Tuple[float, float]
                          ) -> Optional[Tuple[float, float, float]]:
        """Nearest in-gate candidate, scored by distance + confidence +
        size similarity. Returns (cx, cy, size) or None."""
        gate = self._gate_radius()
        best, best_score = None, float("inf")
        for (cx, cy, w, h, conf) in candidates:
            dist = math.hypot(cx - anchor[0], cy - anchor[1])
            if dist > gate:
                continue
            size = (w + h) / 2.0
            size_pen = (abs(size - self.last_size) / self.last_size
                        if self.last_size else 0.0)
            score = (dist / gate) + 0.5 * (1.0 - conf) + 0.3 * size_pen
            if score < best_score:
                best_score, best = score, (cx, cy, size)
        return best

    # ── Main per-frame step ────────────────────────────────────────────────
    def step(self, candidates: List[Tuple]
             ) -> Optional[Tuple[float, float]]:
        """Feed gated candidates (already court- and size-filtered).
        Returns the confirmed ball position for this frame, or None
        (tentative/coasting/inactive positions are internal only)."""
        candidates = [c for c in candidates if self._size_ok(c[2], c[3])]

        if self.state == TrackState.INACTIVE:
            return self._step_inactive(candidates)
        if self.state == TrackState.TENTATIVE:
            return self._step_tentative(candidates)
        return self._step_confirmed_or_coasting(candidates)

    def _step_inactive(self, candidates) -> None:
        if candidates:
            best = max(candidates, key=lambda c: c[4])
            self.tentative_buf = [(best[0], best[1])]
            self.last_size = (best[2] + best[3]) / 2.0
            self.state = TrackState.TENTATIVE
        return None

    def _step_tentative(self, candidates) -> None:
        anchor = self.tentative_buf[-1]
        picked = self._select_candidate(candidates, anchor)
        if picked is None:
            # one miss kills a provisional track — re-acquire from scratch
            self._reset()
            return None
        cx, cy, size = picked
        self.tentative_buf.append((cx, cy))
        self.last_size = size
        if len(self.tentative_buf) >= TENTATIVE_HITS_REQUIRED:
            # Promote: init Kalman with velocity from the tentative run
            # (fixes the zero-velocity cold-start problem)
            (x0, y0), (x1, y1) = self.tentative_buf[0], self.tentative_buf[-1]
            n = len(self.tentative_buf) - 1
            self._init_kf(x1, y1, (x1 - x0) / n, (y1 - y0) / n)
            self.velocity = (float(self.kf.x[2]), float(self.kf.x[3]))
            self.consecutive_hits = TENTATIVE_HITS_REQUIRED
            self.state = TrackState.CONFIRMED
        return None

    def _step_confirmed_or_coasting(self, candidates
                                    ) -> Optional[Tuple[float, float]]:
        self.kf.predict()
        anchor = self._kf_pos()
        picked = self._select_candidate(candidates, anchor)

        if picked is not None:
            cx, cy, size = picked
            # Gate 4 — physics: gate on POSITION innovation. Kicks are
            # huge accelerations and pass; teleports are huge position
            # jumps and never do.
            innovation = math.hypot(cx - anchor[0], cy - anchor[1])
            if innovation > MAX_INNOVATION_PX:
                picked = None

        if picked is None:
            # Miss -> coast (predict-only) up to COAST_MAX_FRAMES
            self.consecutive_hits = 0
            self.coast_frames += 1
            self.state = TrackState.COASTING
            if self.coast_frames > COAST_MAX_FRAMES:
                self._reset()
            return None

        cx, cy, size = picked
        self.kf.update(np.array([[cx], [cy]]))
        self.velocity = (float(self.kf.x[2]), float(self.kf.x[3]))
        self.last_size = size
        self.coast_frames = 0
        self.consecutive_hits += 1
        self.state = TrackState.CONFIRMED
        return self._kf_pos()

    def _reset(self) -> None:
        self.__init__()

    @property
    def speed(self) -> float:
        return math.hypot(*self.velocity)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  EVENT-TRIGGERED SHOT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def predict_with_friction(pos: Tuple[float, float],
                          vel: Tuple[float, float],
                          n_frames: int
                          ) -> List[Tuple[float, float]]:
    """Constant velocity x floor-friction decay. On a futsal floor a
    kicked ball travels essentially straight with mild deceleration."""
    px, py = pos
    vx, vy = vel
    out = []
    for _ in range(n_frames):
        vx *= FRICTION
        vy *= FRICTION
        px += vx
        py += vy
        out.append((px, py))
    return out


def _segments_intersect(p1, p2, p3, p4) -> bool:
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4) and
            ccw(p1, p2, p3) != ccw(p1, p2, p4))


def _path_hits_plane(start: Tuple[float, float],
                     predicted: List[Tuple[float, float]],
                     plane: GoalPlane) -> bool:
    """True if the predicted path enters the goal-plane POLYGON —
    point-in-polygon is what makes 'on target' mean on target, not
    merely crossing an infinite extended line."""
    prev = np.array(start)
    for pt in predicted:
        if plane.contains(pt[0], pt[1]):
            return True
        curr = np.array(pt)
        for (e1, e2) in plane.edges():
            if _segments_intersect(prev, curr, e1, e2):
                return True
        prev = curr
    return False


@dataclass
class PendingShot:
    """A provisionally-counted shot awaiting hindsight validation: did
    the ball genuinely arrive near the plane in the next ~0.8 s, or was
    the prediction contradicted by reality? Predicted-only counts are
    always inflated — this converts the metric to verified shots."""
    side: str
    frame_idx: int
    plane_width: float
    frames_left: int = VALIDATE_FRAMES
    min_dist: float = float("inf")
    resolved: bool = False
    validated: bool = False

    def observe(self, dist_to_plane: float) -> None:
        self.min_dist = min(self.min_dist, dist_to_plane)

    def tick(self) -> None:
        self.frames_left -= 1
        if self.frames_left <= 0 and not self.resolved:
            self.resolved = True
            self.validated = (self.min_dist
                              <= VALIDATE_DIST_RATIO * self.plane_width)


class ShotCounter:
    """Per-side provisional + validated counters with cooldown."""

    def __init__(self):
        self.provisional = {"left": 0, "right": 0}
        self.validated   = {"left": 0, "right": 0}
        self._last_count_frame = {"left": -10**9, "right": -10**9}
        self.pending: List[PendingShot] = []

    def in_cooldown(self, side: str, frame_idx: int) -> bool:
        return frame_idx - self._last_count_frame[side] < COOLDOWN_FRAMES

    def count(self, side: str, frame_idx: int, plane: GoalPlane) -> None:
        self.provisional[side] += 1
        self._last_count_frame[side] = frame_idx
        self.pending.append(PendingShot(side=side, frame_idx=frame_idx,
                                        plane_width=plane.width))

    def update_pending(self, ball_pos: Optional[Tuple[float, float]],
                       planes: Dict[str, Optional[GoalPlane]]) -> None:
        for p in self.pending:
            if p.resolved:
                continue
            plane = planes.get(p.side)
            if ball_pos is not None and plane is not None:
                p.observe(plane.distance(*ball_pos))
            p.tick()
            if p.resolved and p.validated:
                self.validated[p.side] += 1

    def finalize(self) -> None:
        """End of video: unresolved windows resolve on what they saw."""
        for p in self.pending:
            if not p.resolved:
                p.resolved = True
                p.validated = (p.min_dist
                               <= VALIDATE_DIST_RATIO * p.plane_width)
                if p.validated:
                    self.validated[p.side] += 1


# ══════════════════════════════════════════════════════════════════════════════
# 8.  RENDERING
# ══════════════════════════════════════════════════════════════════════════════

class BallTrail:
    """Confirmed measurements ONLY — never coasted predictions, never
    tentative detections. Light render-only EMA for visual smoothness;
    a None break marker on track loss so the trail visibly ends instead
    of connecting to wherever the ball reappears."""

    def __init__(self, color=(0, 220, 255)):
        self.trail: collections.deque = collections.deque(maxlen=TRAIL_MAX_LEN)
        self.color = color
        self._ema: Optional[Tuple[float, float]] = None

    def push_confirmed(self, pos: Tuple[float, float]) -> None:
        if self._ema is None:
            self._ema = pos
        else:
            self._ema = (TRAIL_EMA_ALPHA * pos[0]
                         + (1 - TRAIL_EMA_ALPHA) * self._ema[0],
                         TRAIL_EMA_ALPHA * pos[1]
                         + (1 - TRAIL_EMA_ALPHA) * self._ema[1])
        self.trail.append(self._ema)

    def push_break(self) -> None:
        self._ema = None
        if self.trail and self.trail[-1] is not None:
            self.trail.append(None)

    def shift(self, dx: float, dy: float) -> None:
        shifted = collections.deque(maxlen=TRAIL_MAX_LEN)
        for p in self.trail:
            shifted.append(None if p is None else (p[0] + dx, p[1] + dy))
        self.trail = shifted
        if self._ema is not None:
            self._ema = (self._ema[0] + dx, self._ema[1] + dy)

    def draw(self, frame: np.ndarray) -> None:
        pts = list(self.trail)
        n = len(pts)
        for i in range(1, n):
            p0, p1 = pts[i - 1], pts[i]
            if p0 is None or p1 is None:
                continue
            age = i / n
            col = tuple(int(c * (0.25 + 0.75 * age)) for c in self.color)
            cv2.line(frame, (int(p0[0]), int(p0[1])),
                     (int(p1[0]), int(p1[1])), col,
                     max(1, int(round(4 * age))), cv2.LINE_AA)


def draw_prediction(frame: np.ndarray,
                    predicted: List[Tuple[float, float]],
                    start: Tuple[float, float]) -> None:
    """Dashed red ray — the 1-second predicted trajectory."""
    prev = start
    for i, pt in enumerate(predicted):
        if i % 2 == 0:
            cv2.line(frame, (int(prev[0]), int(prev[1])),
                     (int(pt[0]), int(pt[1])), (0, 60, 255), 2, cv2.LINE_AA)
        prev = pt


def draw_hud(frame: np.ndarray, counter: ShotCounter,
             frame_idx: int, total: int, panning: bool) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 56), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    text = (f"Shots on Target  L: {counter.provisional['left']}"
            f" ({counter.validated['left']} ok)   "
            f"R: {counter.provisional['right']}"
            f" ({counter.validated['right']} ok)")
    cv2.putText(frame, text, (12, 36), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                (0, 255, 180), 2, cv2.LINE_AA)
    if panning:
        cv2.putText(frame, "PAN", (w - 90, 36), cv2.FONT_HERSHEY_DUPLEX,
                    0.7, (0, 160, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{frame_idx}/{total}", (w - 220, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (200, 200, 200), 1,
                cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run() -> None:
    print(f"[INFO] Device: {DEVICE}")

    print("[INFO] Building court mask ...")
    court = CourtMask(VIDEO_PATH)

    print("[INFO] Loading ball detector (RF-DETR Medium) ...")
    ball_det = BallDetector(BALL_MODEL_PATH)
    print("[INFO] Loading player/goalpost detector (YOLO11-L) ...")
    player_det = PlayerDetector(PLAYER_MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {VIDEO_PATH}")
    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {frame_w}x{frame_h} @ {src_fps:.1f} fps, "
          f"{n_frames} frames")

    writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"),
                             src_fps, (frame_w, frame_h))

    cam_motion = CameraMotion(court.off_court)
    plane_est  = GoalPlaneEstimator(frame_w)
    track      = BallTrack()
    trail      = BallTrail()
    counter    = ShotCounter()
    shot_votes = {"left": 0, "right": 0}
    predict_frames = max(1, int(round(src_fps * PREDICT_SECONDS)))
    print(f"[INFO] Trajectory horizon: {predict_frames} frames "
          f"({PREDICT_SECONDS:.1f} s)")
    last_players: Dict[str, list] = {"players": [], "goalposts": []}

    frame_idx = 0
    pbar = tqdm(total=n_frames, desc="Processing")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ── Camera motion: shift everything that lives in pixel coords ──────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dx, dy, panning = cam_motion.update(gray)
        if dx or dy:
            track.apply_camera_shift(dx, dy)
            trail.shift(dx, dy)
            for plane in plane_est.planes.values():
                if plane is not None:
                    plane.shift(dx, dy)

        # ── Players / goalposts (strided — decoration + plane lifecycle) ────
        if frame_idx % PLAYER_DET_STRIDE == 1 or PLAYER_DET_STRIDE == 1:
            last_players = player_det.detect(frame)
            plane_est.mark_seen(last_players["goalposts"], frame_idx)

        # ── Goal plane: SAM2-small ONLY on frames with a goalpost bbox ──────
        if frame_idx % SAM2_UPDATE_STRIDE == 0 and last_players["goalposts"]:
            plane_est.feed(frame, last_players["goalposts"])
        plane_est.expire(frame_idx)

        # ── Ball: detect -> Gate 1 (court) -> track (Gates 2-4 inside) ──────
        candidates = [c for c in ball_det.detect(frame)
                      if court.contains(c[0], c[1])]
        confirmed_pos = track.step(candidates)

        if confirmed_pos is not None:
            trail.push_confirmed(confirmed_pos)
        elif track.state == TrackState.INACTIVE:
            trail.push_break()
        # COASTING/TENTATIVE: push nothing — coasted predictions are
        # internal only; the trail must contain confirmed measurements.

        # ── Trajectory prediction + shot decision (plane-gated) ─────────────
        # Once a goal plane is locked, every confirmed ball frame gets a
        # 1-second trajectory prediction. A shot is counted when the
        # predicted path hits a plane on VOTES_REQUIRED consecutive
        # frames while the ball moves toward that goal above MIN_SHOT_SPEED.
        shot_fired = False
        prediction: List[Tuple[float, float]] = []

        # The speed gate doubles as the post-block cut: a blocked/saved
        # ball decelerates below MIN_SHOT_SPEED within a couple of
        # frames, so the stale velocity ray is never drawn after a block.
        if (confirmed_pos is not None and not panning
                and plane_est.any_plane and track.speed > MIN_SHOT_SPEED):
            prediction = predict_with_friction(confirmed_pos, track.velocity,
                                               predict_frames)
            vx, vy = track.velocity
            for side, plane in plane_est.planes.items():
                if plane is None:
                    continue
                gcx, gcy = plane.centroid
                toward = ((gcx - confirmed_pos[0]) * vx
                          + (gcy - confirmed_pos[1]) * vy) > 0
                if toward and _path_hits_plane(confirmed_pos, prediction,
                                               plane):
                    shot_votes[side] += 1
                else:
                    shot_votes[side] = 0

                if (shot_votes[side] >= VOTES_REQUIRED
                        and not counter.in_cooldown(side, frame_idx)):
                    counter.count(side, frame_idx, plane)
                    shot_votes[side] = 0
                    shot_fired = True
                    print(f"\n[SHOT] Frame {frame_idx}: shot on "
                          f"{side.upper()} goal (provisional).")
        else:
            shot_votes = {"left": 0, "right": 0}

        # ── Hindsight validation of provisional shots ────────────────────────
        counter.update_pending(confirmed_pos, plane_est.planes)

        # ── Draw ─────────────────────────────────────────────────────────────
        for plane in plane_est.planes.values():
            if plane is not None:
                plane.draw(frame, (0, 255, 100) if plane.side == "left"
                           else (0, 100, 255))
        for (x1, y1, x2, y2, _c) in last_players["players"]:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (220, 100, 0), 2)
        trail.draw(frame)
        if prediction and confirmed_pos is not None:
            draw_prediction(frame, prediction, confirmed_pos)
        if confirmed_pos is not None:
            fx, fy = int(confirmed_pos[0]), int(confirmed_pos[1])
            cv2.circle(frame, (fx, fy), 8, (0, 220, 255), -1)
            cv2.circle(frame, (fx, fy), 8, (0, 0, 0), 1)
        if shot_fired:
            cv2.rectangle(frame, (0, 0), (frame_w, frame_h), (0, 0, 255), 12)
            cv2.putText(frame, "SHOT ON TARGET!", (frame_w // 4, frame_h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 5,
                        cv2.LINE_AA)
        draw_hud(frame, counter, frame_idx, n_frames, panning)

        writer.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    counter.finalize()

    # ── Final report: provisional (live overlay) vs validated (the truth) ───
    print("\n" + "=" * 60)
    print("SHOTS ON TARGET — FINAL REPORT")
    print("=" * 60)
    for side in ("left", "right"):
        print(f"  {side.capitalize():>5} goal: "
              f"provisional {counter.provisional[side]}  |  "
              f"validated {counter.validated[side]}")
    total_p = sum(counter.provisional.values())
    total_v = sum(counter.validated.values())
    print(f"  {'Total':>5}     : provisional {total_p}  |  "
          f"validated {total_v}")
    print(f"\n[DONE] Annotated video: {OUTPUT_PATH}")


run()
