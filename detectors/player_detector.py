"""
detectors/player_detector.py — GSFA Player Detection Module

Wraps GSFA_PLAYER_DETECTION.pt (YOLOv11, 3 classes):
    0 → active_player
    1 → goal_post
    2 → referee

Import:
    from detectors.player_detector import PlayerDetector

Quick start:
    detector = PlayerDetector()

    # For homography (2 fps — fast)
    detections = detector.process_video(video_path, sample_every=15)

    # For player heatmap/tracking (every frame)
    detections = detector.process_video(video_path, sample_every=1)

    # Single frame
    dets = detector.detect(frame, frame_idx=0, fps=30.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# OUTPUT TYPES
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Single detected object in one frame."""
    class_id:     int
    class_name:   str                        # "active_player" | "referee" | "goal_post"
    bbox:         tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence:   float
    foot_point:   tuple[int, int]            # bottom-centre — use for ground-plane projection
    centre_point: tuple[int, int]            # bbox centre
    team_id:      Optional[int] = None       # 0 or 1 — set by TeamClassifier; None = unclassified
    is_goalkeeper: bool         = False      # set by GoalkeeperDetector
    track_id:     Optional[int] = None       # set by PlayerTracker
    embedding:    Optional[np.ndarray] = None  # 768-D SigLIP, set by GSFATeamClassifier.classify


@dataclass
class FrameDetections:
    """All detections for one video frame."""
    frame_idx:   int
    timestamp_s: float
    players:     list[Detection] = field(default_factory=list)
    referees:    list[Detection] = field(default_factory=list)
    goal_posts:  list[Detection] = field(default_factory=list)
    all:         list[Detection] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DETECTOR
# ---------------------------------------------------------------------------

class PlayerDetector:

    MODEL_PATH = r"C:\Users\Admin\OneDrive\Desktop\CZ\GSFA_PLAYER_DETECTION.pt"

    CLASS_NAMES: dict[int, str] = {
        0: "active_player",
        1: "goal_post",
        2: "referee",
    }

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        conf: float = 0.50,
        device: str = "cpu",
    ) -> None:
        self.conf   = conf
        self.device = device
        self.model  = YOLO(model_path)

    # ------------------------------------------------------------------
    # Single frame
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        fps: float = 30.0,
    ) -> FrameDetections:
        """Run inference on one BGR frame. Returns FrameDetections."""
        results = self.model(frame, conf=self.conf, device=self.device, verbose=False)
        return self._parse(results[0], frame_idx, fps)

    # ------------------------------------------------------------------
    # Full video
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        sample_every: int = 1,
        progress: bool = True,
    ) -> list[FrameDetections]:
        """
        Run inference on a video file and return per-frame detections.

        Args:
            video_path:   Path to the input video.
            sample_every: Run inference every Nth frame.
                          sample_every=1  → every frame  (30 fps — use for heatmap/tracking)
                          sample_every=15 → every 15th   (2 fps  — use for homography)
            progress:     Print progress every 200 sampled frames.

        Returns:
            List of FrameDetections — only sampled frames are included.
            Frames skipped by sample_every are NOT in the list.
            Use .frame_idx to know which frame each entry belongs to.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"PlayerDetector: cannot open video: {video_path}")

        fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_f   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results_out: list[FrameDetections] = []
        fidx      = 0
        sampled   = 0

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
                    print(f"  [PlayerDetector] frame {fidx:>5}/{total_f} ({pct:.0f}%)  "
                          f"sampled={sampled}")

            fidx += 1

        cap.release()
        return results_out

    # ------------------------------------------------------------------
    # Visualisation helper
    # ------------------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        detections: FrameDetections,
        show_conf: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes on a copy of the frame.
            active_player → green
            referee       → yellow
            goal_post     → cyan

        Returns annotated copy (original unchanged).
        """
        COLOURS = {
            "active_player": (0, 255, 0),
            "referee":       (0, 215, 255),
            "goal_post":     (255, 215, 0),
        }
        out = frame.copy()
        for det in detections.all:
            x1, y1, x2, y2 = det.bbox
            colour = COLOURS.get(det.class_name, (200, 200, 200))
            label  = det.class_name + (f" {det.confidence:.2f}" if show_conf else "")
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(out, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
            cv2.circle(out, det.foot_point, 4, colour, -1)
        return out

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse(self, r, frame_idx: int, fps: float) -> FrameDetections:
        fd = FrameDetections(frame_idx=frame_idx, timestamp_s=frame_idx / max(fps, 1.0))
        if r.boxes is None or len(r.boxes) == 0:
            return fd
        for box in r.boxes:
            cid = int(box.cls[0].cpu().numpy())
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
