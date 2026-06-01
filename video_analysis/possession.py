"""
video_analysis/possession.py — Ball possession analysis for GSFA football matches.

Produces an annotated output video showing:
  • Supervision ellipse circles under each player's feet (team-coloured, no full bboxes)
  • Running possession percentage bar across the top of the frame
  • Pass stats overlay (successful vs inaccurate passes per team)

Run:
    python video_analysis/possession.py
"""

from __future__ import annotations

import math
import sys
from collections import deque
from dataclasses import dataclass
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
VIDEO_PATH         = r"C:\Users\Admin\Downloads\Video Project 8.mp4"
OUTPUT_PATH        = r"data/output/possession_output.mp4"

BALL_CLASS_ID   = 1
BALL_CONF       = 0.35
COAST_FRAMES    = 8
HYSTERESIS_N    = 5
PIXEL_PZ        = 80    # possession zone radius px
PIXEL_DZ        = 130   # duel zone radius px
MIN_HOLD_FRAMES = 25    # consecutive frames required before registering a possessor
BBOX_EXPAND_PCT = 0.05  # expand bboxes by 5% for overlap check
POSSESS_BAR_H   = 42    # pixel height of the possession bar at top of frame

POSSESS_TEAM0   = "team0"
POSSESS_TEAM1   = "team1"
POSSESS_DUEL    = "duel"
POSSESS_LOOSE   = "loose"
POSSESS_UNKNOWN = "unknown"

# BGR colours
TEAM_BGR  = {0: (255, 80, 0), 1: (0, 80, 255), None: (160, 160, 160)}
GK_BGR    = (0, 215, 255)    # gold
BALL_BGR  = (0, 255, 255)    # cyan
REF_BGR   = (80, 220, 80)    # light green

# ---------------------------------------------------------------------------
# BALL DETECTION
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


# ---------------------------------------------------------------------------
# BALL STATE MACHINE
# ---------------------------------------------------------------------------

class BallStateMachine:
    DETECTED = "detected"
    COASTING = "coasting"
    LOST     = "lost"

    def __init__(self, coast_frames: int = COAST_FRAMES) -> None:
        self.coast_frames = coast_frames
        self.state        = self.LOST
        self.last_ball:   Optional[BallDetection] = None
        self.missing_for  = 0

    def update(
        self,
        ball_det: Optional[BallDetection],
    ) -> tuple[str, Optional[BallDetection]]:
        if ball_det is not None:
            self.state       = self.DETECTED
            self.missing_for = 0
            self.last_ball   = ball_det
        else:
            self.missing_for += 1
            if self.missing_for <= self.coast_frames:
                self.state = self.COASTING
            else:
                self.state     = self.LOST
                self.last_ball = None
        return self.state, self.last_ball


# ---------------------------------------------------------------------------
# POSSESSION ENGINE
# ---------------------------------------------------------------------------

def _expand_bbox(bbox: tuple, pct: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    dw = int((x2 - x1) * pct)
    dh = int((y2 - y1) * pct)
    return (x1 - dw, y1 - dh, x2 + dw, y2 + dh)


def _boxes_overlap(a: tuple, b: tuple) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


class PossessionEngine:
    def __init__(
        self,
        hysteresis_n: int   = HYSTERESIS_N,
        expand_pct:   float = BBOX_EXPAND_PCT,
        pixel_pz:     int   = PIXEL_PZ,
        pixel_dz:     int   = PIXEL_DZ,
    ) -> None:
        self.hysteresis_n = hysteresis_n
        self.expand_pct   = expand_pct
        self.pixel_pz     = pixel_pz
        self.pixel_dz     = pixel_dz
        self._buffer:     deque[str]             = deque(maxlen=hysteresis_n)
        self._committed:  str                    = POSSESS_UNKNOWN
        self._possessing: Optional[Detection]    = None

    def get_possessing_player(self) -> Optional[Detection]:
        return self._possessing

    def update(
        self,
        players:    list[Detection],
        ball_state: str,
        ball_det:   Optional[BallDetection],
    ) -> tuple[str, str]:
        if ball_state == BallStateMachine.LOST or ball_det is None:
            self._buffer.clear()
            self._committed  = POSSESS_UNKNOWN
            self._possessing = None
            return POSSESS_UNKNOWN, "none"

        raw, method, raw_player = self._raw_possession(players, ball_det)

        if raw in (POSSESS_LOOSE, POSSESS_UNKNOWN):
            self._buffer.clear()
            self._committed  = raw
            self._possessing = None
        else:
            self._buffer.append(raw)
            if len(self._buffer) == self.hysteresis_n and len(set(self._buffer)) == 1:
                self._committed  = self._buffer[0]
                self._possessing = (
                    raw_player
                    if self._committed in (POSSESS_TEAM0, POSSESS_TEAM1)
                    else None
                )

        return self._committed, method

    def _raw_possession(
        self,
        players:  list[Detection],
        ball_det: BallDetection,
    ) -> tuple[str, str, Optional[Detection]]:
        active   = [p for p in players if p.team_id in (0, 1)]
        ball_exp = _expand_bbox(ball_det.bbox, self.expand_pct)

        # Stage 1 — bounding box overlap
        overlapping: list[Detection] = [
            p for p in active
            if _boxes_overlap(_expand_bbox(p.bbox, self.expand_pct), ball_exp)
        ]
        if len(overlapping) == 1:
            p = overlapping[0]
            return (POSSESS_TEAM0 if p.team_id == 0 else POSSESS_TEAM1), "bbox", p
        if len(overlapping) >= 2:
            teams = {p.team_id for p in overlapping}
            if len(teams) == 1:
                p = overlapping[0]
                return (POSSESS_TEAM0 if p.team_id == 0 else POSSESS_TEAM1), "bbox", p
            return POSSESS_DUEL, "bbox", None

        # Stage 2 — pixel distance (PZ / DZ)
        bx, by = ball_det.centre
        dists  = sorted(
            [(math.dist((p.foot_point[0], p.foot_point[1]), (bx, by)), p) for p in active],
            key=lambda x: x[0],
        )
        in_pz = [(d, p) for d, p in dists if d <= self.pixel_pz]
        if len(in_pz) == 1:
            p = in_pz[0][1]
            return (POSSESS_TEAM0 if p.team_id == 0 else POSSESS_TEAM1), "px", p
        if len(in_pz) > 1:
            teams_pz = {p.team_id for _, p in in_pz}
            if len(teams_pz) == 1:
                p = in_pz[0][1]
                return (POSSESS_TEAM0 if p.team_id == 0 else POSSESS_TEAM1), "px", p
            return POSSESS_DUEL, "px", None

        in_dz = [(d, p) for d, p in dists if d <= self.pixel_dz]
        if in_dz:
            teams_dz = {p.team_id for _, p in in_dz}
            if len(teams_dz) > 1:
                return POSSESS_DUEL, "px", None
            p = in_dz[0][1]
            return (POSSESS_TEAM0 if p.team_id == 0 else POSSESS_TEAM1), "px", p

        return POSSESS_LOOSE, "px", None


# ---------------------------------------------------------------------------
# PASS TRACKER
# ---------------------------------------------------------------------------

class PassTracker:
    def __init__(self, min_hold_frames: int = MIN_HOLD_FRAMES) -> None:
        self.min_hold_frames = min_hold_frames
        self._last_team:  Optional[int]   = None
        self._last_foot:  Optional[tuple] = None
        self._hold_count: int             = 0
        self.passes: dict[int, dict[str, int]] = {
            0: {"successful": 0, "inaccurate": 0},
            1: {"successful": 0, "inaccurate": 0},
        }

    def update(
        self,
        possession_label:  str,
        possessing_player: Optional[Detection],
    ) -> None:
        if possession_label not in (POSSESS_TEAM0, POSSESS_TEAM1) or possessing_player is None:
            return

        cur_team = possessing_player.team_id
        cur_foot = possessing_player.foot_point

        if self._last_foot is None:
            self._last_team  = cur_team
            self._last_foot  = cur_foot
            self._hold_count = 1
            return

        same_player = math.dist(cur_foot, self._last_foot) < 90

        if same_player:
            self._hold_count += 1
        else:
            if self._hold_count >= self.min_hold_frames and self._last_team is not None:
                if cur_team == self._last_team:
                    self.passes[self._last_team]["successful"] += 1
                else:
                    self.passes[self._last_team]["inaccurate"] += 1
            self._last_team  = cur_team
            self._last_foot  = cur_foot
            self._hold_count = 1

    def summary(self) -> str:
        lines = ["--- Pass Summary ---"]
        for tid in (0, 1):
            s   = self.passes[tid]["successful"]
            i   = self.passes[tid]["inaccurate"]
            acc = f"{100 * s / max(1, s + i):.0f}%"
            lines.append(f"  Team {tid}: {s} successful  {i} inaccurate  ({acc} accuracy)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# POSSESSION STATS
# ---------------------------------------------------------------------------

class PossessionStats:
    def __init__(self) -> None:
        self.frame_counts: dict[str, int] = {
            POSSESS_TEAM0: 0, POSSESS_TEAM1: 0,
            POSSESS_DUEL:  0, POSSESS_LOOSE:  0, POSSESS_UNKNOWN: 0,
        }
        self.total = 0

    def update(self, label: str) -> None:
        self.frame_counts[label] = self.frame_counts.get(label, 0) + 1
        self.total += 1

    def percentages(self) -> tuple[float, float]:
        active = (
            self.frame_counts[POSSESS_TEAM0]
            + self.frame_counts[POSSESS_TEAM1]
            + self.frame_counts[POSSESS_DUEL]
        )
        if active == 0:
            return 0.0, 0.0
        return (
            100.0 * self.frame_counts[POSSESS_TEAM0] / active,
            100.0 * self.frame_counts[POSSESS_TEAM1] / active,
        )

    def summary(self) -> str:
        t0, t1 = self.percentages()
        return "\n".join([
            "--- Possession Summary ---",
            f"  Team 0 : {t0:.1f}%",
            f"  Team 1 : {t1:.1f}%",
            f"  Duel   : {100*self.frame_counts[POSSESS_DUEL]/max(1,self.total):.1f}%",
            f"  Loose  : {100*self.frame_counts[POSSESS_LOOSE]/max(1,self.total):.1f}%",
        ])


# ---------------------------------------------------------------------------
# SUPERVISION ANNOTATORS
# ---------------------------------------------------------------------------
# Index 0 = team0 (blue), Index 1 = team1 (red), Index 2 = unclassified (grey)
_PALETTE = sv.ColorPalette.from_hex(["#0050FF", "#FF5000", "#A0A0A0"])

_ellipse_ann   = sv.EllipseAnnotator(color=_PALETTE, thickness=2)
_triangle_ann  = sv.TriangleAnnotator(color=sv.Color.from_hex("#00FFFF"), base=16, height=16, color_lookup=sv.ColorLookup.INDEX)
_label_ann   = sv.LabelAnnotator(
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
    frame:            np.ndarray,
    player_dets:      FrameDetections,
    ball_det:         Optional[BallDetection],
    possession_label: str,
    stats:            PossessionStats,
) -> np.ndarray:
    out        = frame.copy()
    H_vid, W_vid = out.shape[:2]
    font       = cv2.FONT_HERSHEY_SIMPLEX

    # --- Player ellipses via supervision ---
    sv_dets, labels = _to_sv(player_dets.players)
    if len(sv_dets) > 0:
        out = _ellipse_ann.annotate(scene=out, detections=sv_dets)
        out = _label_ann.annotate(scene=out, detections=sv_dets, labels=labels)

    # GK: redraw ellipse in gold to override team colour
    for p in player_dets.players:
        if p.is_goalkeeper:
            x1, y1, x2, y2 = p.bbox
            cx = (x1 + x2) // 2
            rx = max((x2 - x1) // 2, 10)
            cv2.ellipse(out, (cx, y2), (rx, 6), 0, -45, 225, GK_BGR, 2)

    # Goal posts — black rectangle outline
    for gp in player_dets.goal_posts:
        x1, y1, x2, y2 = gp.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), 2)

    # --- Ball — supervision triangle marker ---
    if ball_det is not None:
        ball_sv = sv.Detections(xyxy=np.array([[*ball_det.bbox]], dtype=float))
        out = _triangle_ann.annotate(scene=out, detections=ball_sv)

    # --- Possession bar (top strip) ---
    t0_pct, t1_pct = stats.percentages()
    bx1, bx2 = 10, W_vid - 10
    by1, by2 = 8, 8 + POSSESS_BAR_H
    bar_w    = bx2 - bx1

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

    ball_sm    = BallStateMachine()
    poss_eng   = PossessionEngine()
    pass_track = PassTracker()
    stats      = PossessionStats()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = PlayerTracker(fps=fps)
    W_vid   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_vid   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer  = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W_vid, H_vid),
    )

    BALL_MOVEMENT_PX = 15   # pixels ball must move between frames to count as "in play"

    print(f"[Possession] {total_f} frames  {W_vid}x{H_vid}  {fps:.1f} fps → {out_path}")

    fidx           = 0
    last_ball_pos  = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        player_dets = player_det.detect(frame, fidx, fps)
        team_clf.classify(frame, player_dets)
        gk_det.classify(player_dets)
        tracker.update(frame, player_dets.players)

        raw_ball       = ball_det_model.detect(frame)
        b_state, b_det = ball_sm.update(raw_ball)

        # Only update possession stats when ball is moving (not dead ball)
        ball_moving = (
            b_det is not None
            and (last_ball_pos is None
                 or math.dist(b_det.centre, last_ball_pos) > BALL_MOVEMENT_PX)
        )
        if b_det is not None:
            last_ball_pos = b_det.centre

        label, _method = poss_eng.update(player_dets.players, b_state, b_det)
        pass_track.update(label, poss_eng.get_possessing_player())
        if ball_moving:
            stats.update(label)

        out_frame = draw_frame(frame, player_dets, b_det, label, stats)
        writer.write(out_frame)

        if fidx % 300 == 0:
            t0, t1 = stats.percentages()
            print(f"  frame {fidx:>5}/{total_f}  T0={t0:.1f}%  T1={t1:.1f}%  state={b_state}  label={label}")

        fidx += 1

    cap.release()
    writer.release()
    print(f"\n[Possession] Saved → {out_path}")
    print(stats.summary())
    print(pass_track.summary())


if __name__ == "__main__":
    run()
