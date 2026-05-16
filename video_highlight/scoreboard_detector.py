"""
scoreboard_detector.py — Dynamic scoreboard ROI calibration.

Detects the scoreboard bar's right edge within a user-defined universal search
box, then derives score-digit ROIs relative to that edge. This handles overlays
that shift horizontally when team acronyms have different lengths (e.g. 3-char
"AFC" vs 5-char "AKEFA" widens the bar by ~13 normalized units).

Usage:
    from scoreboard_detector import calibrate_rois, UNIVERSAL_BOX
    rois = calibrate_rois(cap, (frame_h, frame_w), UNIVERSAL_BOX)
    if rois:
        reader.update_rois(rois)
"""

import numpy as np
import cv2

# Universal search box — normalized 0-1000 coords (ymin, xmin, ymax, xmax).
# Derived from two sample frames: one with a 3-char acronym (small bar, xmax~569)
# and one with a 5-char acronym (large bar, xmax~582). The box covers both.
UNIVERSAL_BOX = (10, 410, 155, 590)

# Score digit column width in normalized units — always the rightmost slice of the bar.
_DIGIT_WIDTH_NORM = 28

# Minimum fraction of pixels in a vertical column that must be "dark" (bar background)
# to count that column as still inside the bar.
_DARK_THRESHOLD_V = 80      # HSV V-channel max for "dark" pixel
_DARK_FRAC_MIN   = 0.35     # at least 35% of column pixels must be dark


def _norm_to_px(ymin_n, xmin_n, ymax_n, xmax_n, h, w):
    x1 = int(xmin_n / 1000 * w)
    y1 = int(ymin_n / 1000 * h)
    x2 = int(xmax_n / 1000 * w)
    y2 = int(ymax_n / 1000 * h)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def _sample_frames(cap, duration_sec: float = 30.0, n: int = 10) -> list:
    """Return n evenly-spaced frames from [0, duration_sec]. Resets cap to frame 0."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frame = min(total - 1, int(duration_sec * fps))
    if max_frame <= 0:
        return []

    frames = []
    positions = [int(i * max_frame / max(1, n - 1)) for i in range(n)]
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return frames


def detect_bar_right_edge(frame: np.ndarray, universal_box: tuple) -> int | None:
    """
    Find the rightmost pixel column (in frame coords) that still belongs to the
    scoreboard bar within `universal_box`.

    Returns the right-edge x-coordinate in frame pixels, or None if not found.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _norm_to_px(*universal_box, h, w)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    crop_h, crop_w = v_channel.shape
    # Scan columns right-to-left; find rightmost column with enough dark pixels.
    for col in range(crop_w - 1, -1, -1):
        col_pixels = v_channel[:, col]
        dark_frac = np.sum(col_pixels < _DARK_THRESHOLD_V) / crop_h
        if dark_frac >= _DARK_FRAC_MIN:
            # Convert back to frame x-coordinate
            return x1 + col

    return None


def _px_to_norm(px_val: int, dimension: int) -> int:
    """Convert a pixel coordinate to normalized 0-1000 scale."""
    return int(px_val / dimension * 1000)


def calibrate_rois(
    cap,
    frame_shape: tuple,
    universal_box: tuple = UNIVERSAL_BOX,
    duration_sec: float = 30.0,
    n_samples: int = 10,
    min_detections: int = 4,
) -> dict | None:
    """
    Sample frames from the first `duration_sec` seconds, detect the scoreboard
    bar's right edge in each, and return a calibrated ROI dict (same format as
    ocr_reader._ROIS) or None if calibration fails.

    The returned dict can be passed directly to ScoreReader.update_rois().
    """
    frame_h, frame_w = frame_shape[:2]

    frames = _sample_frames(cap, duration_sec=duration_sec, n=n_samples)
    if not frames:
        return None

    right_edges = []
    for frame in frames:
        edge = detect_bar_right_edge(frame, universal_box)
        if edge is not None:
            right_edges.append(edge)

    if len(right_edges) < min_detections:
        return None

    # Median is robust to the odd frame where the bar is obscured by a replay graphic.
    median_edge_px = int(np.median(right_edges))
    R = _px_to_norm(median_edge_px, frame_w)   # normalized right edge

    digit_x1 = max(0, R - _DIGIT_WIDTH_NORM)
    digit_x2 = min(1000, R)

    return {
        # Score digit columns — right edge of bar, split top/bottom for each team.
        # Y ranges match the original tight hardcoded values (derived from test image analysis).
        # Only X is dynamic — Y position of digits is consistent across all videos.
        "team1_score": (47, digit_x1,  77, digit_x2),
        "team2_score": (87, digit_x1, 118, digit_x2),
        # Timer stays on the left side of the bar — doesn't shift with name length.
        "timer":       (33, 412, 124, 480),
    }
