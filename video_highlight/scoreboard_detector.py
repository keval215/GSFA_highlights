"""
scoreboard_detector.py — Returns fixed wide ROIs covering the universal scoreboard box.

Approach: instead of trying to detect the bar's right edge dynamically (which is
brittle on intro/transition frames), we OCR the entire universal box directly.
EasyOCR with a digit-only allowlist filters out team names automatically, and
the rightmost-digit selection in _parse_digits_easyocr picks the actual score.

Usage:
    from scoreboard_detector import calibrate_rois, UNIVERSAL_BOX
    rois = calibrate_rois(cap, (frame_h, frame_w))
    reader.update_rois(rois)
"""

# Universal search box — normalized 0-1000 coords (ymin, xmin, ymax, xmax).
# Always contains the full scoreboard regardless of team name length / camera angle.
UNIVERSAL_BOX = (10, 410, 155, 590)

# Fixed ROIs — tight score columns covering both short and long team name variants.
_FIXED_ROIS = {
    "team1_score": (25, 540,  85, 585),   # Team A (top row) — score digit only
    "team2_score": (90, 540, 150, 585),   # Team B (bottom row) — score digit only
    "timer":       (10, 412, 155, 480),   # left side: timer column
}


def calibrate_rois(cap=None, frame_shape=None, universal_box=UNIVERSAL_BOX, **_) -> dict:
    """
    Return the fixed wide ROIs covering the universal scoreboard box.

    Signature kept compatible with the previous calibration API so pipeline.py
    doesn't need to change. The cap / frame_shape args are unused.
    """
    return dict(_FIXED_ROIS)
