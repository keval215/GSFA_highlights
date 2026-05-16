"""
ocr_reader.py — Stage 1 + 2: scoreboard ROI cropping and OCR digit reading.

Uses EasyOCR (GPU-optional, works on Windows without PaddlePaddle runtime issues).
PaddlePaddle 3.x has a known Windows PIR runtime bug; EasyOCR is the reliable
fallback listed in approach.md.

Scoreboard overlay is a fixed broadcast graphic burned into the top-center.
Normalized coordinates (0-1000 scale). Default ROIs are fallback values; the
pipeline calibrates them per-video via scoreboard_detector.calibrate_rois().

Pixel conversion: pixel = (norm / 1000) * dimension
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Default scoreboard ROI definitions — normalized 0-1000 coords (ymin, xmin, ymax, xmax).
# Used as fallback when auto-calibration fails (e.g. no scoreboard in first 30s).
# ---------------------------------------------------------------------------
_DEFAULT_ROIS = {
    "timer":       (33, 436, 124, 497),
    "team1_score": (47, 542,  77, 558),
    "team2_score": (87, 542, 118, 558),
}

# Upscale before OCR — small crops (score digits can be ~17x21 px) need enlarging
OCR_UPSCALE = 4


def _norm_to_px(ymin_n, xmin_n, ymax_n, xmax_n, h, w):
    x1 = int(xmin_n / 1000 * w)
    y1 = int(ymin_n / 1000 * h)
    x2 = int(xmax_n / 1000 * w)
    y2 = int(ymax_n / 1000 * h)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def crop_roi(frame: np.ndarray, roi_name: str, rois: dict) -> np.ndarray:
    """Return the upscaled crop for a named ROI from a full video frame."""
    h, w = frame.shape[:2]
    coords = rois[roi_name]
    x1, y1, x2, y2 = _norm_to_px(*coords, h, w)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return crop
    return cv2.resize(crop, None, fx=OCR_UPSCALE, fy=OCR_UPSCALE,
                      interpolation=cv2.INTER_CUBIC)


def _parse_digits_easyocr(results: list, max_value: int = 99) -> int | None:
    """
    Pick the highest-confidence valid integer from an EasyOCR result list,
    rejecting values above `max_value`. Each result item is (bbox, text, conf).

    Best-confidence beats first-found: when EasyOCR returns both a clean `1`
    detection and a noisy `10` hallucination on the same crop, the confident
    `1` wins.
    """
    best_conf, best_val = -1.0, None
    for (_bbox, text, conf) in results:
        digits = "".join(c for c in text if c.isdigit())
        if not digits:
            continue
        val = int(digits)
        if val > max_value:
            continue
        if conf > best_conf:
            best_conf, best_val = conf, val
    return best_val


def _parse_timer_easyocr(results: list) -> str | None:
    """Extract MM:SS string from EasyOCR results."""
    for (_bbox, text, conf) in results:
        cleaned = text.strip()
        if cleaned:
            return cleaned
    return None


class ScoreReader:
    """
    Reads home and away scores from a video frame using EasyOCR.

    EasyOCR is used instead of PaddleOCR because PaddlePaddle 3.x has a
    known Windows runtime bug (ConvertPirAttribute2RuntimeAttribute NotImplemented).

    Usage:
        reader = ScoreReader()
        home, away = reader.read(frame)   # returns (int|None, int|None)
        timer = reader.read_timer(frame)  # returns "MM:SS" string or None
    """

    def __init__(self, gpu: bool = False):
        import easyocr
        # allowlist restricts recognition to digits only for score crops
        # (reduces false positives from font artefacts)
        self._ocr = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        self._digit_allowlist = "0123456789"
        self._rois = dict(_DEFAULT_ROIS)

    def update_rois(self, rois: dict) -> None:
        """Replace the active ROI set with calibrated values from scoreboard_detector."""
        self._rois = rois

    def read(self, frame: np.ndarray) -> tuple[int | None, int | None]:
        """Return (team1_score, team2_score) from a BGR frame, or None if unreadable."""
        home_crop = crop_roi(frame, "team1_score", self._rois)
        away_crop = crop_roi(frame, "team2_score", self._rois)

        home, away = None, None
        if home_crop.size:
            res = self._ocr.readtext(home_crop, allowlist=self._digit_allowlist,
                                     detail=1, paragraph=False)
            home = _parse_digits_easyocr(res, max_value=50)

        if away_crop.size:
            res = self._ocr.readtext(away_crop, allowlist=self._digit_allowlist,
                                     detail=1, paragraph=False)
            away = _parse_digits_easyocr(res, max_value=50)

        return home, away

    def read_timer(self, frame: np.ndarray) -> str | None:
        """Return MM:SS timer string or None."""
        crop = crop_roi(frame, "timer", self._rois)
        if not crop.size:
            return None
        # Timer allows digits and colon
        res = self._ocr.readtext(crop, allowlist="0123456789:",
                                 detail=1, paragraph=False)
        return _parse_timer_easyocr(res)
