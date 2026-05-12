"""
endgame_detector.py — 1-fps state-machine OCR for HALF TIME / FULL TIME / PENALTIES.

State sequence (chronological, one-way):
    PRE_HALFTIME -> POST_HALFTIME -> POST_FULLTIME -> PENALTIES_ACTIVE -> DONE

Per-state ROIs (normalized 0-1000, [ymin, xmin, ymax, xmax]) on full frame:
    PRE_HALFTIME   : halftime banner          (775, 197, 834, 803)
    POST_HALFTIME  : fulltime text            (789, 220, 808, 268)
    POST_FULLTIME  : central penalty box      (24, 430, 122, 497)
    PENALTIES_ACTIVE: central penalty box     (24, 430, 122, 497) — read score

Fuzzy match threshold: 0.65 against {"HALF TIME", "FULL TIME", "PENALTIES"}.

The detector returns events for the pipeline to act on:
    {"type": "halftime",  "t": <sec>}  -> save 30s clip starting at t
    {"type": "fulltime",  "t": <sec>}  -> no clip, just state transition
    {"type": "penalties_start", "t": <sec>, "score": "0-1"}
    {"type": "penalties_score",  "t": <sec>, "score": "0-2"}  (each change)
    {"type": "penalties_end",    "t_start": <sec>, "t_end": <sec>, "final_score": "0-3"}
"""

import re
from difflib import SequenceMatcher

import cv2
import numpy as np

# --- ROIs (normalized 0-1000, ymin/xmin/ymax/xmax) ---
ROI_HALFTIME_BANNER = (775, 197, 834, 803)
ROI_FULLTIME_TEXT   = (789, 220, 808, 268)
ROI_PENALTY_BOX     = (24,  430, 122, 497)

# OCR config
UPSCALE         = 8
FUZZY_THRESHOLD = 0.65
PENALTY_STABLE_SEC  = 60.0   # score must remain unchanged this long to declare end
FULLTIME_COOLDOWN_SEC = 15.0  # safety net after halftime; primary defense is the
                              # hue gate in _check_fulltime
# Hue gate (OpenCV 0-179 scale, user-calibrated): halftime banner is orange,
# fulltime banner is dark navy. Require navy + reject orange before OCR.
HALFTIME_HUE_CENTER = 15
HALFTIME_HUE_TOL    = 12
FULLTIME_HUE_CENTER = 117
FULLTIME_HUE_TOL    = 15
FULLTIME_SAT_MIN    = 60

# State enum (string for readability in logs)
S_PRE_HALFTIME    = "PRE_HALFTIME"
S_POST_HALFTIME   = "POST_HALFTIME"
S_POST_FULLTIME   = "POST_FULLTIME"
S_PENALTIES_ACTIVE = "PENALTIES_ACTIVE"
S_DONE            = "DONE"


def _norm(s: str) -> str:
    return "".join(c for c in s.upper() if c.isalnum())


def _fuzzy(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _crop_roi(frame: np.ndarray, roi, scale: int = UPSCALE) -> np.ndarray:
    h, w = frame.shape[:2]
    ymin, xmin, ymax, xmax = roi
    x1 = max(0, int(xmin / 1000 * w))
    y1 = max(0, int(ymin / 1000 * h))
    x2 = min(w, int(xmax / 1000 * w))
    y2 = min(h, int(ymax / 1000 * h))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return crop
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _best_fuzzy(detections: list, target: str) -> float:
    """Return the best fuzzy score for `target` across detections (and concatenation)."""
    if not detections:
        return 0.0
    texts = [d[1] for d in detections]
    best = max(_fuzzy(t, target) for t in texts)
    concat = " ".join(texts)
    return max(best, _fuzzy(concat, target))


_PEN_SCORE_RE = re.compile(r"(\d+)\s*[-–—]\s*(\d+)")


def _parse_penalty_score(detections: list) -> str | None:
    """
    Extract 'N-N' from detections. Looks at each text and at the concatenation.
    """
    if not detections:
        return None
    candidates = [d[1] for d in detections]
    candidates.append(" ".join(candidates))
    for text in candidates:
        m = _PEN_SCORE_RE.search(text)
        if m:
            return f"{m.group(1)}-{m.group(2)}"
    return None


class EndgameDetector:
    """
    1-fps state-machine OCR for HALF TIME / FULL TIME / PENALTIES.

    Usage:
        det = EndgameDetector(ocr=easyocr_reader)
        # call once per second of video:
        events = det.update(frame, t_sec)
        for ev in events:
            ...
        # at end of video, flush any pending penalties session:
        events = det.finalize(t_end)
    """

    def __init__(self, ocr=None, gpu: bool = False):
        if ocr is None:
            import easyocr
            ocr = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        self._ocr = ocr

        self.state = S_PRE_HALFTIME
        # Halftime cooldown — block fulltime scans for FULLTIME_COOLDOWN_SEC
        # after halftime fires, otherwise the still-visible orange banner
        # fuzzy-matches "FULL TIME" on the very next 1-fps tick.
        self._halftime_t: float | None = None
        # Penalty session tracking
        self._pen_start_t: float | None = None
        self._pen_last_score: str | None = None
        self._pen_last_change_t: float | None = None

    # ---- OCR helpers ----
    def _read(self, frame: np.ndarray, roi, allowlist: str | None = None) -> list:
        crop = _crop_roi(frame, roi)
        if crop.size == 0:
            return []
        kwargs = dict(detail=1, paragraph=False)
        if allowlist:
            kwargs["allowlist"] = allowlist
        return self._ocr.readtext(crop, **kwargs)

    # ---- Per-state checks ----
    def _check_halftime(self, frame, t):
        res = self._read(frame, ROI_HALFTIME_BANNER)
        if _best_fuzzy(res, "HALF TIME") >= FUZZY_THRESHOLD:
            self._halftime_t = t
            self.state = S_POST_HALFTIME
            return [{"type": "halftime", "t": t}]
        return []

    def _check_fulltime(self, frame, t):
        if self._halftime_t is not None and (t - self._halftime_t) < FULLTIME_COOLDOWN_SEC:
            return []
        h, w = frame.shape[:2]
        ymin, xmin, ymax, xmax = ROI_FULLTIME_TEXT
        x1 = max(0, int(xmin / 1000 * w))
        y1 = max(0, int(ymin / 1000 * h))
        x2 = min(w, int(xmax / 1000 * w))
        y2 = min(h, int(ymax / 1000 * h))
        bg = frame[y1:y2, x1:x2]
        if bg.size == 0:
            return []
        hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
        mean_h = float(hsv[:, :, 0].mean())
        mean_s = float(hsv[:, :, 1].mean())
        # Reject if the ROI is still the orange halftime banner.
        if (abs(mean_h - HALFTIME_HUE_CENTER) <= HALFTIME_HUE_TOL
                and mean_s >= FULLTIME_SAT_MIN):
            return []
        # Require the navy fulltime banner before running OCR.
        if not (abs(mean_h - FULLTIME_HUE_CENTER) <= FULLTIME_HUE_TOL
                and mean_s >= FULLTIME_SAT_MIN):
            return []

        res = self._read(frame, ROI_FULLTIME_TEXT)
        if _best_fuzzy(res, "FULL TIME") >= FUZZY_THRESHOLD:
            self.state = S_POST_FULLTIME
            return [{"type": "fulltime", "t": t}]
        return []

    def _check_penalty_start(self, frame, t):
        res = self._read(frame, ROI_PENALTY_BOX, allowlist="0123456789 -PENALTIES")
        if _best_fuzzy(res, "PENALTIES") < FUZZY_THRESHOLD:
            return []
        score = _parse_penalty_score(res) or "0-0"
        self._pen_start_t = t
        self._pen_last_score = score
        self._pen_last_change_t = t
        self.state = S_PENALTIES_ACTIVE
        return [{"type": "penalties_start", "t": t, "score": score}]

    def _track_penalty(self, frame, t):
        res = self._read(frame, ROI_PENALTY_BOX, allowlist="0123456789 -PENALTIES")
        events = []
        score = _parse_penalty_score(res)
        if score and score != self._pen_last_score:
            self._pen_last_score = score
            self._pen_last_change_t = t
            events.append({"type": "penalties_score", "t": t, "score": score})

        # End-of-shootout: no score change for PENALTY_STABLE_SEC
        if (self._pen_last_change_t is not None
                and (t - self._pen_last_change_t) >= PENALTY_STABLE_SEC):
            events.append({
                "type": "penalties_end",
                "t_start": self._pen_start_t,
                "t_end":   t,
                "final_score": self._pen_last_score,
            })
            self.state = S_DONE
        return events

    # ---- Public API ----
    def update(self, frame: np.ndarray, t: float) -> list[dict]:
        """Run the active state's ROI check on `frame` at video time `t` (seconds)."""
        if self.state == S_PRE_HALFTIME:
            return self._check_halftime(frame, t)
        if self.state == S_POST_HALFTIME:
            return self._check_fulltime(frame, t)
        if self.state == S_POST_FULLTIME:
            return self._check_penalty_start(frame, t)
        if self.state == S_PENALTIES_ACTIVE:
            return self._track_penalty(frame, t)
        return []

    def finalize(self, t_end: float) -> list[dict]:
        """
        Close out any pending state at video end. If penalties were active and
        never reached the 60s-stable end condition, emit a penalties_end event
        anchored at t_end so the clip still gets saved.
        """
        if self.state == S_PENALTIES_ACTIVE and self._pen_start_t is not None:
            ev = {
                "type": "penalties_end",
                "t_start": self._pen_start_t,
                "t_end":   t_end,
                "final_score": self._pen_last_score,
                "reason": "video_ended",
            }
            self.state = S_DONE
            return [ev]
        return []
