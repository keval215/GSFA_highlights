"""
test_scoreboard.py — Test the current scoreboard detection logic on a single image.
Run with: python test_scoreboard.py

Reads:  data/images/image.png
Writes: data/images/result.png

Uses the production approach: fixed wide ROIs covering the universal box, with
rightmost-digit selection in OCR parsing.
"""

import cv2
import numpy as np

# ── Production ROIs (matches scoreboard_detector.py / ocr_reader.py _DEFAULT_ROIS) ──
UNIVERSAL_BOX = (10, 410, 155, 590)
ROIS = {
    "team1_score": (25, 540,  85, 585),   # Team A (top row) — score digit
    "team2_score": (90, 540, 150, 585),   # Team B (bottom row) — score digit
    "timer":       (10, 412, 155, 480),   # left side
}


def _norm_to_px(ymin_n, xmin_n, ymax_n, xmax_n, h, w):
    x1 = int(xmin_n / 1000 * w)
    y1 = int(ymin_n / 1000 * h)
    x2 = int(xmax_n / 1000 * w)
    y2 = int(ymax_n / 1000 * h)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


# ── Load image ──
frame = cv2.imread("data/images/image.png")
assert frame is not None, "Could not load data/images/image.png"
h, w = frame.shape[:2]
print(f"Image loaded: {w}x{h}")

# ── Draw ROIs ──
result = frame.copy()

# Draw universal box (blue)
ux1, uy1, ux2, uy2 = _norm_to_px(*UNIVERSAL_BOX, h, w)
cv2.rectangle(result, (ux1, uy1), (ux2, uy2), (255, 100, 0), 2)
cv2.putText(result, "Universal Box", (ux1, uy1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1)

colors = {
    "team1_score": (0, 255, 255),   # yellow
    "team2_score": (0, 165, 255),   # orange
    "timer":       (255, 0, 255),   # magenta
}
for name, coords in ROIS.items():
    rx1, ry1, rx2, ry2 = _norm_to_px(*coords, h, w)
    color = colors[name]
    cv2.rectangle(result, (rx1, ry1), (rx2, ry2), color, 2)
    cv2.putText(result, name, (rx1, ry1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

# ── Try EasyOCR ──
try:
    import easyocr
    OCR_UPSCALE = 4
    print("\nInitialising EasyOCR...")
    ocr = easyocr.Reader(["en"], gpu=False, verbose=False)

    def read_score(roi_name):
        ymin_n, xmin_n, ymax_n, xmax_n = ROIS[roi_name]
        rx1, ry1, rx2, ry2 = _norm_to_px(ymin_n, xmin_n, ymax_n, xmax_n, h, w)
        crop = frame[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            return None
        crop_up = cv2.resize(crop, None, fx=OCR_UPSCALE, fy=OCR_UPSCALE,
                             interpolation=cv2.INTER_CUBIC)
        results = ocr.readtext(crop_up, allowlist="0123456789", detail=1, paragraph=False)

        # Rightmost-digit selection (matches production _parse_digits_easyocr)
        candidates = []
        for (bbox, text, conf) in results:
            digits = "".join(c for c in text if c.isdigit())
            if not digits:
                continue
            val = int(digits)
            if val > 50:
                continue
            x_center = (bbox[0][0] + bbox[2][0]) / 2
            candidates.append((x_center, conf, val))
        if not candidates:
            return None
        candidates.sort(key=lambda c: (-c[0], -c[1]))
        return candidates[0][2]

    team1 = read_score("team1_score")
    team2 = read_score("team2_score")

    print(f"\n── Score Results ──")
    print(f"  Team 1 (PSA)   goals: {team1}")
    print(f"  Team 2 (AKEFA) goals: {team2}")

    label = f"PSA {team1}  -  {team2} AKEFA"
    cv2.putText(result, label, (ux1, uy2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

except ImportError:
    print("\n[INFO] easyocr not installed — skipping OCR, showing ROI boxes only.")

# ── Save ──
cv2.imwrite("data/images/result.png", result)
print("\nSaved: data/images/result.png")
