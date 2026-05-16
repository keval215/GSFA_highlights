"""
test_scoreboard.py — Test the current scoreboard detection logic on a single image.
Run with: python test_scoreboard.py

Reads:  data/images/image.png
Writes: data/images/result.png
"""

import cv2
import numpy as np

# ── Current constants from scoreboard_detector.py ──
UNIVERSAL_BOX    = (10, 410, 155, 590)   # (ymin, xmin, ymax, xmax) normalized 0-1000
_DIGIT_WIDTH_NORM = 28
_DARK_THRESHOLD_V = 80
_DARK_FRAC_MIN    = 0.35


def _norm_to_px(ymin_n, xmin_n, ymax_n, xmax_n, h, w):
    x1 = int(xmin_n / 1000 * w)
    y1 = int(ymin_n / 1000 * h)
    x2 = int(xmax_n / 1000 * w)
    y2 = int(ymax_n / 1000 * h)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def _px_to_norm(px_val, dimension):
    return int(px_val / dimension * 1000)


def detect_bar_right_edge(frame):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _norm_to_px(*UNIVERSAL_BOX, h, w)
    crop = frame[y1:y2, x1:x2]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    crop_h, crop_w = v_channel.shape

    for col in range(crop_w - 1, -1, -1):
        dark_frac = np.sum(v_channel[:, col] < _DARK_THRESHOLD_V) / crop_h
        if dark_frac >= _DARK_FRAC_MIN:
            return x1 + col   # frame pixel x-coordinate

    return None


def derive_rois(edge_px, frame_w):
    R = _px_to_norm(edge_px, frame_w)
    digit_x1 = max(0, R - _DIGIT_WIDTH_NORM)
    digit_x2 = min(1000, R)
    return {
        "team1_score": (25, digit_x1,  84, digit_x2),
        "team2_score": (85, digit_x1, 150, digit_x2),
        "timer":       (10, 412,      155, 480),
    }


# ── Load image ──
frame = cv2.imread("data/images/image.png")
assert frame is not None, "Could not load data/images/image.png"
h, w = frame.shape[:2]
print(f"Image loaded: {w}x{h}")

# ── Run detection ──
result = frame.copy()

# Draw universal search box (blue)
ux1, uy1, ux2, uy2 = _norm_to_px(*UNIVERSAL_BOX, h, w)
cv2.rectangle(result, (ux1, uy1), (ux2, uy2), (255, 100, 0), 2)
cv2.putText(result, "Universal Box", (ux1, uy1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1)

edge_px = detect_bar_right_edge(frame)

if edge_px is None:
    print("[FAIL] Bar right edge NOT detected — dark threshold too strict for this image.")
    print("       Check the universal box region and the scoreboard background contrast.")
    cv2.putText(result, "BAR NOT DETECTED", (ux1, uy1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
else:
    edge_norm = _px_to_norm(edge_px, w)
    print(f"[OK]  Bar right edge detected at x={edge_px}px  (normalized={edge_norm})")

    rois = derive_rois(edge_px, w)
    print(f"\nCalibrated ROIs:")
    for name, coords in rois.items():
        print(f"  {name}: {coords}")

    # Draw detected right edge (green vertical line)
    cv2.line(result, (edge_px, uy1), (edge_px, uy2), (0, 255, 0), 2)
    cv2.putText(result, f"bar right edge x={edge_px}", (edge_px - 110, uy2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Draw each calibrated ROI
    colors = {
        "team1_score": (0, 255, 255),   # yellow
        "team2_score": (0, 165, 255),   # orange
        "timer":       (255, 0, 255),   # magenta
    }
    for name, coords in rois.items():
        rx1, ry1, rx2, ry2 = _norm_to_px(*coords, h, w)
        color = colors[name]
        cv2.rectangle(result, (rx1, ry1), (rx2, ry2), color, 2)
        cv2.putText(result, name, (rx1, ry1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    # ── Try EasyOCR if available ──
    try:
        import easyocr
        OCR_UPSCALE = 4
        ocr = easyocr.Reader(["en"], gpu=False, verbose=False)

        def read_score(roi_name):
            ymin_n, xmin_n, ymax_n, xmax_n = rois[roi_name]
            rx1, ry1, rx2, ry2 = _norm_to_px(ymin_n, xmin_n, ymax_n, xmax_n, h, w)
            crop = frame[ry1:ry2, rx1:rx2]
            if crop.size == 0:
                return None
            crop_up = cv2.resize(crop, None, fx=OCR_UPSCALE, fy=OCR_UPSCALE,
                                 interpolation=cv2.INTER_CUBIC)
            results = ocr.readtext(crop_up, allowlist="0123456789", detail=1, paragraph=False)
            best_conf, best_val = -1.0, None
            for (_, text, conf) in results:
                digits = "".join(c for c in text if c.isdigit())
                if not digits:
                    continue
                val = int(digits)
                if val > 50:
                    continue
                if conf > best_conf:
                    best_conf, best_val = conf, val
            return best_val

        team1 = read_score("team1_score")
        team2 = read_score("team2_score")

        print(f"\n── Score Results ──")
        print(f"  Team 1 (PSA)  goals: {team1}")
        print(f"  Team 2 (AKEFA) goals: {team2}")

        label = f"PSA {team1}  -  {team2} AKEFA"
        cv2.putText(result, label, (ux1, uy2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    except ImportError:
        print("\n[INFO] easyocr not installed — skipping OCR, showing ROI boxes only.")

# ── Save result ──
cv2.imwrite("data/images/result.png", result)
print("\nSaved: data/images/result.png")
