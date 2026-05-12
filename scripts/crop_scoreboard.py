"""
crop_scoreboard.py — crop and inspect scoreboard overlay regions from a match image.

Normalised coordinate system: 0-1000 on both axes (independent of image resolution).
  pixel_x = (norm_x / 1000) * image_width
  pixel_y = (norm_y / 1000) * image_height

Regions defined from the test image analysis (top-center scoreboard layout):
  Entire scoreboard bar  : [ymin=24, xmin=434, ymax=127, xmax=563]
  Timer  (MM:SS)         : [ymin=33, xmin=436, ymax=124, xmax=497]
  Team 1 row   (AFC)     : [ymin=44, xmin=501, ymax=78,  xmax=558]
  Team 1 name  (AFC)     : [ymin=46, xmin=518, ymax=77,  xmax=545]
  Team 1 score (0)       : [ymin=47, xmin=545, ymax=77,  xmax=561]
  Team 2 row   (VFC)     : [ymin=85, xmin=501, ymax=122, xmax=558]
  Team 2 name  (VFC)     : [ymin=87, xmin=518, ymax=118, xmax=545]
  Team 2 score (0)       : [ymin=87, xmin=545, ymax=118, xmax=561]

Usage:
  python scripts/crop_scoreboard.py
  python scripts/crop_scoreboard.py --image data/images/image.png --out data/diagnostics/scoreboard
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Scoreboard ROI definitions  (normalised 0-1000 coords: ymin, xmin, ymax, xmax)
# ---------------------------------------------------------------------------
REGIONS = {
    "scoreboard_bar": (24, 434, 127, 563),
    "timer":          (33, 436, 124, 497),
    "team1_row":      (44, 501,  78, 558),
    "team1_name":     (46, 518,  77, 545),
    "team1_score":    (47, 542,  77, 558),
    "team2_row":      (85, 501, 122, 558),
    "team2_name":     (87, 518, 118, 545),
    "team2_score":    (87, 542, 118, 558),
}

# Colour per region for the annotated overview (BGR)
REGION_COLORS = {
    "scoreboard_bar": (0,   255, 255),   # yellow
    "timer":          (255, 128,   0),   # blue
    "team1_row":      (200, 200, 200),   # light grey
    "team1_name":     (0,   200,   0),   # green
    "team1_score":    (0,   255,   0),   # bright green
    "team2_row":      (180, 180, 180),   # grey
    "team2_name":     (0,    80, 200),   # orange-red
    "team2_score":    (0,    0,  255),   # red
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm_to_px(ymin_n, xmin_n, ymax_n, xmax_n, h, w):
    """Convert normalised (0-1000) coords to pixel (x1,y1,x2,y2)."""
    x1 = int(xmin_n / 1000 * w)
    y1 = int(ymin_n / 1000 * h)
    x2 = int(xmax_n / 1000 * w)
    y2 = int(ymax_n / 1000 * h)
    return x1, y1, x2, y2


def crop_region(img, ymin_n, xmin_n, ymax_n, xmax_n):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = norm_to_px(ymin_n, xmin_n, ymax_n, xmax_n, h, w)
    # clamp to image bounds
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Crop scoreboard regions from a match image")
parser.add_argument("--image", default="data/images/image.png",
                    help="Path to input image")
parser.add_argument("--out",   default="data/diagnostics/scoreboard",
                    help="Output directory for crop PNGs")
parser.add_argument("--scale", type=float, default=3.0,
                    help="Upscale factor when saving small crops (default 3x)")
args = parser.parse_args()

img_path = Path(args.image)
if not img_path.exists():
    print(f"[ERROR] Image not found: {img_path}")
    print(f"        Put your test image there and re-run.")
    sys.exit(1)

img = cv2.imread(str(img_path))
if img is None:
    print(f"[ERROR] cv2 could not read: {img_path}")
    sys.exit(1)

h, w = img.shape[:2]
print(f"[INFO] Image: {img_path.name}  ({w}×{h} px)")

out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

# --- Draw annotated overview ---
annotated = img.copy()

print(f"\n{'Region':<20}  {'Pixels (x1,y1,x2,y2)':<26}  Saved")
print("-" * 65)

for name, (ymin_n, xmin_n, ymax_n, xmax_n) in REGIONS.items():
    crop, (x1, y1, x2, y2) = crop_region(img, ymin_n, xmin_n, ymax_n, xmax_n)

    if crop.size == 0:
        print(f"  {name:<18}  EMPTY (coords out of bounds) — skip")
        continue

    # Upscale small crops so they're readable when opened
    if args.scale > 1.0:
        crop_big = cv2.resize(crop, None, fx=args.scale, fy=args.scale,
                              interpolation=cv2.INTER_CUBIC)
    else:
        crop_big = crop

    out_path = out_dir / f"{name}.png"
    cv2.imwrite(str(out_path), crop_big)

    # Draw on annotated overview
    color = REGION_COLORS.get(name, (255, 255, 255))
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    label_y = y1 - 6 if y1 > 20 else y2 + 14
    cv2.putText(annotated, name, (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    print(f"  {name:<18}  ({x1:4d},{y1:4d}) -> ({x2:4d},{y2:4d})  {out_path.name}")

# Save annotated overview
overview_path = out_dir / "overview_annotated.png"
cv2.imwrite(str(overview_path), annotated)
print(f"\n[INFO] Annotated overview saved: {overview_path}")
print(f"[INFO] All crops saved to:       {out_dir}/")
print(f"\n[NEXT] Run PaddleOCR on these crops:")
print(f"       python scripts/ocr_scoreboard.py --crops {out_dir}")
