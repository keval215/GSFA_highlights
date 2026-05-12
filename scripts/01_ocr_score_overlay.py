# Script 1: OCR Score Overlay Detection
# ----------------------------------------
# Runs EasyOCR on the scoreboard ROI (top-center bar) of a frame.
# Uses normalized 0-1000 [ymin, xmin, ymax, xmax] coords with a padding margin.
# EasyOCR is used instead of PaddleOCR because PaddlePaddle 3.x has a known
# Windows PIR runtime bug (ConvertPirAttribute2RuntimeAttribute NotImplemented).
#
# Required packages (install with pip if missing):
#   pip install easyocr
#   pip install opencv-python
#
# Usage:
#   python scripts/01_ocr_score_overlay.py
#   python scripts/01_ocr_score_overlay.py --image path/to/frame.jpg
#   python scripts/01_ocr_score_overlay.py --video path/to/video.mp4 --frame 0

import argparse
import sys
import os

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="OCR score overlay on football frames")
parser.add_argument(
    "--image",
    default="data/images/image.png",
    help="Path to a single image frame (default: data/images/image.png)",
)
parser.add_argument(
    "--video",
    default=None,
    help="Path to a video file. If provided, --image is ignored.",
)
parser.add_argument(
    "--frame",
    type=int,
    default=0,
    help="Which frame index to extract from the video (default: 0)",
)
parser.add_argument(
    "--pad",
    type=int,
    default=10,
    help="Pixel padding around the scoreboard ROI before OCR (default: 10)",
)
parser.add_argument(
    "--save-crop",
    action="store_true",
    help="Save the cropped ROI to data/images/ocr_crop_debug.jpg for inspection",
)
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Load frame
# ---------------------------------------------------------------------------
def load_frame_from_video(video_path: str, frame_index: int) -> np.ndarray:
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        print("        Place your video at that path and re-run the script.")
        sys.exit(1)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index >= total:
        print(f"[ERROR] frame index {frame_index} out of range (video has {total} frames)")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"[ERROR] Could not read frame {frame_index} from {video_path}")
        sys.exit(1)
    return frame


def load_frame(args) -> np.ndarray:
    if args.video is not None:
        print(f"[INFO] Loading frame {args.frame} from video: {args.video}")
        return load_frame_from_video(args.video, args.frame)

    image_path = args.image
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found: {image_path}")
        print("        Place a sample frame at that path (e.g. data/images/sample_frame.jpg) and re-run.")
        sys.exit(1)
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] cv2.imread failed for: {image_path}")
        sys.exit(1)
    print(f"[INFO] Loaded image: {image_path}  shape={frame.shape}")
    return frame


frame = load_frame(args)
h, w = frame.shape[:2]

# ---------------------------------------------------------------------------
# Crop scoreboard ROI (normalized 0-1000 coords, top-center bar)
# ---------------------------------------------------------------------------
SCOREBOARD_NORM = (24, 434, 127, 563)  # ymin, xmin, ymax, xmax
ymin_n, xmin_n, ymax_n, xmax_n = SCOREBOARD_NORM
x1 = max(0, int(xmin_n / 1000 * w) - args.pad)
y1 = max(0, int(ymin_n / 1000 * h) - args.pad)
x2 = min(w, int(xmax_n / 1000 * w) + args.pad)
y2 = min(h, int(ymax_n / 1000 * h) + args.pad)
roi = frame[y1:y2, x1:x2]
crop_h, crop_w = roi.shape[:2]

print(f"[INFO] Frame size: {w}x{h}  |  Scoreboard ROI: ({x1},{y1}) -> ({x2},{y2})  ({crop_w}x{crop_h})")

if args.save_crop:
    out_path = "data/images/ocr_crop_debug.jpg"
    cv2.imwrite(out_path, roi)
    print(f"[INFO] Saved crop to: {out_path}")

# ---------------------------------------------------------------------------
# EasyOCR
# ---------------------------------------------------------------------------
try:
    import easyocr
except ImportError:
    print("[ERROR] EasyOCR is not installed.")
    print("        Run:  pip install easyocr")
    sys.exit(1)

print("[INFO] Initialising EasyOCR (first run may download models)...")
reader = easyocr.Reader(["en"], gpu=False, verbose=False)

print("[INFO] Running OCR on scoreboard crop...")
results = reader.readtext(roi, detail=1, paragraph=False)

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("OCR RESULTS (scoreboard crop)")
print("=" * 60)

if not results:
    print("[INFO] No text detected in the crop.")
else:
    for (box, text, confidence) in results:
        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        print(f"  [{x_min:3d},{y_min:3d} -> {x_max:3d},{y_max:3d}]  conf={confidence:.2f}  text='{text}'")

print("=" * 60)
print("\n[NEXT STEP] Review the detected text above.")
print("  - If scores/team names appear, note the bounding box coordinates.")
print("  - Adjust --pad if the overlay is being clipped or too much background is included.")
print("  - Use --save-crop to visually inspect what is being sent to OCR.")
