"""
ocr_test.py — Validate PaddleOCR accuracy on a single frame or image before
running the full pipeline. Saves the upscaled crops it feeds to OCR so you
can visually inspect them.

Usage:
  python video_highlight/ocr_test.py --input data/images/image.png
  python video_highlight/ocr_test.py --input data/videos/test.mp4 --frame 300
"""

import argparse
import sys
from pathlib import Path

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--input",  default="data/images/image.png")
parser.add_argument("--frame",  type=int, default=0,
                    help="Frame index to sample (for video input, default 0)")
parser.add_argument("--out",    default="data/diagnostics/scoreboard",
                    help="Where to save OCR crop images")
args = parser.parse_args()

sys.path.insert(0, str(Path(__file__).parent))
from ocr_reader import ScoreReader, crop_roi

inp = Path(args.input)
if not inp.exists():
    print(f"[ERROR] Not found: {inp}"); sys.exit(1)

# Load frame
if inp.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
    cap = cv2.VideoCapture(str(inp))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"[ERROR] Could not read frame {args.frame}"); sys.exit(1)
    print(f"[INFO] Loaded frame {args.frame} from {inp.name}")
else:
    frame = cv2.imread(str(inp))
    if frame is None:
        print(f"[ERROR] Could not read image: {inp}"); sys.exit(1)
    print(f"[INFO] Loaded image {inp.name}  ({frame.shape[1]}x{frame.shape[0]})")

out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

# Save crops that will be fed to OCR
for roi_name in ("timer", "team1_score", "team2_score"):
    crop = crop_roi(frame, roi_name)
    if crop.size:
        p = out_dir / f"ocr_input_{roi_name}.png"
        cv2.imwrite(str(p), crop)
        print(f"  Crop saved: {p}  ({crop.shape[1]}x{crop.shape[0]})")

print("\n[INFO] Initialising PaddleOCR...")
reader = ScoreReader()

home, away = reader.read(frame)
timer = reader.read_timer(frame)

print(f"\n  Timer        : {timer}")
print(f"  Team 1 score : {home}")
print(f"  Team 2 score : {away}")

if home is None and away is None:
    print("\n[WARNING] Both scores unreadable.")
    print("  Suggestions:")
    print("  1. Check crops in data/diagnostics/scoreboard/ — are the digits visible?")
    print("  2. Try --frame N for a frame where the scoreboard is clearly lit.")
    print("  3. Increase OCR_UPSCALE in ocr_reader.py (currently 4x).")
else:
    print(f"\n[OK] OCR working. Score: {home}-{away}")
