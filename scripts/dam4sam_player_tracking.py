# =============================================================================
# GSFA — individual player tracking with DAM4SAM (Google Colab)
# https://github.com/jovanavidenovic/DAM4SAM  (CVPR 2025)
# =============================================================================
# WHAT THIS IS
#   DAM4SAM is a SINGLE-OBJECT tracker: its distractor-aware memory follows one
#   target at a time. Multi-object tracking is NOT supported upstream (repo
#   issue #14). So "individual player detection" here means: you pick the
#   player(s) you want, and we run one DAM4SAM tracker per player, compositing
#   every player's mask into one output video.
#
# HOW YOU SELECT THE PLAYERS (Colab-friendly, no draw-a-box GUI)
#   Colab can't pop up an interactive cv2 window, so instead:
#     1. YOLO runs on the FIRST frame and finds every player box.
#     2. We draw those boxes NUMBERED on frame 0 and show it inline.
#     3. You set KEEP_IDS to the numbers you want (or [] to track them all).
#   A manual PLAYER_BOXES list is also supported if you'd rather type boxes.
#
#   DAM4SAM is an *online* tracker (initialize() on frame 0, then track() each
#   frame), so all the per-player instances run together in one pass.
#
# HOW TO RUN
#   Runtime -> Change runtime type -> GPU, then run each CELL in order.
# =============================================================================


# =========================== CELL 1: install ================================
# Clone DAM4SAM, install it (its vendored sam2 comes with it), download the
# SAM2.1 checkpoints. Takes a few minutes.

!git clone https://github.com/jovanavidenovic/DAM4SAM.git /content/DAM4SAM
%cd /content/DAM4SAM
# DAM4SAM's own dependency list — includes the VOT toolkit that its
# dam4sam_tracker.py imports (`from vot.region...`), which `pip install -e .`
# alone does NOT pull in.
!pip install -q -r requirements.txt || echo "requirements.txt install had issues — continuing"
!pip install -q -e .
!pip install -q ultralytics
# Force the EXACT VOT versions DAM4SAM pins (requirements.txt: vot-toolkit==0.7.1,
# vot-trax==4.0.2). The current PyPI vot-toolkit moved RegionType out of
# vot.region, which breaks dam4sam_tracker.py's `from vot.region import
# RegionType`. Pinned + last so it wins (pip will downgrade a newer vot-toolkit
# back to 0.7.1). vot-trax is a C extension that may not build on Colab's
# Python 3.12, but the RegionType import doesn't need it — so don't block on it.
!pip install -q "vot-toolkit==0.7.1"
!pip install -q "vot-trax==4.0.2" || echo "vot-trax build skipped (RegionType import doesn't need it)"
# SAM2.1 checkpoints — the repo ships a downloader; fall back to a direct pull
# of the large checkpoint if the script location differs in your clone.
!cd checkpoints && bash download_ckpts.sh || ( \
    mkdir -p /content/DAM4SAM/checkpoints && \
    wget -q -O /content/DAM4SAM/checkpoints/sam2.1_hiera_large.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt )

from google.colab import drive
drive.mount('/content/drive')


# =========================== CELL 2: config =================================
import os

# Run from the repo root so `import dam4sam_tracker` and the vendored sam2/
# configs resolve.
os.chdir("/content/DAM4SAM")

CLIP_PATH    = "/content/2.mp4"                # video to track (upload it to /content/)
OUTPUT_PATH  = "/content/dam4sam_players_out.mp4"
TRACKER_NAME = "sam21pp-L"                     # DAM4SAM preset: sam21pp-{T,S,B,L}. L = most accurate.

# --- player selection ---
YOLO_MODEL_PATH = "/content/drive/MyDrive/aiff_v2.pt"  # detector for frame-0 boxes ("" to skip)
PLAYER_CONF     = 0.50
KEEP_IDS        = [0]                          # which numbered detections to track. [0] = just
                                               # the first detected player (single-object trackers
                                               # like DAM4SAM are happiest with one target). Put
                                               # more numbers to track more, or [] to track all.
PLAYER_BOXES    = []                           # manual override: list of (x, y, w, h) on frame 0

# DAM4SAM's run_bbox_example.py prompts with (x, y, w, h) from cv2.selectROI.
# If your clone expects xyxy instead, set this False.
BBOX_IS_XYWH = True

COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (0, 128, 255), (128, 0, 255),
]
MASK_ALPHA = 0.5

assert CLIP_PATH, "Upload your clip and set CLIP_PATH."
assert os.path.exists(CLIP_PATH), f"Clip not found: {CLIP_PATH}"


# ===================== CELL 3: imports + first frame ========================
import cv2
import numpy as np
from PIL import Image
from google.colab.patches import cv2_imshow

from dam4sam_tracker import DAM4SAMTracker

cap = cv2.VideoCapture(CLIP_PATH)
ok, first_frame = cap.read()
FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
H, W = first_frame.shape[:2]
cap.release()
assert ok, f"Could not read frames from {CLIP_PATH}"
print(f"Clip: {W}x{H} @ {FPS:.1f} fps")


# =================== CELL 4: detect + pick players ==========================
# Runs YOLO on frame 0, draws numbered boxes, and shows them. Set KEEP_IDS in
# CELL 2 to the numbers you want to track, then re-run this cell to confirm.

def detect_boxes_xywh(frame_bgr):
    if PLAYER_BOXES:
        return [tuple(int(v) for v in b) for b in PLAYER_BOXES]
    if not YOLO_MODEL_PATH:
        raise ValueError("Set YOLO_MODEL_PATH or fill PLAYER_BOXES to pick players.")
    from ultralytics import YOLO
    yolo = YOLO(YOLO_MODEL_PATH)
    player_ids = [i for i, n in yolo.names.items() if "player" in n.lower()] or [0]
    res = yolo.predict(frame_bgr, conf=PLAYER_CONF, verbose=False)[0]
    boxes = []
    if res.boxes is not None:
        cls = res.boxes.cls.cpu().numpy().astype(int)
        xyxy = res.boxes.xyxy.cpu().numpy()
        for (x1, y1, x2, y2) in xyxy[np.isin(cls, player_ids)]:
            boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
    return boxes

all_boxes = detect_boxes_xywh(first_frame)
sel = KEEP_IDS if KEEP_IDS else list(range(len(all_boxes)))
PICKED = [all_boxes[i] for i in sel]

preview = first_frame.copy()
for i, (x, y, w, h) in enumerate(all_boxes):
    on = i in sel
    col = (0, 255, 0) if on else (128, 128, 128)
    cv2.rectangle(preview, (x, y), (x + w, y + h), col, 2)
    cv2.putText(preview, str(i), (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
print(f"Found {len(all_boxes)} players; tracking {len(PICKED)} "
      f"(green = kept). Set KEEP_IDS in CELL 2 to change the selection.")
cv2_imshow(preview)


# ===================== CELL 5: track + render ===============================
def overlay_mask(frame_bgr, mask, color):
    m = np.asarray(mask)
    if m.ndim > 2:
        m = m.squeeze()
    m = m.astype(bool)
    if m.shape[:2] != frame_bgr.shape[:2] or not m.any():
        return m
    frame_bgr[m] = (MASK_ALPHA * np.array(color) + (1 - MASK_ALPHA) * frame_bgr[m]).astype(np.uint8)
    return m

def draw_label(frame_bgr, mask, text, color):
    ys, xs = np.where(mask)
    if len(xs):
        cv2.putText(frame_bgr, text, (int(xs.min()), max(0, int(ys.min()) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

def prompt_box(xywh):
    x, y, w, h = xywh
    return (x, y, w, h) if BBOX_IS_XYWH else (x, y, x + w, y + h)

assert PICKED, "No players selected — set KEEP_IDS or PLAYER_BOXES."
print(f"Tracking {len(PICKED)} player(s) with DAM4SAM ({TRACKER_NAME}).")

trackers = [DAM4SAMTracker(TRACKER_NAME) for _ in PICKED]
writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

cap = cv2.VideoCapture(CLIP_PATH)
frame_idx = 0
while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break
    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    overlay = frame_bgr.copy()
    for pid, (trk, box) in enumerate(zip(trackers, PICKED)):
        out = trk.initialize(img_pil, None, bbox=prompt_box(box)) if frame_idx == 0 else trk.track(img_pil)
        mask = out.get("pred_mask") if isinstance(out, dict) else out
        if mask is not None:
            color = COLORS[pid % len(COLORS)]
            m = overlay_mask(overlay, mask, color)
            draw_label(overlay, m, f"P{pid}", color)
    writer.write(overlay)
    frame_idx += 1

cap.release()
writer.release()
print(f"Done — {frame_idx} frames written to {OUTPUT_PATH}")


# ===================== CELL 6: re-encode + download =========================
!ffmpeg -y -loglevel error -i {OUTPUT_PATH} -vcodec libx264 -pix_fmt yuv420p /content/dam4sam_players_h264.mp4
from google.colab import files
files.download("/content/dam4sam_players_h264.mp4")
