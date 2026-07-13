# =============================================================================
# GSFA — individual player tracking with SAMURAI (Google Colab)
# https://github.com/yangchris11/samurai  (motion-aware SAM2 tracking)
# =============================================================================
# WHAT THIS IS
#   SAMURAI adds a motion-aware memory to SAM2 for robust SINGLE-OBJECT
#   tracking. Multi-object tracking is NOT supported upstream (repo issue #110);
#   its demo only ever tracks obj_id=0. So "individual player detection" here
#   means: you pick the player(s) you want, we run SAMURAI once PER player over
#   the clip, then composite every player's mask into one output video.
#
# HOW YOU SELECT THE PLAYERS (Colab-friendly, no draw-a-box GUI)
#   Colab can't pop up an interactive cv2 window, so instead:
#     1. YOLO runs on the FIRST frame and finds every player box.
#     2. We draw those boxes NUMBERED on frame 0 and show it inline.
#     3. You set KEEP_IDS to the numbers you want (or [] to track them all).
#   A manual PLAYER_BOXES list is also supported.
#
# WHY ONE PASS PER PLAYER (unlike the DAM4SAM script)
#   SAMURAI uses SAM2's *offline* video predictor: init_state() ingests the
#   whole clip, then propagate_in_video() sweeps it for ONE prompted object. To
#   follow N players we run N sweeps over the same extracted frames and merge
#   the masks at render time — linear in the number of players.
#
# HOW TO RUN
#   Runtime -> Change runtime type -> GPU, then run each CELL in order.
# =============================================================================


# =========================== CELL 1: install ================================
# Clone SAMURAI, install its vendored sam2, download the SAM2.1 checkpoints.

!git clone https://github.com/yangchris11/samurai.git /content/samurai
%cd /content/samurai/sam2
!pip install -q -e .
!pip install -q -e ".[notebooks]"
!pip install -q ultralytics
!cd /content/samurai/sam2/checkpoints && bash download_ckpts.sh

from google.colab import drive
drive.mount('/content/drive')


# =========================== CELL 2: config =================================
import os

# Run from the samurai repo root so the hydra configs under configs/samurai/
# and the vendored sam2 package resolve.
os.chdir("/content/samurai")

CLIP_PATH   = "/content/2.mp4"                              # video to track (upload it to /content/)
OUTPUT_PATH = "/content/samurai_players_out.mp4"
MODEL_PATH  = "/content/samurai/sam2/checkpoints/sam2.1_hiera_large.pt"
FRAMES_DIR  = "/content/samurai_frames"                     # scratch dir for extracted JPEG frames

# --- player selection ---
YOLO_MODEL_PATH = "/content/drive/MyDrive/aiff_v2.pt"       # detector for frame-0 boxes ("" to skip)
PLAYER_CONF     = 0.50
KEEP_IDS        = [0]                                       # which numbered detections to track. [0] = just
                                                            # the first detected player (single-object
                                                            # trackers like SAMURAI are happiest with one
                                                            # target). Add numbers to track more, [] = all.
PLAYER_BOXES    = []                                        # manual override: list of (x, y, w, h) on frame 0

COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (0, 128, 255), (128, 0, 255),
]
MASK_ALPHA = 0.5

assert CLIP_PATH, "Upload your clip and set CLIP_PATH."
assert os.path.exists(CLIP_PATH), f"Clip not found: {CLIP_PATH}"


# ================= CELL 3: imports + extract frames =========================
import gc
import shutil

import cv2
import numpy as np
import torch
from google.colab.patches import cv2_imshow

from sam2.build_sam import build_sam2_video_predictor


def determine_model_cfg(model_path):
    """Map a SAMURAI checkpoint filename to its hydra config (mirrors the
    repo's scripts/demo.py)."""
    name = os.path.basename(model_path).lower()
    if "large" in name:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    if "base_plus" in name:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    if "small" in name:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    if "tiny" in name:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    raise ValueError(f"Cannot infer SAMURAI config from checkpoint name: {model_path}")


# SAM2's init_state loads a directory of frames, so decode the clip once.
shutil.rmtree(FRAMES_DIR, ignore_errors=True)
os.makedirs(FRAMES_DIR)
cap = cv2.VideoCapture(CLIP_PATH)
FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
N_FRAMES = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imwrite(os.path.join(FRAMES_DIR, f"{N_FRAMES:05d}.jpg"), frame)
    N_FRAMES += 1
cap.release()
print(f"Extracted {N_FRAMES} frames ({W}x{H} @ {FPS:.1f} fps) to {FRAMES_DIR}")

first_frame = cv2.imread(os.path.join(FRAMES_DIR, "00000.jpg"))


# =================== CELL 4: detect + pick players ==========================
# Runs YOLO on frame 0, draws numbered boxes, shows them. Set KEEP_IDS in
# CELL 2 to the numbers you want, then re-run this cell to confirm.

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


# ===================== CELL 5: track (one sweep per player) =================
assert PICKED, "No players selected — set KEEP_IDS or PLAYER_BOXES."
print(f"Tracking {len(PICKED)} player(s) with SAMURAI.")

model_cfg = determine_model_cfg(MODEL_PATH)
predictor = build_sam2_video_predictor(model_cfg, MODEL_PATH, device="cuda:0")

# masks_by_frame[frame_idx][player_id] = boolean mask
masks_by_frame = [dict() for _ in range(N_FRAMES)]

for pid, (x, y, bw, bh) in enumerate(PICKED):
    bbox_xyxy = np.array([x, y, x + bw, y + bh], dtype=np.float32)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(FRAMES_DIR, offload_video_to_cpu=True)
        predictor.add_new_points_or_box(state, box=bbox_xyxy, frame_idx=0, obj_id=0)
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
            mask = (mask_logits[0, 0] > 0.0).cpu().numpy()
            if mask.any():
                masks_by_frame[frame_idx][pid] = mask
    del state
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  player P{pid}: swept {N_FRAMES} frames")


# ===================== CELL 6: render + download ============================
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

writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
for frame_idx in range(N_FRAMES):
    frame = cv2.imread(os.path.join(FRAMES_DIR, f"{frame_idx:05d}.jpg"))
    for pid, mask in masks_by_frame[frame_idx].items():
        color = COLORS[pid % len(COLORS)]
        m = overlay_mask(frame, mask, color)
        draw_label(frame, m, f"P{pid}", color)
    writer.write(frame)
writer.release()
print(f"Done — {N_FRAMES} frames written to {OUTPUT_PATH}")

!ffmpeg -y -loglevel error -i {OUTPUT_PATH} -vcodec libx264 -pix_fmt yuv420p /content/samurai_players_h264.mp4
from google.colab import files
files.download("/content/samurai_players_h264.mp4")
