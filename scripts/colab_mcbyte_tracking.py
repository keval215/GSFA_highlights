# =============================================================================
# GSFA — McByte individual player tracking, full mask mode (Google Colab)
# https://arxiv.org/pdf/2506.01373  |  https://trackers.roboflow.com/develop/trackers/mcbyte/
# =============================================================================
# Pipeline (3 stages):
#   1. DETECTION        — your YOLOv11 model (aiff_v2.pt on Drive) finds players
#   2. TEAM CLASSIFY     — SigLIP embeddings -> UMAP -> KMeans(2), same recipe
#                          as GSFATeamClassifier in the main repo
#   3. TRACK (McByte)    — YOLO boxes go into McByteTracker.update() every frame.
#                          Full mode also feeds the RGB frame, which drives an
#                          internal SAM(vit_b) box-prompt + Cutie mask-propagation
#                          pipeline used as an extra association cue on top of
#                          McByte's IoU/Hungarian matching + camera-motion
#                          compensation. Reported HOTA: SoccerNet 85.0,
#                          SportsMOT 76.5 — both directly relevant to futsal.
#
# WHY THIS SCRIPT EXISTS (read before trusting the output)
#   McByte's `lost_track_buffer` controls how long a LOST track's id-slot is
#   kept alive (frames, expressed at 30fps, scaled by frame_rate below) — set
#   to LOST_BUFFER_SECONDS here. It is NOT appearance-based re-identification.
#   Re-matching within that window still needs the returning box to land on
#   the tracklet's Kalman-predicted position (positive IoU); Cutie's mask
#   propagation only runs while the object stays visible somewhere in frame —
#   it has nothing to propagate against while a player is fully off-screen.
#   So a 5s buffer keeps the id-slot open for 5s, it does NOT guarantee a
#   player who left frame and re-entered elsewhere on the pitch gets the same
#   id back. CELL 9's gap report measures this empirically instead of assuming
#   it. This script deliberately does NOT bolt on a hand-rolled appearance
#   gallery (unlike scripts/colab_sam2_player_tracking.py) — the point here is
#   to see what McByte alone actually does. If the gap report shows real id
#   breaks on re-entry, port that script's gallery (SigLIP similarity +
#   jersey-number OCR veto) on top of this tracker's output.
#
# How to run:
#   - Runtime -> Change runtime type -> GPU (T4 works)
#   - Paste each "CELL n" block into its own Colab cell, in order
#   - Upload your clip to the Colab session, set VIDEO_PATH in CELL 2, run all
#
# Outputs (in /content/output/):
#   tracked_output_h264.mp4  — boxes + track_id + team colour
#   tracks.csv               — frame, track_id, team, bbox
#
# Known limitations (deliberate, to keep this a clean read on McByte itself):
#   - No custom re-id gallery (see above) — that's the whole point of the test.
#   - Goalkeepers wear a third kit, so KMeans(2) lumps each GK into whichever
#     team cluster is closer (same caveat as the SAM2 script).
#   - Team is voted online per track (majority so far), so early frames of a
#     new track may show a shaky team colour before the vote settles.
# =============================================================================


# =============================== CELL 1: setup ===============================
# trackers[mask] pulls in torch/torchvision + rf-segment-anything (SAM vit_b)
# + rf-cutie. SAM/Cutie checkpoints download automatically on first use — no
# manual wget needed (unlike raw SAM2 in the other Colab script).

!pip install -q "trackers[mask]" ultralytics transformers umap-learn scikit-learn

from google.colab import drive
drive.mount('/content/drive')


# =============================== CELL 2: config ==============================
import os

MODEL_PATH = "/content/drive/MyDrive/aiff_v2.pt"   # your YOLOv11 weights on Drive
VIDEO_PATH = "/content/2.mp4"                      # video to track (upload it to /content/)

TARGET_FPS       = 15    # resample video to this fps (keeps VRAM/RAM in budget on T4)
MAX_FRAMES       = 600   # hard cap on frames processed (600 @ 15fps = 40s)
PLAYER_CONF      = 0.50  # detection confidence gate
TEAM_FIT_STRIDE  = 2     # sample every Nth resampled frame when collecting fit crops
TEAM_VOTE_STRIDE = 2     # re-classify each track's team every Nth frame, majority vote

# --- the question this script exists to answer ---
LOST_BUFFER_SECONDS = 5.0   # how long McByte keeps a lost track's id-slot alive.
                             # This is the buffer size, not a re-id guarantee — see
                             # the header comment and CELL 9's gap report.

# --- McByte mask pipeline (full mode) ---
SAM_MODEL_TYPE          = "vit_b"   # SAM's smallest backbone — kept small on purpose for 4-16GB GPUs
CUTIE_MAX_INTERNAL_SIZE = 480       # Cutie's internal working resolution (downscaled, restored on output)
ENABLE_ISOLATED_MASK_MATCHING = False  # let masks rescue below-threshold IoU candidates; costs more compute

# --- team classifier fit-quality gates (same as production GSFATeamClassifier) ---
MIN_CROP_PX = 32
BLUR_THRESHOLD = 80
FIT_SILHOUETTE_MIN = 0.20

OUTPUT_DIR = "/content/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

assert VIDEO_PATH and os.path.exists(VIDEO_PATH), f"Video not found: {VIDEO_PATH}"
assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"


# ====================== CELL 3: imports, device, models ======================
import csv
import time
from collections import Counter, defaultdict

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from transformers import AutoProcessor, SiglipVisionModel
from ultralytics import YOLO
import umap

from trackers import McByteTracker, McByteMaskConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "Enable a GPU runtime (Runtime -> Change runtime type -> GPU)"
print(f"Device: {torch.cuda.get_device_name(0)} "
      f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

# --- YOLO detector ---
yolo = YOLO(MODEL_PATH)
print("Model classes:", yolo.names)
PLAYER_CLASS_IDS = [i for i, n in yolo.names.items() if "player" in n.lower()]
if not PLAYER_CLASS_IDS:
    PLAYER_CLASS_IDS = [0]  # fallback: assume class 0 is the player class
print(f"Player class ids: {PLAYER_CLASS_IDS}")

# --- SigLIP embedder (same backbone as GSFATeamClassifier) ---
SIGLIP_ID = "google/siglip-base-patch16-224"
siglip_processor = AutoProcessor.from_pretrained(SIGLIP_ID)
siglip = SiglipVisionModel.from_pretrained(SIGLIP_ID).to(DEVICE).eval()


# ========================= CELL 4: helper functions ===========================

def detect_players(frame_bgr):
    """Run YOLO on one frame, return player boxes + confidences (N,4) / (N,)."""
    res = yolo.predict(frame_bgr, conf=PLAYER_CONF, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    cls = res.boxes.cls.cpu().numpy().astype(int)
    keep = np.isin(cls, PLAYER_CLASS_IDS)
    boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)[keep]
    confs = res.boxes.conf.cpu().numpy().astype(np.float32)[keep]
    return boxes, confs


def torso_crop(frame_bgr, box, min_px=4):
    """Top 55% of the player bbox — jersey region, matches the repo's classifier."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, x2 = max(0, x1), min(w, x2)
    y1 = max(0, y1)
    y2 = min(h, y1 + int((y2 - y1) * 0.55))
    if x2 - x1 < min_px or y2 - y1 < min_px:
        return None
    return frame_bgr[y1:y2, x1:x2]


def sharpness_score(crop_bgr):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def is_sharp(crop_bgr, threshold=BLUR_THRESHOLD):
    return sharpness_score(crop_bgr) >= threshold


@torch.inference_mode()
def embed_crops(crops_bgr, batch_size=32):
    """SigLIP-embed a list of BGR crops -> (N, 768) numpy."""
    out = []
    for i in range(0, len(crops_bgr), batch_size):
        pil = [
            Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
            for c in crops_bgr[i : i + batch_size]
        ]
        inputs = siglip_processor(images=pil, return_tensors="pt").to(DEVICE)
        out.append(siglip(**inputs).pooler_output.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def iter_resampled_frames(video_path, target_fps, max_frames):
    """Yield (kept_idx, frame_bgr) resampled from source fps to target_fps,
    stopping at max_frames. Two independent passes (CELL 5 and CELL 8) each
    call this fresh — cheap relative to detection/embedding cost, and avoids
    holding the whole clip's frames in RAM the way JPEG pre-extraction does."""
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_idx, kept, last_kept = 0, 0, -1
    while kept < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        idx = int(src_idx * target_fps / src_fps)
        if idx > last_kept:
            yield kept, frame
            last_kept = idx
            kept += 1
        src_idx += 1
    cap.release()


_probe_cap = cv2.VideoCapture(VIDEO_PATH)
SRC_FPS = _probe_cap.get(cv2.CAP_PROP_FPS) or 30.0
FRAME_W = int(_probe_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_H = int(_probe_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
_probe_cap.release()
print(f"{SRC_FPS:.1f} fps source, {FRAME_W}x{FRAME_H} -> resampling to {TARGET_FPS} fps")


# ================= CELL 5: detection probe + team-fit crops ===================

probe_boxes = []
fit_crops = []
n_resampled = 0
for kept_idx, frame in iter_resampled_frames(VIDEO_PATH, TARGET_FPS, MAX_FRAMES):
    n_resampled += 1
    boxes, _ = detect_players(frame)
    if kept_idx < 30:  # ~2s at 15fps
        probe_boxes.append(len(boxes))
    if kept_idx % TEAM_FIT_STRIDE == 0:
        for box in boxes:
            crop = torso_crop(frame, box, min_px=MIN_CROP_PX)
            if crop is not None and is_sharp(crop):
                fit_crops.append(crop)

print(f"Resampled {n_resampled} frames")
print(f"Detections per frame (first {len(probe_boxes)} frames): {probe_boxes}")
if probe_boxes:
    print(f"  min={min(probe_boxes)} max={max(probe_boxes)} "
          f"avg={sum(probe_boxes)/len(probe_boxes):.1f}")
    if max(probe_boxes) < 6:
        print("  WARNING: few players found even at the best frame — try lowering "
              f"PLAYER_CONF (currently {PLAYER_CONF}) and re-running from CELL 3.")
print(f"Collected {len(fit_crops)} torso crops for team fitting")


# ============== CELL 6: team classifier fit (SigLIP -> UMAP -> KMeans) ========

fit_embeddings = embed_crops(fit_crops)
del fit_crops

reducer = umap.UMAP(n_components=3, random_state=42)
projected = reducer.fit_transform(fit_embeddings)
del fit_embeddings

team_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(projected)
print("Team clusters fitted:", np.bincount(team_kmeans.labels_))

sil_score = silhouette_score(projected, team_kmeans.labels_)
print(f"Fit silhouette score: {sil_score:.3f} (want >= {FIT_SILHOUETTE_MIN})")
if sil_score < FIT_SILHOUETTE_MIN:
    print("  WARNING: teams didn't separate cleanly — try lowering PLAYER_CONF, "
          "check lighting/camera angle consistency, or confirm jerseys are visually distinct.")


def classify_crops(crops_bgr):
    emb = embed_crops(crops_bgr)
    return team_kmeans.predict(reducer.transform(emb))


# ==================== CELL 7: build the McByte tracker =========================
# lost_track_buffer's unit is "frames at 30fps" regardless of the video's own
# rate — frame_rate below is what converts it to real elapsed frames for THIS
# clip's TARGET_FPS. See header comment for what this buffer does and doesn't do.

lost_track_buffer = round(LOST_BUFFER_SECONDS * 30)

mask_config = McByteMaskConfig(
    device=DEVICE,
    sam_model_type=SAM_MODEL_TYPE,
    cutie_max_internal_size=CUTIE_MAX_INTERNAL_SIZE,
)

tracker = McByteTracker(
    lost_track_buffer=lost_track_buffer,
    frame_rate=TARGET_FPS,
    enable_mask_manager=True,
    mask_config=mask_config,
    enable_isolated_mask_matching=ENABLE_ISOLATED_MASK_MATCHING,
)
print(f"McByteTracker ready — full mask mode (SAM {SAM_MODEL_TYPE} + Cutie), "
      f"lost_track_buffer={lost_track_buffer} => {lost_track_buffer/30:.1f}s real time "
      f"at {TARGET_FPS}fps")

torch.cuda.reset_peak_memory_stats()


# ==================== CELL 8: tracking pass + render + CSV =====================

writer = cv2.VideoWriter(
    os.path.join(OUTPUT_DIR, "tracked_output.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"),
    TARGET_FPS,
    (FRAME_W, FRAME_H),
)
csv_path = os.path.join(OUTPUT_DIR, "tracks.csv")

TEAM_COLORS = {0: (60, 200, 255), 1: (255, 120, 60)}  # BGR: amber vs blue-ish
track_votes = defaultdict(Counter)
track_frames = defaultdict(list)  # tracker_id -> list of resampled frame indices seen

t0 = time.time()
with open(csv_path, "w", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["frame", "timestamp_s", "track_id", "team", "x1", "y1", "x2", "y2", "confidence"])

    for kept_idx, frame_bgr in iter_resampled_frames(VIDEO_PATH, TARGET_FPS, MAX_FRAMES):
        boxes, confs = detect_players(frame_bgr)
        detections = (
            sv.Detections(xyxy=boxes, confidence=confs, class_id=np.zeros(len(boxes), dtype=int))
            if len(boxes) else sv.Detections.empty()
        )

        # McByte's SAM/Cutie backends want RGB frames — passing BGR here silently
        # degrades mask quality instead of erroring, so don't skip this conversion.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            tracked = tracker.update(detections, frame=frame_rgb)
        except torch.cuda.OutOfMemoryError:
            print(f"Frame {kept_idx}: CUDA OOM in the mask pipeline. McByte disables "
                  "mask-conditioned association after 3 consecutive OOMs and falls back "
                  "to IoU-only for the rest of the run — if this keeps happening, lower "
                  "CUDA_MAX_INTERNAL_SIZE or drop SAM_MODEL_TYPE, or move to a bigger GPU.")
            tracked = sv.Detections.empty()

        out_frame = frame_bgr
        if len(tracked):
            crops, crop_idx = [], []
            for i in range(len(tracked)):
                if kept_idx % TEAM_VOTE_STRIDE == 0:
                    crop = torso_crop(frame_bgr, tracked.xyxy[i])
                    if crop is not None:
                        crops.append(crop)
                        crop_idx.append(i)
            if crops:
                teams = classify_crops(crops)
                for i, team in zip(crop_idx, teams):
                    track_votes[int(tracked.tracker_id[i])][int(team)] += 1

            for i in range(len(tracked)):
                tid = int(tracked.tracker_id[i])
                x1, y1, x2, y2 = tracked.xyxy[i]
                conf = float(tracked.confidence[i]) if tracked.confidence is not None else -1.0
                team = track_votes[tid].most_common(1)[0][0] if track_votes[tid] else 0
                color = TEAM_COLORS[team]

                csv_writer.writerow([kept_idx, kept_idx / TARGET_FPS, tid, team,
                                      *[round(v, 1) for v in (x1, y1, x2, y2)], conf])
                track_frames[tid].append(kept_idx)

                cv2.rectangle(out_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"#{tid}"
                cv2.putText(out_frame, label, (int(x1), int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(out_frame, label, (int(x1), int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(out_frame, f"frame {kept_idx}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(out_frame)

        if kept_idx % 100 == 0:
            print(f"  frame {kept_idx}  active_ids={len(tracked)}  elapsed={time.time()-t0:.1f}s")

writer.release()
elapsed = time.time() - t0
print(f"\nDone: {n_resampled} frames in {elapsed:.1f}s ({n_resampled/max(elapsed,1e-6):.1f} fps overall)")
print(f"Wrote {OUTPUT_DIR}/tracked_output.mp4 and {csv_path}")

peak_gb = torch.cuda.max_memory_allocated() / 1e9
total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"Peak CUDA memory: {peak_gb:.2f} GB / {total_gb:.2f} GB total")
if peak_gb > 0.9 * total_gb:
    print("  -> Close to the VRAM ceiling on this GPU.")


# ========================= CELL 9: gap report ===================================
# Does the same track_id survive a re-entry, or does a new id appear near where
# an old one vanished? This is the actual answer to "can McByte handle a player
# re-entering after ~5s" for THIS clip — not a guess from the buffer setting.

print(f"\n--- Gap report (lost_track_buffer = {lost_track_buffer} frames "
      f"= {lost_track_buffer/30:.1f}s, at {TARGET_FPS} fps) ---")
print(f"Unique track IDs seen: {len(track_frames)}")

events = []
for tid, frames in track_frames.items():
    frames = sorted(frames)
    for a, b in zip(frames, frames[1:]):
        gap = b - a
        if gap > 1:
            events.append((tid, a, b, gap, gap / TARGET_FPS))

if not events:
    print("No re-appearance gaps observed for any track (continuous throughout).")
else:
    events.sort(key=lambda e: -e[3])
    print(f"{len(events)} within-track gap(s) found (same id before/after — a successful "
          "McByte re-match; still required IoU overlap on return):")
    for tid, a, b, gap_f, gap_s in events[:20]:
        near_limit = "  <= near buffer limit" if gap_f > 0.8 * lost_track_buffer else ""
        print(f"  track_id={tid:>4}  frame {a:>5} -> {b:>5}  gap={gap_f:>4}f ({gap_s:.2f}s){near_limit}")

print(
    "\nThis only lists gaps McByte itself bridged under the SAME id. A real player "
    "who exited and came back as a NEW id shows up as an unrelated track_id starting "
    "near where another one permanently stopped — check tracked_output.mp4 visually "
    "for the specific re-entry you care about."
)


# ======================= CELL 10: preview / download ============================

!ffmpeg -y -loglevel error -i /content/output/tracked_output.mp4 \
    -vcodec libx264 -crf 24 /content/output/tracked_output_h264.mp4

from google.colab import files
files.download("/content/output/tracked_output_h264.mp4")
files.download("/content/output/tracks.csv")
