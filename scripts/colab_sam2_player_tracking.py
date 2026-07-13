# =============================================================================
# GSFA — SAM2 individual player tracking (Google Colab)
# =============================================================================
# Pipeline (3 stages, as discussed):
#   1. DETECTION        — your YOLOv11 model (aiff_v2.pt on Drive) finds players
#   2. TEAM CLASSIFY    — SigLIP embeddings -> UMAP -> KMeans(2), same recipe
#                         as GSFATeamClassifier in the main repo
#   3. INDIVIDUAL TRACK — YOLO boxes are fed as *box prompts* into SAM2's video
#                         predictor; SAM2's memory bank keeps each player's
#                         identity through occlusion / brief frame exits.
#                         No segmentation model needed — boxes are enough.
#
# How to run:
#   - Runtime -> Change runtime type -> GPU (T4 works; A100 is much faster)
#   - Paste each "CELL n" block below into its own Colab cell, in order
#   - Upload your 30s video to the Colab session, paste its path into
#     VIDEO_PATH in CELL 2, then run everything
#
# Outputs (in /content/output/):
#   tracked_output.mp4  — mask overlay per player, coloured by team, with a
#                         stable ID label that survives occlusion
#   tracks.csv          — frame, track_id, team, bbox — queryable motion data
#
# Known limitations of this demo script (deliberate, to keep it simple):
#   - A new player can take up to ~REDETECT_STRIDE frames to pick up an id
#     after entering frame.
#   - A hard camera cut can spawn a duplicate id for someone already tracked
#     (no scene-cut detector here).
#   - Goalkeepers wear a third kit, so KMeans(2) lumps each GK into whichever
#     team cluster is closer.
#   - By default (ENABLE_ROLLING_MEMORY=True) SAM2 runs one continuous session
#     for the whole clip, evicting old frame memory instead of resetting at
#     hard chunk boundaries — see evict_old_memory(). This reaches into SAM2
#     internals that aren't part of its documented public API; set
#     ENABLE_ROLLING_MEMORY=False to fall back to the older, proven
#     independent-CHUNK_FRAMES-sessions behavior if it misbehaves.
#   - Re-id (appearance similarity + jersey-number OCR veto) uses a capped
#     gallery of MAX_GALLERY_SHOTS crops per track, not every crop ever seen.
#   - Jersey-number OCR is a veto on top of embedding similarity, never the
#     sole re-id signal — broadcast-distance numbers are frequently illegible.
# =============================================================================


# =============================== CELL 1: setup ===============================
# Installs + SAM2 checkpoint download. Takes a couple of minutes.

!pip install -q ultralytics transformers umap-learn
!pip install -q "git+https://github.com/facebookresearch/sam2.git"
!pip install -q easyocr   # jersey-number re-id veto. EasyOCR is torch-based, so it
                          # shares the notebook's existing PyTorch stack — unlike
                          # PaddleOCR, which bundles its own OpenMP runtime and hard-
                          # crashes the Colab kernel when initialised alongside torch.

!mkdir -p /content/checkpoints
!wget -q -O /content/checkpoints/sam2.1_hiera_small.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

from google.colab import drive
drive.mount('/content/drive')


# =============================== CELL 2: config ==============================
import os

MODEL_PATH = "/content/drive/MyDrive/aiff_v2.pt"   # your YOLOv11 weights on Drive
VIDEO_PATH = "/content/2.mp4"                      # video to track (upload it to /content/)

SAM2_CHECKPOINT = "/content/checkpoints/sam2.1_hiera_small.pt"
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_s.yaml"

TARGET_FPS      = 15     # resample video to this fps (keeps SAM2 RAM in budget on Colab)
MAX_FRAMES      = 600    # hard cap on frames processed
PLAYER_CONF     = 0.50   # detection confidence gate (lower to ~0.35-0.40 if CELL 5's
                          # detection-count diagnostic finds fewer players than your roster)
TEAM_FIT_STRIDE = 2      # sample every Nth frame when collecting crops to fit teams
TEAM_VOTE_STRIDE = 2     # re-classify each track every Nth frame, majority vote
REDETECT_STRIDE = 10     # re-run detection every N frames (~0.6s @ 15fps)
IOU_MATCH_THRESH = 0.3   # new detection counts as "already tracked" above this IoU
CHUNK_FRAMES = 150       # frames per independent SAM2 session (~10s @ 15fps) — keeps
                          # SAM2's own memory bounded on long clips instead of growing
                          # for the whole video. Only used when ENABLE_ROLLING_MEMORY
                          # is False.
ENABLE_ROLLING_MEMORY = True  # run ONE continuous SAM2 session for the whole clip,
                          # bounding memory by evicting old frame memory (see
                          # evict_old_memory()) instead of a hard reset every
                          # CHUNK_FRAMES — keeps temporal identity across what used to
                          # be a chunk boundary. This reaches into SAM2 internals that
                          # aren't part of its documented public API: unverified until
                          # run in Colab. Set False to fall back to the proven
                          # chunked-reset behavior below.
MEMORY_WINDOW_FRAMES = 150  # how many recent frames of SAM2 memory to keep when
                          # ENABLE_ROLLING_MEMORY is True — same magnitude as
                          # CHUNK_FRAMES, but a rolling eviction window instead of a
                          # hard reset boundary
LOST_AFTER_FRAMES = 30   # ~2s @ 15fps — a track not seen this long stops being
                          # re-seeded at chunk boundaries (still eligible for re-id below)
REID_SIM_THRESH = 0.85   # cosine similarity above which a new, unmatched detection is
                          # treated as a previously-lost track reappearing — matched
                          # against every look in that track's gallery, not just the
                          # latest one
MAX_GALLERY_SHOTS = 5    # cap on how many crops/embeddings a track's re-id gallery
                          # keeps — a single frozen reference is too fragile to one bad
                          # frame (blur/occlusion/angle), but unbounded growth is
                          # wasted RAM/disk; once full, a new crop only displaces the
                          # gallery's current weakest (least sharp) entry
OCR_CONF_THRESH = 0.75   # jersey-number OCR reads below this confidence are treated
                          # as illegible (expected often, at broadcast distance) —
                          # OCR is a re-id veto/tie-breaker on top of embedding
                          # similarity, never the sole signal
MAX_MASK_AREA_FRAC = 0.20  # reject a mask covering more of the frame than this fraction
MAX_AREA_GROWTH = 3.0    # reject a mask more than this many times bigger than that
                          # track's last known size
DUPLICATE_IOU_THRESH = 0.6  # two different-numbered tracks overlapping at least this
                          # much, both seen very recently, are the same real player —
                          # merge them, keeping the older id
MIN_CROP_PX = 32         # reject torso crops smaller than this on a side (matches
                          # production GSFATeamClassifier's MIN_CROP_PX)
BLUR_THRESHOLD = 80      # reject blurry fit crops below this Laplacian variance
FIT_SILHOUETTE_MIN = 0.20  # warn if the two team clusters don't separate cleanly

FRAMES_DIR = "/content/frames"
OUTPUT_DIR = "/content/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

assert VIDEO_PATH, "Upload your video to the Colab session and paste its path into VIDEO_PATH"
assert os.path.exists(VIDEO_PATH), f"Video not found: {VIDEO_PATH}"
assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"


# ====================== CELL 3: imports, device, models ======================
import csv
import shutil
from collections import Counter, defaultdict

import re

import cv2
import easyocr
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from transformers import AutoProcessor, SiglipVisionModel
from ultralytics import YOLO
import umap

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "Enable a GPU runtime (Runtime -> Change runtime type -> GPU)"

# bf16 needs Ampere+ (A100/L4); T4 gets fp16
AUTOCAST_DTYPE = (
    torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
)
print(f"Device: {torch.cuda.get_device_name(0)} | autocast dtype: {AUTOCAST_DTYPE}")

# --- YOLO detector ---
yolo = YOLO(MODEL_PATH)
print("Model classes:", yolo.names)

# Work out which class ids are players / referees from the class names.
PLAYER_CLASS_IDS = [i for i, n in yolo.names.items() if "player" in n.lower()]
REFEREE_CLASS_IDS = [i for i, n in yolo.names.items() if "referee" in n.lower()]
if not PLAYER_CLASS_IDS:
    PLAYER_CLASS_IDS = [0]  # fallback: assume class 0 is the player class
print(f"Player class ids: {PLAYER_CLASS_IDS} | referee class ids: {REFEREE_CLASS_IDS}")

# --- SigLIP embedder (same backbone as GSFATeamClassifier) ---
SIGLIP_ID = "google/siglip-base-patch16-224"
siglip_processor = AutoProcessor.from_pretrained(SIGLIP_ID)
siglip = SiglipVisionModel.from_pretrained(SIGLIP_ID).to(DEVICE).eval()

# --- EasyOCR (jersey-number re-id veto, secondary signal only) ---
# Runs on CPU on purpose: OCR only fires on gallery-admitted crops (a handful
# per track, not per-frame), so throughput isn't the bottleneck, and keeping it
# off the GPU avoids competing with SAM2 for VRAM.
ocr_reader = easyocr.Reader(["en"], gpu=False)


def detect_players(frame_bgr):
    """Run YOLO on one frame, return player boxes as float32 (N, 4) xyxy."""
    res = yolo.predict(frame_bgr, conf=PLAYER_CONF, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    cls = res.boxes.cls.cpu().numpy().astype(int)
    keep = np.isin(cls, PLAYER_CLASS_IDS)
    return res.boxes.xyxy.cpu().numpy().astype(np.float32)[keep]


def iou(box_a, box_b):
    """Standard IoU between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


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
    """Laplacian variance — higher is sharper. Used both as a blur gate and to
    rank gallery candidates against each other."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def is_sharp(crop_bgr, threshold=BLUR_THRESHOLD):
    """Blur check — matches production's fit-time filter."""
    return sharpness_score(crop_bgr) >= threshold


def ocr_jersey_number(crop_bgr, upscale=3):
    """OCR a jersey number off a torso crop. Returns (digits, confidence), or
    (None, 0.0) if nothing digit-only clears OCR_CONF_THRESH — expect this often,
    broadcast-distance jersey numbers are frequently occluded/illegible, which is
    why OCR is only ever used as a veto on top of embedding similarity, never the
    sole re-id signal."""
    h, w = crop_bgr.shape[:2]
    big = cv2.resize(crop_bgr, (w * upscale, h * upscale), interpolation=cv2.INTER_CUBIC)
    # allowlist restricts recognition to digits — jersey numbers only, which also
    # rejects stray text on kit/boards. detail=1 -> (bbox, text, confidence).
    results = ocr_reader.readtext(big, allowlist="0123456789", detail=1)
    best_digits, best_conf = None, 0.0
    for _box, text, conf in results:
        digits = re.sub(r"\D", "", text)
        if digits and conf > best_conf:
            best_digits, best_conf = digits, float(conf)
    if best_conf < OCR_CONF_THRESH:
        return None, 0.0
    return best_digits, best_conf


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


# ========================= CELL 4: extract frames ============================
# SAM2's video predictor wants a directory of JPEG frames named 00000.jpg, ...
# We resample to TARGET_FPS so a 30s clip stays within Colab's RAM budget.

shutil.rmtree(FRAMES_DIR, ignore_errors=True)
os.makedirs(FRAMES_DIR)

cap = cv2.VideoCapture(VIDEO_PATH)
src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

n_frames, src_idx, last_kept = 0, 0, -1
while n_frames < MAX_FRAMES:
    ok, frame = cap.read()
    if not ok:
        break
    kept_idx = int(src_idx * TARGET_FPS / src_fps)
    if kept_idx > last_kept:
        cv2.imwrite(os.path.join(FRAMES_DIR, f"{n_frames:05d}.jpg"), frame)
        last_kept = kept_idx
        n_frames += 1
    src_idx += 1
cap.release()

print(f"{src_fps:.1f} fps source -> {n_frames} frames at {TARGET_FPS} fps "
      f"({frame_w}x{frame_h})")


def read_frame(idx):
    return cv2.imread(os.path.join(FRAMES_DIR, f"{idx:05d}.jpg"))


# ================= CELL 5: detection pass (stage 1 of 3) =====================
# (a) Diagnostic: print per-frame detection counts for the first ~2s so a bad
#     PLAYER_CONF threshold (or a class-id mismatch) is obvious immediately,
#     instead of silently costing us players later.
# (b) Collect torso crops across the whole clip to fit the team classifier.
#
# NOTE: there's no more single "anchor frame" — Cell 7 now seeds tracks at
# frame 0 and re-detects periodically through the whole clip, so a player
# missed here isn't lost permanently the way it used to be.

probe_frames = min(30, n_frames)  # ~2s at 15fps
probe_counts = [len(detect_players(read_frame(i))) for i in range(probe_frames)]
print(f"Detections per frame (first {probe_frames} frames): {probe_counts}")
print(f"  min={min(probe_counts)} max={max(probe_counts)} "
      f"avg={sum(probe_counts) / len(probe_counts):.1f}")
if max(probe_counts) < 6:
    print("  WARNING: even the best of these frames found few players — "
          "if this is well below your actual roster size, try lowering "
          "PLAYER_CONF (currently "
          f"{PLAYER_CONF}) in CELL 2 and re-running from CELL 3.")

fit_crops = []
for idx in range(0, n_frames, TEAM_FIT_STRIDE):
    frame = read_frame(idx)
    for box in detect_players(frame):
        crop = torso_crop(frame, box, min_px=MIN_CROP_PX)
        if crop is not None and is_sharp(crop):
            fit_crops.append(crop)
print(f"Collected {len(fit_crops)} torso crops for team fitting "
      f"(after {MIN_CROP_PX}px size + blur filtering)")


# ============== CELL 6: team classifier fit (stage 2 of 3) ===================
# SigLIP -> UMAP(3d) -> KMeans(2): the GSFATeamClassifier recipe.

from sklearn.metrics import silhouette_score

fit_embeddings = embed_crops(fit_crops)
del fit_crops  # thousands of full-res crops held live for no reason after this

reducer = umap.UMAP(n_components=3, random_state=42)
projected = reducer.fit_transform(fit_embeddings)
del fit_embeddings  # only the projection is needed from here on

team_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(projected)
print("Team clusters fitted:", np.bincount(team_kmeans.labels_))

# Fit-quality guard (matches production's FIT_SILHOUETTE_MIN check): if the
# two clusters don't separate cleanly, team assignments downstream will be
# noisy — flag it instead of silently trusting a bad fit.
sil_score = silhouette_score(projected, team_kmeans.labels_)
print(f"Fit silhouette score: {sil_score:.3f} (want >= {FIT_SILHOUETTE_MIN})")
if sil_score < FIT_SILHOUETTE_MIN:
    print("  WARNING: teams didn't separate cleanly — try lowering PLAYER_CONF, "
          "check for inconsistent lighting/camera angle across the clip, or "
          "confirm both team jerseys are visually distinct.")


def classify_crops(crops_bgr):
    """Return a team id (0/1) per crop."""
    emb = embed_crops(crops_bgr)
    return team_kmeans.predict(reducer.transform(emb))


# ================ CELL 7: SAM2 tracking (stage 3 of 3) =======================
# Seed SAM2 with detections, propagate in short segments, and at each segment
# boundary re-run detection and inject a *new* SAM2 object for any detection
# that doesn't IoU-match an already-tracked box — this lets a player missed
# earlier (occluded, off-camera, or after a camera-angle change) still pick
# up a track partway through the clip. Within a segment, propagate_in_video's
# memory bank keeps identity through occlusion on its own.
#
# SAM2's own memory use grows with total frames x objects propagated. By
# default (ENABLE_ROLLING_MEMORY=True) we run ONE session for the whole clip
# and bound its memory with evict_old_memory(), which prunes cached frame
# outputs older than MEMORY_WINDOW_FRAMES — this keeps temporal identity
# continuous instead of losing it at a hard boundary. Set
# ENABLE_ROLLING_MEMORY=False to fall back to the older, proven approach of
# independent CHUNK_FRAMES-sized sessions, each dropped before the next starts.
#
# In the fallback (chunked) path, at each chunk boundary only tracks seen
# within LOST_AFTER_FRAMES are re-seeded from their last box; anything older
# relies on appearance re-id (SigLIP similarity + jersey-number OCR veto)
# instead of a guessed position if it reappears.
#
# consume() also rejects masks that are implausibly large or that suddenly
# ballooned past a track's last known size — SAM2 sometimes outlines bare
# floor instead of admitting it lost the player.
#
# merge_duplicate_tracks() catches a fast-moving player outrunning the
# re-detection gap and getting handed a second id: if two ids are sitting on
# top of each other, they're the same player — keep the older, drop the newer.
#
# Re-id gallery: last_embedding[obj_id] holds at most MAX_GALLERY_SHOTS
# (embedding, sharpness) pairs, admitted via _admit_to_gallery() — a single
# frozen reference is too fragile to one bad frame, but unbounded growth is
# wasted RAM/disk. track_ocr[obj_id] holds that track's single most confident
# jersey-number read across its gallery, used only as a veto in
# inject_new_detections() when it confidently disagrees with a re-id candidate.

from sam2.build_sam import build_sam2_video_predictor
import gc

CHUNK_DIR = "/content/chunk_frames"
CROPS_DIR = os.path.join(OUTPUT_DIR, "crops")   # per-track crop gallery, written to
                          # disk instead of RAM — one file per track per gallery slot
os.makedirs(CROPS_DIR, exist_ok=True)


def make_chunk_dir(start, end):
    """Symlink a contiguous frame range into its own 0-indexed directory —
    SAM2's init_state loads a whole directory, so chunking needs one. Only used
    when ENABLE_ROLLING_MEMORY is False (FRAMES_DIR is already 0-indexed
    globally, so the rolling-memory path points init_state at it directly)."""
    shutil.rmtree(CHUNK_DIR, ignore_errors=True)
    os.makedirs(CHUNK_DIR)
    for local_idx, global_idx in enumerate(range(start, end)):
        os.symlink(
            os.path.abspath(os.path.join(FRAMES_DIR, f"{global_idx:05d}.jpg")),
            os.path.join(CHUNK_DIR, f"{local_idx:05d}.jpg"),
        )
    return CHUNK_DIR


def evict_old_memory(inference_state, current_frame_idx, keep_last_n):
    """Bound SAM2's memory bank instead of a hard chunk-boundary reset — the
    Det-SAM2 technique, ported onto the official predictor's own inference_state
    rather than depending on that (stale) repo. Prunes each tracked object's
    cached non-conditioning frame outputs older than keep_last_n frames; the
    conditioning (first) frame is untouched since propagate_in_video needs it.

    NOTE: reaches into inference_state's internal structure, which isn't part of
    SAM2's documented public API and can shift between versions — validate this
    still does something (see the A/B check in the plan's verification section)
    the first time this runs against a real installed sam2 version."""
    per_obj = inference_state.get("output_dict_per_obj", {})
    for obj_state in per_obj.values():
        non_cond = obj_state.get("non_cond_frame_outputs")
        if not non_cond:
            continue
        stale = [f for f in list(non_cond) if current_frame_idx - f > keep_last_n]
        for f in stale:
            del non_cond[f]


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


FRAME_AREA = frame_w * frame_h

next_obj_id = 0
last_known_box = {}     # obj_id -> most recent visible box, used for IoU matching + re-seeding
last_seen_frame = {}    # obj_id -> global frame idx it was last actually visible
last_embedding = defaultdict(list)  # obj_id -> that track's re-id gallery, capped at
                          # MAX_GALLERY_SHOTS entries of (embedding, sharpness_score) —
                          # see flush_votes(), which only admits crops that clear the
                          # blur/size gates and displaces the weakest entry once full
track_ocr = {}           # obj_id -> (digits, confidence) — the single most confident
                          # jersey-number read across that track's gallery so far, used
                          # as a re-id veto (see inject_new_detections); most tracks
                          # will have no entry here, broadcast-distance numbers are
                          # frequently illegible
merged_away = set()     # obj_ids that turned out to be a duplicate of another track —
                          # permanently ignored from here on (see merge_duplicate_tracks)

# Per (global) frame per object we keep: bbox (for CSV + labels) and a
# half-res bit-packed mask (for rendering) — full-res masks for a whole
# clip would blow Colab's RAM.
track_boxes = defaultdict(dict)   # frame_idx -> {obj_id: (x1, y1, x2, y2)}
track_masks = defaultdict(dict)   # frame_idx -> {obj_id: (packed_bits, shape)}

# Rolling-batch team voting: embed crops in small batches as they come in
# instead of buffering the whole clip's crops before one final pass. Reuses
# the same embeddings for team voting AND updating last_embedding (re-id).
votes = defaultdict(Counter)
_vote_buffer_crops, _vote_buffer_keys = [], []


def _gallery_crop_path(obj_id, slot):
    return os.path.join(CROPS_DIR, f"track_{obj_id:03d}_{slot}.jpg")


def _admit_to_gallery(obj_id, emb, crop):
    """Add (emb, crop) to obj_id's capped re-id gallery, if it's sharp/large
    enough to trust as a reference — displacing the gallery's current weakest
    entry once MAX_GALLERY_SHOTS is reached, rather than growing forever."""
    h, w = crop.shape[:2]
    if min(h, w) < MIN_CROP_PX:
        return
    score = sharpness_score(crop)
    if score < BLUR_THRESHOLD:
        return
    gallery = last_embedding[obj_id]
    if len(gallery) < MAX_GALLERY_SHOTS:
        slot = len(gallery)
        gallery.append((emb, score))
    else:
        slot = min(range(len(gallery)), key=lambda i: gallery[i][1])
        if score <= gallery[slot][1]:
            return  # not better than what's already kept in this slot
        gallery[slot] = (emb, score)
    cv2.imwrite(_gallery_crop_path(obj_id, slot), crop)

    digits, conf = ocr_jersey_number(crop)
    if digits is not None and conf > track_ocr.get(obj_id, (None, 0.0))[1]:
        track_ocr[obj_id] = (digits, conf)


def flush_votes():
    global _vote_buffer_crops, _vote_buffer_keys
    if not _vote_buffer_crops:
        return
    embeddings = embed_crops(_vote_buffer_crops)
    teams = team_kmeans.predict(reducer.transform(embeddings))
    for obj_id, emb, team, crop in zip(_vote_buffer_keys, embeddings, teams, _vote_buffer_crops):
        votes[obj_id][int(team)] += 1
        _admit_to_gallery(obj_id, emb, crop)
    _vote_buffer_crops, _vote_buffer_keys = [], []


def consume(global_frame_idx, obj_ids, mask_logits):
    frame = read_frame(global_frame_idx) if global_frame_idx % TEAM_VOTE_STRIDE == 0 else None
    for i, obj_id in enumerate(obj_ids):
        if obj_id in merged_away:
            continue  # this id turned out to be a duplicate of another track
        mask = (mask_logits[i, 0] > 0).cpu().numpy()
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue  # occluded / out of frame right now — SAM2 remembers it
        box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

        # Reject implausible masks instead of trusting them blindly — SAM2
        # sometimes outlines something (e.g. bare floor) instead of admitting
        # it lost the real player. Treat a rejected mask the same as "not
        # visible this frame" rather than rendering a phantom blob.
        area = box_area(box)
        if area > MAX_MASK_AREA_FRAC * FRAME_AREA:
            continue
        prev_box = last_known_box.get(obj_id)
        if prev_box is not None:
            prev_area = box_area(prev_box)
            if prev_area > 0 and area > MAX_AREA_GROWTH * prev_area:
                continue

        track_boxes[global_frame_idx][obj_id] = box
        last_known_box[obj_id] = box
        last_seen_frame[obj_id] = global_frame_idx
        small = mask[::2, ::2]
        track_masks[global_frame_idx][obj_id] = (np.packbits(small), small.shape)
        if frame is not None:
            crop = torso_crop(frame, box)
            if crop is not None:
                _vote_buffer_crops.append(crop)
                _vote_buffer_keys.append(obj_id)
                if len(_vote_buffer_crops) >= 32:
                    flush_votes()


def merge_duplicate_tracks(global_frame_idx, active_ids):
    """If two different-numbered tracks are sitting almost exactly on top of
    each other right now, they're the same real player — one track drifted
    onto another. Keep the older id, permanently drop the newer one."""
    candidates = [
        oid for oid in active_ids
        if oid not in merged_away
        and global_frame_idx - last_seen_frame.get(oid, -10**9) <= REDETECT_STRIDE
    ]
    for i, a in enumerate(candidates):
        if a in merged_away:
            continue
        for b in candidates[i + 1:]:
            if b in merged_away:
                continue
            if iou(last_known_box[a], last_known_box[b]) >= DUPLICATE_IOU_THRESH:
                keep_id, drop_id = min(a, b), max(a, b)
                votes[keep_id].update(votes[drop_id])
                del votes[drop_id]
                merged_away.add(drop_id)
                active_ids.discard(drop_id)
                print(f"Frame {global_frame_idx}: merged duplicate track "
                      f"{drop_id} into {keep_id}")


def inject_new_detections(predictor, inference_state, local_frame_idx, global_frame_idx, active_ids):
    """Add a SAM2 object for any detection that doesn't match an active track —
    reviving a lost track by appearance if it matches one, else minting a new id."""
    global next_obj_id
    added = 0
    frame = read_frame(global_frame_idx)
    for box in detect_players(frame):
        best_iou = max((iou(box, last_known_box[oid]) for oid in active_ids), default=0.0)
        if best_iou >= IOU_MATCH_THRESH:
            continue  # already tracked by an active track

        matched_id, best_sim = None, -1.0
        crop = torso_crop(frame, box)
        if crop is not None:
            lost_ids = [oid for oid in last_embedding
                        if oid not in active_ids and oid not in merged_away]
            if lost_ids:
                emb = embed_crops([crop])[0]
                for oid in lost_ids:
                    # match against the best of that track's saved gallery,
                    # not just its single most recent look
                    sim = max(cosine_sim(emb, g_emb) for g_emb, _ in last_embedding[oid])
                    if sim > best_sim:
                        best_sim, matched_id = sim, oid
                if best_sim < REID_SIM_THRESH:
                    matched_id = None

                # Jersey-number OCR as a veto on top of the embedding match — never
                # the sole signal, since broadcast-distance numbers are frequently
                # illegible. Only overrides a match when BOTH sides read a
                # confident, disagreeing number.
                if matched_id is not None:
                    cand_digits, _ = ocr_jersey_number(crop)
                    stored_digits, _ = track_ocr.get(matched_id, (None, 0.0))
                    if (cand_digits is not None and stored_digits is not None
                            and cand_digits != stored_digits):
                        print(f"Frame {global_frame_idx}: OCR mismatch — rejecting "
                              f"re-id match to track {matched_id} (stored "
                              f"#{stored_digits} vs candidate #{cand_digits})")
                        matched_id = None

        if matched_id is not None:
            obj_id = matched_id
            print(f"Frame {global_frame_idx}: re-identified track {obj_id} "
                  f"(similarity {best_sim:.2f}) instead of a new id")
        else:
            obj_id = next_obj_id
            next_obj_id += 1
            added += 1

        predictor.add_new_points_or_box(
            inference_state, frame_idx=local_frame_idx, obj_id=obj_id, box=box
        )
        last_known_box[obj_id] = tuple(float(v) for v in box)
        last_seen_frame[obj_id] = global_frame_idx
        active_ids.add(obj_id)
    return added


def _run_segment(predictor, inference_state, chunk_start, local_start, segment_len, active_ids):
    """Propagate one REDETECT_STRIDE-sized segment, consume its output, merge
    duplicates, and inject any new detections for the next segment. Shared by
    both the rolling-memory and chunked-reset paths below."""
    for local_idx, obj_ids, mask_logits in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=local_start,
        max_frame_num_to_track=segment_len,
    ):
        consume(chunk_start + local_idx, obj_ids, mask_logits)

    merge_duplicate_tracks(chunk_start + local_start + segment_len - 1, active_ids)


with torch.inference_mode(), torch.autocast("cuda", dtype=AUTOCAST_DTYPE):
    if ENABLE_ROLLING_MEMORY:
        # One continuous session for the whole clip — FRAMES_DIR is already
        # 0-indexed globally, so no per-chunk symlink directory is needed here.
        # Memory is bounded by evict_old_memory() instead of a hard reset, so
        # temporal identity survives past what used to be a chunk boundary.
        predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
        inference_state = predictor.init_state(
            video_path=FRAMES_DIR,
            offload_video_to_cpu=True,   # keep the frame tensor in system RAM, not VRAM
            offload_state_to_cpu=True,
        )

        active_ids = set()
        seeded = inject_new_detections(predictor, inference_state, 0, 0, active_ids)
        print(f"Frame 0: seeded {seeded} tracks")

        for local_start in range(0, n_frames, REDETECT_STRIDE):
            segment_len = min(REDETECT_STRIDE, n_frames - local_start)
            _run_segment(predictor, inference_state, 0, local_start, segment_len, active_ids)
            evict_old_memory(inference_state, local_start + segment_len - 1, MEMORY_WINDOW_FRAMES)

            next_local_start = local_start + segment_len
            if next_local_start < n_frames:
                flush_votes()  # keep last_embedding current before the re-id check below
                added = inject_new_detections(
                    predictor, inference_state, next_local_start, next_local_start, active_ids
                )
                if added:
                    print(f"Frame {next_local_start}: +{added} new track(s) (total {next_obj_id})")

        del predictor, inference_state
        gc.collect()
        torch.cuda.empty_cache()

    else:
        # Proven fallback: full session teardown/rebuild every CHUNK_FRAMES.
        for chunk_start in range(0, n_frames, CHUNK_FRAMES):
            chunk_end = min(chunk_start + CHUNK_FRAMES, n_frames)
            chunk_len = chunk_end - chunk_start
            make_chunk_dir(chunk_start, chunk_end)

            predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
            inference_state = predictor.init_state(
                video_path=CHUNK_DIR,
                offload_video_to_cpu=True,   # keep the frame tensor in system RAM, not VRAM
                offload_state_to_cpu=True,
            )

            active_ids = set()
            if chunk_start == 0:
                seeded = inject_new_detections(predictor, inference_state, 0, 0, active_ids)
                print(f"Frame 0: seeded {seeded} tracks")
            else:
                # only re-seed tracks actually seen recently — a track that's
                # been gone longer than LOST_AFTER_FRAMES is left out here and
                # instead relies on appearance re-id if it reappears
                for obj_id, box in last_known_box.items():
                    if obj_id in merged_away:
                        continue
                    if chunk_start - last_seen_frame.get(obj_id, -10**9) <= LOST_AFTER_FRAMES:
                        predictor.add_new_points_or_box(
                            inference_state, frame_idx=0, obj_id=obj_id,
                            box=np.array(box, dtype=np.float32),
                        )
                        active_ids.add(obj_id)
                added = inject_new_detections(predictor, inference_state, 0, chunk_start, active_ids)
                if added:
                    print(f"Frame {chunk_start}: +{added} new track(s) (total {next_obj_id})")

            for local_start in range(0, chunk_len, REDETECT_STRIDE):
                segment_len = min(REDETECT_STRIDE, chunk_len - local_start)
                _run_segment(predictor, inference_state, chunk_start, local_start, segment_len, active_ids)

                next_local_start = local_start + segment_len
                if next_local_start < chunk_len:
                    global_idx = chunk_start + next_local_start
                    flush_votes()  # keep last_embedding current before the re-id check below
                    added = inject_new_detections(
                        predictor, inference_state, next_local_start, global_idx, active_ids
                    )
                    if added:
                        print(f"Frame {global_idx}: +{added} new track(s) (total {next_obj_id})")

            # drop this chunk's SAM2 state before the next chunk starts — this is
            # the actual RAM fix, don't skip it
            del predictor, inference_state
            gc.collect()
            torch.cuda.empty_cache()

    flush_votes()

shutil.rmtree(CHUNK_DIR, ignore_errors=True)

final_track_count = next_obj_id - len(merged_away)
print(f"Tracked {final_track_count} players across {len(track_boxes)} frames "
      f"({len(merged_away)} duplicate id(s) merged away)")
if final_track_count < 6:
    print("  WARNING: still tracking very few players — re-check the "
          "detection-count diagnostic in CELL 5 and consider lowering "
          "PLAYER_CONF in CELL 2.")

track_team = {obj_id: c.most_common(1)[0][0] for obj_id, c in votes.items()}
for obj_id in range(next_obj_id):
    track_team.setdefault(obj_id, 0)  # fallback covers merged ids too, so
                                       # CELL 8 can still render their few
                                       # pre-merge frames without crashing
print("Team assignment:", {oid: t for oid, t in sorted(track_team.items())
                            if oid not in merged_away})


# ==================== CELL 8: render output + tracks CSV =====================
TEAM_COLORS = {0: (60, 200, 255), 1: (255, 120, 60)}  # BGR: amber vs blue-ish
MASK_ALPHA = 0.45

writer = cv2.VideoWriter(
    os.path.join(OUTPUT_DIR, "tracked_output.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"),
    TARGET_FPS,
    (frame_w, frame_h),
)

csv_path = os.path.join(OUTPUT_DIR, "tracks.csv")
with open(csv_path, "w", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["frame", "track_id", "team", "x1", "y1", "x2", "y2"])

    for frame_idx in range(n_frames):
        frame = read_frame(frame_idx)
        overlay = frame.copy()
        for obj_id, (packed, shape) in track_masks.get(frame_idx, {}).items():
            small = np.unpackbits(packed)[: shape[0] * shape[1]].reshape(shape)
            mask = cv2.resize(
                small.astype(np.uint8), (frame_w, frame_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            color = TEAM_COLORS[track_team[obj_id]]
            overlay[mask] = color

            x1, y1, x2, y2 = track_boxes[frame_idx][obj_id]
            csv_writer.writerow([frame_idx, obj_id, track_team[obj_id], x1, y1, x2, y2])
            label = f"#{obj_id}"
            cv2.putText(frame, label, (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(frame, label, (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frame = cv2.addWeighted(overlay, MASK_ALPHA, frame, 1 - MASK_ALPHA, 0)
        cv2.putText(frame, f"frame {frame_idx}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)

writer.release()
print(f"Wrote {OUTPUT_DIR}/tracked_output.mp4 and {csv_path}")


# ======================= CELL 9: preview / download ==========================
# Re-encode to h264 so it plays inline / downloads small, then download both.

!ffmpeg -y -loglevel error -i /content/output/tracked_output.mp4 \
    -vcodec libx264 -crf 24 /content/output/tracked_output_h264.mp4

from google.colab import files
files.download("/content/output/tracked_output_h264.mp4")
files.download("/content/output/tracks.csv")
