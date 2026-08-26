# =============================================================================
# GSFA — BoT-SORT player tracking with long-term re-identification (Colab)
# =============================================================================
# Pipeline (4 stages):
#   1. DETECTION        — your YOLO model (Drive .pt) finds players/referees/
#                         ball/goal posts
#   2. TEAM CLASSIFY    — SigLIP embeddings -> UMAP -> KMeans(2), identical
#                         recipe to GSFATeamClassifier
#   3. TRACK            — production's exact BoT-SORT config (boxmot BotSort,
#                         reid_model=None, external SigLIP embeddings) — same
#                         as tracking/player_tracker.py. BoT-SORT's own memory
#                         only covers ~2s (track_buffer=60 @30fps, scaled
#                         internally by frame_rate/30); anyone gone longer
#                         gets a brand-new internal id with nothing checking
#                         whether it's someone already seen.
#   4. LONG-TERM RE-ID  — a raw_id -> stable_id remap layer sitting on top of
#                         BoT-SORT: every new internal id BoT-SORT mints is
#                         checked against a capped appearance gallery of every
#                         "lost" stable id (kept alive up to
#                         LOST_GALLERY_TTL_SECONDS after last seen), with a
#                         small local VLM (Qwen2-VL-2B-Instruct, 4-bit) called
#                         ONLY to break ties in an ambiguous cosine-similarity
#                         band, and a jersey-number OCR veto on top of that.
#
# What's different from colab_sam2_player_tracking.py, and why:
#   - Tracker backbone is BoT-SORT (box-only, motion+appearance, ECC camera
#     compensation) — the same tracker production actually runs — not SAM2
#     (box+mask, video memory bank). No segmentation cost, no SAM2-internals
#     memory-eviction hacks.
#   - BoT-SORT's own re-id horizon is ~2 seconds (track_buffer, scaled by
#     frame_rate/30 inside boxmot itself). This script adds a SEPARATE,
#     much longer-lived (default 3 minutes) gallery layer on top, because a
#     player leaving frame for a throw-in, a sub, an off-pitch treatment, or
#     a hard camera cut easily exceeds 2 seconds but is still clearly the
#     same person to a human watching the footage.
#   - A small local VLM adjudicates only the middle band of re-id similarity
#     scores that are too ambiguous for cosine similarity alone to trust —
#     never the primary signal, never called on every match.
#   - SigLIP embeddings are mean-pooled over last_hidden_state (matches
#     production GSFATeamClassifier._embed()), NOT pooler_output as in
#     colab_sam2_player_tracking.py's embed_crops() — that script's embedding
#     recipe was a divergence from production; this one is not.
#
# How to run:
#   - Runtime -> Change runtime type -> GPU (T4 works)
#   - Paste each "CELL n" block below into its own Colab cell, in order
#   - Fill in MODEL_PATH in CELL 2 with your YOLO .pt on Drive, and
#     VIDEO_PATH with your uploaded clip
#
# Outputs (in /content/output/):
#   tracked_output.mp4  — boxes only (no masks), coloured by team, labelled
#                         with the STABLE id (survives long re-id gaps)
#   tracks.csv           — frame, stable_id, raw_track_id, team, bbox, conf
#   reid_log.csv          — every re-id decision made (new track, high-conf
#                         match, VLM tie-break + verdict, OCR veto, gallery
#                         expiry, duplicate merge) — audit trail for merges
#
# Known limitations of this demo script (deliberate, to keep it simple):
#   - Long-term re-id only fires the moment BoT-SORT mints a genuinely new
#     internal id; it never overrides BoT-SORT's own within-buffer decisions.
#   - A lost stable id's gallery is permanently forgotten after
#     LOST_GALLERY_TTL_SECONDS with nothing seen — a later reappearance mints
#     a new stable id, same as a first-time appearance.
#   - The VLM tie-break trusts two still crops; extreme motion blur or a
#     jersey occluded in both crops can still produce a wrong verdict — it is
#     one signal among cosine similarity + OCR, not a ground-truth oracle.
#   - Goalkeepers wear a third kit, so KMeans(2) lumps each GK into whichever
#     team cluster is closer (same known limitation as the SAM2 script).
# =============================================================================


# =============================== CELL 1: setup ===============================
# Installs. No SAM2 checkpoint download needed — BoT-SORT ships via boxmot.

!pip install -q ultralytics boxmot "transformers>=4.45.0" umap-learn accelerate bitsandbytes
!pip install -q easyocr   # jersey-number re-id veto. EasyOCR is torch-based, so it
                          # shares the notebook's existing PyTorch stack (same
                          # reasoning as the SAM2 script).

from google.colab import drive
drive.mount('/content/drive')


# =============================== CELL 2: config ==============================
import os

MODEL_PATH = ""   # <-- REQUIRED: your YOLO .pt on Drive, e.g.
                  #     "/content/drive/MyDrive/aiff_v2.pt" — left empty on
                  #     purpose, fill in before running. Classes are never
                  #     hardcoded: CELL 3 auto-detects player/ball/referee/
                  #     goal_post ids from yolo.names by keyword match.
VIDEO_PATH = "/content/clip.mp4"   # video to track (upload it to /content/)

MAX_FRAMES      = 5400   # hard cap (~3 min @30fps) — Colab session budget guard
PLAYER_CONF     = 0.50   # detection confidence gate (lower to ~0.35-0.40 if CELL 5's
                          # detection-count diagnostic finds fewer players than your roster)
TEAM_FIT_STRIDE = 2       # sample every Nth frame when collecting crops to fit teams
TEAM_VOTE_STRIDE = 2      # re-classify each track every Nth frame, majority vote

CMC_METHOD = "ecc"        # BoT-SORT global motion compensation — mirrors
                          # tracking/player_tracker.py
TRACK_BUFFER_FRAMES_AT_30FPS = 60   # identical name/value to production —
                          # BoT-SORT's OWN re-id horizon (~2s). boxmot scales
                          # this internally by frame_rate/30 (its own formula:
                          # buffer_size = int(frame_rate/30*track_buffer)).

LOST_GALLERY_TTL_SECONDS = 180   # 3 minutes wall-clock — how long a lost stable
                          # id's gallery survives before being permanently
                          # forgotten. Long enough to cover a substitution, an
                          # off-pitch treatment, or several camera cuts; short
                          # enough that two different players who happen to
                          # share a similar build/kit don't stay eligible for
                          # cross-matching for the whole clip.
REID_SIM_HIGH = 0.85     # >= this: accept re-id outright, no VLM call
REID_SIM_LOW  = 0.65     # below this: reject outright, mint a new stable id.
                          # [REID_SIM_LOW, REID_SIM_HIGH) is the ambiguous band
                          # where the VLM tie-break fires.
MAX_GALLERY_SHOTS = 5    # cap on how many crops/embeddings a track's re-id
                          # gallery keeps — a single frozen reference is too
                          # fragile to one bad frame (blur/occlusion/angle),
                          # but unbounded growth is wasted RAM/disk; once
                          # full, a new crop only displaces the gallery's
                          # current weakest (least sharp) entry
OCR_CONF_THRESH = 0.75   # jersey-number OCR reads below this confidence are
                          # treated as illegible — OCR is a re-id veto/
                          # tie-breaker on top of embedding+VLM, never the
                          # sole signal
DUPLICATE_IOU_THRESH = 0.6  # two different stable ids overlapping at least
                          # this much, both seen very recently, are the same
                          # real player — merge them, keeping the older id
MIN_CROP_PX = 32          # reject torso crops smaller than this on a side
                          # (matches production GSFATeamClassifier's MIN_CROP_PX)
BLUR_THRESHOLD = 80       # reject blurry fit/gallery crops below this
                          # Laplacian variance
FIT_SILHOUETTE_MIN = 0.20  # warn if the two team clusters don't separate cleanly

VLM_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"   # natively takes 2 images in one
                          # prompt — a direct comparison in one forward pass,
                          # no composite-image workaround needed. 4-bit here
                          # is deliberately conservative for a 2B model (would
                          # fit fine in fp16 on a T4) so the same call stays
                          # plausible on a 4GB local GPU later.
VLM_MAX_NEW_TOKENS = 8

# No TARGET_FPS resampling (unlike the SAM2 script): every re-id threshold
# below is defined in wall-clock time converted from the SOURCE fps, so
# decoupling frame count from real time would complicate every threshold for
# no compute-budget reason — this pipeline is far cheaper per-frame than
# SAM2's video memory bank was.

FRAMES_DIR = "/content/frames"
OUTPUT_DIR = "/content/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

assert MODEL_PATH, ("Fill in MODEL_PATH with your YOLO .pt path on Drive "
                     "before running — see CELL 2.")
assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"
assert VIDEO_PATH, "Upload your video to the Colab session and paste its path into VIDEO_PATH"
assert os.path.exists(VIDEO_PATH), f"Video not found: {VIDEO_PATH}"


# ====================== CELL 3: imports, device, models ======================
import csv
import re
import shutil
from collections import Counter, defaultdict

import cv2
import easyocr
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    SiglipVisionModel,
)
from ultralytics import YOLO
import umap

from boxmot.trackers.bbox.botsort.botsort import BotSort

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "Enable a GPU runtime (Runtime -> Change runtime type -> GPU)"

# --- probe source fps (needed before BotSort + long-term thresholds) ---
_probe_cap = cv2.VideoCapture(VIDEO_PATH)
SRC_FPS = _probe_cap.get(cv2.CAP_PROP_FPS) or 30.0
_probe_cap.release()

# --- YOLO detector ---
yolo = YOLO(MODEL_PATH)
print("Model classes:", yolo.names)

# Work out which class ids are players / referees / ball / goal posts from
# the class names — never hardcode ids, this model's names may differ from
# production's (active_player/ball/goal_post/referee).
PLAYER_CLASS_IDS    = [i for i, n in yolo.names.items() if "player" in n.lower()]
REFEREE_CLASS_IDS   = [i for i, n in yolo.names.items() if "referee" in n.lower()]
BALL_CLASS_IDS       = [i for i, n in yolo.names.items() if "ball" in n.lower()]
GOAL_POST_CLASS_IDS  = [i for i, n in yolo.names.items() if "goal" in n.lower()]
if not PLAYER_CLASS_IDS:
    PLAYER_CLASS_IDS = [0]  # fallback: assume class 0 is the player class
print(f"Player class ids: {PLAYER_CLASS_IDS} | referee: {REFEREE_CLASS_IDS} | "
      f"ball: {BALL_CLASS_IDS} | goal_post: {GOAL_POST_CLASS_IDS}")

# --- SigLIP embedder (same backbone + recipe as GSFATeamClassifier) ---
SIGLIP_ID = "google/siglip-base-patch16-224"
siglip_processor = AutoProcessor.from_pretrained(SIGLIP_ID)
siglip = SiglipVisionModel.from_pretrained(SIGLIP_ID).to(DEVICE).eval()

# --- BoT-SORT — production's exact config (tracking/player_tracker.py) ---
tracker = BotSort(
    reid_model            = None,
    with_reid             = True,
    cmc_method            = CMC_METHOD,
    track_high_thresh     = 0.5,
    track_low_thresh      = 0.1,
    new_track_thresh      = 0.6,
    match_thresh          = 0.8,
    proximity_thresh      = 0.5,
    appearance_thresh     = 0.25,
    track_buffer          = TRACK_BUFFER_FRAMES_AT_30FPS,
    frame_rate            = int(round(SRC_FPS)),
    fuse_first_associate  = False,
)

# Same formula boxmot uses internally for max_time_lost — the long-term re-id
# layer must never race BoT-SORT's own within-buffer recovery.
LONG_TERM_GAP_FRAMES = int(SRC_FPS / 30.0 * TRACK_BUFFER_FRAMES_AT_30FPS)
LOST_GALLERY_TTL_FRAMES = int(round(SRC_FPS * LOST_GALLERY_TTL_SECONDS))
print(f"Source fps: {SRC_FPS:.1f} | BoT-SORT own buffer: {LONG_TERM_GAP_FRAMES} frames "
      f"| long-term gallery TTL: {LOST_GALLERY_TTL_FRAMES} frames")

# --- EasyOCR (jersey-number re-id veto, secondary signal only) ---
# CPU on purpose: fires only on gallery-admitted crops (a handful per track,
# not per-frame), so throughput isn't the bottleneck, and keeping it off the
# GPU avoids competing with SigLIP/the VLM for VRAM.
ocr_reader = easyocr.Reader(["en"], gpu=False)

# --- Qwen2-VL (re-id tie-breaker, ambiguous similarity band only) ---
_vlm_bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
vlm = Qwen2VLForConditionalGeneration.from_pretrained(
    VLM_MODEL_ID, quantization_config=_vlm_bnb_cfg, device_map="cuda:0"
)
vlm.eval()


def detect_players(frame_bgr):
    """Run YOLO on one frame, return player boxes+conf as float32 (N, 5) xyxy+conf."""
    res = yolo.predict(frame_bgr, conf=PLAYER_CONF, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.empty((0, 5), dtype=np.float32)
    cls = res.boxes.cls.cpu().numpy().astype(int)
    keep = np.isin(cls, PLAYER_CLASS_IDS)
    xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)[keep]
    conf = res.boxes.conf.cpu().numpy().astype(np.float32)[keep]
    return np.concatenate([xyxy, conf[:, None]], axis=1)


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


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


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
    why OCR is only ever used as a veto on top of embedding similarity (and the
    VLM tie-break), never the sole re-id signal."""
    h, w = crop_bgr.shape[:2]
    big = cv2.resize(crop_bgr, (w * upscale, h * upscale), interpolation=cv2.INTER_CUBIC)
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
    """SigLIP-embed a list of BGR crops -> (N, 768) numpy. Mean-pools
    last_hidden_state — the PRODUCTION recipe (GSFATeamClassifier._embed()),
    not SigLIP's pooler_output (which colab_sam2_player_tracking.py used —
    a divergence from production, not repeated here)."""
    out = []
    for i in range(0, len(crops_bgr), batch_size):
        pil = [
            Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
            for c in crops_bgr[i : i + batch_size]
        ]
        inputs = siglip_processor(images=pil, return_tensors="pt").to(DEVICE)
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = siglip(**inputs)
        emb = torch.mean(outputs.last_hidden_state, dim=1).float().cpu().numpy()
        out.append(emb)
    return np.concatenate(out, axis=0)


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def vlm_same_player(crop_a_bgr, crop_b_bgr):
    """Ask the VLM whether two torso crops are the same player. Returns
    (verdict: bool, raw_text: str) for logging. Defaults to False (no match)
    on any unparseable response — a missed re-id costs one extra id, a wrong
    merge corrupts two players' data, so an ambiguous VLM answer must not
    confirm a match."""
    img_a = Image.fromarray(cv2.cvtColor(crop_a_bgr, cv2.COLOR_BGR2RGB))
    img_b = Image.fromarray(cv2.cvtColor(crop_b_bgr, cv2.COLOR_BGR2RGB))
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img_a},
        {"type": "image", "image": img_b},
        {"type": "text", "text": (
            "Image 1 and Image 2 are cropped torso/jersey photos of a soccer "
            "player from broadcast video. Based on jersey colour, pattern, and "
            "any visible number, are these the same individual player? "
            "Answer with exactly one word: yes or no.")},
    ]}]
    text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = vlm_processor(text=[text], images=[img_a, img_b], return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out_ids = vlm.generate(**inputs, max_new_tokens=VLM_MAX_NEW_TOKENS)
    reply = vlm_processor.batch_decode(
        out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip().lower()
    return reply.startswith("yes"), reply


# ========================= CELL 4: extract frames ============================
# Native source fps, no resampling — the whole re-id state machine's
# thresholds are wall-clock-derived from SRC_FPS, so decoupling frame count
# from real time would complicate every threshold for no benefit here (this
# pipeline is far cheaper per-frame than SAM2's video memory bank was).

shutil.rmtree(FRAMES_DIR, ignore_errors=True)
os.makedirs(FRAMES_DIR)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

n_frames = 0
while n_frames < MAX_FRAMES:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imwrite(os.path.join(FRAMES_DIR, f"{n_frames:05d}.jpg"), frame)
    n_frames += 1
cap.release()

print(f"{SRC_FPS:.1f} fps source -> {n_frames} frames ({frame_w}x{frame_h})")


def read_frame(idx):
    return cv2.imread(os.path.join(FRAMES_DIR, f"{idx:05d}.jpg"))


# ================= CELL 5: detection pass (stage 1 of 4) =====================
# (a) Diagnostic: print per-frame detection counts for the first ~2s so a bad
#     PLAYER_CONF threshold (or a class-id mismatch) is obvious immediately.
# (b) Collect torso crops across the whole clip to fit the team classifier.

probe_frames = min(int(SRC_FPS * 2), n_frames)  # ~2s
probe_counts = [len(detect_players(read_frame(i))) for i in range(probe_frames)]
print(f"Detections per frame (first {probe_frames} frames): {probe_counts}")
print(f"  min={min(probe_counts)} max={max(probe_counts)} "
      f"avg={sum(probe_counts) / len(probe_counts):.1f}")
if max(probe_counts) < 6:
    print("  WARNING: even the best of these frames found few players — "
          "if this is well below your actual roster size, try lowering "
          f"PLAYER_CONF (currently {PLAYER_CONF}) in CELL 2 and re-running "
          "from CELL 3.")

fit_crops = []
for idx in range(0, n_frames, TEAM_FIT_STRIDE):
    frame = read_frame(idx)
    for box in detect_players(frame)[:, :4]:
        crop = torso_crop(frame, box, min_px=MIN_CROP_PX)
        if crop is not None and is_sharp(crop):
            fit_crops.append(crop)
print(f"Collected {len(fit_crops)} torso crops for team fitting "
      f"(after {MIN_CROP_PX}px size + blur filtering)")


# ============== CELL 6: team classifier fit (stage 2 of 4) ===================
# SigLIP -> UMAP(3d) -> KMeans(2): the GSFATeamClassifier recipe.

from sklearn.metrics import silhouette_score

fit_embeddings = embed_crops(fit_crops)
del fit_crops  # thousands of full-res crops held live for no reason after this

reducer = umap.UMAP(n_components=3, random_state=42)
projected = reducer.fit_transform(fit_embeddings)
del fit_embeddings  # only the projection is needed from here on

team_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(projected)
print("Team clusters fitted:", np.bincount(team_kmeans.labels_))

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


# ========== CELL 7: BoT-SORT tracking + long-term re-id (stage 3+4 of 4) =====
# Every frame: detect -> embed -> tracker.update() -> split rows into
# already-mapped (raw_id already resolved to a stable_id) vs. brand-new raw
# ids -> resolve brand-new ids through the long-term re-id state machine
# (gallery cosine similarity -> VLM tie-break on ambiguous scores -> OCR veto)
# -> duplicate-merge -> per-frame gallery-TTL expiry. Every downstream
# artifact (video labels, tracks.csv, galleries, OCR reads, team votes) is
# keyed by stable_id, never BoT-SORT's own raw_id.

raw_to_stable = {}          # BoT-SORT raw id -> our stable id (only place the
                             # two id spaces meet)
stable_next_id = 0
stable_last_seen = {}        # stable_id -> global frame idx last actually visible
stable_last_box = {}         # stable_id -> most recent visible box
gallery = defaultdict(list)  # stable_id -> [(embedding, sharpness), ...],
                             # capped at MAX_GALLERY_SHOTS, weakest evicted first
track_ocr = {}               # stable_id -> (digits, confidence) — single most
                             # confident jersey-number read across the gallery
merged_away = set()          # stable ids permanently absorbed into another

votes = defaultdict(Counter)  # stable_id -> team-vote counter
track_boxes = defaultdict(dict)  # frame_idx -> {stable_id: (x1,y1,x2,y2,conf)}

reid_log = []  # list of dicts -> reid_log.csv

_vote_buffer_crops, _vote_buffer_keys = [], []


def _gallery_crop_path(stable_id, slot):
    return os.path.join(OUTPUT_DIR, "crops", f"track_{stable_id:03d}_{slot}.jpg")


os.makedirs(os.path.join(OUTPUT_DIR, "crops"), exist_ok=True)


def _admit_to_gallery(stable_id, emb, crop):
    """Add (emb, crop) to stable_id's capped re-id gallery, if it's sharp/large
    enough to trust as a reference — displacing the gallery's current weakest
    entry once MAX_GALLERY_SHOTS is reached, rather than growing forever."""
    h, w = crop.shape[:2]
    if min(h, w) < MIN_CROP_PX:
        return
    score = sharpness_score(crop)
    if score < BLUR_THRESHOLD:
        return
    g = gallery[stable_id]
    if len(g) < MAX_GALLERY_SHOTS:
        slot = len(g)
        g.append((emb, score))
    else:
        slot = min(range(len(g)), key=lambda i: g[i][1])
        if score <= g[slot][1]:
            return  # not better than what's already kept in this slot
        g[slot] = (emb, score)
    cv2.imwrite(_gallery_crop_path(stable_id, slot), crop)

    digits, conf = ocr_jersey_number(crop)
    if digits is not None and conf > track_ocr.get(stable_id, (None, 0.0))[1]:
        track_ocr[stable_id] = (digits, conf)


def _best_gallery_crop_path(stable_id):
    g = gallery.get(stable_id)
    if not g:
        return None
    slot = max(range(len(g)), key=lambda i: g[i][1])
    path = _gallery_crop_path(stable_id, slot)
    return path if os.path.exists(path) else None


def flush_votes():
    global _vote_buffer_crops, _vote_buffer_keys
    if not _vote_buffer_crops:
        return
    embeddings = embed_crops(_vote_buffer_crops)
    teams = team_kmeans.predict(reducer.transform(embeddings))
    for stable_id, emb, team, crop in zip(_vote_buffer_keys, embeddings, teams, _vote_buffer_crops):
        votes[stable_id][int(team)] += 1
        _admit_to_gallery(stable_id, emb, crop)
    _vote_buffer_crops, _vote_buffer_keys = [], []


def _log(frame_idx, event, **fields):
    reid_log.append({"frame": frame_idx, "event": event, **fields})


def _lost_candidates(frame_idx):
    """Stable ids that BoT-SORT itself has given up on (past its own ~2s
    buffer) but whose gallery hasn't expired yet — eligible for long-term
    re-id. Excludes anything currently active or already merged away."""
    return [
        sid for sid, last in stable_last_seen.items()
        if sid not in merged_away
        and gallery.get(sid)
        and LONG_TERM_GAP_FRAMES < (frame_idx - last) <= LOST_GALLERY_TTL_FRAMES
    ]


def resolve_new_raw_ids(frame_idx, new_raw_ids, raw_id_to_crop_emb):
    """Batched resolution: compute each brand-new raw id's best lost-gallery
    candidate + similarity first, then assign in descending-similarity order
    so two new arrivals in the same frame can't both claim the same lost id."""
    global stable_next_id
    proposals = []  # (similarity, raw_id, candidate_stable_id, crop, emb)
    lost_ids = _lost_candidates(frame_idx)

    for raw_id in new_raw_ids:
        crop, emb = raw_id_to_crop_emb[raw_id]
        if crop is None or not lost_ids:
            proposals.append((-1.0, raw_id, None, crop, emb))
            continue
        best_sim, best_sid = -1.0, None
        for sid in lost_ids:
            sim = max(cosine_sim(emb, g_emb) for g_emb, _ in gallery[sid])
            if sim > best_sim:
                best_sim, best_sid = sim, sid
        proposals.append((best_sim, raw_id, best_sid, crop, emb))

    proposals.sort(key=lambda p: p[0], reverse=True)
    claimed = set()

    for sim, raw_id, cand_sid, crop, emb in proposals:
        matched_sid = None

        if cand_sid is not None and cand_sid not in claimed:
            if sim >= REID_SIM_HIGH:
                matched_sid = cand_sid
                _log(frame_idx, "reid_high_conf", raw_id=raw_id,
                     matched_stable_id=cand_sid, similarity=sim)
            elif sim >= REID_SIM_LOW:
                ref_path = _best_gallery_crop_path(cand_sid)
                ref_crop = cv2.imread(ref_path) if ref_path else None
                if ref_crop is not None:
                    verdict, raw_text = vlm_same_player(crop, ref_crop)
                    if verdict:
                        matched_sid = cand_sid
                    _log(frame_idx, "reid_vlm_checked", raw_id=raw_id,
                         candidate_stable_id=cand_sid, similarity=sim,
                         vlm_used=True, vlm_verdict=verdict, vlm_raw_text=raw_text)
                else:
                    _log(frame_idx, "reid_vlm_skipped_no_ref", raw_id=raw_id,
                         candidate_stable_id=cand_sid, similarity=sim)
            else:
                _log(frame_idx, "reid_below_threshold", raw_id=raw_id,
                     candidate_stable_id=cand_sid, similarity=sim)

        # Jersey-number OCR veto — applied regardless of how the match was
        # reached, on top of embedding similarity AND the VLM verdict. Only
        # overrides when BOTH sides read a confident, disagreeing number.
        if matched_sid is not None:
            cand_digits, _ = ocr_jersey_number(crop) if crop is not None else (None, 0.0)
            stored_digits, _ = track_ocr.get(matched_sid, (None, 0.0))
            if (cand_digits is not None and stored_digits is not None
                    and cand_digits != stored_digits):
                _log(frame_idx, "ocr_veto", raw_id=raw_id,
                     candidate_stable_id=matched_sid,
                     ocr_candidate_digits=cand_digits, ocr_stored_digits=stored_digits)
                matched_sid = None

        if matched_sid is not None:
            raw_to_stable[raw_id] = matched_sid
            claimed.add(matched_sid)
            print(f"Frame {frame_idx}: re-identified stable id {matched_sid} "
                  f"(raw {raw_id}, similarity {sim:.2f}) instead of a new id")
        else:
            matched_sid = stable_next_id
            stable_next_id += 1
            raw_to_stable[raw_id] = matched_sid
            _log(frame_idx, "new_track", raw_id=raw_id, matched_stable_id=matched_sid)

        stable_last_seen[matched_sid] = frame_idx
        if crop is not None and emb is not None:
            _admit_to_gallery(matched_sid, emb, crop)


def merge_duplicate_tracks(frame_idx, active_sids):
    """If two different stable ids are sitting almost exactly on top of each
    other right now, they're the same real player — keep the older id,
    permanently drop the newer one (absorb its gallery/OCR/votes)."""
    candidates = [sid for sid in active_sids if sid not in merged_away]
    for i, a in enumerate(candidates):
        if a in merged_away:
            continue
        for b in candidates[i + 1:]:
            if b in merged_away:
                continue
            if a not in stable_last_box or b not in stable_last_box:
                continue
            if iou(stable_last_box[a], stable_last_box[b]) >= DUPLICATE_IOU_THRESH:
                keep_id, drop_id = min(a, b), max(a, b)
                gallery[keep_id].extend(gallery[drop_id])
                gallery[keep_id] = sorted(gallery[keep_id], key=lambda g: g[1], reverse=True)[:MAX_GALLERY_SHOTS]
                votes[keep_id].update(votes[drop_id])
                del votes[drop_id]
                if drop_id in track_ocr and track_ocr[drop_id][1] > track_ocr.get(keep_id, (None, 0.0))[1]:
                    track_ocr[keep_id] = track_ocr[drop_id]

                # Redirect every raw BoT-SORT id currently pointing at drop_id
                # onto keep_id — without this, BoT-SORT would keep emitting
                # rows for that raw id every subsequent frame (it has no idea
                # we merged anything), and those rows would silently vanish
                # from the output the instant drop_id lands in merged_away,
                # instead of continuing on-screen under the surviving id.
                for raw_id, sid in list(raw_to_stable.items()):
                    if sid == drop_id:
                        raw_to_stable[raw_id] = keep_id

                gallery.pop(drop_id, None)
                stable_last_seen.pop(drop_id, None)
                stable_last_box.pop(drop_id, None)
                track_ocr.pop(drop_id, None)

                merged_away.add(drop_id)
                active_sids.discard(drop_id)
                _log(frame_idx, "duplicate_merge", matched_stable_id=keep_id,
                     candidate_stable_id=drop_id)
                print(f"Frame {frame_idx}: merged duplicate stable id {drop_id} into {keep_id}")


def expire_stale_galleries(frame_idx):
    """Permanently forget any stable id whose gallery has outlived
    LOST_GALLERY_TTL_FRAMES with nothing seen — a later reappearance is
    treated as a brand-new player, same cost as a first-ever appearance."""
    stale = [
        sid for sid, last in stable_last_seen.items()
        if sid not in merged_away and (frame_idx - last) > LOST_GALLERY_TTL_FRAMES
    ]
    for sid in stale:
        del stable_last_seen[sid]
        gallery.pop(sid, None)
        track_ocr.pop(sid, None)
        stable_last_box.pop(sid, None)
        _log(frame_idx, "gallery_expired", matched_stable_id=sid)


active_sids = set()

for frame_idx in range(n_frames):
    frame = read_frame(frame_idx)
    dets = detect_players(frame)  # (N, 5) xyxy+conf

    crops = [torso_crop(frame, box) for box in dets[:, :4]]
    valid_crops = [c for c in crops if c is not None]
    embs_valid = embed_crops(valid_crops) if valid_crops else np.empty((0, 768), dtype=np.float32)
    emb_iter = iter(embs_valid)
    embs = np.zeros((len(dets), 768), dtype=np.float32)
    for i, c in enumerate(crops):
        if c is not None:
            embs[i] = next(emb_iter)

    if len(dets) == 0:
        out = tracker.update(np.empty((0, 6), dtype=np.float32), frame,
                              embs=np.empty((0, 1), dtype=np.float32))
    else:
        dets6 = np.concatenate([dets, np.zeros((len(dets), 1), dtype=np.float32)], axis=1)  # + cls=0
        out = tracker.update(dets6, frame, embs=embs)
    out_arr = np.asarray(out)

    new_raw_ids = []
    raw_id_to_crop_emb = {}

    if out_arr.size:
        for row in out_arr:
            raw_id = int(row[4])
            det_ind = int(row[7])
            crop = crops[det_ind] if 0 <= det_ind < len(crops) else None
            emb = embs[det_ind] if 0 <= det_ind < len(embs) else None
            if raw_id not in raw_to_stable:
                new_raw_ids.append(raw_id)
                raw_id_to_crop_emb[raw_id] = (crop, emb)

    if new_raw_ids:
        resolve_new_raw_ids(frame_idx, new_raw_ids, raw_id_to_crop_emb)

    active_sids = set()
    if out_arr.size:
        for row in out_arr:
            raw_id = int(row[4])
            det_ind = int(row[7])
            sid = raw_to_stable[raw_id]
            if sid in merged_away:
                continue
            box = tuple(float(v) for v in row[0:4])
            conf = float(row[5])
            stable_last_seen[sid] = frame_idx
            stable_last_box[sid] = box
            track_boxes[frame_idx][sid] = box + (conf,)
            active_sids.add(sid)

            crop = crops[det_ind] if 0 <= det_ind < len(crops) else None
            if crop is not None and frame_idx % TEAM_VOTE_STRIDE == 0:
                _vote_buffer_crops.append(crop)
                _vote_buffer_keys.append(sid)
                if len(_vote_buffer_crops) >= 32:
                    flush_votes()

    merge_duplicate_tracks(frame_idx, active_sids)
    if frame_idx % 30 == 0:
        expire_stale_galleries(frame_idx)

flush_votes()
expire_stale_galleries(n_frames)

track_team = {sid: c.most_common(1)[0][0] for sid, c in votes.items()}
final_track_count = stable_next_id - len(merged_away)
print(f"Tracked {final_track_count} players across {len(track_boxes)} frames "
      f"({len(merged_away)} duplicate id(s) merged away)")
if final_track_count < 6:
    print("  WARNING: still tracking very few players — re-check the "
          "detection-count diagnostic in CELL 5 and consider lowering "
          "PLAYER_CONF in CELL 2.")


# ==================== CELL 8: render output + CSV exports ====================
TEAM_COLORS = {0: (60, 200, 255), 1: (255, 120, 60)}  # BGR: amber vs blue-ish

writer = cv2.VideoWriter(
    os.path.join(OUTPUT_DIR, "tracked_output.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"),
    SRC_FPS,
    (frame_w, frame_h),
)

tracks_csv_path = os.path.join(OUTPUT_DIR, "tracks.csv")
with open(tracks_csv_path, "w", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["frame", "stable_id", "team", "x1", "y1", "x2", "y2", "conf"])

    for frame_idx in range(n_frames):
        frame = read_frame(frame_idx)
        for sid, (x1, y1, x2, y2, conf) in track_boxes.get(frame_idx, {}).items():
            if sid in merged_away:
                continue
            team = track_team.get(sid, 0)
            color = TEAM_COLORS[team]
            csv_writer.writerow([frame_idx, sid, team, x1, y1, x2, y2, conf])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"#{sid}"
            cv2.putText(frame, label, (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(frame, label, (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"frame {frame_idx}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)

writer.release()

reid_log_path = os.path.join(OUTPUT_DIR, "reid_log.csv")
_reid_fields = ["frame", "event", "raw_id", "matched_stable_id", "candidate_stable_id",
                "similarity", "vlm_used", "vlm_verdict", "vlm_raw_text",
                "ocr_candidate_digits", "ocr_stored_digits"]
with open(reid_log_path, "w", newline="") as f:
    csv_writer = csv.DictWriter(f, fieldnames=_reid_fields, extrasaction="ignore")
    csv_writer.writeheader()
    for row in reid_log:
        csv_writer.writerow(row)

print(f"Wrote {OUTPUT_DIR}/tracked_output.mp4, {tracks_csv_path}, and {reid_log_path}")


# ======================= CELL 9: preview / download ==========================
# Re-encode to h264 so it plays inline / downloads small, then download all outputs.

!ffmpeg -y -loglevel error -i /content/output/tracked_output.mp4 \
    -vcodec libx264 -crf 24 /content/output/tracked_output_h264.mp4

from google.colab import files
files.download("/content/output/tracked_output_h264.mp4")
files.download("/content/output/tracks.csv")
files.download("/content/output/reid_log.csv")
