# Script 2: Player Detection + Team Classification (HSV Color Histogram + KMeans)
# -------------------------------------------------------------------------------
# Detects players with YOLOv11m, extracts HSV color histograms from the TORSO
# region of each bounding box (NOT the full box), and clusters into 2 teams with
# KMeans. This fixes the SigLIP+UMAP approach which encoded background context
# (bench seats, court surface) rather than jersey color, causing spatial clusters
# (on-court vs bench) instead of color clusters (team A vs team B).
#
# Why torso-only HSV instead of SigLIP+UMAP:
#   - SigLIP embeds the ENTIRE crop including background -> bench players cluster
#     differently because the bleacher background dominates over jersey color.
#   - HSV histograms on the torso ROI (upper-middle of bbox) capture jersey color
#     and almost nothing else: no court, no bench, minimal skin contamination.
#   - Works on a panning camera with no fixed court reference.
#   - 100x faster than SigLIP forward passes: pure NumPy, no GPU required.
#
# Torso ROI definition (configurable via constants below):
#   - Vertical: rows [bbox_h * TORSO_TOP_FRAC .. bbox_h * TORSO_BOT_FRAC]
#   - Horizontal: cols [bbox_w * TORSO_LR_FRAC .. bbox_w * (1 - TORSO_LR_FRAC)]
#
# Required packages:
#   pip install supervision ultralytics scikit-learn opencv-python numpy tqdm
#
# Usage:
#   python scripts/02_team_classification.py
#   python scripts/02_team_classification.py --input data/videos/match.mp4 \
#       --output data/videos/team_classification_output.mp4
#   python scripts/02_team_classification.py --input data/videos/match.mp4 \
#       --output data/videos/out.mp4 --conf 0.35 --stride 30 --kmeans-init 20
#
# Known limitations:
#   - Referees (third jersey color) will be assigned to one of the two teams.
#     Extend to k=3 clusters and post-filter if referee separation is needed.
#   - Goalkeepers often wear a unique color; if GK kit is very different from
#     outfield kit, they may cluster with the opposing team. Add a GK-specific
#     class filter or override if necessary.
#   - Very dark or white/grey jerseys with low saturation may cluster poorly in
#     HSV Hue space; the pipeline falls back to Value+Saturation channels which
#     helps, but may not fully resolve monochrome jersey pairs.
#   - Camera zoom changes alter the torso crop size but not its color content,
#     so classification remains stable across zoom levels.

import argparse
import sys
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Torso ROI fractions within each bounding box
# These isolate the jersey region and avoid head/skin (top) and shorts/legs (bottom)
TORSO_TOP_FRAC = 0.25   # start 25% down from the top of the box (skip head)
TORSO_BOT_FRAC = 0.65   # end 65% down (stop before shorts/legs)
TORSO_LR_FRAC  = 0.15   # trim 15% from each side (avoid arm/background edges)

# HSV histogram parameters
H_BINS = 36              # hue bins (360 degrees / 36 = 10 deg per bin)
S_BINS = 32              # saturation bins
V_BINS = 16              # value bins — less discriminative but helps for dark kits
HIST_NORMALIZE = True    # L1-normalise so feature is scale-invariant

# KMeans
KMEANS_CLUSTERS = 3
KMEANS_NINIT    = 20     # more inits = more stable for color-based clustering
KMEANS_MAXITER  = 500
KMEANS_SEED     = 42

# Referee identification
# Hue ~40° (OpenCV scale 0-180) = yellow — confirmed from diagnostic output.
# Any cluster whose dominant hue bin falls within ±2 bins of this is the referee cluster.
REFEREE_HUE_DEG   = 40.0
REFEREE_HUE_TOL   = 20.0   # ±20° tolerance around the referee hue

# Bench-exclusion zone: only apply the sideline-cutoff logic when a referee's
# bounding-box centre is in the BOTTOM fraction of the frame.
# Referees on the upper/mid court are ignored for this filter.
REFEREE_BOTTOM_ZONE_FRAC = 0.60   # referee centre-y > 60% of frame height = bottom zone

# Minimum crop area (pixels) to be accepted as a valid detection
MIN_CROP_AREA = 400      # ~20x20px; smaller detections are too noisy

# YOLO
YOLO_CLASSES = [0]       # COCO class 0 = person

# Frame sampling stride for the fit phase (frames between samples)
DEFAULT_STRIDE = 30      # ~1fps at 30fps video

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Player detection + HSV-torso team classification"
)
parser.add_argument("--input",  default="data/videos/test.mp4",
                    help="Path to input video file")
parser.add_argument("--output", default="data/videos/team_classification_output.mp4",
                    help="Path to output annotated video")
parser.add_argument("--model",  default="yolo11m.pt",
                    help="YOLO model weights file")
parser.add_argument("--device", default="cuda",
                    help="Inference device: 'cuda' or 'cpu'")
parser.add_argument("--conf",   type=float, default=0.3,
                    help="YOLO detection confidence threshold")
parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                    help="Frame stride for fit-phase sampling (default: 30)")
parser.add_argument("--kmeans-init", type=int, default=KMEANS_NINIT,
                    help="Number of KMeans re-initialisations (default: 20)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

try:
    import supervision as sv
except ImportError:
    print("[ERROR] supervision not installed.  Run:  pip install supervision")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics not installed.  Run:  pip install ultralytics")
    sys.exit(1)

try:
    from sklearn.cluster import KMeans
except ImportError:
    print("[ERROR] scikit-learn not installed.  Run:  pip install scikit-learn")
    sys.exit(1)

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

# ---------------------------------------------------------------------------
# CUDA check
# ---------------------------------------------------------------------------

device = args.device
try:
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available — falling back to CPU.")
        device = "cpu"
except ImportError:
    # torch is only needed for YOLO inference; ultralytics bundles it
    device = "cpu"

# ---------------------------------------------------------------------------
# Input video check
# ---------------------------------------------------------------------------

input_path = Path(args.input)
if not input_path.exists():
    print(f"[ERROR] Input video not found: {args.input}")
    print("        Place your clip at that path and re-run.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Core helper: torso crop extraction
# ---------------------------------------------------------------------------

def extract_torso_crop(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """
    Crop the torso region from a full player bounding box.

    This is the critical fix over the SigLIP approach: instead of passing the
    entire bounding box (which includes court/bench background), we extract only
    the jersey-bearing torso region. This means color features reflect jersey
    color and are not contaminated by whatever is behind the player.

    Args:
        frame: Full video frame in BGR.
        xyxy:  [x1, y1, x2, y2] bounding box coordinates (float or int).

    Returns:
        BGR crop of the torso region, or empty array if the box is too small.
    """
    x1, y1, x2, y2 = map(int, xyxy)
    # Clamp to frame bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    bw = x2 - x1
    bh = y2 - y1
    if bw < 4 or bh < 4:
        return np.empty((0, 0, 3), dtype=np.uint8)

    # Compute torso sub-region within the bbox
    ty1 = y1 + int(bh * TORSO_TOP_FRAC)
    ty2 = y1 + int(bh * TORSO_BOT_FRAC)
    tx1 = x1 + int(bw * TORSO_LR_FRAC)
    tx2 = x2 - int(bw * TORSO_LR_FRAC)

    if ty2 <= ty1 or tx2 <= tx1:
        return np.empty((0, 0, 3), dtype=np.uint8)

    return frame[ty1:ty2, tx1:tx2]


# ---------------------------------------------------------------------------
# Core helper: HSV histogram feature extraction
# ---------------------------------------------------------------------------

def compute_hsv_histogram(crop: np.ndarray) -> np.ndarray:
    """
    Compute a normalised HSV histogram for a BGR crop.

    We use all three HSV channels concatenated into a single feature vector.
    Hue is weighted highest (more bins) because jersey hue is the primary
    discriminator between teams. Saturation and Value provide fallback
    discrimination for dark or low-saturation jerseys.

    Args:
        crop: BGR image crop (the torso region).

    Returns:
        1-D float32 feature vector of length (H_BINS + S_BINS + V_BINS).
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], None, [H_BINS], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [S_BINS], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [V_BINS], [0, 256]).flatten()

    feature = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)

    if HIST_NORMALIZE and feature.sum() > 0:
        feature /= feature.sum()   # L1 normalise

    return feature


# ---------------------------------------------------------------------------
# TeamClassifier: mirrors the roboflow sports/common/team.py interface
# (fit / predict) but uses torso HSV histograms instead of SigLIP embeddings
# ---------------------------------------------------------------------------

class TeamClassifier:
    """
    Classifies player detections into two teams by clustering HSV color
    histograms extracted from the torso region of each bounding box.

    Interface is intentionally compatible with the roboflow sports TeamClassifier
    so it can be swapped in wherever that class is used:

        classifier = TeamClassifier(device=device)
        classifier.fit(crops)
        team_ids = classifier.predict(crops)

    The 'device' argument is accepted for interface compatibility but ignored —
    all computation is CPU-based NumPy/OpenCV, which is fast enough.

    Args:
        device:      Accepted for API compatibility; unused.
        batch_size:  Accepted for API compatibility; unused.
        n_clusters:  Number of team clusters (default 2).
        n_init:      KMeans re-initialisations (default KMEANS_NINIT).
    """

    def __init__(
        self,
        device: str = "cpu",
        batch_size: int = 32,
        n_clusters: int = KMEANS_CLUSTERS,
        n_init: int = KMEANS_NINIT,
    ) -> None:
        self.device = device           # stored but unused
        self.batch_size = batch_size   # stored but unused
        self.n_clusters = n_clusters
        self.cluster_model = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=KMEANS_MAXITER,
            random_state=KMEANS_SEED,
        )
        self._fitted = False

    # ------------------------------------------------------------------
    # Feature extraction (public so callers can inspect features)
    # ------------------------------------------------------------------

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract torso HSV histogram features from a list of full player crops.

        Each crop should be the FULL player bounding box as returned by
        sv.crop_image(). The torso sub-region is extracted internally here.

        Args:
            crops: List of BGR numpy arrays (full player bounding box crops).

        Returns:
            Float32 array of shape (N, H_BINS + S_BINS + V_BINS).
        """
        features = []
        for crop in crops:
            if crop is None or crop.size < MIN_CROP_AREA:
                # Pad with zeros so index alignment is preserved
                features.append(np.zeros(H_BINS + S_BINS + V_BINS, dtype=np.float32))
                continue

            h, w = crop.shape[:2]
            # Apply torso fractions directly to the crop (crop coords start at 0)
            ty1 = int(h * TORSO_TOP_FRAC)
            ty2 = int(h * TORSO_BOT_FRAC)
            tx1 = int(w * TORSO_LR_FRAC)
            tx2 = w - int(w * TORSO_LR_FRAC)

            torso = crop[ty1:ty2, tx1:tx2] if (ty2 > ty1 and tx2 > tx1) else crop

            if torso.size < MIN_CROP_AREA:
                # Fallback: use full crop if torso slice is degenerate
                torso = crop

            features.append(compute_hsv_histogram(torso))

        return np.array(features, dtype=np.float32)

    # ------------------------------------------------------------------
    # fit / predict — mirrors roboflow sports/common/team.py interface
    # ------------------------------------------------------------------

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit KMeans on HSV histogram features extracted from the provided crops.

        Call this once on a representative sample of player crops from the video
        (typically ~1fps sampling across the full match).

        Args:
            crops: List of BGR numpy arrays (full player bounding box crops).
        """
        print(f"[INFO] Extracting HSV torso features from {len(crops)} crops...")
        features = self.extract_features(crops)
        print(f"[INFO] Feature matrix: {features.shape}  "
              f"(each vector = {H_BINS}H + {S_BINS}S + {V_BINS}V bins)")

        print(f"[INFO] Fitting KMeans (k={self.n_clusters}, n_init={self.cluster_model.n_init})...")
        self.cluster_model.fit(features)
        self._fitted = True

        # Log cluster centroids and identify the referee cluster by hue proximity
        centroids = self.cluster_model.cluster_centers_
        self.referee_cluster_id = -1
        best_dist = float("inf")
        for i, centroid in enumerate(centroids):
            dominant_hue_bin = int(np.argmax(centroid[:H_BINS]))
            dominant_hue_deg = dominant_hue_bin * (180.0 / H_BINS)
            dist = abs(dominant_hue_deg - REFEREE_HUE_DEG)
            print(f"  Cluster {i}: dominant hue bin={dominant_hue_bin} "
                  f"(~{dominant_hue_deg:.0f}° OpenCV HSV)")
            if dist < best_dist and dist <= REFEREE_HUE_TOL:
                best_dist = dist
                self.referee_cluster_id = i
        if self.referee_cluster_id >= 0:
            ref_hue = int(np.argmax(centroids[self.referee_cluster_id, :H_BINS])) * (180.0 / H_BINS)
            print(f"[INFO] Referee cluster identified: cluster {self.referee_cluster_id} "
                  f"(hue ≈ {ref_hue:.0f}°, tolerance ±{REFEREE_HUE_TOL}°)")
        else:
            print(f"[WARNING] No cluster matched referee hue {REFEREE_HUE_DEG}° ±{REFEREE_HUE_TOL}°. "
                  f"Referee will be identified by lowest per-frame count instead.")

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict team cluster labels for a list of player crops.

        Returns integer array of shape (N,) with raw KMeans cluster IDs (0..k-1).
        """
        if not self._fitted:
            raise RuntimeError("TeamClassifier.predict() called before fit(). "
                               "Call fit() first with training crops.")
        if len(crops) == 0:
            return np.array([], dtype=int)

        features = self.extract_features(crops)
        return self.cluster_model.predict(features)


# ---------------------------------------------------------------------------
# Frame-level crop collector (fit phase)
# ---------------------------------------------------------------------------

def collect_player_crops(
    video_path: str,
    yolo_model: "YOLO",
    conf: float,
    stride: int,
) -> List[np.ndarray]:
    """
    Sample frames at the given stride, detect players, and return full
    bounding-box crops. The TeamClassifier will internally extract torso
    sub-regions during feature extraction.

    Args:
        video_path: Path to input video.
        yolo_model: Loaded YOLO model instance.
        conf:       Detection confidence threshold.
        stride:     Frame interval between samples.

    Returns:
        List of BGR crop arrays (one per detected player per sampled frame).
    """
    frame_gen = sv.get_video_frames_generator(source_path=video_path, stride=stride)
    crops = []

    desc = f"Collecting crops (stride={stride})"
    iterator = tqdm(frame_gen, desc=desc) if _TQDM_AVAILABLE else frame_gen

    for frame in iterator:
        results = yolo_model(
            frame, conf=conf, classes=YOLO_CLASSES,
            device=device, verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(results)

        for xyxy in detections.xyxy:
            crop = sv.crop_image(frame, xyxy)
            if crop is not None and crop.size >= MIN_CROP_AREA:
                crops.append(crop)

    return crops


# ---------------------------------------------------------------------------
# Load YOLO model
# ---------------------------------------------------------------------------

print(f"[INFO] Loading YOLO: {args.model}")
yolo_model = YOLO(args.model)

# ---------------------------------------------------------------------------
# FIT PHASE — collect crops and fit the classifier
# ---------------------------------------------------------------------------

print(f"\n[INFO] === FIT PHASE: sampling 1 frame every {args.stride} frames ===")
crops = collect_player_crops(args.input, yolo_model, args.conf, args.stride)
print(f"[INFO] Collected {len(crops)} player crops.")

MIN_CROPS_FOR_FIT = KMEANS_CLUSTERS * 10
if len(crops) < MIN_CROPS_FOR_FIT:
    print(f"[ERROR] Need at least {MIN_CROPS_FOR_FIT} crops for KMeans (got {len(crops)}).")
    print("        Try lowering --conf or using a longer clip.")
    sys.exit(1)

classifier = TeamClassifier(
    device=device,
    n_clusters=KMEANS_CLUSTERS,
    n_init=args.kmeans_init,
)
classifier.fit(crops)
print("[INFO] Classifier ready.\n")

# ---------------------------------------------------------------------------
# Supervision annotation setup
# ---------------------------------------------------------------------------

TEAM_COLORS = [
    sv.Color(r=220, g=50,  b=50),    # Team 1 — red
    sv.Color(r=50,  g=100, b=220),   # Team 2 — blue
    sv.Color(r=180, g=180, b=180),   # Referee/Other — grey
]

box_annotators = [
    sv.BoxAnnotator(color=TEAM_COLORS[0], thickness=2, color_lookup=sv.ColorLookup.INDEX),
    sv.BoxAnnotator(color=TEAM_COLORS[1], thickness=2, color_lookup=sv.ColorLookup.INDEX),
    sv.BoxAnnotator(color=TEAM_COLORS[2], thickness=2, color_lookup=sv.ColorLookup.INDEX),
]

label_annotators = [
    sv.LabelAnnotator(color=TEAM_COLORS[0], text_scale=0.5, text_thickness=1, color_lookup=sv.ColorLookup.INDEX),
    sv.LabelAnnotator(color=TEAM_COLORS[1], text_scale=0.5, text_thickness=1, color_lookup=sv.ColorLookup.INDEX),
    sv.LabelAnnotator(color=TEAM_COLORS[2], text_scale=0.5, text_thickness=1, color_lookup=sv.ColorLookup.INDEX),
]

# ---------------------------------------------------------------------------
# Video writer setup
# ---------------------------------------------------------------------------

video_info = sv.VideoInfo.from_video_path(args.input)
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    str(output_path), fourcc, video_info.fps,
    (video_info.width, video_info.height)
)
if not out.isOpened():
    print(f"[ERROR] Could not create output video: {args.output}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# INFERENCE PHASE — process every frame
# ---------------------------------------------------------------------------

print(f"[INFO] === INFERENCE PHASE: processing {video_info.total_frames} frames ===")

frame_gen = sv.get_video_frames_generator(source_path=args.input)
if _TQDM_AVAILABLE:
    frame_gen = tqdm(frame_gen, total=video_info.total_frames, desc="Annotating")

try:
    for frame_idx, frame in enumerate(frame_gen):
        # --- Detect players ---
        results = yolo_model(
            frame, conf=args.conf, classes=YOLO_CLASSES,
            device=device, verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(results)

        if len(detections) == 0:
            out.write(frame)
            continue

        # --- Crop full bounding boxes (TeamClassifier extracts torso internally) ---
        all_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
        valid_mask   = [c is not None and c.size >= MIN_CROP_AREA for c in all_crops]
        valid_crops  = [c for c, v in zip(all_crops, valid_mask) if v]
        valid_indices = [i for i, v in enumerate(valid_mask) if v]

        if not valid_crops:
            out.write(frame)
            continue

        # --- Classify: get raw k=3 cluster IDs ---
        raw_ids = classifier.predict(valid_crops)
        valid_xyxy = detections.xyxy[valid_indices]
        frame_h = frame.shape[0]

        # --- Identify referee cluster for this frame ---
        # Primary: use the hue-matched cluster found during fit().
        # Fallback: lowest per-frame count if hue match failed.
        ref_cluster = classifier.referee_cluster_id
        if ref_cluster < 0:
            cluster_ids, counts = np.unique(raw_ids, return_counts=True)
            ref_cluster = int(cluster_ids[np.argmin(counts)])

        # --- Map raw cluster IDs → semantic labels (0=Team1, 1=Team2, 2=Referee) ---
        team_clusters = [c for c in range(KMEANS_CLUSTERS) if c != ref_cluster]
        cluster_to_label = {ref_cluster: 2}
        for rank, cid in enumerate(team_clusters):
            cluster_to_label[cid] = rank   # 0 or 1
        frame_labels = np.array([cluster_to_label[int(r)] for r in raw_ids])

        # --- Bench-exclusion: referee in bottom zone defines a sideline cutoff ---
        # Find bottom-zone referees (centre-y > REFEREE_BOTTOM_ZONE_FRAC * frame_h).
        # Their bounding-box bottom edge (y2) becomes the cutoff line.
        # Any non-referee detection whose centre-y is below that line is discarded
        # (bench players sitting alongside the court below the referee).
        ref_mask = frame_labels == 2
        cutoff_y = frame_h   # default: no cutoff
        if ref_mask.any():
            ref_boxes = valid_xyxy[ref_mask]       # shape (N_ref, 4)
            ref_cy = (ref_boxes[:, 1] + ref_boxes[:, 3]) / 2.0
            bottom_zone_refs = ref_boxes[ref_cy > frame_h * REFEREE_BOTTOM_ZONE_FRAC]
            if len(bottom_zone_refs):
                cutoff_y = float(bottom_zone_refs[:, 3].max())   # max y2 of bottom-zone refs

        # Build keep mask: referees always kept; players only if centre-y < cutoff_y
        keep = np.ones(len(frame_labels), dtype=bool)
        for i, lbl in enumerate(frame_labels):
            if lbl != 2:   # player detection
                cy = (valid_xyxy[i, 1] + valid_xyxy[i, 3]) / 2.0
                if cy > cutoff_y:
                    keep[i] = False   # below referee sideline → discard (bench player)

        final_labels = frame_labels[keep]
        final_xyxy   = valid_xyxy[keep]

        LABEL_NAMES = ["Team 1", "Team 2", "Referee"]

        # --- Annotate frame ---
        annotated = frame.copy()
        for grp in range(3):
            mask = final_labels == grp
            if not mask.any():
                continue
            grp_xyxy = final_xyxy[mask]
            det = sv.Detections(xyxy=grp_xyxy)
            annotated = box_annotators[grp].annotate(scene=annotated, detections=det)
            annotated = label_annotators[grp].annotate(
                scene=annotated, detections=det,
                labels=[LABEL_NAMES[grp]] * int(mask.sum())
            )

        out.write(annotated)

        # Progress log when tqdm is unavailable
        if not _TQDM_AVAILABLE and frame_idx % 100 == 0:
            pct = 100 * frame_idx / max(video_info.total_frames, 1)
            print(f"  {frame_idx}/{video_info.total_frames} frames ({pct:.1f}%)")

finally:
    out.release()

print(f"\n[DONE] Output saved to: {args.output}")
print("[NOTE] If team colours are swapped, swap the two TEAM_COLORS entries in the script.")
print("[NOTE] Check the 'dominant hue' log lines above to verify the two clusters")
print("       correspond to distinct jersey hues (~30 deg apart is a good sign).")
