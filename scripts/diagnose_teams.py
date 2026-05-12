"""
Diagnostic script: visualise KMeans team clustering results.

Outputs:
  data/diagnostics/
    cluster_hue_histogram.png  — hue distribution per cluster
    cluster_grid_0.png         — grid of 30 sample crops from cluster 0
    cluster_grid_1.png         — grid of 30 sample crops from cluster 1
    all_crops_pca.png          — PCA scatter of all 84-dim HSV features, coloured by cluster
    torso_crops/               — all individual torso crops saved as JPEGs

Run:
  python scripts/diagnose_teams.py
  python scripts/diagnose_teams.py --input data/videos/match.mp4 --stride 30 --k 2
"""

import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input",  default="data/videos/test.mp4")
parser.add_argument("--model",  default="yolo11m.pt")
parser.add_argument("--device", default="cuda")
parser.add_argument("--conf",   type=float, default=0.3)
parser.add_argument("--stride", type=int,   default=30)
parser.add_argument("--k",      type=int,   default=2,  help="number of KMeans clusters")
parser.add_argument("--samples",type=int,   default=30, help="crops to show in grid per cluster")
parser.add_argument("--out-dir",default="data/diagnostics")
args = parser.parse_args()

OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
TORSO_DIR = OUT / "torso_crops"
TORSO_DIR.mkdir(exist_ok=True)

# Torso fractions — must match 02_team_classification.py
TORSO_TOP_FRAC = 0.25
TORSO_BOT_FRAC = 0.65
TORSO_LR_FRAC  = 0.15

H_BINS = 36
S_BINS = 32
V_BINS = 16
MIN_CROP_AREA = 400

YOLO_CLASSES = [0]

# ---------------------------------------------------------------------------
# Dependency imports
# ---------------------------------------------------------------------------
try:
    import supervision as sv
except ImportError:
    print("[ERROR] pip install supervision"); sys.exit(1)
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] pip install ultralytics"); sys.exit(1)
try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
except ImportError:
    print("[ERROR] pip install scikit-learn"); sys.exit(1)
try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

try:
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available — using CPU")
        args.device = "cpu"
except ImportError:
    args.device = "cpu"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_torso(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    ty1 = int(h * TORSO_TOP_FRAC)
    ty2 = int(h * TORSO_BOT_FRAC)
    tx1 = int(w * TORSO_LR_FRAC)
    tx2 = w - int(w * TORSO_LR_FRAC)
    if ty2 > ty1 and tx2 > tx1:
        return crop[ty1:ty2, tx1:tx2]
    return crop


def hsv_histogram(crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [H_BINS], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [S_BINS], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [V_BINS], [0, 256]).flatten()
    feat = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    if feat.sum() > 0:
        feat /= feat.sum()
    return feat


def dominant_rgb(cluster_hue_bins: np.ndarray) -> tuple:
    """Convert the dominant hue bin to an RGB colour for plotting."""
    bin_idx = int(np.argmax(cluster_hue_bins))
    hue_deg = bin_idx * (180.0 / H_BINS)          # OpenCV hue: 0-180
    hsv_pixel = np.uint8([[[int(hue_deg), 220, 200]]])
    bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return (bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)   # RGB for matplotlib


# ---------------------------------------------------------------------------
# Step 1 — collect crops
# ---------------------------------------------------------------------------
print(f"[INFO] Loading YOLO: {args.model}")
yolo = YOLO(args.model)

print(f"[INFO] Sampling video (stride={args.stride})…")
frame_gen = sv.get_video_frames_generator(source_path=args.input, stride=args.stride)
it = tqdm(frame_gen, desc="Collecting crops") if _TQDM else frame_gen

full_crops: List[np.ndarray] = []
torso_crops: List[np.ndarray] = []

for frame in it:
    results = yolo(frame, conf=args.conf, classes=YOLO_CLASSES,
                   device=args.device, verbose=False)[0]
    dets = sv.Detections.from_ultralytics(results)
    for xyxy in dets.xyxy:
        crop = sv.crop_image(frame, xyxy)
        if crop is None or crop.size < MIN_CROP_AREA:
            continue
        torso = extract_torso(crop)
        if torso.size < MIN_CROP_AREA:
            torso = crop
        full_crops.append(crop)
        torso_crops.append(torso)

print(f"[INFO] Collected {len(full_crops)} player crops.")

# ---------------------------------------------------------------------------
# Step 2 — extract features and cluster
# ---------------------------------------------------------------------------
print("[INFO] Extracting HSV features…")
features = np.array([hsv_histogram(t) for t in torso_crops], dtype=np.float32)

print(f"[INFO] Fitting KMeans k={args.k}…")
km = KMeans(n_clusters=args.k, n_init=20, max_iter=500, random_state=42)
labels = km.fit_predict(features)

for i, centroid in enumerate(km.cluster_centers_):
    dom_bin = np.argmax(centroid[:H_BINS])
    dom_hue = dom_bin * (180.0 / H_BINS)
    print(f"  Cluster {i}: dominant hue bin={dom_bin} (~{dom_hue:.0f}° OpenCV HSV)  "
          f"  size={np.sum(labels == i)}")

# ---------------------------------------------------------------------------
# Step 3 — save individual torso crops
# ---------------------------------------------------------------------------
print("[INFO] Saving individual torso crops…")
for idx, (torso, label) in enumerate(zip(torso_crops, labels)):
    fname = TORSO_DIR / f"cluster{label}_crop{idx:04d}.jpg"
    cv2.imwrite(str(fname), torso)
print(f"[INFO] Saved {len(torso_crops)} torso crops to {TORSO_DIR}/")

# ---------------------------------------------------------------------------
# Step 4 — Plot 1: Hue histograms per cluster
# ---------------------------------------------------------------------------
hue_bins_deg = np.arange(H_BINS) * (180.0 / H_BINS)
cluster_colors_rgb = [dominant_rgb(km.cluster_centers_[i][:H_BINS]) for i in range(args.k)]

fig, axes = plt.subplots(1, args.k, figsize=(6 * args.k, 4), sharey=True)
if args.k == 1:
    axes = [axes]

for i, ax in enumerate(axes):
    mask = labels == i
    if mask.sum() == 0:
        continue
    mean_hue_hist = features[mask, :H_BINS].mean(axis=0)
    ax.bar(hue_bins_deg, mean_hue_hist, width=180.0 / H_BINS,
           color=cluster_colors_rgb[i], edgecolor="black", linewidth=0.4)
    ax.set_title(f"Cluster {i}  (n={mask.sum()})\n"
                 f"dominant hue ≈ {np.argmax(mean_hue_hist) * (180.0/H_BINS):.0f}°",
                 fontsize=12)
    ax.set_xlabel("Hue (OpenCV 0-180°)")
    ax.set_ylabel("Normalised frequency")
    ax.set_xlim(0, 180)

fig.suptitle("Mean hue distribution per cluster  (torso crops)", fontsize=14)
plt.tight_layout()
out_hue = OUT / "cluster_hue_histogram.png"
fig.savefig(out_hue, dpi=120)
plt.close(fig)
print(f"[INFO] Saved: {out_hue}")

# ---------------------------------------------------------------------------
# Step 5 — Plot 2: Crop grids per cluster (full player crops, not torso)
# ---------------------------------------------------------------------------
THUMB_H, THUMB_W = 96, 48   # thumbnail size for grid
GRID_COLS = 10

def make_grid(crops_subset: List[np.ndarray], n_samples: int) -> np.ndarray:
    samples = crops_subset[:n_samples]
    thumbs = [cv2.resize(c, (THUMB_W, THUMB_H)) for c in samples]
    n = len(thumbs)
    n_rows = (n + GRID_COLS - 1) // GRID_COLS
    # pad to full grid
    pad = n_rows * GRID_COLS - n
    blank = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
    thumbs += [blank] * pad
    rows = []
    for r in range(n_rows):
        row = np.hstack(thumbs[r * GRID_COLS:(r + 1) * GRID_COLS])
        rows.append(row)
    return np.vstack(rows)


for i in range(args.k):
    mask = np.where(labels == i)[0]
    # shuffle for variety
    rng = np.random.default_rng(42)
    idx_sample = rng.choice(mask, size=min(args.samples, len(mask)), replace=False)
    subset = [full_crops[j] for j in sorted(idx_sample)]
    grid_img = make_grid(subset, args.samples)
    out_grid = OUT / f"cluster_grid_{i}.png"
    cv2.imwrite(str(out_grid), grid_img)
    print(f"[INFO] Saved: {out_grid}  (cluster {i}, {len(mask)} total crops)")

# ---------------------------------------------------------------------------
# Step 6 — Plot 3: PCA scatter coloured by cluster
# ---------------------------------------------------------------------------
print("[INFO] Computing PCA for scatter plot…")
pca = PCA(n_components=2, random_state=42)
reduced = pca.fit_transform(features)

fig, ax = plt.subplots(figsize=(8, 6))
palette = plt.cm.tab10.colors

for i in range(args.k):
    mask = labels == i
    dom_bin = int(np.argmax(km.cluster_centers_[i, :H_BINS]))
    dom_hue = dom_bin * (180.0 / H_BINS)
    ax.scatter(
        reduced[mask, 0], reduced[mask, 1],
        s=12, alpha=0.6,
        color=cluster_colors_rgb[i],
        label=f"Cluster {i}  (hue≈{dom_hue:.0f}°, n={mask.sum()})"
    )

ax.set_title("PCA of HSV torso features — coloured by KMeans cluster", fontsize=13)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.legend(fontsize=11)
plt.tight_layout()
out_pca = OUT / "all_crops_pca.png"
fig.savefig(out_pca, dpi=120)
plt.close(fig)
print(f"[INFO] Saved: {out_pca}")

# ---------------------------------------------------------------------------
# Step 7 — Plot 4: Dominant colour swatches per cluster
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4 * args.k, 2))
for i in range(args.k):
    rect = mpatches.FancyBboxPatch(
        (i / args.k + 0.02, 0.1), 0.9 / args.k, 0.8,
        boxstyle="round,pad=0.02",
        facecolor=cluster_colors_rgb[i], edgecolor="black", linewidth=1.5,
        transform=ax.transAxes
    )
    ax.add_patch(rect)
    dom_hue = np.argmax(km.cluster_centers_[i, :H_BINS]) * (180.0 / H_BINS)
    ax.text(
        (i + 0.5) / args.k, 0.5,
        f"Cluster {i}\nhue≈{dom_hue:.0f}°\nn={np.sum(labels==i)}",
        ha="center", va="center", transform=ax.transAxes,
        fontsize=13, fontweight="bold",
        color="white" if sum(cluster_colors_rgb[i]) < 1.5 else "black"
    )
ax.set_axis_off()
ax.set_title("Dominant jersey colour per cluster", fontsize=13, pad=8)
plt.tight_layout()
out_swatch = OUT / "cluster_colour_swatches.png"
fig.savefig(out_swatch, dpi=120)
plt.close(fig)
print(f"[INFO] Saved: {out_swatch}")

# ---------------------------------------------------------------------------
print(f"\n[DONE] All diagnostics saved to: {OUT}/")
print("  cluster_hue_histogram.png  — hue distribution per cluster")
print("  cluster_colour_swatches.png — dominant colour per cluster")
print("  cluster_grid_0.png / _1.png  — sample player crops per cluster")
print("  all_crops_pca.png           — PCA scatter of all HSV features")
print(f"  torso_crops/               — {len(torso_crops)} individual torso JPEGs")
