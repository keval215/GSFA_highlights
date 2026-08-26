"""
modules/team_classifier/test_team_classifier.py — Visual test for team classification

Usage:
    # SigLIP + UMAP + KMeans (default)
    highlights\\Scripts\\python.exe modules/team_classifier/test_team_classifier.py

    # Colour histogram + UMAP + KMeans
    highlights\\Scripts\\python.exe modules/team_classifier/test_team_classifier.py --colour

Output:
    data/debug/team_test/         → SigLIP results
    data/debug/team_test_colour/  → Colour histogram results

    Two files per frame:
      frame_XXXXX.jpg  — full annotated frame (blue=team0, red=team1)
      crops_XXXXX.jpg  — grid of torso crops bordered by team colour
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from modules.detectors.player_detector import PlayerDetector
from modules.team_classifier.team_classifier import GSFATeamClassifier
from modules.team_classifier.colour_histogram import ColourHistogramTeamClassifier

# ---------------------------------------------------------------------------
# ARGS
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Team classifier visual test")
parser.add_argument(
    "--colour", action="store_true",
    help="Use colour histogram classifier instead of SigLIP"
)
parser.add_argument(
    "--force-refit", action="store_true",
    help="Ignore saved pkl and refit from scratch"
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Mirror video_analysis/possession.py exactly so the dumped crops/labels match
# what the real run produces (same video, same detector weights, same device,
# same conf, same class filter, same fit cadence).
VIDEO_PATH           = r"C:\Users\Admin\Downloads\test.mp4"
PLAYER_MODEL_WEIGHTS = r"C:\Users\Admin\OneDrive\Desktop\CZ\aiff_v1.pt"
DEVICE               = "cuda"
PLAYER_CONF          = 0.55
DETECT_CLASSES       = [0, 1, 2]   # active_player, ball, goal_post (referee=3 filtered)
N_PREVIEW            = 10
SAMPLE_EVERY         = 30          # fit cadence — matches possession's default (1 fps @ 30fps)

if args.colour:
    OUT_DIR = Path(r"D:\GSFA_highlights\data\debug\team_test_colour")
    MODE    = "COLOUR HISTOGRAM"
else:
    OUT_DIR = Path(r"D:\GSFA_highlights\data\debug\team_test")
    MODE    = "SIGLIP"

PKL_PATH = None   # auto-derived from video path per match

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
print("=" * 60)
print(f"  TEAM CLASSIFIER TEST  [{MODE}]")
print("=" * 60)
print(f"  Video : {VIDEO_PATH}")
print(f"  PKL   : auto → data/cache/<video_stem>_{'team_colour' if args.colour else 'team_siglip'}.pkl")
print(f"  Output: {OUT_DIR}")
print()

# ---------------------------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------------------------
print("[1/3] Loading PlayerDetector …")
player_det = PlayerDetector(model_path=PLAYER_MODEL_WEIGHTS, device=DEVICE,
                            classes=DETECT_CLASSES, player_conf=PLAYER_CONF)

print(f"[2/3] Fitting / loading {MODE} classifier …")
if args.colour:
    clf = ColourHistogramTeamClassifier()
else:
    clf = GSFATeamClassifier(device=DEVICE)

clf.fit_from_video_or_load(
    video_path   = VIDEO_PATH,
    player_det   = player_det,
    sample_every = SAMPLE_EVERY,
    progress     = True,
    force_refit  = args.force_refit,
)

# ---------------------------------------------------------------------------
# SAMPLE FRAMES
# ---------------------------------------------------------------------------
print(f"\n[3/3] Sampling {N_PREVIEW} frames …")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    sys.exit(f"ERROR: Cannot open {VIDEO_PATH}")

total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0

sample_indices = set(int(i * total_f / N_PREVIEW) for i in range(N_PREVIEW))

TEAM_COLOURS = {0: (255, 80, 0), 1: (0, 80, 255), None: (160, 160, 160)}
THUMB_H, THUMB_W, BORDER = 96, 64, 4

team0_total = team1_total = player_total = 0
diag_emb: list[np.ndarray] = []   # 768-d SigLIP per classified player (for fit diagnostics)
diag_lbl: list[int] = []          # its assigned team_id
fidx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if fidx in sample_indices:
        # Detect active_players only
        dets = player_det.detect(frame, frame_idx=fidx, fps=fps)
        dets.all = dets.players
        dets.referees = []
        dets.goal_posts = []

        clf.classify(frame, dets)

        t0 = sum(1 for p in dets.players if p.team_id == 0)
        t1 = sum(1 for p in dets.players if p.team_id == 1)
        team0_total  += t0
        team1_total  += t1
        player_total += len(dets.players)

        # Collect SigLIP embeddings + assigned labels for post-run fit diagnostics
        for p in dets.players:
            if p.embedding is not None and p.team_id is not None:
                diag_emb.append(p.embedding)
                diag_lbl.append(p.team_id)

        # --- annotated full frame ---
        if args.colour:
            annotated = ColourHistogramTeamClassifier.draw(frame, dets)
        else:
            annotated = GSFATeamClassifier.draw(frame, dets)

        hud = (f"F{fidx:05d}  players={len(dets.players)}  "
               f"T0(blue)={t0}  T1(red)={t1}  [{MODE}]")
        cv2.putText(annotated, hud, (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(OUT_DIR / f"frame_{fidx:05d}.jpg"), annotated)

        # --- torso crop grid ---
        thumbs = []
        for p in dets.players:
            if args.colour:
                crop = ColourHistogramTeamClassifier._torso_crop(frame, p.bbox)
            else:
                crop = GSFATeamClassifier._torso_crop(frame, p.bbox)
            if crop.size == 0:
                continue
            thumb  = cv2.resize(crop, (THUMB_W, THUMB_H))
            colour = TEAM_COLOURS[p.team_id]
            thumb  = cv2.copyMakeBorder(thumb, BORDER, BORDER, BORDER, BORDER,
                                        cv2.BORDER_CONSTANT, value=colour)
            label_bar = np.zeros((18, thumb.shape[1], 3), dtype=np.uint8)
            label = f"T{p.team_id}" if p.team_id is not None else "?"
            cv2.putText(label_bar, label, (4, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
            thumbs.append(np.vstack([thumb, label_bar]))

        crops_path = None
        if thumbs:
            COLS = 10
            rows = []
            for i in range(0, len(thumbs), COLS):
                row_t = thumbs[i:i + COLS]
                while len(row_t) < COLS:
                    row_t.append(np.zeros_like(row_t[0]))
                rows.append(np.hstack(row_t))
            crops_path = OUT_DIR / f"crops_{fidx:05d}.jpg"
            cv2.imwrite(str(crops_path), np.vstack(rows))

        print(f"  frame {fidx:>5}  players={len(dets.players):>2}  T0={t0}  T1={t1}"
              + (f"  crops → {crops_path.name}" if crops_path else ""))

    fidx += 1

cap.release()

# ---------------------------------------------------------------------------
# FIT DIAGNOSTICS (SigLIP) — is the clustering actually separating teams?
# ---------------------------------------------------------------------------
if not args.colour and diag_emb:
    feats  = np.asarray(diag_emb, dtype=np.float32)
    labels = np.asarray(diag_lbl, dtype=int)
    proj   = clf._classifier.reducer.transform(feats)      # (N, 3) UMAP space
    sizes  = np.bincount(labels, minlength=2)

    np.save(OUT_DIR / "umap_projections.npy", proj)
    np.save(OUT_DIR / "umap_labels.npy", labels)

    print()
    print("=" * 60)
    print("  FIT DIAGNOSTICS  [SIGLIP]")
    print("=" * 60)
    print(f"  Classified players : {len(labels)}")
    print(f"  Cluster sizes      : T0={sizes[0]}  T1={sizes[1]}")
    if len(np.unique(labels)) > 1 and len(labels) > 2:
        from sklearn.metrics import silhouette_score
        sil = silhouette_score(proj, labels)
        print(f"  Silhouette (UMAP)  : {sil:.3f}  (>0.5 clean split, <0.2 weak/degenerate)")
    else:
        print("  Silhouette (UMAP)  : n/a — only ONE cluster populated (degenerate fit)")
    print(f"  Saved              : umap_projections.npy + umap_labels.npy → {OUT_DIR}")
    print("  Interpretation:")
    print("   - one cluster ~0 OR silhouette <0.2 → SigLIP isn't separating the")
    print("     jerseys (crops background-dominated); not a _torso_crop code bug.")
    print("   - both clusters healthy + good silhouette but wrong borders in")
    print("     crops_XXXXX.jpg → crop grabs the wrong region; inspect _torso_crop.")
    print("=" * 60)

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print(f"  RESULTS  [{MODE}]")
print("=" * 60)
print(f"  Frames sampled  : {N_PREVIEW}")
print(f"  Players detected: {player_total}")
print(f"  Team 0 (blue)   : {team0_total}")
print(f"  Team 1 (red)    : {team1_total}")
print(f"  Debug frames    : {OUT_DIR}")
print()
print("  Check crops_XXXXX.jpg:")
print("  - Blue border = team 0,  Red border = team 1")
print("  - Same jersey colour should have the same border colour")
print("=" * 60)
