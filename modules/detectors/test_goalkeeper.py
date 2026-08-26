"""
modules/detectors/test_goalkeeper.py — Visual test for goalkeeper colour classification

Run:
    cd D:\\GSFA_highlights
    highlights\\Scripts\\python.exe modules/detectors/test_goalkeeper.py

Output:
    data/debug/gk_test/frame_XXXXX.jpg   — 10 annotated frames
      Yellow box + "GK-T0/T1" = goalkeeper (jersey colour matched TEAM{0,1}_GK_COLOUR)
      Blue  box  + "T0"       = team 0 outfield
      Red   box  + "T1"       = team 1 outfield
"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from modules.detectors.player_detector import PlayerDetector
from modules.detectors.goalkeeper_detector import GoalkeeperDetector
from modules.team_classifier.colour_histogram import ColourHistogramTeamClassifier

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VIDEO_PATH   = r"C:\Users\Admin\Downloads\2.mp4"
OUT_DIR      = Path(r"D:\GSFA_highlights\data\debug\gk_test")
N_PREVIEW    = 10
SAMPLE_EVERY = 30
FORCE_REFIT  = False   # set True to ignore cached team-classifier pkl and refit

# Hex or CSS-name jersey colour of each team's goalkeeper — set these to
# match the actual footage before running.
# team0 outfield = dark blue, team1 outfield = orange (not used by this
# script — ColourHistogramTeamClassifier fits outfield colours unsupervised
# from the video, it doesn't take reference colours).
TEAM0_GK_COLOUR = "#90EE90"   # light green
TEAM1_GK_COLOUR = "#FFC0CB"   # pink

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------------------------
print("=" * 60)
print("  GOALKEEPER COLOUR CLASSIFICATION TEST")
print("=" * 60)
print(f"  Video        : {VIDEO_PATH}")
print(f"  Output       : {OUT_DIR}")
print(f"  team0_gk_colour: {TEAM0_GK_COLOUR}")
print(f"  team1_gk_colour: {TEAM1_GK_COLOUR}")
print()

print("[1/2] Loading PlayerDetector …")
player_det = PlayerDetector()

print("[2/2] Fitting / loading TeamClassifier …")
team_clf = ColourHistogramTeamClassifier()
team_clf.fit_from_video_or_load(
    video_path   = VIDEO_PATH,
    player_det   = player_det,
    sample_every = SAMPLE_EVERY,
    progress     = True,
    force_refit  = FORCE_REFIT,
)

# No fit stage — GoalkeeperDetector just needs the two reference colours.
gk_det = GoalkeeperDetector(TEAM0_GK_COLOUR, TEAM1_GK_COLOUR)

# ---------------------------------------------------------------------------
# SAMPLE FRAMES
# ---------------------------------------------------------------------------
print(f"\nSampling {N_PREVIEW} frames …")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    sys.exit(f"ERROR: Cannot open {VIDEO_PATH}")

total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0

sample_indices = set(int(i * total_f / N_PREVIEW) for i in range(N_PREVIEW))

fidx = 0
gk_found_total = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if fidx in sample_indices:
        # Full pipeline: detect → team classify → GK classify (independent passes)
        dets = player_det.detect(frame, frame_idx=fidx, fps=fps)
        dets.all      = dets.players
        dets.referees = []
        team_clf.classify(frame, dets)
        gk_det.classify(frame, dets)

        gks = [p for p in dets.players if p.is_goalkeeper]
        t0  = [p for p in dets.players if not p.is_goalkeeper and p.team_id == 0]
        t1  = [p for p in dets.players if not p.is_goalkeeper and p.team_id == 1]
        gk_found_total += len(gks)

        # Draw players (GK=yellow, T0=blue, T1=red)
        annotated = GoalkeeperDetector.draw(frame, dets)

        # HUD
        hud = f"F{fidx:05d}  GK={len(gks)}  T0={len(t0)}  T1={len(t1)}"
        cv2.putText(annotated, hud, (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),       3, cv2.LINE_AA)
        cv2.putText(annotated, hud, (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),  2, cv2.LINE_AA)

        out_path = OUT_DIR / f"frame_{fidx:05d}.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"  frame {fidx:>5}  GK={len(gks)}  T0={len(t0)}  T1={len(t1)}  "
              f"→ {out_path.name}")

    fidx += 1

cap.release()

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("  RESULTS")
print("=" * 60)
print(f"  Frames sampled : {N_PREVIEW}")
print(f"  GKs found total: {gk_found_total}  (expect up to {N_PREVIEW*2} if both keepers always visible)")
print(f"  Debug frames   : {OUT_DIR}")
print()
print("  Legend:")
print("  - Yellow box  GK-T0/T1  = goalkeeper (colour match)")
print("  - Blue box    T0         = team 0 outfield")
print("  - Red  box    T1         = team 1 outfield")
print("=" * 60)
