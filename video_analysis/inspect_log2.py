"""
Cross-check the six keypoints that dominate H computation (3,4,7,8,9,12)
against what is visible in the debug frames.

For each keypoint, print:
  - its world coordinate from PITCH_KEYPOINTS
  - its pixel position in frame 0 (the clearest early frame)
  - what physical court marking that pixel actually sits on
    (human must verify visually, but we print the numbers)

Also compute: given frame 0's H, where does the wireframe boundary project?
This tells us which world corners land where in pixel space.
"""
import json
import numpy as np

PITCH_KEYPOINTS = {
    0:  ( 0.0, 10.0),
    1:  ( 0.0, 20.0),
    2:  (10.0, 20.0),
    3:  (20.0, 20.0),
    4:  (30.0, 20.0),
    5:  (40.0, 20.0),
    6:  (40.0, 10.0),
    7:  (40.0,  0.0),
    8:  (30.0,  0.0),
    9:  (20.0,  0.0),
   10:  (10.0,  0.0),
   11:  ( 0.0,  0.0),
   12:  (20.0, 10.0),
}

EXCLUDE = {0, 6}
KP_CONF_THRESH = 0.50
PITCH_W, PITCH_H = 40.0, 20.0

with open("D:/GSFA_highlights/video_analysis/keypoint_log.json") as f:
    data = json.load(f)

# --- Frame 0 keypoints and H ---
d0 = data[0]
print("=== Frame 0 — the canonical 6-point H-direct frame ===")
print(f"n_used={d0['n_used']}  H_solved={d0['H_solved']}")
print()

src_pts, dst_pts = [], []
for kd in d0["keypoints"]:
    i, c, px, py = kd["idx"], kd["conf"], kd["px"], kd["py"]
    if c >= KP_CONF_THRESH and i in PITCH_KEYPOINTS and i not in EXCLUDE:
        src_pts.append([px, py])
        dst_pts.append(list(PITCH_KEYPOINTS[i]))

import cv2
src_np = np.array(src_pts, dtype=np.float32)
dst_np = np.array(dst_pts, dtype=np.float32)
H, mask = cv2.findHomography(src_np, dst_np, cv2.RANSAC, 5.0)
inliers = int(mask.sum()) if mask is not None else 0
print(f"Recomputed H  inliers={inliers}/{len(src_pts)}")
print()

# Project the four field boundary corners back to image space
H_inv = np.linalg.inv(H)

def w2i(wx, wy):
    p = H_inv @ np.array([wx, wy, 1.0])
    return p[:2] / p[2]

print("=== Field boundary corners projected to pixel space (frame 0) ===")
corners = [
    ("far-left   pt11", 0.0,       0.0),
    ("far-right  pt7 ", PITCH_W,   0.0),
    ("near-right pt5 ", PITCH_W,   PITCH_H),
    ("near-left  pt1 ", 0.0,       PITCH_H),
]
for label, wx, wy in corners:
    px, py = w2i(wx, wy)
    print(f"  {label}  world=({wx:.0f},{wy:.0f})  ->  pixel=({px:.0f},{py:.0f})")

print()
print("=== Halfway line endpoints ===")
for label, wx, wy in [("pt9 far", 20.0, 0.0), ("pt3 near", 20.0, 20.0)]:
    px, py = w2i(wx, wy)
    print(f"  {label}  world=({wx:.0f},{wy:.0f})  ->  pixel=({px:.0f},{py:.0f})")

print()
print("=== Goal line endpoints ===")
for label, wx, wy in [
    ("left-goal top   ", 0.0, 8.5),
    ("left-goal bottom", 0.0, 11.5),
    ("right-goal top  ", 40.0, 8.5),
    ("right-goal bot  ", 40.0, 11.5),
]:
    px, py = w2i(wx, wy)
    print(f"  {label}  world=({wx:.0f},{wy:.1f})  ->  pixel=({px:.0f},{py:.0f})")

print()
print("=== Per-keypoint pixel positions in frame 0 ===")
print(f"  {'kp':>4}  {'world_x':>8}  {'world_y':>8}  {'px':>6}  {'py':>6}  {'conf':>6}  used")
for kd in sorted(d0["keypoints"], key=lambda x: x["idx"]):
    i, c, px, py = kd["idx"], kd["conf"], kd["px"], kd["py"]
    if c < 0.10:
        continue
    wcoord = PITCH_KEYPOINTS.get(i, ("?","?"))
    used = "YES" if (c >= KP_CONF_THRESH and i in PITCH_KEYPOINTS and i not in EXCLUDE) else "-"
    print(f"  {i:>4}  {wcoord[0]:>8}  {wcoord[1]:>8}  {px:>6.0f}  {py:>6.0f}  {c:>6.3f}  {used}")

print()
# --- Cross-check: for the kps actually used, compute reprojection error ---
print("=== Reprojection check — project world coord back to image, compare to detected pixel ===")
print(f"  {'kp':>4}  {'det_px':>8}  {'reproj_px':>10}  {'err_px':>8}")
for kd in sorted(d0["keypoints"], key=lambda x: x["idx"]):
    i, c, px, py = kd["idx"], kd["conf"], kd["px"], kd["py"]
    if not (c >= KP_CONF_THRESH and i in PITCH_KEYPOINTS and i not in EXCLUDE):
        continue
    wx, wy = PITCH_KEYPOINTS[i]
    rpx, rpy = w2i(wx, wy)
    err = np.sqrt((px - rpx)**2 + (py - rpy)**2)
    print(f"  {i:>4}  ({px:>5.0f},{py:>5.0f})  ({rpx:>5.0f},{rpy:>5.0f})  {err:>8.1f} px")

print()
# --- Summary of n_used breakdown and H solve rate ---
print("=== n_used distribution across all 484 frames ===")
from collections import Counter
ctr = Counter(d["n_used"] for d in data)
for k in sorted(ctr):
    solved = sum(1 for d in data if d["n_used"] == k and d["H_solved"])
    print(f"  n_used={k}: {ctr[k]} frames   H_solved={solved}")
