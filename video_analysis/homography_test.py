"""
homography_test.py — Keypoint detection + homography reprojection pipeline.

Steps:
  1. Run YOLO11m-pose inference per frame (box_conf ≥ 0.25)
  2. Filter keypoints: keep conf ≥ 0.50
  3. Compute H per frame via findHomography (RANSAC, ≥ 4 pts)
     When a frame has < 4 valid keypoints → interpolate H from nearest
     neighbours that did have a valid H.
  4. For each frame: project pitch wireframe through H⁻¹ onto the image
     Drawn: field boundary + halfway line (matching the 13-keypoint template)
  5. Write annotated video + debug frames + JSON log.

Keypoint layout (world coords):
  pt11 ——— pt10 ——— pt9 ——— pt8 ——— pt7     ← far touchline
  |                  |                  |
  pt0               pt12               pt6   ← goal post level (mid height)
  |                  |                  |
  pt1 ——— pt2 ——— pt3 ——— pt4 ——— pt5     ← near touchline

  Corners:          pt1, pt11, pt7, pt5
  Goal post level:  pt0 (left), pt6 (right)
  Halfway line:     pt9 (far), pt12 (centre/kickoff), pt3 (near)
  Intermediate:     pt2 divides pt1→pt3, pt4 divides pt3→pt5
                    pt10 divides pt11→pt9, pt8 divides pt9→pt7

Outputs (all inside video_analysis/):
  homography_overlay.mp4
  debug/frame_XXXX.jpg      every DEBUG_EVERY frames
  keypoint_log.json

Usage:
  cd D:\\GSFA_highlights
  highlights\\Scripts\\python.exe video_analysis\\homography_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
MODEL_PATH = r"C:\Users\Admin\OneDrive\Desktop\CZ\GSFA_keypoint\final_best.pt"
VIDEO_PATH = r"C:\Users\Admin\Downloads\Video Project 8.mp4"
OUT_DIR    = Path(r"D:\GSFA_highlights\video_analysis")
OUT_VIDEO  = OUT_DIR / "homography_overlay.mp4"
DEBUG_DIR  = OUT_DIR / "debug"
LOG_PATH   = OUT_DIR / "keypoint_log.json"

# ---------------------------------------------------------------------------
# THRESHOLDS
# ---------------------------------------------------------------------------
BOX_CONF_THRESH  = 0.25   # field bounding-box confidence floor
KP_CONF_THRESH   = 0.50   # keypoint confidence required for H computation
MIN_KP_FOR_H     = 4      # need at least this many confident points to solve H (OpenCV minimum)
DEBUG_EVERY      = 30     # save a labeled debug frame every N frames
EXCLUDE_FROM_H   = {0, 6} # goal posts — 3-D structures, not painted on the floor

# ---------------------------------------------------------------------------
# PITCH COORDINATE MAP
#   x : 0 m  =  left touchline   →  PITCH_W m  =  right touchline
#   y : 0 m  =  far touchline    →  PITCH_H m  =  near touchline (camera side)
#
#  Layout matches the 13-keypoint YOLO template exactly:
#
#  pt11 ——— pt10 ——— pt9 ——— pt8 ——— pt7     y=0  (far)
#  |                  |                  |
#  pt0               pt12               pt6         (mid height)
#  |                  |                  |
#  pt1 ——— pt2 ——— pt3 ——— pt4 ——— pt5     y=H  (near)
# ---------------------------------------------------------------------------
PITCH_W, PITCH_H = 40.0, 20.0   # metres — real futsal pitch

PITCH_KEYPOINTS: dict[int, tuple[float, float]] = {
    0:  ( 0.0, 10.0),  # left goal post — mid of left goal line  [excluded from H]
    1:  ( 0.0, 20.0),  # near-left corner
    2:  (10.0, 20.0),  # near touchline mid-left  (W/4)
    3:  (20.0, 20.0),  # halfway × near touchline
    4:  (30.0, 20.0),  # near touchline mid-right (3W/4)
    5:  (40.0, 20.0),  # near-right corner
    6:  (40.0, 10.0),  # right goal post — mid of right goal line [excluded from H]
    7:  (40.0,  0.0),  # far-right corner
    8:  (30.0,  0.0),  # far touchline mid-right  (3W/4)
    9:  (20.0,  0.0),  # halfway × far touchline
   10:  (10.0,  0.0),  # far touchline mid-left   (W/4)
   11:  ( 0.0,  0.0),  # far-left corner
   12:  (20.0, 10.0),  # centre spot / kickoff point
}

# ---------------------------------------------------------------------------
# PITCH WIREFRAME  (world coords, metres)
# Drawn from the 13-keypoint template only — no penalty areas, no arcs.
#
#  pt11 ——— pt10 ——— pt9 ——— pt8 ——— pt7     ← far touchline
#  |                  |                  |
#  pt0               pt12               pt6   ← goal post level
#  |                  |                  |
#  pt1 ——— pt2 ——— pt3 ——— pt4 ——— pt5     ← near touchline
# ---------------------------------------------------------------------------

def _poly_pts(points: list[tuple[float, float]]) -> np.ndarray:
    """Convert list of (x, y) world points to float32 array for polylines."""
    return np.array(points, dtype=np.float32)


# Field boundary — four corners as a closed rectangle
WIREFRAME_BOUNDARY = [
    _poly_pts([(0.0, 0.0), (40.0, 0.0), (40.0, 20.0), (0.0, 20.0)]),
]

# Halfway line — pt9 (20,0) → pt3 (20,20)
WIREFRAME_LINES = [
    _poly_pts([(20.0, 0.0), (20.0, 20.0)]),
]

# Goal mouths — standard futsal 3 m wide, centred on y=10 → y 8.5–11.5
WIREFRAME_GOALS = [
    _poly_pts([( 0.0,  8.5), ( 0.0, 11.5)]),   # left goal
    _poly_pts([(40.0,  8.5), (40.0, 11.5)]),   # right goal
]

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def world_to_img(pts_world: np.ndarray, H_inv: np.ndarray) -> np.ndarray:
    """Project Nx2 world points into image pixel coords via H⁻¹."""
    pts_h = np.column_stack([pts_world,
                             np.ones(len(pts_world), dtype=np.float32)])
    proj = (H_inv @ pts_h.T).T
    proj /= proj[:, 2:3]          # divide by w
    return proj[:, :2]


def draw_wireframe(frame: np.ndarray, H_inv: np.ndarray) -> None:
    """Overlay pitch wireframe onto frame using H_inv (modifies in-place)."""
    h_img, w_img = frame.shape[:2]

    def _draw_poly(world_pts: np.ndarray, color: tuple, closed: bool,
                   thickness: int = 2) -> None:
        img_pts = world_to_img(world_pts, H_inv)
        # Keep only points roughly inside a generous margin
        mask = ((img_pts[:, 0] > -200) & (img_pts[:, 0] < w_img + 200) &
                (img_pts[:, 1] > -200) & (img_pts[:, 1] < h_img + 200))
        img_pts = img_pts[mask]
        if len(img_pts) < 2:
            return
        pts_int = img_pts.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(frame, [pts_int], isClosed=closed,
                      color=color, thickness=thickness, lineType=cv2.LINE_AA)

    # Field boundary — white closed rectangle
    for poly in WIREFRAME_BOUNDARY:
        _draw_poly(poly, color=(255, 255, 255), closed=True, thickness=3)

    # Halfway line — white
    for line in WIREFRAME_LINES:
        _draw_poly(line, color=(255, 255, 255), closed=False, thickness=2)

    # Goal mouths — yellow (3 m wide on each goal line)
    for goal in WIREFRAME_GOALS:
        _draw_poly(goal, color=(0, 215, 255), closed=False, thickness=4)


def draw_keypoints_debug(frame: np.ndarray,
                         kp_px: list[tuple[int, int]],
                         kp_conf: list[float],
                         used_indices: set[int]) -> None:
    """Draw numbered keypoints on frame for debugging."""
    for i, (px, conf) in enumerate(zip(kp_px, kp_conf)):
        if px == (0, 0) or conf < 0.10:
            continue
        color = (0, 255, 0) if i in used_indices else (
                 (0, 165, 255) if conf >= 0.25 else (60, 60, 200))
        cv2.circle(frame, px, 7, color, -1)
        cv2.putText(frame, f"{i}({conf:.2f})", (px[0] + 6, px[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)


def lerp_H(H_prev: np.ndarray | None, H_next: np.ndarray | None,
           alpha: float) -> np.ndarray | None:
    """Linearly interpolate between two homography matrices."""
    if H_prev is None and H_next is None:
        return None
    if H_prev is None:
        return H_next
    if H_next is None:
        return H_prev
    H = (1.0 - alpha) * H_prev + alpha * H_next
    return H / H[2, 2]          # renormalise


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run(video_path: str = VIDEO_PATH,
        model_path: str = MODEL_PATH) -> None:

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"[2/4] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"ERROR: Cannot open {video_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_vid      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_f / fps

    print(f"    {w_vid}×{h_vid}  {fps:.1f} fps  {total_f} frames  "
          f"({duration_s:.1f} s)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_VIDEO), fourcc, fps, (w_vid, h_vid))
    if not writer.isOpened():
        sys.exit(f"ERROR: Cannot open output writer → {OUT_VIDEO}")

    # -----------------------------------------------------------------------
    # Pass 1: inference — collect per-frame detections & compute H where possible
    # -----------------------------------------------------------------------
    print(f"[3/4] Running inference + homography computation …")

    frame_data: list[dict] = []    # full per-frame log
    H_list: list[np.ndarray | None] = []  # H per frame (None if not solved)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for fidx in range(total_f):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=BOX_CONF_THRESH, verbose=False)
        r = results[0]

        kp_data: list[dict] = []
        H: np.ndarray | None = None
        src_pts: list[list[float]] = []   # image pixel coords
        dst_pts: list[list[float]] = []   # world metre coords

        if r.keypoints is not None and len(r.keypoints) > 0:
            kp    = r.keypoints[0]
            confs = kp.conf[0].cpu().numpy()
            xys   = kp.xy[0].cpu().numpy()

            for i in range(min(13, len(confs))):
                c  = float(confs[i])
                px = float(xys[i][0])
                py = float(xys[i][1])
                kp_data.append({"idx": i, "px": px, "py": py, "conf": c})

                if c >= KP_CONF_THRESH and i in PITCH_KEYPOINTS and i not in EXCLUDE_FROM_H:
                    src_pts.append([px, py])
                    dst_pts.append(list(PITCH_KEYPOINTS[i]))

        if len(src_pts) >= MIN_KP_FOR_H:
            src_np = np.array(src_pts, dtype=np.float32)
            dst_np = np.array(dst_pts, dtype=np.float32)
            H_raw, mask = cv2.findHomography(src_np, dst_np,
                                              cv2.RANSAC, 5.0)
            if H_raw is not None:
                if np.linalg.cond(H_raw) > 1e7:
                    H_raw = None   # truly degenerate (near-singular) — interpolate instead
                else:
                    inliers = int(mask.sum()) if mask is not None else 0
                    if inliers >= MIN_KP_FOR_H:
                        H = H_raw

        H_list.append(H)
        frame_data.append({
            "frame":    fidx,
            "ts":       fidx / fps,
            "keypoints": kp_data,
            "n_used":   len(src_pts),
            "H_solved": H is not None,
        })

        if fidx % 50 == 0:
            solved = sum(1 for h in H_list if h is not None)
            pct = fidx / max(1, total_f) * 100
            print(f"    frame {fidx:>4}/{total_f}  ({pct:4.0f}%)  "
                  f"H solved so far: {solved}")

    cap.release()
    print(f"    Inference done.  H solved in "
          f"{sum(1 for h in H_list if h is not None)}/{len(H_list)} frames.")

    # -----------------------------------------------------------------------
    # Interpolate H for frames where it wasn't solvable
    # -----------------------------------------------------------------------
    print("    Interpolating H for skipped frames …")

    # Find index of prev/next solved H for every frame
    solved_idx = [i for i, h in enumerate(H_list) if h is not None]

    if not solved_idx:
        print("  ⚠️  WARNING: No frame had ≥4 confident keypoints. "
              "Overlay will be skipped.\n"
              "  Try lowering KP_CONF_THRESH or check PITCH_KEYPOINTS map.")
    else:
        first_solved = solved_idx[0]
        last_solved  = solved_idx[-1]
        long_gap_warn = 0

        for fidx in range(len(H_list)):
            if H_list[fidx] is not None:
                continue  # already solved

            # Find nearest solved frames before and after
            prev_idx = None
            next_idx = None
            for si in solved_idx:
                if si < fidx:
                    prev_idx = si
                elif si > fidx and next_idx is None:
                    next_idx = si
                    break

            if prev_idx is None:
                H_list[fidx] = H_list[first_solved]
            elif next_idx is None:
                H_list[fidx] = H_list[last_solved]
            else:
                gap = next_idx - prev_idx
                if gap > fps * 3:          # warn if gap > 3 s
                    long_gap_warn += 1
                alpha = (fidx - prev_idx) / gap
                H_list[fidx] = lerp_H(H_list[prev_idx], H_list[next_idx],
                                      alpha)

        if long_gap_warn:
            print(f"  ⚠️  DOMAIN SHIFT WARNING: {long_gap_warn} frame(s) have "
                  f"no H neighbours within 3 s.  Consider adding training data "
                  f"from this venue.")

    # -----------------------------------------------------------------------
    # Pass 2: write output video with wireframe overlay
    # -----------------------------------------------------------------------
    print(f"[4/4] Writing overlay video → {OUT_VIDEO}")
    cap2 = cv2.VideoCapture(video_path)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for fidx in range(len(H_list)):
        ret, frame = cap2.read()
        if not ret:
            break

        H = H_list[fidx]
        fdat = frame_data[fidx]

        # Reconstruct used-in-H keypoint indices for debug colouring
        used_kp = set()
        kp_px_list: list[tuple[int, int]] = [(0, 0)] * 13
        kp_conf_list: list[float]         = [0.0]    * 13
        for kd in fdat["keypoints"]:
            i  = kd["idx"]
            c  = kd["conf"]
            px = (int(kd["px"]), int(kd["py"]))
            if i < 13:
                kp_px_list[i]   = px
                kp_conf_list[i] = c
            if c >= KP_CONF_THRESH and i in PITCH_KEYPOINTS:
                used_kp.add(i)

        # Draw wireframe overlay
        if H is not None:
            try:
                H_inv = np.linalg.inv(H)
                draw_wireframe(frame, H_inv)
            except np.linalg.LinAlgError:
                pass

        # Draw keypoints (debug)
        draw_keypoints_debug(frame, kp_px_list, kp_conf_list, used_kp)

        # Status text
        status = (f"F{fidx:04d}  "
                  f"kp_used={fdat['n_used']}  "
                  f"{'H-direct' if fdat['H_solved'] else 'H-interp'}")
        cv2.putText(frame, status, (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 128), 2, cv2.LINE_AA)

        writer.write(frame)

        # Save debug frames
        if fidx % DEBUG_EVERY == 0:
            dbg_path = DEBUG_DIR / f"frame_{fidx:04d}.jpg"
            cv2.imwrite(str(dbg_path), frame)

    cap2.release()
    writer.release()

    # -----------------------------------------------------------------------
    # Save JSON log
    # -----------------------------------------------------------------------
    with open(LOG_PATH, "w") as f:
        json.dump(frame_data, f, indent=2)
    print(f"    Log saved → {LOG_PATH}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    solved   = sum(1 for h in H_list if h is not None)
    n_direct = sum(1 for fd in frame_data if fd["H_solved"])
    print("\n" + "=" * 60)
    print("  HOMOGRAPHY TEST COMPLETE")
    print("=" * 60)
    print(f"  Frames total       : {len(H_list)}")
    print(f"  H solved directly  : {n_direct}  "
          f"({100*n_direct/max(1,len(H_list)):.0f}%)")
    print(f"  H interpolated     : {solved - n_direct}")
    print(f"  Output video       : {OUT_VIDEO}")
    print(f"  Debug frames       : {DEBUG_DIR}/frame_XXXX.jpg")
    print(f"  Keypoint log       : {LOG_PATH}")
    print()
    print("  VISUAL CHECK:")
    print("  - Play homography_overlay.mp4.")
    print("  - White field boundary + white halfway line should")
    print("    track the real pitch markings as the camera pans.")
    print("  - If they drift, adjust PITCH_KEYPOINTS at the top of this file.")
    print("=" * 60)


if __name__ == "__main__":
    run()
