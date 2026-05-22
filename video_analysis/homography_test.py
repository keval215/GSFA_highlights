"""
homography_test.py — Keypoint detection + homography reprojection pipeline.

Steps:
  1. Run YOLO11m-pose inference per frame (box_conf ≥ 0.25)
  2. Filter keypoints: keep conf ≥ 0.50
  3. Compute H per frame via findHomography (RANSAC, ≥ 4 pts)
     When a frame has < 4 valid keypoints → interpolate H from nearest
     neighbours that did have a valid H.
  4. For each frame: project pitch wireframe through H⁻¹ onto the image
     Drawn: court boundary, halfway line, centre circle, penalty D-arcs
  5. Write annotated video + debug frames + JSON log.

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
import math
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
MIN_KP_FOR_H     = 4      # need at least this many confident points to solve H
DEBUG_EVERY      = 30     # save a labeled debug frame every N frames

# ---------------------------------------------------------------------------
# PITCH COORDINATE MAP
#   x : 0 m  =  left goal line   →  40 m  =  right goal line
#   y : 0 m  =  far touchline    →  20 m  =  near touchline (camera side)
#
#  ⚠️  INITIAL GUESSES — verified against debug/frame_0000.jpg.
#      If the wireframe overlay is off, adjust the numbers here and re-run.
#      Confident assignments (from per-frame tracking + visual inspection):
#        KP3  → appears bottom-left  frame 0  → near-left corner  (0, 20)
#        KP4  → tracks to image centre         → halfway × near   (20, 20)
#        KP9  → appears far-left     frame 0  → far-left corner   (0, ~3)
#        KP12 → appears left-mid     frame 0  → left goal midline (0, 10)
#        KP7  → upper-right, hi-conf, pairs KP11 → right penalty far (34, 0)
#        KP8  → upper-left, hi-conf           → left penalty far  (6,  0)
# ---------------------------------------------------------------------------
PITCH_KEYPOINTS: dict[int, tuple[float, float]] = {
    0:  (20.0,  0.0),   # halfway × far touchline
    1:  ( 0.0,  0.0),   # far-left corner  (pairs with KP5)
    2:  (40.0,  0.0),   # far-right corner
    3:  ( 0.0, 20.0),   # near-left corner                ← LIKELY
    4:  (20.0, 20.0),   # halfway × near touchline        ← CONFIDENT
    5:  (40.0, 20.0),   # near-right corner  (pairs with KP1)
    6:  (40.0, 10.0),   # right goal midpoint
    7:  (34.0,  0.0),   # right penalty area × far touch  ← LIKELY (pairs KP11)
    8:  ( 6.0,  0.0),   # left  penalty area × far touch  ← LIKELY
    9:  ( 0.0,  3.0),   # far-left, just inside far corner ← LIKELY
   10:  (40.0,  3.0),   # far-right, just inside far corner
   11:  ( 6.0, 20.0),   # left  penalty area × near touch  (pairs KP7)
   12:  ( 0.0, 10.0),   # left goal midpoint               ← LIKELY
}

# ---------------------------------------------------------------------------
# PITCH WIREFRAME  (world coords, metres)
# These use KNOWN court geometry — independent of keypoint guesses.
# ---------------------------------------------------------------------------
PITCH_W, PITCH_H = 40.0, 20.0   # metres
PENALTY_SPOT_X   = 6.0           # left penalty spot x
CIRCLE_R         = 3.0           # centre-circle radius
PENALTY_ARC_R    = 6.0           # penalty D-arc radius


def _poly_pts(points: list[tuple[float, float]]) -> np.ndarray:
    """Convert list of (x, y) world points to float32 array for polylines."""
    return np.array(points, dtype=np.float32)


def _circle_poly(cx: float, cy: float, r: float, n: int = 32) -> np.ndarray:
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(angles),
                            cy + r * np.sin(angles)]).astype(np.float32)


def _arc_poly(cx: float, cy: float, r: float,
              a_start: float, a_end: float, n: int = 24) -> np.ndarray:
    angles = np.linspace(a_start, a_end, n)
    return np.column_stack([cx + r * np.cos(angles),
                            cy + r * np.sin(angles)]).astype(np.float32)


# Court boundary
WIREFRAME_BOUNDARY = [
    _poly_pts([(0, 0), (PITCH_W, 0), (PITCH_W, PITCH_H), (0, PITCH_H)]),
]

# Halfway line
WIREFRAME_LINES = [
    _poly_pts([(PITCH_W / 2, 0), (PITCH_W / 2, PITCH_H)]),
]

# Centre circle
WIREFRAME_CIRCLES = [
    _circle_poly(PITCH_W / 2, PITCH_H / 2, CIRCLE_R),
]

# Left penalty D-arc (faces inward, i.e., toward field / +x direction)
# Semicircle centered on left penalty spot (6, 10), arc from ~270° to ~90° (field side)
_lpa_cx, _lpa_cy = PENALTY_SPOT_X, PITCH_H / 2
WIREFRAME_ARCS = [
    _arc_poly(_lpa_cx, _lpa_cy, PENALTY_ARC_R,
              a_start=-math.pi / 2, a_end=math.pi / 2, n=24),   # left arc
    _arc_poly(PITCH_W - _lpa_cx, _lpa_cy, PENALTY_ARC_R,
              a_start=math.pi / 2, a_end=3 * math.pi / 2, n=24),  # right arc (mirror)
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

    # Boundary — white
    for poly in WIREFRAME_BOUNDARY:
        _draw_poly(poly, color=(255, 255, 255), closed=True, thickness=3)

    # Halfway line — white
    for line in WIREFRAME_LINES:
        _draw_poly(line, color=(255, 255, 255), closed=False, thickness=2)

    # Centre circle — cyan
    for circle in WIREFRAME_CIRCLES:
        _draw_poly(circle, color=(0, 255, 255), closed=True, thickness=2)

    # Penalty arcs — yellow
    for arc in WIREFRAME_ARCS:
        _draw_poly(arc, color=(0, 215, 255), closed=False, thickness=2)


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

                if c >= KP_CONF_THRESH and i in PITCH_KEYPOINTS:
                    src_pts.append([px, py])
                    dst_pts.append(list(PITCH_KEYPOINTS[i]))

        if len(src_pts) >= MIN_KP_FOR_H:
            src_np = np.array(src_pts, dtype=np.float32)
            dst_np = np.array(dst_pts, dtype=np.float32)
            H_raw, mask = cv2.findHomography(src_np, dst_np,
                                              cv2.RANSAC, 5.0)
            if H_raw is not None:
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
    print("  - The white court boundary + cyan circle + yellow arcs")
    print("    should track the real court markings as the camera pans.")
    print("  - If they don't, adjust PITCH_KEYPOINTS at the top of this file.")
    print("=" * 60)


if __name__ == "__main__":
    run()
