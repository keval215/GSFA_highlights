"""
simple_homography.py — Keypoint detection + homography reprojection pipeline.

Steps:
  1. Run YOLO11m-pose inference per frame (box_conf ≥ 0.25)
  2. Filter keypoints: keep conf ≥ KP_CONF_THRESH (0.50 default)
     Corner keypoints (pt1, pt5, pt7, pt11) use a lower threshold (0.35)
     because painted corners are geometrically unambiguous even at moderate conf.
  3. Compute H per frame via findHomography (RANSAC, ≥ 4 pts)
     Reject H if the used keypoints span < 60% of pitch width (left-side
     extrapolation guard — Fix B).
     When a frame has no valid H → interpolate H from nearest
     neighbours that did have a valid H.
  4. For each frame: project pitch wireframe through H⁻¹ onto the image
     Drawn: field boundary + halfway line + goal mouths
  5. Write annotated video + debug frames + JSON log.
  6. At the end: generate a top-down pitch coverage heatmap (PNG) showing
     which parts of the pitch were visible across the whole video.

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

Outputs (all inside data/ at the repo root):
  homography_overlay.mp4
  debug/frame_XXXX.jpg      every DEBUG_EVERY frames
  keypoint_log.json
  pitch_coverage_map.png    top-down heatmap of visible pitch area

Usage:
  cd D:\\GSFA_highlights
  highlights\\Scripts\\python.exe video_analysis\\simple_homography.py
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
VIDEO_PATH = r"C:\Users\Admin\Downloads\Video Project 10.mp4"
OUT_DIR    = Path(r"D:\GSFA_highlights\data")
OUT_VIDEO  = OUT_DIR / "homography_overlay.mp4"
DEBUG_DIR  = OUT_DIR / "debug"
LOG_PATH   = OUT_DIR / "keypoint_log.json"
MAP_PATH   = OUT_DIR / "pitch_coverage_map.png"

# ---------------------------------------------------------------------------
# THRESHOLDS
# ---------------------------------------------------------------------------
BOX_CONF_THRESH  = 0.25   # field bounding-box confidence floor
KP_CONF_THRESH   = 0.50   # default keypoint confidence required for H computation
MIN_KP_FOR_H     = 4      # need at least this many confident points to solve H
DEBUG_EVERY      = 2000    # save a labeled debug frame every N frames
EXCLUDE_FROM_H   = {0, 6} # goal posts — 3-D structures, not painted on the floor

# Corner keypoints: geometrically unambiguous (painted corners) — use a lower
# confidence threshold so left-side anchors are rescued even in domain-shifted video.
KP_CONF_OVERRIDE: dict[int, float] = {
    1:  0.35,   # near-left corner
    5:  0.35,   # near-right corner
    7:  0.35,   # far-right corner
    11: 0.35,   # far-left corner
}

# ---------------------------------------------------------------------------
# PITCH COORDINATE MAP
#   x : 0 m  =  left touchline   →  PITCH_W m  =  right touchline
#   y : 0 m  =  far touchline    →  PITCH_H m  =  near touchline (camera side)
# ---------------------------------------------------------------------------
PITCH_W, PITCH_H = 40.0, 20.0   # metres — real futsal pitch

PITCH_KEYPOINTS: dict[int, tuple[float, float]] = {
    0:  ( 0.0, 10.0),  # left goal post              [excluded from H]
    1:  ( 0.0, 20.0),  # near-left corner
    2:  (10.0, 20.0),  # near touchline mid-left  (W/4)
    3:  (20.0, 20.0),  # halfway × near touchline
    4:  (30.0, 20.0),  # near touchline mid-right (3W/4)
    5:  (40.0, 20.0),  # near-right corner
    6:  (40.0, 10.0),  # right goal post             [excluded from H]
    7:  (40.0,  0.0),  # far-right corner
    8:  (30.0,  0.0),  # far touchline mid-right  (3W/4)
    9:  (20.0,  0.0),  # halfway × far touchline
   10:  (10.0,  0.0),  # far touchline mid-left   (W/4)
   11:  ( 0.0,  0.0),  # far-left corner
   12:  (20.0, 10.0),  # centre spot / kickoff point
}

# ---------------------------------------------------------------------------
# PITCH WIREFRAME  (world coords, metres)
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
    proj /= proj[:, 2:3]
    return proj[:, :2]


def img_to_world(pts_img: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Project Nx2 image pixel coords into world coords via H."""
    pts_h = np.column_stack([pts_img,
                             np.ones(len(pts_img), dtype=np.float32)])
    proj = (H @ pts_h.T).T
    proj /= proj[:, 2:3]
    return proj[:, :2]


def draw_wireframe(frame: np.ndarray, H_inv: np.ndarray) -> None:
    """Overlay pitch wireframe onto frame using H_inv (modifies in-place)."""
    h_img, w_img = frame.shape[:2]

    def _draw_poly(world_pts: np.ndarray, color: tuple, closed: bool,
                   thickness: int = 2) -> None:
        img_pts = world_to_img(world_pts, H_inv)
        mask = ((img_pts[:, 0] > -200) & (img_pts[:, 0] < w_img + 200) &
                (img_pts[:, 1] > -200) & (img_pts[:, 1] < h_img + 200))
        img_pts = img_pts[mask]
        if len(img_pts) < 2:
            return
        pts_int = img_pts.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(frame, [pts_int], isClosed=closed,
                      color=color, thickness=thickness, lineType=cv2.LINE_AA)

    for poly in WIREFRAME_BOUNDARY:
        _draw_poly(poly, color=(255, 255, 255), closed=True, thickness=3)
    for line in WIREFRAME_LINES:
        _draw_poly(line, color=(255, 255, 255), closed=False, thickness=2)
    


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
        # World coord label — helps verify index→world mapping is correct
        if i in PITCH_KEYPOINTS:
            wx, wy = PITCH_KEYPOINTS[i]
            cv2.putText(frame, f"({wx:.0f},{wy:.0f})", (px[0] + 6, px[1] + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)


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
    return H / H[2, 2]


# ---------------------------------------------------------------------------
# PITCH COVERAGE HEATMAP
# ---------------------------------------------------------------------------

# Scale: how many pixels per metre in the output map
MAP_SCALE = 20   # 20 px/m → 800 × 400 px for a 40×20 m pitch
MAP_MARGIN = 30  # extra pixel margin around the pitch

def build_pitch_canvas() -> np.ndarray:
    """Create a blank top-down pitch background (grass green)."""
    W = int(PITCH_W * MAP_SCALE) + 2 * MAP_MARGIN
    H = int(PITCH_H * MAP_SCALE) + 2 * MAP_MARGIN
    canvas = np.full((H, W, 3), (34, 85, 34), dtype=np.uint8)  # dark green

    def world_to_map(x: float, y: float) -> tuple[int, int]:
        mx = int(x * MAP_SCALE) + MAP_MARGIN
        my = int(y * MAP_SCALE) + MAP_MARGIN
        return mx, my

    # Draw pitch outline and markings in white
    corners = [world_to_map(0, 0), world_to_map(PITCH_W, 0),
               world_to_map(PITCH_W, PITCH_H), world_to_map(0, PITCH_H)]
    cv2.polylines(canvas, [np.array(corners, np.int32).reshape(-1, 1, 2)],
                  True, (255, 255, 255), 2)

    # Halfway line
    cv2.line(canvas, world_to_map(PITCH_W / 2, 0),
             world_to_map(PITCH_W / 2, PITCH_H), (255, 255, 255), 2)

    # Centre circle (radius ~3 m for futsal)
    cx, cy = world_to_map(PITCH_W / 2, PITCH_H / 2)
    cv2.circle(canvas, (cx, cy), int(3 * MAP_SCALE), (255, 255, 255), 2)

    # Centre spot
    cv2.circle(canvas, (cx, cy), 4, (255, 255, 255), -1)

    # Goal mouths (yellow)
    cv2.line(canvas, world_to_map(0, 8.5), world_to_map(0, 11.5), (0, 215, 255), 4)
    cv2.line(canvas, world_to_map(PITCH_W, 8.5), world_to_map(PITCH_W, 11.5), (0, 215, 255), 4)

    # Keypoint labels
    for idx, (wx, wy) in PITCH_KEYPOINTS.items():
        mx, my = world_to_map(wx, wy)
        cv2.circle(canvas, (mx, my), 5, (200, 200, 200), -1)
        cv2.putText(canvas, str(idx), (mx + 5, my - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    return canvas


def accumulate_coverage(
        coverage_acc: np.ndarray,
        H: np.ndarray,
        img_w: int, img_h: int) -> None:
    """
    Project the image corners through H into world space and fill the visible
    polygon on the coverage accumulator (single-channel float32).
    """
    corners_img = np.array([
        [0, 0], [img_w, 0], [img_w, img_h], [0, img_h]
    ], dtype=np.float32)
    corners_world = img_to_world(corners_img, H)

    # Clip to pitch bounds
    corners_world[:, 0] = np.clip(corners_world[:, 0], 0, PITCH_W)
    corners_world[:, 1] = np.clip(corners_world[:, 1], 0, PITCH_H)

    # Convert to map pixels
    map_pts = np.column_stack([
        corners_world[:, 0] * MAP_SCALE + MAP_MARGIN,
        corners_world[:, 1] * MAP_SCALE + MAP_MARGIN,
    ]).astype(np.int32)

    cv2.fillConvexPoly(coverage_acc, map_pts, 1.0)


def render_coverage_map(
        coverage_acc: np.ndarray,
        pitch_canvas: np.ndarray,
        total_frames: int) -> np.ndarray:
    """
    Blend the coverage heatmap with the pitch canvas.
    coverage_acc: H×W float32, values = frame count visible
    """
    # Normalise to 0–1
    max_val = coverage_acc.max()
    if max_val > 0:
        norm = coverage_acc / max_val
    else:
        norm = coverage_acc.copy()

    # Colourmap: TURBO (blue→green→yellow→red) on the coverage data
    heat_u8  = (norm * 255).astype(np.uint8)
    heat_rgb = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)

    # Blend with pitch canvas: where coverage > 0, mix heatmap in
    alpha_mask = (norm > 0).astype(np.float32)
    alpha_3    = np.stack([alpha_mask] * 3, axis=2)
    blend = (pitch_canvas.astype(np.float32) * (1 - alpha_3 * 0.65) +
             heat_rgb.astype(np.float32)    *      alpha_3 * 0.65).astype(np.uint8)

    # Redraw pitch lines on top so they're always visible
    pitch_lines = build_pitch_canvas()
    line_mask   = (pitch_lines.sum(axis=2) > 100)   # white/yellow pixels
    blend[line_mask] = pitch_lines[line_mask]

    # Legend
    H_map, W_map = blend.shape[:2]
    legend_x, legend_y = W_map - 160, H_map - 90
    cv2.rectangle(blend, (legend_x - 10, legend_y - 20),
                  (W_map - 5, H_map - 5), (30, 30, 30), -1)
    pct_covered = float((coverage_acc > 0).sum()) / coverage_acc.size * 100
    cv2.putText(blend, "PITCH COVERAGE", (legend_x, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(blend, f"Frames: {total_frames}",
                (legend_x, legend_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
    cv2.putText(blend, f"Area covered: {pct_covered:.0f}%",
                (legend_x, legend_y + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
    cv2.putText(blend, "Blue=rare  Red=frequent",
                (legend_x, legend_y + 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    return blend


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run(video_path: str = VIDEO_PATH,
        model_path: str = MODEL_PATH) -> None:

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"[2/5] Opening video: {video_path}")
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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # H.264 — handles long frame counts; mp4v crashes on >~100k frames
    writer = cv2.VideoWriter(str(OUT_VIDEO), fourcc, fps, (w_vid, h_vid))
    if not writer.isOpened():
        sys.exit(f"ERROR: Cannot open output writer → {OUT_VIDEO}")

    # -----------------------------------------------------------------------
    # Pass 1: inference — collect per-frame detections & compute H where possible
    # -----------------------------------------------------------------------
    print(f"[3/5] Running inference + homography computation …")
    print(f"      Corner conf override (kp1,5,7,11) → {KP_CONF_OVERRIDE[1]:.2f}")

    frame_data: list[dict] = []
    H_list: list[np.ndarray | None] = []

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
        skip_reason: str = ""

        if r.keypoints is not None and len(r.keypoints) > 0:
            kp    = r.keypoints[0]
            confs = kp.conf[0].cpu().numpy()
            xys   = kp.xy[0].cpu().numpy()

            for i in range(min(13, len(confs))):
                c  = float(confs[i])
                px = float(xys[i][0])
                py = float(xys[i][1])
                kp_data.append({"idx": i, "px": px, "py": py, "conf": c})

                # Use per-keypoint threshold override for corners
                threshold = KP_CONF_OVERRIDE.get(i, KP_CONF_THRESH)
                if c >= threshold and i in PITCH_KEYPOINTS and i not in EXCLUDE_FROM_H:
                    src_pts.append([px, py])
                    dst_pts.append(list(PITCH_KEYPOINTS[i]))

        if len(src_pts) >= MIN_KP_FOR_H:
            # Fix B: geometric spread guard — reject one-sided H
            world_xs = [p[0] for p in dst_pts]
            x_span   = max(world_xs) - min(world_xs)
            if x_span < PITCH_W * 0.3:
                # Not enough horizontal spread — would extrapolate badly
                skip_reason = f"x_span={x_span:.1f}m<{PITCH_W*0.3:.0f}m"
            else:
                src_np  = np.array(src_pts, dtype=np.float32)
                dst_np  = np.array(dst_pts, dtype=np.float32)
                H_raw, mask = cv2.findHomography(src_np, dst_np,
                                                  cv2.RANSAC, 5.0)
                if H_raw is not None:
                    if np.linalg.cond(H_raw) > 1e7:
                        skip_reason = "singular"
                        H_raw = None
                    else:
                        inliers = int(mask.sum()) if mask is not None else 0
                        if inliers >= MIN_KP_FOR_H:
                            H = H_raw
                        else:
                            skip_reason = f"only {inliers} inliers"

        H_list.append(H)
        frame_data.append({
            "frame":       fidx,
            "ts":          fidx / fps,
            "keypoints":   kp_data,
            "n_used":      len(src_pts),
            "H_solved":    H is not None,
            "skip_reason": skip_reason,
        })

        if fidx % 50 == 0:
            solved = sum(1 for h in H_list if h is not None)
            pct = fidx / max(1, total_f) * 100
            print(f"    frame {fidx:>5}/{total_f}  ({pct:4.0f}%)  "
                  f"H solved so far: {solved}")

    cap.release()
    n_direct = sum(1 for h in H_list if h is not None)
    print(f"    Inference done.  H solved directly: {n_direct}/{len(H_list)} frames.")

    # -----------------------------------------------------------------------
    # Interpolate H for frames where it wasn't solvable
    # -----------------------------------------------------------------------
    print("    Interpolating H for skipped frames …")

    solved_idx = [i for i, h in enumerate(H_list) if h is not None]

    if not solved_idx:
        print("  ⚠️  WARNING: No frame had enough well-spread keypoints. "
              "Overlay will be skipped.\n"
              "  Try lowering KP_CONF_THRESH or check PITCH_KEYPOINTS map.")
    else:
        first_solved = solved_idx[0]
        last_solved  = solved_idx[-1]
        long_gap_warn = 0

        for fidx in range(len(H_list)):
            if H_list[fidx] is not None:
                continue

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
                if gap > fps * 3:
                    long_gap_warn += 1
                alpha = (fidx - prev_idx) / gap
                H_list[fidx] = lerp_H(H_list[prev_idx], H_list[next_idx],
                                      alpha)

        if long_gap_warn:
            print(f"  ⚠️  DOMAIN SHIFT WARNING: {long_gap_warn} frame(s) have "
                  f"no H neighbours within 3 s.  Consider adding training data "
                  f"from this venue.")

    # -----------------------------------------------------------------------
    # Pass 2: write output video + accumulate coverage heatmap
    # -----------------------------------------------------------------------
    print(f"[4/5] Writing overlay video → {OUT_VIDEO}")

    pitch_canvas  = build_pitch_canvas()
    map_h, map_w  = pitch_canvas.shape[:2]
    coverage_acc  = np.zeros((map_h, map_w), dtype=np.float32)

    cap2 = cv2.VideoCapture(video_path)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for fidx in range(len(H_list)):
        ret, frame = cap2.read()
        if not ret:
            break

        H    = H_list[fidx]
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
            threshold = KP_CONF_OVERRIDE.get(i, KP_CONF_THRESH)
            if c >= threshold and i in PITCH_KEYPOINTS:
                used_kp.add(i)

        # Draw wireframe overlay and accumulate coverage
        if H is not None:
            try:
                H_inv = np.linalg.inv(H)
                draw_wireframe(frame, H_inv)
                # Only accumulate coverage from directly-solved frames
                # (interpolated frames don't represent new real observations)
                if fdat["H_solved"]:
                    accumulate_coverage(coverage_acc, H, w_vid, h_vid)
            except np.linalg.LinAlgError:
                pass

        draw_keypoints_debug(frame, kp_px_list, kp_conf_list, used_kp)

        status = (f"F{fidx:04d}  "
                  f"kp_used={fdat['n_used']}  "
                  f"{'H-direct' if fdat['H_solved'] else 'H-interp'}"
                  + (f"  [{fdat['skip_reason']}]" if fdat.get("skip_reason") else ""))
        cv2.putText(frame, status, (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 128), 2, cv2.LINE_AA)

        writer.write(frame)

        if fidx % DEBUG_EVERY == 0:
            dbg_path = DEBUG_DIR / f"frame_{fidx:04d}.jpg"
            cv2.imwrite(str(dbg_path), frame)

    cap2.release()
    writer.release()

    # -----------------------------------------------------------------------
    # Pass 3: generate and save pitch coverage map
    # -----------------------------------------------------------------------
    print(f"[5/5] Generating pitch coverage map → {MAP_PATH}")
    coverage_map = render_coverage_map(coverage_acc, pitch_canvas, n_direct)
    cv2.imwrite(str(MAP_PATH), coverage_map)

    # -----------------------------------------------------------------------
    # Save JSON log
    # -----------------------------------------------------------------------
    with open(LOG_PATH, "w") as f:
        json.dump(frame_data, f, indent=2)
    print(f"    Log saved → {LOG_PATH}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    n_interp     = sum(1 for h in H_list if h is not None) - n_direct
    pct_coverage = float((coverage_acc > 0).sum()) / coverage_acc.size * 100

    from collections import Counter
    skip_counts = Counter(fd["skip_reason"] for fd in frame_data if fd.get("skip_reason"))

    print("\n" + "=" * 60)
    print("  SIMPLE HOMOGRAPHY COMPLETE")
    print("=" * 60)
    print(f"  Frames total         : {len(H_list)}")
    print(f"  H solved directly    : {n_direct}  "
          f"({100*n_direct/max(1,len(H_list)):.0f}%)")
    print(f"  H interpolated       : {n_interp}")
    print(f"  Pitch area covered   : {pct_coverage:.0f}%")
    print(f"  Output video         : {OUT_VIDEO}")
    print(f"  Coverage map PNG     : {MAP_PATH}")
    print(f"  Debug frames         : {DEBUG_DIR}/frame_XXXX.jpg")
    print(f"  Keypoint log         : {LOG_PATH}")
    if skip_counts:
        print()
        print("  Skip reason breakdown:")
        for reason, count in skip_counts.most_common():
            print(f"    {reason:45s}: {count}")
    print()
    print("  VISUAL CHECK:")
    print("  - Play homography_overlay.mp4.")
    print("  - White field boundary + halfway line should track")
    print("    real pitch markings as the camera pans.")
    print("  - pitch_coverage_map.png shows which areas of the")
    print("    pitch were visible (blue=rare, red=frequent).")
    print("=" * 60)


# ---------------------------------------------------------------------------
# H-LIST CACHE  (used by heatmap.py and other downstream consumers)
# ---------------------------------------------------------------------------

def _compute_h_list(video_path: str,
                    model_path: str = MODEL_PATH) -> list:
    """
    Minimal inference pass: compute per-frame homography matrices and return
    them as a list of length == total_frame_count.

    Each entry is either a (3,3) np.ndarray or None (unsolvable frame).
    After the raw pass, None-entries are filled via nearest-neighbour
    interpolation — identical logic to run().

    No video is written; no debug images are saved.
    """
    print(f"  [H] Loading keypoint model: {model_path}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  [H] {total_f} frames @ {fps:.1f} fps — running inference …")

    H_list: list = []

    for fidx in range(total_f):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=BOX_CONF_THRESH, verbose=False)
        r = results[0]

        H: np.ndarray | None = None
        src_pts: list = []
        dst_pts: list = []

        if r.keypoints is not None and len(r.keypoints) > 0:
            kp    = r.keypoints[0]
            confs = kp.conf[0].cpu().numpy()
            xys   = kp.xy[0].cpu().numpy()

            for i in range(min(13, len(confs))):
                c  = float(confs[i])
                px = float(xys[i][0])
                py = float(xys[i][1])
                threshold = KP_CONF_OVERRIDE.get(i, KP_CONF_THRESH)
                if c >= threshold and i in PITCH_KEYPOINTS and i not in EXCLUDE_FROM_H:
                    src_pts.append([px, py])
                    dst_pts.append(list(PITCH_KEYPOINTS[i]))

        if len(src_pts) >= MIN_KP_FOR_H:
            world_xs = [p[0] for p in dst_pts]
            x_span   = max(world_xs) - min(world_xs)
            if x_span >= PITCH_W * 0.3:
                src_np  = np.array(src_pts, dtype=np.float32)
                dst_np  = np.array(dst_pts, dtype=np.float32)
                H_raw, mask = cv2.findHomography(src_np, dst_np, cv2.RANSAC, 5.0)
                if H_raw is not None and np.linalg.cond(H_raw) <= 1e7:
                    inliers = int(mask.sum()) if mask is not None else 0
                    if inliers >= MIN_KP_FOR_H:
                        H = H_raw

        H_list.append(H)

        if fidx % 500 == 0:
            solved = sum(1 for h in H_list if h is not None)
            pct    = fidx / max(1, total_f) * 100
            print(f"  [H] frame {fidx:>5}/{total_f} ({pct:4.0f}%)  "
                  f"H-solved so far: {solved}")

    cap.release()

    # Interpolate H for frames where it wasn't solvable
    solved_idx = [i for i, h in enumerate(H_list) if h is not None]
    if solved_idx:
        first_solved = solved_idx[0]
        last_solved  = solved_idx[-1]
        for fidx in range(len(H_list)):
            if H_list[fidx] is not None:
                continue
            prev_idx = next_idx = None
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
                alpha = (fidx - prev_idx) / (next_idx - prev_idx)
                H_list[fidx] = lerp_H(H_list[prev_idx], H_list[next_idx], alpha)

    n_solved = sum(1 for h in H_list if h is not None)
    print(f"  [H] Done. H available for {n_solved}/{len(H_list)} frames "
          f"(after interpolation).")
    return H_list


def build_h_list(video_path: str,
                 model_path: str = MODEL_PATH,
                 force_recompute: bool = False) -> list:
    """
    Public API: return a cached per-frame H-matrix list for *video_path*.

    On first call the homography is computed and saved to
      data/cache/<video_stem>_H.pkl
    Subsequent calls load that file instantly.

    Args:
        video_path:      Path to the source video.
        model_path:      YOLO keypoint model (defaults to MODEL_PATH).
        force_recompute: If True, ignore existing cache and recompute.

    Returns:
        list of length == total_frame_count; each entry is (3,3) ndarray or None.
    """
    try:
        import joblib
        from detectors.cache import cache_path as _cache_path
    except ImportError as e:
        raise ImportError(
            "joblib and detectors.cache are required for build_h_list. "
            f"Original error: {e}"
        )

    pkl = _cache_path(video_path, "H")

    if not force_recompute and pkl.exists():
        print(f"[Homography] Loading cached H_list ({pkl.name}) …")
        return joblib.load(pkl)

    print("[Homography] Cache not found — computing H_list (runs once) …")
    H_list = _compute_h_list(video_path, model_path)

    pkl.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(H_list, pkl)
    print(f"[Homography] Saved {len(H_list)} H matrices → {pkl}")
    return H_list


if __name__ == "__main__":
    run()
