"""
heatmap.py — Full pipeline heatmap generator.

Runs PlayerDetector + ColourHistogramTeamClassifier + GoalkeeperDetector
on every frame of a video, projects each player's foot_point to world-space
(metres) via homography, and writes two team heatmap PNGs:

  data/output/team0_heatmap.png   (blue colourmap)
  data/output/team1_heatmap.png   (red/orange colourmap)

Both PNGs show a top-down futsal pitch diagram with blended heatmap overlay.

Usage:
  cd D:\\GSFA_highlights
  highlights\\Scripts\\python.exe heatmap.py

Edit the CONFIG block below to point at a different video.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG  ← edit these three lines when running on a new video
# ---------------------------------------------------------------------------
VIDEO_PATH   = r"C:\Users\Admin\Downloads\Video Project 10.mp4"
OUT_DIR      = Path(r"D:\GSFA_highlights\data\output")
SAMPLE_EVERY = 1   # 1 = every frame  |  2 = every other frame (2× faster)
# ---------------------------------------------------------------------------

# Pitch dimensions (metres) — must match simple_homography.py PITCH_W/PITCH_H
PITCH_W, PITCH_H = 40.0, 20.0

# Heatmap canvas: 20 px/m → 800 × 400 px
MAP_SCALE   = 20
MAP_MARGIN  = 40           # pixel border around the pitch drawing
CANVAS_W    = int(PITCH_W * MAP_SCALE) + 2 * MAP_MARGIN   # 840
CANVAS_H    = int(PITCH_H * MAP_SCALE) + 2 * MAP_MARGIN   # 440

# Gaussian blur radius for smoothing raw hit counts (σ in pixels ≈ 1 m)
BLUR_SIGMA = 20


# ---------------------------------------------------------------------------
# PITCH DIAGRAM HELPERS
# ---------------------------------------------------------------------------

def _world_to_map(wx: float, wy: float) -> tuple[int, int]:
    """Convert world metres (x left→right, y far→near) to map pixel coords."""
    mx = int(wx * MAP_SCALE) + MAP_MARGIN
    my = int(wy * MAP_SCALE) + MAP_MARGIN
    return mx, my


def build_pitch_canvas() -> np.ndarray:
    """Return a BGR image of a top-down futsal pitch (green + white lines)."""
    canvas = np.full((CANVAS_H, CANVAS_W, 3), (34, 85, 34), dtype=np.uint8)

    # Pitch boundary
    pts = np.array([
        _world_to_map(0, 0), _world_to_map(PITCH_W, 0),
        _world_to_map(PITCH_W, PITCH_H), _world_to_map(0, PITCH_H),
    ], np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], True, (255, 255, 255), 2)

    # Halfway line
    cv2.line(canvas, _world_to_map(PITCH_W / 2, 0),
             _world_to_map(PITCH_W / 2, PITCH_H), (255, 255, 255), 2)

    # Centre circle (radius 3 m)
    cx, cy = _world_to_map(PITCH_W / 2, PITCH_H / 2)
    cv2.circle(canvas, (cx, cy), int(3 * MAP_SCALE), (255, 255, 255), 2)
    cv2.circle(canvas, (cx, cy), 4, (255, 255, 255), -1)

    # Penalty areas (futsal: 6 m radius from post, approximated as rectangles)
    # Left penalty area (x=0..6, y=5..15)
    cv2.rectangle(canvas, _world_to_map(0, 5), _world_to_map(6, 15),
                  (255, 255, 255), 1)
    # Right penalty area (x=34..40, y=5..15)
    cv2.rectangle(canvas, _world_to_map(34, 5), _world_to_map(40, 15),
                  (255, 255, 255), 1)

    # Goal mouths (yellow, 3 m wide centred at y=10)
    cv2.line(canvas, _world_to_map(0, 8.5), _world_to_map(0, 11.5),
             (0, 215, 255), 4)
    cv2.line(canvas, _world_to_map(PITCH_W, 8.5),
             _world_to_map(PITCH_W, 11.5), (0, 215, 255), 4)

    return canvas


def render_team_heatmap(
        heat: np.ndarray,
        team_id: int,
        n_detections: int,
        colourmap: int,
        team_colour_bgr: tuple[int, int, int],
) -> np.ndarray:
    """
    Blend a Gaussian-smoothed heatmap with the pitch canvas.

    Args:
        heat:             (CANVAS_H, CANVAS_W) float32 accumulator.
        team_id:          0 or 1 — used only for the title string.
        n_detections:     Total detection count for the title.
        colourmap:        cv2.COLORMAP_* constant.
        team_colour_bgr:  Swatch colour for the legend.

    Returns:
        BGR image ready to write with cv2.imwrite().
    """
    # 1. Smooth
    blurred = cv2.GaussianBlur(heat, (0, 0), sigmaX=BLUR_SIGMA)

    # 2. Normalise
    max_val = blurred.max()
    if max_val > 0:
        norm_u8 = (blurred / max_val * 255).astype(np.uint8)
    else:
        norm_u8 = blurred.astype(np.uint8)

    # 3. Colourize
    coloured = cv2.applyColorMap(norm_u8, colourmap)   # (H, W, 3)

    # 4. Blend with pitch canvas
    pitch = build_pitch_canvas()
    alpha = np.clip(norm_u8.astype(np.float32) / 255.0, 0.0, 1.0)
    alpha_3 = np.stack([alpha] * 3, axis=2)
    blend = (
        pitch.astype(np.float32) * (1.0 - alpha_3 * 0.75)
        + coloured.astype(np.float32) * alpha_3 * 0.75
    ).astype(np.uint8)

    # 5. Redraw pitch lines so they are always visible
    # Threshold must be above green background (34+85+34=153) but below
    # yellow goal mouth (0+215+255=470) and white lines (255*3=765).
    lines_only = build_pitch_canvas()
    line_mask = lines_only.sum(axis=2) > 300
    blend[line_mask] = lines_only[line_mask]

    # 6. Title bar
    title_h = 50
    output = np.zeros((CANVAS_H + title_h, CANVAS_W, 3), dtype=np.uint8)
    output[title_h:, :] = blend
    cv2.rectangle(output, (0, 0), (CANVAS_W, title_h), (20, 20, 20), -1)

    label = f"Team {team_id} Heatmap   |   {n_detections:,} player-frame detections"
    cv2.putText(output, label, (10, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # 7. Colour swatch in title bar
    sw_x, sw_y = CANVAS_W - 30, 10
    cv2.rectangle(output, (sw_x, sw_y), (CANVAS_W - 5, sw_y + 30),
                  team_colour_bgr, -1)

    return output


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Validate video
    # ------------------------------------------------------------------
    if not Path(VIDEO_PATH).exists():
        sys.exit(f"ERROR: Video not found: {VIDEO_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1 — load / fit all detectors  (cached after first run)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  GSFA HEATMAP PIPELINE")
    print("=" * 60)
    print(f"  Video       : {VIDEO_PATH}")
    print(f"  Sample every: {SAMPLE_EVERY} frame(s)")
    print(f"  Output dir  : {OUT_DIR}")
    print()

    # Player detector
    from detectors.player_detector import PlayerDetector
    print("[1/4] Loading player detector …")
    player_det = PlayerDetector()

    # Colour-histogram team classifier
    from team_classifier.colour_histogram import ColourHistogramTeamClassifier
    print("[2/4] Loading / fitting team classifier (colour histogram) …")
    team_clf = ColourHistogramTeamClassifier()
    team_clf.fit_from_video_or_load(VIDEO_PATH, player_det)

    # Goalkeeper detector
    from detectors.goalkeeper_detector import GoalkeeperDetector
    print("[3/4] Loading / fitting goalkeeper detector …")
    gk_det = GoalkeeperDetector()
    gk_det.fit_from_video_or_load(VIDEO_PATH, player_det, team_clf)

    # Homography H-list
    print("[4/4] Loading / computing homography H-list …")
    from video_analysis.simple_homography import build_h_list
    H_list = build_h_list(VIDEO_PATH)

    # ------------------------------------------------------------------
    # Phase 2 — per-frame accumulation
    # ------------------------------------------------------------------
    print()
    print("Processing frames …")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        sys.exit(f"ERROR: Cannot open video: {VIDEO_PATH}")

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # float32 accumulators in map-pixel space
    heat: dict[int, np.ndarray] = {
        0: np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32),
        1: np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32),
    }
    det_counts: dict[int, int] = {0: 0, 1: 0}

    t_start = time.time()
    fidx    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if fidx % SAMPLE_EVERY == 0:
            # Detect
            dets = player_det.detect(frame, fidx, fps)

            # Classify teams
            team_clf.classify(frame, dets)

            # Mark goalkeepers
            gk_det.classify(dets)

            # Project foot_point → world → accumulate
            H = H_list[fidx] if H_list and fidx < len(H_list) else None
            if H is not None:
                for p in dets.players:
                    if p.team_id not in (0, 1):
                        continue
                    # img_to_world expects (N, 2) array
                    from video_analysis.simple_homography import img_to_world
                    wx, wy = img_to_world(
                        np.array([[p.foot_point[0], p.foot_point[1]]],
                                 dtype=np.float32),
                        H
                    )[0]
                    # Clamp to pitch bounds
                    wx = float(np.clip(wx, 0.0, PITCH_W))
                    wy = float(np.clip(wy, 0.0, PITCH_H))
                    # Convert to map-pixel coords
                    mx = int(wx / PITCH_W * (CANVAS_W - 2 * MAP_MARGIN)) + MAP_MARGIN
                    my = int(wy / PITCH_H * (CANVAS_H - 2 * MAP_MARGIN)) + MAP_MARGIN
                    mx = int(np.clip(mx, 0, CANVAS_W - 1))
                    my = int(np.clip(my, 0, CANVAS_H - 1))
                    heat[p.team_id][my, mx] += 1
                    det_counts[p.team_id] += 1

            # Progress
            if fidx % 300 == 0 and fidx > 0:
                elapsed  = time.time() - t_start
                frames_done = fidx // SAMPLE_EVERY
                fps_proc = frames_done / elapsed if elapsed > 0 else 0
                remaining = (total_f - fidx) / max(1, SAMPLE_EVERY * fps_proc)
                pct = fidx / max(1, total_f) * 100
                print(f"  frame {fidx:>6}/{total_f} ({pct:4.0f}%)  "
                      f"{fps_proc:.1f} frames/s  "
                      f"ETA {remaining/60:.1f} min  "
                      f"T0:{det_counts[0]:,}  T1:{det_counts[1]:,}")

        fidx += 1

    cap.release()
    elapsed_total = time.time() - t_start
    print(f"\n  Done — {fidx} frames in {elapsed_total/60:.1f} min")
    print(f"  Detections: Team 0 = {det_counts[0]:,}  |  Team 1 = {det_counts[1]:,}")

    # ------------------------------------------------------------------
    # Phase 3 — render and save 2 heatmap PNGs
    # ------------------------------------------------------------------
    print()
    print("Rendering heatmaps …")

    TEAM_CONFIG = {
        0: {
            "colourmap":       cv2.COLORMAP_WINTER,   # blue tones
            "team_colour_bgr": (180, 60, 30),          # blue swatch
            "filename":        "team0_heatmap.png",
        },
        1: {
            "colourmap":       cv2.COLORMAP_HOT,       # red/orange tones
            "team_colour_bgr": (30, 60, 210),          # red swatch
            "filename":        "team1_heatmap.png",
        },
    }

    for t_id, cfg in TEAM_CONFIG.items():
        img = render_team_heatmap(
            heat        = heat[t_id],
            team_id     = t_id,
            n_detections= det_counts[t_id],
            colourmap   = cfg["colourmap"],
            team_colour_bgr = cfg["team_colour_bgr"],
        )
        out_path = OUT_DIR / cfg["filename"]
        cv2.imwrite(str(out_path), img)
        print(f"  Saved: {out_path}")

    print()
    print("=" * 60)
    print("  COMPLETE")
    print(f"  team0_heatmap.png  — {det_counts[0]:,} detections (blue)")
    print(f"  team1_heatmap.png  — {det_counts[1]:,} detections (red)")
    print("=" * 60)


if __name__ == "__main__":
    main()
