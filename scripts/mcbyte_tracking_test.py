"""
scripts/mcbyte_tracking_test.py — McByte tracker feasibility test (local GPU)
=============================================================================
Research script, not part of the production pipeline. Answers two questions
for this repo's `docs/codebase/tracking/` decisions:

  1. Does McByte (https://arxiv.org/pdf/2506.01373, roboflow/trackers) run on
     this machine's GPU without CUDA OOM, in both modes:
       - lightweight  (IoU-only, ByteTrack/BoT-SORT-style — no extra deps)
       - full mask    (+ SAM(vit_b) + Cutie, `pip install "trackers[mask]"`)
  2. How McByte behaves when a player leaves the frame and comes back later.

Ground truth on (2): McByte's `lost_track_buffer` only controls how long a
LOST TRACK SLOT is kept alive (in frames, expressed at 30 FPS and scaled by
`frame_rate`) — it does not add appearance-based re-identification. Re-
association within that window still requires the returning detection's box
to have positive IoU against the tracklet's Kalman-predicted position. A
player who is fully out of frame for 5s and returns at a different spot on
the pitch will NOT be IoU-matched even with a 5s buffer — the predicted box
has nowhere to anchor to. Full mode's SAM/Cutie masks don't change this: Cutie
propagates a mask forward each frame the *object is visible*; it has nothing
to propagate against while the player is off-screen. So McByte (either mode)
is a short-horizon CONTINUITY tracker (crowding, brief blocked-by-another-
player occlusion), not a long-horizon RE-ID tracker. This script's --gap-report
demonstrates that empirically: it flags every track whose gap between last-seen
and a same-slot re-match exceeds the buffer, vs. cases where a new track_id
appears near where an old one vanished (a likely identity break).

For true 5s+ re-entry, layer an appearance-embedding gallery on top (this repo
already has one: GSFATeamClassifier's SigLIP embeddings) — the same approach
scripts/colab_sam2_player_tracking.py already built for SAM2 (embedding
similarity + jersey-number OCR veto over a capped gallery of recent exits).

Install:
    pip install trackers                    # lightweight mode only
    pip install "trackers[mask]"             # + SAM(vit_b) + Cutie for full mode

Usage:
    python scripts/mcbyte_tracking_test.py --video path\\to\\clip.mp4
    python scripts/mcbyte_tracking_test.py --video path\\to\\clip.mp4 --mask
    python scripts/mcbyte_tracking_test.py --video path\\to\\clip.mp4 --mask --lost-buffer-seconds 5

Outputs (next to the input video):
    <video>_mcbyte_tracks.csv       frame_idx,timestamp_s,track_id,x1,y1,x2,y2,confidence
    <video>_mcbyte_annotated.mp4    boxes + track_id labels (only with --annotate)
Printed: unique-ID count, peak CUDA memory (empirical answer to the VRAM
question for whatever GPU this runs on), and the gap report.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.detectors.player_detector import PlayerDetector  # noqa: E402


def build_tracker(use_mask: bool, lost_track_buffer: int, frame_rate: float, device: str):
    from trackers import McByteTracker

    kwargs = dict(lost_track_buffer=lost_track_buffer, frame_rate=frame_rate)

    if not use_mask:
        return McByteTracker(**kwargs)

    from trackers import McByteMaskConfig

    mask_config = McByteMaskConfig(device=device)
    return McByteTracker(enable_mask_manager=True, mask_config=mask_config, **kwargs)


def detections_to_sv(dets, frame_shape) -> sv.Detections:
    if not dets:
        return sv.Detections.empty()
    xyxy = np.array([d.bbox for d in dets], dtype=np.float32)
    confidence = np.array([d.confidence for d in dets], dtype=np.float32)
    class_id = np.zeros(len(dets), dtype=int)
    return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)


def report_gaps(track_frames: dict[int, list[int]], buffer_frames: int, fps: float) -> None:
    print(f"\n--- Gap report (lost_track_buffer = {buffer_frames} frames "
          f"= {buffer_frames / fps:.1f}s at {fps:.1f} fps) ---")
    print(f"Unique track IDs seen: {len(track_frames)}")

    events = []
    for tid, frames in track_frames.items():
        frames = sorted(frames)
        for a, b in zip(frames, frames[1:]):
            gap = b - a
            if gap > 1:
                events.append((tid, a, b, gap, gap / fps))

    if not events:
        print("No re-appearance gaps observed for any track (continuous throughout).")
        return

    events.sort(key=lambda e: -e[3])
    print(f"{len(events)} within-track gap(s) found (same ID before/after — a successful "
          f"McByte re-match, still requires IoU overlap on return):")
    for tid, a, b, gap_f, gap_s in events[:20]:
        flag = "  <= near buffer limit, re-match was close" if gap_f > 0.8 * buffer_frames else ""
        print(f"  track_id={tid:>4}  frame {a:>6} -> {b:>6}  gap={gap_f:>4}f ({gap_s:.2f}s){flag}")

    print(
        "\nNote: this only lists gaps McByte itself bridged under the SAME id. It cannot "
        "tell you about a real player who exited and came back as a NEW id — that shows up "
        "as an unrelated track_id starting near where another one permanently stopped. "
        "Cross-check visually via --annotate output for the specific re-entry you care about."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="Path to input clip")
    ap.add_argument("--mask", action="store_true", help="Enable full SAM+Cutie mask-conditioned mode")
    ap.add_argument("--lost-buffer-seconds", type=float, default=5.0,
                     help="Seconds a lost track is kept alive before deletion (default: 5.0)")
    ap.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (default: whole video)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--model-path", default=PlayerDetector.MODEL_PATH)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--annotate", action="store_true", help="Also write an annotated .mp4")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path.name}  {width}x{height} @ {fps:.2f}fps  ({total_frames} frames)")

    detector = PlayerDetector(model_path=args.model_path, conf=args.conf, device=args.device)

    lost_track_buffer = round(args.lost_buffer_seconds * 30)  # McByte's buffer unit is "frames at 30fps"
    tracker = build_tracker(args.mask, lost_track_buffer, frame_rate=fps, device=args.device)
    print(f"McByteTracker ready (mask={'ON (SAM+Cutie)' if args.mask else 'OFF (IoU-only)'}, "
          f"lost_track_buffer={lost_track_buffer} => {lost_track_buffer/30:.1f}s real time)")

    if args.device == "cuda":
        import torch
        torch.cuda.reset_peak_memory_stats()

    writer = None
    if args.annotate:
        out_path = video_path.with_name(video_path.stem + "_mcbyte_annotated.mp4")
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

    csv_path = video_path.with_name(video_path.stem + "_mcbyte_tracks.csv")
    track_frames: dict[int, list[int]] = defaultdict(list)

    t0 = time.time()
    frame_idx = 0
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "timestamp_s", "track_id", "x1", "y1", "x2", "y2", "confidence"])

        while True:
            ret, frame_bgr = cap.read()
            if not ret or (args.max_frames and frame_idx >= args.max_frames):
                break

            frame_dets = detector.detect(frame_bgr, frame_idx=frame_idx, fps=fps)
            sv_dets = detections_to_sv(frame_dets.players, frame_bgr.shape)

            # McByte's SAM/Cutie mask backends expect RGB; IoU-only mode ignores `frame`.
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if args.mask else None
            tracked = tracker.update(sv_dets, frame=frame_rgb)

            for i in range(len(tracked)):
                tid = int(tracked.tracker_id[i])
                x1, y1, x2, y2 = tracked.xyxy[i]
                conf = float(tracked.confidence[i]) if tracked.confidence is not None else -1.0
                w.writerow([frame_idx, frame_idx / fps, tid, *[round(v, 1) for v in (x1, y1, x2, y2)], conf])
                track_frames[tid].append(frame_idx)

            if writer is not None:
                labels = [f"#{int(t)}" for t in tracked.tracker_id] if len(tracked) else []
                annotated = box_annotator.annotate(frame_bgr.copy(), tracked)
                annotated = label_annotator.annotate(annotated, tracked, labels=labels)
                writer.write(annotated)

            if frame_idx % 100 == 0:
                print(f"  frame {frame_idx}/{total_frames}  active_ids={len(tracked)}  "
                      f"elapsed={time.time()-t0:.1f}s")
            frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Annotated video: {out_path}")

    elapsed = time.time() - t0
    print(f"\nDone: {frame_idx} frames in {elapsed:.1f}s ({frame_idx/max(elapsed,1e-6):.1f} fps overall)")
    print(f"Tracks CSV: {csv_path}")

    if args.device == "cuda":
        import torch
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nPeak CUDA memory: {peak_gb:.2f} GB / {total_gb:.2f} GB total on this GPU")
        if peak_gb > 0.9 * total_gb:
            print("  -> Close to the VRAM ceiling. If this OOM'd or you plan to raise "
                  "resolution/frame count, move to a bigger GPU (e.g. Colab T4, 16GB).")

    report_gaps(track_frames, lost_track_buffer, fps)


if __name__ == "__main__":
    main()
