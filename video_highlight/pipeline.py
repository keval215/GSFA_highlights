"""
pipeline.py — Full Stages 1-3 highlight pipeline.

Stage 1: Scoreboard ROI localization (hardcoded normalized coords, fixed overlay)
Stage 2: EasyOCR score reading at 2 fps with stability filter + endgame state machine
Stage 3: Goal/halftime/penalties clip extraction -> chronological highlight reel

Importable:
    from video_highlight.pipeline import run_pipeline
    result = run_pipeline(input_video=..., out_dir=..., output_path=..., match=...)

CLI:
    python video_highlight/pipeline.py --input match.mp4 --match demo
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

import cv2

# Local imports — keep CLI-style fallback for running this file directly
try:
    from ocr_reader import ScoreReader
    from endgame_detector import EndgameDetector, PENALTY_STABLE_SEC, S_PENALTIES_ACTIVE
    from clip_extractor import extract_clip, extract_segment, concat_clips
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from ocr_reader import ScoreReader
    from endgame_detector import EndgameDetector, PENALTY_STABLE_SEC, S_PENALTIES_ACTIVE
    from clip_extractor import extract_clip, extract_segment, concat_clips


def run_pipeline(
    input_video: str | Path,
    out_dir: str | Path,
    output_path: str | Path,
    match: str = "match",
    fps: float = 2.0,
    endgame_fps: float = 1.0,
    pre: float = 15.0,
    post: float = 5.0,
    confirm: int = 3,
    max_goals: int = 25,
    use_gpu: bool = False,
    progress_cb: Callable[[str, float], None] | None = None,
) -> dict:
    """
    Run the full highlight pipeline on `input_video` and write the concatenated
    reel to `output_path`. Returns a dict describing what was detected.

    Args:
        input_video : path to the source mp4
        out_dir     : directory for per-clip mp4s (goals + halftime + penalties)
        output_path : path for the final concatenated highlight reel mp4
        match       : prefix used in halftime/penalty clip filenames
        fps         : score-OCR sample rate (default 2 fps)
        endgame_fps : endgame-OCR sample rate (default 1 fps)
        pre, post   : seconds before/after each goal to include in clips
        confirm     : consecutive samples a score change must persist
        max_goals   : abort if more than this many goals detected (sanity cap)
        use_gpu     : enable GPU for EasyOCR (requires CUDA + GPU build of torch)
        progress_cb : optional callback(stage: str, fraction: float in [0,1])

    Returns:
        {
          "goals":         [...goal dicts...],
          "endgame_clips": [...endgame clip dicts...],
          "output_path":   Path to the final reel (or None if no clips produced),
          "total_secs":    video duration in seconds,
        }
    """
    input_video = Path(input_video)
    out_dir = Path(out_dir)
    output_path = Path(output_path)

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _progress(stage: str, frac: float) -> None:
        if progress_cb is not None:
            try:
                progress_cb(stage, max(0.0, min(1.0, frac)))
            except Exception:
                pass  # never let a callback break the pipeline

    # --- Stage 1: load OCR + endgame detector ---
    _progress("init", 0.0)
    print("[Stage 1] Initialising EasyOCR score reader + endgame detector...")
    reader = ScoreReader(gpu=use_gpu)
    endgame = EndgameDetector(ocr=reader._ocr)
    print("[Stage 1] Ready.\n")

    # --- Stage 2: scan ---
    print(f"[Stage 2] Scanning: {input_video.name}")
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_secs = total_frames / video_fps
    stride = max(1, int(video_fps / fps))
    endgame_stride = max(1, int(video_fps / endgame_fps))

    print(f"         Video: {total_frames} frames @ {video_fps:.1f} fps "
          f"({total_secs/60:.1f} min)")
    print(f"         Score OCR sample rate  : every {stride} frames (~{fps} fps)")
    print(f"         Endgame OCR sample rate: every {endgame_stride} frames (~{endgame_fps} fps)")
    print(f"         Stability filter: {confirm} consecutive matches required\n")

    endgame_clips: list[dict] = []
    goals: list[dict] = []

    home_score: int | None = None
    away_score: int | None = None
    pending_home: int | None = None
    pending_away: int | None = None
    pending_count = 0
    pending_frame = 0
    pending_large_jump = False
    recovery_home: int | None = None
    recovery_away: int | None = None
    recovery_count = 0
    RECOVERY_THRESHOLD = 6  # ~3s at 2 fps

    def _handle_endgame_events(events: list[dict]) -> None:
        for ev in events:
            t = ev.get("t")
            mm, ss = divmod(int(t or 0), 60)
            if ev["type"] == "halftime":
                t_end = min(total_secs, t + 30.0)
                endgame_clips.append({
                    "type": "halftime", "t_start": t, "t_end": t_end,
                    "label": "HALFTIME",
                })
                print(f"  [HALFTIME]  detected at {mm}m{ss:02d}s -> save 30s clip ({t:.1f}-{t_end:.1f}s)")
            elif ev["type"] == "fulltime":
                print(f"  [FULLTIME]  detected at {mm}m{ss:02d}s -> penalties now eligible")
            elif ev["type"] == "penalties_start":
                print(f"  [PENALTIES] started at {mm}m{ss:02d}s  score={ev.get('score')}")
            elif ev["type"] == "penalties_score":
                print(f"  [PENALTIES] score change at {mm}m{ss:02d}s -> {ev.get('score')}")
            elif ev["type"] == "penalties_end":
                t0 = ev["t_start"]; t1 = ev["t_end"]
                if ev.get("reason") == "video_ended":
                    t_clip_end = t1
                else:
                    t_clip_end = max(t0 + 1.0, t1 - PENALTY_STABLE_SEC * 0.5)
                endgame_clips.append({
                    "type": "penalties", "t_start": t0, "t_end": t_clip_end,
                    "label": f"PENALTIES {ev.get('final_score')}",
                })
                mm0, ss0 = divmod(int(t0), 60); mm1, ss1 = divmod(int(t_clip_end), 60)
                print(f"  [PENALTIES] ended -> clip {mm0}m{ss0:02d}s..{mm1}m{ss1:02d}s  final={ev.get('final_score')}")

    frame_idx = 0
    scan_start = time.time()
    last_progress_report = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Progress reporting (cheap, every ~2s of real time)
        now = time.time()
        if progress_cb is not None and (now - last_progress_report) > 1.0:
            _progress("scanning", frame_idx / max(1, total_frames))
            last_progress_report = now

        if frame_idx % endgame_stride == 0:
            ts = frame_idx / video_fps
            _handle_endgame_events(endgame.update(frame, ts))

        if frame_idx % stride == 0:
            ts = frame_idx / video_fps

            # Skip goal-score OCR while penalty shootout is active.
            if endgame.state == S_PENALTIES_ACTIVE:
                frame_idx += 1
                continue

            h, a = reader.read(frame)

            if h is None and a is None:
                frame_idx += 1
                continue

            # Baseline requires BOTH scores readable to avoid bogus 0-None -> 0-0 goals.
            if home_score is None:
                if h is None or a is None:
                    frame_idx += 1
                    continue
                home_score, away_score = h, a
                print(f"  [init]  Starting score: {home_score}-{away_score}")
                frame_idx += 1
                continue

            h = h if h is not None else home_score
            a = a if a is not None else away_score
            score_changed = (h != home_score) or (a != away_score)

            if score_changed:
                h_delta = (h or 0) - (home_score or 0)
                a_delta = (a or 0) - (away_score or 0)

                if h_delta < 0 or a_delta < 0:
                    same_recovery = (h == recovery_home and a == recovery_away)
                    if same_recovery:
                        recovery_count += 1
                    else:
                        recovery_home, recovery_away, recovery_count = h, a, 1

                    if recovery_count >= RECOVERY_THRESHOLD:
                        mm, ss = divmod(int(ts), 60)
                        print(f"  [RECOVER]  state was poisoned {home_score}-{away_score}, "
                              f"correcting to {h}-{a} at {mm}m{ss:02d}s")
                        goals.append({
                            "index": len(goals) + 1,
                            "timestamp": ts,
                            "score_before": "?",
                            "score_after": f"{h}-{a}",
                            "rapid": True,
                        })
                        home_score, away_score = h, a
                        recovery_count = 0
                        pending_count = 0
                    frame_idx += 1
                    continue

                recovery_count = 0

                if h_delta > 4 or a_delta > 4:
                    pending_count = 0
                    frame_idx += 1
                    continue

                is_large_jump = (h_delta > 1 or a_delta > 1)

                same_candidate = (h == pending_home and a == pending_away)
                if same_candidate:
                    pending_count += 1
                else:
                    pending_home = h
                    pending_away = a
                    pending_count = 1
                    pending_frame = frame_idx
                    pending_large_jump = is_large_jump

                if pending_count >= confirm:
                    goal_ts = pending_frame / video_fps
                    goals.append({
                        "index": len(goals) + 1,
                        "timestamp": goal_ts,
                        "score_before": f"{home_score}-{away_score}",
                        "score_after": f"{h}-{a}",
                        "rapid": pending_large_jump,
                    })
                    mm, ss = divmod(int(goal_ts), 60)
                    tag = " RAPID" if pending_large_jump else ""
                    print(f"  [GOAL #{len(goals)}{tag}]  {home_score}-{away_score} -> {h}-{a}  "
                          f"at {mm}m{ss:02d}s  ({goal_ts:.1f}s)")

                    home_score, away_score = h, a
                    pending_count = 0
                    pending_large_jump = False

                    if len(goals) >= max_goals:
                        print(f"[WARNING] Reached max-goals cap ({max_goals}). Stopping scan.")
                        break
            else:
                pending_count = 0

        frame_idx += 1

    cap.release()
    _handle_endgame_events(endgame.finalize(total_secs))
    _progress("scanning", 1.0)

    print(f"\n[Stage 2] Scan complete. {len(goals)} goal(s) and "
          f"{len(endgame_clips)} endgame clip(s) detected in "
          f"{total_secs/60:.1f} min of footage.\n")

    result: dict = {
        "goals": goals,
        "endgame_clips": endgame_clips,
        "output_path": None,
        "total_secs": total_secs,
    }

    if not goals and not endgame_clips:
        print("[INFO] No score changes or endgame events detected.")
        return result

    # --- Stage 3: extract + concat ---
    _progress("extracting", 0.0)
    print(f"[Stage 3] Extracting {len(goals)} goal clip(s) + {len(endgame_clips)} endgame clip(s)...")
    timeline: list[tuple[float, Path]] = []

    total_clips = len(goals) + len(endgame_clips)
    done = 0

    for goal in goals:
        path = extract_clip(
            video_path=str(input_video),
            t_goal=goal["timestamp"],
            clip_index=goal["index"],
            out_dir=out_dir,
            pre_roll=pre,
            post_roll=post,
        )
        if path:
            timeline.append((goal["timestamp"], path))
        done += 1
        _progress("extracting", done / max(1, total_clips))

    for ec in endgame_clips:
        fname = f"{match}_{ec['type']}.mp4"
        path = extract_segment(
            video_path=str(input_video),
            t_start=ec["t_start"],
            t_end=ec["t_end"],
            out_path=out_dir / fname,
        )
        if path:
            timeline.append((ec["t_start"], path))
        done += 1
        _progress("extracting", done / max(1, total_clips))

    timeline.sort(key=lambda x: x[0])
    clip_paths = [p for _, p in timeline]

    print(f"\n[Stage 3] {len(clip_paths)} clip(s) ready.")
    print("[Stage 3] Concatenating into highlight reel (chronological)...")
    _progress("concatenating", 0.5)
    concat_clips(clip_paths, output_path)
    _progress("done", 1.0)

    result["output_path"] = output_path

    # Summary
    print("\n" + "=" * 60)
    print("  HIGHLIGHT PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Input video : {input_video}")
    print(f"  Goals found    : {len(goals)}")
    for g in goals:
        print(f"    #{g['index']:02d}  {g['score_before']} -> {g['score_after']}  "
              f"at {g['timestamp']/60:.2f} min")
    print(f"  Endgame clips  : {len(endgame_clips)}")
    for ec in endgame_clips:
        print(f"    {ec['label']:<18}  {ec['t_start']/60:.2f} -> {ec['t_end']/60:.2f} min")
    print(f"  Clips saved : {out_dir}/")
    print(f"  Reel output : {output_path}")
    print(f"  Elapsed     : {time.time()-scan_start:.1f}s")
    print("=" * 60)

    return result


def _main_cli() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Futsal highlight generator (Stages 1-3)")
    p.add_argument("--input", default="data/videos/test.mp4")
    p.add_argument("--out-dir", default="clips")
    p.add_argument("--output", default="highlights.mp4")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--pre", type=float, default=15.0)
    p.add_argument("--post", type=float, default=5.0)
    p.add_argument("--confirm", type=int, default=3)
    p.add_argument("--max-goals", type=int, default=25)
    p.add_argument("--match", default="match")
    p.add_argument("--endgame-fps", type=float, default=1.0)
    p.add_argument("--gpu", action="store_true", help="Enable GPU for EasyOCR")
    args = p.parse_args()

    try:
        run_pipeline(
            input_video=args.input,
            out_dir=args.out_dir,
            output_path=args.output,
            match=args.match,
            fps=args.fps,
            endgame_fps=args.endgame_fps,
            pre=args.pre,
            post=args.post,
            confirm=args.confirm,
            max_goals=args.max_goals,
            use_gpu=args.gpu,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main_cli())
