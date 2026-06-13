"""
simulate/slice_match.py — ffmpeg-slice a full-match mp4 into 60 s clips.

Output layout (consumed by send_clips.py):
    <out_dir>/<half>_<minute>.mp4   e.g. clips/1_1.mp4, 1_2.mp4, ...

Usage:
    python simulate/slice_match.py --input match.mp4 --out-dir clips
    python simulate/slice_match.py --input match.mp4 --out-dir clips --half 2
    python simulate/slice_match.py --input match.mp4 --out-dir clips --copy
        (--copy uses stream copy: fast but cuts land on keyframes, so clip
         boundaries can be off by up to a GOP; default re-encodes for
         frame-exact 60 s clips)
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path


def probe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def slice_match(input_path: str, out_dir: Path, half: int, copy: bool) -> list[Path]:
    duration = probe_duration(input_path)
    n_clips  = math.ceil(duration / 60.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []

    for i in range(n_clips):
        minute = i + 1
        dest   = out_dir / f"{half}_{minute}.mp4"
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", str(i * 60), "-t", "60",
               "-i", input_path]
        if copy:
            cmd += ["-c", "copy"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-an"]
        cmd += [str(dest)]
        print(f"[slice] minute {minute}/{n_clips} → {dest}")
        subprocess.run(cmd, check=True)
        clips.append(dest)

    return clips


def main() -> None:
    ap = argparse.ArgumentParser(description="Slice a full match into 60 s clips")
    ap.add_argument("--input", required=True, help="full-match mp4")
    ap.add_argument("--out-dir", default="data/sim_clips", help="output directory")
    ap.add_argument("--half", type=int, default=1, help="half number for filenames")
    ap.add_argument("--copy", action="store_true",
                    help="stream-copy instead of re-encoding (faster, keyframe-aligned cuts)")
    args = ap.parse_args()

    if not Path(args.input).exists():
        sys.exit(f"input not found: {args.input}")
    clips = slice_match(args.input, Path(args.out_dir), args.half, args.copy)
    print(f"[slice] done — {len(clips)} clips in {args.out_dir}")


if __name__ == "__main__":
    main()
