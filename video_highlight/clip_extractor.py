"""
clip_extractor.py — Stage 3b: FFmpeg-based clip cutting with fade-out and concat.

Each goal clip: [t_goal - PRE_ROLL, t_goal + POST_ROLL]
Fade-out: 0.5s at the end of every clip (video + audio).
Final reel: all clips concatenated in chronological order.
"""

import subprocess
import sys
from pathlib import Path

PRE_ROLL  = 10.0   # seconds before the detected goal timestamp
POST_ROLL =  5.0   # seconds after
FADE_DURATION = 0.5  # seconds of fade-out at end of each clip


def _find_ffmpeg() -> str:
    """Locate ffmpeg binary — checks PATH first, then common install locations."""
    import shutil
    found = shutil.which("ffmpeg")
    if found:
        return found
    candidates = [
        r"C:\Users\Admin\miniconda3\Library\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    raise FileNotFoundError(
        "ffmpeg not found. Open a new terminal so the winget PATH update takes effect, "
        "or add ffmpeg to PATH manually."
    )

_FFMPEG_BIN = _find_ffmpeg()


def _ffmpeg(*args: str) -> subprocess.CompletedProcess:
    cmd = [_FFMPEG_BIN, "-y", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[FFmpeg ERROR]\n{result.stderr[-2000:]}", file=sys.stderr)
    return result


def extract_clip(
    video_path: str,
    t_goal: float,
    clip_index: int,
    out_dir: Path,
    pre_roll: float = PRE_ROLL,
    post_roll: float = POST_ROLL,
    fade: float = FADE_DURATION,
) -> Path | None:
    """
    Cut a single goal clip from video_path and apply a fade-out.

    Args:
        video_path: Input match video.
        t_goal:     Timestamp (seconds) when the goal was confirmed.
        clip_index: Clip number (used in filename).
        out_dir:    Directory to save the clip.
        pre_roll:   Seconds before t_goal to include.
        post_roll:  Seconds after t_goal to include.
        fade:       Fade-out duration in seconds.

    Returns:
        Path to the saved clip, or None on failure.
    """
    t_start = max(0.0, t_goal - pre_roll)
    duration = pre_roll + post_roll
    fade_start = duration - fade  # when fade-out begins within the clip

    out_path = out_dir / f"goal_{clip_index:03d}.mp4"

    r = _ffmpeg(
        "-ss", f"{t_start:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-vf", f"fade=t=out:st={fade_start:.3f}:d={fade:.3f}",
        "-af", f"afade=t=out:st={fade_start:.3f}:d={fade:.3f}",
        "-c:v", "libopenh264", "-b:v", "2M",
        "-c:a", "aac", "-b:a", "192k",
        str(out_path),
    )
    if r.returncode == 0:
        print(f"  [clip] Saved: {out_path.name}  ({t_start:.1f}s -> {t_start+duration:.1f}s)")
        return out_path
    return None


def extract_segment(
    video_path: str,
    t_start: float,
    t_end: float,
    out_path: Path,
    fade: float = FADE_DURATION,
) -> Path | None:
    """
    Cut an arbitrary [t_start, t_end] segment from video_path to out_path
    with a fade-out at the end. Used for halftime / penalty clips that
    don't fit the goal pre/post-roll model.
    """
    if t_end <= t_start:
        print(f"[segment] Bad range: {t_start} -> {t_end}")
        return None
    duration = t_end - t_start
    fade_start = max(0.0, duration - fade)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = _ffmpeg(
        "-ss", f"{t_start:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-vf", f"fade=t=out:st={fade_start:.3f}:d={fade:.3f}",
        "-af", f"afade=t=out:st={fade_start:.3f}:d={fade:.3f}",
        "-c:v", "libopenh264", "-b:v", "2M",
        "-c:a", "aac", "-b:a", "192k",
        str(out_path),
    )
    if r.returncode == 0:
        print(f"  [segment] Saved: {out_path.name}  ({t_start:.1f}s -> {t_end:.1f}s, {duration:.1f}s)")
        return out_path
    return None


def concat_clips(clip_paths: list[Path], output_path: Path) -> bool:
    """
    Concatenate clip_paths into a single highlight reel using FFmpeg concat demuxer.

    Args:
        clip_paths:  Ordered list of clip files.
        output_path: Final output MP4 path.

    Returns:
        True on success.
    """
    if not clip_paths:
        print("[concat] No clips to concatenate.")
        return False

    # Write concat manifest
    manifest = output_path.parent / "_concat_list.txt"
    manifest.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in clip_paths),
        encoding="utf-8"
    )

    r = _ffmpeg(
        "-f", "concat",
        "-safe", "0",
        "-i", str(manifest),
        "-c", "copy",
        str(output_path),
    )

    manifest.unlink(missing_ok=True)

    if r.returncode == 0:
        print(f"\n[concat] Highlight reel saved: {output_path}")
        return True
    return False
