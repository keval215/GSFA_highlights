"""
simulate/send_clips.py — POST sliced clips to the ingestion API.

Reads <clips_dir>/<half>_<minute>.mp4 files (from slice_match.py), sorts
them by (half, minute) and POSTs them to /api/clips.

Usage:
    # real-time rehearsal — one clip per minute, like a live match:
    python simulate/send_clips.py --api http://VM_IP:8000 --clips-dir clips \
        --match-id sim_match_01 --interval 60

    # fast-forward — next POST as soon as the 202 arrives:
    python simulate/send_clips.py --api http://VM_IP:8000 --clips-dir clips \
        --match-id sim_match_01 --fast-forward
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import requests

_CLIP_RE = re.compile(r"^(\d+)_(\d+)\.mp4$")


def find_clips(clips_dir: Path) -> list[tuple[int, int, Path]]:
    clips = []
    for f in clips_dir.iterdir():
        m = _CLIP_RE.match(f.name)
        if m:
            clips.append((int(m.group(1)), int(m.group(2)), f))
    return sorted(clips)


def send(api: str, match_id: str, half: int, minute: int, path: Path,
         team0: str | None, team1: str | None, venue: str | None) -> dict:
    data = {"match_id": match_id, "half": str(half), "minute": str(minute)}
    if team0: data["team0_name"] = team0
    if team1: data["team1_name"] = team1
    if venue: data["venue_id"]   = venue
    with open(path, "rb") as f:
        resp = requests.post(
            f"{api.rstrip('/')}/api/clips",
            data=data,
            files={"file": (path.name, f, "video/mp4")},
            timeout=120,
        )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    ap = argparse.ArgumentParser(description="POST 60 s clips to /api/clips")
    ap.add_argument("--api", required=True, help="e.g. http://1.2.3.4:8000")
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--interval", type=float, default=None,
                    help="seconds between POSTs (60 = real-time rehearsal)")
    ap.add_argument("--fast-forward", action="store_true",
                    help="next POST as soon as the 202 arrives")
    ap.add_argument("--team0", default=None)
    ap.add_argument("--team1", default=None)
    ap.add_argument("--venue", default=None)
    args = ap.parse_args()

    if not args.fast_forward and args.interval is None:
        sys.exit("specify --interval SECONDS or --fast-forward")

    clips = find_clips(Path(args.clips_dir))
    if not clips:
        sys.exit(f"no <half>_<minute>.mp4 clips found in {args.clips_dir}")

    print(f"[send] {len(clips)} clips → {args.api} (match_id={args.match_id})")
    for i, (half, minute, path) in enumerate(clips):
        t0 = time.monotonic()
        body = send(args.api, args.match_id, half, minute, path,
                    args.team0, args.team1, args.venue)
        dup = "  (duplicate)" if body.get("duplicate") else ""
        print(f"[send] h{half} m{minute} → 202 in {time.monotonic()-t0:.1f}s{dup}")

        if not args.fast_forward and i < len(clips) - 1:
            wait = max(0.0, args.interval - (time.monotonic() - t0))
            time.sleep(wait)

    print("[send] done")


if __name__ == "__main__":
    main()
