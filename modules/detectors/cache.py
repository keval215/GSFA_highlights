"""
modules/detectors/cache.py — Shared cache path utility

All per-match pkl files are stored under data/cache/
named after the video file so each match gets its own cache.

Usage:
    from modules.detectors.cache import cache_path
    path = cache_path(r"C:/path/to/Video Project 8.mp4", "team_siglip")
    # -> data/cache/video_project_8_team_siglip.pkl
"""

from pathlib import Path

CACHE_DIR = Path(r"D:\GSFA_highlights\data\cache")


def cache_path(video_path: str, suffix: str) -> Path:
    """
    Derive a match-specific pkl path from the video filename.

    Args:
        video_path: Full path to the video file.
        suffix:     Short label for this classifier, e.g. "team_siglip",
                    "team_colour", "goalkeeper".

    Returns:
        Path like: data/cache/video_project_8_team_siglip.pkl
    """
    stem = Path(video_path).stem.lower().replace(" ", "_")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{stem}_{suffix}.pkl"
