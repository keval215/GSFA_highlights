"""
tracking/player_tracker.py — Phase 1 player tracking for GSFA highlights.

Wraps deep-sort-realtime's DeepSort with externally-supplied SigLIP
appearance embeddings (already computed by GSFATeamClassifier — no second
forward pass). Writes Detection.track_id in-place on each player.

Detections without an embedding (crops too small to embed) are skipped;
they keep track_id=None.
"""

from __future__ import annotations

from typing import List

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from detectors.player_detector import Detection


def _iou(a: tuple[int, int, int, int], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class PlayerTracker:
    """DeepSORT wrapper that reuses SigLIP embeddings written onto Detection."""

    def __init__(self, fps: float):
        # Short max_age — futsal direction changes make stale Kalman predictions
        # risky. Lean on appearance instead via tight max_cosine_distance.
        self.ds = DeepSort(
            max_age=max(1, int(0.5 * fps)),
            n_init=2,
            max_iou_distance=0.7,
            max_cosine_distance=0.2,
            nn_budget=30,
            embedder=None,   # we supply external SigLIP embeds
        )

    def update(self, frame: np.ndarray, players: List[Detection]) -> None:
        """Assign track_id to each player in-place. Players without an
        embedding are skipped (track_id stays None)."""
        tracked = [p for p in players if p.embedding is not None]
        if not tracked:
            # Still need to tick the tracker so max_age accounting advances.
            self.ds.update_tracks([], embeds=None, frame=frame)
            return

        raw_dets = []
        for p in tracked:
            x1, y1, x2, y2 = p.bbox
            raw_dets.append(
                ([float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                 float(p.confidence),
                 str(p.team_id) if p.team_id is not None else "?")
            )
        embeds = np.stack([p.embedding for p in tracked]).astype(np.float32)

        tracks = self.ds.update_tracks(raw_dets, embeds=embeds, frame=frame)

        # Map confirmed tracks back to input detections via IoU.
        confirmed = [t for t in tracks if t.is_confirmed()]
        if not confirmed:
            return

        used = set()
        for t in confirmed:
            l, t_, r, b = t.to_ltrb()
            best_iou = 0.3   # minimum overlap to accept the mapping
            best_idx = -1
            for i, p in enumerate(tracked):
                if i in used:
                    continue
                score = _iou(p.bbox, (l, t_, r, b))
                if score > best_iou:
                    best_iou = score
                    best_idx = i
            if best_idx >= 0:
                tracked[best_idx].track_id = int(t.track_id)
                used.add(best_idx)
