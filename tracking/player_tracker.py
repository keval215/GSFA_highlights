"""
tracking/player_tracker.py — Player tracking for GSFA highlights.

BoT-SORT (boxmot) with:
  • Built-in ECC global motion compensation — survives fast camera pans
  • External SigLIP appearance embeddings (reused from GSFATeamClassifier)
  • Two-stage ByteTrack association (high-conf then low-conf)

Writes track_id in-place on each Detection. Detections without an
embedding are still tracked (motion-only association), but ID stability
across pans relies on having an embedding for those that do.
"""

from __future__ import annotations

from typing import List

import numpy as np
from boxmot.trackers.bbox.botsort.botsort import BotSort

from detectors.player_detector import Detection


# Track buffer is scaled internally by frame_rate/30 — we want lost tracks
# kept alive for ~2 seconds, which covers a typical pass duration but
# stops well short of long ReID territory.
_TRACK_BUFFER_FRAMES_AT_30FPS = 60


class PlayerTracker:
    """BoT-SORT wrapper that consumes external SigLIP embeddings."""

    def __init__(self, fps: float) -> None:
        self.tracker = BotSort(
            reid_model         = None,
            with_reid          = True,
            cmc_method         = "ecc",
            track_high_thresh  = 0.5,
            track_low_thresh   = 0.1,
            new_track_thresh   = 0.6,
            match_thresh       = 0.8,
            proximity_thresh   = 0.5,
            appearance_thresh  = 0.25,
            track_buffer       = _TRACK_BUFFER_FRAMES_AT_30FPS,
            frame_rate         = int(round(fps)),
            fuse_first_associate = False,
        )

    def update(self, frame: np.ndarray, players: List[Detection]) -> None:
        """Run the tracker on one frame's player detections.

        Writes track_id onto each Detection in-place. Detections whose
        SigLIP embedding is missing are still tracked (motion-only) but
        will have weaker appearance-based recovery.
        """
        if not players:
            self.tracker.update(np.empty((0, 6), dtype=np.float32), frame)
            return

        # Build Nx6 dets array: x1, y1, x2, y2, conf, cls
        dets = np.array(
            [[*p.bbox, p.confidence, 0] for p in players],
            dtype=np.float32,
        )

        emb_dim = None
        for p in players:
            if p.embedding is not None:
                emb_dim = int(p.embedding.shape[0])
                break

        if emb_dim is None:
            embs = None
        else:
            embs = np.zeros((len(players), emb_dim), dtype=np.float32)
            for i, p in enumerate(players):
                if p.embedding is not None:
                    embs[i] = p.embedding.astype(np.float32)

        out = self.tracker.update(dets, frame, embs=embs)
        out_arr = np.asarray(out)
        if out_arr.size == 0:
            return

        # boxmot rows: x1, y1, x2, y2, id, conf, cls, det_ind
        for row in out_arr:
            det_ind = int(row[7])
            if 0 <= det_ind < len(players):
                players[det_ind].track_id = int(row[4])
