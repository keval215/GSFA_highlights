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

from modules.detectors.player_detector import Detection


# SigLIP embedding width produced by GSFATeamClassifier (mean of SigLIP's
# last_hidden_state; google/siglip-base-patch16-224 → hidden_size=768). Used as
# the fallback embs width on frames where NO detection carried an embedding, so
# boxmot still receives a real (all-zeros) appearance matrix instead of None.
_SIGLIP_EMB_DIM = 768


class PlayerTracker:
    """BoT-SORT wrapper that consumes external SigLIP embeddings.

    All BoT-SORT association thresholds below default to today's futsal-tuned
    values but are real constructor parameters — a ruleset config supplies
    its own values explicitly (e.g. classic football's larger roster means
    more simultaneous tracks/occlusion, so these are prime candidates to
    differ per sport).
    """

    def __init__(
        self,
        fps: float,
        cmc_method: str = "ecc",
        *,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
        track_buffer_frames_at_30fps: int = 60,
    ) -> None:
        self.tracker = BotSort(
            reid_model         = None,
            with_reid          = True,
            cmc_method         = cmc_method,
            track_high_thresh  = track_high_thresh,
            track_low_thresh   = track_low_thresh,
            new_track_thresh   = new_track_thresh,
            match_thresh       = match_thresh,
            proximity_thresh   = proximity_thresh,
            appearance_thresh  = appearance_thresh,
            # Track buffer is scaled internally by frame_rate/30 — the
            # default keeps lost tracks alive for ~2 seconds, which covers a
            # typical pass duration but stops well short of long ReID territory.
            track_buffer       = track_buffer_frames_at_30fps,
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
            # Pass an empty (not None) embs array so boxmot never falls back to
            # its internal ReID model (which is None — we feed external SigLIP
            # embeddings instead). embs=None on this path is what crashed with
            # 'NoneType' object has no attribute 'get_features'.
            self.tracker.update(
                np.empty((0, 6), dtype=np.float32),
                frame,
                embs=np.empty((0, 1), dtype=np.float32),
            )
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

        # When NO detection in this frame carried a SigLIP embedding, fall back
        # to the known SigLIP width and still hand boxmot an all-zeros embs
        # matrix rather than None. embs=None makes boxmot reach for its internal
        # ReID model (we built it with reid_model=None) → 'NoneType' object has
        # no attribute 'get_features'. All-zero appearance vectors give a
        # constant appearance distance, so the frame degrades to motion-only
        # association — the documented behaviour for embedding-less detections.
        if emb_dim is None:
            emb_dim = _SIGLIP_EMB_DIM
        embs = np.zeros((len(players), emb_dim), dtype=np.float32)
        for i, p in enumerate(players):
            if p.embedding is not None:
                embs[i] = p.embedding.astype(np.float32)

        out = self.tracker.update(dets, frame, embs=embs)
        out_arr = np.asarray(out)
        if out_arr.size == 0:
            return

        # boxmot rows: x1, y1, x2, y2, id, conf, cls, det_ind
        # row[0:4] is BoT-SORT's Kalman-smoothed box for the track — stash it on
        # the Detection so drawing can use the steady box instead of the raw
        # per-frame YOLO bbox (which wobbles). Raw .bbox is left untouched so
        # foot-zone / carrier geometry is unchanged.
        for row in out_arr:
            det_ind = int(row[7])
            if 0 <= det_ind < len(players):
                players[det_ind].track_id      = int(row[4])
                players[det_ind].smoothed_bbox = (
                    int(row[0]), int(row[1]), int(row[2]), int(row[3]))
