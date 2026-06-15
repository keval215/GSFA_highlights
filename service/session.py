"""
service/session.py — MatchSession + MatchSessionManager.

A MatchSession owns all cross-clip state for one match (implementation.md,
Conflict 2): the fitted team classifier, BoT-SORT tracker, ball Kalman,
pass FSM and the processed-frame counter, all carried across clip
boundaries so a pass released at 0:59 of clip N and received at 0:01 of
clip N+1 resolves normally.

The minute-bucketing/correction logic itself lives in service/stats.py
(dependency-free, unit-tested); this module wires it to the CV classes.

Crash recovery: counters/events are durable in SQL after every clip and
the team fit pkl is on disk. Tracker/Kalman internals are NOT pickled —
on worker restart we accept one boundary discontinuity (trackers re-init,
FSM starts IDLE, at most one in-flight pass lost).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import joblib
import numpy as np

from detectors.player_detector import PlayerDetector
from team_classifier.team_classifier import GSFATeamClassifier, TeamSpec
from tracking.player_tracker import PlayerTracker
from video_analysis import possession as vp
from video_analysis.possession import (
    BallDetector,
    BallTracker,
    CarrierEngine,
    PassEventTracker,
)

from service import config, db, stats

log = logging.getLogger("gsfa.session")

# The dependency-free constants in stats.py must mirror the pipeline's.
assert stats.LBL_TEAM0 == vp.POSSESS_TEAM0
assert stats.LBL_TEAM1 == vp.POSSESS_TEAM1
assert stats.LBL_LOOSE == vp.POSSESS_LOOSE
assert stats.LBL_OOF == vp.POSSESS_OOF
assert stats.EVT_COMPLETED == vp.EVT_COMPLETED
assert stats.EVT_INTERCEPTION == vp.EVT_INTERCEPTION
assert stats.EVT_BALL_LOST == vp.EVT_BALL_LOST

_TRAVEL_PHASES = (vp.PHASE_CAND_REL, vp.PHASE_TRAVEL, vp.PHASE_CAND_RCV)


# ---------------------------------------------------------------------------
# Shared (process-wide) model bundle — loaded once, lives in VRAM
# ---------------------------------------------------------------------------

@dataclass
class ModelBundle:
    player_det: PlayerDetector
    ball_det:   BallDetector

    @staticmethod
    def load() -> "ModelBundle":
        log.info("Loading models (player=%s, ball=%s, device=%s)",
                 config.player_weights(), config.ball_weights(), config.DEVICE)
        return ModelBundle(
            player_det=PlayerDetector(
                model_path=config.player_weights(),
                device=config.DEVICE,
            ),
            ball_det=BallDetector(weights=config.ball_weights()),
        )


# ---------------------------------------------------------------------------
# Team-fit helpers (clip-1 dense fit + silhouette quality guard)
# ---------------------------------------------------------------------------

def collect_crops(clip_path: str, player_det: PlayerDetector,
                  sample_every: int) -> list[np.ndarray]:
    """Torso crops from every Nth frame of a clip, sharp + big enough only.
    Reuses GSFATeamClassifier's crop filters so the fit distribution matches
    the batch pipeline."""
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise IOError(f"collect_crops: cannot open {clip_path}")
    crops: list[np.ndarray] = []
    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fidx % sample_every == 0:
            dets = player_det.detect(frame, frame_idx=fidx)
            for p in dets.players:
                crop = GSFATeamClassifier._torso_crop(frame, p.bbox)
                if (crop.shape[0] >= 32 and crop.shape[1] >= 32
                        and GSFATeamClassifier._is_sharp(crop)):
                    crops.append(crop)
        fidx += 1
    cap.release()
    return crops


def fit_and_score(clf: GSFATeamClassifier, crops: list[np.ndarray],
                  score_sample: int = 600) -> float:
    """Fit SigLIP→UMAP→KMeans on the crops, then return the KMeans
    silhouette score computed on the UMAP projections of a subsample."""
    from sklearn.metrics import silhouette_score

    if len(crops) < 10:
        raise RuntimeError(f"fit_and_score: only {len(crops)} crops — not enough to fit")
    clf._classifier.fit(crops)
    clf._is_fitted = True

    sample = crops if len(crops) <= score_sample else [
        crops[i] for i in np.linspace(0, len(crops) - 1, score_sample, dtype=int)
    ]
    features    = clf._classifier.extract_features(sample)
    projections = clf._classifier.reducer.transform(features)
    labels      = clf._classifier.cluster_model.predict(projections)
    if len(set(labels.tolist())) < 2:
        return -1.0
    return float(silhouette_score(projections, labels))


# ---------------------------------------------------------------------------
# MatchSession
# ---------------------------------------------------------------------------

class MatchSession:
    """All cross-clip state for one match. Created by the worker on the
    first clip it sees for a match_id; the team fit is loaded from disk if
    a pkl exists (worker restart / late clip after eviction)."""

    def __init__(self, match_id: str, models: ModelBundle) -> None:
        self.match_id = match_id
        self.models   = models

        self.state_dir = config.MATCH_STATE_DIR / match_id
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.fit_pkl_path = self.state_dir / "team_siglip.pkl"

        self.team_clf: Optional[GSFATeamClassifier] = None
        self.fit_status: str = "pending"           # pending | ok | refit | failed
        self._fit_crops_clip1: list[np.ndarray] = []  # kept only while status == refit

        # Cross-clip CV state (Conflict 2). PlayerTracker frame rate uses the
        # processed-frame rate because it only ever sees strided frames.
        self.tracker      = PlayerTracker(fps=config.TARGET_PROCESS_FPS,
                                          cmc_method=config.CMC_METHOD)
        self.ball_tracker = BallTracker()
        self.carrier_eng  = CarrierEngine()
        self.pass_track   = PassEventTracker()

        self.proc_idx        = 0    # processed-frame counter, session-global
        self.n_events_seen   = 0    # how many pass_track.events already written
        self.last_written: Optional[tuple[int, int]] = None  # (half, minute)
        # Travel frames pending when the previous clip ended — the prior-minute
        # share of the next retroactive adjustment (Conflict 3 split).
        self.carryover_travel_frames = 0

        self.last_touched = time.monotonic()

        if self.fit_pkl_path.exists():
            self.team_clf   = joblib.load(self.fit_pkl_path)
            self.fit_status = "ok"
            log.info("[%s] team fit loaded from %s", match_id, self.fit_pkl_path)

    # ------------------------------------------------------------------
    # Team fit (clip 1, dense sampling, quality guard)
    # ------------------------------------------------------------------

    def ensure_fit(self, clip_path: str) -> str:
        """Fit on this clip if not fitted yet; refit on clip 2 with combined
        samples if clip-1 quality was below threshold. Returns fit_status.
        No refits after the fit is committed — re-running KMeans mid-match
        can swap team 0/1 labels and corrupt stats."""
        self.last_touched = time.monotonic()

        if self.fit_status == "ok":
            return self.fit_status

        crops = collect_crops(clip_path, self.models.player_det, config.FIT_SAMPLE_EVERY)
        if self.fit_status == "refit":
            crops = self._fit_crops_clip1 + crops

        clf = GSFATeamClassifier(device=config.DEVICE)
        try:
            score = fit_and_score(clf, crops)
        except RuntimeError as exc:
            log.error("[%s] team fit failed: %s", self.match_id, exc)
            self.fit_status = "failed"
            raise

        if score >= config.FIT_SILHOUETTE_MIN:
            self.team_clf   = clf
            self.fit_status = "ok"
            self._fit_crops_clip1 = []
            self._resolve_team_names(clf, crops)
            joblib.dump(clf, self.fit_pkl_path)
            log.info("[%s] team fit ok (silhouette=%.3f, crops=%d) → %s",
                     self.match_id, score, len(crops), self.fit_pkl_path)
        elif self.fit_status == "pending":
            # Below threshold on clip 1 — keep the samples, redo on clip 2.
            # Use this (low-quality) fit for clip 1 so its stats are not lost.
            self.team_clf   = clf
            self.fit_status = "refit"
            self._fit_crops_clip1 = crops
            log.warning("[%s] team fit below threshold (silhouette=%.3f) — "
                        "will refit on clip 2 with combined samples",
                        self.match_id, score)
        else:
            # Combined clip-1+2 refit still poor — commit it anyway and move
            # on (no later refits), but record the quality problem.
            self.team_clf   = clf
            self.fit_status = "ok"
            self._fit_crops_clip1 = []
            self._resolve_team_names(clf, crops)
            joblib.dump(clf, self.fit_pkl_path)
            log.error("[%s] refit still below threshold (silhouette=%.3f) — "
                      "committing anyway", self.match_id, score)
        return self.fit_status

    def _resolve_team_names(self, clf: GSFATeamClassifier,
                            crops: list[np.ndarray]) -> None:
        """Map the two clusters to the caller-supplied team names by jersey
        colour. Runs once, when the fit is committed; the result is pickled
        with clf so later clips only look it up. No-op (team_id_to_name stays
        None → payload uses team0/team1) if the caller did not supply both
        names and both colours, or if colour resolution fails."""
        conn = db.get_conn()
        try:
            raw = db.get_team_specs(conn, self.match_id)
        finally:
            conn.close()
        if not raw:
            return
        specs = [TeamSpec(name=n, colour=c) for n, c in raw]
        try:
            mapping = clf.resolve_team_names(specs, crops)
            log.info("[%s] team colours resolved → %s", self.match_id, mapping)
        except (ValueError, RuntimeError) as exc:
            log.warning("[%s] team colour resolution failed (%s) — "
                        "falling back to team0/team1 labels", self.match_id, exc)

    # ------------------------------------------------------------------
    # Per-clip processing hooks (called by clip_processor)
    # ------------------------------------------------------------------

    def split_adjustment(self, n: int) -> tuple[int, int]:
        """Split an adjustment of n provisional frames into
        (prior_minute_frames, current_minute_frames), consuming the
        boundary carryover."""
        prior, current, self.carryover_travel_frames = stats.split_adjustment(
            n, self.carryover_travel_frames
        )
        return prior, current

    def finish_clip(self, half: int, minute: int) -> None:
        """Record boundary state after a clip is fully processed."""
        self.last_written = (half, minute)
        if self.pass_track.phase in _TRAVEL_PHASES:
            self.carryover_travel_frames = self.pass_track._travel_frames_so_far
        else:
            self.carryover_travel_frames = 0
        self.last_touched = time.monotonic()

    def new_events(self) -> list:
        """Pass events resolved since the last call (i.e. during this clip)."""
        evts = self.pass_track.events[self.n_events_seen:]
        self.n_events_seen = len(self.pass_track.events)
        return evts


# ---------------------------------------------------------------------------
# MatchSessionManager
# ---------------------------------------------------------------------------

class MatchSessionManager:
    """dict[match_id → MatchSession] with idle eviction. One worker process
    ⇒ trivial memory at ~1 concurrent match."""

    def __init__(self, models: ModelBundle) -> None:
        self.models = models
        self._sessions: dict[str, MatchSession] = {}

    def get_or_create(self, match_id: str) -> MatchSession:
        sess = self._sessions.get(match_id)
        if sess is None:
            sess = MatchSession(match_id, self.models)
            self._sessions[match_id] = sess
        sess.last_touched = time.monotonic()
        return sess

    def evict_idle(self) -> None:
        now = time.monotonic()
        for mid in [m for m, s in self._sessions.items()
                    if now - s.last_touched > config.SESSION_IDLE_EVICT_S]:
            log.info("[%s] evicting idle session (fit pkl stays on disk)", mid)
            del self._sessions[mid]

    @property
    def active_matches(self) -> list[str]:
        return list(self._sessions.keys())
