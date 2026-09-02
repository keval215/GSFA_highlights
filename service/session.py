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

import json
import logging
import time
from typing import Optional

import cv2
import joblib
import numpy as np

from modules.detectors.goalkeeper_detector import GoalkeeperDetector
from modules.detectors.player_detector import PlayerDetector
from modules.team_classifier.team_classifier import GSFATeamClassifier, TeamSpec
from modules.tracking.player_tracker import PlayerTracker
from modules.possession import labels as vp_labels
from modules.possession.pass_event_tracker import (
    PHASE_CAND_REL,
    PHASE_CAND_RCV,
    PHASE_TRAVEL,
)
from modules.possession import (
    BallTracker,
    CarrierEngine,
    PassEventTracker,
)
from rulesets import RulesetConfig

from service import config, db, stats

log = logging.getLogger("gsfa.session")

# The dependency-free constants in stats.py must mirror the pipeline's.
assert stats.LBL_TEAM_A == vp_labels.POSSESS_TEAM_A
assert stats.LBL_TEAM_B == vp_labels.POSSESS_TEAM_B
assert stats.LBL_LOOSE == vp_labels.POSSESS_LOOSE
assert stats.LBL_OOF == vp_labels.POSSESS_OOF
assert stats.EVT_COMPLETED == vp_labels.EVT_COMPLETED
assert stats.EVT_INTERCEPTION == vp_labels.EVT_INTERCEPTION
assert stats.EVT_BALL_LOST == vp_labels.EVT_BALL_LOST

_TRAVEL_PHASES = (PHASE_CAND_REL, PHASE_TRAVEL, PHASE_CAND_RCV)


# ---------------------------------------------------------------------------
# Shared (process-wide) model bundle — lives in VRAM
# ---------------------------------------------------------------------------

class ModelBundle:
    """One PlayerDetector per ruleset actually in use, loaded lazily on
    first match of that ruleset (not all rulesets need VRAM if only one is
    ever requested on a given deployment)."""

    def __init__(self) -> None:
        self._player_dets: dict[str, PlayerDetector] = {}

    def player_detector(self, ruleset: RulesetConfig) -> PlayerDetector:
        det = self._player_dets.get(ruleset.name)
        if det is None:
            weights = config.player_weights(ruleset.name)
            log.info("Loading player model for ruleset=%s (weights=%s, device=%s)",
                     ruleset.name, weights, config.DEVICE)
            det = PlayerDetector(
                model_path=weights,
                device=config.DEVICE,
                player_conf=ruleset.player_conf,
                ball_conf=ruleset.ball_conf,
                class_names=ruleset.class_names,
            )
            self._player_dets[ruleset.name] = det
        return det


# ---------------------------------------------------------------------------
# Team-fit helpers (clip-1 dense fit + silhouette quality guard)
# ---------------------------------------------------------------------------

def collect_crops(clip_path: str, player_det: PlayerDetector,
                  sample_every: int, ruleset: RulesetConfig) -> list[np.ndarray]:
    """Torso crops from every Nth frame of a clip, sharp + big enough only.
    Reuses GSFATeamClassifier's crop filters (parametrized by the match's
    ruleset) so the fit distribution matches the batch pipeline."""
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
                crop = GSFATeamClassifier._torso_crop(frame, p.bbox, ruleset.torso_ratio)
                if (crop.shape[0] >= ruleset.min_crop_px and crop.shape[1] >= ruleset.min_crop_px
                        and GSFATeamClassifier._is_sharp(crop, ruleset.blur_threshold)):
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

    def __init__(self, match_id: str, models: ModelBundle, ruleset: RulesetConfig) -> None:
        self.match_id = match_id
        self.models   = models
        self.ruleset  = ruleset
        self.player_det = models.player_detector(ruleset)

        self.state_dir = config.MATCH_STATE_DIR / match_id
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.fit_pkl_path  = self.state_dir / "team_siglip.pkl"
        self.fit_meta_path = self.state_dir / "fit_meta.json"

        self.team_clf: Optional[GSFATeamClassifier] = None
        self.fit_status: str = "pending"           # pending | ok | refit | failed
        self._fit_crops_clip1: list[np.ndarray] = []  # kept only while status == refit

        # Which fit_generation (matches.fit_generation) this session's current
        # fit was made under, and which CV cluster id (0/1) resolved to
        # team_a_name/team_a_colour under that fit — see _resolve_team_names /
        # reset_for_new_generation. Both are persisted in fit_meta_path
        # alongside the pkl so a disk-reloaded session (worker restart, or a
        # late clip after idle eviction) recovers them instead of silently
        # reverting to generation 1 / raw cluster order.
        self.fit_generation: int = 1
        self.team_a_cluster_id: Optional[int] = None

        self.gk_det: Optional[GoalkeeperDetector] = None
        self._gk_colour_invalid = False   # set once if the DB colours fail to parse
        # No fit stage — GoalkeeperDetector just needs the two reference
        # colours (known constants), so construction is cheap and retried
        # every clip via ensure_gk_ready() until the DB has both colours.

        # Cross-clip CV state (Conflict 2), all parametrized by the match's
        # ruleset. PlayerTracker frame rate uses the processed-frame rate
        # because it only ever sees strided frames.
        self.tracker      = PlayerTracker(
            fps                           = config.TARGET_PROCESS_FPS,
            cmc_method                    = config.CMC_METHOD,
            track_high_thresh             = ruleset.track_high_thresh,
            track_low_thresh              = ruleset.track_low_thresh,
            new_track_thresh              = ruleset.new_track_thresh,
            match_thresh                  = ruleset.match_thresh,
            proximity_thresh              = ruleset.proximity_thresh,
            appearance_thresh             = ruleset.appearance_thresh,
            track_buffer_frames_at_30fps  = ruleset.track_buffer_frames_at_30fps,
        )
        self.ball_tracker = BallTracker(
            coast_frames = ruleset.kalman_coast_frames,
            gate_sigma   = ruleset.kalman_gate_sigma,
        )
        self.carrier_eng  = CarrierEngine(
            hysteresis_n     = ruleset.carrier_hysteresis_n,
            foot_zone_ratio  = ruleset.foot_zone_ratio,
            foot_zone_min_px = ruleset.foot_zone_min_px,
            foot_zone_max_px = ruleset.foot_zone_max_px,
        )
        self.pass_track   = PassEventTracker(
            release_sustain  = ruleset.release_sustain,
            reception_settle = ruleset.reception_settle,
            travel_min_gap   = ruleset.travel_min_gap,
            travel_timeout   = ruleset.travel_timeout_frames,
        )

        self.proc_idx        = 0    # processed-frame counter, session-global
        self.n_events_seen   = 0    # how many pass_track.events already written
        self.last_written: Optional[tuple[int, int]] = None  # (half, minute)
        # team_a_cluster_id AS IT WAS when the last_written row was actually
        # written — NOT necessarily today's team_a_cluster_id. A boundary
        # PriorCorrection targets that row, so it must be reoriented against
        # the orientation that row was written under, not the current clip's
        # (they can differ, e.g. clip 1 commits before colour resolution
        # succeeds — team_a_cluster_id is still None — and clip 2's combined
        # refit resolves it, possibly to the other cluster). See
        # clip_processor.process_clip / stats.orient_for_team_a.
        self.last_written_team_a_cluster_id: Optional[int] = None
        # Travel frames pending when the previous clip ended — the prior-minute
        # share of the next retroactive adjustment (Conflict 3 split).
        self.carryover_travel_frames = 0

        self.last_touched = time.monotonic()

        if self.fit_pkl_path.exists():
            # Route through GSFATeamClassifier.load (not raw joblib.load) so
            # _use_fast_processor() re-creates the fast SigLIP processor and
            # overrides the slow one pickled into the pkl. Raw joblib.load
            # bypasses that and keeps the slow PIL processor → ~90ms/frame.
            self.team_clf   = GSFATeamClassifier.load(self.fit_pkl_path, progress=False)
            self.fit_status = "ok"
            self._load_fit_meta()
            log.info("[%s] team fit loaded from %s (fit_generation=%d)",
                     match_id, self.fit_pkl_path, self.fit_generation)

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

        crops = collect_crops(clip_path, self.player_det, config.FIT_SAMPLE_EVERY, self.ruleset)
        if self.fit_status == "refit":
            crops = self._fit_crops_clip1 + crops

        clf = GSFATeamClassifier(
            device            = config.DEVICE,
            torso_ratio       = self.ruleset.torso_ratio,
            blur_threshold    = self.ruleset.blur_threshold,
            min_crop_px       = self.ruleset.min_crop_px,
            centre_crop_ratio = self.ruleset.centre_crop_ratio,
        )
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
            self._write_fit_meta()
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
            self._write_fit_meta()
            log.error("[%s] refit still below threshold (silhouette=%.3f) — "
                      "committing anyway", self.match_id, score)
        return self.fit_status

    def _resolve_team_names(self, clf: GSFATeamClassifier,
                            crops: list[np.ndarray]) -> None:
        """Map the two clusters to the caller-supplied team names by jersey
        colour, and record which cluster resolved to team_a_name as
        self.team_a_cluster_id (used by clip_processor's
        orient_for_team_a() to keep stored team_a/team_b orientation
        colour-consistent). Runs once, when the fit is committed; the
        team_id_to_name mapping is pickled with clf so later clips only look
        it up. team_a_cluster_id stays None (orient_for_team_a becomes a
        no-op, raw cluster order is used) if the caller did not supply both
        names and both colours, or if colour resolution fails — on a
        mid-match re-fit (fit_generation > 1) that failure is logged loudly
        since it silently corrupts team_a/team_b orientation until the next
        successful re-fit, a documented v1 limitation."""
        conn = db.get_conn()
        try:
            raw = db.get_team_specs(conn, self.match_id)
        finally:
            conn.close()
        if not raw:
            self.team_a_cluster_id = None
            return
        specs = [TeamSpec(name=n, colour=c) for n, c in raw]   # specs[0] is team_a
        try:
            mapping = clf.resolve_team_names(specs, crops)
            self.team_a_cluster_id = next(
                (cid for cid, name in mapping.items() if name == specs[0].name), None
            )
            log.info("[%s] team colours resolved → %s (team_a_cluster_id=%s)",
                     self.match_id, mapping, self.team_a_cluster_id)
        except (ValueError, RuntimeError) as exc:
            self.team_a_cluster_id = None
            log.warning("[%s] team colour resolution failed (%s) — "
                        "falling back to raw cluster order for team_a/team_b",
                        self.match_id, exc)
            if self.fit_generation > 1:
                log.error("[%s] colour resolution failed on a re-fit (fit_generation=%d) "
                          "— team_a/team_b orientation for this and following minutes may "
                          "not match team_a_colour until the next successful re-fit",
                          self.match_id, self.fit_generation)

    def _write_fit_meta(self) -> None:
        """Sidecar written alongside every committed fit pkl (joblib.dump),
        so fit_generation / team_a_cluster_id survive a disk reload
        (worker restart, or a late clip after idle eviction) instead of
        silently reverting to generation 1 / unresolved orientation."""
        meta = {"fit_generation": self.fit_generation, "team_a_cluster_id": self.team_a_cluster_id}
        self.fit_meta_path.write_text(json.dumps(meta), encoding="utf-8")

    def _load_fit_meta(self) -> None:
        """Restore fit_generation / team_a_cluster_id from fit_meta_path on a
        disk-reloaded fit. A missing/corrupt sidecar (e.g. a pkl committed
        before this feature existed) degrades to generation 1 / unresolved
        orientation rather than raising — should not crash the worker."""
        try:
            meta = json.loads(self.fit_meta_path.read_text(encoding="utf-8"))
            self.fit_generation    = int(meta.get("fit_generation", 1))
            self.team_a_cluster_id = meta.get("team_a_cluster_id")
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            log.warning("[%s] fit_meta.json missing/unreadable at %s — assuming "
                        "fit_generation=1, team_a orientation unresolved",
                        self.match_id, self.fit_meta_path)
            self.fit_generation    = 1
            self.team_a_cluster_id = None

    def reset_for_new_generation(self, n: int) -> None:
        """Mid-match team/GK colour change (matches.fit_generation advanced
        past this session's own): discard the committed team fit and GK
        classifier and start clean, as if this were clip 1 again, under
        fit_generation n. Re-running KMeans on the SAME clusters mid-match
        can swap team 0/1 labels and corrupt stats (see ensure_fit's
        docstring) — a genuinely new fit_generation is the one case where
        that's exactly what we want, since the jerseys themselves changed.

        Cross-clip CV state that has nothing to do with jersey colour —
        tracker / ball_tracker / carrier_eng / pass_track, proc_idx,
        n_events_seen, last_written, carryover_travel_frames — is left
        untouched, so an in-flight pass isn't lost across the reset."""
        if self.fit_pkl_path.exists():
            self.fit_pkl_path.unlink()
        if self.fit_meta_path.exists():
            self.fit_meta_path.unlink()
        self.team_clf   = None
        self.fit_status = "pending"
        self._fit_crops_clip1 = []
        self.gk_det = None
        self._gk_colour_invalid = False
        self.team_a_cluster_id = None
        self.fit_generation = n
        self.last_touched = time.monotonic()
        log.info("[%s] session reset for fit_generation=%d (team/GK colours changed)",
                 self.match_id, n)

    # ------------------------------------------------------------------
    # Goalkeeper classifier readiness (no fit stage — just needs the two
    # reference colours from the DB; cheap enough to retry every clip)
    # ------------------------------------------------------------------

    def ensure_gk_ready(self) -> None:
        """Construct self.gk_det if not already done and the DB now has both
        GK reference colours. No-op once constructed. Never raises — GK
        classification is an overlay on top of the core possession stats."""
        if self.gk_det is not None or self._gk_colour_invalid:
            return

        conn = db.get_conn()
        try:
            gk_colours = db.get_gk_colours(conn, self.match_id)
        finally:
            conn.close()
        if not gk_colours:
            return

        team_a_gk_colour, team_b_gk_colour = gk_colours
        try:
            self.gk_det = GoalkeeperDetector(
                team_a_gk_colour, team_b_gk_colour,
                max_colour_dist   = self.ruleset.max_gk_colour_dist,
                torso_ratio       = self.ruleset.torso_ratio,
                centre_crop_ratio = self.ruleset.centre_crop_ratio,
            )
            log.info("[%s] GK classifier ready (colours=%s/%s)",
                     self.match_id, team_a_gk_colour, team_b_gk_colour)
        except ValueError as exc:
            self._gk_colour_invalid = True
            log.warning("[%s] invalid GK colour (%s) — GK classification disabled",
                        self.match_id, exc)

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
        # Snapshot NOW (this clip's own team_a_cluster_id, unchanged since
        # ensure_fit ran before process_clip) — becomes "the orientation this
        # row was written under" for whichever future clip's boundary
        # correction, if any, ends up targeting it.
        self.last_written_team_a_cluster_id = self.team_a_cluster_id
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

    def get_or_create(
        self, match_id: str, ruleset: RulesetConfig,
        want_generation: Optional[int] = None,
    ) -> tuple[MatchSession, bool]:
        """`ruleset` is only used when creating a new session — an existing
        session keeps whatever ruleset it was created with (a match's
        ruleset is fixed for its lifetime, see db.ensure_match).

        `want_generation` (db.get_fit_generation's current value) is
        compared against the session's own fit_generation — new, cached, and
        disk-reloaded sessions alike (a disk reload restores fit_generation
        from the fit_meta.json sidecar, so a stale sidecar next to an
        up-to-date pkl is caught here too). When want_generation is higher,
        the session is reset via reset_for_new_generation before being
        returned. Returns (session, just_reset)."""
        sess = self._sessions.get(match_id)
        if sess is None:
            sess = MatchSession(match_id, self.models, ruleset)
            self._sessions[match_id] = sess
        just_reset = False
        if want_generation is not None and want_generation > sess.fit_generation:
            sess.reset_for_new_generation(want_generation)
            just_reset = True
        sess.last_touched = time.monotonic()
        return sess, just_reset

    def evict_idle(self) -> None:
        now = time.monotonic()
        for mid in [m for m, s in self._sessions.items()
                    if now - s.last_touched > config.SESSION_IDLE_EVICT_S]:
            log.info("[%s] evicting idle session (fit pkl stays on disk)", mid)
            del self._sessions[mid]

    @property
    def active_matches(self) -> list[str]:
        return list(self._sessions.keys())
