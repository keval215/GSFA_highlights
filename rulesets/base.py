"""RulesetConfig — every sport-tunable parameter the CV pipeline needs, in one place.

Everything here used to be a scattered Python module constant (in
video_analysis/possession.py, detectors/goalkeeper_detector.py,
team_classifier/team_classifier.py, tracking/player_tracker.py) with no way
to give futsal and classic 11-a-side football different values without
editing source. This dataclass is the single seam: `rulesets/futsal.py` and
`rulesets/classic.py` each fill in one instance; `modules/*` objects are
constructed from whichever instance the match's ruleset selects.

Deliberately NOT here: infra/operational settings that apply uniformly
regardless of sport (device, target processing fps, camera-motion-compensation
method, batch window, DB/blob/queue config) — those stay in service/config.py,
unchanged by this refactor.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RulesetConfig:
    name: str

    # --- Detection (modules/detectors/player_detector.py) -------------------
    player_model_weights: str
    player_conf: float = 0.50
    ball_conf: float = 0.25

    # --- Team classifier (modules/team_classifier/team_classifier.py) -------
    # Camera-framing-dependent — classic football's typically wider broadcast
    # angle is a real candidate to need different values than futsal's
    # close court-side framing.
    torso_ratio: float = 0.55
    blur_threshold: float = 80.0
    min_crop_px: int = 32
    centre_crop_ratio: float = 0.50

    # --- Goalkeeper (modules/detectors/goalkeeper_detector.py) --------------
    max_gk_colour_dist: float = 60.0

    # --- Tracking (modules/tracking/player_tracker.py) ----------------------
    # BoT-SORT association thresholds. 11-a-side means more simultaneous
    # tracks/occlusion than 5-a-side, so these are prime candidates to differ.
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    match_thresh: float = 0.8
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    track_buffer_frames_at_30fps: int = 60

    # --- Ball tracking (modules/possession/ball_tracker.py) -----------------
    kalman_coast_frames: int = 12
    kalman_gate_sigma: float = 6.0

    # --- Carrier engine (modules/possession/carrier_engine.py) --------------
    foot_zone_ratio: float = 0.45
    foot_zone_min_px: int = 20
    foot_zone_max_px: int = 140
    carrier_hysteresis_n: int = 3

    # --- Pass event FSM (modules/possession/pass_event_tracker.py) ----------
    # Calibrated against `reference_fps`; a classic-football pitch is longer,
    # so average pass travel time is longer too — travel_timeout_frames in
    # particular is a prime candidate to differ from futsal's value.
    release_sustain: int = 1
    reception_settle: int = 2
    travel_min_gap: int = 1
    travel_timeout_frames: int = 22
    reference_fps: float = 15.0
