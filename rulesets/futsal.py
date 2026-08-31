"""FUTSAL — today's exact tuning, extracted verbatim from the pre-refactor
module constants. Loading this ruleset must not change behavior for any
existing futsal match.

Sources (pre-refactor):
  detectors/player_detector.py    PlayerDetector.MODEL_PATH / conf defaults
  team_classifier/team_classifier.py   MIN_CROP_PX / TORSO_RATIO / BLUR_THRESHOLD / CENTRE_CROP_RATIO
  detectors/goalkeeper_detector.py     MAX_GK_COLOUR_DIST
  tracking/player_tracker.py           BoT-SORT thresholds hardcoded in __init__
  video_analysis/possession.py         BallTracker / CarrierEngine / PassEventTracker constants
"""

from __future__ import annotations

from rulesets.base import RulesetConfig

FUTSAL = RulesetConfig(
    name="futsal",
    # Matches video_analysis/possession.py's pre-refactor PLAYER_MODEL_WEIGHTS
    # exactly (NOT PlayerDetector.MODEL_PATH — those were always two
    # independently-tracked checkpoint paths, not one falling back to the
    # other). Keeping this literal preserves video_analysis/run.py's default
    # behavior unchanged. The production service never reads this default —
    # it always resolves weights via the PLAYER_WEIGHTS env var
    # (service/config.py), so this only affects local dev runs.
    player_model_weights=r"C:\Users\Admin\OneDrive\Desktop\CZ\aiff_v2.pt",
    player_conf=0.50,
    ball_conf=0.25,
    # Unified 4-class YOLOv11m schema (equals PlayerDetector.CLASS_NAMES; kept
    # explicit so futsal behaviour is unchanged if the default ever moves).
    class_names={0: "active_player", 1: "ball", 2: "goal_post", 3: "referee"},
    torso_ratio=0.55,
    blur_threshold=80.0,
    min_crop_px=32,
    centre_crop_ratio=0.50,
    max_gk_colour_dist=60.0,
    track_high_thresh=0.5,
    track_low_thresh=0.1,
    new_track_thresh=0.6,
    match_thresh=0.8,
    proximity_thresh=0.5,
    appearance_thresh=0.25,
    track_buffer_frames_at_30fps=60,
    kalman_coast_frames=12,
    kalman_gate_sigma=6.0,
    foot_zone_ratio=0.45,
    foot_zone_min_px=20,
    foot_zone_max_px=140,
    carrier_hysteresis_n=3,
    release_sustain=1,
    reception_settle=2,
    travel_min_gap=1,
    travel_timeout_frames=22,
    reference_fps=15.0,
)
