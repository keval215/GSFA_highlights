"""CLASSIC — 11-a-side football ruleset.

Structurally complete and committed as the starting production profile for
11-a-side. The values below are tuned for classic football's wider broadcast
camera framing (whole pitch visible vs. futsal's close court-side shot),
longer pitch, and larger roster (more simultaneous tracks / occlusion). They
are reasoned starting points, not values calibrated frame-by-frame against a
large 11-a-side corpus — expect to nudge them once real match telemetry is in
(the fields most likely to move are noted inline).

`player_model_weights` here is only the local-dev default for
`video_analysis/run.py`; the production service resolves classic weights from
the `CLASSIC_PLAYER_WEIGHTS` env var (see `service/config.py::player_weights`),
which must point at a real classic-trained YOLOv11m checkpoint on the VM.
"""

from __future__ import annotations

from rulesets.base import RulesetConfig

CLASSIC = RulesetConfig(
    name="classic",
    # Local-dev default only (video_analysis/run.py). The service ignores this
    # and requires CLASSIC_PLAYER_WEIGHTS in the environment.
    player_model_weights=r"C:\Users\Admin\OneDrive\Desktop\CZ\GSFA_CLASSIC_PLAYER_DETECTION.pt",
    player_conf=0.50,
    ball_conf=0.25,
    # GSFA_CLASSIC_PLAYER_DETECTION.pt / GSL_v1.pt is a 3-class model:
    #   0 active_players, 1 ball, 2 refree  (the model's own spelling; no goal_post).
    # Map id 2 to "referee" so refs land in FrameDetections.referees, not
    # goal_posts. Classic produces no goal-post detections by design.
    class_names={0: "active_player", 1: "ball", 2: "referee"},
    # Wider broadcast framing → the player bbox is smaller on screen, so take a
    # larger top fraction of it as the jersey crop to keep enough pixels for the
    # SigLIP embedding (futsal uses 0.55 on its tighter, larger boxes).
    torso_ratio=0.65,
    # Smaller, softer torso crops from the wide shot carry less high-frequency
    # detail, so their Laplacian variance runs lower than futsal's close-up
    # crops. Drop the sharp-enough floor from 80 → 55 so the team fit is not
    # starved of otherwise-usable crops. Revisit against real footage.
    blur_threshold=55.0,
    min_crop_px=24,
    centre_crop_ratio=0.50,
    max_gk_colour_dist=60.0,
    track_high_thresh=0.5,
    track_low_thresh=0.1,
    new_track_thresh=0.6,
    match_thresh=0.8,
    proximity_thresh=0.5,
    appearance_thresh=0.25,
    # 11 players/side means more simultaneous tracks and occlusion than futsal's
    # 5; a longer buffer keeps lost tracks alive through more traffic before
    # giving up.
    track_buffer_frames_at_30fps=90,
    # Longer pitch → harder/longer kicks; the ball can plausibly leave frame or
    # blur out for longer stretches, so let the Kalman filter coast further
    # before declaring the track lost.
    kalman_coast_frames=20,
    kalman_gate_sigma=8.0,
    # Wider camera framing shrinks on-screen player size, so the foot-zone pixel
    # bounds shrink to match.
    foot_zone_ratio=0.45,
    foot_zone_min_px=14,
    foot_zone_max_px=100,
    carrier_hysteresis_n=3,
    release_sustain=1,
    reception_settle=2,
    travel_min_gap=1,
    # A full-size pitch means longer average pass travel time than futsal's tight
    # court; this is the single number most likely to need real recalibration.
    travel_timeout_frames=45,
    reference_fps=15.0,
)
