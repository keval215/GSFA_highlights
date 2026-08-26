"""CLASSIC — 11-a-side football ruleset.

Structurally complete (every field the pipeline needs is set), but several
numeric values below are starting points, not calibrated numbers — marked
PLACEHOLDER. Classic football's wider broadcast camera framing (whole pitch
visible vs. futsal's close court-side shot), longer pitch, and larger roster
(more simultaneous tracks/occlusion) are all reasons these are expected to
differ from futsal's tuned values, but the actual numbers need calibration
against real 11-a-side footage before this ruleset is production-ready.
`player_model_weights` in particular MUST be replaced with a real
classic-trained YOLOv11m checkpoint before this ruleset is usable — there is
no trained classic model yet.
"""

from __future__ import annotations

from rulesets.base import RulesetConfig

CLASSIC = RulesetConfig(
    name="classic",
    # PLACEHOLDER — no classic-trained checkpoint exists yet. The service
    # requires the CLASSIC_PLAYER_WEIGHTS env var to be set before this
    # ruleset can actually be used (see service/config.py); this default only
    # matters for standalone local runs.
    player_model_weights=r"C:\Users\Admin\OneDrive\Desktop\CZ\GSFA_CLASSIC_PLAYER_DETECTION.pt",
    player_conf=0.50,
    ball_conf=0.25,
    # PLACEHOLDER — wider broadcast framing means smaller player crops in
    # absolute pixels; torso_ratio/blur_threshold likely need retuning once
    # real classic footage is available.
    torso_ratio=0.55,
    blur_threshold=80.0,
    min_crop_px=24,
    centre_crop_ratio=0.50,
    max_gk_colour_dist=60.0,
    track_high_thresh=0.5,
    track_low_thresh=0.1,
    new_track_thresh=0.6,
    match_thresh=0.8,
    proximity_thresh=0.5,
    appearance_thresh=0.25,
    # PLACEHOLDER — 11 players/side means more simultaneous tracks and
    # occlusion than futsal's 5; longer buffer keeps lost tracks alive
    # through more traffic before giving up.
    track_buffer_frames_at_30fps=90,
    # PLACEHOLDER — longer pitch means longer/harder kicks; the ball can
    # plausibly leave frame or blur out for longer stretches.
    kalman_coast_frames=18,
    kalman_gate_sigma=8.0,
    # PLACEHOLDER — wider camera framing shrinks player size on screen,
    # so the foot-zone pixel bounds likely need to shrink to match.
    foot_zone_ratio=0.45,
    foot_zone_min_px=14,
    foot_zone_max_px=100,
    carrier_hysteresis_n=3,
    release_sustain=1,
    reception_settle=2,
    travel_min_gap=1,
    # PLACEHOLDER — a full-size pitch means longer average pass travel time
    # than futsal's tight court; this is the single number most likely to
    # need real recalibration.
    travel_timeout_frames=45,
    reference_fps=15.0,
)
