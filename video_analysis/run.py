"""
video_analysis/run.py — Local dev entrypoint: process one whole video file,
render an annotated .mp4, print a possession/pass summary.

Pipeline (same modules/ pipeline the Azure service uses):
  Detect (unified YOLOv11m: players + ball) -> SigLIP team -> GK -> BoT-SORT (+GMC)
  -> BallTracker (Kalman smoothing) -> CarrierEngine (bbox-relative foot-zone)
  -> PassEventTracker (release / travel / reception phases)
  -> PossessionStats (strict denominator)

Which sport's tuning is used is selected via --ruleset (futsal | classic),
resolved through rulesets.registry — see rulesets/base.py for the full list
of parameters a ruleset controls.

Run:
    python video_analysis/run.py
    python video_analysis/run.py --ruleset classic
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.detectors.goalkeeper_detector import GoalkeeperDetector
from modules.detectors.player_detector import Detection, FrameDetections, PlayerDetector
from modules.possession import (
    BallDetection,
    BallTracker,
    CarrierEngine,
    CarrierState,
    PassEventTracker,
    PossessionStats,
    best_ball,
)
from modules.possession.pass_event_tracker import (
    PHASE_CAND_REL,
    PHASE_CAND_RCV,
    PHASE_IDLE,
    PHASE_POSS,
    PHASE_TRAVEL,
)
from modules.team_classifier.team_classifier import GSFATeamClassifier
from modules.tracking.player_tracker import PlayerTracker
from rulesets import DEFAULT_RULESET, RulesetConfig, get_ruleset

# ---------------------------------------------------------------------------
# CONFIGURATION (local-dev-only — not ruleset-controlled)
# ---------------------------------------------------------------------------

DEVICE      = "cuda"   # CUDA is available locally; "cpu" works but is slow at imgsz 960
VIDEO_PATH  = r"C:\Users\Admin\Downloads\6a8edb537fe372ce49da9eb4_part-00003.mp4"
OUTPUT_PATH = r"data/output/possession_output.mp4"
CMC_METHOD  = "ecc"

# Debug / speed run
PROCESS_DURATION_SEC = 480      # stop after this many seconds

# Visuals
POSSESS_BAR_H = 42
TEAM_BGR = {0: (255, 80, 0), 1: (0, 80, 255), None: (160, 160, 160)}
GK_BGR   = (0, 215, 255)
BALL_BGR = (0, 255, 255)
REF_BGR  = (80, 220, 80)

# Possession/event labels re-exported here for the print statements below.
from modules.possession.labels import (  # noqa: E402
    EVT_BALL_LOST,
    EVT_COMPLETED,
    EVT_INTERCEPTION,
)


# ---------------------------------------------------------------------------
# SUPERVISION ANNOTATORS
# ---------------------------------------------------------------------------
# Index 0 = team_a (blue), 1 = team_b (red), 2 = unclassified (grey)
_PALETTE = sv.ColorPalette.from_hex(["#0050FF", "#FF5000", "#A0A0A0"])

_ellipse_ann  = sv.EllipseAnnotator(color=_PALETTE, thickness=1)
_triangle_ann = sv.TriangleAnnotator(
    color=sv.Color.from_hex("#00FFFF"), base=16, height=16,
    color_lookup=sv.ColorLookup.INDEX,
)
_label_ann = sv.LabelAnnotator(
    color          = _PALETTE,
    text_color     = sv.Color.WHITE,
    text_scale     = 0.32,
    text_thickness = 1,
    text_padding   = 2,
)


def _to_sv(players: list[Detection]) -> tuple[sv.Detections, list[str]]:
    if not players:
        return sv.Detections(xyxy=np.empty((0, 4))), []
    xyxy     = np.array([[*p.bbox] for p in players], dtype=float)
    class_id = np.array(
        [p.team_id if p.team_id is not None else 2 for p in players], dtype=int
    )
    labels = []
    for p in players:
        base = (f"GK-T{p.team_id}" if p.is_goalkeeper
                else (f"T{p.team_id}" if p.team_id is not None else "?"))
        labels.append(f"{base}#{p.track_id}" if p.track_id is not None else base)
    return sv.Detections(xyxy=xyxy, class_id=class_id), labels


# ---------------------------------------------------------------------------
# FRAME RENDERING
# ---------------------------------------------------------------------------

def draw_frame(
    frame:        np.ndarray,
    player_dets:  FrameDetections,
    ball_det:     Optional[BallDetection],
    carrier:      CarrierState,
    poss_stats:   PossessionStats,
    pass_tracker: PassEventTracker,
) -> np.ndarray:
    out = frame.copy()
    H_vid, W_vid = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    sv_dets, labels = _to_sv(player_dets.players)
    if len(sv_dets) > 0:
        out = _ellipse_ann.annotate(scene=out, detections=sv_dets)
        out = _label_ann.annotate(scene=out, detections=sv_dets, labels=labels)

    for p in player_dets.players:
        if p.is_goalkeeper:
            x1, y1, x2, y2 = p.bbox
            cx = (x1 + x2) // 2
            rx = max((x2 - x1) // 2, 10)
            cv2.ellipse(out, (cx, y2), (rx, 6), 0, -45, 225, GK_BGR, 2)

    for gp in player_dets.goal_posts:
        x1, y1, x2, y2 = gp.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), 2)

    if ball_det is not None:
        ball_sv = sv.Detections(xyxy=np.array([[*ball_det.bbox]], dtype=float))
        out = _triangle_ann.annotate(scene=out, detections=ball_sv)

    if carrier.kind == "carrier" and carrier.player is not None:
        cp = carrier.player.foot_point
        cv2.circle(out, cp, 14, (255, 255, 0), 2)

    # Possession bar
    t0_pct, t1_pct = poss_stats.percentages()
    bx1, bx2 = 10, W_vid - 10
    by1, by2 = 8, 8 + POSSESS_BAR_H
    bar_w = bx2 - bx1
    cv2.rectangle(out, (bx1, by1), (bx2, by2), (25, 25, 25), -1)
    t0_w = int(bar_w * t0_pct / 100)
    if t0_w > 0:
        cv2.rectangle(out, (bx1, by1), (bx1 + t0_w, by2), TEAM_BGR[0], -1)
    t1_w = int(bar_w * t1_pct / 100)
    if t1_w > 0:
        cv2.rectangle(out, (bx2 - t1_w, by1), (bx2, by2), TEAM_BGR[1], -1)
    cv2.putText(out, f"T0  {t0_pct:.1f}%",
                (bx1 + 8, by2 - 10), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"{t1_pct:.1f}%  T1",
                (bx2 - 130, by2 - 10), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Pass overlay
    pass_summary = pass_tracker.summary_for_overlay()
    y0 = by2 + 24
    for tid in (0, 1):
        s = pass_summary[tid]
        txt = f"T{tid}  passes: {s['successful']}   inaccurate: {s['inaccurate']}"
        cv2.putText(out, txt, (12, y0 + tid * 22),
                    font, 0.55, TEAM_BGR[tid], 2, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run(video_path: str = VIDEO_PATH, out_path: str = OUTPUT_PATH,
        team_a_gk_colour: str | None = None, team_b_gk_colour: str | None = None,
        ruleset: str = DEFAULT_RULESET,
        model_path: str | None = None, model_classes: str | None = None,
        imgsz: int | None = None, detect_classes: str | None = None) -> None:
    rs: RulesetConfig = get_ruleset(ruleset)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[Possession] ruleset={rs.name!r} — initialising models …")
    player_det = PlayerDetector(
        model_path  = model_path or rs.player_model_weights,
        device      = DEVICE,
        player_conf = rs.player_conf,
        ball_conf   = rs.ball_conf,
        classes     = [int(c) for c in detect_classes.split(",")] if detect_classes else None,
        class_names = rs.class_names,
    )   # classic: players + ball + refs; futsal: + goal posts
    if model_classes:
        # Explicit override for a one-off test model whose class schema differs
        # from the ruleset's map (e.g. no goal_post, different class ids) —
        # instance-only, wins over rs.class_names set above.
        names = model_classes.split(",")
        player_det.CLASS_NAMES = dict(enumerate(names))
    if imgsz:
        player_det.imgsz = imgsz

    team_clf = GSFATeamClassifier(
        device            = DEVICE,
        torso_ratio       = rs.torso_ratio,
        blur_threshold    = rs.blur_threshold,
        min_crop_px       = rs.min_crop_px,
        centre_crop_ratio = rs.centre_crop_ratio,
    )
    team_clf.fit_from_video_or_load(video_path, player_det)

    # team_a_gk_colour/team_b_gk_colour: hex ("#FF6600") or CSS name ("orange"),
    # same format as team colours. Both required to enable GK classification
    # (direct jersey-colour match, runs independently of team_clf); omit
    # either to skip GK classification entirely.
    gk_det = (GoalkeeperDetector(
                  team_a_gk_colour, team_b_gk_colour,
                  max_colour_dist   = rs.max_gk_colour_dist,
                  torso_ratio       = rs.torso_ratio,
                  centre_crop_ratio = rs.centre_crop_ratio,
              )
              if team_a_gk_colour and team_b_gk_colour else None)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W_vid   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_vid   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    import warnings, os
    warnings.filterwarnings("ignore")
    os.environ["TQDM_DISABLE"] = "1"

    frame_step = 1   # process & render every native frame for smooth output
    max_frame  = int(fps * PROCESS_DURATION_SEC)
    eff_fps    = fps / frame_step   # == native fps

    # The event/Kalman constants in the ruleset are calibrated for
    # rs.reference_fps real-time durations. Now that we process at native
    # fps, rescale the frame-count thresholds by eff_fps/reference_fps so the
    # real-time behaviour holds.
    fps_scale = eff_fps / rs.reference_fps
    _sc = lambda n: max(1, round(n * fps_scale))

    tracker      = PlayerTracker(
        fps                           = fps,
        cmc_method                    = CMC_METHOD,
        track_high_thresh             = rs.track_high_thresh,
        track_low_thresh              = rs.track_low_thresh,
        new_track_thresh              = rs.new_track_thresh,
        match_thresh                  = rs.match_thresh,
        proximity_thresh              = rs.proximity_thresh,
        appearance_thresh             = rs.appearance_thresh,
        track_buffer_frames_at_30fps  = rs.track_buffer_frames_at_30fps,
    )
    ball_tracker = BallTracker(
        coast_frames = _sc(rs.kalman_coast_frames),
        gate_sigma   = rs.kalman_gate_sigma,
    )
    carrier_eng  = CarrierEngine(
        hysteresis_n     = _sc(rs.carrier_hysteresis_n),
        foot_zone_ratio  = rs.foot_zone_ratio,
        foot_zone_min_px = rs.foot_zone_min_px,
        foot_zone_max_px = rs.foot_zone_max_px,
    )
    pass_track   = PassEventTracker(
        release_sustain  = _sc(rs.release_sustain),
        reception_settle = _sc(rs.reception_settle),
        travel_min_gap   = _sc(rs.travel_min_gap),
        travel_timeout   = _sc(rs.travel_timeout_frames),
    )
    stats        = PossessionStats()

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        eff_fps,
        (W_vid, H_vid),
    )

    print(
        f"[Possession] first {PROCESS_DURATION_SEC:.0f}s | "
        f"native {fps:.0f}fps → {eff_fps:.0f}fps (step={frame_step}) | "
        f"{W_vid}x{H_vid}"
    )
    print("-" * 60)

    fidx     = 0
    proc_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or fidx >= max_frame:
            break

        if fidx % frame_step != 0:
            fidx += 1
            continue

        t_sec = fidx / fps

        player_dets = player_det.detect(frame, fidx, fps)
        team_clf.classify(frame, player_dets)
        if gk_det is not None:
            gk_det.classify(frame, player_dets)
        tracker.update(frame, player_dets.players)

        raw_ball         = best_ball(player_dets)
        ball_state, ball = ball_tracker.update(raw_ball)

        carrier = carrier_eng.update(player_dets.players, ball_state, ball)

        prev_phase  = pass_track.phase
        prev_passer = pass_track._passer
        prev_n_evts = len(pass_track.events)

        label, adjustments = pass_track.update(carrier, proc_idx)

        # --- Phase-transition logs ---
        cur_phase = pass_track.phase
        if cur_phase != prev_phase:
            p = pass_track._passer
            r = pass_track._receiver
            if cur_phase == PHASE_POSS and prev_phase == PHASE_IDLE:
                print(f"[{t_sec:.2f}s] NEW CARRIER       team={p[1]} track={p[0]}")
            elif cur_phase == PHASE_CAND_REL:
                src = prev_passer or p
                if src:
                    print(f"[{t_sec:.2f}s] CAND RELEASE      team={src[1]} track={src[0]}  ball left foot-zone")
            elif cur_phase == PHASE_TRAVEL and prev_phase == PHASE_CAND_REL:
                if p:
                    print(f"[{t_sec:.2f}s] RELEASE CONFIRMED team={p[1]} track={p[0]}  ball in travel")
            elif cur_phase == PHASE_CAND_RCV:
                if r and p:
                    print(f"[{t_sec:.2f}s] CAND RECEPTION    receiver team={r[1]} track={r[0]}  from team={p[1]} track={p[0]}")
            elif cur_phase == PHASE_TRAVEL and prev_phase == PHASE_CAND_RCV:
                print(f"[{t_sec:.2f}s] RECEPTION DROPPED receiver lost ball, back to travel")

        proc_idx += 1

        # --- Resolved-event logs ---
        for evt in pass_track.events[prev_n_evts:]:
            t_evt = evt.end_frame / eff_fps
            if evt.kind == EVT_COMPLETED:
                print(
                    f"[{t_evt:.2f}s] PASS COMPLETED    "
                    f"T{evt.from_team_id}#{evt.from_track_id} → T{evt.to_team_id}#{evt.to_track_id}  "
                    f"travel={evt.travel_frames}f"
                )
            elif evt.kind == EVT_INTERCEPTION:
                print(
                    f"[{t_evt:.2f}s] INTERCEPTED       "
                    f"T{evt.from_team_id}#{evt.from_track_id} → T{evt.to_team_id}#{evt.to_track_id}  "
                    f"travel={evt.travel_frames}f"
                )
            elif evt.kind == EVT_BALL_LOST:
                print(
                    f"[{t_evt:.2f}s] BALL LOST         "
                    f"from T{evt.from_team_id}#{evt.from_track_id}  "
                    f"travel={evt.travel_frames}f"
                )

        stats.update(label)
        if adjustments:
            stats.apply_adjustments(adjustments)

        out_frame = draw_frame(frame, player_dets, ball, carrier, stats, pass_track)
        writer.write(out_frame)

        fidx += 1

    cap.release()
    writer.release()
    print("-" * 60)
    print(f"\n[Possession] Saved → {out_path}")
    print(stats.summary())
    print(pass_track.summary())


def _parse_args() -> "argparse.Namespace":
    import argparse
    from rulesets.registry import available_rulesets

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", default=VIDEO_PATH, help="Input video path")
    p.add_argument("--out", default=OUTPUT_PATH, help="Output annotated .mp4 path")
    p.add_argument("--ruleset", default=DEFAULT_RULESET, choices=available_rulesets(),
                    help="Which sport's tuning to use")
    p.add_argument("--team-a-gk-colour", default=None)
    p.add_argument("--team-b-gk-colour", default=None)
    p.add_argument("--model-path", default=None,
                    help="Override the ruleset's player_model_weights (test a different .pt)")
    p.add_argument("--model-classes", default=None,
                    help="Comma-separated class names in id order, e.g. "
                         "'active_player,ball,referee' — overrides the ruleset's class_names "
                         "map for a one-off test model whose class schema differs")
    p.add_argument("--imgsz", type=int, default=None,
                    help="Override PlayerDetector's inference resolution (default 960)")
    p.add_argument("--detect-classes", default=None,
                    help="Comma-separated class ids to run inference on, e.g. '0,1' for "
                         "player+ball only — passed straight to the Ultralytics call as an "
                         "allow-list (default: all classes in the model)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        video_path      = args.video,
        out_path        = args.out,
        team_a_gk_colour = args.team_a_gk_colour,
        team_b_gk_colour = args.team_b_gk_colour,
        ruleset         = args.ruleset,
        model_path      = args.model_path,
        model_classes   = args.model_classes,
        imgsz           = args.imgsz,
        detect_classes  = args.detect_classes,
    )
