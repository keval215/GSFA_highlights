---
name: possession-pipeline-architecture
description: Complete possession+pass pipeline architecture, file locations, thresholds, and model details (as of branch test)
type: project
---

Full pipeline lives in video_analysis/possession.py. Flat single-file Colab version at gsfa_colab_pipeline.py.

## Models

- **Player/referee/goalpost detector:** YOLOv11, file GSFA_PLAYER_DETECTION.pt
  - class 0 = active_player, class 1 = goal_post, class 2 = referee
  - conf=0.50, device=cpu
- **Ball detector:** RFDETRMedium (.pth checkpoint, gsfa_ball_detection.pth)
  - num_classes=2, resolution=576, class_id=1 = ball, conf=0.25
- **Team classifier:** GSFATeamClassifier — SigLIP(768-D) + UMAP(→3-D) + KMeans(k=2)
  - Crop: top 55% of player bbox (TORSO_RATIO=0.55), min 32px, blur filter (Laplacian var > 80)
  - Fit: 1fps sampling (every 30 frames), saves pkl to data/cache/<stem>_team_siglip.pkl
  - Also stores 768-D embedding on each Detection for tracker ReID
- **Tracker:** BoT-SORT (boxmot), with_reid=True, cmc_method="ecc"
  - track_high_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.6
  - match_thresh=0.8, proximity_thresh=0.5, appearance_thresh=0.25
  - track_buffer=60 frames (at 30fps), reid_model=None (embeddings passed externally)
- **Goalkeeper detector:** spatial fit — finds closest player per goal_post over video;
  assigns each GK zone to nearer team centroid; saves pkl data/cache/<stem>_goalkeeper.pkl

## Possession calculation

Frame rate subsampled to TARGET_PROCESS_FPS=15fps (frame_step = round(native_fps/15)).

**Ball-to-player assignment (CarrierEngine):**
- foot_zone_radius = clamp(0.45 * bbox_height, 20px, 140px)
- Distance: ball_centre to player foot_point (bottom-centre of bbox)
- Only team-0 or team-1 players with a track_id qualify
- Multiple players in zone → CONTESTED; none → LOOSE; ball LOST → OOF
- Hysteresis: 3 consecutive frames must agree before committing a carrier change

**PossessionStats denominator:** only team0_frames + team1_frames (excludes loose/contested/OOF).
Retroactive corrections applied when PassEventTracker resolves interceptions (flip_to) or ball_lost (drop).

## Pass accuracy / inaccurate pass detection

3-phase FSM (PassEventTracker) operating on CarrierState transitions (processed-frame units at 15fps):

1. **CAND_RELEASE:** ball leaves carrier's foot zone; wait 1 frame to confirm it's not a dribble
2. **TRAVEL:** release confirmed; wait for a reception or timeout (22 frames = ~1.47s)
   - Travel min gap before a receiver is accepted: 1 processed frame
3. **CAND_RECEPTION:** ball enters new player's foot zone; wait 2 frames to settle

Resolution:
- Receiver same team → EVT_COMPLETED → counted as "successful" pass for that team
- Receiver other team → EVT_INTERCEPTION → counted as "inaccurate" pass on PASSER's team
- Ball returns to passer → EVT_DRIBBLE_CANCEL → ignored for pass stats
- Timeout (22 frames) → EVT_BALL_LOST → provisional possession credit dropped to OOF

**Pass accuracy = successful / (successful + inaccurate) per team**

## Drawing / annotation

- EllipseAnnotator (sv) under players, coloured by team_id
- LabelAnnotator: "T0#trackid", "GK-T1#trackid" etc.
- Goalkeeper: extra cyan ellipse at foot level
- Goal post: black rectangle
- Ball: cyan triangle (sv TriangleAnnotator)
- Carrier: yellow circle (r=14) at carrier foot_point
- Possession bar: top of frame, 42px high, dark background; team0 fills left, team1 fills right
- Pass overlay: text below possession bar per team

## Output

VideoWriter: mp4v codec, TARGET_PROCESS_FPS=15fps, native resolution.
