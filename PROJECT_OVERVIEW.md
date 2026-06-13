# GSFA Highlights — Project Overview

> Sport: Futsal (6v6, indoor stadium)
> Camera: Panning sideline broadcast camera
> Goal: Automatically analyse full-match video to produce possession stats, pass accuracy, shots-on-target counts, pitch coverage heatmaps, and eventually automated highlight reels.

---

## 1. What this project does

The GSFA Highlights pipeline takes raw match footage as input and produces:

- An annotated output video with live possession bars, pass counters, team labels, and ball trails.
- End-of-match statistics: possession % per team, pass count, pass accuracy (successful vs intercepted).
- Shots-on-target counts (provisional live + hindsight-validated) with a predicted ball trajectory overlay.
- A top-down pitch coverage heatmap showing which areas of the 40 x 20 m futsal court were visible across the video.
- Per-team player position heatmaps projected onto the top-down pitch diagram.

### High-level data flow

```
Raw video (.mp4)
    │
    ├─ [PlayerDetector]       YOLO11 – detects players, goal posts, referee
    ├─ [BallDetector]         RF-DETR Medium – detects ball each frame
    ├─ [GSFATeamClassifier]   SigLIP → UMAP → KMeans – assigns team 0 / team 1
    ├─ [GoalkeeperDetector]   Spatial fit – identifies GK per goal, assigns team
    ├─ [PlayerTracker]        BoT-SORT + SigLIP ReID – stable track IDs
    ├─ [BallTracker]          Kalman filter – smooths ball position, handles occlusion
    ├─ [CarrierEngine]        Foot-zone proximity – who has the ball each frame
    ├─ [PassEventTracker]     3-phase FSM – detects pass, interception, dribble
    ├─ [PossessionStats]      Denominator-strict running totals
    ├─ [ShotDetector]         Trajectory prediction + SAM2 goal planes (shots_on_t.py)
    ├─ [SimpleHomography]     YOLO pose keypoints → H matrix → pitch coordinates
    └─ Annotated output video + PNG heatmaps + JSON log
```

Everything runs frame-by-frame at a sub-sampled 15 fps (from the native frame rate). Caches are written per match so the slow fit phases (SigLIP training, GK detection, homography) are skipped on repeat runs.

---

## 2. Azure infrastructure

### 2a. Virtual Machine

| Property | Value |
|---|---|
| Name | gsfa-highlights |
| SKU | Standard NC4as T4 v3 |
| GPU | 1x NVIDIA T4 (16 GB VRAM) |
| OS | Ubuntu 24.04 LTS |
| Public IP | 4.186.40.179 (dynamic — see note below) |
| Resource group | GSFAStatus |
| Region | Central India Zone 1 |
| Subscription ID | a56f6649-7402-4bcb-b7df-c5b74d70d180 |

**Connection:**
```bash
ssh azureuser@4.186.40.179
```
The IP is dynamic — it changes if the VM is deallocated. The docs recommend converting to a static IP in the Azure portal (or using the DNS label `gsfa-highlights.centralindia.cloudapp.azure.com`).

**What the VM runs:** The entire pipeline runs inside a Docker container (`docker compose up -d --build`) with GPU passthrough via the NVIDIA Container Toolkit. The FastAPI web app (`server.py`) listens on port 8000 and accepts match video uploads, runs the pipeline, and uploads the output to Azure Blob Storage.

**GPU setup:** NVIDIA driver 550 + CUDA 12.4. The Dockerfile base image is the official CUDA 12.4.1 Ubuntu image. GPU mode is activated by setting `USE_GPU=1` in `/etc/gsfa-highlights.env`.

**Disk layout:**
- `/` — ~10 GB OS + Docker images
- `/var/lib/docker` — ~15 GB container layers (CUDA base is ~6 GB)
- `/mnt/data/jobs/<job_id>/` — transient job working directory, deleted after each run (up to ~15 GB per in-flight job; only one job runs at a time via `_job_sem`)

**Cost guidance:**
| Scenario | Approx. cost |
|---|---|
| VM running 24/7 | ~$380/month |
| VM deallocated (stopped) | ~$0.01/hr (disk only) |
| Recommended | Start VM only when processing a match, then deallocate |

```bash
# Deallocate from local machine (Azure CLI):
az vm deallocate --resource-group GSFAStatus --name gsfa-highlights
az vm start      --resource-group GSFAStatus --name gsfa-highlights
```

**NSG rules:** Only SSH (22) and the app port (8000) should be open, restricted to your own IP range. Port 8000 must not be opened to `0.0.0.0/0` — there is no auth layer on the FastAPI app in its current state.

### 2b. Azure Blob Storage

| Property | Value |
|---|---|
| Storage account | gsfastorage |
| Resource group | GSFAStatus |
| Region | Central India |
| SKU | Standard_LRS |
| Container name | highlights |
| Public access | off |

**What is stored:** Only the final annotated output video (highlight reel) is uploaded to Blob. Raw match footage is transient — it is uploaded by the browser to `/mnt/data/jobs/<job_id>/` on the VM, processed, and deleted on completion.

**How it is accessed:** `webapp/blob.py` uploads the output video and returns a time-limited SAS URL to the caller. It reads the connection string from the `AZURE_STORAGE_CONNECTION_STRING` environment variable.

**Secret location:** The connection string is stored in `/etc/gsfa-highlights.env` on the VM (file mode 600, not committed to the repo). Obtain it from the Azure portal: Storage account → Access keys → Connection string.

```bash
# Example env file structure (values redacted):
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...;AccountKey=<REDACTED>;EndpointSuffix=core.windows.net
USE_GPU=1
MAX_UPLOAD_GB=15
```

### 2c. Google Colab (non-Azure cloud path)

Two scripts (`gsfa_colab_pipeline.py`, `shots_on_t.py`) are written to run directly in Google Colab on a T4 GPU runtime. They are self-contained single-file versions of the modular local pipeline. Model weights are copied to `/content/` before running. No Azure infrastructure is involved in the Colab path — output videos are saved locally in the Colab session.

---

## 3. Processing pipeline — stage by stage

### 3a. Player and ball detection

**Player detector** (`detectors/player_detector.py`, class `PlayerDetector`)

- Model: `GSFA_PLAYER_DETECTION.pt` — YOLOv11 fine-tuned on GSFA futsal footage.
- Three output classes: `0 = active_player`, `1 = goal_post`, `2 = referee`.
- Inference confidence threshold: 0.50.
- Runs on CPU by default; set `device="cuda"` for GPU.
- Output per frame: `FrameDetections` dataclass containing lists of `Detection` objects, each with `bbox (x1,y1,x2,y2)`, `foot_point` (bottom-centre of bbox — used for ground-plane projection), `centre_point`, `team_id`, `is_goalkeeper`, `track_id`, and a 768-D `embedding`.

**Ball detector** (`BallDetector` class in `video_analysis/possession.py` and `gsfa_colab_pipeline.py`)

- Model: `gsfa_ball_detection.pth` — RF-DETR Medium checkpoint, `num_classes=2`, resolution 576.
- Ball is `class_id=1`, confidence threshold 0.25 (intentionally low — Kalman gating filters false positives).
- Input frame is converted BGR→RGB→PIL before inference.
- Returns the highest-confidence ball detection in each frame, or `None`.

In `shots_on_t.py` a more aggressive gated cascade is applied before the tracker:
- **Gate 1 (Court mask):** HSV-segmented futsal court (blue floor, HSV `[90,40,40]`–`[130,255,255]`). Candidates outside the court mask are discarded immediately.
- **Gate 2 (Size/shape):** 0.5x–2x expected ball diameter (~18 px at typical camera distance).
- **Gate 3 (Distance gate):** Adaptive radius = `GATE_BASE_PX + 1.5 * |velocity|` (~55 px base). Only the nearest in-gate candidate is considered.
- **Gate 4 (Physics gate):** Hard position-innovation ceiling of 60 px/frame. Kicks are large accelerations and pass; teleports are large position jumps and fail.

### 3b. Ball tracking (Kalman filter)

**In the possession pipeline** (`BallTracker` class):

- Classic constant-velocity Kalman filter: state `[x, y, vx, vy]`, measurement `[x, y]`.
- Process noise: `diag([4, 4, 25, 25])`. Measurement noise: `diag([9, 9])`.
- Coasts for up to 12 frames after the last accepted detection (emits predicted position), then transitions to `LOST`.
- Mahalanobis gate (`gate_sigma=6.0`) rejects outlier detections.
- Three states: `DETECTED` / `COASTING` / `LOST`.

**In the shots pipeline** (`BallTrack` class in `shots_on_t.py`):

- FilterPy Kalman filter with the same constant-velocity model but separate noise tuning.
- Four-state machine: `INACTIVE → TENTATIVE → CONFIRMED → COASTING`.
- TENTATIVE requires 3 consecutive consistent detections to promote to CONFIRMED, so a single false positive can never start a real track.
- Camera motion compensation: the track state is shifted by the per-frame pan vector before association, so the gate stays glued to the ball in current-frame pixel coordinates.

### 3c. Team classification

**Active classifier: `GSFATeamClassifier`** (`team_classifier/team_classifier.py`)

This is the production classifier. It must always be used — never swap in `ColourHistogramTeamClassifier`.

**How it works:**

1. At fit time, sample 1 fps from the video, detect players, crop the top 55% of each bbox (jersey region only — excludes legs and court floor), filter out blurry crops (Laplacian variance threshold 80), and collect all valid crops.
2. Run all crops through Google's SigLIP vision model (`google/siglip-base-patch16-224`) to get 768-D embeddings.
3. Reduce (N, 768) → (N, 3) with UMAP.
4. Cluster into k=2 with KMeans — one cluster per team.
5. Save the fitted model to `data/cache/<video_stem>_team_siglip.pkl`.

**At inference time (per frame):**

- Same torso crop is extracted.
- Crops go through SigLIP → UMAP → KMeans predict.
- `team_id` (0 or 1) and the raw 768-D SigLIP embedding are written onto each `Detection` in-place.
- The 768-D embedding is also passed to BoT-SORT as the ReID feature.

**Why this works better than colour histograms:** SigLIP embeddings carry texture, logo, and colour information together. The torso crop + blur gate removes background contamination that caused the earlier approach (SigLIP on full bbox) to cluster by pitch position rather than jersey colour on the panning camera. The 768-D embedding doubles as the ReID feature for the tracker.

**Deprecated alternative: `ColourHistogramTeamClassifier`** (`team_classifier/colour_histogram.py`)

HSV histogram (H: 64 bins + S: 32 bins) → (N, 96) → KMeans(k=2) directly on histograms. Not wired into any active pipeline. Used only in `heatmap.py` (which is a utility script, not the main possession pipeline).

### 3d. Goalkeeper detection

**Class:** `GoalkeeperDetector` (`detectors/goalkeeper_detector.py`)

**Fit phase (once per match):**
1. Sample 1 fps; for each detected goal post, find the closest active player by foot-point distance.
2. Average the closest-player positions over all sampled frames to get two GK zone centroids (one per post, sorted left-to-right).
3. Compute average positions of all team-0 and team-1 outfield players; assign each GK zone to the nearer team.
4. Save to `data/cache/<video_stem>_goalkeeper.pkl`.

**Per-frame classify:**
- For each detected goal post, find the closest active player, mark `is_goalkeeper=True`, and override `team_id` with the fitted assignment (corrects any misclassification by the team classifier near the goal).

### 3e. Player tracking

**Class:** `PlayerTracker` (`tracking/player_tracker.py`)

- Algorithm: BoT-SORT from the `boxmot` library.
- ECC global motion compensation (`cmc_method="ecc"`) handles camera pans — the entire track state shifts with the camera.
- ReID: the 768-D SigLIP embeddings from `GSFATeamClassifier.classify()` are passed as the appearance matrix. Detections without an embedding fall back to motion-only association.
- Key thresholds: `track_high_thresh=0.5`, `track_low_thresh=0.1`, `new_track_thresh=0.6`, `match_thresh=0.8`, `proximity_thresh=0.5`, `appearance_thresh=0.25`.
- Track buffer: 60 frames at the native frame rate (~2 seconds) — long enough to survive a pass but short enough not to re-use IDs that have left the frame.
- Writes `track_id` onto each `Detection` in-place.

### 3f. Possession detection (CarrierEngine)

**Class:** `CarrierEngine` (`video_analysis/possession.py`, duplicated in `gsfa_colab_pipeline.py`)

The video is sub-sampled to 15 fps (`TARGET_PROCESS_FPS`; `frame_step = round(native_fps / 15)`).

**Ball-to-player assignment per processed frame:**

1. For each tracked player (team 0 or team 1, with a `track_id`), compute the foot-zone radius: `clamp(0.45 * bbox_height, 20px, 140px)`. This scales naturally with the player's apparent size in the panning camera.
2. Measure Euclidean distance from the ball centre to the player's foot point.
3. Collect all players within their foot zone radius.
4. Zero players in zone → `LOOSE`. Multiple teams in zone → `CONTESTED`. One team only → `carrier` state for the closest player.
5. Ball tracker `LOST` → `OOF` (out of frame).

**Hysteresis:** A state change is only committed after 3 consecutive processed frames agree (prevents flicker from single-frame occlusions).

**PossessionStats denominator:** Only `team0_frames + team1_frames` count. Loose, contested, and OOF frames are excluded from the percentage calculation.

### 3g. Pass detection (PassEventTracker)

**Class:** `PassEventTracker` (`video_analysis/possession.py`, duplicated in `gsfa_colab_pipeline.py`)

A 3-phase finite state machine operating at 15 fps on `CarrierState` transitions:

| Phase | Trigger | Action |
|---|---|---|
| `IDLE` | Any carrier detected | → `IN_POSSESSION` |
| `IN_POSSESSION` | Ball leaves carrier's foot zone | → `CAND_RELEASE` |
| `CAND_RELEASE` | Wait 1 frame — same carrier returns → dribble cancel; otherwise | → `TRAVEL` |
| `TRAVEL` | Another player enters foot zone (after ≥1 frame gap) | → `CAND_RECEPTION` |
| `TRAVEL` | 22 frames (~1.47 s) without reception | → `EVT_BALL_LOST` → `IDLE` |
| `CAND_RECEPTION` | Same candidate holds for 2 frames | → resolve reception |
| Resolution: same team | — | `EVT_COMPLETED` (successful pass) |
| Resolution: other team | — | `EVT_INTERCEPTION` (inaccurate pass, attributed to passer's team) |

**Pass accuracy = successful / (successful + inaccurate) per team.**

**Retroactive possession corrections:** When an interception is resolved, travel frames that were credited to the passer's team are flipped to the receiving team (`flip_to` adjustment). When `EVT_BALL_LOST` fires, travel frames are dropped to OOF (`drop` adjustment). These corrections keep the possession denominator honest.

### 3h. Shots on target (`shots_on_t.py`)

This is a separate, standalone Colab script that runs an independent pipeline focused on counting shots on target. It uses the same ball and player models but adds:

**Goal plane detection via SAM2:**
- YOLO detects goal post bboxes. SAM2-small (`sam2.1_hiera_small.pt`) segments the goal post, white-pixel HSV filtering and convex-hull approximation extract the 4 corners of the goal opening.
- Side assignment: goal post whose bbox centre is left of the frame midline → LEFT goal, else RIGHT.
- The plane exists only while YOLO is actively detecting the goal post bbox. If YOLO does not see it for 12 frames, the plane is dropped (prevents drift across camera pans).
- When a new extraction matches the existing plane (centroid difference < plane width), an EMA with alpha=0.35 refines it; otherwise it is replaced outright.

**Camera motion compensation:**
- Median sparse optical flow on off-court features (inverse of the HSV court mask) estimates the per-frame pan vector `(dx, dy)`.
- The ball track state, trail, and all goal planes are shifted by this vector each frame.
- Pan exceeding 2.5 px/frame suppresses shot detection entirely for that frame.

**Shot detection logic:**
1. Every CONFIRMED ball frame predicts the trajectory 1.2 seconds ahead using a constant-velocity + floor-friction model (friction decay 0.985 per frame).
2. The predicted path is tested against each goal plane polygon using point-in-polygon.
3. A shot is counted when the path hits the plane on ≥3 consecutive frames while the ball moves toward that goal at ≥6 px/frame (minimum speed gate).
4. Cooldown: 45 frames (~1.5 s) between counted shots per goal to prevent duplicates.

**Hindsight validation:**
- A provisional shot is counted live (shown in the HUD).
- For the next ~0.8 s (24 frames) of confirmed track, the script checks whether the ball actually came within 60% of the goal-plane width from the plane. If not, the count is revoked.
- The final report shows both `provisional` and `validated` counts.

### 3i. Keypoint homography and pitch mapping

**File:** `video_analysis/simple_homography.py`, reused by `heatmap.py`

**How it works:**
- Model: `final_best.pt` — YOLO11m-pose trained on "Football field" single-class with 13 keypoints, 76 epochs on Colab (mAP50-pose=0.985 on training data).
- The model outputs 13 keypoints per detected field, each with `(x, y, confidence)`.
- Keypoints 0 and 6 (goal posts, 3-D structures) are excluded from homography computation. Corners (1, 5, 7, 11) use a lower confidence threshold (0.35) since they are geometrically unambiguous even at moderate confidence.
- A geometric spread guard rejects homographies where the used keypoints span less than 30% of the pitch width (prevents extrapolation errors when the camera sees only one end of the court).
- Per-frame homography H maps image pixel → world metres (40 m × 20 m pitch, origin at far-left corner).
- Frames without a solvable H (too few confident keypoints) are filled by linear interpolation of H from adjacent solved frames.
- The result: the pitch boundary, halfway line, and goal mouths are overlaid onto the video as a white wireframe; a top-down coverage heatmap (`pitch_coverage_map.png`) shows which areas of the pitch were visible.

**13 keypoint layout** (world coords in metres, origin = far-left corner):

```
pt11(0,0) — pt10(10,0) — pt9(20,0) — pt8(30,0) — pt7(40,0)   ← far touchline
   |                         |                        |
  pt0(0,10)               pt12(20,10)              pt6(40,10)  ← goal-post level
   |                         |                        |
pt1(0,20) — pt2(10,20) — pt3(20,20) — pt4(30,20) — pt5(40,20) ← near touchline
```

**Caching:** `build_h_list()` saves the per-frame H matrix list to `data/cache/<video_stem>_H.pkl`; subsequent calls load it instantly.

### 3j. Player heatmaps (`heatmap.py`)

Runs PlayerDetector + GoalkeeperDetector + homography on every frame, projects each player's foot point to world coordinates via H, and accumulates hit counts in a 2-D grid. Gaussian blur (sigma=20 px) smooths the raw counts. Outputs two PNG files:
- `data/output/team0_heatmap.png` (COLORMAP_WINTER, blue tones)
- `data/output/team1_heatmap.png` (COLORMAP_HOT, red/orange tones)

Note: `heatmap.py` uses `ColourHistogramTeamClassifier` (not `GSFATeamClassifier`) because it predates the SigLIP integration and is a utility script, not the main pipeline.

### 3k. Early-stage OCR scripts (legacy / exploratory)

`scripts/01_ocr_score_overlay.py` and related scripts in `scripts/` are early prototypes that used EasyOCR on the scorebug ROI to detect score changes. These were part of the v1 approach described in `approach.md`. The current main pipeline (`video_analysis/possession.py`) does not use OCR at all — possession is derived from player and ball tracking, not from reading the scoreboard. The OCR feature was removed (commit `e4126bc`).

---

## 4. Repository structure

```
D:\GSFA_highlights\
│
├── video_analysis/
│   ├── possession.py          Main modular pipeline (local runs)
│   ├── simple_homography.py   Keypoint pose → H matrix → pitch wireframe + heatmap
│   ├── inspect_log.py         Utility: print keypoint_log.json stats
│   └── inspect_log2.py        Utility: deeper log analysis
│
├── detectors/
│   ├── player_detector.py     PlayerDetector (YOLO11), Detection/FrameDetections types
│   ├── goalkeeper_detector.py GoalkeeperDetector (spatial fit)
│   ├── cache.py               Shared cache path utility (data/cache/)
│   └── test_goalkeeper.py     Unit test for goalkeeper detection
│
├── team_classifier/
│   ├── team_classifier.py     GSFATeamClassifier (SigLIP + UMAP + KMeans) [ACTIVE]
│   └── colour_histogram.py    ColourHistogramTeamClassifier [DEPRECATED, not wired into main pipeline]
│
├── tracking/
│   └── player_tracker.py      PlayerTracker (BoT-SORT + ECC + SigLIP ReID)
│
├── scripts/                   Exploratory / diagnostic scripts
│   ├── 01_ocr_score_overlay.py   OCR prototype (EasyOCR on scorebug)
│   ├── 02_team_classification.py Team classifier test harness
│   ├── crop_scoreboard.py        Utility: crop scoreboard region
│   ├── diagnose_teams.py         Team classification debugging
│   └── test_endgame_ocr.py       End-game OCR test
│
├── data/
│   ├── cache/
│   │   ├── video_project_8_team_siglip.pkl   Fitted GSFATeamClassifier for match 8
│   │   └── video_project_8_goalkeeper.pkl    Fitted GoalkeeperDetector for match 8
│   ├── debug/team_test/          Debug frame crops from team classifier testing
│   └── colour_team_classifier.pkl            Colour histogram classifier (legacy)
│
├── gsfa_colab_pipeline.py    Single-file Colab version of the full possession+pass pipeline
├── shots_on_t.py             Single-file Colab shots-on-target pipeline (SAM2 + ball tracking)
├── heatmap.py                Per-team player position heatmap generator
├── approach.md               Design document describing v1 strategy and v2 roadmap
├── requirements.txt          Python dependencies
│
├── docs/
│   └── azure_deploy.md       Full Azure VM setup guide (Docker, CUDA, Blob Storage, NSG)
│
├── highlights/               Python virtual environment (in .gitignore)
└── .claude/
    ├── agents/opencv-highlight-pipeline.md  Sub-agent definition
    └── agent-memory/opencv-highlight-pipeline/  Persistent agent memory files
```

---

## 5. How to run

### Prerequisites

Python 3.11 or 3.13 with the following packages (see `requirements.txt`):

```
torch torchvision  (cu124 wheel — requires CUDA 12.4 for GPU)
opencv-python
transformers       (for SigLIP via sports library)
umap-learn
supervision
ultralytics        (YOLO)
scikit-learn
rfdetr             (ball detector)
boxmot             (BoT-SORT tracker)
joblib
Pillow
more-itertools
tqdm
```

The `sports` library (Roboflow, provides `sports.common.team.TeamClassifier` — the SigLIP+UMAP+KMeans wrapper) must be installed from source:
```bash
pip install git+https://github.com/roboflow/sports.git
```

For the shots-on-target script only, SAM2 is also required:
```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

### Model weights

Model weights are not committed to the repo. They live at local paths on the developer machine or must be copied to `/content/` in Colab:

| Model | Default local path | Purpose |
|---|---|---|
| GSFA_PLAYER_DETECTION.pt | `C:\Users\Admin\OneDrive\Desktop\CZ\GSFA_PLAYER_DETECTION.pt` | YOLOv11 player/post/referee |
| gsfa_ball_detection.pth | `C:\Users\Admin\OneDrive\Desktop\CZ\gsfa_ball_detection.pth` | RF-DETR ball detector |
| final_best.pt | `C:\Users\Admin\OneDrive\Desktop\CZ\GSFA_keypoint\final_best.pt` | YOLO11m-pose field keypoints |

### Running the modular pipeline (local)

```bash
cd D:\GSFA_highlights
highlights\Scripts\python.exe video_analysis\possession.py
```

The `VIDEO_PATH`, `BALL_MODEL_WEIGHTS`, and `OUTPUT_PATH` constants at the top of `video_analysis/possession.py` must be edited before running. On first run, the team classifier and goalkeeper detector are fitted from scratch (~minutes); on subsequent runs the cached `.pkl` files are loaded in seconds.

### Running the Colab pipeline

1. Open Google Colab, select Runtime → T4 GPU.
2. Copy `gsfa_colab_pipeline.py` into a single cell (or upload it and run `!python gsfa_colab_pipeline.py`).
3. Upload model weights to `/content/` and edit the CONFIG block at the top.
4. Run the cell. Install command at the top of the file.

```python
# Colab install (paste first):
!pip install ultralytics supervision rfdetr boxmot joblib umap-learn scikit-learn Pillow
!pip install git+https://github.com/roboflow/sports.git
```

### Running shots-on-target (Colab)

Same Colab workflow with `shots_on_t.py`. The script auto-installs missing packages on first run. SAM2 checkpoint (`sam2.1_hiera_small.pt`) is downloaded automatically on first goalpost detection. Edit the `CONFIG` block at the top for video path and model paths.

### Running homography overlay

```bash
cd D:\GSFA_highlights
highlights\Scripts\python.exe video_analysis\simple_homography.py
```

Outputs: `data/homography_overlay.mp4`, `data/pitch_coverage_map.png`, `data/keypoint_log.json`, and debug frames in `data/debug/`.

### Running team heatmaps

```bash
cd D:\GSFA_highlights
highlights\Scripts\python.exe heatmap.py
```

Outputs: `data/output/team0_heatmap.png`, `data/output/team1_heatmap.png`.

### On the Azure VM (production path)

The deployment is Docker-based. See `docs/azure_deploy.md` for the full guide. The high-level steps are:

1. Create `/etc/gsfa-highlights.env` with `AZURE_STORAGE_CONNECTION_STRING`, `USE_GPU=1`, `MAX_UPLOAD_GB=15`.
2. Clone the repo to `/opt/gsfa-highlights/repo`.
3. `docker compose up -d --build`.
4. Verify: `curl http://localhost:8000/health` → `{"ok":true,"use_gpu":true,...}`.

The FastAPI app (`server.py`, not committed to the repo at time of writing) accepts multipart video uploads up to 15 GB, processes them through the pipeline, and uploads the output reel to the `highlights` container in Azure Blob Storage, returning a SAS URL.

---

## 6. Key configuration parameters

All tunable parameters are in the CONFIG block at the top of each script. The most important ones:

| Parameter | Value | Meaning |
|---|---|---|
| `TARGET_PROCESS_FPS` | 15.0 | Sub-sample rate for the main analysis loop |
| `FOOT_ZONE_RATIO` | 0.45 | Foot zone radius = 45% of player bbox height |
| `FOOT_ZONE_MIN_PX` | 20 | Minimum foot zone radius |
| `FOOT_ZONE_MAX_PX` | 140 | Maximum foot zone radius |
| `CARRIER_HYSTERESIS_N` | 3 | Frames of agreement required before committing carrier change |
| `TRAVEL_TIMEOUT_FRAMES` | 22 | ~1.47 s — pass travel timeout at 15 fps |
| `KALMAN_COAST_FRAMES` | 12 | Frames ball tracker emits predicted position after losing detection |
| `TORSO_RATIO` | 0.55 | Top 55% of player bbox used for jersey crop |
| `BLUR_THRESHOLD` | 80 | Laplacian variance — blurry crops below this are skipped at fit time |
| `TC_SAMPLE_EVERY` | 30 | Sample every 30th frame (1 fps at 30 fps) for team classifier fitting |
| `PITCH_W / PITCH_H` | 40.0 / 20.0 m | Real futsal court dimensions |

---

## 7. What is not in the repo

The following are referenced in the code or docs but are not committed:

- Model weights (`GSFA_PLAYER_DETECTION.pt`, `gsfa_ball_detection.pth`, `final_best.pt`) — stored locally on the developer's OneDrive.
- `webapp/blob.py` and `server.py` (FastAPI web app) — referenced in `docs/azure_deploy.md` but not present in the repo at the time this overview was written.
- `Dockerfile` and `docker-compose.yml` — referenced in `docs/azure_deploy.md` but not committed.
- `/etc/gsfa-highlights.env` — secrets file on the VM, never committed.
- Large video files — all match footage (`Video Project *.mp4`) is local only.
- The `data/` directory (except cache `.pkl` files) is in `.gitignore`.
- The `highlights/` Python virtual environment is in `.gitignore`.
