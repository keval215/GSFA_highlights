# Futsal Highlight Generation Pipeline — My Approach

## Context

- **Sport:** Futsal (6v6, indoor stadium)
- **Camera:** Single fixed camera
- **Overlays:** Custom-designed, fixed position at the start of the match, no replays in match
- **Constraint:** No audio-based signals (excluded by design)
- **Goal:** Auto-generate highlight reels from full-match footage without player/ball tracking

---

## Solution Overview

The pipeline combines two complementary signals:

1. **Overlay-based event detection** — read the scorebug to catch every goal with high precision
2. **Visual motion analysis** — use optical flow and goal-area presence to catch saves, near-misses, and attacking phases

No deep-learning fine-tuning, no audio processing, no player tracking in v1.

---

## Stage 1 — Overlay Localization (one-time per match)

### Approach: OpenCV Template Matching

Since the overlay design is custom and its position is fixed at the start of every match, template matching is sufficient and avoids any model training.

**Steps:**

1. Save one reference image of the empty overlay (clean scorebug, no scores filled in) as `overlay_template.png`.
2. On the first frame of the match, run `cv2.matchTemplate` to locate the overlay's anchor point in the frame.
3. Cache those pixel coordinates for the rest of the match — the overlay does not move.
4. From those coordinates, derive the sub-regions for the **home score digit** and **away score digit** as fixed offsets.

**Code outline:**

```python
import cv2

# Load the template once
template = cv2.imread("overlay_template.png", cv2.IMREAD_GRAYSCALE)
th, tw = template.shape

# Run template matching on the first frame
gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
_, max_val, _, top_left = cv2.minMaxLoc(result)

# Anchor point of the overlay
overlay_x, overlay_y = top_left

# Derive score-digit ROIs as fixed offsets from anchor
# (these offsets are measured once from the overlay design)
HOME_DIGIT_OFFSET = (x1, y1, x2, y2)
AWAY_DIGIT_OFFSET = (x3, y3, x4, y4)

home_roi = lambda f: f[overlay_y + y1 : overlay_y + y2,
                       overlay_x + x1 : overlay_x + x2]
away_roi = lambda f: f[overlay_y + y3 : overlay_y + y4,
                       overlay_x + x3 : overlay_x + x4]
```

**Why this works:**
- Real-time on CPU (no GPU needed for this stage)
- Zero training data
- Tolerates small position shifts between recordings
- Single point of failure (anchor lookup) — easy to debug

---

## Stage 2 — Score Reading (OCR)

### Approach: PaddleOCR with digit-only configuration

For v0, no training. Test a single image on the OCR first to validate accuracy on the actual overlay font before integrating into the pipeline.

**Why PaddleOCR over alternatives:**

| OCR | Verdict |
|-----|---------|
| **PaddleOCR** | Best balance of accuracy and speed for digits in clean overlays. Use this. |
| EasyOCR | Easiest to install but less accurate on digit-only tasks. Acceptable backup. |
| TrOCR | Transformer-based, very accurate but heavy and slow. Overkill for 0–9 digits. |
| Tesseract | Older, weaker on stylized fonts and small text. Skip. |

**Initial validation step:**

1. Extract one frame from a match where the score is clearly visible.
2. Crop the home-score digit and the away-score digit using the ROIs from Stage 1.
3. Run PaddleOCR on each crop.
4. Verify the recognized digits match what's actually on screen.
5. If accuracy is poor, fall back to training a tiny digit CNN later.

**Code outline:**

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)

def read_digit(crop):
    # Optional: upscale small crops (digits can be tiny in the overlay)
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    result = ocr.ocr(crop, cls=False)
    if result and result[0]:
        text = result[0][0][1][0]    # extract recognized string
        digits = ''.join(c for c in text if c.isdigit())
        return int(digits) if digits else None
    return None
```

**Sampling rate:** sample at 1–2 fps. Reading the score on every frame is wasteful — score only changes during goals.

**Stability filter:** a score change must persist for ≥3 consecutive samples before being committed as a real change. This handles any rare OCR misreads.

---

## Stage 3 — Goal Detection and Highlight Clipping

**Logic:**

1. Maintain a rolling state: `(home_score, away_score)`.
2. On each sampled frame, OCR the digits.
3. If the score changes and the change persists for ≥3 samples, mark that timestamp as `t_goal`.
4. Extract the highlight clip: `[t_goal - 10s, t_goal + 5s]`.
   - 8 seconds before captures the buildup (futsal possessions are fast — 6–10 seconds is the right window).
   - 5 seconds after captures the immediate reaction (the ball going in, brief celebration start).
5. Save the clip with FFmpeg.

**Sanity rules:**
- Scores only increase, never decrease.
- Score deltas of more than 1 in a single change are suspicious — log and skip.
- Cap goals per match at a sane maximum (e.g., 25) to catch runaway OCR errors.

---

## Stage 4 — Visual Motion Analysis (Tier 2)

This is the layer that catches the events the overlay can never tell us about: **saves, near-misses, attacking phases, breakaways, goalmouth scrambles.**

### 4a — Mean Optical Flow Magnitude

Detects fast collective motion = transitions, counter-attacks, chaotic play near goal.

```python
import cv2
import numpy as np

def frame_motion(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return mag.mean()
```

**Usage:**

- Run at a downsampled resolution (e.g., 320×180) for speed — magnitude is what matters, not detail.
- Compute `frame_motion()` for every consecutive frame pair (or every Nth pair to save compute).
- Build a 1D time series of motion values across the match.
- Compute a rolling baseline (e.g., 60-second median).
- Flag windows where motion exceeds the 90th percentile of the baseline for ≥2 consecutive seconds → candidate exciting window.

### 4b — Spatial Heat-Map of Activity

Detects which part of the court is active. Concentrated activity in a goal area = attacking phase.

**Approach:**

1. Divide the frame into a 4×3 grid (12 cells covering the court).
2. For each consecutive frame pair, compute optical flow and split it into the 12 cells.
3. Compute the mean flow magnitude per cell.
4. Build a per-cell time series.
5. Flag intervals where the cells corresponding to either goal area show high motion for ≥3 seconds.

**Rules:**
- Map the two goal-area cells once at setup (since the camera is fixed).
- A "goal-area attacking phase" = sustained high motion in one of those cells, ≥3 seconds.
- Clip `[t_start - 4s, t_end + 3s]` for the highlight.

### 4c — Goal-Area Presence

Detects how many players are in a goal area. Count ≥3 in one area for ≥2 seconds = high-pressure moment, likely a save or chance.

**Two implementation options:**

**Option 1 — Background subtraction (no model, fast):**

```python
backsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16)

def players_in_region(frame, region_mask):
    fg = backsub.apply(frame)
    fg_in_region = cv2.bitwise_and(fg, fg, mask=region_mask)
    contours, _ = cv2.findContours(fg_in_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours by minimum area to count player-sized blobs
    players = [c for c in contours if cv2.contourArea(c) > MIN_PLAYER_AREA]
    return len(players)
```

This is rough but works. Empty court = no foreground; players in the goal area = blobs in the region mask.

**Option 2 — YOLO person detector (better accuracy):**

Use a pretrained YOLOv8n. No fine-tuning needed since "person" is a default class.

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")

def players_in_region(frame, region_polygon):
    results = model(frame, classes=[0], verbose=False)  # class 0 = person
    boxes = results[0].boxes.xyxy.cpu().numpy()
    centers = [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in boxes]
    return sum(1 for c in centers if cv2.pointPolygonTest(region_polygon, c, False) >= 0)
```

**Either way:**
- Define the two goal-area polygons once at setup.
- Count players per region every 0.5s.
- Flag intervals where count ≥3 in either region for ≥2 seconds.
- Clip `[t_start - 5s, t_end + 3s]`.

---

## Stage 5 — Highlight Stitching

**Inputs:** list of `(t_start, t_end, event_type)` tuples from all stages above.

**Steps:**

1. Sort by `t_start`.
2. Merge overlapping windows (a goal often coincides with a motion spike — don't double-clip).
3. Use FFmpeg's `concat` demuxer to stitch clips:

```bash
ffmpeg -f concat -safe 0 -i clips.txt -c copy highlights.mp4
```

Where `clips.txt` is a list of `file 'clip_001.mp4'` lines.

4. Optional: add 0.5s fade transitions between clips.

---

## Build Order

| Step | Goal | Effort |
|------|------|--------|
| 1 | Validate OCR on a single overlay crop | 30 min |
| 2 | Implement overlay localization with template matching | 2 hours |
| 3 | Implement score-change detection over a full match | 4 hours |
| 4 | Test goal-clip extraction on a real match | 2 hours |
| 5 | Add optical flow magnitude detector | 4 hours |
| 6 | Add spatial heatmap | 4 hours |
| 7 | Add goal-area presence detection | 6 hours |
| 8 | Stitching pipeline with FFmpeg | 2 hours |

Realistic timeline: ~1 week for v1 working end-to-end on one match.

---

## Tier 5 — Future Path (Player and Ball Tracking)

This is the eventual depth layer if v1 turns out to need more nuance. **Not part of v1 — defer until proven necessary.**

### Why defer

- Significant engineering effort (multi-week)
- Requires labeled futsal data (no public dataset exists for futsal specifically)
- Ball is small, fast, and frequently occluded — hardest part of the system
- v1 (overlay + motion) likely covers 80%+ of highlight-worthy moments already

### What it would unlock

From player and ball tracks, we can derive event semantics that motion alone cannot:

- **Sudden ball acceleration** → shot taken (distinguishes a real shot from a pass)
- **Ball entering goal area + GK movement** → save attempt
- **Player going to ground + nearby player + brief stoppage** → foul
- **GK leaving the penalty area** → distinctive futsal "flying GK" power-play tactic — usually signals tactical aggression, often a key moment
- **Ball trajectory hitting the post/crossbar** → near-miss (currently invisible to v1)
- **Long-range shot detection** (ball traveling >10m before a save/goal) → quality moment

### Implementation sketch

**Player tracking:**
1. Fine-tune YOLOv8 on a few hundred labeled futsal frames (player, GK, referee, ball as classes).
2. Use ByteTrack or BoT-SORT on top for multi-object tracking.
3. Output: per-player track with timestamps, court positions (after homography), team identity.

**Ball tracking:**
1. Ball detection is the bottleneck — a futsal ball is small (~20–30 px in a 1080p frame from typical camera distance).
2. Options:
   - Train a dedicated small-object detector on ball crops only
   - Use a TrackNet-style architecture (designed for small fast-moving balls in racquet sports — works for futsal)
3. Apply trajectory smoothing (Kalman filter) to handle occlusions.

**Court calibration:**
1. Compute a homography from camera image to court coordinates using the visible court lines.
2. This converts pixel positions to real-world meters, enabling speed/distance metrics.

**Event detection from tracks:**
- Build classical rules over track features (ball speed, player density, region transitions).
- Or train a small classifier on track-derived features (ball speed, player count per region, distances to goal) for `goal / shot / save / foul / pass / nothing`.

### Decision criterion

Only build Tier 5 if v1 evaluation on real matches shows that:
- **Highlight precision is good but recall is bad** (you're missing too many interesting moments), AND
- **The specific missed moments require semantic understanding** (e.g., users complain about missed great saves, missed posts, missed nutmegs)

If the missed moments are mostly things audio cues would have caught — well, audio is excluded by design, so this becomes the only path forward.

---

## Open Questions to Resolve During v1

1. What's the actual OCR accuracy on the real overlay font? Test on 50+ frames before committing.
2. What's the right backtrack window for futsal goals? Start with 8s, tune on real data.
3. What threshold percentile defines "high motion" for this specific camera setup?
4. Does background subtraction give clean enough player blobs, or is YOLOv8 needed?
5. How long is a typical highlight reel for a single futsal match — and is that what users actually want?



Note: Before training our player clustering model, we need to gather training data. To do this, we'll sample one frame per second, detect players within those frames, and then crop them out. from tqdm import tqdm

SOURCE_VIDEO_PATH = "/content/121364_0.mp4"
PLAYER_ID = 2
STRIDE = 30

frame_generator = sv.get_video_frames_generator(
    source_path=SOURCE_VIDEO_PATH, stride=STRIDE)

crops = []
for frame in tqdm(frame_generator, desc='collecting crops'):
    result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
    detections = sv.Detections.from_inference(result)
    detections = detections.with_nms(threshold=0.5, class_agnostic=True)
    detections = detections[detections.class_id == PLAYER_ID]
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
    crops += players_crops 

    Note: Next, we'll run SigLIP to calculate embeddings for each of the crops. 
    import torch
from transformers import AutoProcessor, SiglipVisionModel

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(DEVICE)
EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH) 
import numpy as np
from more_itertools import chunked

BATCH_SIZE = 32

crops = [sv.cv2_to_pillow(crop) for crop in crops]
batches = chunked(crops, BATCH_SIZE)
data = []
with torch.no_grad():
    for batch in tqdm(batches, desc='embedding extraction'):
        inputs = EMBEDDINGS_PROCESSOR(images=batch, return_tensors="pt").to(DEVICE)
        outputs = EMBEDDINGS_MODEL(**inputs)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
        data.append(embeddings)

data = np.concatenate(data)


Note: Using UMAP, we project our embeddings from (N, 768) to (N, 3) and then perform a two-cluster division using KMeans. 
import umap
from sklearn.cluster import KMeans

REDUCER = umap.UMAP(n_components=3)
CLUSTERING_MODEL = KMeans(n_clusters=2)
To simplify the use of the SigLIP, UMAP, and KMeans combo, I've packaged all these models into a TeamClassifier that you can find in the sports repository.

Note: Time to assign goalkeepers to teams. We'll use a simple heuristic: calculate the average position (centroid) of the players belonging to both teams and then assign the goalkeeper to the team whose average position is closer.