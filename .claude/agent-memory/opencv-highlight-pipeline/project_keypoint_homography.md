---
name: project-keypoint-homography
description: Model inspection findings and open questions for the keypoint homography reprojection pipeline (homography_test.py)
metadata:
  type: project
---

## Model facts (confirmed 2026-05-22)

- Path: `C:\Users\Admin\OneDrive\Desktop\CZ\GSFA_keypoint\final_best.pt`
- Task: pose, class = "Football field" (single class, max_det=1 at training)
- Keypoint shape: [13, 3] — exactly 13 keypoints per detection, each with (x, y, conf)
- Backbone: yolo11m-pose (Ultralytics 8.4.53, trained 76 epochs on Colab GPU)
- Training mAP50(P)=0.985, mAP50-95(P)=0.935 — strong pose accuracy on training data
- Training data yaml path in checkpoint: `/content/GSFA_Keypoint-1/data.yaml` (Colab path, not accessible)
- Box confidence on TEST_1_YOLO.mp4 is low (~0.28–0.36); max_det=1 was used at training time
- Keypoint confidences on TEST_1_YOLO.mp4 are generally very low (most < 0.5); rarely more than 1–2 keypoints above 0.5 in any frame
- val_batch0_pred.jpg shows full-field overhead-ish camera angles — training data appears to differ from panning sideline video

## Open questions (pending user answers)

1. What is the test video path? (TEST_1_YOLO.mp4 at `C:\Users\Admin\OneDrive\Desktop\CZ\` is ~30s 1080p, may be the intended video — needs confirmation)
2. What are the pitch coordinate mappings for all 13 keypoints (index 0–12 → (x_m, y_m) on 40×20m court)?
3. Should the confidence threshold be lowered below 0.5 given low scores on this video, or does user have a different test video where detections are stronger?
4. The model was trained with max_det=1 — at inference we should set max_det=1 and take that single detection rather than selecting by box conf.

**Why:** The data.yaml with keypoint label definitions was on a Colab runtime and is not locally accessible. The 13-keypoint layout (which corners, penalty spots, etc.) must be confirmed by the user before the pitch coordinate map can be hard-coded.

**How to apply:** Do not write homography_test.py until the user provides the keypoint→pitch-coordinate mapping. Prototype can be written assuming a placeholder mapping, but it must be parameterized so the user can fill in the real values.
