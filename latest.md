# Clip-processing performance — latest status

_Goal: get a 60s clip under 100s at full 15fps, no accuracy loss (frame-skipping is off the table — it cost ~50% accuracy)._

## Hardware
Azure **NC4as T4 v3**: 1× T4 GPU (16GB VRAM), **4 vCPUs (the tight resource)**, 28GB RAM.

## What's been implemented (committed to `test`)
1. **fp16 + imgsz on YOLO** — `detectors/player_detector.py` (`half=True` on CUDA, `imgsz=640`) + new `detect_batch()`.
2. **RF-DETR `optimize_for_inference()` + `torch.inference_mode()`** — `video_analysis/possession.py` `BallDetector`; new `detect_batch()` (batched with per-frame fallback).
3. **`classify_batch()`** — `team_classifier/team_classifier.py`: embeds crops from K frames in one SigLIP pass (`classify()` left intact for local callers).
4. **Two-pass `process_clip`** — `service/clip_processor.py`: Pass 1 batched GPU (detect/embed/ball), Pass 2 sequential stateful logic (tracker/Kalman/carrier/pass) unchanged. Plus **per-stage timing log**.
5. **`CMC_METHOD` + `CLIP_BATCH_WINDOW`** config flags — `service/config.py`; tracker reads `CMC_METHOD` (default `ecc`) via `PlayerTracker(cmc_method=...)`.

## Profiling result (clip h1 m6, 902 frames, total 209.9s)
Per-frame ms:
| stage | ms | note |
|---|---|---|
| decode | 4.3 | fine |
| **player_det (YOLO)** | **10.0** | fp16 working ✅ |
| **team_clf (SigLIP)** | **90.0** | bottleneck #2 |
| **ball_det (RF-DETR)** | **100.4** | bottleneck #1 |
| tracker (ECC) | 28.1 | |
| ball_kalman / carrier / pass | ~0 | |

**Batching gave no speedup on team_clf / ball_det** → they are NOT GPU-throughput-bound.

## Diagnosis (confirmed)
- **SigLIP is on `cuda:0`** (verified: `self.device = cuda`, weights on `cuda:0`). So the 90ms is **CPU-side preprocessing**, not GPU compute: `sports/common/team.py:73-84` runs `sv.cv2_to_pillow()` + HuggingFace `AutoProcessor` (PIL resize/normalize) on CPU per crop, then a tiny GPU forward. On 4 vCPUs this dominates; batching the GPU forward can't fix per-crop CPU preprocessing.
- **RF-DETR Medium @576** is genuinely heavy (~100ms); the batched `predict(list)` likely fell back to per-frame.

## Next fixes (proposed, both accuracy-safe — NOT yet implemented)
1. **SigLIP fast processor** — load `AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH, use_fast=True)` (torchvision-backed) after fit/load; same normalization, much faster. Targets the 90ms.
2. **Ball detection stride + Kalman coast** — `BALL_DETECT_EVERY=2` config; run RF-DETR every 2nd processed frame, feed `None` to `BallTracker.update` on off-frames (it already coasts via `KALMAN_COAST_FRAMES`). ~halves ball_det. A/B-checkable.

**Expected:** ~232ms/frame → ~122ms/frame ≈ **110s**; push stride to 3 or improve SigLIP further to cross <100s.

## Open datapoint before implementing
Run `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1` during a clip:
- Low GPU% → confirms CPU/preprocessing-bound → fast-processor fix is correct.
- High GPU% → pivot to fp16-on-RF-DETR + real batching instead.

## Optional / deferred
- `CMC_METHOD=ecc→sof` (cheaper CPU camera-motion) — validate events/minute_stats before adopting.
- Per-track `team_id` caching — limited by ReID needing per-frame embeddings.
- Producer-thread pipeline parallelism (overlap decode/GPU with sequential pass).

## Verification rule
Items 1-5 above must keep `minute_stats`/`events` **identical** for a clip (fp16 sub-pixel jitter aside). The ball-stride and CMC changes are the only ones allowed a small delta — A/B compare before adopting.
