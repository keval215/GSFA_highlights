"""
jersey_recognize.py — offline tracklet jersey-number recognition (Layer 1+2+5).

Goal: get a number for EVERY player, "anyhow". Pure per-frame OCR can't, so:
  * TRACKLET TEMPORAL VOTING — OCR every gated frame across a track's whole
    life and confidence-vote, instead of a few best crops. Dozens of attempts
    per player → one robust number even at a low per-frame hit rate.
  * SUPER-RES-LITE preprocessing on the number patch (Lanczos x4 + CLAHE
    contrast + unsharp) so borderline / low-contrast (RSFC white-on-stripe)
    numbers become legible.
  * Exports a REVIEW SHEET + best crop per track so a human can confirm/fill
    the residual in a few clicks (jersey_review.* ) — that is what guarantees
    100% coverage. See jersey_review + the roster loader.

Outputs (under video_analysis/jersey/):
  jersey_numbers.json   {tid: {number, conf, votes, team, frames, best_crop}}
  crops/t<tid>.png      best (most legible) crop per track, for review
  jersey_review_sheet.png   montage for eyeballing

Run:
  .venv311/Scripts/python.exe video_analysis/jersey_recognize.py
Env: CV_ROOT, STEP (frame stride, default 3), GPU=1
"""
import os
import json
import pickle
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(os.environ.get("CV_ROOT", r"D:/cv project"))
FR = ROOT / "local_pipeline_out" / "frames_2min"
TRACK_PKL = ROOT / "local_pipeline_out" / "track_data_classed_2min.pkl"
OUT = ROOT / "video_analysis" / "jersey"
CROPS = OUT / "crops"
STEP = int(os.environ.get("STEP", "3"))
MIN_H, MIN_SHARP = 100, 55
MIN_CONF = 0.45
ASSIGN_W, ASSIGN_C, MARGIN = 2.0, 2, 1.4
LO, HI = 1, 99

OUT.mkdir(parents=True, exist_ok=True)
CROPS.mkdir(parents=True, exist_ok=True)

import easyocr
reader = easyocr.Reader(["en"], gpu=os.environ.get("GPU") == "1", verbose=False)
_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def is_yellow(c):
    h = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    return (((h[:, :, 0] >= 20) & (h[:, :, 0] <= 45) & (h[:, :, 1] > 100) & (h[:, :, 2] > 120)).mean()) > 0.15


def sharpness(c):
    return float(cv2.Laplacian(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def enhance(roi):
    """super-res-lite: Lanczos x4 -> CLAHE on L -> unsharp."""
    roi = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
    roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(roi, (0, 0), 2.0)
    return cv2.addWeighted(roi, 1.6, blur, -0.6, 0)


def read_number(crop):
    h = crop.shape[0]
    roi = crop[int(h * 0.12):int(h * 0.55), :]
    if roi.size == 0:
        return None
    res = reader.readtext(enhance(roi), allowlist="0123456789", min_size=10, text_threshold=0.5)
    best = None
    for _, txt, conf in res:
        txt = txt.strip()
        if not txt.isdigit() or conf < MIN_CONF:
            continue
        n = int(txt)
        if not (LO <= n <= HI):
            continue
        if best is None or conf > best[1]:
            best = (n, float(conf))
    return best


# ---- load tracks + per-track team (torso V/S/white KMeans) ----
td = pickle.load(open(TRACK_PKL, "rb"))
N = max(td) + 1
N = min(N, int(os.environ.get("MAXF", "0")) or N)
players = {f: {t: bb for t, (bb, c) in fd.items() if c == 0} for f, fd in td.items()}


def torso_feat(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    return np.array([v.mean(), s.mean(), ((v > 175) & (s < 60)).mean()])


# ---- main pass: vote numbers per track, keep best crop ----
votes = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))   # tid -> {num:[w,c]}
frames_seen = Counter()
best_by_conf = {}        # tid -> (conf, crop)
best_by_sharp = {}       # tid -> (sharp, crop)
teamfeat = defaultdict(list)

for f in tqdm(range(0, N, STEP), desc="ocr"):
    fd = players.get(f, {})
    if not fd:
        continue
    fr = cv2.imread(str(FR / f"{f:06d}.jpg"))
    if fr is None:
        continue
    for tid, (x1, y1, x2, y2) in fd.items():
        h = y2 - y1
        crop = fr[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0 or h < MIN_H or (x2 - x1) < 22 or is_yellow(crop):
            continue
        sh = sharpness(crop)
        if sh < MIN_SHARP:
            continue
        frames_seen[tid] += 1
        teamfeat[tid].append(torso_feat(crop[:int(h * .5)]))
        if tid not in best_by_sharp or sh > best_by_sharp[tid][0]:
            best_by_sharp[tid] = (sh, crop.copy())
        got = read_number(crop)
        if got is None:
            continue
        num, conf = got
        v = votes[tid][num]; v[0] += conf; v[1] += 1
        if tid not in best_by_conf or conf > best_by_conf[tid][0]:
            best_by_conf[tid] = (conf, crop.copy())

# ---- team per track via pooled KMeans on torso feature ----
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
tids = [t for t in teamfeat if teamfeat[t]]
team_of = {}
if len(tids) >= 2:
    raw = np.stack([np.mean(teamfeat[t], 0) for t in tids])
    km = KMeans(2, n_init=10, random_state=0).fit(StandardScaler().fit_transform(raw))
    rsfc = 0 if raw[km.labels_ == 0, 2].mean() >= raw[km.labels_ == 1, 2].mean() else 1
    team_of = {t: (0 if l == rsfc else 1) for t, l in zip(tids, km.labels_)}

# ---- decide number per track + export ----
results = {}
for tid in frames_seen:
    ranked = sorted(votes.get(tid, {}).items(), key=lambda kv: -kv[1][0])
    number, conf, locked = None, 0.0, False
    if ranked:
        top_num, (top_w, top_c) = ranked[0]
        number, conf = top_num, round(top_w, 2)
        second = ranked[1][1][0] if len(ranked) > 1 else 0.0
        locked = top_w >= ASSIGN_W and top_c >= ASSIGN_C and (second == 0 or top_w >= MARGIN * second)
    # best crop for review: prefer the one that produced the top read, else sharpest
    crop = best_by_conf.get(tid, (0, None))[1]
    if crop is None:
        crop = best_by_sharp.get(tid, (0, None))[1]
    crop_path = ""
    if crop is not None:
        crop_path = str(CROPS / f"t{tid}.png")
        cv2.imwrite(crop_path, crop)
    results[tid] = {
        "number": number if locked else None,
        "guess": number, "conf": conf, "locked": locked,
        "votes": {str(k): round(v[0], 2) for k, v in votes.get(tid, {}).items()},
        "team": team_of.get(tid), "frames": frames_seen[tid], "best_crop": crop_path,
    }

# keep only tracks that are plausibly real players (seen enough)
real = {t: r for t, r in results.items() if r["frames"] >= 3}
json.dump(real, open(OUT / "jersey_numbers.json", "w"), indent=1)

locked = {t: r for t, r in real.items() if r["number"] is not None}
print(f"\n[jersey] tracks considered: {len(real)}")
print(f"[jersey] AUTO-locked numbers: {len(locked)}  -> "
      + ", ".join(f"t{t}=#{r['number']}({r['conf']})" for t, r in sorted(locked.items())))
need = [t for t in real if real[t]["number"] is None]
print(f"[jersey] need human review: {len(need)} tracks -> {need}")

# ---- review sheet montage (sorted by dwell) ----
order = sorted(real, key=lambda t: -real[t]["frames"])
cols, tw, th = 6, 130, 200
rows = (len(order) + cols - 1) // cols
sheet = np.full((rows * (th + 26) + 40, cols * (tw + 8) + 8, 3), (28, 28, 28), np.uint8)
cv2.putText(sheet, "Jersey review — green=auto-locked, red=needs number",
            (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
for i, t in enumerate(order):
    r = real[t]
    cp = r["best_crop"]
    tile = cv2.imread(cp) if cp and os.path.exists(cp) else None
    tile = cv2.resize(tile, (tw, th)) if tile is not None else np.zeros((th, tw, 3), np.uint8)
    col = (90, 220, 90) if r["number"] is not None else (90, 90, 230)
    cv2.rectangle(tile, (0, 0), (tw - 1, th - 1), col, 2)
    ry, rx = 40 + (i // cols) * (th + 26), 8 + (i % cols) * (tw + 8)
    sheet[ry:ry + th, rx:rx + tw] = tile
    lab = f"t{t} #{r['number']}" if r["number"] is not None else f"t{t} #? ({r['guess']})"
    cv2.putText(sheet, lab, (rx, ry + th + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
cv2.imwrite(str(OUT / "jersey_review_sheet.png"), sheet)
print(f"[jersey] wrote {OUT/'jersey_numbers.json'} + review sheet + {len(order)} crops")
