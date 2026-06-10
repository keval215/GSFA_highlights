"""
find_jersey.py — self-contained jersey-number finder.

Video in -> per-track jersey numbers out. Does detection + BoT-SORT tracking +
tracklet OCR voting (super-res-lite) in one script. No other project files
needed; the GSFA player model sits next to this script in the zip.

Run (Colab/local):
    GSFA_VIDEO=/path/to/match.mp4 python find_jersey.py
Env:
    GSFA_MODEL  player detector  (default: GSFA_PLAYER_DETECTION.pt next to this file)
    GSFA_VIDEO  match video                              (required)
    GSFA_OUT    output dir       (default: ./jersey_out)
    STEP        OCR every Nth frame (default 2)
    GPU         1 = easyocr on GPU (default 1)
"""
import os
import json
import glob
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = os.environ.get("GSFA_MODEL", str(HERE / "GSFA_PLAYER_DETECTION.pt"))
VIDEO = os.environ.get("GSFA_VIDEO", "")
OUT = Path(os.environ.get("GSFA_OUT", "jersey_out"))
STEP = int(os.environ.get("STEP", "2"))
GPU = os.environ.get("GPU", "1") == "1"

if not VIDEO:
    cand = sorted(glob.glob(str(HERE / "*.mp4")))
    VIDEO = cand[0] if cand else ""
assert os.path.exists(MODEL), f"model not found: {MODEL}"
assert VIDEO and os.path.exists(VIDEO), f"video not found (set GSFA_VIDEO): {VIDEO}"
(OUT / "crops").mkdir(parents=True, exist_ok=True)
print(f"model: {MODEL}\nvideo: {VIDEO}\nout:   {OUT}")

MIN_H, MIN_SHARP, MIN_CONF = 100, 55, 0.45
ASSIGN_W, ASSIGN_C, MARGIN, LO, HI = 2.0, 2, 1.4, 1, 99

import easyocr
from ultralytics import YOLO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

reader = easyocr.Reader(["en"], gpu=GPU, verbose=False)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def is_yellow(c):
    h = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    return (((h[:, :, 0] >= 20) & (h[:, :, 0] <= 45) & (h[:, :, 1] > 100) & (h[:, :, 2] > 120)).mean()) > 0.15


def sharp(c):
    return float(cv2.Laplacian(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def enhance(roi):
    roi = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.addWeighted(roi, 1.6, cv2.GaussianBlur(roi, (0, 0), 2.0), -0.6, 0)


def read_number(crop):
    h = crop.shape[0]
    roi = crop[int(h * 0.12):int(h * 0.55), :]
    if roi.size == 0:
        return None
    best = None
    for _, t, c in reader.readtext(enhance(roi), allowlist="0123456789", min_size=10, text_threshold=0.5):
        t = t.strip()
        if t.isdigit() and c >= MIN_CONF and LO <= int(t) <= HI and (best is None or c > best[1]):
            best = (int(t), float(c))
    return best


# ---- Pass 1: detect + BoT-SORT track ----
print("tracking ...")
det = YOLO(MODEL)
tracks = {}
for i, r in enumerate(det.track(source=VIDEO, tracker="botsort.yaml", persist=True,
                                stream=True, verbose=False, conf=0.2, iou=0.5, classes=[0])):
    tracks[i] = {}
    if r.boxes is None or r.boxes.id is None:
        continue
    for tid, bb in zip(r.boxes.id.cpu().numpy().astype(int), r.boxes.xyxy.cpu().numpy().astype(int)):
        tracks[i][int(tid)] = tuple(bb.tolist())
print(f"{len(tracks)} frames tracked")

# ---- Pass 2: tracklet OCR voting ----
print("reading numbers ...")
votes = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
seen = Counter()
best_crop = {}
teamfeat = defaultdict(list)
cap = cv2.VideoCapture(VIDEO)
f = -1
while True:
    ok, fr = cap.read()
    f += 1
    if not ok:
        break
    if f % STEP or f not in tracks:
        continue
    for tid, (x1, y1, x2, y2) in tracks[f].items():
        h = y2 - y1
        crop = fr[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0 or h < MIN_H or (x2 - x1) < 22 or is_yellow(crop) or sharp(crop) < MIN_SHARP:
            continue
        seen[tid] += 1
        hsv = cv2.cvtColor(crop[:h // 2], cv2.COLOR_BGR2HSV)
        teamfeat[tid].append([hsv[:, :, 2].mean(), hsv[:, :, 1].mean(),
                              ((hsv[:, :, 2] > 175) & (hsv[:, :, 1] < 60)).mean()])
        got = read_number(crop)
        score = got[1] if got else sharp(crop) * 1e-5
        if tid not in best_crop or score > best_crop[tid][0]:
            best_crop[tid] = (score, crop.copy())
        if got:
            v = votes[tid][got[0]]
            v[0] += got[1]
            v[1] += 1
cap.release()

# ---- team per track (for review colour) ----
tids = [t for t in teamfeat if teamfeat[t]]
team = {}
if len(tids) >= 2:
    raw = np.stack([np.mean(teamfeat[t], 0) for t in tids])
    km = KMeans(2, n_init=10, random_state=0).fit(StandardScaler().fit_transform(raw))
    a = 0 if raw[km.labels_ == 0, 2].mean() >= raw[km.labels_ == 1, 2].mean() else 1
    team = {t: (0 if l == a else 1) for t, l in zip(tids, km.labels_)}

# ---- decide + export ----
res = {}
for t in seen:
    rk = sorted(votes.get(t, {}).items(), key=lambda kv: -kv[1][0])
    num = conf = None
    locked = False
    if rk:
        num, (w, c) = rk[0][0], rk[0][1]
        conf = round(w, 2)
        sec = rk[1][1][0] if len(rk) > 1 else 0
        locked = w >= ASSIGN_W and c >= ASSIGN_C and (sec == 0 or w >= MARGIN * sec)
    cp = ""
    if t in best_crop:
        cp = str(OUT / "crops" / f"t{t}.png")
        cv2.imwrite(cp, best_crop[t][1])
    res[str(t)] = {"number": num if locked else None, "guess": num, "conf": conf or 0.0,
                   "team": team.get(t), "frames": seen[t], "best_crop": cp}
res = {t: r for t, r in res.items() if r["frames"] >= 3}
json.dump(res, open(OUT / "jersey_numbers.json", "w"), indent=1)
locked = {t: r["number"] for t, r in res.items() if r["number"] is not None}
print(f"\nAUTO-locked {len(locked)} numbers: " + ", ".join(f"t{t}=#{n}" for t, n in locked.items()))
print(f"need review: {sum(1 for r in res.values() if r['number'] is None)} tracks")

# ---- review sheet ----
order = sorted(res, key=lambda t: -res[t]["frames"])
cols, tw, th = 6, 130, 200
rows = (len(order) + cols - 1) // cols
sheet = np.full((rows * (th + 26) + 40, cols * (tw + 8) + 8, 3), (28, 28, 28), np.uint8)
cv2.putText(sheet, "Jersey review (green=auto, red=fill)", (8, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
for i, t in enumerate(order):
    r = res[t]
    tile = cv2.imread(r["best_crop"]) if r["best_crop"] else None
    tile = cv2.resize(tile, (tw, th)) if tile is not None else np.zeros((th, tw, 3), np.uint8)
    col = (90, 220, 90) if r["number"] is not None else (90, 90, 230)
    cv2.rectangle(tile, (0, 0), (tw - 1, th - 1), col, 2)
    ry, rx = 40 + (i // cols) * (th + 26), 8 + (i % cols) * (tw + 8)
    sheet[ry:ry + th, rx:rx + tw] = tile
    cv2.putText(sheet, f"t{t} #{r['number'] if r['number'] is not None else '?'}",
                (rx, ry + th + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
cv2.imwrite(str(OUT / "jersey_review_sheet.png"), sheet)
print(f"\nsaved -> {OUT}/  (jersey_numbers.json, jersey_review_sheet.png, crops/)")
