"""
number_ocr.py — jersey-number OCR anchor for player tracking.

Appearance re-ID (OSNet/SigLIP) cannot separate same-team players (identical
kits). The jersey NUMBER is the only per-player-unique signal. It is unreadable
most frames (small / front-facing / blurred) but, when a player's back faces the
camera at size, easyocr reads it reliably (conf ~0.9-1.0, repeatably).

So this is a SPARSE, per-track ANCHOR — not a per-frame classifier:
  * OCR a track only on gated crops (big enough, sharp, not a ref/sub bib),
    throttled to every Nth frame, and stop once the number is locked.
  * Accumulate confidence-weighted votes per track; lock a number once the
    evidence is strong and consistent.
  * Use locked numbers to (a) label true identity and (b) reconnect fragments:
    two tracks with the same (team, number) are the same player.

Plugs into the GSFA `tracking/player_tracker.py` flow — it duck-types the
`Detection` object (needs `.track_id`, `.bbox=(x1,y1,x2,y2)`, `.team_id`).

Wire-in (in possession.py, after tracker.update + team/gk classify):
    ocr = JerseyOCRAnchor(gpu=False)
    ...per frame...
    ocr.update(frame, dets.players, frame_idx)
    num = ocr.number_for(p.track_id)          # int | None
    canon = ocr.reconnect_map()               # {track_id: canonical_track_id}

Demo (validate on cached futsal tracks):
    .venv311/Scripts/python.exe video_analysis/number_ocr.py
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class JerseyOCRAnchor:
    def __init__(
        self,
        reader=None,
        gpu: bool = False,
        min_height: int = 110,          # px; numbers below this are illegible
        min_sharpness: float = 60.0,    # Laplacian variance; rejects motion blur
        ocr_every: int = 3,             # throttle: OCR a track at most 1/ocr_every frames
        min_conf: float = 0.50,         # per-read confidence floor
        upscale: int = 4,               # cubic upscale of the back region before OCR
        assign_weight: float = 2.0,     # min summed confidence to lock a number
        assign_count: int = 2,          # min distinct frames a number must appear in
        assign_margin: float = 1.5,     # winner must beat runner-up weight by this factor
        max_reads_per_track: int = 60,  # stop OCRing a track after this many reads
        number_range: Tuple[int, int] = (1, 99),
    ):
        self._reader = reader
        self._gpu = gpu
        self.min_height = min_height
        self.min_sharpness = min_sharpness
        self.ocr_every = ocr_every
        self.min_conf = min_conf
        self.upscale = upscale
        self.assign_weight = assign_weight
        self.assign_count = assign_count
        self.assign_margin = assign_margin
        self.max_reads = max_reads_per_track
        self.lo, self.hi = number_range

        # tid -> {number: [weight_sum, distinct_frame_count]}
        self._votes: Dict[int, Dict[int, list]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
        self._reads: Dict[int, int] = defaultdict(int)
        self._seen: Dict[int, int] = defaultdict(int)     # gated crops seen per track
        self._best: Dict[int, tuple] = {}                 # tid -> (score, crop) for review
        self._last_ocr: Dict[int, int] = {}
        self._assigned: Dict[int, int] = {}       # tid -> locked number
        self._team: Dict[int, int] = {}           # tid -> team_id (last seen)
        self._roster: Dict[int, int] = {}         # human-confirmed tid -> number (wins)

    def set_roster(self, roster: Dict[int, int]) -> None:
        """Human-confirmed {track_id: number} (from jersey_review). Takes
        precedence over live OCR in number_for()."""
        self._roster = {int(k): int(v) for k, v in roster.items() if v is not None}

    # -- lazy easyocr reader (so import is cheap) --
    @property
    def reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(["en"], gpu=self._gpu, verbose=False)
        return self._reader

    # -- gates --
    @staticmethod
    def _is_yellow(crop: np.ndarray) -> bool:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        return (((h >= 20) & (h <= 45) & (s > 100) & (v > 120)).mean()) > 0.15

    @staticmethod
    def _sharpness(crop: np.ndarray) -> float:
        return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

    def _gate(self, crop: np.ndarray) -> bool:
        if crop.size == 0 or crop.shape[0] < self.min_height or crop.shape[1] < 25:
            return False
        if self._is_yellow(crop):            # referees / sideline bibs
            return False
        if self._sharpness(crop) < self.min_sharpness:
            return False
        return True

    def _read_number(self, crop: np.ndarray) -> Optional[Tuple[int, float]]:
        h = crop.shape[0]
        back = crop[int(h * 0.15):int(h * 0.55), :]          # upper back = number zone
        if back.size == 0:
            return None
        back = cv2.resize(back, None, fx=self.upscale, fy=self.upscale,
                          interpolation=cv2.INTER_CUBIC)
        res = self.reader.readtext(back, allowlist="0123456789",
                                   min_size=10, text_threshold=0.5)
        best = None
        for _, txt, conf in res:
            txt = txt.strip()
            if not txt.isdigit() or conf < self.min_conf:
                continue
            n = int(txt)
            if not (self.lo <= n <= self.hi):                # plausible jersey number
                continue
            if best is None or conf > best[1]:
                best = (n, float(conf))
        return best

    def update(self, frame: np.ndarray, players: List, frame_idx: int) -> None:
        """OCR gated player crops, accumulate per-track number votes."""
        for p in players:
            tid = getattr(p, "track_id", None)
            if tid is None:
                continue
            tid = int(tid)
            if getattr(p, "team_id", None) is not None:
                self._team[tid] = int(p.team_id)
            if tid in self._assigned or self._reads[tid] >= self.max_reads:
                continue
            if frame_idx - self._last_ocr.get(tid, -10 ** 9) < self.ocr_every:
                continue
            x1, y1, x2, y2 = (int(v) for v in p.bbox)
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if not self._gate(crop):
                continue
            self._last_ocr[tid] = frame_idx
            self._seen[tid] += 1
            self._consider_crop(tid, crop, self._sharpness(crop) * 1e-5)
            got = self._read_number(crop)
            if got is None:
                continue
            num, conf = got
            self._reads[tid] += 1
            self._consider_crop(tid, crop, conf)         # conf >> sharpness → wins
            v = self._votes[tid][num]
            v[0] += conf
            v[1] += 1
            self._maybe_lock(tid)

    def _consider_crop(self, tid: int, crop: np.ndarray, score: float) -> None:
        prev = self._best.get(tid)
        if prev is None or score > prev[0]:
            self._best[tid] = (score, crop.copy())

    def _maybe_lock(self, tid: int) -> None:
        ranked = sorted(self._votes[tid].items(), key=lambda kv: -kv[1][0])
        if not ranked:
            return
        top_num, (top_w, top_c) = ranked[0]
        if top_w < self.assign_weight or top_c < self.assign_count:
            return
        if len(ranked) > 1:
            second_w = ranked[1][1][0]
            if second_w > 0 and top_w < self.assign_margin * second_w:
                return
        self._assigned[tid] = top_num

    # -- queries --
    def number_for(self, tid: Optional[int], provisional: bool = True) -> Optional[int]:
        """Locked number, or (if provisional) the current leading candidate."""
        if tid is None:
            return None
        tid = int(tid)
        if tid in self._roster:          # human-confirmed wins
            return self._roster[tid]
        if tid in self._assigned:
            return self._assigned[tid]
        if provisional and self._votes.get(tid):
            return max(self._votes[tid].items(), key=lambda kv: kv[1][0])[0]
        return None

    def assignments(self) -> Dict[int, int]:
        """Locked {track_id: jersey_number} only."""
        return dict(self._assigned)

    def export(self, json_path, crops_dir=None) -> dict:
        """Dump per-track review material (same schema as jersey_recognize.py):
        {tid: {number, guess, conf, votes, team, frames, best_crop}} + best crop
        images. Feeds jersey_review.py -> jersey_roster.json -> set_roster()."""
        import json
        from pathlib import Path
        jp = Path(json_path)
        cd = Path(crops_dir) if crops_dir else jp.parent / "crops"
        cd.mkdir(parents=True, exist_ok=True)
        out = {}
        for tid in self._seen:
            ranked = sorted(self._votes.get(tid, {}).items(), key=lambda kv: -kv[1][0])
            guess = ranked[0][0] if ranked else None
            conf = round(ranked[0][1][0], 2) if ranked else 0.0
            crop_path = ""
            if tid in self._best:
                crop_path = str(cd / f"t{tid}.png")
                cv2.imwrite(crop_path, self._best[tid][1])
            out[str(tid)] = {
                "number": self._assigned.get(tid),
                "guess": guess, "conf": conf,
                "votes": {str(k): round(v[0], 2) for k, v in self._votes.get(tid, {}).items()},
                "team": self._team.get(tid), "frames": self._seen[tid],
                "best_crop": crop_path,
            }
        json.dump(out, open(jp, "w"), indent=1)
        return out

    def reconnect_map(self) -> Dict[int, int]:
        """Fragments sharing (team, number) are the same player → map every
        track_id to the smallest track_id in its (team, number) group."""
        groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        effective = {**self._assigned, **self._roster}   # human roster wins
        for tid, num in effective.items():
            groups[(self._team.get(tid, -1), num)].append(tid)
        out: Dict[int, int] = {}
        for tids in groups.values():
            canon = min(tids)
            for t in tids:
                out[t] = canon
        return out


# ---------------------------------------------------------------------------
# Demo / validation on the cached futsal tracks
# ---------------------------------------------------------------------------
def _demo():
    import pickle
    from pathlib import Path
    ROOT = Path(os.environ.get("CV_ROOT", r"D:/cv project"))
    FR = ROOT / "local_pipeline_out" / "frames_2min"
    td = pickle.load(open(ROOT / "local_pipeline_out" / "track_data_classed_2min.pkl", "rb"))
    N = max(td) + 1
    STEP = int(os.environ.get("DEMO_STEP", "3"))
    MAXF = int(os.environ.get("DEMO_MAXF", "0")) or N

    class _Det:
        __slots__ = ("track_id", "bbox", "team_id")
        def __init__(s, t, b): s.track_id, s.bbox, s.team_id = t, b, None

    ocr = JerseyOCRAnchor(gpu=os.environ.get("GPU") == "1")
    n_frames = 0
    for f in range(0, MAXF, STEP):
        fd = td.get(f, {})
        players = [_Det(tid, bb) for tid, (bb, c) in fd.items() if c == 0]
        if not players:
            continue
        fr = cv2.imread(str(FR / f"{f:06d}.jpg"))
        if fr is None:
            continue
        ocr.update(fr, players, f)
        n_frames += 1
        if f % 600 == 0:
            print(f"  frame {f}: {len(ocr.assignments())} numbers locked so far")
    asg = ocr.assignments()
    print(f"\n[demo] processed {n_frames} frames")
    print(f"[demo] locked jersey numbers for {len(asg)} tracks:")
    for tid, num in sorted(asg.items()):
        print(f"    track {tid:4d} -> #{num}")
    rc = {k: v for k, v in ocr.reconnect_map().items() if k != v}
    print(f"[demo] reconnect (same team+number): {rc}")


if __name__ == "__main__":
    _demo()
