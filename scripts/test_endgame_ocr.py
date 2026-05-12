"""
test_endgame_ocr.py — Validate EasyOCR detection of HALF TIME / FULL TIME /
PENALTY SHOOTOUT overlays + penalty scores on the three test stills in
data/images/. Normalized 0-1000 [ymin, xmin, ymax, xmax] coords.

Run:
  paddle/Scripts/python.exe scripts/test_endgame_ocr.py
"""
import sys
from pathlib import Path

import cv2

# Normalized 0-1000 ROIs (ymin, xmin, ymax, xmax)
HALFTIME_BANNER  = (775, 197, 834, 803)   # +8px on ymax (=14 normalized) for taller crop
FULLTIME_TEXT    = (789, 220, 808, 268)
PENALTY_OVERLAY  = (16,    8, 190, 290)   # legacy top-left overlay (kept for ref)
PENALTY_TEXT     = (30,   95,  45, 205)   # legacy tight text (too thin)

# New central scoreboard penalty overlay (replaces the timer during penalties)
PENALTY_BOX        = (24, 430, 122, 497)
PENALTY_SCORE_LINE = (42, 445,  75, 482)  # "0 - 1"
PENALTY_LABEL      = (85, 438, 105, 488)  # "PENALTIES"

UPSCALE = 8


def norm_to_px(roi, h, w):
    ymin, xmin, ymax, xmax = roi
    x1 = max(0, int(xmin / 1000 * w))
    y1 = max(0, int(ymin / 1000 * h))
    x2 = min(w, int(xmax / 1000 * w))
    y2 = min(h, int(ymax / 1000 * h))
    return x1, y1, x2, y2


def crop(img, roi, scale=UPSCALE):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = norm_to_px(roi, h, w)
    c = img[y1:y2, x1:x2]
    if c.size == 0:
        return c, (x1, y1, x2, y2)
    return cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), (x1, y1, x2, y2)


def _norm(s: str) -> str:
    return "".join(c for c in s.upper() if c.isalnum())


def fuzzy_ratio(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def run_ocr(reader, img, roi, allowlist=None, save_path=None, target=None):
    """Returns (all_texts, best_text, best_conf, fuzzy_score_vs_target, px)."""
    crop_img, px = crop(img, roi)
    if crop_img.size == 0:
        return [], "<EMPTY>", 0.0, 0.0, px
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), crop_img)
    kwargs = dict(detail=1, paragraph=False)
    if allowlist:
        kwargs["allowlist"] = allowlist
    res = reader.readtext(crop_img, **kwargs)
    if not res:
        return [], "<NO TEXT>", 0.0, 0.0, px
    all_texts = [(r[1], float(r[2])) for r in res]
    if target:
        # Score every detection against the target, also try concatenation of all
        concat = " ".join(t for t, _ in all_texts)
        scores = [(t, c, fuzzy_ratio(t, target)) for t, c in all_texts]
        scores.append((concat, max(c for _, c in all_texts), fuzzy_ratio(concat, target)))
        best = max(scores, key=lambda s: s[2])
        return all_texts, best[0], best[1], best[2], px
    best = max(res, key=lambda r: r[2])
    return all_texts, best[1], float(best[2]), 0.0, px


def main():
    import easyocr
    print("[INFO] Loading EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    img_dir = Path("data/images")
    out_dir = Path("data/diagnostics/endgame")

    cases = [
        ("halftime.png",  "HALF TIME banner",      HALFTIME_BANNER,    None,            "HALF TIME"),
        ("fulltime.png",  "FULL TIME text",        FULLTIME_TEXT,      None,            "FULL TIME"),
        ("penalties.png", "Penalty overlay full",  PENALTY_OVERLAY,    None,            "PENALTY SHOOTOUT"),
        ("penalties.png", "Central penalty box",   PENALTY_BOX,        None,            "PENALTIES"),
        ("penalties.png", "Penalty score '0 - 1'", PENALTY_SCORE_LINE, "0123456789 -",  None),
        ("penalties.png", "PENALTIES label",       PENALTY_LABEL,      None,            "PENALTIES"),
    ]

    print(f"\n{'Image':<15} {'Region':<22} {'Pixels':<26} {'Match':<6} BestText (fuzzy vs target)")
    print("-" * 105)
    for fname, label, roi, allow, target in cases:
        path = img_dir / fname
        if not path.exists():
            print(f"  MISSING: {path}"); continue
        img = cv2.imread(str(path))
        if img is None:
            print(f"  UNREADABLE: {path}"); continue
        out_path = out_dir / f"{path.stem}__{label.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        all_texts, text, conf, fuzzy, px = run_ocr(reader, img, roi, allowlist=allow, save_path=out_path, target=target)
        if target:
            verdict = "OK" if fuzzy >= 0.65 else "MISS"
            print(f"  {fname:<13} {label:<22} ({px[0]:3d},{px[1]:3d})-({px[2]:3d},{px[3]:3d})  {verdict:<6} '{text}'  fuzzy={fuzzy:.2f}  all={all_texts}")
        else:
            print(f"  {fname:<13} {label:<22} ({px[0]:3d},{px[1]:3d})-({px[2]:3d},{px[3]:3d})  conf={conf:.2f}  '{text}'  all={all_texts}")

    print(f"\n[INFO] Crops saved to {out_dir}/")


if __name__ == "__main__":
    main()
