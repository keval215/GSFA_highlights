"""
jersey_review.py — human confirm/fill step that GUARANTEES a number per track.

Reads jersey_recognize.py output (jersey_numbers.json + crops), lets you confirm
the auto-locked numbers and fill the residual, and writes a locked roster
  jersey_roster.json   { "<track_id>": <number|null> }
which the pipeline uses as the source of truth (overrides live OCR).

A futsal roster is ~12-16 players, and most are pre-filled by OCR, so this is a
few keystrokes per match.

Modes:
  # 1) Template (works anywhere): write an editable JSON pre-filled with guesses,
  #    you edit the "number" fields, then import it.
  python video_analysis/jersey_review.py --template
  #    -> edit video_analysis/jersey/jersey_roster_template.json, then:
  python video_analysis/jersey_review.py --from-template

  # 2) Interactive CLI: shows each track (open the crop path), type the number.
  python video_analysis/jersey_review.py

For Colab use the ipywidgets cell (image + textbox per track) — see README.
"""
import os
import json
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CV_ROOT", r"D:/cv project"))
OUT = Path(os.environ.get("JERSEY_DIR", str(ROOT / "video_analysis" / "jersey")))
NUMS = OUT / "jersey_numbers.json"
ROSTER = OUT / "jersey_roster.json"
TEMPLATE = OUT / "jersey_roster_template.json"


def _load():
    data = json.load(open(NUMS))
    # order by dwell (most active first); keys are strings in json
    return dict(sorted(data.items(), key=lambda kv: -kv[1]["frames"]))


def make_template():
    data = _load()
    tmpl = {tid: {"number": r["number"], "ocr_guess": r["guess"], "conf": r["conf"],
                  "team": r["team"], "frames": r["frames"], "crop": r["best_crop"]}
            for tid, r in data.items()}
    json.dump(tmpl, open(TEMPLATE, "w"), indent=1)
    n_need = sum(1 for r in data.values() if r["number"] is None)
    print(f"[review] wrote {TEMPLATE}")
    print(f"[review] {len(tmpl)} tracks; {n_need} have number=null — open each 'crop' "
          f"image, set its \"number\", then run: --from-template")


def from_template():
    tmpl = json.load(open(TEMPLATE))
    roster = {tid: (int(v["number"]) if v.get("number") is not None else None)
              for tid, v in tmpl.items()}
    json.dump(roster, open(ROSTER, "w"), indent=1)
    filled = {k: v for k, v in roster.items() if v is not None}
    print(f"[review] wrote {ROSTER}: {len(filled)}/{len(roster)} tracks numbered")


def interactive():
    data = _load()
    roster = {}
    print("Enter=accept shown number | type N=set number | s=skip(no number) | q=quit\n")
    for tid, r in data.items():
        cur = r["number"] if r["number"] is not None else r["guess"]
        prompt = (f"track {tid}  team={r['team']}  frames={r['frames']}  "
                  f"crop={r['best_crop']}\n  number [{cur}]: ")
        try:
            ans = input(prompt).strip()
        except EOFError:
            print("(no stdin — use --template mode)"); return
        if ans.lower() == "q":
            break
        if ans.lower() == "s":
            roster[tid] = None; continue
        if ans == "" and cur is not None:
            roster[tid] = int(cur)
        elif ans.isdigit():
            roster[tid] = int(ans)
        else:
            roster[tid] = None
    json.dump(roster, open(ROSTER, "w"), indent=1)
    filled = {k: v for k, v in roster.items() if v is not None}
    print(f"\n[review] wrote {ROSTER}: {len(filled)}/{len(roster)} tracks numbered")


def load_roster(path=ROSTER):
    """{track_id:int -> number:int} for confirmed entries only."""
    if not Path(path).exists():
        return {}
    raw = json.load(open(path))
    return {int(k): int(v) for k, v in raw.items() if v is not None}


if __name__ == "__main__":
    if "--template" in sys.argv:
        make_template()
    elif "--from-template" in sys.argv:
        from_template()
    else:
        interactive()
