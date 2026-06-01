import json

with open("D:/GSFA_highlights/video_analysis/keypoint_log.json") as f:
    data = json.load(f)

total = len(data)
n_direct = sum(1 for d in data if d["H_solved"])
n_used_dist = {}
for d in data:
    n = d["n_used"]
    n_used_dist[n] = n_used_dist.get(n, 0) + 1

print(f"Total frames: {total}")
print(f"H solved directly: {n_direct} ({100*n_direct/total:.1f}%)")
print(f"n_used distribution: {sorted(n_used_dist.items())}")
print()

solved = [d for d in data if d["H_solved"]]
solved.sort(key=lambda x: -x["n_used"])
print("=== Top 10 directly-solved frames by keypoint count ===")
EXCLUDE = {0, 6}
for d in solved[:10]:
    used_kps = [kd["idx"] for kd in d["keypoints"] if kd["conf"] >= 0.50 and kd["idx"] not in EXCLUDE]
    confs    = {kd["idx"]: round(kd["conf"],3) for kd in d["keypoints"] if kd["conf"] >= 0.50}
    print(f"  frame={d['frame']:4d}  ts={d['ts']:.1f}s  n_used={d['n_used']}  used_kps={used_kps}  confs={confs}")

print()
print("=== All directly-solved frames with per-keypoint pixel positions ===")
for d in solved[:15]:
    used_kps = set(kd["idx"] for kd in d["keypoints"] if kd["conf"] >= 0.50 and kd["idx"] not in EXCLUDE)
    print(f"  frame={d['frame']:4d}  ts={d['ts']:.1f}s  n_used={d['n_used']}")
    for kd in sorted(d["keypoints"], key=lambda x: x["idx"]):
        if kd["conf"] < 0.10:
            continue
        marker = "<-- USED IN H" if kd["idx"] in used_kps else ""
        print(f"    kp{kd['idx']:2d}: px=({kd['px']:.0f},{kd['py']:.0f})  conf={kd['conf']:.3f}  {marker}")
    print()
