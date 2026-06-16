"""
test.py — upload a 1-minute clip to the ingestion API.

The upload returns in ~1-2 s; the worker processes the clip asynchronously and
pushes the advance-stats callback once when it finishes. To see the real callback
JSON, watch the worker logs for the `advance-stats ... body={...}` line.
"""

import requests

API_BASE  = "https://aiff.clubduelz.in"
CLIP_PATH = r"C:\Users\Admin\Downloads\Video Project_1min.mp4"

MATCH_ID = "test_match_001"
HALF     = 1
MINUTE   = 13

with open(CLIP_PATH, "rb") as f:
    response = requests.post(
        f"{API_BASE}/api/clips",
        files={"file": ("clip.mp4", f, "video/mp4")},
        data={
            "match_id":     MATCH_ID,
            "half":         str(HALF),
            "minute":       str(MINUTE),
            "team0_name":   "RFC",
            "team0_colour": "#0000FF",
            "team1_name":   "BFA",
            "team1_colour": "#800080",
        },
    )

print(f"Upload status:   {response.status_code}")
print(f"Upload response: {response.json()}")
