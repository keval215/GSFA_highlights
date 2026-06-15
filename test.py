import requests

url = "https://aiff.clubduelz.in/api/clips"

with open(r"C:\Users\Admin\Downloads\Video Project_1min.mp4", "rb") as f:
    response = requests.post(
        url,
        files={"file": ("clip.mp4", f, "video/mp4")},
        data={
            "match_id":     "test_match_001",
            "half":         "1",
            "minute":       "1",
            "team0_name":   "RFC",
            "team0_colour": "#0000FF",
            "team1_name":   "BFA",
            "team1_colour": "#800080"
        }
    )

print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")