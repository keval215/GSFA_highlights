---
name: team-classifier-architecture
description: Which team classifier is active in the main pipeline and why the architecture evolved
type: project
---

The main pipeline (video_analysis/possession.py, branch `test`) uses **GSFATeamClassifier** — SigLIP (768-D) + UMAP (→3-D) + KMeans(k=2) — NOT ColourHistogramTeamClassifier.

The key fix vs the naive SigLIP approach was switching the crop to the **top 55% of the player bbox** (TORSO_RATIO=0.55) and adding a **blur filter** (Laplacian variance > 80). These two changes remove background contamination and blurry crops that were causing spatial clustering instead of jersey-colour clustering on the panning futsal camera.

The ColourHistogramTeamClassifier (HSV h=64bins + s=32bins, KMeans directly) exists in team_classifier/colour_histogram.py as a drop-in alternative but is not wired into possession.py. An earlier project memory described it as the "fix" — that was an intermediate state. The current production path is GSFATeamClassifier.

**Why:** SigLIP embeddings are richer than HSV histograms and enable the PlayerTracker (BoT-SORT) to also consume the 768-D feature for ReID. The torso crop + blur gate is sufficient to remove the background contamination that originally broke SigLIP on this panning camera.

**How to apply:** Always wire GSFATeamClassifier in possession.py and any future pipeline scripts. Never swap in ColourHistogramTeamClassifier without flagging it to the user. The pkl cache is at data/cache/<video_stem>_team_siglip.pkl.

See also: [[user-team-classifier-rule]]
