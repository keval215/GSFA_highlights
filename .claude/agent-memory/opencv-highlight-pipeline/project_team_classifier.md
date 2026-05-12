---
name: Team Classifier Architecture Decision
description: Why the GSFA pipeline uses HSV torso histograms instead of SigLIP+UMAP for team classification
type: project
---

The original team classifier (02_team_classification.py) used SigLIP + UMAP + KMeans on full player bounding box crops. This caused spatial clustering (on-court vs bench) instead of jersey-color clustering because SigLIP encoded background context (bleachers, court surface) rather than jersey features.

**Why:** Camera is PANNING (not fixed), so court-polygon spatial filters cannot correct for it. Full-box SigLIP embeddings are background-contaminated on a panning camera.

**How to apply:** The fix is torso-only HSV histogram features (TORSO_TOP_FRAC=0.25, TORSO_BOT_FRAC=0.65, TORSO_LR_FRAC=0.15) fed directly into KMeans (k=2). No UMAP, no SigLIP, no GPU required. TeamClassifier class keeps same fit()/predict() interface for drop-in compatibility. The roboflow sports/common/team.py reference implementation uses the same broken SigLIP approach — we intentionally diverge from it on this project.
