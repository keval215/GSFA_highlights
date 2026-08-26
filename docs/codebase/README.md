# GSFA Highlights — Codebase Documentation

This folder documents the entire `GSFA_highlights` codebase: what every package and
file does, what it deliberately does **not** do, and how the pieces connect.

> Scope note: the system analyses **futsal / football match video** to produce
> **ball-possession and pass statistics** (and some experimental highlight/heatmap
> tooling). It does not edit or cut highlight reels yet — the `events` table is the
> raw material a future highlight-reel builder would consume.

---

## How to read this documentation

Start here, then follow the reading order:

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** — the big picture. The two ways the code
   runs (local script vs. cloud service), the full per-frame data flow, and a
   module-to-module connection map. **Read this first.**
2. **[GLOSSARY.md](GLOSSARY.md)** — domain terms (carrier, foot zone, OOF, travel,
   coast, possession denominator, etc.). Keep it open while reading the rest.
3. The per-package folders below, in dependency order (low-level → high-level).

---

## Repository map

```
GSFA_highlights/
├── modules/               The shared CV pipeline package
│   ├── detectors/         Object detection + per-frame parsing (YOLOv11m), GK logic, cache
│   ├── team_classifier/   Assign each player to team 0/1 (SigLIP and colour-histogram variants)
│   ├── tracking/          BoT-SORT multi-object tracker (stable per-player track IDs)
│   └── possession/        Ball tracking, carrier engine, pass FSM, possession stats
├── rulesets/              Per-sport RulesetConfig profiles (futsal / classic) selecting all of the above
├── video_analysis/        run.py (local dev CLI) + homography/keypoint tooling
├── service/               The production cloud service (FastAPI ingest + GPU worker + Azure + SQL)
├── sql/                   Azure SQL schema (matches, minute_stats, post_processing, events, callback_outbox)
├── scripts/               Standalone tools (CSV export, clip upload, team-diagnostics, tracker research)
├── data/                  (gitignored) cache pkls + output videos/PNGs
├── docs/                  Documentation (this tree + API.md + azure_deploy.md)
├── Dockerfile             Single image for both api + worker
├── docker-compose.yml     Runs api (CPU) + worker (GPU) from that image
├── requirements.txt       Local dev deps   |  requirements-service.txt  Service deps
└── .github/workflows/     CI: push to `main` → SSH redeploy on the Azure VM
```

`heatmap.py` and `shots_on_t.py` (former root-level standalone tools) have been deleted
from the repo — see [scripts/README.md](scripts/README.md).

---

## Package documentation index

| Package | Doc | What lives there |
|---|---|---|
| `modules/detectors/` | [detectors/README.md](detectors/README.md) | YOLOv11m detection, `Detection`/`FrameDetections` types, goalkeeper detection, cache paths |
| `modules/team_classifier/` | [team_classifier/README.md](team_classifier/README.md) | SigLIP team classifier (production) + colour-histogram variant |
| `modules/tracking/` | [tracking/README.md](tracking/README.md) | BoT-SORT wrapper that consumes external SigLIP embeddings |
| `modules/possession/` + `video_analysis/` | [video_analysis/README.md](video_analysis/README.md) | Ball tracking → carrier → pass FSM → possession stats (`modules/possession/`), + `run.py` (local CLI) and homography tools |
| `rulesets/` | [rulesets/README.md](rulesets/README.md) | Per-sport `RulesetConfig` profiles (futsal production default, classic placeholder) |
| `service/` | [service/README.md](service/README.md) | The production service: ingest API, GPU worker, sessions, stats, DB, Azure adapters |
| `sql/` | [sql/README.md](sql/README.md) | Database schema + why cumulative numbers are computed on read |
| scripts / standalone | [scripts/README.md](scripts/README.md) | `scripts/*` (CSV export, clip upload, tracker research) |
| infra | [infra/README.md](infra/README.md) | Dockerfile, compose, CI/CD, requirements, environment |

---

## The two execution modes (at a glance)

The same CV pipeline classes are used in two very different entry points:

| | **Local script mode** | **Production service mode** |
|---|---|---|
| Entry point | `python video_analysis/run.py [--ruleset futsal\|classic]` | `service/api.py` (ingest) + `service/worker.py` (GPU) |
| Input | One whole video file on disk | 60-second clips uploaded over HTTP, one per match-minute |
| Output | An annotated `.mp4` + printed summary | Rows in Azure SQL + an HTTP callback with cumulative stats |
| Team fit | `fit_from_video_or_load()` over the video | `Session.ensure_fit()` on clip 1, silhouette quality guard |
| State | Lives for one process run | Persists across clips in a `MatchSession` (+ pkl on disk) |
| Rendering | Yes (draws overlays) | **No** — the service never renders video |

Both paths share: `PlayerDetector`, `GSFATeamClassifier`, `PlayerTracker`,
`BallTracker`, `CarrierEngine`, `PassEventTracker` (`modules/`) — each constructed from a
**`RulesetConfig`** (`rulesets/`) selected per run/match. The possession **math** is
written once and reused — see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Conventions used in these docs

- **"Does NOT"** sections are first-class — they record deliberate scope boundaries so
  you don't go looking for behaviour that was intentionally left out.
- **"Connections"** sections name the exact modules/functions on each side of a link.
- Code references are written as `file.py:symbol` so they're easy to grep.
- These docs were generated from a read-through of the source (CI now deploys off
  `main`, not `test`). If code and docs disagree, the code wins — please update the
  relevant file here.
