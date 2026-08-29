# GSFA_highlights

Futsal/football match-video analysis system. From video it derives, per team (team_a/team_b):
**possession %**, **completed passes**, **interceptions**, and **ball-lost events**, written to
an append-only event log. It does **not** yet edit/cut highlight reels — the `events` table is
raw material for a future highlight-reel builder.

## Two execution modes (same CV pipeline underneath)

| | Local script | Production service |
|---|---|---|
| Entry | `python video_analysis/possession.py` | `service/api.py` (ingest) + `service/worker.py` (GPU) |
| Input | one whole video file on disk | 60s clips over HTTP, one per match-minute (+ a whole-match `post_processing` upload path) |
| Output | annotated `.mp4` + printed summary | Azure SQL rows + HTTP callback with cumulative stats |
| Renders video? | yes | **no** |

Both share `PlayerDetector`, `GSFATeamClassifier`, `PlayerTracker`, `BallTracker`, `CarrierEngine`,
`PassEventTracker` — the possession math is written once. Perception chain per frame:
`PlayerDetector (unified YOLOv11m: active_player/ball/goal_post/referee)` → `GSFATeamClassifier
(SigLIP)` → `GoalkeeperDetector` → `PlayerTracker (BoT-SORT)` → `BallTracker (Kalman)` →
`CarrierEngine (foot-zone)` → `PassEventTracker (release/travel/receive FSM)`.

Deep-dive entry point: **[docs/codebase/ARCHITECTURE.md](docs/codebase/ARCHITECTURE.md)**.
Full doc map: [docs/codebase/README.md](docs/codebase/README.md).

| Package | Doc |
|---|---|
| `detectors/` | [docs/codebase/detectors/README.md](docs/codebase/detectors/README.md) |
| `team_classifier/` | [docs/codebase/team_classifier/README.md](docs/codebase/team_classifier/README.md) |
| `tracking/` | [docs/codebase/tracking/README.md](docs/codebase/tracking/README.md) |
| `video_analysis/` | [docs/codebase/video_analysis/README.md](docs/codebase/video_analysis/README.md) |
| `service/` | [docs/codebase/service/README.md](docs/codebase/service/README.md) |
| `sql/` | [docs/codebase/sql/README.md](docs/codebase/sql/README.md) |
| scripts / standalone | [docs/codebase/scripts/README.md](docs/codebase/scripts/README.md) |
| infra (Docker, CI/CD) | [docs/codebase/infra/README.md](docs/codebase/infra/README.md) |
| Service API | [docs/API.md](docs/API.md) |
| Azure deployment | [docs/azure_deploy.md](docs/azure_deploy.md) |

## Facts worth knowing before touching anything

- **Team classifier: always `GSFATeamClassifier` (SigLIP embeddings). Never `ColourHistogramTeamClassifier`** — explicit, standing production decision, not a stale option to reconsider.
- **Docker must use deadsnakes Python 3.11.** Ubuntu's default `3.11.0rc1` segfaults `torch.jit.script` inside rfdetr-related code paths.
- **The unified YOLOv11m model replaced the old two-model setup** (separate YOLOv11 + RF-DETR). One forward pass now emits all four classes; the RF-DETR ball model and weights are gone. `heatmap.py` and `shots_on_t.py` are standalone/experimental and still use the older separate detectors — don't "fix" them to match the unified model unless asked.
- **Never start `video_analysis/possession.py`, the service worker, or any clip-processing run without explicit user permission.** Doc/code reading and editing never needs this.
- **Never push code, or deploy/copy anything to the production VM (no `git push`, no `scp`/`rsync` to `gsfa-highlights`, no remote `docker compose up`).** The user pushes and deploys themselves. Editing local files and preparing a commit message is fine; executing the push/deploy is not.
- **If code and docs disagree, the code wins** — but disagreements shouldn't accumulate; see below.

## Memory & docs workflow

- **Update docs alongside the code change, not deferred.** When a change alters documented behavior (API contract, config, architecture, data flow), update the relevant `docs/` file(s) in the same change. This project has been burned by drift before — `docs/API.md` claimed `half`/`minute`/GK colours were required well after the code made them optional — don't let that recur.
- **`/update-doc`** is the periodic/explicit full-repo reconciliation sweep — for drift missed above, or changes made outside a Claude Code session (direct edits, other tools). It reconciles `docs/` with everything that's changed since the last sync (tracked as a commit SHA in `.claude/agent-memory/docs-maintainer/docs_last_synced.md`), and — as part of that same run — reconciles the Supermemory MCP server (containerTag `sm_project_gsfa`) so its stored facts track the same commit.
- **`/sm-recall <question>`** is the cheap first move for an architecture/design question before reading full docs or source — a Supermemory hit is tens of tokens versus a whole README or file. Falls back to docs/source itself if recall comes up empty or stale.
- The **`docs-maintainer`** agent owns both of the above and also answers architecture questions directly (read-only) without invoking either command.
