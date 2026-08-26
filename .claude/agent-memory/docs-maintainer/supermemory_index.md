---
name: supermemory_index
description: Maps docs/ sections to verbatim Supermemory content under containerTag sm_project_gsfa, plus the last-synced commit SHA for each. Consult before saving/forgetting so /update-doc replaces facts precisely instead of duplicating them.
type: reference
---

containerTag for every row below: `sm_project_gsfa`

**Known limitation (learned 2026-07-07, do not re-litigate):** `mcp__supermemory__add_memory`'s
`forget` action only matches against a document's small, auto-generated `memoryEntries`
(atomic one-sentence facts Supermemory derives asynchronously) — never the raw saved
`content` string, and never the auto-generated `summary`. Both were tried directly
against real saved documents and failed ("No matching memory found to forget... only
document chunks matched"). Several documents never get decomposed into `memoryEntries`
at all (empty array), making them **permanently un-forgettable by content** via this
tool. Consequence: **do not rely on `forget` to remove a stale document.** Always
attempt it (cheap, sometimes works if entries exist), but treat this index — not
Supermemory's own state — as the source of truth for "what's canonical." If `forget`
reports success, mark the row's "Forget attempted" cell `ok`; if it reports "no
matching memory," mark it `stale-orphan (harmless)` and leave the new row as the one
`recall()`/`search_memory` consumers should trust. Never spend more than one retry on a
failed forget.

**Actual MCP tool names in this environment** (the docs-maintainer role instructions say
`mcp__supermemory__recall` / `mcp__supermemory__memory(action:"save"/"forget")`, but the
tools actually exposed are `mcp__supermemory__search_memory` for recall and
`mcp__supermemory__add_memory` with `action: "save"|"forget"` for save/forget — use
these).

| Doc file / section | Verbatim saved content (current) | Document id | Last-synced SHA | Forget attempted (prior version) |
|---|---|---|---|---|
| docs/codebase/README.md — scope/overview | "GSFA_highlights is a futsal/football match-video analysis system (repo at D:\GSFA_highlights). It derives per-team (team0/team1) possession %, completed passes, interceptions, and ball-lost events from video, producing an append-only event log. It does not yet edit/cut highlight reels — the events table is raw material for a future highlight-reel builder." | bbRq4ZfckPe8HznZTiWLE6 | 2ecc1eea3eff027bb1e683b42c010434a312f527 | n/a (unchanged this sync — content still accurate) |
| docs/codebase/ARCHITECTURE.md §1,5 — per-frame pipeline chain | "GSFA_highlights per-frame pipeline chain (perception stage): PlayerDetector (unified YOLOv11m, modules/detectors/player_detector.py...) -> GSFATeamClassifier (modules/team_classifier/, SigLIP, CUDA-only fail-fast) -> GoalkeeperDetector (modules/detectors/goalkeeper_detector.py) runs INDEPENDENTLY of/in parallel with GSFATeamClassifier... Then 'who has the ball' stage: PlayerTracker (BoT-SORT, modules/tracking/) -> BallTracker (Kalman) -> CarrierEngine (foot-zone) -> PassEventTracker (FSM), all in modules/possession/... Every numeric tuning parameter now comes from one RulesetConfig dataclass (rulesets/base.py) selected per match/run." | PKySyafFrRQqDKHjMUVtUp | c7ef0b6245310cfc783835ca081a7624219711fc + uncommitted modules/rulesets refactor (see docs_last_synced.md) | attempted, `stale-orphan (harmless)` |
| docs/codebase/README.md + ARCHITECTURE.md §2-3 — two execution modes | "GSFA_highlights has two execution modes... Mode A: local script `video_analysis/run.py [--ruleset futsal\|classic]` -> run() (replaces deleted possession.py's run()), PROCESS_DURATION_SEC now 480s... Mode B: production service... Both endpoints now require team0_colour/team1_colour/team0_gk_colour/team1_gk_colour on every request, and accept an optional `ruleset` field... Mode B never renders video." | DRZJj4C3nYTozU7P2V8dMZ | c7ef0b6245310cfc783835ca081a7624219711fc + uncommitted refactor | `ok` |
| docs/codebase/ARCHITECTURE.md §4 / detectors/README.md — wire-format types | "GSFA_highlights shared wire-format types, defined in modules/detectors/player_detector.py (moved from top-level detectors/): Detection (...) and FrameDetections (...). modules/possession/ (split out of the deleted video_analysis/possession.py) adds BallDetection and CarrierState..." | egmRVtFdj5qZ51aEiKZ9jj | c7ef0b6245310cfc783835ca081a7624219711fc + uncommitted refactor | attempted, `stale-orphan (harmless)` |
| docs/codebase/service/README.md — worker clip-handling flow | "GSFA_highlights service-mode worker flow (service/worker.py Worker._handle...): ...4) ruleset resolved via get_ruleset(db.get_match_ruleset(match_id)) then MatchSessionManager.get_or_create(match_id, ruleset)... 5) clip_processor.process_clip() first calls session.ensure_gk_ready()... For kind=post_processing whole-match jobs: queue message deleted BEFORE processing (no retry), MatchSession constructed directly (never via manager)... ModelBundle lazily loads PlayerDetector per ruleset... FIT_SAMPLE_EVERY default raised 5->30... docker-compose.yml mem_limit/mem_reservation on both containers (added 2026-07-07)." | uZZq4Tq8iLLE9Sa5GZEXa4 | c7ef0b6245310cfc783835ca081a7624219711fc + uncommitted refactor | attempted, `stale-orphan (harmless)` |
| docs/codebase/ARCHITECTURE.md §7 / service/README.md — correctness invariants | "GSFA_highlights correctness invariants: (1) Provisional credit + retroactive adjustment... (2) Cumulative-on-read... (3) Transactional outbox... (4) Idempotency + ordering guards..." | ey78oW697x8dnGn64bKfQz | 2ecc1eea3eff027bb1e683b42c010434a312f527 | n/a (unchanged this sync — content still accurate) |
| docs/codebase/ARCHITECTURE.md §6 / detectors/README.md — unified YOLOv11m change | "GSFA_highlights recent architectural change: ...one unified YOLOv11m model... The old separate RF-DETR ball model and weights were removed entirely. The root-level heatmap.py and shots_on_t.py standalone tools, which were out of scope for that unification and kept the older separate detectors, have themselves since been deleted from the repo entirely (along with test.py and several old OCR/team-classification probe scripts under scripts/)." | FQNUxURB4SpSpHJVgbGBvu | c7ef0b6245310cfc783835ca081a7624219711fc + uncommitted refactor | attempted, `stale-orphan (harmless)` |
| docs/codebase/README.md — repository map | "GSFA_highlights repository layout (as of the modules/ + rulesets/ refactor): modules/ is the shared CV pipeline package containing modules/detectors/, modules/team_classifier/, modules/tracking/, modules/possession/... rulesets/ holds per-sport RulesetConfig profiles... video_analysis/ now holds only run.py... scripts/ has ops tools plus several new standalone Colab tracker-research scripts... root-level heatmap.py, shots_on_t.py, test.py all deleted." | 95ZzTpSVC3beKTyKk4Ddnk | c7ef0b6245310cfc783835ca081a7624219711fc + uncommitted refactor | attempted, `stale-orphan (harmless)` |
| docs/codebase/team_classifier/README.md — SigLIP vs colour-histogram decision + CUDA-only | "GSFA_highlights team classifier: GSFATeamClassifier is CUDA-only as of the 'Enforce CUDA-only SigLIP' commit — __init__ defaults device='cuda' (was 'cpu') and asserts torch.cuda.is_available()... no CPU fallback. Always use GSFATeamClassifier, never ColourHistogramTeamClassifier... crop/blur/quality knobs are now constructor parameters (RulesetConfig-overridable), defaults unchanged." | FBdT8vP6jAuz73dzTnJZmQ | c7ef0b6245310cfc783835ca081a7624219711fc + uncommitted refactor | `ok` |
| docs/codebase/infra/README.md — Docker/Python 3.11 constraint | "GSFA_highlights deployment constraint: Docker must use deadsnakes Python 3.11 — Ubuntu's default Python 3.11.0rc1 segfaults torch.jit.script inside rfdetr-related code paths." | iVhwjMCGRbFoJcBr8wzai7 | 2ecc1eea3eff027bb1e683b42c010434a312f527 | n/a (unchanged this sync — content still accurate) |
| docs/codebase/detectors/README.md — GoalkeeperDetector rewrite (colour match) | "GSFA_highlights GoalkeeperDetector was rewritten (modules/detectors/goalkeeper_detector.py): no longer fit-then-classify... now fit-free: caller supplies two reference jersey colours... classify(frame, detections) finds per-colour the single closest-matching player (reusing GSFATeamClassifier's HSV colour-vector static helpers)... In the service, team0_gk_colour/team1_gk_colour are now required fields... MatchSession.ensure_gk_ready() lazily constructs the detector..." | oqNwWHjGSf6XYqLUYyjd9q | c7ef0b6245310cfc783835ca081a7624219711fc + uncommitted refactor | n/a (new row, no prior version) |
| docs/codebase/rulesets/README.md — new rulesets/ package | "GSFA_highlights rulesets/ package: one RulesetConfig frozen dataclass (rulesets/base.py) bundles every sport-tunable CV parameter... rulesets/futsal.py (FUTSAL) is the production default — byte-identical to pre-refactor hardcoded values. rulesets/classic.py (CLASSIC, 11-a-side) is NOT production-ready — several PLACEHOLDER fields, no trained checkpoint yet. rulesets/registry.py provides get_ruleset/available_rulesets/DEFAULT_RULESET... sql/schema.sql matches.ruleset column (migration v4, NOT yet applied to live Azure DB)." | sNiumPoswvVCPLHzUDVMNy | c7ef0b6245310cfc783835ca081a7624219711fc + uncommitted refactor | n/a (new row, no prior version) |

**Note on "Last-synced SHA" for this sync (2026-08-24):** the bulk of what changed
(modules/ + rulesets/ reorganisation) is **uncommitted working-tree state**, not yet at
a commit SHA — see `docs_last_synced.md` for the full explanation. Rows above marked
"`c7ef0b6... + uncommitted refactor`" reflect that combined state as of this sync date;
if the refactor is later committed, the next `/update-doc` should re-verify these rows
still match rather than assuming the SHA alone confirms freshness.

**Migration note (2026-07-07):** the original 10 documents (pre-refactor) were saved
under `containerTag: sm_project_default` before being moved to `sm_project_gsfa`;
forgetting the `sm_project_default` originals mostly failed (see limitation note above)
and those are harmless stale orphans left in that other tag. Not re-attempted.
