---
name: recurring_discrepancies
description: Known code/doc discrepancies found during syncs that were deliberately not silently fixed — surface these again if still true.
type: project
---

Open items as of the 2026-07-06 sync (commit `2ecc1ee`):

- **`PlayerDetector.classes` allow-list is unused in production.** `detectors/player_detector.py`
  added a `classes: list[int] | None` constructor param with a comment claiming "Local
  possession passes `[0, 1, 2]` to hard-filter referees" — but `video_analysis/possession.py`'s
  `run()` does **not** pass `classes=` to `PlayerDetector(...)` (verified via grep, confirmed
  again on the commit that introduced it). The only caller that actually uses `classes=[0,1,2]`
  is the manual debug harness `team_classifier/test_team_classifier.py`. Docs describe the
  param neutrally (no claim about local possession using it) — don't add that claim until
  `possession.py` actually passes it.
- **`sql/schema.sql` has no inline migration note for the newer additions.** The v2 migration
  (`next_clip_seq_h1/h2`) has an inline `ALTER TABLE` comment; the newer
  `minute_stats.clip_duration_seconds` column and the whole `post_processing` table do not —
  anyone deploying against a pre-existing DB has to hand-write those ALTER/CREATE statements.
  Flagged, not fixed (would require editing schema.sql, out of docs-maintainer scope).
- **Previous manual doc edits (not via /update-doc) broke docs/API.md.** Commit `a04fea8`
  ("feat - addes post processing logic") hand-edited `docs/API.md` to add `POST
  /post-processing` but in doing so deleted the entire `GET /metrics` section and the
  `## Callback (outbox)` header/intro — leaving orphaned body-shape text under a `GET /health`
  section. Restored in this sync. Lesson: don't assume commits that touch `docs/*.py` files by
  the same author who touches code are safe — always diff the actual doc content, not just
  trust that "docs were already updated."
