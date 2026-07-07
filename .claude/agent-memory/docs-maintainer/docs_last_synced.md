---
name: docs_last_synced
description: Commit SHA + date of the last successful /update-doc sync — diff from here next time.
type: project
---

Last synced HEAD: `2ecc1eea3eff027bb1e683b42c010434a312f527` on 2026-07-06.

**Why:** No prior `docs-last-synced` marker existed (first real sync). Baseline used
was `08a2639` (the commit that added the docs/codebase tree) — verified that
`08a2639`'s doc tree already reflected the YOLOv11m-unification commit `bc3504b`
(which predates it chronologically), so the actual delta reconciled was
`08a2639..2ecc1ee` (commits `12ea300`, `f2f00ae`, `a04fea8`, `2ecc1ee` — "csv export",
"clip metadata field", "post processing logic" x2).

**How to apply:** Next `/update-doc` run should `git diff 2ecc1eea3eff..HEAD` (plus
working tree) rather than falling back to log-scanning.
