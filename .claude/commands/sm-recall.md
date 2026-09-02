---
description: Cheap Supermemory recall for this repo (containerTag sm_project_gsfa) before reading full docs/source.
---

Call `mcp__supermemory__search_memory` with `query: "$ARGUMENTS"`,
`containerTag: "sm_project_gsfa"`, `includeProfile: false`.

Present the returned memories concisely.

These reflect the repo as of the last `/update-doc` sync (see
`.claude/agent-memory/thor/docs_last_synced.md` for the SHA/date) — if the question
concerns something changed after that SHA, or recall returns nothing or looks
low-confidence, fall back to `.claude/agent-memory/shared/CONTEXT.md`,
`docs/codebase/ARCHITECTURE.md`, the owning package README, or the source directly.

Do not treat recall output alone as sufficient basis for edits or load-bearing
decisions — verify against current source first.
