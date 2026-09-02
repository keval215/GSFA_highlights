# AGENTS.md — GSFA_highlights agent fleet

A four-agent system for this repo. **ironman** routes; **hawkeye**, **thor**, and
**captain** do the work. Each specialist is also usable on its own.

> This file is the playbook. Agent behaviour is defined in `.claude/agents/<codename>.md`;
> this file is the shared contract between them and the human-readable map for you.

## Roster

| Codename | Role | File | Tools | Writes |
|---|---|---|---|---|
| **ironman** | Orchestrator / router | `.claude/agents/ironman.md` | `Agent(hawkeye, thor, captain)`, Read, Grep, Glob, Bash | `.claude/agent-memory/shared/TASKS.md` |
| **hawkeye** | Web research | `.claude/agents/hawkeye.md` | WebSearch, WebFetch, Read, Write, Supermemory | `.claude/agent-memory/shared/research/*.md` |
| **thor** | Reader — shared context + `docs/` | `.claude/agents/thor.md` | Read, Grep, Glob, Edit, Write, Bash, Supermemory | `.claude/agent-memory/shared/CONTEXT.md`, `docs/**` (Mode 3 only) |
| **captain** | Coder + self-verify | `.claude/agents/captain.md` | full (inherits all) | application code |

`thor` **replaces the old `docs-maintainer` agent** — it folds in `/update-doc` (Mode 3),
`/sm-recall`, and the Supermemory index protocol, and adds the shared-context job (Mode 2)
and repo Q&A (Mode 1). Its memory lives at `.claude/agent-memory/thor/`.

## How to run it

| Want | Command |
|---|---|
| Every session starts as the orchestrator | already set — `"agent": "ironman"` in `.claude/settings.json` |
| One session as the orchestrator | `claude --agent ironman` |
| Ask the reader a repo question directly | `claude --agent thor "how does the carrier engine pick the ball holder?"` |
| Run one specialist directly | `claude --agent <hawkeye\|thor\|captain> "<task>"` |
| Background run | `claude --bg --agent ironman "<request>"` |
| Delegate once from a normal session | "use the ironman agent to …" |

## Routing table

| The request is… | Owner |
|---|---|
| "find out / research / what's the state of the art on …" | **hawkeye** |
| "how does X work / where is Y / why was Z decided" | **thor** (Mode 1) |
| "code changed — bring shared context up to date" | **thor** (Mode 2) |
| "/update-doc" or "update/sync the docs" | **thor** (Mode 3) |
| "implement / fix / refactor / add …" | **captain** |
| anything spanning ≥2 of the above | **ironman** decomposes and dispatches |

## Task-list protocol (ironman)

1. Read `.claude/agent-memory/shared/CONTEXT.md`; if stale/thin, have `thor` refresh it first.
2. Write `.claude/agent-memory/shared/TASKS.md`: ordered checkbox items, each with an
   **owner**, **acceptance criteria**, and **dependencies**.
3. Dispatch in dependency order, parallel where independent; pass each agent the task, its
   acceptance criteria, and "read `CONTEXT.md` first".
4. On return: verify against acceptance criteria → tick the box + one-line result note; on
   failure, re-dispatch with corrections (≤2 retries, then surface the blocker).
5. If `captain` changed documented behaviour → final task to `thor` to reconcile
   `docs/` + Supermemory (or tell the user to run `/update-doc`).
6. Relay one consolidated summary. Subagent raw reports don't reach the user — ironman conveys.

## Definition of done

- Every `TASKS.md` box ticked has evidence in the owning subagent's report.
- `captain` changes carry a verification log: `py_compile` + relevant tests + `git diff
  --stat` + `/code-review` on the diff + a non-pipeline smoke check. No "done" without it.
- Documented-behaviour changes are reconciled by `thor`, not left to drift.
- `hawkeye` claims are each cited (URL + access date).

## Standing prohibitions (inherited by every agent, from `CLAUDE.md`)

- **Never run** `video_analysis/possession.py`, the service worker, or any clip-processing
  run without explicit user permission. Verification never needs them.
- **Never `git push`**, and never deploy/copy to the production VM (`gsfa-highlights`).
  Preparing a local commit + message is fine.
- Team classifier is always `GSFATeamClassifier` (SigLIP); never the colour-histogram variant.
- Docker uses deadsnakes Python 3.11.
- `heatmap.py` / `shots_on_t.py` are standalone/experimental — don't align them to the
  unified YOLOv11m model unless the task says so.
- If code and docs disagree, **code wins** — but flag it so drift doesn't accumulate.

## Shared memory

```
.claude/agent-memory/
├── shared/
│   ├── CONTEXT.md          living digest, every agent reads first   (owner: thor)
│   ├── TASKS.md            current request's checklist              (owner: ironman)
│   └── research/<topic>.md sourced briefs                           (owner: hawkeye)
├── ironman/  hawkeye/  captain/   per-agent durable notes (MEMORY.md index + fact files)
└── thor/     docs_last_synced.md, supermemory_index.md, recurring_discrepancies.md, MEMORY.md
```

Plus **Supermemory** (`containerTag sm_project_gsfa`): durable cross-session facts —
architecture rows (via `thor/supermemory_index.md`, the source of truth because
Supermemory's `forget`-by-content is unreliable) and `research`-tagged findings from hawkeye.

## Known issue — `mtnewswire` MCP

The broken `plugin:financial-analysis:mtnewswire` server was removed by deleting its key
from `~/.claude/plugins/cache/claude-for-financial-services/financial-analysis/0.1.1/.mcp.json`.
That cache dir is version-pinned: a plugin reinstall/update re-adds the entry because
upstream `anthropics/financial-services` still ships the dead URL
(`https://vast-mcp.blueskyapi.com`). If the startup error returns, re-delete the key, or
disable the whole `financial-analysis` plugin in `~/.claude/settings.json`.
