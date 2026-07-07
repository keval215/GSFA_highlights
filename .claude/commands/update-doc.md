---
description: Reconcile the docs/ tree with code changes since the docs were last synced (delegates to the docs-maintainer agent).
---

Launch the **docs-maintainer** agent in **update mode** to bring the documentation in `docs/` up to date with the code changes made since the docs were last synced.

Instruct the agent to:

1. Determine the delta — read its `docs-last-synced` project memory (a commit SHA + date) and diff from there to `HEAD`, including uncommitted changes (`git status --short`, working + staged diff). If there is no marker, fall back to recent `git log` plus the working diff.
2. Map each behaviour/interface/threshold/env-var/schema/data-flow change to the doc file that owns it (the `docs/codebase/<package>/README.md` tree, `docs/API.md`, `docs/azure_deploy.md`, `ARCHITECTURE.md`, `GLOSSARY.md`).
3. Edit only the affected docs, surgically, preserving the house style ("What it does" / "Does NOT" / "Connections" sections, `file.py:symbol` references, factual terse tone). Verify each change against the actual source before documenting it.
4. Sync Supermemory for each section edited in step 3: consult `.claude/agent-memory/docs-maintainer/supermemory_index.md` for what's currently stored (containerTag `sm_project_gsfa`), save the replacement fact, best-effort `forget` the prior verbatim entry (may report "no match" — that's a known tool limitation, not an error to fix), and update the index row (new content, new id, new SHA, forget outcome).
5. Report a concise per-file summary of what changed and why, and flag any code/doc discrepancies it did not silently fix.
6. Update the `docs-last-synced` memory to the current `HEAD` SHA and today's date.

Pass through any extra scope the user typed after the command (e.g. a specific package or file to focus on): $ARGUMENTS

Do not let the agent run the pipeline or edit any non-docs file. The agent may call the Supermemory MCP tools (`memory`, `recall`) and may edit `supermemory_index.md` under its agent-memory directory — this is memory bookkeeping, not a docs/code edit.
