# TASKS — implement implementation.md (threaded producer/consumer split for process_clip) — 2026-09-03

Source brief: `implementation.md` (repo root, untracked). Goal: overlap the GPU/decode
pass and the CPU tracker/FSM pass in `service/clip_processor.py::process_clip()` via a
bounded-queue single-producer / main-thread-consumer split, gated behind a new
`CLIP_PIPELINE_THREADED` env var (default OFF). Includes Fix 1 (guaranteed producer
shutdown on every consumer exit path) and Fix 2 (`cap.release()` moves into the producer).

Constraints passed to every agent: read `.claude/agent-memory/shared/CONTEXT.md` first;
never run the pipeline / worker / any clip-processing run; never `git push` or deploy to
the VM.

- [x] 1. Refresh CONTEXT.md to HEAD — owner: thor — deps: none
      DONE: header now `cf9d23e`; §3/§4 state Approach A as landed (6c128c2); §5 records
      full process_clip() baseline + service/config.py `# --- Tunables ---` block for the
      new knob. Note: thor/docs_last_synced.md still at 5cf10e3 — flagged for /update-doc.
      accept: header SHA == `cf9d23e` (currently stale at `5cf10e3`); "Approach A" is now
      committed (6c128c2/cf9d23e) so recent-changes reflects that; note uncommitted
      `M video_analysis/run.py`; capture the current single-threaded structure of
      `service/clip_processor.py::process_clip()` (decode loop + flush_window Pass1/Pass2)
      so captain has an accurate baseline.

- [x] 2. Implement the threaded split in service/clip_processor.py — owner: captain — deps: 1
      DONE: `CLIP_PIPELINE_THREADED` (default False) in service/config.py; process_clip
      split into _pass1 (producer thread + decode + GPU) / _pass2 (main-thread consumer,
      relocated verbatim) behind the knob; serial path byte-for-byte unchanged. Fix 1
      (bounded drain+join+log.error), Fix 2 (cap.release in producer finally), Fix 5
      (producer exc re-raised post-join) all in. New tests/test_clip_processor_threaded.py
      (5 tests incl. required fault-injection). py_compile OK; `pytest tests/` 80 passed /
      1 pre-existing skip; /code-review 6 findings addressed; diff scoped to service/ +
      new test. Deviations: (a) 30s-timeout path also raises RuntimeError on the CLEAN
      consumer path (brief said log-only) — Pass2 exceptions still take precedence;
      (b) shutdown drain is a bounded loop not one-shot. Both defensible; revert path
      offered. On-VM perf/bit-identical validation still open (forbidden here).
      accept:
      - new `CLIP_PIPELINE_THREADED` bool in `service/config.py` (or wherever service env
        vars live), default False; when False, behaviour is byte-for-byte the current
        serial path (no thread created).
      - when True: single daemon producer thread does decode + Pass 1 (player_det.detect_batch
        → team_clf.classify_batch → gk_det.classify → best_ball); main thread consumes a
        `queue.Queue(maxsize=2)` and runs Pass 2 (tracker/ball_tracker/carrier_eng/pass_track
        /counters/correction bookkeeping) — logic relocated verbatim, not re-derived.
      - Fix 1: try/finally around the consumer loop → `stop_event.set()`, drain queue,
        `t.join(timeout=30)`, loud `log.error` if join times out; original exception from
        Pass 2 still propagates unchanged.
      - Fix 2: `cap.release()` lives only in the producer's finally; main thread never
        touches `cap` post-split. Producer always emits the `None` sentinel (clean or error).
      - producer exceptions captured and re-raised on the main thread after join.
      - frame-stride / window semantics identical to current code.
      - a fault-injection unit test: monkeypatch a Pass 2 call to raise mid-clip, assert
        (a) producer thread is joined (no leak), (b) the exception propagates out of
        process_clip unchanged. Add to service tests.
      - verification: `python -m py_compile` clean; existing `service/` tests pass;
        `/code-review` on the diff clean; diff scoped to `service/` only (no
        `video_analysis/run.py`, no other packages).
      - NOT in scope for captain (requires forbidden pipeline/VM runs — surface as open):
        real wall-clock measurement on the T4, full multi-clip end-to-end run, bit-identical
        diff on real processed clips. Captain should instead argue equivalence via code
        review + any offline/mocked test it can run without the pipeline.

- [x] 3. Docs reconciliation — owner: thor — deps: 2
      DONE: edited docs/API.md (env-var table row), docs/codebase/service/README.md
      (config + clip_processor sections), docs/codebase/ARCHITECTURE.md (Mode B flow +
      module map), and CONTEXT.md §4/§5. API HTTP contract unchanged. azure_deploy.md /
      NEEDED_FROM_YOU.md don't exist. Supermemory NOT synced (change uncommitted).
      A full `/update-doc` (5cf10e3..HEAD) is still warranted separately.
      accept: if task 2 landed the env var + threading model, update
      `docs/codebase/service/README.md` (+ config/env-var docs, ARCHITECTURE.md if it
      describes the clip loop) to document `CLIP_PIPELINE_THREADED` and the
      producer/consumer design; otherwise confirm no doc impact. (User may instead prefer
      to run `/update-doc` — offer that.)
