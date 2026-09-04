# Shared CONTEXT — GSFA_highlights

> Refreshed: 2026-09-03 · HEAD: `cf9d23e` (working tree: uncommitted `CLIP_PIPELINE_THREADED`
> split in `service/clip_processor.py` + `service/config.py` + new test; `M video_analysis/run.py` churn)
> Owner: thor. Every agent reads this file first. Keep it to ~one screen.

## 1. What this repo is
Futsal/football match-video analysis. From video it derives, per team (`team_a`/`team_b`):
possession %, completed passes, interceptions, ball-lost events → append-only `events` +
`minute_stats` tables. Does **not** yet cut highlight reels — those tables are raw material.

Two execution modes, one shared CV pipeline:
- **Local script:** `python video_analysis/run.py [--ruleset futsal|classic]` — one whole
  video file, renders an annotated `.mp4` + printed summary.
- **Production service:** `service/api.py` (ingest, no GPU) + `service/worker.py` (GPU) —
  60 s clips over HTTP, one per match-minute (+ a whole-match `/post-processing` path).
  Output is Azure SQL rows + an HTTP advance-stats callback. **No video render.**

## 2. Key modules / perception chain (per frame)
`PlayerDetector` (`modules/detectors/player_detector.py`, unified YOLOv11m) →
`GSFATeamClassifier` (`modules/team_classifier/team_classifier.py`, SigLIP→UMAP→KMeans, CUDA-only) →
`GoalkeeperDetector` (`modules/detectors/goalkeeper_detector.py`, fit-free colour match, runs
independently of the team classifier) →
`PlayerTracker` (`modules/tracking/player_tracker.py`, BoT-SORT, consumes SigLIP embeddings) →
`BallTracker` (`modules/possession/ball_tracker.py`, Kalman + coast) →
`CarrierEngine` (`modules/possession/carrier_engine.py`, foot-zone) →
`PassEventTracker` (`modules/possession/pass_event_tracker.py`, release/travel/receive FSM).
Labels/constants: `modules/possession/labels.py` (mirrored torch-free in `service/stats.py`;
`service/session.py` asserts the two match at import). Tuning: one `RulesetConfig`
(`rulesets/base.py`), selected per match/run.

## 3. Load-bearing invariants
- Team classifier is always `GSFATeamClassifier` (SigLIP). Never the colour-histogram variant.
- Unified YOLOv11m; class set is per-ruleset via `RulesetConfig.class_names`
  (futsal 4-class incl. `goal_post`, classic separate 3-class checkpoint, no `goal_post`).
- Docker uses deadsnakes Python 3.11 (Ubuntu default 3.11.0rc1 segfaults `torch.jit.script`).
- Naming is `team_a` / `team_b` end-to-end (since v6); classic is the default ruleset.
- A match's `ruleset` is fixed at creation. `matches` names are fill-in-only (COALESCE).
- The four team/GK **colour** columns are mutable mid-match — every request overwrites
  them and a genuine colour change bumps `matches.fit_generation`, driving a per-match
  team/GK re-fit (tracker / ball / carrier / pass-FSM state preserved, no worker restart).
  `team_a`/`team_b` column assignment is colour-anchored at write time
  (`stats.orient_for_team_a`), not raw KMeans label order. Superseded `minute_stats` /
  `events` rows carry `superseded = 1` and are excluded from cumulative reads.
  (Landed `6c128c2`, "Approach A".)
- Never run the pipeline / worker / clip processing without explicit user permission.
- Never `git push` or deploy to the production VM.

## 4. Recent changes
- **Uncommitted working tree:**
  - `service/config.py` + `service/clip_processor.py` + new
    `tests/test_clip_processor_threaded.py` — opt-in threaded producer/consumer split in
    `process_clip()` behind a new `CLIP_PIPELINE_THREADED` bool env var (default **False**).
    Off = the current serial decode loop, byte-for-byte unchanged, **no thread created**.
    On = a daemon producer thread runs decode + Pass 1 into a `queue.Queue(maxsize=2)`;
    the main thread consumes and runs Pass 2. Pass 1 / Pass 2 relocated verbatim into
    `_pass1` / `_pass2` closures. No API / SQL / threshold / `team_a`-`team_b` change.
    Docs reconciled 2026-09-03 (`docs/API.md` env table, `service/README.md`,
    `ARCHITECTURE.md`).
  - `M video_analysis/run.py` — local-dev constants churn (hardcoded `VIDEO_PATH` swapped,
    debug `TEAM_BGR`/`_PALETTE` colours changed, module-level `MODEL_PATH` /
    `TEAM_A_GK_COLOUR` / `TEAM_B_GK_COLOUR` removed so the args default to `None` /
    ruleset weights). No service or pipeline behaviour change.
- `cf9d23e` — per-clip timing log promoted DEBUG→INFO in `clip_processor.py`; urllib3
  chatter silenced (`service/logging_setup.py`).
- `6c128c2` — "Approach A" mid-match team/GK colour-change handling **landed** (this is
  the diff that the 2026-09-01 CONTEXT flagged as uncommitted). Spans `service/api.py`,
  `clip_processor.py`, `db.py`, `session.py`, `stats.py`, `worker.py`, `sql/schema.sql`
  (+ new `tests/`). Key points:
  - schema "Migration v7": `matches.fit_generation INT NOT NULL DEFAULT 1`;
    `minute_stats.superseded` + `events.superseded` `BIT NOT NULL DEFAULT 0`.
  - `db.ensure_match`: colour columns become `COALESCE(?, existing)` (overwrite) and bump
    `fit_generation` when a normalised incoming colour differs from stored; names still
    COALESCE. New `db.get_fit_generation`, `mark_minutes_superseded` (minute_stats+events
    ≤ given half/minute), `bump_fit_generation`. `cumulative_read` gains `AND superseded = 0`.
  - `MatchSession`: `fit_generation` + `team_a_cluster_id` (backed by `fit_meta.json`
    sidecar next to `team_siglip.pkl`), `reset_for_new_generation()`,
    `last_written_team_a_cluster_id` snapshot in `finish_clip()`.
    `MatchSessionManager.get_or_create(..., want_generation=)` → `(session, just_reset)`;
    resets that match's team classifier + GK detector when DB generation is ahead,
    preserving tracker / ball / carrier / pass-FSM state. No worker restart.
  - `worker._handle`: reads `get_fit_generation`, passes as `want_generation`; on reset
    calls `mark_minutes_superseded(match_id, last_half, last_minute)`. `post_processing`
    path builds its own session directly — unaffected.
  - `stats.orient_for_team_a(result, team_a_cluster_id, correction_team_a_cluster_id=…)` —
    reorders a clip's minute_row/events/correction so `team_a`/`0` == the cluster matching
    `team_a_colour`. Applied in `clip_processor` at write time, never at read time.
    `build_payload` lost `team_id_to_name`; `db.write_clip_result` lost `team_names`.
  - New endpoint `POST /api/matches/{match_id}/reset-fit` (optional JSON body of the 6
    name/colour fields): unconditionally bumps `fit_generation`; 404 unknown match;
    422 on a present-but-blank field. Returns `{match_id, fit_generation}`.
- `5cf10e3` — detector per-ruleset class map (classic 3-class, no `goal_post`).
- `3697434` — rename `team0`/`team1` → `team_a`/`team_b`; classic is now the default ruleset.
- `3844daa` / `471d643` — Docker + CV pipeline merged into `modules/`, config-driven rulesets.

## 5. Upcoming-task baseline — `clip_processor.py` + service config
`service/clip_processor.py::process_clip(session, clip_path, half, minute,
clip_duration_seconds, clip_blob_path=None) -> ClipResult` (file is ~347 lines):
- Guards: raises if `session.team_clf is None`; calls `session.ensure_gk_ready()`.
- Opens `cap = cv2.VideoCapture(clip_path)`; `frame_step = max(1, round(fps /
  config.TARGET_PROCESS_FPS))`, `window_k = max(1, config.CLIP_BATCH_WINDOW)`.
- **Decode loop — serial by default, opt-in threaded** (gated by
  `config.CLIP_PIPELINE_THREADED`, default False):
  - **Serial (default):** `while True` (~line 172): `ret, frame = cap.read()` → skip unless
    `fidx % frame_step == 0` → append to `win_frames` / `win_fidx` → when
    `len(win_frames) >= window_k` call `flush_window()` (= `_pass1` then `_pass2`). One
    final `flush_window()` after the loop for the trailing partial window. `cap.release()`
    in that loop's `finally:`. **No thread created — byte-for-byte the pre-threading path.**
  - **Threaded (`=True`):** one `daemon` producer thread runs the decode loop + `_pass1`
    into a `queue.Queue(maxsize=2)`; the main thread consumes and runs `_pass2`. `None`
    sentinel to stop; `cap.release()` moves into the producer's `finally`. On any consumer
    exit (clean / Pass 2 exception / KeyboardInterrupt): `stop_event.set()`, drain queue,
    `producer.join` bounded to 30 s, `log.error` naming `match_id` if it won't die, and a
    `RuntimeError` on the clean path if still wedged past the timeout. Producer-thread
    exceptions captured and re-raised on the main thread after the join. Timing dict /
    INFO log still emitted in both modes; threaded `total` is summed stage time, not
    wall-clock.
- `_pass1(frames, fidxs)` / `_pass2(frames, dets_list, balls)` — the two passes relocated
  verbatim into closures (`_pass2` keeps the `nonlocal correction,
  correction_team_a_cluster_id, n_processed`); `flush_window()` is now just
  `_pass1` → `_pass2` for the serial path. Behaviour below is unchanged:
  - **Pass 1 — batched GPU, stateless:** `session.player_det.detect_batch(win_frames,
    win_fidx, fps)` → `session.team_clf.classify_batch(win_frames, dets_list)` →
    `session.gk_det.classify(frame, dets)` per frame (when `gk_det` set) →
    `balls = [best_ball(d) for d in dets_list]` (ball from the same unified pass, every
    processed frame — no separate model/stride).
  - **Pass 2 — sequential CPU, stateful**, per frame in window order: `session.tracker.update`
    → `session.ball_tracker.update` → `session.carrier_eng.update` → `session.pass_track.update`
    (then `session.proc_idx += 1`) → `counters.add_label(label)`. For each returned
    adjustment tuple `(kind, team_id, n)`: `session.split_adjustment(n)` →
    `counters.apply_adjustment(...)`; if `prior_n > 0` and `session.last_written` is set,
    build one `PriorCorrection` (boundary correction) and snapshot
    `correction_team_a_cluster_id = session.last_written_team_a_cluster_id`. At most one
    per clip; a second is logged as an error and dropped.
  - Per-stage ms accrue in dict `t` (decode / player_det / team_clf / ball_det / tracker /
    ball_kalman / carrier / pass).
- **Post-loop tail:** iterate `session.new_events()` → build `EventRow`s (kind via
  `KIND_MAP`) + `counters.count_event(...)` → `session.finish_clip(half, minute)` → INFO
  timing log when `n_processed` → assemble `ClipResult(minute_row=counters.to_minute_row(
  match_id, half, minute, clip_blob_path, clip_duration_seconds), correction=correction,
  events=event_rows)` → `return orient_for_team_a(result, session.team_a_cluster_id,
  correction_team_a_cluster_id)`.

Service env-vars are all defined in **`service/config.py`** (`os.environ.get`; required
ones via `_required`). The `# --- Tunables ---` block is where a new knob is added:
`TARGET_PROCESS_FPS` (15), `CMC_METHOD` (ecc), `CLIP_BATCH_WINDOW` (16),
`CLIP_PIPELINE_THREADED` (false — opt-in producer/consumer split, above), `DEVICE` (cuda),
`FIT_SAMPLE_EVERY` (30), `FIT_SILHOUETTE_MIN` (0.20), `QUEUE_VISIBILITY_SEC` (90),
`MAX_DEQUEUE_COUNT` (3), `ORDERING_RETRIES`/`_RETRY_DELAY`, `CALLBACK_RETRIES`/`_BACKOFF_BASE`,
`SESSION_IDLE_EVICT_S` (30 min), `MAX_UPLOAD_GB` (2), `API_PORT` (8000). Model weights via
`config.player_weights(ruleset)` (futsal → `PLAYER_WEIGHTS`, else `<RULESET>_PLAYER_WEIGHTS`).

## 6. Open research findings
- None. `.claude/agent-memory/shared/research/` holds only `README.md` (no briefs yet).

## 7. Pointers
- Big picture: `docs/codebase/ARCHITECTURE.md` (mid-match colour-change flow is in §3 + §7)
- Terms: `docs/codebase/GLOSSARY.md` (`fit_generation`, `superseded`, `team_a_cluster_id`)
- Service detail: `docs/codebase/service/README.md`; HTTP contract: `docs/API.md`
- Schema: `docs/codebase/sql/README.md` (migration v7)
- Per-package detail: `docs/codebase/<package>/README.md`
- Last docs sync SHA/date: `.claude/agent-memory/thor/docs_last_synced.md`
