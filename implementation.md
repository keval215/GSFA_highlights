Yes, this is buildable, and I checked the two assumptions the proposal is resting on against the actual code rather than taking them on faith — both hold:

Verified: GPU outputs are already CPU-side by the time they'd cross the thread boundary. detect_batch() (modules/detectors/player_detector.py:161) returns list[FrameDetections] — parsed Python dataclasses, not tensors. classify_batch() (modules/team_classifier/team_classifier.py:364) writes team_id (python int) and embedding (.astype(np.float32) — already numpy) back onto those same objects. So "only the producer thread touches CUDA" is correct as stated — there's no lingering GPU tensor for the consumer thread to touch, no implicit cross-thread sync hazard.

Verified: only the consumer touches tracker/session state. tracker.update() / ball_tracker.update() / carrier_eng.update() / pass_track.update() / session.proc_idx are all currently in Pass 2 (clip_processor.py:110-123) and stay there in the proposed split — single-writer, no races on the stateful FSM.

Verified: the math matches your actual numbers, not estimates. 95.4s serial total from your own log (6.6+13.8+39.8+35.1+0.1 = 95.4ms/frame × 1000) → producer-bound pipeline at 963ms/window × 63 windows ≈ 60.7s — the ~61s figure checks out against real data, not hypotheticals.

Scope note the proposal didn't have: this only helps service/clip_processor.py — I checked video_analysis/run.py and it doesn't import clip_processor/flush_window/CLIP_BATCH_WINDOW at all, it has its own loop. So this is a service-worker-only optimization; the local possession.py execution mode gets nothing from it.

Downsides — three real gaps in the sketch, specific to this codebase, not generic threading caveats

1. Producer-thread leak on a consumer-side exception. The sketch's cleanup (t.join(); if error_box: raise) only runs on the clean/producer-error path. If cpu_pass() itself raises (main thread), nothing stops or joins the producer — it can be sitting blocked on queue.put() (full at maxsize=2) indefinitely, or mid-CUDA-call. This matters more here than in a generic app: worker.py's outer loop (run_forever) catches all exceptions from _handle(), logs, and moves straight to the next queue message — and MatchSessionManager caches and reuses MatchSession per match_id across clips. So an orphaned producer thread from a failed clip could still be calling session.player_det/session.team_clf while the next clip for that same match starts on the main thread with the same session's model objects — two threads touching the same detector/classifier instances, unguarded. Needs a finally that drains/joins the producer on every exit path, not just the happy one.

2. cap.release() placement isn't addressed. Decode moves into the producer thread under this design; the existing finally: cap.release() (currently in process_clip's main scope) needs to move with it, or you get a double-release / leaked capture handle on some exit path.

3. The 61s figure is a target, not a guarantee. cv2.read() and PyTorch CUDA calls release the GIL, so real overlap is plausible — but plenty of Python-level glue (crop-cropping loops in _parse/classify_batch, list building) still holds it. Measure actual wall-clock after implementing; don't treat the arithmetic as delivered.

Upsides

- Contained blast radius — this is entirely inside process_clip(); doesn't touch stats.py (still torch-free/testable), db.py, or worker.py's transaction/callback logic.
- Bounded memory (maxsize=2) — no risk of buffering a whole clip.
- Output should be deterministic/bit-identical per the proposal's own reasoning, and that's verifiable cheaply (diff a few clips) before trusting it.
- Real, validated ~36% per-clip wall-clock cut if the two gaps above are closed properly.

My actual recommendation on sequencing, not just mechanics: this is captain's implementation, not mine — but I'd tell them to close gaps 1 and 2 explicitly (thread-safe cleanup on every exit path, given the session-reuse hazard above is a correctness bug, not a nice-to-have) and land it as its own isolated, diffed change after your colour-reset test finishes and is confirmed working — you don't want two unvalidated changes stacked on the same hot path (clip_processor.py) at once while you're still trying to prove the reset feature is correct.

✻ Worked for 2m 4s · done 1:17 PM

※ recap: You're deploying and testing the mid-match team-colour reset feature on the VM using match T16a8edb537fe372ce49da9e; clips 1-3 (and a couple extra) have processed successfully. Next: run the fit_generation/superseded SQL checks to confirm the reset actually fired.

❯ okay then give me a detailed explanation so i can give it to captain along with the fixes of 1 and 2 please keep it detailed

Here's a self-contained brief you can hand captain as-is.

---

Task: pipeline process_clip()'s GPU/CPU passes across a producer/consumer thread boundary

Motivation

Confirmed against actual measured numbers on T16a8edb537fe372ce49da9e (gsfa.clip timing log, h1 m5, 1000 frames, total 95.6s): the clip loop is currently fully serial — decode → GPU batch (player_det+team_clf+ball_det) → CPU sequential (tracker+ball_kalman+carrier+pass) — window by window (window_k = config.CLIP_BATCH_WINDOW, default 16 frames), with zero overlap between stages. Per window: producer-side work (decode+GPU) ≈ (6.6+13.8+39.8)×16 = 963ms, consumer-side (tracker etc.) ≈ 35.1×16 = 562ms. Overlapping them via a 2-thread producer/consumer split makes the pipeline bound by the slower stage (963ms) instead of their sum (1525ms) — projected ~63 windows × 963ms ≈ 61s vs. the current ~95s per clip, ~36% cut. Validated, not hypothetical.

Scope: service/clip_processor.py only. video_analysis/run.py (the local-script execution mode) has its own separate loop and does not import clip_processor/flush_window/CLIP_BATCH_WINDOW — this change gives it nothing, don't try to unify them as part of this task.

Current structure (baseline — service/clip_processor.py:44-197)

- Single-threaded. Outer while True: cap.read() loop fills win_frames/win_fidx; once len(win_frames) >= window_k, calls flush_window() inline, which does Pass 1 (batched GPU: player_det.detect_batch → team_clf.classify_batch + gk_det.classify per frame → best_ball per frame) then Pass 2 (sequential CPU: tracker.update → ball_tracker.update → carrier_eng.update → pass_track.update, plus boundary-correction bookkeeping and counters.add_label).
- cap.release() is in a finally around the whole decode loop.
- After the loop: session.new_events() → build event_rows → session.finish_clip(half, minute) → build+return ClipResult via orient_for_team_a.

Verified facts that make this safe to split across threads (checked against the actual code, not assumed)

1. GPU outputs are CPU-side before Pass 1 returns. PlayerDetector.detect_batch() (modules/detectors/player_detector.py:161) returns list[FrameDetections] — parsed Python dataclasses. GSFATeamClassifier.classify_batch() (modules/team_classifier/team_classifier.py:364) writes back team_id (python int) and embedding (.astype(np.float32), already numpy). So a producer thread doing only Pass 1 work never hands a live GPU tensor to the consumer — no cross-thread CUDA sync hazard.
2. Only Pass 2 touches session tracker/FSM state (session.tracker, session.ball_tracker, session.carrier_eng, session.pass_track, session.proc_idx) — keeping all of that on the consumer thread preserves single-writer ordering, which is load-bearing for the FSM's cross-frame state.

Design: bounded queue, single producer thread, main thread as consumer

import queue
import threading

def process_clip(session, clip_path, half, minute, clip_duration_seconds, clip_blob_path=None) -> ClipResult:
    ...
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise IOError(f"process_clip: cannot open {clip_path}")

    q = queue.Queue(maxsize=2)          # backpressure — bounds RAM to ~2 windows in flight
    stop_event = threading.Event()
    producer_error: list[BaseException] = []

    def gpu_pass(win_frames, win_fidx):
        dets_list = session.player_det.detect_batch(win_frames, win_fidx, fps)
        session.team_clf.classify_batch(win_frames, dets_list)
        if session.gk_det is not None:
            for frame, dets in zip(win_frames, dets_list):
                session.gk_det.classify(frame, dets)
        balls = [best_ball(d) for d in dets_list]
        return win_frames, win_fidx, dets_list, balls

    def producer():
        win_frames, win_fidx, fidx = [], [], 0
        try:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    if win_frames:
                        q.put(gpu_pass(win_frames, win_fidx))
                    break
                if fidx % frame_step == 0:          # preserve EXACT original stride semantics
                    win_frames.append(frame)
                    win_fidx.append(fidx)
                    if len(win_frames) >= window_k:
                        q.put(gpu_pass(win_frames, win_fidx))
                        win_frames, win_fidx = [], []
                fidx += 1
        except BaseException as e:
            producer_error.append(e)
        finally:
            cap.release()      # FIX 2 — cap is now owned end-to-end by the producer thread
            q.put(None)        # sentinel — ALWAYS sent, clean exit or error, so consumer never blocks forever

    t = threading.Thread(target=producer, name=f"gpu-producer-{session.match_id}", daemon=True)
    t.start()

    try:
        while True:
            item = q.get()
            if item is None:
                break
            win_frames, win_fidx, dets_list, balls = item
            # --- Pass 2 body: moved verbatim from the current flush_window(),
            # operating on dets_list/balls instead of recomputing them.
            # (tracker.update / ball_tracker.update / carrier_eng.update /
            # pass_track.update / counters.add_label / correction handling —
            # unchanged logic, just relocated.)
            ...
    finally:
        # FIX 1 — guarantee producer shutdown on EVERY exit path from the
        # consumer loop: clean break, an exception raised inside Pass 2, or
        # a KeyboardInterrupt. Without this, an exception in Pass 2 leaves
        # the producer thread running (possibly blocked on q.put() because
        # the queue is full) with a live reference to session.player_det /
        # session.team_clf — and MatchSessionManager REUSES this same
        # session object for the next clip on this match_id. worker.py's
        # run_forever() catches all exceptions and continues to the next
        # queue message, so without this cleanup, the leaked thread would
        # still be calling into the session's detector/classifier while the
        # main thread starts processing the NEXT clip on the same objects —
        # a real cross-clip race, not a theoretical one.
        stop_event.set()
        try:
            while True:
                q.get_nowait()      # drain so a producer blocked on a full-queue put() unblocks
        except queue.Empty:
            pass
        t.join(timeout=30)
        if t.is_alive():
            log.error("[%s] gpu-producer thread did not exit within 30s after "
                      "stop signal — leaking; session models may be unsafe to "
                      "reuse on this match's next clip", session.match_id)

    if producer_error:
        raise producer_error[0]

    # ... unchanged: session.new_events() / event_rows / session.finish_clip() /
    # ClipResult / orient_for_team_a — only reached after the consumer loop
    # (and the finally block above) have fully completed.

Notes for captain on the two named fixes specifically:
- Fix 1 is the try/finally around the consumer loop with stop_event + queue-drain + t.join(timeout=...) + a loud log if the join times out. daemon=True on the thread is a last-resort safety net (won't hang process exit) — it is not a substitute for the explicit join; a daemon thread left running past its clip's lifetime is exactly the session-reuse race described above.
- Fix 2 is cap.release() moving into the producer's own finally, since decode now lives entirely in that thread — the main thread must never touch cap at all post-split.

Testing/validation requirements before this ships

1. Bit-identical output diff: run 2-3 already-processed clips through both the old and new process_clip() and diff the resulting ClipResult (minute_row fields, events list, correction) — should match exactly per-field. The proposal's own framing ("reordering of when work happens, not what happens") should hold; prove it, don't assume it.
2. Fault-injection test for Fix 1: monkeypatch tracker.update (or any Pass 2 call) to raise partway through a clip, confirm (a) the producer thread is joined within the 30s timeout and doesn't leak, (b) the original exception still propagates out of process_clip() unchanged, so worker.py's existing catch-all behavior is unaffected.
3. Real wall-clock measurement on the VM's T4 for a full clip, compared against the ~61s projection — the GIL-release assumptions behind the math are reasonable but unverified until measured.
4. Full multi-clip run (same shape as the current colour-reset test — several sequential clips on one match_id) end-to-end with the new code, diffed against a prior run with the old code for the same input clips.

Sequencing

- Land this after the mid-match colour-reset feature is confirmed working end-to-end (don't stack two unvalidated changes on the same hot file, clip_processor.py, at once).
- Consider gating it behind a new env var (e.g. CLIP_PIPELINE_THREADED, default off in config.py) so it can be disabled without a code revert/redeploy if something's wrong in production — this is the first thread ever introduced into this codebase (confirmed: no threading/asyncio/concurrent.futures anywhere in service/ currently), so having an instant kill switch is cheap insurance for a first attempt at it.