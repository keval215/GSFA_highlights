"""Tests for the optional producer/consumer threading in
service/clip_processor.process_clip (config.CLIP_PIPELINE_THREADED).

All fakes — no GPU, no real model, no real video file (cv2.VideoCapture is
monkeypatched). Covers:
  * serial path creates no thread when the knob is off;
  * threaded and serial paths produce an identical ClipResult;
  * a Pass 2 exception on the threaded path propagates out of process_clip
    unchanged AND leaves no producer thread alive (Fix 1 + Fix 2).
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from service import clip_processor, config, stats


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

_FRAME = object()          # opaque — no fake below inspects frame content
_LABELS = (stats.LBL_TEAM_A, stats.LBL_TEAM_B, stats.LBL_LOOSE, stats.LBL_OOF)


class _FakeDet:
    """Stands in for one FrameDetections. `.players` for Pass 2, `.balls`
    for modules.possession.best_ball (empty → best_ball returns None)."""
    players: list = []
    balls: list = []


class _FakeCap:
    """Minimal cv2.VideoCapture stand-in: yields `n_raw` frames then EOF."""

    def __init__(self, n_raw: int) -> None:
        self._n = n_raw
        self._i = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, _prop) -> float:
        return 30.0

    def read(self):
        if self._i >= self._n:
            return False, None
        self._i += 1
        return True, _FRAME

    def release(self) -> None:
        self.released = True


class _FakePlayerDet:
    def detect_batch(self, frames, fidxs, fps):
        return [_FakeDet() for _ in frames]


class _FakeTeamClf:
    def classify_batch(self, frames, dets_list):
        return None


class _FakeTracker:
    def update(self, frame, players):
        return None


class _FakeBallTracker:
    def update(self, raw_ball):
        return None, None


class _FakeCarrierEng:
    def update(self, players, ball_state, ball):
        return None


class _FakePassTrack:
    """Deterministic label per processed frame; optionally raises on the
    Nth update() call to simulate a mid-clip Pass 2 failure."""

    def __init__(self, raise_on_call: int | None = None, exc: BaseException | None = None):
        self.calls = 0
        self._raise_on_call = raise_on_call
        self._exc = exc

    def update(self, carrier, proc_idx):
        self.calls += 1
        if self._raise_on_call is not None and self.calls == self._raise_on_call:
            raise self._exc
        return _LABELS[proc_idx % len(_LABELS)], []


class _FakeSession:
    def __init__(self, pass_track: _FakePassTrack | None = None) -> None:
        self.match_id = "m_thread_test"
        self.team_clf = _FakeTeamClf()
        self.gk_det = None
        self.player_det = _FakePlayerDet()
        self.tracker = _FakeTracker()
        self.ball_tracker = _FakeBallTracker()
        self.carrier_eng = _FakeCarrierEng()
        self.pass_track = pass_track or _FakePassTrack()
        self.proc_idx = 0
        self.last_written = None
        self.last_written_team_a_cluster_id = None
        self.team_a_cluster_id = None          # orient_for_team_a → no-op
        self.finished = None

    def ensure_gk_ready(self) -> None:
        pass

    def split_adjustment(self, n):            # unused (fakes emit no adjustments)
        return 0, n

    def new_events(self):
        return []

    def finish_clip(self, half, minute):
        self.finished = (half, minute)


class _InjectedError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_cap(monkeypatch):
    """Patch cv2.VideoCapture in clip_processor to hand out a _FakeCap.
    Returns a dict so a test can read back the last capture created."""
    holder: dict = {}

    def _factory(_path):
        cap = _FakeCap(n_raw=132)            # step 2 → 66 processed → 4 full windows + tail
        holder["cap"] = cap
        return cap

    monkeypatch.setattr(clip_processor.cv2, "VideoCapture", _factory)
    return holder


def _run(session) -> "stats.ClipResult":
    return clip_processor.process_clip(
        session, "unused.mp4", half=1, minute=3, clip_duration_seconds=60.0,
        clip_blob_path="blob/x.mp4",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_serial_path_creates_no_thread(fake_cap, monkeypatch):
    monkeypatch.setattr(config, "CLIP_PIPELINE_THREADED", False)

    def _no_thread(*a, **k):
        raise AssertionError("serial path must not create a thread")

    monkeypatch.setattr(clip_processor.threading, "Thread", _no_thread)

    result = _run(_FakeSession())

    assert result.minute_row.match_id == "m_thread_test"
    # 66 processed frames, labels cycle team_a/team_b/loose/oof → 17/17/16/16
    row = result.minute_row
    total = (row.frames_team_a + row.frames_team_b + row.frames_loose + row.frames_oof)
    assert total == 66
    assert fake_cap["cap"].released is True


def test_threaded_and_serial_produce_identical_result(monkeypatch):
    # Two independent runs over the same synthetic clip — one serial, one
    # threaded — must yield an equal ClipResult (pure reordering of work).
    def _cap_factory(_path):
        return _FakeCap(n_raw=132)

    monkeypatch.setattr(clip_processor.cv2, "VideoCapture", _cap_factory)

    n_threads_before = threading.active_count()

    monkeypatch.setattr(config, "CLIP_PIPELINE_THREADED", False)
    serial = _run(_FakeSession())

    monkeypatch.setattr(config, "CLIP_PIPELINE_THREADED", True)
    threaded = _run(_FakeSession())

    assert threaded == serial
    assert threading.active_count() == n_threads_before  # producer joined
    assert not any(th.name.startswith("clip-producer-") and th.is_alive()
                   for th in threading.enumerate())


def test_threaded_pass2_exception_propagates_and_joins_producer(fake_cap, monkeypatch):
    monkeypatch.setattr(config, "CLIP_PIPELINE_THREADED", True)

    boom = _InjectedError("boom in pass 2")
    session = _FakeSession(_FakePassTrack(raise_on_call=5, exc=boom))

    n_threads_before = threading.active_count()

    with pytest.raises(_InjectedError) as ei:
        _run(session)

    # (b) the injected exception surfaces unchanged
    assert ei.value is boom
    assert "boom in pass 2" in str(ei.value)

    # (a) no producer thread left alive
    assert not any(
        th.name.startswith("clip-producer-") and th.is_alive()
        for th in threading.enumerate()
    )
    assert threading.active_count() == n_threads_before

    # Fix 2: the capture was released (by the producer's own finally)
    assert fake_cap["cap"].released is True


def test_threaded_shutdown_joins_a_still_running_producer(monkeypatch):
    """Regression guard for the shutdown drain: Pass 1 is slow enough that the
    producer is still executing (queue full, blocked on put or mid-detect) when
    Pass 2 raises. The consumer's `finally` must keep draining until the
    producer unblocks, reaches its stop check and exits — no leaked thread."""
    monkeypatch.setattr(config, "CLIP_PIPELINE_THREADED", True)

    def _cap_factory(_path):
        return _FakeCap(n_raw=132)

    monkeypatch.setattr(clip_processor.cv2, "VideoCapture", _cap_factory)

    release = threading.Event()

    class _SlowPlayerDet:
        def __init__(self):
            self.calls = 0

        def detect_batch(self, frames, fidxs, fps):
            self.calls += 1
            if self.calls >= 2:            # windows 2+ stall → queue fills up
                release.wait(timeout=3.0)
            return [_FakeDet() for _ in frames]

    boom = _InjectedError("boom while producer busy")
    session = _FakeSession(_FakePassTrack(raise_on_call=3, exc=boom))
    session.player_det = _SlowPlayerDet()

    n_before = threading.active_count()
    try:
        with pytest.raises(_InjectedError, match="boom while producer busy"):
            _run(session)
    finally:
        release.set()

    assert not any(th.name.startswith("clip-producer-") and th.is_alive()
                   for th in threading.enumerate())
    assert threading.active_count() == n_before


def test_threaded_producer_exception_propagates(fake_cap, monkeypatch):
    # An exception raised in Pass 1 (producer thread) must be captured and
    # re-raised on the main thread with its original type.
    monkeypatch.setattr(config, "CLIP_PIPELINE_THREADED", True)

    session = _FakeSession()

    def _boom_detect(frames, fidxs, fps):
        raise _InjectedError("boom in pass 1")

    monkeypatch.setattr(session.player_det, "detect_batch", _boom_detect)

    with pytest.raises(_InjectedError, match="boom in pass 1"):
        _run(session)

    assert not any(
        th.name.startswith("clip-producer-") and th.is_alive()
        for th in threading.enumerate()
    )
    assert fake_cap["cap"].released is True
