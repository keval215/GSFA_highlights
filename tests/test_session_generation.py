"""Unit tests for the mid-match team/GK colour-change reset logic added to
service/session.py: MatchSession.reset_for_new_generation, the fit_meta.json
sidecar (write/load), _resolve_team_names → team_a_cluster_id, and
MatchSessionManager.get_or_create's want_generation reset behavior.

Uses a fake ModelBundle (never loads a real YOLOv11m PlayerDetector) since
none of these tests call ensure_fit/process_clip — only session-state
bookkeeping, which is dependency-free of the actual CV models.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from rulesets import get_ruleset
from service import config, db
from service.session import MatchSession, MatchSessionManager


class _FakeModels:
    """Stands in for service.session.ModelBundle."""

    def player_detector(self, ruleset):
        return object()


class _NullConn:
    def close(self):
        pass


class _FakeClf:
    """Stands in for GSFATeamClassifier for _resolve_team_names tests."""

    def __init__(self, mapping):
        self._mapping = mapping

    def resolve_team_names(self, specs, crops):
        return self._mapping


@pytest.fixture(autouse=True)
def _match_state_dir(tmp_path, monkeypatch):
    # MatchSession.__init__ creates config.MATCH_STATE_DIR / match_id — point
    # it at a throwaway tmp dir instead of the real deployment path.
    monkeypatch.setattr(config, "MATCH_STATE_DIR", tmp_path)


@pytest.fixture
def ruleset():
    return get_ruleset("classic")


@pytest.fixture
def models():
    return _FakeModels()


def _session(models, ruleset, match_id="m1"):
    return MatchSession(match_id, models, ruleset)


# ---------------------------------------------------------------------------
# reset_for_new_generation
# ---------------------------------------------------------------------------

def test_reset_clears_fit_state_and_sets_generation(models, ruleset):
    sess = _session(models, ruleset)
    sess.team_clf = object()
    sess.fit_status = "ok"
    sess.gk_det = object()
    sess._gk_colour_invalid = True
    sess.team_a_cluster_id = 1
    sess._fit_crops_clip1 = [1, 2, 3]

    sess.reset_for_new_generation(2)

    assert sess.team_clf is None
    assert sess.fit_status == "pending"
    assert sess.gk_det is None
    assert sess._gk_colour_invalid is False
    assert sess.team_a_cluster_id is None
    assert sess._fit_crops_clip1 == []
    assert sess.fit_generation == 2


def test_reset_leaves_cross_clip_cv_state_untouched(models, ruleset):
    sess = _session(models, ruleset)
    tracker, ball_tracker, carrier_eng, pass_track = (
        sess.tracker, sess.ball_tracker, sess.carrier_eng, sess.pass_track
    )
    sess.proc_idx = 42
    sess.n_events_seen = 3
    sess.last_written = (1, 5)
    sess.last_written_team_a_cluster_id = 1
    sess.carryover_travel_frames = 7

    sess.reset_for_new_generation(2)

    assert sess.tracker is tracker
    assert sess.ball_tracker is ball_tracker
    assert sess.carrier_eng is carrier_eng
    assert sess.pass_track is pass_track
    assert sess.proc_idx == 42
    assert sess.n_events_seen == 3
    assert sess.last_written == (1, 5)
    assert sess.last_written_team_a_cluster_id == 1
    assert sess.carryover_travel_frames == 7


# ---------------------------------------------------------------------------
# finish_clip — last_written_team_a_cluster_id snapshot (code-review finding:
# a boundary PriorCorrection must be reoriented against the orientation the
# TARGET row was written under, not necessarily today's team_a_cluster_id —
# finish_clip is what records that snapshot for the next clip to read)
# ---------------------------------------------------------------------------

def test_finish_clip_snapshots_current_team_a_cluster_id(models, ruleset):
    sess = _session(models, ruleset)
    sess.team_a_cluster_id = 1

    sess.finish_clip(1, 3)

    assert sess.last_written == (1, 3)
    assert sess.last_written_team_a_cluster_id == 1


def test_finish_clip_snapshot_does_not_retroactively_change_on_later_resolution(models, ruleset):
    # Clip 1 finishes before colour resolution succeeds (cluster id still None) ...
    sess = _session(models, ruleset)
    sess.finish_clip(1, 1)
    assert sess.last_written_team_a_cluster_id is None

    # ... clip 2's combined refit resolves it to cluster 1 — that must NOT
    # rewrite the snapshot already taken for clip 1's row.
    sess.team_a_cluster_id = 1
    snapshot_before_clip_2_finishes = sess.last_written_team_a_cluster_id
    assert snapshot_before_clip_2_finishes is None


def test_reset_deletes_pkl_and_meta_sidecar(models, ruleset):
    sess = _session(models, ruleset)
    sess.fit_pkl_path.write_text("fake pkl")
    sess.fit_meta_path.write_text(json.dumps({"fit_generation": 1, "team_a_cluster_id": 0}))

    sess.reset_for_new_generation(2)

    assert not sess.fit_pkl_path.exists()
    assert not sess.fit_meta_path.exists()


def test_reset_is_safe_when_no_fit_files_exist(models, ruleset):
    sess = _session(models, ruleset)
    sess.reset_for_new_generation(2)   # no pkl/meta ever written — must not raise
    assert sess.fit_generation == 2


# ---------------------------------------------------------------------------
# fit_meta.json sidecar write / load
# ---------------------------------------------------------------------------

def test_write_fit_meta_round_trips(models, ruleset):
    sess = _session(models, ruleset)
    sess.fit_generation = 3
    sess.team_a_cluster_id = 1
    sess._write_fit_meta()

    meta = json.loads(sess.fit_meta_path.read_text())
    assert meta == {"fit_generation": 3, "team_a_cluster_id": 1}


def test_load_fit_meta_restores_state(models, ruleset):
    sess = _session(models, ruleset)
    sess.fit_meta_path.write_text(json.dumps({"fit_generation": 4, "team_a_cluster_id": 1}))
    sess.fit_generation, sess.team_a_cluster_id = 1, None

    sess._load_fit_meta()

    assert sess.fit_generation == 4
    assert sess.team_a_cluster_id == 1


def test_load_fit_meta_missing_file_degrades_to_defaults(models, ruleset):
    sess = _session(models, ruleset)
    assert not sess.fit_meta_path.exists()
    sess.fit_generation = 99

    sess._load_fit_meta()

    assert sess.fit_generation == 1
    assert sess.team_a_cluster_id is None


def test_load_fit_meta_corrupt_file_degrades_to_defaults(models, ruleset):
    sess = _session(models, ruleset)
    sess.fit_meta_path.write_text("not json")

    sess._load_fit_meta()

    assert sess.fit_generation == 1
    assert sess.team_a_cluster_id is None


# ---------------------------------------------------------------------------
# _resolve_team_names → team_a_cluster_id
# ---------------------------------------------------------------------------

def test_resolve_team_names_sets_team_a_cluster_id_straight(models, ruleset, monkeypatch):
    sess = _session(models, ruleset)
    monkeypatch.setattr(db, "get_conn", lambda: _NullConn())
    monkeypatch.setattr(db, "get_team_specs",
                        lambda conn, mid: [("Alpha", "#FF0000"), ("Bravo", "#0000FF")])
    clf = _FakeClf({0: "Alpha", 1: "Bravo"})

    sess._resolve_team_names(clf, crops=["dummy"])

    assert sess.team_a_cluster_id == 0


def test_resolve_team_names_sets_team_a_cluster_id_swapped(models, ruleset, monkeypatch):
    sess = _session(models, ruleset)
    monkeypatch.setattr(db, "get_conn", lambda: _NullConn())
    monkeypatch.setattr(db, "get_team_specs",
                        lambda conn, mid: [("Alpha", "#FF0000"), ("Bravo", "#0000FF")])
    clf = _FakeClf({0: "Bravo", 1: "Alpha"})   # KMeans landed the clusters swapped

    sess._resolve_team_names(clf, crops=["dummy"])

    assert sess.team_a_cluster_id == 1


def test_resolve_team_names_no_specs_leaves_cluster_id_none(models, ruleset, monkeypatch):
    sess = _session(models, ruleset)
    sess.team_a_cluster_id = 0   # prove it gets reset to None, not left stale
    monkeypatch.setattr(db, "get_conn", lambda: _NullConn())
    monkeypatch.setattr(db, "get_team_specs", lambda conn, mid: None)
    clf = _FakeClf({0: "Alpha", 1: "Bravo"})

    sess._resolve_team_names(clf, crops=["dummy"])

    assert sess.team_a_cluster_id is None


def test_resolve_team_names_failure_leaves_cluster_id_none(models, ruleset, monkeypatch):
    sess = _session(models, ruleset)
    monkeypatch.setattr(db, "get_conn", lambda: _NullConn())
    monkeypatch.setattr(db, "get_team_specs",
                        lambda conn, mid: [("Alpha", "#FF0000"), ("Bravo", "#0000FF")])

    class _FailingClf:
        def resolve_team_names(self, specs, crops):
            raise ValueError("boom")

    sess._resolve_team_names(_FailingClf(), crops=["dummy"])

    assert sess.team_a_cluster_id is None


# ---------------------------------------------------------------------------
# MatchSessionManager.get_or_create
# ---------------------------------------------------------------------------

def test_get_or_create_new_session_no_reset(models, ruleset):
    mgr = MatchSessionManager(models)
    sess, just_reset = mgr.get_or_create("m1", ruleset, want_generation=1)
    assert just_reset is False
    assert sess.fit_generation == 1


def test_get_or_create_returns_same_session_on_cache_hit(models, ruleset):
    mgr = MatchSessionManager(models)
    sess1, _ = mgr.get_or_create("m1", ruleset, want_generation=1)
    sess2, just_reset = mgr.get_or_create("m1", ruleset, want_generation=1)
    assert sess1 is sess2
    assert just_reset is False


def test_get_or_create_resets_in_place_when_want_generation_is_higher(models, ruleset):
    mgr = MatchSessionManager(models)
    sess, _ = mgr.get_or_create("m1", ruleset, want_generation=1)
    sess.team_clf = object()
    sess.fit_status = "ok"
    tracker = sess.tracker

    sess2, just_reset = mgr.get_or_create("m1", ruleset, want_generation=2)

    assert sess2 is sess            # same MatchSession object, reset in place
    assert just_reset is True
    assert sess2.team_clf is None
    assert sess2.fit_status == "pending"
    assert sess2.fit_generation == 2
    assert sess2.tracker is tracker  # cross-clip CV state untouched by the reset


def test_get_or_create_no_reset_when_want_generation_not_higher(models, ruleset):
    mgr = MatchSessionManager(models)
    sess, _ = mgr.get_or_create("m1", ruleset, want_generation=2)
    sess.team_clf = object()

    sess2, just_reset = mgr.get_or_create("m1", ruleset, want_generation=2)

    assert just_reset is False
    assert sess2.team_clf is not None


def test_get_or_create_want_generation_none_never_resets(models, ruleset):
    mgr = MatchSessionManager(models)
    sess, _ = mgr.get_or_create("m1", ruleset, want_generation=5)
    sess.team_clf = object()

    sess2, just_reset = mgr.get_or_create("m1", ruleset)   # want_generation defaults to None

    assert just_reset is False
    assert sess2.team_clf is not None
