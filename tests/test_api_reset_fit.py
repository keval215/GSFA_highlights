"""Unit tests for POST /api/matches/{match_id}/reset-fit (service/api.py).

Uses FastAPI's TestClient without entering it as a context manager, so the
app's `startup` event (which constructs real Azure blob/queue clients) never
fires — this endpoint touches neither, only service.db, which is
monkeypatched onto the sqlite stand-in from tests/conftest.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient

from service import api, db


@pytest.fixture
def client(monkeypatch, sqlite_conn_factory):
    monkeypatch.setattr(db, "get_conn", sqlite_conn_factory)
    return TestClient(api.app)


def test_reset_fit_bumps_generation_no_body(client, sqlite_conn_factory):
    conn = sqlite_conn_factory()
    db.ensure_match(conn, "m1", team_a_colour="#FF0000")

    resp = client.post("/api/matches/m1/reset-fit")

    assert resp.status_code == 200
    body = resp.json()
    assert body == {"match_id": "m1", "fit_generation": 2}


def test_reset_fit_with_body_overwrites_supplied_fields(client, sqlite_conn_factory):
    conn = sqlite_conn_factory()
    db.ensure_match(conn, "m1", team_a_name="Alpha", team_a_colour="#FF0000")

    resp = client.post("/api/matches/m1/reset-fit", json={
        "team_a_colour": "#00FF00",
        "team_a_name": "Charlie",
    })

    assert resp.status_code == 200
    assert resp.json()["fit_generation"] == 2

    cur = conn.cursor()
    cur.execute("SELECT team_a_name, team_a_colour FROM matches WHERE match_id = ?", "m1")
    assert tuple(cur.fetchone()) == ("Charlie", "#00FF00")


def test_reset_fit_unknown_match_is_404(client):
    resp = client.post("/api/matches/does-not-exist/reset-fit")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Blank-field validation (code-review finding: a supplied "" would otherwise
# silently overwrite a real stored colour via COALESCE(?, existing) and
# permanently break get_team_specs' `not all(row)` colour-presence check)
# ---------------------------------------------------------------------------

def test_reset_fit_rejects_blank_colour(client, sqlite_conn_factory):
    conn = sqlite_conn_factory()
    db.ensure_match(conn, "m1", team_a_colour="#FF0000")

    resp = client.post("/api/matches/m1/reset-fit", json={"team_a_colour": ""})

    assert resp.status_code == 422
    # The stored colour must be untouched — no partial write before the reject.
    cur = conn.cursor()
    cur.execute("SELECT team_a_colour, fit_generation FROM matches WHERE match_id = ?", "m1")
    assert tuple(cur.fetchone()) == ("#FF0000", 1)


def test_reset_fit_rejects_whitespace_only_name(client, sqlite_conn_factory):
    conn = sqlite_conn_factory()
    db.ensure_match(conn, "m1", team_a_name="Alpha")

    resp = client.post("/api/matches/m1/reset-fit", json={"team_a_name": "   "})

    assert resp.status_code == 422


def test_reset_fit_omitted_field_is_still_fine(client, sqlite_conn_factory):
    conn = sqlite_conn_factory()
    db.ensure_match(conn, "m1", team_a_name="Alpha")

    resp = client.post("/api/matches/m1/reset-fit", json={"team_b_name": "Bravo"})

    assert resp.status_code == 200


def test_reset_fit_repeated_calls_keep_incrementing(client, sqlite_conn_factory):
    conn = sqlite_conn_factory()
    db.ensure_match(conn, "m1")

    first = client.post("/api/matches/m1/reset-fit").json()["fit_generation"]
    second = client.post("/api/matches/m1/reset-fit").json()["fit_generation"]

    assert (first, second) == (2, 3)


def test_reset_fit_partial_body_leaves_other_fields_unset(client, sqlite_conn_factory):
    conn = sqlite_conn_factory()
    db.ensure_match(conn, "m1", team_a_name="Alpha", team_b_name="Bravo")

    client.post("/api/matches/m1/reset-fit", json={"team_a_name": "Charlie"})

    cur = conn.cursor()
    cur.execute("SELECT team_a_name, team_b_name FROM matches WHERE match_id = ?", "m1")
    assert tuple(cur.fetchone()) == ("Charlie", "Bravo")
