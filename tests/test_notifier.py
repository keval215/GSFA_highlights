"""Notifier behavior: strict in-order delivery, exponential backoff on
failure, exhaustion ⇒ failed + continue with the next row.

requests/pyodbc are mocked, so these run anywhere; if the service deps
(requests) are missing the module is skipped.
"""

import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("requests")
pytest.importorskip("pyodbc", reason="pyodbc needed to import service.db")

from service import notifier
from service.db import OutboxRow


def _rows():
    return [
        OutboxRow(outbox_id=1, match_id="m", half=1, minute=1, payload={"minute": 1}, attempts=0),
        OutboxRow(outbox_id=2, match_id="m", half=1, minute=2, payload={"minute": 2}, attempts=0),
        OutboxRow(outbox_id=3, match_id="m", half=2, minute=1, payload={"minute": 1}, attempts=0),
    ]


def _resp(code):
    r = mock.Mock()
    r.status_code = code
    return r


@mock.patch("service.notifier.time.sleep")
@mock.patch("service.notifier.requests.post")
def test_in_order_delivery(post, _sleep, monkeypatch):
    monkeypatch.setenv("CALLBACK_URL", "http://cb.example/hook")
    post.return_value = _resp(200)
    sent, failed = [], []
    with mock.patch.object(notifier.db, "fetch_pending", return_value=_rows()), \
         mock.patch.object(notifier.db, "mark_sent", side_effect=lambda c, oid, a: sent.append(oid)), \
         mock.patch.object(notifier.db, "mark_failed", side_effect=lambda c, oid, a: failed.append(oid)):
        notifier.send_pending_for_match(mock.Mock(), "m")

    assert sent == [1, 2, 3]            # strict (half, minute) order
    assert failed == []
    payloads = [c.kwargs["json"] for c in post.call_args_list]
    assert [p["minute"] for p in payloads] == [1, 2, 1]


@mock.patch("service.notifier.time.sleep")
@mock.patch("service.notifier.requests.post")
def test_exhaustion_marks_failed_and_continues(post, sleep, monkeypatch):
    monkeypatch.setenv("CALLBACK_URL", "http://cb.example/hook")
    # First row always 500s; second row succeeds.
    post.side_effect = [_resp(500), _resp(500), _resp(500), _resp(200), _resp(200), _resp(200)]
    sent, failed = [], []
    rows = _rows()
    with mock.patch.object(notifier.db, "fetch_pending", return_value=rows), \
         mock.patch.object(notifier.db, "mark_sent", side_effect=lambda c, oid, a: sent.append(oid)), \
         mock.patch.object(notifier.db, "mark_failed", side_effect=lambda c, oid, a: failed.append(oid)):
        notifier.send_pending_for_match(mock.Mock(), "m")

    assert failed == [1]                # exhausted after 3 attempts
    assert sent == [2, 3]               # later rows still delivered
    # Exponential backoff between the 3 attempts of row 1: 1s then 2s.
    waits = [c.args[0] for c in sleep.call_args_list[:2]]
    assert waits == [1.0, 2.0]


@mock.patch("service.notifier.requests.post")
def test_no_callback_url_leaves_rows_pending(post, monkeypatch):
    monkeypatch.delenv("CALLBACK_URL", raising=False)
    with mock.patch.object(notifier.db, "fetch_pending") as fetch:
        notifier.send_pending_for_match(mock.Mock(), "m")
    fetch.assert_not_called()
    post.assert_not_called()
