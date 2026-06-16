"""
service/notifier.py — outbox sender.

Runs in the worker process after each clip's SQL transaction commits.
Sends pending callback_outbox rows for the match strictly in
(half, minute) order via HTTP POST to the tournament-duelz advance-stats
endpoint (CALLBACK_URL base + /v1/pvt/tournament-duelz/{match_id}/advance-stats),
authenticated with the X-Super-Admin-Key header. Each POST is a full overwrite
of the duel's advance_stats subdocument; the server replies 204.

Retry: CALLBACK_RETRIES attempts with exponential backoff
(CALLBACK_BACKOFF_BASE * 2^n seconds). On exhaustion the row is marked
'failed', the failure is logged loudly, and the NEXT row is still sent —
SQL remains the source of truth, the callback is best-effort. A 401
(bad/missing super-admin key) fails fast without burning retries.

If CALLBACK_URL or SUPER_ADMIN_KEY is unset, rows simply stay 'pending'
(nothing is lost; they will be sent once both are configured and the
notifier runs again).
"""

from __future__ import annotations

import json
import logging
import time

import requests

from service import config, db

log = logging.getLogger("gsfa.notifier")

_TIMEOUT_S = 10


def send_pending_for_match(conn, match_id: str) -> None:
    # match_id is the tournament-duel ObjectID → drives the per-duel URL.
    url = config.advance_stats_url(match_id)
    key = config.super_admin_key()
    if not url or not key:
        log.warning(
            "CALLBACK_URL/SUPER_ADMIN_KEY not set — outbox rows for %s stay pending",
            match_id,
        )
        return

    headers = {"X-Super-Admin-Key": key}
    for row in db.fetch_pending(conn, match_id):
        # Full body at DEBUG (real cumulative numbers); concise confirmation at
        # INFO only once the send succeeds (below).
        log.debug("advance-stats %s h%d m%d → POST %s body=%s",
                  match_id, row.half, row.minute, url, json.dumps(row.payload))
        attempts = row.attempts
        sent = False
        for i in range(config.CALLBACK_RETRIES):
            attempts += 1
            try:
                resp = requests.post(url, json=row.payload, headers=headers,
                                     timeout=_TIMEOUT_S)
                if 200 <= resp.status_code < 300:
                    sent = True
                    break
                log.warning(
                    "callback %s h%d m%d attempt %d → HTTP %d",
                    match_id, row.half, row.minute, attempts, resp.status_code,
                )
                # A bad/missing super-admin key won't fix itself on retry.
                if resp.status_code == 401:
                    break
            except requests.RequestException as exc:
                log.warning(
                    "callback %s h%d m%d attempt %d → %s",
                    match_id, row.half, row.minute, attempts, exc,
                )
            if i < config.CALLBACK_RETRIES - 1:
                time.sleep(config.CALLBACK_BACKOFF_BASE * (2 ** i))

        if sent:
            log.info("advance-stats %s h%d m%d sent", match_id, row.half, row.minute)
            db.mark_sent(conn, row.outbox_id, attempts)
        else:
            db.mark_failed(conn, row.outbox_id, attempts)
            log.error(
                "CALLBACK FAILED PERMANENTLY match=%s half=%d minute=%d after %d attempts "
                "— marked failed, continuing (stats remain in SQL)",
                match_id, row.half, row.minute, attempts,
            )
