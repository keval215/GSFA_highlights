"""
service/notifier.py — outbox sender.

Runs in the worker process after each clip's SQL transaction commits.
Sends pending callback_outbox rows for the match strictly in
(half, minute) order via HTTP POST to CALLBACK_URL.

Retry: CALLBACK_RETRIES attempts with exponential backoff
(CALLBACK_BACKOFF_BASE * 2^n seconds). On exhaustion the row is marked
'failed', the failure is logged loudly, and the NEXT row is still sent —
SQL remains the source of truth, the callback is best-effort.

If CALLBACK_URL is unset, rows simply stay 'pending' (nothing is lost;
they will be sent once the URL is configured and the notifier runs again).
"""

from __future__ import annotations

import logging
import time

import requests

from service import config, db

log = logging.getLogger("gsfa.notifier")

_TIMEOUT_S = 10


def send_pending_for_match(conn, match_id: str) -> None:
    url = config.callback_url()
    if not url:
        log.warning("CALLBACK_URL not set — outbox rows for %s stay pending", match_id)
        return

    for row in db.fetch_pending(conn, match_id):
        attempts = row.attempts
        sent = False
        for i in range(config.CALLBACK_RETRIES):
            attempts += 1
            try:
                resp = requests.post(url, json=row.payload, timeout=_TIMEOUT_S)
                if 200 <= resp.status_code < 300:
                    sent = True
                    break
                log.warning(
                    "callback %s h%d m%d attempt %d → HTTP %d",
                    match_id, row.half, row.minute, attempts, resp.status_code,
                )
            except requests.RequestException as exc:
                log.warning(
                    "callback %s h%d m%d attempt %d → %s",
                    match_id, row.half, row.minute, attempts, exc,
                )
            if i < config.CALLBACK_RETRIES - 1:
                time.sleep(config.CALLBACK_BACKOFF_BASE * (2 ** i))

        if sent:
            db.mark_sent(conn, row.outbox_id, attempts)
        else:
            db.mark_failed(conn, row.outbox_id, attempts)
            log.error(
                "CALLBACK FAILED PERMANENTLY match=%s half=%d minute=%d after %d attempts "
                "— marked failed, continuing (stats remain in SQL)",
                match_id, row.half, row.minute, attempts,
            )
