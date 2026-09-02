"""
service/logging_setup.py — one logging configuration for both entry points.

Both the api (service/api.py) and the worker (service/worker.py) want the same
three things:
  1. Timestamps in IST (the team reads logs in Indian time, not the container's UTC).
  2. Noisy third-party HTTP loggers silenced (the Azure SDK logs every
     Blob/Queue request+response at INFO; urllib3 logs every connection at
     DEBUG) so they don't flood the log.
  3. Our own gsfa.* lines at LOG_LEVEL (default INFO).

Call configure() once at startup (api: at import; worker: in main()).
"""

from __future__ import annotations

import logging
import os
import time

# IST is a fixed UTC+05:30 with no daylight-saving, so a constant offset is always
# correct. We render IST by offsetting the epoch seconds and formatting as gmtime —
# this keeps it independent of the container's TZ / tzdata.
_IST_OFFSET_SECONDS = 5 * 3600 + 30 * 60  # 19800


def _ist_converter(secs: float | None = None) -> time.struct_time:
    return time.gmtime((secs if secs is not None else time.time()) + _IST_OFFSET_SECONDS)


_FORMAT = "%(asctime)s IST %(name)s %(levelname)s %(message)s"


def configure(component: str) -> None:
    """Configure root logging for `component` ("api" | "worker").

    Idempotent enough for our use: api calls it at import, worker in main().
    """
    # IST timestamps for every Formatter that uses %(asctime)s (ours, azure, uvicorn).
    # staticmethod() so the function isn't bound as a method when accessed via an
    # instance (logging calls self.converter(record.created) with one arg); a plain
    # function would receive `self` too and raise. The stdlib default (time.localtime)
    # sidesteps this only because it's a C builtin.
    logging.Formatter.converter = staticmethod(_ist_converter)

    level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, format=_FORMAT)

    # The Azure SDK's HTTP logging policy logs every Blob/Queue request+response at
    # INFO, which floods the log (especially the api on each clip upload). Cap it.
    logging.getLogger("azure").setLevel(logging.WARNING)

    # urllib3 logs every connection + retry at DEBUG; with LOG_LEVEL=DEBUG this
    # floods the log during callbacks / blob I/O. Keep it at WARNING regardless.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
