"""
service/queueing.py — Azure Queue Storage wrapper.

One message per clip:
    {"match_id": str, "half": int, "minute": int, "blob_path": str,
     "clip_duration_seconds": float}

Azure Queue Storage is approximately FIFO, so the worker enforces ordering
itself (see worker.py); this module only provides enqueue/dequeue/poison
mechanics. dequeue_count > MAX_DEQUEUE_COUNT moves the message to the
poison queue.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from azure.core.exceptions import ResourceExistsError
from azure.storage.queue import QueueClient

from service import config


@dataclass
class ClipMessage:
    match_id:  str
    half:      int
    minute:    int
    blob_path: str
    clip_duration_seconds: float = 60.0
    # How many times the worker deferred this message because it arrived
    # out of order (carried in the message body across re-sends).
    ordering_retries: int = 0
    # Queue bookkeeping needed for delete/update:
    message_id:    str = ""
    pop_receipt:   str = ""
    dequeue_count: int = 0


class ClipQueue:
    def __init__(self, conn_str: str | None = None) -> None:
        conn = conn_str or config.storage_conn_str()
        self._queue  = QueueClient.from_connection_string(conn, config.QUEUE_NAME)
        self._poison = QueueClient.from_connection_string(conn, config.POISON_QUEUE_NAME)

    def ensure_queues(self) -> None:
        for q in (self._queue, self._poison):
            try:
                q.create_queue()
            except ResourceExistsError:
                pass

    # --- producer (api) ---

    def enqueue(
        self,
        match_id: str,
        half: int,
        minute: int,
        blob_path: str,
        clip_duration_seconds: float,
    ) -> None:
        self._queue.send_message(json.dumps({
            "match_id": match_id,
            "half": half,
            "minute": minute,
            "blob_path": blob_path,
            "clip_duration_seconds": clip_duration_seconds,
        }))

    # --- consumer (worker) ---

    def dequeue(self, visibility_timeout: int | None = None) -> Optional[ClipMessage]:
        """Pop one message or None. The message stays invisible for
        visibility_timeout seconds; call delete() after successful processing."""
        msgs = self._queue.receive_messages(
            max_messages=1,
            visibility_timeout=visibility_timeout or config.QUEUE_VISIBILITY_SEC,
        )
        for m in msgs:
            try:
                body = json.loads(m.content)
            except (json.JSONDecodeError, TypeError):
                self._move_to_poison_raw(m.content or "")
                self._queue.delete_message(m)
                continue
            return ClipMessage(
                match_id         = body["match_id"],
                half             = int(body["half"]),
                minute           = int(body["minute"]),
                blob_path        = body["blob_path"],
                clip_duration_seconds = float(body.get("clip_duration_seconds", 60.0)),
                ordering_retries = int(body.get("ordering_retries", 0)),
                message_id       = m.id,
                pop_receipt      = m.pop_receipt,
                dequeue_count    = int(m.dequeue_count or 0),
            )
        return None

    def delete(self, msg: ClipMessage) -> None:
        self._queue.delete_message(msg.message_id, msg.pop_receipt)

    def defer(self, msg: ClipMessage, delay_seconds: int) -> None:
        """Out-of-order message: delete + re-send hidden for delay_seconds,
        with ordering_retries bumped in the body (dequeue_count resets, so
        poison counting stays reserved for actual processing crashes)."""
        self._queue.delete_message(msg.message_id, msg.pop_receipt)
        self._queue.send_message(
            json.dumps({
                "match_id": msg.match_id, "half": msg.half,
                "minute": msg.minute, "blob_path": msg.blob_path,
                "clip_duration_seconds": msg.clip_duration_seconds,
                "ordering_retries": msg.ordering_retries + 1,
            }),
            visibility_timeout=delay_seconds,
        )

    def move_to_poison(self, msg: ClipMessage) -> None:
        self._poison.send_message(json.dumps({
            "match_id": msg.match_id, "half": msg.half,
            "minute": msg.minute, "blob_path": msg.blob_path,
            "clip_duration_seconds": msg.clip_duration_seconds,
        }))
        self._queue.delete_message(msg.message_id, msg.pop_receipt)

    def _move_to_poison_raw(self, content: str) -> None:
        self._poison.send_message(content)

    # --- metrics ---

    def depths(self) -> tuple[int, int]:
        """(main queue depth, poison queue depth) — approximate counts."""
        main   = self._queue.get_queue_properties().approximate_message_count or 0
        poison = self._poison.get_queue_properties().approximate_message_count or 0
        return int(main), int(poison)
