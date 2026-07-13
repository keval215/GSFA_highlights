"""
service/queueing.py — Azure Queue Storage wrapper.

One message per work item:
    clip:
        {"kind": "clip", "match_id": str, "half": int, "minute": int,
         "blob_path": str, "clip_duration_seconds": float}
    post-processing:
        {"kind": "post_processing", "match_id": str, "blob_path": str,
         "team0_name": str | null, "team1_name": str | null,
         "team0_colour": str | null, "team1_colour": str | null,
         "team0_gk_colour": str | null, "team1_gk_colour": str | null}

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
    blob_path: str
    kind: str = "clip"
    half:      int = 0
    minute:    int = 0
    clip_duration_seconds: float = 60.0
    team0_name: Optional[str] = None
    team1_name: Optional[str] = None
    team0_colour: Optional[str] = None
    team1_colour: Optional[str] = None
    team0_gk_colour: Optional[str] = None
    team1_gk_colour: Optional[str] = None
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
        kind: str = "clip",
        team0_name: Optional[str] = None,
        team1_name: Optional[str] = None,
        team0_colour: Optional[str] = None,
        team1_colour: Optional[str] = None,
    ) -> None:
        body = {
            "kind": kind,
            "match_id": match_id,
            "half": half,
            "minute": minute,
            "blob_path": blob_path,
            "clip_duration_seconds": clip_duration_seconds,
        }
        if team0_name is not None:
            body["team0_name"] = team0_name
        if team1_name is not None:
            body["team1_name"] = team1_name
        if team0_colour is not None:
            body["team0_colour"] = team0_colour
        if team1_colour is not None:
            body["team1_colour"] = team1_colour
        self._queue.send_message(json.dumps(body))

    def enqueue_post_processing(
        self,
        match_id: str,
        blob_path: str,
        team0_name: Optional[str] = None,
        team1_name: Optional[str] = None,
        team0_colour: Optional[str] = None,
        team1_colour: Optional[str] = None,
        team0_gk_colour: Optional[str] = None,
        team1_gk_colour: Optional[str] = None,
    ) -> None:
        body = {
            "kind": "post_processing",
            "match_id": match_id,
            "blob_path": blob_path,
        }
        if team0_name is not None:
            body["team0_name"] = team0_name
        if team1_name is not None:
            body["team1_name"] = team1_name
        if team0_colour is not None:
            body["team0_colour"] = team0_colour
        if team1_colour is not None:
            body["team1_colour"] = team1_colour
        if team0_gk_colour is not None:
            body["team0_gk_colour"] = team0_gk_colour
        if team1_gk_colour is not None:
            body["team1_gk_colour"] = team1_gk_colour
        self._queue.send_message(json.dumps(body))

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
                blob_path        = body["blob_path"],
                kind             = str(body.get("kind", "clip")),
                half             = int(body.get("half", 0)),
                minute           = int(body.get("minute", 0)),
                clip_duration_seconds = float(body.get("clip_duration_seconds", 60.0)),
                team0_name       = body.get("team0_name"),
                team1_name       = body.get("team1_name"),
                team0_colour     = body.get("team0_colour"),
                team1_colour     = body.get("team1_colour"),
                team0_gk_colour  = body.get("team0_gk_colour"),
                team1_gk_colour  = body.get("team1_gk_colour"),
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
                "kind": msg.kind,
                "match_id": msg.match_id, "half": msg.half,
                "minute": msg.minute, "blob_path": msg.blob_path,
                "clip_duration_seconds": msg.clip_duration_seconds,
                "team0_name": msg.team0_name,
                "team1_name": msg.team1_name,
                "team0_colour": msg.team0_colour,
                "team1_colour": msg.team1_colour,
                "team0_gk_colour": msg.team0_gk_colour,
                "team1_gk_colour": msg.team1_gk_colour,
                "ordering_retries": msg.ordering_retries + 1,
            }),
            visibility_timeout=delay_seconds,
        )

    def move_to_poison(self, msg: ClipMessage) -> None:
        self._poison.send_message(json.dumps({
            "kind": msg.kind,
            "match_id": msg.match_id, "half": msg.half,
            "minute": msg.minute, "blob_path": msg.blob_path,
            "clip_duration_seconds": msg.clip_duration_seconds,
            "team0_name": msg.team0_name,
            "team1_name": msg.team1_name,
            "team0_colour": msg.team0_colour,
            "team1_colour": msg.team1_colour,
            "team0_gk_colour": msg.team0_gk_colour,
            "team1_gk_colour": msg.team1_gk_colour,
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
