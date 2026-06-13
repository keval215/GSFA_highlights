"""
service/blob.py — Azure Blob Storage wrapper.

api uploads incoming clips; the worker downloads them for processing and
deletes them after the SQL transaction commits. Blob name layout:

    <CLIPS_CONTAINER>/<match_id>/<half>_<minute>.mp4
"""

from __future__ import annotations

from pathlib import Path

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient

from service import config


def blob_name(match_id: str, half: int, minute: int) -> str:
    return f"{match_id}/{half}_{minute}.mp4"


class ClipBlobStore:
    def __init__(self, conn_str: str | None = None, container: str | None = None) -> None:
        self._service   = BlobServiceClient.from_connection_string(conn_str or config.storage_conn_str())
        self._container = container or config.CLIPS_CONTAINER
        self._client    = self._service.get_container_client(self._container)

    def ensure_container(self) -> None:
        try:
            self._client.create_container()
        except ResourceExistsError:
            pass

    def exists(self, name: str) -> bool:
        return self._client.get_blob_client(name).exists()

    def upload_stream(self, name: str, stream, overwrite: bool = False) -> str:
        """Upload a file-like object. Returns the blob path (container-relative)."""
        self._client.get_blob_client(name).upload_blob(stream, overwrite=overwrite)
        return name

    def download_to(self, name: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            self._client.get_blob_client(name).download_blob().readinto(f)
        return dest

    def delete(self, name: str) -> None:
        try:
            self._client.get_blob_client(name).delete_blob()
        except Exception:
            # Best-effort cleanup; an orphaned clip blob is harmless and
            # idempotent re-processing tolerates a missing one.
            pass
