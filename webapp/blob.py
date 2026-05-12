"""
blob.py — Azure Blob upload + SAS URL generation for delivering finished
highlight reels to the user.

Requires env var AZURE_STORAGE_CONNECTION_STRING (set on the VM).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from azure.storage.blob import (
    BlobSasPermissions,
    BlobServiceClient,
    ContentSettings,
    generate_blob_sas,
)


def _connection_string() -> str:
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        raise RuntimeError(
            "AZURE_STORAGE_CONNECTION_STRING is not set; "
            "set it in the VM's env file before starting the service."
        )
    return conn


def _container_name() -> str:
    return os.environ.get("BLOB_CONTAINER", "highlights")


def upload_and_sas(local_path: Path, blob_name: str) -> str:
    """
    Upload `local_path` to the configured Blob container and return a
    time-limited SAS download URL that triggers Save As in the browser.

    Container name defaults to env BLOB_CONTAINER (default 'highlights').
    SAS expiry defaults to env BLOB_SAS_DAYS (default 7).
    """
    container = _container_name()
    sas_days = int(os.environ.get("BLOB_SAS_DAYS", "7"))

    svc = BlobServiceClient.from_connection_string(_connection_string())

    try:
        svc.create_container(container)
    except Exception:
        pass

    blob = svc.get_blob_client(container=container, blob=blob_name)
    with open(local_path, "rb") as f:
        blob.upload_blob(
            f,
            overwrite=True,
            content_settings=ContentSettings(content_type="video/mp4"),
        )

    sas = generate_blob_sas(
        account_name=svc.account_name,
        container_name=container,
        blob_name=blob_name,
        account_key=svc.credential.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(timezone.utc) + timedelta(days=sas_days),
        content_disposition=f'attachment; filename="{blob_name}"',
    )
    return f"{blob.url}?{sas}"


def delete_blob(blob_name: str) -> bool:
    """
    Delete `blob_name` from the configured container. Returns True if the
    blob was removed (or was already gone), False on any other error.
    """
    container = _container_name()
    try:
        svc = BlobServiceClient.from_connection_string(_connection_string())
        blob = svc.get_blob_client(container=container, blob=blob_name)
        blob.delete_blob(delete_snapshots="include")
        return True
    except Exception as e:
        msg = str(e).lower()
        if "blobnotfound" in msg or "the specified blob does not exist" in msg:
            return True
        return False
