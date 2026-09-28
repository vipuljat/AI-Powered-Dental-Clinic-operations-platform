"""The one object-storage client abstraction (architecture.md §4's
"storage_uri + metadata row in Postgres, binary in object storage" rule) —
used for CSV export files, call recordings, dashboard export PDFs/CSVs.

Owns the environment pivot to an in-memory fake in test mode. Does not know
about any specific bucket "folder"/prefix convention beyond the ``key``
string callers pass in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from app.common.exceptions.errors import NotFoundError

if TYPE_CHECKING:
    from app.core.config import Settings


@runtime_checkable
class ObjectStorageClient(Protocol):
    """The interface every object-storage implementation (real or fake) satisfies."""

    async def put_object(self, key: str, data: bytes, content_type: str) -> str:
        """Store ``data`` under ``key`` and return the resulting ``storage_uri``."""
        ...

    async def get_object(self, key: str) -> bytes:
        """Return the bytes stored under ``key``.

        Raises ``app.common.exceptions.errors.NotFoundError`` if ``key`` does
        not exist — a missing object must surface as a 404, not a silently
        empty file (see e.g. GET /engagement/calls/{id} and any
        export-download path).
        """
        ...


class S3StorageClient:
    """boto3 S3-compatible client (MinIO dev / S3 prod, identical API per
    architecture.md §9).
    """

    def __init__(
        self,
        endpoint: str | None,
        bucket: str,
        access_key: str | None,
        secret_key: str | None,
    ) -> None:
        # Imported lazily so importing this module never requires boto3 to
        # reach out over the network, and so test-mode processes (which use
        # InMemoryStorageClient exclusively) never construct this client at
        # all.
        import boto3

        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    async def put_object(self, key: str, data: bytes, content_type: str) -> str:
        self._client.put_object(
            Bucket=self._bucket, Key=key, Body=data, ContentType=content_type
        )
        return f"s3://{self._bucket}/{key}"

    async def get_object(self, key: str) -> bytes:
        from botocore.exceptions import ClientError

        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code in ("NoSuchKey", "404"):
                raise NotFoundError(f"No object found for key '{key}'.") from exc
            raise
        return response["Body"].read()


class InMemoryStorageClient:
    """dict[str, bytes]-backed fake, used only when
    ``settings.environment == "test"`` per project_rules.testing.
    """

    _BUCKET_PLACEHOLDER = "test-bucket"

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    async def put_object(self, key: str, data: bytes, content_type: str) -> str:
        # content_type is intentionally not persisted alongside the bytes:
        # this fake only needs to round-trip `get_object`/`put_object`, and
        # no caller reads a stored content-type back through this interface.
        self._objects[key] = data
        return f"memory://{self._BUCKET_PLACEHOLDER}/{key}"

    async def get_object(self, key: str) -> bytes:
        try:
            return self._objects[key]
        except KeyError as exc:
            raise NotFoundError(f"No object found for key '{key}'.") from exc


def get_storage_client(settings: "Settings") -> ObjectStorageClient:
    """Factory selecting the storage client for the current environment.

    project_rules.testing: the object-storage client is switched to an
    in-memory dict-backed fake in test mode — a real boto3/S3 client is not
    constructed there because no MinIO/S3 endpoint is available in the test
    process.
    """

    if settings.environment == "test":
        return InMemoryStorageClient()

    return S3StorageClient(
        endpoint=settings.object_storage_endpoint,
        bucket=settings.object_storage_bucket,
        access_key=settings.object_storage_access_key,
        secret_key=settings.object_storage_secret_key,
    )
