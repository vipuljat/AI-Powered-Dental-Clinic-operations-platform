"""Unit tests for app/core/storage.py.

Covers `InMemoryStorageClient` (the test-mode fake per project_rules.testing),
`get_storage_client`'s environment-driven dispatch between it and
`S3StorageClient`, and the documented `s3://{bucket}/{key}` / `NotFoundError`
contracts of the `ObjectStorageClient` protocol.

Two techniques borrow this codebase's existing pattern (see
tests/unit/app/core/test_db_types.py's pgvector shim) of substituting an
external dependency at its own module boundary rather than reaching into an
undocumented private attribute of the class under test:

- `get_storage_client`'s non-test branch is verified by monkeypatching the
  module-level `S3StorageClient` name `app.core.storage` itself resolves at
  call time, rather than constructing a real boto3/S3 client (whose
  constructor args -- which `Settings` fields feed `endpoint`/`bucket`/
  `access_key`/`secret_key` -- this file's spec does not name).
- `S3StorageClient.put_object`'s return-value contract is verified by
  monkeypatching `boto3.client` (the actual external boundary) to return an
  in-process fake, so no real network call is made.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import boto3
import pytest

import app.core.storage as storage_module
from app.common.exceptions.errors import NotFoundError
from app.core.storage import InMemoryStorageClient, S3StorageClient, get_storage_client


# ---------------------------------------------------------------------------
# InMemoryStorageClient
# ---------------------------------------------------------------------------

async def test_in_memory_client_put_then_get_round_trips_bytes():
    client = InMemoryStorageClient()

    await client.put_object("exports/report.csv", b"a,b,c\n1,2,3", "text/csv")
    data = await client.get_object("exports/report.csv")

    assert data == b"a,b,c\n1,2,3"


async def test_in_memory_client_put_object_returns_memory_uri_containing_the_key():
    client = InMemoryStorageClient()

    uri = await client.put_object("call-recordings/abc.wav", b"\x00\x01", "audio/wav")

    assert uri.startswith("memory://")
    assert "call-recordings/abc.wav" in uri


async def test_in_memory_client_get_object_raises_not_found_for_missing_key():
    client = InMemoryStorageClient()

    with pytest.raises(NotFoundError):
        await client.get_object("does/not/exist.csv")


async def test_in_memory_client_is_isolated_per_instance():
    # Watch out: a missing object must surface as NotFoundError, not an
    # empty/default byte string leaking across unrelated client instances.
    first_client = InMemoryStorageClient()
    second_client = InMemoryStorageClient()

    await first_client.put_object("only-in-first.csv", b"hello", "text/csv")

    with pytest.raises(NotFoundError):
        await second_client.get_object("only-in-first.csv")


# ---------------------------------------------------------------------------
# get_storage_client: environment-driven dispatch
# ---------------------------------------------------------------------------

def test_get_storage_client_returns_in_memory_client_in_test_environment():
    settings = SimpleNamespace(
        environment="test",
        object_storage_endpoint=None,
        object_storage_bucket="dental-platform",
        object_storage_access_key=None,
        object_storage_secret_key=None,
    )

    client = get_storage_client(settings)

    assert isinstance(client, InMemoryStorageClient)


def test_get_storage_client_dispatches_to_s3_client_outside_test_environment(monkeypatch):
    captured = {}

    class _FakeS3Client:
        def __init__(self, endpoint, bucket, access_key, secret_key):
            captured["args"] = (endpoint, bucket, access_key, secret_key)

    monkeypatch.setattr(storage_module, "S3StorageClient", _FakeS3Client)

    settings = SimpleNamespace(
        environment="production",
        object_storage_endpoint="https://minio.internal:9000",
        object_storage_bucket="dental-platform",
        object_storage_access_key="access-key",
        object_storage_secret_key="secret-key",
    )

    client = storage_module.get_storage_client(settings)

    assert isinstance(client, _FakeS3Client)
    assert not isinstance(client, InMemoryStorageClient)


# ---------------------------------------------------------------------------
# S3StorageClient
# ---------------------------------------------------------------------------

def test_s3_storage_client_constructs_without_a_real_network_call(monkeypatch):
    monkeypatch.setattr(boto3, "client", MagicMock(return_value=MagicMock()))

    client = S3StorageClient(
        endpoint="http://localhost:9000",
        bucket="dental-platform",
        access_key="access-key",
        secret_key="secret-key",
    )

    assert isinstance(client, S3StorageClient)


async def test_s3_storage_client_put_object_returns_s3_uri(monkeypatch):
    fake_boto_client = MagicMock()
    monkeypatch.setattr(boto3, "client", MagicMock(return_value=fake_boto_client))
    if hasattr(storage_module, "client"):
        monkeypatch.setattr(storage_module, "client", MagicMock(return_value=fake_boto_client))

    client = S3StorageClient(
        endpoint="http://localhost:9000",
        bucket="dental-platform",
        access_key="access-key",
        secret_key="secret-key",
    )

    uri = await client.put_object("exports/report.csv", b"a,b,c", "text/csv")

    assert uri == "s3://dental-platform/exports/report.csv"
