from __future__ import annotations

import os
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StoredObject:
    provider: str
    uri: str
    bucket: str
    key: str
    metadata: dict[str, Any]


class S3CompatibleObjectStore:
    """Small S3-compatible object store adapter.

    This follows the same basic pattern used by automated-w2s-research: use
    boto3, allow custom endpoints for RunPod/S3-compatible stores, and keep the
    framework-facing contract provider-neutral.
    """

    def __init__(
        self,
        *,
        bucket: str | None = None,
        prefix: str | None = None,
        endpoint_url: str | None = None,
        region_name: str | None = None,
    ) -> None:
        self.bucket = bucket or os.environ.get("AO_S3_BUCKET") or os.environ.get("S3_BUCKET")
        if not self.bucket:
            raise ValueError("S3 artifact storage requires AO_S3_BUCKET or S3_BUCKET")
        self.prefix = (prefix if prefix is not None else os.environ.get("AO_S3_PREFIX", "agentic-opt/artifacts/")).strip("/")
        self.endpoint_url = endpoint_url or os.environ.get("AO_S3_ENDPOINT_URL") or os.environ.get("S3_ENDPOINT_URL")
        self.region_name = region_name or os.environ.get("AO_S3_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        self._client = None

    def upload_path(self, *, source: Path, artifact_id: str, filename: str | None = None) -> StoredObject:
        upload_source = source
        cleanup_path: Path | None = None
        metadata: dict[str, Any] = {"source_path": str(source), "packed": False}
        if source.is_dir():
            fd, raw_path = tempfile.mkstemp(prefix=f"{artifact_id}_", suffix=".tar.gz")
            os.close(fd)
            cleanup_path = Path(raw_path)
            with tarfile.open(cleanup_path, "w:gz") as archive:
                archive.add(source, arcname=source.name)
            upload_source = cleanup_path
            metadata["packed"] = True
            metadata["archive_format"] = "tar.gz"
        key_name = filename or upload_source.name
        key = "/".join(part for part in (self.prefix, artifact_id, key_name) if part)
        self.client.upload_file(str(upload_source), self.bucket, key)
        if cleanup_path is not None:
            cleanup_path.unlink(missing_ok=True)
        return StoredObject(
            provider="s3",
            uri=f"s3://{self.bucket}/{key}",
            bucket=self.bucket,
            key=key,
            metadata={
                **metadata,
                "bucket": self.bucket,
                "key": key,
                "endpoint_url": self.endpoint_url,
                "region_name": self.region_name,
            },
        )

    @property
    def client(self):
        if self._client is None:
            self._client = self._make_client()
        return self._client

    def _make_client(self):
        try:
            import boto3  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("S3 artifact storage requires optional dependency 'boto3'") from exc
        kwargs: dict[str, Any] = {"region_name": self.region_name}
        access_key = os.environ.get("AO_AWS_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AO_AWS_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
        if access_key and secret_key:
            kwargs["aws_access_key_id"] = access_key
            kwargs["aws_secret_access_key"] = secret_key
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        return boto3.client("s3", **kwargs)

