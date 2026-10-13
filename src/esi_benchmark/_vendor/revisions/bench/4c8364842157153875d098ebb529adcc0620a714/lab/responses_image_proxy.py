#!/usr/bin/env python3
"""Offload Responses API data-URL images to R2 before forwarding a request."""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import threading
import time
from typing import Any, Protocol
from urllib.parse import urlparse
import uuid

import httpx


_HOP_BY_HOP_HEADERS = {
    b"connection",
    b"keep-alive",
    b"proxy-authenticate",
    b"proxy-authorization",
    b"te",
    b"trailer",
    b"transfer-encoding",
    b"upgrade",
}
_IMAGE_MIME = re.compile(r"image/[A-Za-z0-9!#$&^_.+-]+\Z", re.ASCII | re.IGNORECASE)
_SUMMARY_FIELDS = {
    "requests_received": 0,
    "requests_forwarded": 0,
    "responses_completed": 0,
    "images_rewritten": 0,
    "unique_images_uploaded": 0,
    "cache_hits": 0,
    "decoded_image_bytes": 0,
    "source_request_bytes": 0,
    "forwarded_request_bytes": 0,
    "bytes_removed": 0,
    "upload_seconds": 0.0,
    "upstream_seconds": 0.0,
    "rewrite_errors": 0,
    "upstream_errors": 0,
    "metrics_dropped": 0,
    "remote_objects_deleted": 0,
    "cleanup_errors": 0,
}
_R2_ENV_KEYS = {
    "R2_ACCOUNT_ID",
    "R2_ENDPOINT_URL",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET",
    "R2_REGION",
}
_ENV_KEY = re.compile(r"[A-Z][A-Z0-9_]*\Z", re.ASCII)
_MAX_ENV_FILE_BYTES = 16 * 1024


class ImageRewriteError(ValueError):
    """The request could not be safely rewritten."""


class ObjectStore(Protocol):
    async def upload(
        self, digest: str, media_type: str, content: bytes
    ) -> None: ...

    async def signed_get_url(self, digest: str) -> str: ...

    async def cleanup_uploaded(self) -> CleanupResult: ...


@dataclass(frozen=True)
class CleanupResult:
    deleted: int = 0
    errors: int = 0


@dataclass(frozen=True)
class _ImageRef:
    owner: dict[str, Any]
    media_type: str
    content: bytes
    digest: str


@dataclass(frozen=True)
class _OffloadResult:
    url: str
    uploaded: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _connection_tokens(headers: list[tuple[bytes, bytes]]) -> set[bytes]:
    tokens: set[bytes] = set()
    for name, value in headers:
        if name.lower() == b"connection":
            tokens.update(part.strip().lower() for part in value.split(b",") if part.strip())
    return tokens


def _forward_headers(
    headers: list[tuple[bytes, bytes]], *, request: bool
) -> list[tuple[bytes, bytes]]:
    blocked = _HOP_BY_HOP_HEADERS | _connection_tokens(headers)
    if request:
        # The destination host and rewritten body length belong to httpx.
        blocked |= {b"host", b"content-length"}
    return [(name, value) for name, value in headers if name.lower() not in blocked]


def _decode_data_image(url: str) -> tuple[str, bytes] | None:
    prefix, separator, encoded = url.partition(",")
    lower_prefix = prefix.lower()
    if not (
        lower_prefix.startswith("data:image/")
        and lower_prefix.endswith(";base64")
    ):
        return None
    if not separator:
        raise ImageRewriteError("image data URL has no payload")
    media_type = prefix[5:-7]
    if not _IMAGE_MIME.fullmatch(media_type):
        raise ImageRewriteError("image data URL has an invalid media type")
    try:
        content = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise ImageRewriteError("image data URL has invalid base64") from exc
    if not content:
        raise ImageRewriteError("image data URL is empty")
    return media_type.lower(), content


def _collect_images(value: Any, *, max_nodes: int = 2_000_000) -> list[_ImageRef]:
    refs: list[_ImageRef] = []
    stack = [value]
    visited = 0
    while stack:
        current = stack.pop()
        visited += 1
        if visited > max_nodes:
            raise ImageRewriteError("request JSON is too deeply nested or too large")
        if isinstance(current, dict):
            if current.get("type") == "input_image" and isinstance(
                current.get("image_url"), str
            ):
                decoded = _decode_data_image(current["image_url"])
                if decoded is not None:
                    media_type, content = decoded
                    refs.append(
                        _ImageRef(
                            owner=current,
                            media_type=media_type,
                            content=content,
                            digest=hashlib.sha256(content).hexdigest(),
                        )
                    )
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return refs


def _validate_signed_url(url: str) -> None:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ImageRewriteError("object store returned an invalid HTTPS URL")


class _ImageOffloader:
    def __init__(self, store: ObjectStore) -> None:
        self._store = store
        self._uploads: dict[str, asyncio.Task[None]] = {}
        self._signed_urls: dict[str, asyncio.Task[str]] = {}
        self._lock = asyncio.Lock()

    async def offload(self, image: _ImageRef) -> _OffloadResult:
        async with self._lock:
            upload_task = self._uploads.get(image.digest)
            uploaded = upload_task is None
            if upload_task is None:
                upload_task = asyncio.create_task(
                    self._store.upload(
                        image.digest, image.media_type, image.content
                    )
                )
                self._uploads[image.digest] = upload_task
            url_task = self._signed_urls.get(image.digest)
            if url_task is None:
                url_task = asyncio.create_task(
                    self._sign_after_upload(image.digest, upload_task)
                )
                self._signed_urls[image.digest] = url_task
        try:
            url = await asyncio.shield(url_task)
        except BaseException:
            async with self._lock:
                upload_failed = upload_task.done() and (
                    upload_task.cancelled() or upload_task.exception() is not None
                )
                url_failed = url_task.done() and (
                    url_task.cancelled() or url_task.exception() is not None
                )
                if upload_failed:
                    if self._uploads.get(image.digest) is upload_task:
                        self._uploads.pop(image.digest, None)
                if url_failed:
                    if self._signed_urls.get(image.digest) is url_task:
                        self._signed_urls.pop(image.digest, None)
            raise
        return _OffloadResult(url=url, uploaded=uploaded)

    async def _sign_after_upload(
        self, digest: str, upload_task: asyncio.Task[None]
    ) -> str:
        await asyncio.shield(upload_task)
        url = await self._store.signed_get_url(digest)
        _validate_signed_url(url)
        return url


class RunMetrics:
    """A bounded, secret-free JSONL map plus an atomically updated summary."""

    def __init__(
        self,
        run_dir: Path,
        *,
        max_jsonl_bytes: int = 16 * 1024 * 1024,
        max_jsonl_records: int = 100_000,
    ) -> None:
        self.transport_dir = run_dir / "transport"
        self.transport_dir.mkdir(parents=True, exist_ok=True)
        self.image_map_path = self.transport_dir / "image_map.jsonl"
        self.summary_path = self.transport_dir / "summary.json"
        self._max_jsonl_bytes = max_jsonl_bytes
        self._max_jsonl_records = max_jsonl_records
        self._jsonl_bytes = (
            self.image_map_path.stat().st_size if self.image_map_path.exists() else 0
        )
        self._jsonl_records = 0
        if self._jsonl_bytes <= self._max_jsonl_bytes and self.image_map_path.exists():
            with self.image_map_path.open("rb") as stream:
                self._jsonl_records = sum(1 for _ in stream)
        self._summary: dict[str, int | float | str] = {
            "schema_version": 1,
            "updated_at_utc": _utc_now(),
            **_SUMMARY_FIELDS,
        }
        self._lock = asyncio.Lock()
        self._write_summary()

    def _write_summary(self) -> None:
        self._summary["updated_at_utc"] = _utc_now()
        temporary = self.summary_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self._summary, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.summary_path)

    async def update(
        self,
        *,
        increments: dict[str, int | float] | None = None,
        images: list[dict[str, Any]] | None = None,
    ) -> None:
        async with self._lock:
            for key, amount in (increments or {}).items():
                if key not in _SUMMARY_FIELDS:
                    raise KeyError(key)
                self._summary[key] = self._summary[key] + amount  # type: ignore[operator]
            for image in images or []:
                record = {
                    "schema_version": 1,
                    "timestamp_utc": _utc_now(),
                    **image,
                }
                line = (
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                ).encode("utf-8")
                if (
                    self._jsonl_records >= self._max_jsonl_records
                    or self._jsonl_bytes + len(line) > self._max_jsonl_bytes
                ):
                    self._summary["metrics_dropped"] = (
                        int(self._summary["metrics_dropped"]) + 1
                    )
                    continue
                with self.image_map_path.open("ab") as stream:
                    stream.write(line)
                self._jsonl_records += 1
                self._jsonl_bytes += len(line)
            self._write_summary()


def load_r2_env_file(path: str | Path) -> dict[str, str]:
    """Read a small owner-only dotenv file without executing or exporting it."""

    env_path = Path(path)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("this platform cannot safely open an R2 env file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    try:
        descriptor = os.open(env_path, flags)
    except OSError as exc:
        raise RuntimeError("R2 env file could not be opened safely") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("R2 env file must be a regular file")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise RuntimeError("R2 env file must not grant group or other permissions")
        if metadata.st_size > _MAX_ENV_FILE_BYTES:
            raise RuntimeError("R2 env file exceeds 16 KiB")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(_MAX_ENV_FILE_BYTES + 1)
        if len(raw) > _MAX_ENV_FILE_BYTES:
            raise RuntimeError("R2 env file exceeds 16 KiB")
    finally:
        os.close(descriptor)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("R2 env file must be UTF-8") from exc

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if (
            not separator
            or not _ENV_KEY.fullmatch(key)
            or key not in _R2_ENV_KEYS
            or key in values
        ):
            raise RuntimeError(f"invalid R2 env entry on line {line_number}")
        quoted = value[:1] in {'"', "'"} or value[-1:] in {'"', "'"}
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        elif quoted:
            raise RuntimeError(f"invalid R2 env value on line {line_number}")
        elif any(character.isspace() for character in value):
            raise RuntimeError(f"invalid R2 env value on line {line_number}")
        if not value or "\x00" in value or "\r" in value or "\n" in value:
            raise RuntimeError(f"empty or invalid R2 env value on line {line_number}")
        values[key] = value
    return values


class R2ObjectStore:
    """Small async wrapper around a boto3 R2 client."""

    def __init__(
        self,
        *,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        region: str,
        prefix: str,
        signed_url_seconds: int = 86_400,
        upload_proxy: str | None = None,
    ) -> None:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise RuntimeError(
                "R2 mode requires boto3; install it in the proxy's isolated venv"
            ) from exc

        if not all((endpoint, access_key, secret_key, bucket)):
            raise RuntimeError("R2 endpoint, credentials, and bucket are required")
        parsed_endpoint = urlparse(endpoint)
        if (
            parsed_endpoint.scheme != "https"
            or not parsed_endpoint.hostname
            or not parsed_endpoint.hostname.endswith(".r2.cloudflarestorage.com")
            or parsed_endpoint.username is not None
            or parsed_endpoint.password is not None
            or parsed_endpoint.path not in {"", "/"}
            or parsed_endpoint.query
            or parsed_endpoint.fragment
        ):
            raise RuntimeError("R2 endpoint must be a Cloudflare S3 HTTPS endpoint")
        clean_prefix = prefix.strip("/")
        if not clean_prefix or ".." in clean_prefix.split("/"):
            raise RuntimeError("R2 prefix must be a non-empty safe object prefix")
        if not 60 <= signed_url_seconds <= 604_800:
            raise RuntimeError("signed URL lifetime must be between 60 and 604800 seconds")
        proxy_config: dict[str, str] = {}
        if upload_proxy is not None:
            parsed_proxy = urlparse(upload_proxy)
            if (
                parsed_proxy.scheme not in {"http", "https"}
                or not parsed_proxy.hostname
                or parsed_proxy.username is not None
                or parsed_proxy.password is not None
                or parsed_proxy.path not in {"", "/"}
                or parsed_proxy.query
                or parsed_proxy.fragment
            ):
                raise RuntimeError("R2 upload proxy must be a plain HTTP(S) origin")
            proxy_config = {"https": upload_proxy}

        self._bucket = bucket
        self._prefix = clean_prefix
        self._signed_url_seconds = signed_url_seconds
        self._uploaded_keys: set[str] = set()
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint.rstrip("/"),
            region_name=region or "auto",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(
                signature_version="s3v4",
                proxies=proxy_config,
                retries={"mode": "standard", "total_max_attempts": 3},
            ),
        )

    @classmethod
    def from_env_file(
        cls,
        env_file: str | Path,
        *,
        prefix: str,
        signed_url_seconds: int = 86_400,
        upload_proxy: str | None = None,
    ) -> R2ObjectStore:
        values = load_r2_env_file(env_file)
        endpoint = values.get("R2_ENDPOINT_URL", "").strip()
        account_id = values.get("R2_ACCOUNT_ID", "").strip()
        if not endpoint and account_id:
            endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
        required = {
            "endpoint": endpoint,
            "access key": values.get("R2_ACCESS_KEY_ID", ""),
            "secret key": values.get("R2_SECRET_ACCESS_KEY", ""),
            "bucket": values.get("R2_BUCKET", ""),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError("R2 env file is missing required settings")
        return cls(
            endpoint=endpoint,
            access_key=required["access key"],
            secret_key=required["secret key"],
            bucket=required["bucket"],
            region=values.get("R2_REGION", "auto"),
            prefix=prefix,
            signed_url_seconds=signed_url_seconds,
            upload_proxy=upload_proxy,
        )

    def _key(self, digest: str) -> str:
        return f"{self._prefix}/{digest}"

    async def upload(self, digest: str, media_type: str, content: bytes) -> None:
        key = self._key(digest)
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self._bucket,
            Key=key,
            Body=content,
            ContentLength=len(content),
            ContentType=media_type,
            Metadata={"sha256": digest},
        )
        self._uploaded_keys.add(key)

    async def signed_get_url(self, digest: str) -> str:
        return await asyncio.to_thread(
            self._client.generate_presigned_url,
            "get_object",
            Params={"Bucket": self._bucket, "Key": self._key(digest)},
            ExpiresIn=self._signed_url_seconds,
        )

    async def cleanup_uploaded(self) -> CleanupResult:
        keys = sorted(self._uploaded_keys)
        deleted = 0
        errors = 0
        for offset in range(0, len(keys), 1000):
            batch = keys[offset : offset + 1000]
            try:
                response = await asyncio.to_thread(
                    self._client.delete_objects,
                    Bucket=self._bucket,
                    Delete={
                        "Objects": [{"Key": key} for key in batch],
                        "Quiet": True,
                    },
                )
                response_errors = response.get("Errors", [])
                failed = {
                    item.get("Key")
                    for item in response_errors
                    if isinstance(item, dict) and isinstance(item.get("Key"), str)
                }
                if len(failed) != len(response_errors):
                    errors += len(batch)
                    continue
                deleted_keys = set(batch) - failed
                deleted += len(deleted_keys)
                errors += len(failed)
                self._uploaded_keys.difference_update(deleted_keys)
            except Exception:
                errors += len(batch)
                continue
        return CleanupResult(deleted=deleted, errors=errors)


class ResponsesImageProxy:
    def __init__(
        self,
        *,
        upstream_url: str,
        store: ObjectStore,
        metrics: RunMetrics,
        client: httpx.AsyncClient | None = None,
        client_timeout: httpx.Timeout | None = None,
        max_body_bytes: int = 2 * 1024 * 1024 * 1024,
    ) -> None:
        parsed = urlparse(upstream_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or not parsed.path.rstrip("/").endswith("/v1/responses")
        ):
            raise ValueError("upstream URL must be HTTP(S)")
        self._upstream_url = upstream_url
        self._store = store
        self._offloader = _ImageOffloader(store)
        self._metrics = metrics
        self._max_body_bytes = max_body_bytes
        self._client = client or httpx.AsyncClient(
            trust_env=False, timeout=client_timeout or httpx.Timeout(900.0)
        )
        self._owns_client = client is None
        self._closed = False
        self._close_lock = asyncio.Lock()

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope["type"] != "http":
            return
        if scope.get("path") != "/v1/responses":
            await self._error(send, 404, "not_found")
            return
        if scope.get("method") != "POST":
            await self._error(send, 405, "method_not_allowed")
            return

        request_id = uuid.uuid4().hex[:16]
        try:
            body = await self._read_body(receive)
        except ImageRewriteError:
            await self._metrics.update(increments={"rewrite_errors": 1})
            await self._error(send, 413, "request_too_large")
            return
        await self._metrics.update(
            increments={"requests_received": 1, "source_request_bytes": len(body)}
        )

        started = time.monotonic()
        try:
            payload = json.loads(body)
            refs = _collect_images(payload)
            results = await asyncio.gather(
                *(self._offloader.offload(image) for image in refs)
            )
            for image, result in zip(refs, results, strict=True):
                image.owner["image_url"] = result.url
            forwarded_body = (
                json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
                if refs
                else body
            )
        except (ImageRewriteError, UnicodeDecodeError, json.JSONDecodeError):
            await self._metrics.update(increments={"rewrite_errors": 1})
            await self._error(send, 400, "invalid_responses_request")
            return
        except Exception:
            await self._metrics.update(increments={"rewrite_errors": 1})
            await self._error(send, 502, "image_offload_failed")
            return

        upload_seconds = time.monotonic() - started
        await self._metrics.update(
            increments={
                "images_rewritten": len(refs),
                "unique_images_uploaded": sum(result.uploaded for result in results),
                "cache_hits": sum(not result.uploaded for result in results),
                "decoded_image_bytes": sum(len(image.content) for image in refs),
                "forwarded_request_bytes": len(forwarded_body),
                "bytes_removed": max(0, len(body) - len(forwarded_body)),
                "upload_seconds": upload_seconds,
            },
            images=[
                {
                    "request_id": request_id,
                    "sha256": image.digest,
                    "media_type": image.media_type,
                    "bytes": len(image.content),
                    "uploaded": result.uploaded,
                    "cache_hit": not result.uploaded,
                }
                for image, result in zip(refs, results, strict=True)
            ],
        )

        headers = _forward_headers(list(scope.get("headers", [])), request=True)
        request = httpx.Request(
            "POST", self._upstream_url, headers=headers, content=forwarded_body
        )
        await self._metrics.update(increments={"requests_forwarded": 1})
        upstream_started = time.monotonic()
        response: httpx.Response | None = None
        response_started = False
        try:
            # Intentionally one attempt: provider retries belong to the caller.
            response = await self._client.send(request, stream=True)
            response_headers = _forward_headers(
                list(response.headers.raw), request=False
            )
            await send(
                {
                    "type": "http.response.start",
                    "status": response.status_code,
                    "headers": response_headers,
                }
            )
            response_started = True
            async for chunk in response.aiter_raw():
                await send(
                    {"type": "http.response.body", "body": chunk, "more_body": True}
                )
            await send({"type": "http.response.body", "body": b""})
        except Exception:
            await self._metrics.update(
                increments={
                    "upstream_errors": 1,
                    "upstream_seconds": time.monotonic() - upstream_started,
                }
            )
            if not response_started:
                await self._error(send, 502, "upstream_failed")
            return
        finally:
            if response is not None:
                await response.aclose()
        await self._metrics.update(
            increments={
                "responses_completed": 1,
                "upstream_seconds": time.monotonic() - upstream_started,
            }
        )

    async def _read_body(self, receive: Any) -> bytes:
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                raise ImageRewriteError("client disconnected")
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self._max_body_bytes:
                raise ImageRewriteError("request body is too large")
            chunks.append(chunk)
            if not message.get("more_body", False):
                return b"".join(chunks)

    async def _lifespan(self, receive: Any, send: Any) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await self.close()
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            try:
                cleanup = await self._store.cleanup_uploaded()
            except Exception:
                cleanup = CleanupResult(errors=1)
            await self._metrics.update(
                increments={
                    "remote_objects_deleted": cleanup.deleted,
                    "cleanup_errors": cleanup.errors,
                }
            )
            if self._owns_client:
                try:
                    await self._client.aclose()
                except Exception:
                    pass

    @staticmethod
    async def _error(send: Any, status: int, error_type: str) -> None:
        body = json.dumps(
            {"error": {"type": error_type}}, separators=(",", ":")
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class ResponsesImageProxyHandle:
    def __init__(
        self,
        *,
        base_url: str,
        server: Any,
        thread: threading.Thread,
        listener: socket.socket,
        app: ResponsesImageProxy,
        close_timeout: float,
    ) -> None:
        self.base_url = base_url
        self._server = server
        self._thread = thread
        self._listener = listener
        self._app = app
        self._close_timeout = close_timeout
        self._closed = False
        self._lock = threading.Lock()

    def close(self) -> None:
        """Stop the proxy and clean its run objects; repeated calls never raise."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._server.should_exit = True
                self._thread.join(self._close_timeout)
                if self._thread.is_alive():
                    self._server.force_exit = True
                    self._thread.join(self._close_timeout)
                if self._thread.is_alive():
                    asyncio.run(self._app.close())
            except BaseException:
                pass
            finally:
                try:
                    asyncio.run(self._app.close())
                except BaseException:
                    pass
                try:
                    self._listener.close()
                except BaseException:
                    pass

    def __enter__(self) -> ResponsesImageProxyHandle:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def start_responses_image_proxy(
    *,
    upstream_url: str,
    run_dir: str | Path,
    r2_env: str | Path,
    r2_prefix: str,
    r2_upload_proxy: str | None = None,
    signed_url_seconds: int = 86_400,
    max_body_bytes: int = 2 * 1024 * 1024 * 1024,
    upstream_read_timeout: float = 900.0,
    startup_timeout: float = 15.0,
    close_timeout: float = 15.0,
) -> ResponsesImageProxyHandle:
    """Start an embedded proxy on an ephemeral loopback port."""

    import uvicorn

    store = R2ObjectStore.from_env_file(
        r2_env,
        prefix=r2_prefix,
        signed_url_seconds=signed_url_seconds,
        upload_proxy=r2_upload_proxy,
    )
    metrics = RunMetrics(Path(run_dir))
    app = ResponsesImageProxy(
        upstream_url=upstream_url,
        store=store,
        metrics=metrics,
        client_timeout=httpx.Timeout(
            connect=30.0,
            read=upstream_read_timeout,
            write=900.0,
            pool=30.0,
        ),
        max_body_bytes=max_body_bytes,
    )
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            access_log=False,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        exited = threading.Event()

        def run() -> None:
            try:
                server.run(sockets=[listener])
            except BaseException:
                pass
            finally:
                exited.set()

        thread = threading.Thread(
            target=run,
            name=f"responses-image-proxy-{port}",
            daemon=True,
        )
        thread.start()
        deadline = time.monotonic() + startup_timeout
        while not server.started and time.monotonic() < deadline:
            if exited.wait(0.01):
                break
        if not server.started:
            server.should_exit = True
            server.force_exit = True
            thread.join(min(close_timeout, 5.0))
            try:
                asyncio.run(app.close())
            except BaseException:
                pass
            raise RuntimeError("embedded image proxy failed to start") from None
        return ResponsesImageProxyHandle(
            base_url=f"http://127.0.0.1:{port}/v1",
            server=server,
            thread=thread,
            listener=listener,
            app=app,
            close_timeout=close_timeout,
        )
    except BaseException:
        try:
            asyncio.run(app.close())
        except BaseException:
            pass
        try:
            listener.close()
        except BaseException:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--r2-env", type=Path, required=True)
    parser.add_argument("--r2-prefix", required=True)
    parser.add_argument("--r2-upload-proxy")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--signed-url-seconds", type=int, default=86_400)
    parser.add_argument("--max-body-mib", type=int, default=2048)
    parser.add_argument("--upstream-read-timeout", type=float, default=900.0)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.max_body_mib <= 0:
        parser.error("--max-body-mib must be positive")

    store = R2ObjectStore.from_env_file(
        args.r2_env,
        prefix=args.r2_prefix,
        signed_url_seconds=args.signed_url_seconds,
        upload_proxy=args.r2_upload_proxy,
    )
    metrics = RunMetrics(args.run_dir)
    timeout = httpx.Timeout(
        connect=30.0,
        read=args.upstream_read_timeout,
        write=900.0,
        pool=30.0,
    )
    app = ResponsesImageProxy(
        upstream_url=args.upstream_url,
        store=store,
        metrics=metrics,
        client_timeout=timeout,
        max_body_bytes=args.max_body_mib * 1024 * 1024,
    )
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
