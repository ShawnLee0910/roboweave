"""Pure model/scene control protocol and Unix transport.

This module must remain free of simulator imports so a model client can attach
to a resident scene without importing OmniGibson or acquiring GPU resources.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
from typing import Any, Mapping

SCENE_PROTOCOL_VERSION = "behavior-hl-bench-try-scene-control/v1"
CONTROL_SOCKET_NAME = "s"
MAX_UNIX_SOCKET_PATH_BYTES = 107
MAX_SCENE_RESPONSE_BYTES = 256 * 1024
BENCH_TRY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_READY_PATH = BENCH_TRY_ROOT / "runtime" / "task25" / "ready.json"


class ModelClientError(RuntimeError):
    pass


class SceneTransportTimeout(ModelClientError):
    """Reply not received; execution may have completed. Never replay the action."""

    def __init__(self, command, sequence, timeout_seconds):
        super().__init__('scene control response timed out; execution outcome is unknown')
        self.command = command
        self.sequence = sequence
        self.timeout_seconds = timeout_seconds


class SceneRequestError(ModelClientError):
    """A typed, privacy-safe failure returned by the resident scene."""

    def __init__(
        self,
        *,
        error_type: str,
        message: str,
        error_id: str | None,
        recovery: Mapping[str, Any] | None,
        error_origin: str = "resident",
    ) -> None:
        if error_origin not in {"resident", "transport"}:
            raise ValueError("scene error origin is invalid")
        super().__init__(f"{error_type}: {message}")
        self.error_type = error_type
        self.error_id = error_id
        self.recovery = None if recovery is None else dict(recovery)
        self.error_origin = error_origin


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _read_strict_json(path: Path, *, max_bytes: int) -> Mapping[str, Any]:
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise ModelClientError(f"JSON artifact is missing or oversized: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ModelClientError("JSON artifact is not an object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_json_bytes(value) + b"\n")
    temporary.replace(path)


class SceneClient:
    """Validated client for one resident scene's Unix control socket."""

    def __init__(
        self,
        *,
        ready_path: Path = DEFAULT_READY_PATH,
        expected_task_id: int | None = None,
        expected_task_name: str | None = None,
        expected_split: str | None = None,
        expected_instance_id: str | None = None,
        runtime_root: Path | None = None,
    ) -> None:
        ready_path = Path(ready_path).resolve()
        runtime_registry = (
            BENCH_TRY_ROOT / "runtime" if runtime_root is None else Path(runtime_root)
        ).resolve()
        if (
            ready_path.name != "ready.json"
            or not ready_path.is_relative_to(runtime_registry)
            or ready_path.parent.parent != runtime_registry
        ):
            raise ValueError("scene ready path escaped the runtime registry")
        ready = _read_strict_json(ready_path, max_bytes=64 * 1024)
        if (
            ready.get("protocol_version") != SCENE_PROTOCOL_VERSION
            or ready.get("model_bound") is not False
            or ready.get("prompt_bound") is not False
            or ready.get("provider_bound") is not False
        ):
            raise ModelClientError("resident scene identity is invalid")
        self.socket_path = Path(ready["socket_path"]).resolve()
        self.control_token = ready["control_token"]
        self.scene_run_dir = Path(ready["run_dir"]).resolve()
        if (
            not isinstance(self.control_token, str)
            or len(self.control_token) < 32
            or not self.socket_path.is_socket()
            or not self.scene_run_dir.is_dir()
        ):
            raise ModelClientError("resident scene control endpoint is invalid")
        expected = (
            expected_task_id,
            expected_task_name,
            expected_split,
            expected_instance_id,
        )
        if any(value is not None for value in expected):
            if (
                isinstance(expected_task_id, bool)
                or not isinstance(expected_task_id, int)
                or not isinstance(expected_task_name, str)
                or not expected_task_name.strip()
                or not isinstance(expected_split, str)
                or not expected_split.strip()
                or not isinstance(expected_instance_id, str)
                or not expected_instance_id.strip()
            ):
                raise ValueError("expected resident task identity is invalid")
            status = self.request("status")
            if (
                status.get("task_id") != expected_task_id
                or status.get("task_name") != expected_task_name
                or status.get("task_split") != expected_split
                or str(status.get("task_instance_id")) != expected_instance_id
            ):
                raise ModelClientError("resident scene task identity does not match")

    def request(self, command: str, **fields: Any) -> Mapping[str, Any]:
        request = {
            "protocol_version": SCENE_PROTOCOL_VERSION,
            "control_token": self.control_token,
            "command": command,
            **fields,
        }
        encoded = _canonical_json_bytes(request) + b"\n"
        if len(encoded) > 64 * 1024:
            raise ModelClientError("scene request exceeds size limit")
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # Action placement can exceed five minutes. Stay below the CLI's 900s
        # tool limit; the outer episode wall budget remains authoritative.
        timeout_seconds = 600 if command == 'step' else 300
        connection.settimeout(timeout_seconds)
        try:
            try:
                connection.connect(str(self.socket_path))
                connection.sendall(encoded)
                stream = connection.makefile("rb")
                line = stream.readline(MAX_SCENE_RESPONSE_BYTES + 1)
            except TimeoutError as exc:
                raise SceneTransportTimeout(command, fields.get('sequence'), timeout_seconds) from exc
            except OSError as exc:
                raise ModelClientError("scene control transport failed") from exc
        finally:
            connection.close()
        if not line or len(line) > MAX_SCENE_RESPONSE_BYTES:
            raise ModelClientError("scene response is missing or oversized")
        try:
            response = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelClientError("scene response is not strict UTF-8 JSON") from exc
        if not isinstance(response, Mapping):
            raise ModelClientError("scene response is not a JSON object")
        if response.get("status") == "ERROR":
            error_type = response.get("error_type")
            message = response.get("message")
            error_id = response.get("error_id")
            recovery = response.get("recovery")
            if (
                not isinstance(error_type, str)
                or not isinstance(message, str)
                or (error_id is not None and not isinstance(error_id, str))
                or (recovery is not None and not isinstance(recovery, Mapping))
            ):
                raise ModelClientError("scene error response contract changed")
            raise SceneRequestError(
                error_type=error_type,
                message=message,
                error_id=error_id,
                recovery=recovery,
                error_origin="resident",
            )
        if (
            response.get("protocol_version") != SCENE_PROTOCOL_VERSION
            or response.get("command") != command
            or not isinstance(response.get("result"), Mapping)
        ):
            raise ModelClientError("scene response contract changed")
        return response["result"]


def control_socket_path(runtime_root: Path) -> Path:
    path = Path(runtime_root).resolve() / CONTROL_SOCKET_NAME
    if len(os.fsencode(path)) > MAX_UNIX_SOCKET_PATH_BYTES:
        raise ValueError("resident control socket path exceeds AF_UNIX limit")
    return path

__all__ = [
    "DEFAULT_READY_PATH",
    "CONTROL_SOCKET_NAME",
    "ModelClientError",
    "SceneClient",
    "SceneRequestError",
    "MAX_UNIX_SOCKET_PATH_BYTES",
    "SCENE_PROTOCOL_VERSION",
    "control_socket_path",
]
