from __future__ import annotations

import ast
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import secrets
import stat
import traceback
from types import SimpleNamespace
from typing import Any, Mapping


LAB_ROOT = (Path(__file__).resolve().parents[2] / "src/roboweave/_vendor/bench/lab")
SCENE_SERVER_PATH = LAB_ROOT / "scene_server.py"


def _load_scene_error_boundary() -> dict[str, Any]:
    """Load the exact CPU-only boundary without importing OmniGibson."""

    tree = ast.parse(SCENE_SERVER_PATH.read_text(encoding="utf-8"))
    wanted_assignments = {
        "SCENE_PROTOCOL_VERSION",
        "SCENE_SERVER_VERSION",
        "MAX_REQUEST_BYTES",
        "PRIVATE_ERROR_LOG_RELATIVE_PATH",
        "MAX_PRIVATE_ERROR_ENTRY_BYTES",
        "MAX_PRIVATE_ERROR_LOG_BYTES",
        "_SECRET_ASSIGNMENT_PATTERN",
        "_BEARER_PATTERN",
        "_URL_CREDENTIAL_PATTERN",
        "_PRIVATE_KEY_PATTERN",
        "_KNOWN_REQUEST_OPERATIONS",
    }
    wanted_definitions = {
        "SceneProtocolError",
        "SceneIntegrationError",
        "_redact_private_error_text",
        "_bounded_private_error_payload",
        "_append_bounded_private_jsonl",
        "_private_request_operation",
        "_exception_fingerprint",
        "PrivateSceneErrorRecorder",
        "UnixSceneServer",
        "_canonical_json_bytes",
        "_error_response",
        "_write_json",
        "_write_private_json",
    }
    selected: list[ast.stmt] = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ]
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            if node.name in wanted_definitions:
                selected.append(node)
        elif isinstance(node, ast.Assign):
            names = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
            if names & wanted_assignments:
                selected.append(node)
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace: dict[str, Any] = {
        "Any": Any,
        "Mapping": Mapping,
        "Path": Path,
        "datetime": datetime,
        "timezone": timezone,
        "fcntl": fcntl,
        "hashlib": hashlib,
        "json": json,
        "os": os,
        "re": re,
        "secrets": secrets,
        "socket": SimpleNamespace(AF_UNIX=1, SOCK_STREAM=1),
        "stat": stat,
        "traceback": traceback,
        "send_response_or_client_gone": lambda connection, payload: True,
    }
    exec(compile(module, str(SCENE_SERVER_PATH), "exec"), namespace)
    return namespace


def test_public_error_is_generic_and_private_record_keeps_builder_detail(
    tmp_path: Path,
) -> None:
    boundary = _load_scene_error_boundary()
    control_token = "control-token-value-that-must-never-escape"
    processor = SimpleNamespace(control_token=control_token)
    server = boundary["UnixSceneServer"](
        processor=processor,
        socket_path=tmp_path / "control.sock",
        ready_path=tmp_path / "ready.json",
        run_dir=tmp_path / "simulator" / "run",
        runtime_root=tmp_path,
    )

    class R1ProB1ComponentBuilderError(RuntimeError):
        pass

    try:
        raise R1ProB1ComponentBuilderError(
            "phase=before_restricted_kinematic_base_step "
            "type=symbolic_grasp password=hunter2 "
            f"control_token={control_token} object_id=17 xyz=(1,2,3)"
        )
    except R1ProB1ComponentBuilderError as exc:
        public = server._error_response_for_exception(
            exc,
            request_operation="step",
        )

    assert set(public) == {
        "protocol_version",
        "status",
        "error_type",
        "message",
        "error_id",
    }
    assert public["status"] == "ERROR"
    assert public["error_type"] == "INTERNAL_ENV_ERROR"
    assert public["message"] == "INTERNAL_ENV_ERROR"
    assert len(public["error_id"]) == 32
    public_text = json.dumps(public, sort_keys=True)
    for forbidden in (
        "R1ProB1ComponentBuilderError",
        "before_restricted_kinematic_base_step",
        "symbolic_grasp",
        "hunter2",
        control_token,
        "object_id",
        "xyz",
        str(tmp_path),
    ):
        assert forbidden not in public_text

    log_path = tmp_path / "private" / "scene_server_errors.jsonl"
    assert stat.S_IMODE(log_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(log_path.parent.stat().st_mode) == 0o700
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    private = json.loads(
        lines[0],
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert private["error_id"] == public["error_id"]
    assert private["request_operation"] == "step"
    assert private["exception_type"] == "R1ProB1ComponentBuilderError"
    assert "phase=before_restricted_kinematic_base_step" in private["message"]
    assert "type=symbolic_grasp" in private["message"]
    assert "R1ProB1ComponentBuilderError" in private["traceback"]
    assert "[REDACTED]" in private["message"]
    assert "hunter2" not in private["message"]
    assert control_token not in private["message"]
    datetime.fromisoformat(private["occurred_at_utc"])

    redacted_json = boundary["_redact_private_error_text"](
        '{"api_key": "sk-sensitive-value", "safe": "phase/type"}',
        sensitive_values=(),
    )
    assert "sk-sensitive-value" not in redacted_json
    assert "phase/type" in redacted_json


def test_public_error_exposes_only_bounded_recovery_classification(
    tmp_path: Path,
) -> None:
    boundary = _load_scene_error_boundary()
    status = {
        "scene_resident": True,
        "session_valid": True,
        "episode_active": False,
        "episode_id": None,
        "next_sequence": 0,
        "rolling_checkpoint_action_counts": [3, 6],
        "private_target_name": "must-not-escape",
    }
    processor = SimpleNamespace(
        control_token="t" * 64,
        scene=SimpleNamespace(status=lambda: status),
    )
    server = boundary["UnixSceneServer"](
        processor=processor,
        socket_path=tmp_path / "control.sock",
        ready_path=tmp_path / "ready.json",
        run_dir=tmp_path / "simulator" / "run",
        runtime_root=tmp_path,
    )

    public = server._error_response_for_exception(
        RuntimeError("private object and pose detail"),
        request_operation="step",
    )

    assert public["error_type"] == "RECOVERABLE_SCENE_ERROR"
    assert public["message"] == "RECOVERABLE_SCENE_ERROR"
    assert public["recovery"] == {
        "scene_resident": True,
        "session_valid": True,
        "episode_active": False,
        "episode_id": None,
        "next_sequence": 0,
        "rolling_checkpoint_action_counts": [3, 6],
        "error_fingerprint": public["recovery"]["error_fingerprint"],
    }
    assert re.fullmatch(
        r"[0-9a-f]{64}", public["recovery"]["error_fingerprint"]
    )
    assert "private_target_name" not in json.dumps(public)
    assert "private object" not in json.dumps(public)


def test_protocol_error_is_distinct_without_exposing_its_message(
    tmp_path: Path,
) -> None:
    boundary = _load_scene_error_boundary()
    processor = SimpleNamespace(
        control_token="t" * 64,
        scene=SimpleNamespace(
            status=lambda: {
                "scene_resident": True,
                "session_valid": True,
                "episode_active": True,
                "episode_id": "episode",
                "next_sequence": 4,
                "rolling_checkpoint_action_counts": [3],
            }
        ),
    )
    server = boundary["UnixSceneServer"](
        processor=processor,
        socket_path=tmp_path / "control.sock",
        ready_path=tmp_path / "ready.json",
        run_dir=tmp_path / "simulator" / "run",
        runtime_root=tmp_path,
    )

    public = server._error_response_for_exception(
        boundary["SceneProtocolError"]("secret sequence detail"),
        request_operation="step",
    )

    assert public["error_type"] == "SCENE_PROTOCOL_ERROR"
    assert public["message"] == "SCENE_PROTOCOL_ERROR"
    assert "secret sequence detail" not in json.dumps(public)


def test_integration_error_is_distinct_and_checkpoint_recoverable(
    tmp_path: Path,
) -> None:
    boundary = _load_scene_error_boundary()
    processor = SimpleNamespace(
        control_token="t" * 64,
        scene=SimpleNamespace(
            status=lambda: {
                "scene_resident": True,
                "session_valid": True,
                "episode_active": False,
                "episode_id": None,
                "next_sequence": 0,
                "rolling_checkpoint_action_counts": [6],
            }
        ),
    )
    server = boundary["UnixSceneServer"](
        processor=processor,
        socket_path=tmp_path / "control.sock",
        ready_path=tmp_path / "ready.json",
        run_dir=tmp_path / "simulator" / "run",
        runtime_root=tmp_path,
    )

    public = server._error_response_for_exception(
        boundary["SceneIntegrationError"]("private marker wiring detail"),
        request_operation="step",
    )

    assert public["error_type"] == "INTEGRATION_ERROR"
    assert public["message"] == "INTEGRATION_ERROR"
    assert public["recovery"]["session_valid"] is True
    assert public["recovery"]["rolling_checkpoint_action_counts"] == [6]
    assert "private marker wiring detail" not in json.dumps(public)


def test_private_jsonl_has_per_entry_and_total_hard_bounds(tmp_path: Path) -> None:
    boundary = _load_scene_error_boundary()
    boundary["MAX_PRIVATE_ERROR_ENTRY_BYTES"] = 1024
    boundary["MAX_PRIVATE_ERROR_LOG_BYTES"] = 1500
    recorder = boundary["PrivateSceneErrorRecorder"](runtime_root=tmp_path)
    huge = RuntimeError("line\\n" * 100_000)

    assert recorder.record(
        error_id="a" * 32,
        exception=huge,
        request_operation="step",
    ) is True
    assert recorder.record(
        error_id="b" * 32,
        exception=huge,
        request_operation="step",
    ) is False

    path = tmp_path / "private" / "scene_server_errors.jsonl"
    payload = path.read_bytes()
    assert len(payload) <= 1500
    lines = payload.splitlines()
    assert len(lines) == 1
    assert len(lines[0]) + 1 <= 1024
    record = json.loads(lines[0])
    assert record["entry_truncated"] is True
    assert record["traceback_truncated"] is True


def test_handler_exception_and_client_disconnect_do_not_stop_loop(
    tmp_path: Path,
) -> None:
    boundary = _load_scene_error_boundary()
    sent: list[dict[str, Any]] = []

    class Processor:
        control_token = "t" * 64
        stop_requested = False

        def __init__(self) -> None:
            self.calls = 0

        def handle(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("phase=during_restricted secret=private-value")
            self.stop_requested = True
            return {
                "protocol_version": boundary["SCENE_PROTOCOL_VERSION"],
                "command": "status",
                "result": {"alive": True},
            }

    requests = [
        {"command": "step"},
        {"command": "status"},
    ]

    class Connection:
        def __init__(self, request: Mapping[str, Any]) -> None:
            self.request = request

        def __enter__(self) -> "Connection":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def makefile(self, mode: str) -> io.BytesIO:
            assert mode == "rb"
            return io.BytesIO(json.dumps(self.request).encode("utf-8") + b"\n")

    class FakeSocket:
        def __init__(self) -> None:
            self.connections = [Connection(request) for request in requests]

        def bind(self, path: str) -> None:
            Path(path).touch()

        def listen(self, count: int) -> None:
            assert count == 1

        def accept(self) -> tuple[Connection, None]:
            return self.connections.pop(0), None

        def close(self) -> None:
            return None

    fake_socket = FakeSocket()
    boundary["socket"] = SimpleNamespace(
        AF_UNIX=1,
        SOCK_STREAM=1,
        socket=lambda *args: fake_socket,
    )

    def disconnect_once(connection: Connection, payload: bytes) -> bool:
        sent.append(json.loads(payload))
        return len(sent) != 1

    boundary["send_response_or_client_gone"] = disconnect_once
    processor = Processor()
    server = boundary["UnixSceneServer"](
        processor=processor,
        socket_path=tmp_path / "control.sock",
        ready_path=tmp_path / "ready.json",
        run_dir=tmp_path / "simulator" / "run",
        runtime_root=tmp_path,
    )
    result = server.serve()

    assert processor.calls == 2
    assert result["status"] == "PASS"
    assert result["commands_processed"] == 1
    assert sent[0]["error_type"] == "INTERNAL_ENV_ERROR"
    assert sent[0]["message"] == "INTERNAL_ENV_ERROR"
    assert sent[1]["result"] == {"alive": True}
    private = json.loads(
        (tmp_path / "private" / "scene_server_errors.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert private["request_operation"] == "step"
    assert "phase=during_restricted" in private["message"]
    assert "private-value" not in private["message"]
