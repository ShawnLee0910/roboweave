"""Isolated TokenPlan Qwen model adapter for one resident episode."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping

import fcntl

from behavior_hl.embodied.runtime.model_visual_history import (
    MODEL_SENT_FRAME_MAX,
)
from behavior_hl.embodied.runtime.observation import PublicRobotObservation
from behavior_hl.project_identity import (
    PROJECT_ROOT,
    capture_project_identity,
)


TOKENPLAN_RESIDENT_MODEL = os.environ.get(
    "BEHAVIOR_HL_TOKENPLAN_MODEL",
    "qwen3.8-max-preview",
).strip()
if not TOKENPLAN_RESIDENT_MODEL:
    raise RuntimeError("BEHAVIOR_HL_TOKENPLAN_MODEL must not be empty")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
DEFAULT_CHILD_SCRIPT = PROJECT_ROOT / "scripts" / (
    "qoderclicn_resident_decision.py"
    if os.environ.get("BEHAVIOR_HL_MODEL_BACKEND") == "qoderclicn"
    else "tokenplan_resident_decision.py"
)


class TokenPlanResidentModelError(RuntimeError):
    """Fail-closed model-process or response-contract failure."""


def _acquire_global_api_slot(
    *,
    maximum_wait_seconds: float = 300.0,
) -> tuple[Any | None, str | None]:
    root_value = os.environ.get("BEHAVIOR_HL_TOKENPLAN_GLOBAL_LOCK", "").strip()
    if not root_value:
        return None, None
    raw_slots = os.environ.get("BEHAVIOR_HL_TOKENPLAN_GLOBAL_SLOTS", "2")
    try:
        slot_count = int(raw_slots)
    except ValueError as exc:
        raise TokenPlanResidentModelError("global API slot count is invalid") from exc
    if not 1 <= slot_count <= 8:
        raise TokenPlanResidentModelError("global API slot count must be in [1, 8]")
    root = Path(root_value).resolve()
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + maximum_wait_seconds
    while True:
        for index in range(slot_count):
            path = root / f"slot_{index:02d}.lock"
            stream = path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
                continue
            stream.seek(0)
            stream.truncate()
            stream.write(f"pid={os.getpid()} acquired_at={time.time()}\n")
            stream.flush()
            return stream, str(path)
        if time.monotonic() >= deadline:
            raise TokenPlanResidentModelError(
                "global API slot acquisition timed out"
            )
        time.sleep(0.2)


class TokenPlanResidentModel:
    """Call Qwen in an isolated Python process using the exact public PNG."""

    def __init__(
        self,
        *,
        frame_artifact_provider: Callable[
            [PublicRobotObservation], Mapping[str, Any]
        ],
        policy_dir: Path,
        python_executable: Path = Path(
            os.environ.get("ESI_MODEL_PYTHON", sys.executable)
        ),
        child_script: Path = DEFAULT_CHILD_SCRIPT,
        wall_timeout_seconds: int = 90,
        http_timeout_seconds: int | None = None,
        maximum_attempts: int = 3,
    ) -> None:
        if not callable(frame_artifact_provider):
            raise TypeError("frame_artifact_provider must be callable")
        if (
            isinstance(wall_timeout_seconds, bool)
            or not isinstance(wall_timeout_seconds, int)
            or wall_timeout_seconds <= 0
        ):
            raise ValueError("wall_timeout_seconds must be positive")
        if (
            isinstance(maximum_attempts, bool)
            or not isinstance(maximum_attempts, int)
            or maximum_attempts < 1
            or maximum_attempts > 5
        ):
            raise ValueError("maximum_attempts must be in [1, 5]")
        if http_timeout_seconds is not None and (
            isinstance(http_timeout_seconds, bool)
            or not isinstance(http_timeout_seconds, int)
            or http_timeout_seconds <= 0
        ):
            raise ValueError("http_timeout_seconds must be positive")
        self._frame_artifact_provider = frame_artifact_provider
        self.policy_dir = Path(policy_dir).resolve()
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        self.python_executable = Path(python_executable).resolve()
        self.child_script = Path(child_script).resolve()
        capture_project_identity(
            expected_root=PROJECT_ROOT,
            module_files={
                "tokenplan_resident_model": Path(__file__),
                "tokenplan_resident_child": self.child_script,
            },
        )
        self.wall_timeout_seconds = wall_timeout_seconds
        self.http_timeout_seconds = http_timeout_seconds
        self.maximum_attempts = maximum_attempts
        self._call_index = 0

    def complete(
        self,
        prompt: str,
        observation: PublicRobotObservation,
    ) -> str:
        return self.complete_with_visual_history(prompt, (observation,))

    def complete_with_visual_history(
        self,
        prompt: str,
        observations: tuple[PublicRobotObservation, ...],
    ) -> str:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be non-empty")
        if (
            not isinstance(observations, tuple)
            or not observations
            or len(observations) > MODEL_SENT_FRAME_MAX
            or any(
                not isinstance(item, PublicRobotObservation)
                for item in observations
            )
        ):
            raise TypeError(
                "observations must be a tuple of one to "
                f"{MODEL_SENT_FRAME_MAX} public frames"
            )
        pngs: list[bytes] = []
        png_hashes: list[str] = []
        for observation in observations:
            artifact = self._frame_artifact_provider(observation)
            if not isinstance(artifact, Mapping):
                raise TokenPlanResidentModelError(
                    "frame artifact provider returned a non-mapping"
                )
            png = artifact.get("png_bytes")
            png_sha256 = artifact.get("png_sha256")
            if not isinstance(png, (bytes, bytearray, memoryview)):
                raise TokenPlanResidentModelError(
                    "frame artifact has no encoded PNG bytes"
                )
            png = bytes(png)
            actual_png_hash = hashlib.sha256(png).hexdigest()
            if (
                not png.startswith(_PNG_SIGNATURE)
                or png_sha256 != actual_png_hash
            ):
                raise TokenPlanResidentModelError(
                    "frame artifact PNG identity is invalid"
                )
            pngs.append(png)
            png_hashes.append(actual_png_hash)

        self._call_index += 1
        call_index = self._call_index
        call_dir = self.policy_dir / f"call_{call_index:04d}"
        call_dir.mkdir(parents=True, exist_ok=False)
        frame_paths = [
            call_dir / f"model_input_{index:02d}.png"
            for index in range(len(pngs))
        ]
        request_path = call_dir / "request.json"
        for frame_path, png in zip(frame_paths, pngs, strict=True):
            frame_path.write_bytes(png)
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        request = {
            "schema_version": "behavior-hl-tokenplan-resident-request-v2",
            "model": TOKENPLAN_RESIDENT_MODEL,
            "call_index": call_index,
            "prompt": prompt,
            "prompt_sha256": prompt_sha256,
            "png_paths": [str(path) for path in frame_paths],
            "png_sha256s": png_hashes,
            "project_root": str(PROJECT_ROOT),
        }
        _write_json(request_path, request)

        environment = os.environ.copy()
        environment.pop("LD_PRELOAD", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["BEHAVIOR_HL_PROJECT_ROOT"] = str(PROJECT_ROOT)
        if self.http_timeout_seconds is not None:
            environment["BEHAVIOR_HL_TOKENPLAN_HTTP_TIMEOUT_SECONDS"] = str(
                self.http_timeout_seconds
            )
        expected = {
            "model": TOKENPLAN_RESIDENT_MODEL,
            "call_index": call_index,
            "prompt_sha256": prompt_sha256,
            "png_sha256s": png_hashes,
            "project_root": str(PROJECT_ROOT),
        }
        attempts: list[dict[str, Any]] = []
        last_error: BaseException | None = None
        api_slot_stream, api_slot_path = _acquire_global_api_slot()
        for attempt_index in range(1, self.maximum_attempts + 1):
            attempt_dir = call_dir if attempt_index == 1 else (
                call_dir / f"attempt_{attempt_index:02d}"
            )
            attempt_dir.mkdir(parents=True, exist_ok=attempt_index == 1)
            response_path = attempt_dir / "response.json"
            process_path = attempt_dir / "process.json"
            command = [
                str(self.python_executable),
                str(self.child_script),
                "--request",
                str(request_path),
                "--output",
                str(response_path),
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.child_script.parent.parent,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=self.wall_timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                last_error = exc
                attempt = {
                    "attempt": attempt_index,
                    "status": "timeout",
                    "timeout_seconds": self.wall_timeout_seconds,
                }
                attempts.append(attempt)
                _write_json(process_path, attempt)
                continue
            process = {
                "attempt": attempt_index,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            }
            _write_json(process_path, process)
            if not response_path.is_file():
                attempts.append({**process, "status": "missing_response"})
                continue
            response = json.loads(response_path.read_text(encoding="utf-8"))
            for field, value in expected.items():
                if response.get(field) != value:
                    raise TokenPlanResidentModelError(
                        f"TokenPlan response changed {field}"
                    )
            if completed.returncode != 0 or response.get("status") == (
                "infrastructure_error"
            ):
                attempts.append(
                    {
                        **process,
                        "status": "infrastructure_error",
                        "error_type": response.get("error_type", "unknown"),
                    }
                )
                continue
            raw = response.get("raw_response")
            if not isinstance(raw, str) or not raw.strip():
                raise TokenPlanResidentModelError(
                    "TokenPlan response has no raw model text"
                )
            if response.get("status") == "valid_json_object":
                original_reasoning = response.get("original_reasoning")
                if (
                    not isinstance(original_reasoning, str)
                    or not original_reasoning.strip()
                ):
                    raise TokenPlanResidentModelError(
                        "TokenPlan valid response lost original reasoning"
                    )
            for field in (
                "original_reasoning",
                "provider_reasoning_content",
                "provider_reasoning",
            ):
                if field not in response:
                    raise TokenPlanResidentModelError(
                        f"TokenPlan response omitted {field}"
                    )
            attempts.append({**process, "status": "success"})
            if api_slot_stream is not None:
                api_slot_stream.close()
            _write_json(
                call_dir / "retry_summary.json",
                {
                    "maximum_attempts": self.maximum_attempts,
                    "attempts": attempts,
                    "successful_attempt": attempt_index,
                    "global_api_slot": api_slot_path,
                },
            )
            return raw
        _write_json(
            call_dir / "retry_summary.json",
            {
                "maximum_attempts": self.maximum_attempts,
                "attempts": attempts,
                "successful_attempt": None,
                "global_api_slot": api_slot_path,
            },
        )
        if api_slot_stream is not None:
            api_slot_stream.close()
        message = "TokenPlan resident model exhausted infrastructure retries"
        raise TokenPlanResidentModelError(message) from last_error


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
