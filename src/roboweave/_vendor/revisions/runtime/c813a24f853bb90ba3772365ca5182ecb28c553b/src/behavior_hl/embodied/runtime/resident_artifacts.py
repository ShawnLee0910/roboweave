"""Crash-resilient artifact writer for one resident simulator episode.

Every artifact is replaced atomically and fsynced before it is entered into
the recovery inventory.  Public payloads pass a fail-closed leakage guard.
The writer never reads simulator state; callers provide public observations
and private verifier reports through separate methods.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from behavior_hl.embodied.runtime.artifact_contract import (
    EpisodeArtifactContract,
)
from behavior_hl.embodied.runtime.artifacts import (
    ArtifactInventory,
    ArtifactRecord,
    ArtifactVisibility,
)


_SECRET_TEXT = re.compile(
    r"(?:\bBearer\s+\S+|\bsk-[A-Za-z0-9_-]{12,})",
    flags=re.IGNORECASE,
)
_SECRET_BYTES = re.compile(
    rb"(?:\bBearer\s+\S+|\bsk-[A-Za-z0-9_-]{12,})",
    flags=re.IGNORECASE,
)
_PRIVATE_PUBLIC_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "bddl",
    "containment_truth",
    "goal_truth",
    "object_scope",
    "official_q",
    "official_success",
    "private_verifier",
    "seg_instance",
    "secret",
    "success_truth",
    "world_pose",
)


class ResidentArtifactError(RuntimeError):
    """Base error for resident artifact persistence."""


class ArtifactCompletenessError(ResidentArtifactError):
    """Raised when finalization would certify an incomplete trajectory."""


class PublicArtifactLeakError(ResidentArtifactError):
    """Raised before a sensitive value can enter a public artifact."""


class ArtifactIntegrityError(ResidentArtifactError):
    """Raised when a recovered file no longer matches its recorded hash."""


@dataclass(frozen=True, slots=True)
class ResidentArtifactSummary:
    run_id: str
    status: str
    action_count: int
    manifest_path: str | None
    manifest_sha256: str | None


class ResidentEpisodeArtifactWriter:
    """Incrementally preserve one expensive resident-scene rollout."""

    _STATE_PATH = "system/resident_state.json"

    def __init__(
        self,
        run_dir: Path,
        *,
        run_id: str,
        metadata: Mapping[str, Any] | None = None,
        sensitive_values: Sequence[str] = (),
    ) -> None:
        resolved = Path(run_dir).resolve()
        if resolved == Path(resolved.anchor):
            raise ValueError("run_dir cannot be a filesystem root")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must not be empty")
        self.run_dir = resolved
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id.strip()
        self._sensitive_values = tuple(
            value
            for value in sensitive_values
            if isinstance(value, str) and value
        )
        self._records: dict[str, ArtifactRecord] = {}
        self._steps: dict[int, set[str]] = {}
        self._raw_responses: list[Any] = []
        self._parsed_actions: list[Any] = []
        self._events: list[dict[str, Any]] = []
        self._metadata = dict(metadata or {})
        self._status = "RUNNING"
        self._manifest_path: str | None = None
        self._manifest_sha256: str | None = None
        self._closed = False
        self._guard_public(self._metadata, path="metadata")

        state_path = self.run_dir / self._STATE_PATH
        if state_path.exists():
            self._recover(state_path)
            if self.run_id != run_id.strip():
                raise ResidentArtifactError("run_id does not match recovered state")
            return

        for directory in ("public", "private", "video", "system"):
            (self.run_dir / directory).mkdir(parents=True, exist_ok=True)
        self._write_json(
            logical_name="events",
            relative_path="system/events.json",
            visibility=ArtifactVisibility.SYSTEM,
            payload=self._events,
        )
        self._write_json(
            logical_name="model_raw_responses",
            relative_path="public/model/raw_responses.json",
            visibility=ArtifactVisibility.PUBLIC,
            payload=self._raw_responses,
        )
        self._write_json(
            logical_name="parsed_actions",
            relative_path="public/model/parsed_actions.json",
            visibility=ArtifactVisibility.PUBLIC,
            payload=self._parsed_actions,
        )
        self._append_event("episode_writer_started")

    @property
    def inventory(self) -> ArtifactInventory:
        return ArtifactInventory(self._records.values())

    @property
    def status(self) -> str:
        return self._status

    @property
    def action_count(self) -> int:
        return max(self._steps, default=-1) + 1

    def __enter__(self) -> "ResidentEpisodeArtifactWriter":
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is not None and not self._closed:
            self.record_failure(
                failure_type=exc_type.__name__ if exc_type else "Exception",
                message=str(exc),
            )
        return False

    def record_initial_observation(self, payload: Any) -> ArtifactRecord:
        return self._write_public_payload(
            "initial_frame",
            "public/observations/initial",
            payload,
        )

    def record_initial_predicates(self, report: Any) -> ArtifactRecord:
        return self._write_json(
            logical_name="initial_predicates",
            relative_path="private/verifier/initial.json",
            visibility=ArtifactVisibility.PRIVATE,
            payload=report,
            immutable=True,
        )

    def record_model_exchange(
        self,
        *,
        action_index: int,
        model_input: Any,
        raw_response: Any,
        parsed_action: Any,
    ) -> None:
        self._require_step_index(action_index, create=True)
        self._guard_public(model_input, path="model_input")
        self._guard_public(raw_response, path="raw_response")
        self._guard_public(parsed_action, path="parsed_action")
        self._write_public_payload(
            f"step_{action_index:04d}_model_input",
            f"public/steps/{action_index:04d}/model_input",
            model_input,
            immutable=True,
        )
        self._mark_step(action_index, "model_input")

        self._write_public_payload(
            f"step_{action_index:04d}_model_raw_response",
            f"public/steps/{action_index:04d}/model_raw_response",
            raw_response,
            immutable=True,
        )
        self._write_public_payload(
            f"step_{action_index:04d}_parsed_action",
            f"public/steps/{action_index:04d}/parsed_action",
            parsed_action,
            immutable=True,
        )
        if len(self._raw_responses) > action_index:
            if (
                self._raw_responses[action_index] == raw_response
                and self._parsed_actions[action_index] == parsed_action
            ):
                self._mark_step(action_index, "model_exchange")
                return
            raise ResidentArtifactError(
                "recovered model exchange payload does not match"
            )
        if len(self._raw_responses) != action_index:
            raise ResidentArtifactError(
                "model exchanges must be recorded in contiguous order"
            )
        next_raw = [*self._raw_responses, raw_response]
        next_parsed = [*self._parsed_actions, parsed_action]
        self._raw_responses = next_raw
        self._parsed_actions = next_parsed
        self._write_json(
            logical_name="model_raw_responses",
            relative_path="public/model/raw_responses.json",
            visibility=ArtifactVisibility.PUBLIC,
            payload=next_raw,
        )
        self._write_json(
            logical_name="parsed_actions",
            relative_path="public/model/parsed_actions.json",
            visibility=ArtifactVisibility.PUBLIC,
            payload=next_parsed,
        )
        self._mark_step(action_index, "model_exchange")
        self._append_event("model_exchange_recorded", action_index)

    def record_model_attempt(
        self,
        *,
        sequence: int,
        model_input: Any,
        raw_response: Any,
    ) -> ArtifactRecord:
        """Crash-safely preserve one model attempt before strict parsing.

        An attempt is not an accepted action: it does not create a step,
        increment ``action_count``, or enter accepted raw / parsed histories.
        If parsing later succeeds, :meth:`record_model_exchange` records the
        normal accepted pair separately.
        """

        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
        ):
            raise ValueError("model attempt sequence must be non-negative")
        if sequence != self.action_count:
            raise ResidentArtifactError(
                "model attempt sequence must equal the next action index"
            )
        self._guard_public(model_input, path="model_attempt.model_input")
        self._guard_public(
            raw_response,
            path="model_attempt.raw_response",
        )
        return self._write_json(
            logical_name=f"model_attempt_{sequence:04d}",
            relative_path=f"public/model/attempts/{sequence:04d}.json",
            visibility=ArtifactVisibility.PUBLIC,
            payload={
                "sequence": sequence,
                "model_input": model_input,
                "raw_response": raw_response,
            },
            immutable=True,
        )

    def record_action_before(self, action_index: int, payload: Any) -> ArtifactRecord:
        return self._record_step_public_snapshot(
            action_index, "action_before", payload
        )

    def record_action_after(self, action_index: int, payload: Any) -> ArtifactRecord:
        return self._record_step_public_snapshot(
            action_index, "action_after", payload
        )

    def record_settle_final(self, action_index: int, payload: Any) -> ArtifactRecord:
        return self._record_step_public_snapshot(
            action_index, "settle_final", payload
        )

    def record_private_verifier(
        self,
        action_index: int,
        report: Any,
    ) -> ArtifactRecord:
        self._require_step_index(action_index)
        record = self._write_json(
            logical_name=f"step_{action_index:04d}_verifier",
            relative_path=f"private/steps/{action_index:04d}/verifier.json",
            visibility=ArtifactVisibility.PRIVATE,
            payload=report,
            immutable=True,
        )
        self._mark_step(action_index, "verifier")
        return record

    def record_final_observation(self, payload: Any) -> ArtifactRecord:
        return self._write_public_payload(
            "final_state",
            "public/observations/final",
            payload,
            immutable=True,
        )

    def record_final_predicate_report(self, report: Any) -> ArtifactRecord:
        return self._write_json(
            logical_name="final_predicate_report",
            relative_path="private/verifier/final.json",
            visibility=ArtifactVisibility.PRIVATE,
            payload=report,
            immutable=True,
        )

    def register_episode_video(
        self,
        relative_path: str = "video/episode.mp4",
    ) -> ArtifactRecord:
        self._require_open()
        target = self._validated_target(
            "episode_video",
            relative_path,
            ArtifactVisibility.MEDIA,
        )
        if not target.is_file():
            raise FileNotFoundError(target)
        with target.open("rb") as stream:
            os.fsync(stream.fileno())
        _fsync_directory(target.parent)
        record = self._record_file(
            "episode_video",
            relative_path,
            ArtifactVisibility.MEDIA,
            target,
        )
        self._persist_state()
        return record

    def record_failure(self, *, failure_type: str, message: str) -> None:
        self._require_open()
        safe_type = _bounded_text(failure_type, "failure_type", 200)
        safe_message = self._redact_text(
            _bounded_text(message, "message", 4_000)
        )
        self._status = "INTERRUPTED"
        self._write_json(
            logical_name="failure",
            relative_path="system/failure.json",
            visibility=ArtifactVisibility.SYSTEM,
            payload={
                "failure_type": safe_type,
                "message": safe_message,
                "preserved_action_count": self.action_count,
            },
        )
        self._append_event("episode_interrupted")

    def verify_integrity(self) -> None:
        for record in self._records.values():
            target = self.run_dir / record.relative_path
            if not target.is_file():
                raise ArtifactIntegrityError(
                    f"artifact is missing: {record.relative_path}"
                )
            size, digest = _file_identity(target)
            if size != record.byte_size or digest != record.sha256:
                raise ArtifactIntegrityError(
                    f"artifact hash mismatch: {record.relative_path}"
                )

    def finalize(self, *, status: str) -> ResidentArtifactSummary:
        self._require_open()
        terminal_status = _bounded_text(status, "status", 100)
        self.verify_integrity()
        if "manifest" in self._records:
            manifest = self._records["manifest"]
            payload = json.loads(
                (self.run_dir / manifest.relative_path).read_text(
                    encoding="utf-8"
                )
            )
            if payload.get("status") != terminal_status:
                raise ResidentArtifactError(
                    "recovered manifest status does not match finalize status"
                )
            EpisodeArtifactContract(
                self.action_count
            ).require_complete(self.inventory)
            self._status = terminal_status
            self._manifest_path = manifest.relative_path
            self._manifest_sha256 = manifest.sha256
            self._closed = True
            self._persist_state()
            return ResidentArtifactSummary(
                run_id=self.run_id,
                status=self._status,
                action_count=self.action_count,
                manifest_path=self._manifest_path,
                manifest_sha256=self._manifest_sha256,
            )
        missing_step_fields = {
            index: sorted(
                {
                    "model_input",
                    "model_exchange",
                    "action_before",
                    "action_after",
                    "settle_final",
                    "verifier",
                }.difference(fields)
            )
            for index, fields in self._steps.items()
            if {
                "model_input",
                "model_exchange",
                "action_before",
                "action_after",
                "settle_final",
                "verifier",
            }.difference(fields)
        }
        if missing_step_fields:
            self._append_event("finalize_rejected_incomplete_steps")
            raise ArtifactCompletenessError(
                f"incomplete action artifacts: {missing_step_fields}"
            )
        if len(self._raw_responses) != self.action_count or len(
            self._parsed_actions
        ) != self.action_count:
            self._append_event("finalize_rejected_model_history")
            raise ArtifactCompletenessError(
                "model histories do not match action_count"
            )

        contract = EpisodeArtifactContract(self.action_count)
        required_before_manifest = set(contract.required_logical_names)
        required_before_manifest.remove("manifest")
        missing = self.inventory.missing(required_before_manifest)
        if missing:
            self._append_event("finalize_rejected_missing_artifacts")
            raise ArtifactCompletenessError(
                f"missing required artifacts: {list(missing)}"
            )

        self._status = terminal_status
        manifest_payload = {
            "schema_version": "resident-episode-artifacts-v1",
            "run_id": self.run_id,
            "status": terminal_status,
            "action_count": self.action_count,
            "metadata": self._metadata,
            "artifacts": self.inventory.to_manifest(),
        }
        manifest = self._write_json(
            logical_name="manifest",
            relative_path="system/manifest.json",
            visibility=ArtifactVisibility.SYSTEM,
            payload=manifest_payload,
            immutable=True,
        )
        contract.require_complete(self.inventory)
        self.verify_integrity()
        self._manifest_path = manifest.relative_path
        self._manifest_sha256 = manifest.sha256
        self._closed = True
        self._persist_state()
        return ResidentArtifactSummary(
            run_id=self.run_id,
            status=self._status,
            action_count=self.action_count,
            manifest_path=self._manifest_path,
            manifest_sha256=self._manifest_sha256,
        )

    def _record_step_public_snapshot(
        self,
        action_index: int,
        field: str,
        payload: Any,
    ) -> ArtifactRecord:
        self._require_step_index(action_index)
        record = self._write_public_payload(
            f"step_{action_index:04d}_{field}",
            f"public/steps/{action_index:04d}/{field}",
            payload,
            immutable=True,
        )
        self._mark_step(action_index, field)
        return record

    def _write_public_payload(
        self,
        logical_name: str,
        path_without_suffix: str,
        payload: Any,
        *,
        immutable: bool = True,
    ) -> ArtifactRecord:
        self._guard_public(payload, path=logical_name)
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return self._write_bytes(
                logical_name=logical_name,
                relative_path=f"{path_without_suffix}.bin",
                visibility=ArtifactVisibility.PUBLIC,
                content=bytes(payload),
                immutable=immutable,
            )
        return self._write_json(
            logical_name=logical_name,
            relative_path=f"{path_without_suffix}.json",
            visibility=ArtifactVisibility.PUBLIC,
            payload=payload,
            immutable=immutable,
        )

    def _write_json(
        self,
        *,
        logical_name: str,
        relative_path: str,
        visibility: ArtifactVisibility,
        payload: Any,
        immutable: bool = False,
    ) -> ArtifactRecord:
        try:
            content = (
                json.dumps(
                    payload,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ResidentArtifactError(
                f"{logical_name} is not strict JSON"
            ) from exc
        return self._write_bytes(
            logical_name=logical_name,
            relative_path=relative_path,
            visibility=visibility,
            content=content,
            immutable=immutable,
        )

    def _write_bytes(
        self,
        *,
        logical_name: str,
        relative_path: str,
        visibility: ArtifactVisibility,
        content: bytes,
        immutable: bool = False,
    ) -> ArtifactRecord:
        self._require_open()
        target = self._validated_target(
            logical_name,
            relative_path,
            visibility,
        )
        if immutable and logical_name in self._records:
            existing = self._records[logical_name]
            if (
                existing.relative_path == relative_path
                and existing.byte_size == len(content)
                and existing.sha256
                == hashlib.sha256(content).hexdigest()
            ):
                return existing
            raise ResidentArtifactError(
                f"immutable artifact already exists: {logical_name}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(
            f".{target.name}.tmp-{os.getpid()}"
        )
        try:
            with temporary.open("wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            _fsync_directory(target.parent)
        finally:
            if temporary.exists():
                temporary.unlink()
        record = self._record_file(
            logical_name,
            relative_path,
            visibility,
            target,
        )
        if relative_path != self._STATE_PATH:
            self._persist_state()
        return record

    def _record_file(
        self,
        logical_name: str,
        relative_path: str,
        visibility: ArtifactVisibility,
        target: Path,
    ) -> ArtifactRecord:
        size, digest = _file_identity(target)
        record = ArtifactRecord(
            logical_name=logical_name,
            relative_path=relative_path,
            visibility=visibility,
            byte_size=size,
            sha256=digest,
        )
        self._records[logical_name] = record
        return record

    def _validated_target(
        self,
        logical_name: str,
        relative_path: str,
        visibility: ArtifactVisibility,
    ) -> Path:
        ArtifactRecord(logical_name, relative_path, visibility)
        expected = {
            ArtifactVisibility.PUBLIC: "public",
            ArtifactVisibility.PRIVATE: "private",
            ArtifactVisibility.MEDIA: "video",
            ArtifactVisibility.SYSTEM: "system",
        }[visibility]
        path = PurePosixPath(relative_path)
        if not path.parts or path.parts[0] != expected:
            raise ValueError(
                f"{visibility.value} artifact must use {expected}/"
            )
        target = (self.run_dir / Path(*path.parts)).resolve()
        if self.run_dir not in target.parents:
            raise ValueError("artifact path escaped the run directory")
        existing = self._records.get(logical_name)
        if existing and existing.relative_path != relative_path:
            raise ResidentArtifactError(
                "logical artifact name cannot change relative_path"
            )
        for name, record in self._records.items():
            if name != logical_name and record.relative_path == relative_path:
                raise ResidentArtifactError(
                    "artifact relative_path already belongs to another name"
                )
        return target

    def _require_step_index(self, action_index: int, *, create: bool = False) -> None:
        if (
            isinstance(action_index, bool)
            or not isinstance(action_index, int)
            or action_index < 0
        ):
            raise ValueError("action_index must be non-negative")
        if create:
            if action_index in self._steps:
                return
            if action_index != self.action_count:
                raise ResidentArtifactError(
                    "action indices must be contiguous"
                )
            self._steps[action_index] = set()
            self._persist_state()
        elif action_index not in self._steps:
            raise ResidentArtifactError(
                "record_model_exchange must create the action first"
            )

    def _mark_step(self, action_index: int, field: str) -> None:
        self._steps[action_index].add(field)
        self._persist_state()

    def _append_event(
        self,
        event: str,
        action_index: int | None = None,
    ) -> None:
        item: dict[str, Any] = {
            "sequence": len(self._events),
            "event": event,
        }
        if action_index is not None:
            item["action_index"] = action_index
        next_events = [*self._events, item]
        self._write_json(
            logical_name="events",
            relative_path="system/events.json",
            visibility=ArtifactVisibility.SYSTEM,
            payload=next_events,
        )
        self._events = next_events
        self._persist_state()

    def _guard_public(self, payload: Any, *, path: str) -> None:
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                if not isinstance(key, str):
                    raise PublicArtifactLeakError(
                        f"{path} has a non-string public key"
                    )
                folded = key.casefold()
                if any(
                    fragment in folded
                    for fragment in _PRIVATE_PUBLIC_KEY_FRAGMENTS
                ):
                    raise PublicArtifactLeakError(
                        f"{path}.{key} is private or sensitive"
                    )
                self._guard_public(value, path=f"{path}.{key}")
            return
        if isinstance(payload, (tuple, list)):
            for index, value in enumerate(payload):
                self._guard_public(value, path=f"{path}[{index}]")
            return
        if isinstance(payload, str):
            if _SECRET_TEXT.search(payload) or any(
                secret in payload for secret in self._sensitive_values
            ):
                raise PublicArtifactLeakError(
                    f"{path} contains credential-like text"
                )
            return
        if isinstance(payload, (bytes, bytearray, memoryview)):
            raw = bytes(payload)
            if _SECRET_BYTES.search(raw):
                raise PublicArtifactLeakError(
                    f"{path} contains credential-like bytes"
                )
            for secret in self._sensitive_values:
                if secret.encode("utf-8") in raw:
                    raise PublicArtifactLeakError(
                        f"{path} contains a configured sensitive value"
                    )

    def _redact_text(self, text: str) -> str:
        redacted = _SECRET_TEXT.sub("[REDACTED]", text)
        for secret in self._sensitive_values:
            redacted = redacted.replace(secret, "[REDACTED]")
        return redacted

    def _persist_state(self) -> None:
        payload = {
            "schema_version": "resident-writer-state-v1",
            "run_id": self.run_id,
            "status": self._status,
            "metadata": self._metadata,
            "records": [
                record.to_dict() for record in self.inventory.records
            ],
            "steps": {
                str(index): sorted(fields)
                for index, fields in sorted(self._steps.items())
            },
            "raw_responses": self._raw_responses,
            "parsed_actions": self._parsed_actions,
            "events": self._events,
            "manifest_path": self._manifest_path,
            "manifest_sha256": self._manifest_sha256,
            "closed": self._closed,
        }
        target = self.run_dir / self._STATE_PATH
        content = (
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        try:
            with temporary.open("wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            _fsync_directory(target.parent)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _recover(self, state_path: Path) -> None:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(
                "resident recovery state is unreadable"
            ) from exc
        if state.get("schema_version") != "resident-writer-state-v1":
            raise ArtifactIntegrityError("unknown resident recovery schema")
        recovered_run_id = state.get("run_id")
        if recovered_run_id != self.run_id:
            raise ResidentArtifactError("run_id does not match recovered state")
        records = []
        for item in state.get("records", []):
            records.append(
                ArtifactRecord(
                    logical_name=item["logical_name"],
                    relative_path=item["relative_path"],
                    visibility=ArtifactVisibility(item["visibility"]),
                    byte_size=item["byte_size"],
                    sha256=item["sha256"],
                )
            )
        inventory = ArtifactInventory(records)
        self._records = {
            record.logical_name: record for record in inventory.records
        }
        self._steps = {
            int(index): set(fields)
            for index, fields in state.get("steps", {}).items()
        }
        self._raw_responses = list(state.get("raw_responses", []))
        self._parsed_actions = list(state.get("parsed_actions", []))
        self._events = list(state.get("events", []))
        self._metadata = dict(state.get("metadata", {}))
        self._status = str(state.get("status", "INTERRUPTED"))
        self._manifest_path = state.get("manifest_path")
        self._manifest_sha256 = state.get("manifest_sha256")
        self._closed = bool(state.get("closed", False))
        self.verify_integrity()
        self._reconcile_recovered_steps()

    def _reconcile_recovered_steps(self) -> None:
        """Repair state/aggregates from immutable per-step commit files."""

        patterns = {
            "model_input": re.compile(r"^step_(\d{4})_model_input$"),
            "model_raw_response": re.compile(
                r"^step_(\d{4})_model_raw_response$"
            ),
            "parsed_action": re.compile(
                r"^step_(\d{4})_parsed_action$"
            ),
            "action_before": re.compile(
                r"^step_(\d{4})_action_before$"
            ),
            "action_after": re.compile(
                r"^step_(\d{4})_action_after$"
            ),
            "settle_final": re.compile(
                r"^step_(\d{4})_settle_final$"
            ),
            "verifier": re.compile(r"^step_(\d{4})_verifier$"),
        }
        committed_raw: dict[int, Any] = {}
        committed_parsed: dict[int, Any] = {}
        for name, record in self._records.items():
            for field, pattern in patterns.items():
                match = pattern.fullmatch(name)
                if match is None:
                    continue
                index = int(match.group(1))
                self._steps.setdefault(index, set())
                if field == "model_raw_response":
                    committed_raw[index] = self._read_json_record(record)
                elif field == "parsed_action":
                    committed_parsed[index] = self._read_json_record(record)
                else:
                    self._steps[index].add(field)
                break
        complete_indices = sorted(
            set(committed_raw).intersection(committed_parsed)
        )
        if complete_indices and complete_indices != list(
            range(len(complete_indices))
        ):
            raise ArtifactIntegrityError(
                "recovered model exchanges are not contiguous"
            )
        for index in complete_indices:
            self._steps[index].add("model_exchange")
        rebuilt_raw = [committed_raw[index] for index in complete_indices]
        rebuilt_parsed = [
            committed_parsed[index] for index in complete_indices
        ]
        if self._closed:
            if (
                rebuilt_raw != self._raw_responses
                or rebuilt_parsed != self._parsed_actions
            ):
                raise ArtifactIntegrityError(
                    "closed model histories disagree with step commits"
                )
            return
        self._raw_responses = rebuilt_raw
        self._parsed_actions = rebuilt_parsed
        self._write_json(
            logical_name="model_raw_responses",
            relative_path="public/model/raw_responses.json",
            visibility=ArtifactVisibility.PUBLIC,
            payload=rebuilt_raw,
        )
        self._write_json(
            logical_name="parsed_actions",
            relative_path="public/model/parsed_actions.json",
            visibility=ArtifactVisibility.PUBLIC,
            payload=rebuilt_parsed,
        )
        self._persist_state()

    def _read_json_record(self, record: ArtifactRecord) -> Any:
        path = self.run_dir / record.relative_path
        if path.suffix != ".json":
            raise ArtifactIntegrityError(
                f"model commit is not JSON: {record.relative_path}"
            )
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(
                f"model commit is unreadable: {record.relative_path}"
            ) from exc

    def _require_open(self) -> None:
        if self._closed:
            raise ResidentArtifactError("resident artifact writer is closed")


def _bounded_text(value: str, field: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must not be empty")
    text = value.strip()
    if len(text) > limit:
        text = text[:limit]
    return text


def _file_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
