"""Frozen official BDDL actual-index mappings for large goal tables.

The CPU lazy grounder has its own deterministic raw-index space.  It is useful
for finding a canonical semantic witness, but that raw index is not an index
into ``task.ground_goal_state_options``.  This module is the only bridge from a
canonical semantic identity to an actual official index.  Mappings are
authored in a fresh interpreter using the official BDDL implementation and a
fixed interpreter-start ``PYTHONHASHSEED``.

Runtime lookup is deliberately O(1): validate frozen source / environment
provenance, read one configured index, and identity-attest that one official
option.  Runtime code never scans the live option table.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence


OFFICIAL_ACTUAL_OPTION_MAPPING_VERSION = (
    "behavior2026.official-actual-option-mapping/v1"
)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
OFFICIAL_BEHAVIOR_ROOT = PROJECT_ROOT.parent / "BEHAVIOR-1K-2026"
DEFAULT_MAPPING_PATH = (
    PROJECT_ROOT / "configs" / "b100_official_actual_option_mappings.json"
)


class OfficialActualOptionMappingError(RuntimeError):
    """A frozen actual-index mapping is absent, invalid, or drifted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def semantic_identity_sha256(identity: Sequence[str]) -> str:
    encoded = json.dumps(
        list(identity),
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture_python_ordering_environment() -> dict[str, str]:
    """Capture interpreter properties that can affect official set order."""

    return {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "cache_tag": str(sys.implementation.cache_tag),
        "executable": str(Path(sys.executable).resolve()),
        "python_hash_seed": str(os.environ.get("PYTHONHASHSEED", "")),
        "hash_algorithm": str(sys.hash_info.algorithm),
        "hash_width": str(sys.hash_info.width),
    }


def capture_source_provenance(
    *,
    project_files: Sequence[str],
    official_files: Sequence[str],
) -> dict[str, dict[str, str]]:
    """Hash the exact project and official files used by authoring."""

    result: dict[str, dict[str, str]] = {
        "project_files": {},
        "official_files": {},
    }
    for relative in sorted(set(project_files)):
        path = (PROJECT_ROOT / relative).resolve()
        try:
            path.relative_to(PROJECT_ROOT.resolve())
        except ValueError as exc:
            raise OfficialActualOptionMappingError(
                f"project provenance path escaped root: {relative}"
            ) from exc
        if not path.is_file():
            raise OfficialActualOptionMappingError(
                f"project provenance file is missing: {relative}"
            )
        result["project_files"][relative] = sha256_file(path)
    for relative in sorted(set(official_files)):
        path = (OFFICIAL_BEHAVIOR_ROOT / relative).resolve()
        try:
            path.relative_to(OFFICIAL_BEHAVIOR_ROOT.resolve())
        except ValueError as exc:
            raise OfficialActualOptionMappingError(
                f"official provenance path escaped root: {relative}"
            ) from exc
        if not path.is_file():
            raise OfficialActualOptionMappingError(
                f"official provenance file is missing: {relative}"
            )
        result["official_files"][relative] = sha256_file(path)
    return result


@dataclass(frozen=True, slots=True)
class OfficialActualOptionMapping:
    task_name: str
    definition_id: int
    actual_option_index: int
    official_option_count: int
    semantic_identity: tuple[str, ...]
    semantic_identity_sha256: str
    expanded_problem_sha256: str
    identity_source: str
    compiled_operation_count: int
    compiled_operations_sha256: str
    all_spatial_subject_reference_distinct: bool
    config_sha256: str
    authoring_environment: Mapping[str, str]
    source_provenance: Mapping[str, Mapping[str, str]]
    schema_version: str = OFFICIAL_ACTUAL_OPTION_MAPPING_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OFFICIAL_ACTUAL_OPTION_MAPPING_VERSION:
            raise ValueError("unexpected official actual-option mapping version")
        if not self.task_name:
            raise ValueError("actual-option mapping task name is empty")
        if self.definition_id < 0:
            raise ValueError("actual-option mapping definition ID is invalid")
        if self.actual_option_index < 0:
            raise ValueError("actual-option mapping index is invalid")
        if self.official_option_count < 1:
            raise ValueError("actual-option mapping option count is invalid")
        if self.actual_option_index >= self.official_option_count:
            raise ValueError("actual-option mapping index is out of bounds")
        if not self.semantic_identity:
            raise ValueError("actual-option mapping semantic identity is empty")
        if (
            self.semantic_identity_sha256
            != semantic_identity_sha256(self.semantic_identity)
        ):
            raise ValueError("actual-option mapping semantic identity drifted")
        if len(self.expanded_problem_sha256) != 64:
            raise ValueError("actual-option mapping problem hash is invalid")
        if not self.identity_source:
            raise ValueError("actual-option mapping identity source is empty")
        if self.compiled_operation_count < 1:
            raise ValueError("actual-option mapping operation count is invalid")
        if len(self.compiled_operations_sha256) != 64:
            raise ValueError("actual-option mapping operation hash is invalid")
        if not self.all_spatial_subject_reference_distinct:
            raise ValueError(
                "actual-option mapping contains a spatial self-relation"
            )

    def private_audit_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_name": self.task_name,
            "definition_id": self.definition_id,
            "actual_option_index": self.actual_option_index,
            "official_option_count": self.official_option_count,
            "semantic_identity_sha256": self.semantic_identity_sha256,
            "expanded_problem_sha256": self.expanded_problem_sha256,
            "identity_source": self.identity_source,
            "compiled_operation_count": self.compiled_operation_count,
            "compiled_operations_sha256": self.compiled_operations_sha256,
            "all_spatial_subject_reference_distinct": True,
            "config_sha256": self.config_sha256,
            "runtime_access": "direct_single_index_identity_attestation",
            "runtime_live_scan": False,
            "private_only": True,
        }


def _require_exact_mapping_keys(
    value: Mapping[str, Any],
    *,
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise OfficialActualOptionMappingError(
            f"{label} keys drifted: expected {sorted(expected)}, "
            f"got {sorted(actual)}"
        )


def _load_payload(path: Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    if not source.is_file():
        raise OfficialActualOptionMappingError(
            f"official actual-option mapping file is missing: {source}"
        )
    raw = source.read_bytes()
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OfficialActualOptionMappingError(
            "official actual-option mapping JSON is invalid"
        ) from exc
    if not isinstance(payload, dict):
        raise OfficialActualOptionMappingError(
            "official actual-option mapping root must be an object"
        )
    return payload, hashlib.sha256(raw).hexdigest()


def _require_environment_and_source_provenance(
    payload: Mapping[str, Any],
) -> None:
    environment = payload.get("authoring_environment")
    provenance = payload.get("source_provenance")
    if not isinstance(environment, Mapping) or not isinstance(provenance, Mapping):
        raise OfficialActualOptionMappingError(
            "actual-option mapping lacks authoring provenance"
        )
    current_environment = capture_python_ordering_environment()
    if dict(environment) != current_environment:
        raise OfficialActualOptionMappingError(
            "official option ordering environment drifted: "
            f"expected {dict(environment)}, got {current_environment}"
        )
    project_files = provenance.get("project_files")
    official_files = provenance.get("official_files")
    if not isinstance(project_files, Mapping) or not isinstance(
        official_files, Mapping
    ):
        raise OfficialActualOptionMappingError(
            "actual-option mapping source provenance is invalid"
        )
    current = capture_source_provenance(
        project_files=tuple(str(key) for key in project_files),
        official_files=tuple(str(key) for key in official_files),
    )
    if current != {
        "project_files": dict(project_files),
        "official_files": dict(official_files),
    }:
        raise OfficialActualOptionMappingError(
            "official option ordering source provenance drifted"
        )


def load_official_actual_option_mapping(
    task_name: str,
    *,
    path: Path = DEFAULT_MAPPING_PATH,
    verify_provenance: bool = True,
) -> OfficialActualOptionMapping | None:
    """Load one frozen mapping; return ``None`` only when task is unmapped."""

    payload, config_sha256 = _load_payload(path)
    _require_exact_mapping_keys(
        payload,
        expected={
            "schema_version",
            "authoring_environment",
            "source_provenance",
            "mappings",
        },
        label="mapping root",
    )
    if payload["schema_version"] != OFFICIAL_ACTUAL_OPTION_MAPPING_VERSION:
        raise OfficialActualOptionMappingError(
            "official actual-option mapping schema drifted"
        )
    mappings = payload.get("mappings")
    if not isinstance(mappings, Mapping):
        raise OfficialActualOptionMappingError("mapping records are invalid")
    record = mappings.get(task_name)
    if record is None:
        return None
    if not isinstance(record, Mapping):
        raise OfficialActualOptionMappingError(
            f"mapping record for {task_name} is invalid"
        )
    _require_exact_mapping_keys(
        record,
        expected={
            "task_name",
            "definition_id",
            "actual_option_index",
            "official_option_count",
            "semantic_identity",
            "semantic_identity_sha256",
            "expanded_problem_sha256",
            "identity_source",
            "compiled_operation_count",
            "compiled_operations_sha256",
            "all_spatial_subject_reference_distinct",
        },
        label=f"mapping record {task_name}",
    )
    if record["task_name"] != task_name:
        raise OfficialActualOptionMappingError(
            "mapping key and task name disagree"
        )
    if verify_provenance:
        _require_environment_and_source_provenance(payload)
    environment = payload["authoring_environment"]
    provenance = payload["source_provenance"]
    try:
        return OfficialActualOptionMapping(
            task_name=task_name,
            definition_id=int(record["definition_id"]),
            actual_option_index=int(record["actual_option_index"]),
            official_option_count=int(record["official_option_count"]),
            semantic_identity=tuple(record["semantic_identity"]),
            semantic_identity_sha256=str(record["semantic_identity_sha256"]),
            expanded_problem_sha256=str(record["expanded_problem_sha256"]),
            identity_source=str(record["identity_source"]),
            compiled_operation_count=int(record["compiled_operation_count"]),
            compiled_operations_sha256=str(record["compiled_operations_sha256"]),
            all_spatial_subject_reference_distinct=bool(
                record["all_spatial_subject_reference_distinct"]
            ),
            config_sha256=config_sha256,
            authoring_environment=dict(environment),
            source_provenance={
                "project_files": dict(provenance["project_files"]),
                "official_files": dict(provenance["official_files"]),
            },
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise OfficialActualOptionMappingError(
            f"mapping record for {task_name} failed validation"
        ) from exc


__all__ = [
    "DEFAULT_MAPPING_PATH",
    "OFFICIAL_ACTUAL_OPTION_MAPPING_VERSION",
    "OFFICIAL_BEHAVIOR_ROOT",
    "OfficialActualOptionMapping",
    "OfficialActualOptionMappingError",
    "capture_python_ordering_environment",
    "capture_source_provenance",
    "load_official_actual_option_mapping",
    "semantic_identity_sha256",
    "sha256_file",
]
