"""Validate and route a Physical L2 handoff without replaying a Symbolic state."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


PHYSICAL_CONDITIONS = {
    "action_mode": "Physical",
    "physics_steps_per_action": None,
    "scene_load": "Full",
}


def _handoff_helpers():
    try:
        from lab.instruction_handoff import compile_handoff
    except ModuleNotFoundError:
        from instruction_handoff import compile_handoff
    return compile_handoff


def _read_reference(ref: Mapping[str, Any], *, field: str) -> tuple[Path, dict[str, Any]]:
    path = Path(ref.get("path", "")).resolve()
    expected = ref.get("sha256")
    if not path.is_file() or not isinstance(expected, str):
        raise ValueError(f"Physical {field} reference is incomplete")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Physical {field} reference changed")
    import json

    return path, json.loads(raw)


def is_physical_instruction(handoff_path: Path, instruction_id: str) -> bool:
    compile_handoff = _handoff_helpers()
    matrix = compile_handoff(handoff_path, {"model": 1})
    return _matrix_is_physical(matrix, instruction_id)


def _matrix_is_physical(matrix: Mapping[str, Any], instruction_id: str) -> bool:
    if instruction_id not in matrix["instructions"]:
        raise ValueError("Instruction is absent from handoff: " + instruction_id)
    return all(
        matrix.get("model_conditions", {}).get(key) == value
        for key, value in PHYSICAL_CONDITIONS.items()
    )


def load_physical_instruction(handoff_path: Path, instruction_id: str) -> dict[str, Any]:
    """Return only a fully verified Physical instruction and its live constructor route."""
    compile_handoff = _handoff_helpers()
    matrix = compile_handoff(handoff_path, {"model": 1})
    if not _matrix_is_physical(matrix, instruction_id):
        raise ValueError("handoff instruction is not Physical: " + instruction_id)
    row = matrix["instructions"][instruction_id]
    package_ref = row["package"]
    package_path, package = _read_reference(package_ref, field="package")
    if package.get("runtime", {}).get("adapter") != "registered_l0":
        raise ValueError("Physical instruction requires registered_l0 runtime")
    if package.get("requested_execution_policy") != PHYSICAL_CONDITIONS:
        raise ValueError("Physical package execution policy differs from handoff")
    state = package.get("initial_state_compilation", {})
    if state.get("required") is not True or state.get("status") != "PENDING_NEW_LIVE_READBACK":
        raise ValueError("Physical instruction requires fresh live initial-state readback")
    if state.get("reset_mode") != "RECONSTRUCT_AND_VERIFY_DONOR_PHYSICAL_CONTRACT":
        raise ValueError("Physical instruction has an unsupported reset mode")
    if state.get("donor_success_inherited") is not False:
        raise ValueError("Physical instruction cannot inherit donor success")

    spec = row.get("task_spec")
    if not isinstance(spec, Mapping) or not isinstance(spec.get("physical_template"), Mapping):
        raise ValueError("Physical instruction lacks its physical template")
    if spec.get("task_spec_id") != instruction_id:
        raise ValueError("Physical TaskSpec identity differs from handoff")
    catalog_ref = package["runtime"].get("registration_catalog")
    catalog_path, catalog = _read_reference(catalog_ref, field="registration catalog")
    matches = [
        item
        for item in catalog.get("tasks", [])
        if item.get("task_spec_id", item.get("candidate_id")) == instruction_id
    ]
    if len(matches) != 1:
        raise ValueError("Physical registration is not unique")
    return {
        "matrix": matrix,
        "instruction": row,
        "package": package,
        "package_path": package_path,
        "catalog_path": catalog_path,
        "registered_task": matches[0],
        "initial_state_strategy": "RECONSTRUCT_AND_VERIFY_DONOR_PHYSICAL_CONTRACT",
    }


def native_physical_instruction_arguments(
    handoff_path: Path, instruction_id: str
) -> list[str]:
    value = load_physical_instruction(handoff_path, instruction_id)
    return [
        "--benchmark-package",
        str(value["package_path"]),
        "--construct-and-oracle",
        "--placement-physical-consequences",
        "--physical-human-service",
        "--registered-public-action20-marker-se2-navigation",
    ]
