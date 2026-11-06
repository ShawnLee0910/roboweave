"""Additive, seed-independent L0 reset poses and runtime document overlays."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


OVERLAY_VERSION = "action20-capability-benchmark.l0-reset-pose-overlay/v1"


def _finite_vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_reset_pose_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one compiler row without accepting seed-dependent fields."""

    if record.get("schema_version") != OVERLAY_VERSION:
        raise ValueError("unsupported L0 reset-pose overlay version")
    task_spec_id = record.get("task_spec_id")
    task_name = record.get("task_name")
    if not isinstance(task_spec_id, str) or not task_spec_id:
        raise ValueError("reset-pose record has no task_spec_id")
    if not isinstance(task_name, str) or not task_name:
        raise ValueError("reset-pose record has no task_name")
    if "seed" in record or "random_seed" in record:
        raise ValueError("fixed reset poses must not contain a seed")
    pose = record.get("fixed_robot_pose")
    if not isinstance(pose, Mapping):
        raise ValueError(f"{task_spec_id} has no selected fixed_robot_pose")
    position = _finite_vector(pose.get("position"), 3, "position")
    orientation = _finite_vector(pose.get("orientation"), 4, "orientation")
    norm = math.sqrt(sum(item * item for item in orientation))
    if abs(norm - 1.0) > 1.0e-6:
        raise ValueError("reset-pose orientation must be a unit quaternion")
    view = record.get("initial_camera_view_state")
    if not isinstance(view, Mapping) or view.get("pitch_state") != "center":
        raise ValueError("phase-one overlay requires the canonical R1Pro center reset view")
    if view.get("initial_view_actions") not in ([], ()):
        raise ValueError("center reset view must not require initial view actions")
    result = dict(record)
    result["fixed_robot_pose"] = {
        **dict(pose),
        "position": position,
        "orientation": [item / norm for item in orientation],
    }
    return result


def load_l0_reset_pose_overlay(path: Path) -> Mapping[str, Mapping[str, Any]]:
    """Load the selected rows from an additive JSONL overlay."""

    result: dict[str, Mapping[str, Any]] = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, Mapping):
            raise ValueError(f"overlay line {line_number} is not an object")
        if raw.get("selection_status") != "SELECTED":
            continue
        record = validate_reset_pose_record(raw)
        task_spec_id = record["task_spec_id"]
        if task_spec_id in result:
            raise ValueError(f"duplicate reset-pose task_spec_id: {task_spec_id}")
        result[task_spec_id] = record
    if not result:
        raise ValueError("reset-pose overlay contains no selected rows")
    return result


def patch_scene_template_reset_pose(
    document: Mapping[str, Any], record: Mapping[str, Any]
) -> dict[str, Any]:
    """Patch an in-memory template; the source template remains untouched."""

    validated = validate_reset_pose_record(record)
    output = deepcopy(dict(document))
    metadata = output.setdefault("metadata", {}).setdefault("task", {})
    robot_poses = metadata.setdefault("robot_poses", {})
    pose = validated["fixed_robot_pose"]
    robot_poses["R1Pro"] = [
        {"position": list(pose["position"]), "orientation": list(pose["orientation"])}
    ]
    # Official instance load and task reset prefer the generic robot key.
    for key in tuple(robot_poses):
        if key.lower() in {"robot", "r1pro"}:
            robot_poses[key] = deepcopy(robot_poses["R1Pro"])
    metadata["l0_reset_pose_overlay"] = {
        "schema_version": OVERLAY_VERSION,
        "task_spec_id": validated["task_spec_id"],
        "initial_camera_view_state": deepcopy(validated["initial_camera_view_state"]),
    }
    return output


def patch_tro_state_reset_pose(
    document: Mapping[str, Any], record: Mapping[str, Any]
) -> dict[str, Any]:
    """Patch a generated TRO copy through the official robot_poses reset key."""

    validated = validate_reset_pose_record(record)
    output = deepcopy(dict(document))
    pose = validated["fixed_robot_pose"]
    robot_poses = deepcopy(output.get("robot_poses", {}))
    robot_poses["R1Pro"] = [
        {"position": list(pose["position"]), "orientation": list(pose["orientation"])}
    ]
    for key in tuple(robot_poses):
        if key.lower() in {"robot", "r1pro"}:
            robot_poses[key] = deepcopy(robot_poses["R1Pro"])
    output["robot_poses"] = robot_poses
    return output


def configure_official_l0_reset_pose_runtime(
    evaluator_module: Any,
    *,
    records_by_task_spec_id: Mapping[str, Mapping[str, Any]],
    tro_paths_by_task_spec_id: Mapping[str, Path],
    current_task_spec_id: Callable[[], str],
) -> Callable[[], None]:
    """Overlay official available-task and TRO lookup hooks process-locally."""

    if set(records_by_task_spec_id) != set(tro_paths_by_task_spec_id):
        raise ValueError("runtime reset-pose records and TRO paths are not aligned")
    records = {
        name: validate_reset_pose_record(record)
        for name, record in records_by_task_spec_id.items()
    }
    if any(key != record["task_spec_id"] for key, record in records.items()):
        raise ValueError("runtime reset-pose keys must match task_spec_id")
    paths = {name: Path(path).resolve() for name, path in tro_paths_by_task_spec_id.items()}
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("generated reset-pose TRO path is missing")
    original_available = evaluator_module.load_available_tasks
    original_instance_path = evaluator_module.get_task_instance_path

    def load_available_tasks() -> dict[str, Any]:
        value = dict(original_available())
        record = records[current_task_spec_id()]
        task_name = record["task_name"]
        if task_name not in value or 0 not in value[task_name]:
            raise ValueError(f"official available task is missing: {task_name}")
        task = deepcopy(value[task_name])
        pose = record["fixed_robot_pose"]
        task[0]["robot_start_position"] = list(pose["position"])
        task[0]["robot_start_orientation"] = list(pose["orientation"])
        value[task_name] = task
        return value

    def get_task_instance_path(
        scene_model: str, name: str, *args: Any, **kwargs: Any
    ) -> Any:
        item_id = current_task_spec_id()
        task_name = records[item_id]["task_name"]
        prefix = f"{scene_model}_task_{task_name}_instances/"
        if name.startswith(prefix):
            return str(paths[item_id])
        return original_instance_path(scene_model, name, *args, **kwargs)

    evaluator_module.load_available_tasks = load_available_tasks
    evaluator_module.get_task_instance_path = get_task_instance_path

    def cleanup() -> None:
        evaluator_module.load_available_tasks = original_available
        evaluator_module.get_task_instance_path = original_instance_path

    return cleanup


def attest_l0_reset_pose(robot: Any, record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the readback and record pose drift without a proximity gate."""
    from behavior_hl.embodied.evaluator_bridge.same_scene_reset_attestation import (
        _quaternion_angle,
    )

    expected = validate_reset_pose_record(record)["fixed_robot_pose"]
    raw_position, raw_orientation = robot.get_position_orientation()
    position = _finite_vector(tuple(raw_position), 3, "measured_position")
    orientation = _finite_vector(tuple(raw_orientation), 4, "measured_orientation")
    if math.sqrt(sum(value * value for value in orientation)) <= 1e-8:
        raise ValueError("measured reset orientation has zero norm")
    position_error = math.dist(position, expected["position"])
    orientation_error = _quaternion_angle(orientation, expected["orientation"])
    return {
        "schema_version": "behavior2026.l0-fixed-reset-pose-attestation/v2",
        "task_spec_id": record["task_spec_id"],
        "status": "PASS",
        "reason": None,
        "admission_policy": "finite_readback_pose_drift_diagnostic_only",
        "expected_pose": expected,
        "measured_position": position,
        "measured_orientation": orientation,
        "position_error_m": position_error,
        "orientation_error_rad": orientation_error,
        "robot_pose_modified": False,
    }


def selected_candidate_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    """The requested deterministic C-G ordering after A/B have passed."""

    return (
        -int(candidate["visible_relevant_instance_count"]),
        -int(candidate["visible_role_kind_count"]),
        float(candidate["total_path_length_m"]),
        int(candidate["initial_turn_look_action_count"]),
        tuple(candidate["stable_coordinate_tiebreak"]),
    )


def choose_fixed_reset_pose_candidate(
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Reject incomplete routes, then apply the exact requested ordering."""

    eligible = [
        candidate
        for candidate in candidates
        if candidate.get("all_required_routes_exist") is True
        and candidate.get("under_l0_route_target") is True
        and int(candidate.get("visible_relevant_instance_count", 0)) > 0
    ]
    return min(eligible, key=selected_candidate_key) if eligible else None
