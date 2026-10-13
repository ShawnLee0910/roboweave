"""Read-only validation for CPU Oracle scene-loaded preflight candidates.

The records in this module are static proposals.  They are never a live
collision, visibility, manipulation, or Oracle result.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


MAX_CANDIDATES = 16
POSE_TOLERANCE = 1e-6
QUATERNION_UNIT_TOLERANCE = 1e-3
PREFLIGHT_STATUS = "PENDING_SCENE_LOADED_PREFLIGHT"

# ``camera_pitch_deg`` is an existing diagnostic in the CPU export.  The
# other names below are actual optical/configuration overrides and must not be
# smuggled through the neutral/center view state.
_OPTICAL_OVERRIDE_NAMES = frozenset(
    {
        "camera_config",
        "camera_extrinsics",
        "camera_intrinsics",
        "camera_optics",
        "camera_profile",
        "extrinsics",
        "focal_length",
        "focal_length_mm",
        "horizontal_fov_deg",
        "image_height",
        "image_width",
        "intrinsics",
        "lens",
        "optical_parameters",
        "resolution",
        "vertical_fov_deg",
    }
)


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"{label} is not readable UTF-8 JSONL: {exc}") from exc
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} line {line_number} is invalid JSON: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} line {line_number} must be an object")
        records.append(value)
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_ref(owner: Path, value: Any, label: str) -> Path:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a file reference")
    raw_path = value.get("path")
    expected = value.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label}.path must be a non-empty string")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError(f"{label}.sha256 must be a lowercase SHA-256 digest")
    path = Path(raw_path)
    if not path.is_absolute():
        path = owner.parent / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{label} file is unavailable: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 drifted: {path}")
    return path


def _member_id(record: Mapping[str, Any], label: str) -> str:
    task_spec_id = record.get("task_spec_id")
    candidate_id = record.get("candidate_id")
    value = task_spec_id if task_spec_id is not None else candidate_id
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} has no task_spec_id/candidate_id")
    if candidate_id is not None and candidate_id != value:
        raise ValueError(f"{label} task_spec_id/candidate_id disagree")
    return value


def _member_list(value: Any, label: str, expected_count: int) -> list[str]:
    if not isinstance(value, list) or len(value) != expected_count:
        raise ValueError(f"{label} must contain exactly {expected_count} members")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ValueError(f"{label}[{index}] must be a non-empty string")
        result.append(item)
    if len(set(result)) != expected_count:
        raise ValueError(f"{label} contains duplicate members")
    return result


def _validate_member_records(
    records: Sequence[Mapping[str, Any]],
    expected: Sequence[str],
    label: str,
) -> dict[str, Mapping[str, Any]]:
    if len(records) != len(expected):
        raise ValueError(f"{label} must contain exactly {len(expected)} records")
    actual = [_member_id(record, f"{label}[{index}]") for index, record in enumerate(records)]
    if actual != list(expected):
        raise ValueError(f"{label} membership/order does not match source package")
    return dict(zip(actual, records))


def _finite_vector(value: Any, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{label} must contain exactly {size} values")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{label}[{index}] must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{label}[{index}] must be finite")
        result.append(number)
    return tuple(result)


def _validate_pose(value: Any, label: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    position = _finite_vector(value.get("position"), 3, f"{label}.position")
    orientation = _finite_vector(value.get("orientation"), 4, f"{label}.orientation")
    norm = math.sqrt(sum(component * component for component in orientation))
    if norm <= 1e-8 or abs(norm - 1.0) > QUATERNION_UNIT_TOLERANCE:
        raise ValueError(f"{label}.orientation must be a unit quaternion")
    return position, orientation


def _validate_cell(value: Any, label: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain two cell coordinates")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{label}[{index}] must be a non-negative integer")
        result.append(item)
    return tuple(result)  # type: ignore[return-value]


def _looks_like_optical_override(name: str) -> bool:
    normalized = name.casefold().replace("-", "_")
    return (
        normalized in _OPTICAL_OVERRIDE_NAMES
        or "focal" in normalized
        or normalized.endswith("_intrinsics")
        or normalized.endswith("_extrinsics")
    )


def _validate_view(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if value.get("pitch_state") != "center":
        raise ValueError(f"{label}.pitch_state must remain center")
    if value.get("torso_height_state") != "neutral":
        raise ValueError(f"{label}.torso_height_state must remain neutral")
    if value.get("initial_view_actions") != []:
        raise ValueError(f"{label}.initial_view_actions must be empty")
    for key in value:
        if isinstance(key, str) and _looks_like_optical_override(key):
            raise ValueError(f"{label} contains an optical override: {key}")


def _same_float_vector(lhs: Sequence[float], rhs: Sequence[float]) -> bool:
    return all(
        math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=POSE_TOLERANCE)
        for a, b in zip(lhs, rhs)
    )


def _same_quaternion(lhs: Sequence[float], rhs: Sequence[float]) -> bool:
    return _same_float_vector(lhs, rhs) or _same_float_vector(
        lhs, tuple(-float(value) for value in rhs)
    )


def _validate_static_status(record: Mapping[str, Any], label: str) -> None:
    if record.get("runtime_enabled") is not False:
        raise ValueError(f"{label} cannot claim runtime_enabled")
    if record.get("live_status") != "NOT_RUN":
        raise ValueError(f"{label} cannot claim a Live status")
    if record.get("goal_initial_false_live") != "NOT_RUN":
        raise ValueError(f"{label} cannot claim live goal truth")
    if record.get("render_status") != "NOT_RUN":
        raise ValueError(f"{label} cannot claim rendered validation")


def _validate_oracle_preflight(
    record: Mapping[str, Any], label: str
) -> dict[int, tuple[tuple[float, ...], tuple[float, ...]]]:
    value = record.get("oracle_preflight")
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}.oracle_preflight must be an object")
    if value.get("schema_version") != "action20.cpu-preflight-candidates/v1":
        raise ValueError(f"{label}.oracle_preflight schema is unsupported")
    if value.get("status") != PREFLIGHT_STATUS:
        raise ValueError(f"{label} has a non-pending preflight status")
    if value.get("exact_collision_checks") != "NOT_RUN":
        raise ValueError(f"{label} cannot claim exact collision checks")
    if value.get("final_pose_frozen_by_this_compiler") is not False:
        raise ValueError(f"{label} cannot claim a frozen live pose")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or len(candidates) > MAX_CANDIDATES:
        raise ValueError(f"{label}.oracle_preflight.candidates exceeds {MAX_CANDIDATES}")
    limit = value.get("candidate_limit")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= MAX_CANDIDATES:
        raise ValueError(f"{label}.oracle_preflight.candidate_limit is invalid")
    if len(candidates) > limit:
        raise ValueError(f"{label}.oracle_preflight.candidates exceeds candidate_limit")
    eligible = value.get("eligible_candidate_count")
    if isinstance(eligible, bool) or not isinstance(eligible, int) or eligible < len(candidates):
        raise ValueError(f"{label}.oracle_preflight.eligible_candidate_count is invalid")
    if not isinstance(value.get("exported_candidates_complete"), bool):
        raise ValueError(f"{label}.oracle_preflight.exported_candidates_complete is invalid")
    resolution = value.get("map_resolution_m")
    if isinstance(resolution, bool) or not isinstance(resolution, (int, float)):
        raise ValueError(f"{label}.oracle_preflight.map_resolution_m is invalid")
    if not math.isfinite(float(resolution)) or float(resolution) <= 0:
        raise ValueError(f"{label}.oracle_preflight.map_resolution_m is invalid")
    map_size = value.get("map_size")
    if isinstance(map_size, bool) or not isinstance(map_size, int) or map_size <= 0:
        raise ValueError(f"{label}.oracle_preflight.map_size is invalid")

    poses: dict[int, tuple[tuple[float, ...], tuple[float, ...]]] = {}
    previous_rank = 0
    for index, candidate in enumerate(candidates):
        candidate_label = f"{label}.oracle_preflight.candidates[{index}]"
        if not isinstance(candidate, Mapping):
            raise ValueError(f"{candidate_label} must be an object")
        rank = candidate.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= MAX_CANDIDATES:
            raise ValueError(f"{candidate_label}.rank is invalid")
        if rank <= previous_rank or rank in poses:
            raise ValueError(f"{label} candidate ranks must be unique and sorted")
        previous_rank = rank
        pose = candidate.get("robot_pose")
        position, orientation = _validate_pose(pose, f"{candidate_label}.robot_pose")
        _validate_cell(candidate.get("cell"), f"{candidate_label}.cell")
        _validate_view(
            candidate.get("initial_camera_view_state"),
            f"{candidate_label}.initial_camera_view_state",
        )
        if not isinstance(candidate.get("egress_world_directions"), list):
            raise ValueError(
                f"{candidate_label}.egress_world_directions must be preserved as a list"
            )
        if not isinstance(candidate.get("route_legs"), list):
            raise ValueError(f"{candidate_label}.route_legs must be preserved as a list")
        if candidate.get("manipulation_path_check") != "NOT_RUN":
            raise ValueError(f"{candidate_label} cannot claim manipulation validation")
        if candidate.get("stowed_whole_body_check") != "NOT_RUN":
            raise ValueError(f"{candidate_label} cannot claim whole-body validation")
        for leg_index, leg in enumerate(candidate["route_legs"]):
            if not isinstance(leg, Mapping):
                raise ValueError(f"{candidate_label}.route_legs[{leg_index}] must be an object")
            if "exact_manipulation_check" in leg and leg["exact_manipulation_check"] != "NOT_RUN":
                raise ValueError(
                    f"{candidate_label}.route_legs[{leg_index}] claims manipulation validation"
                )
        poses[rank] = (position, orientation)
    return poses


def _validate_source_package(
    package_path: Path,
    expected_count: int,
) -> tuple[
    list[str],
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    Path,
    Path,
    Mapping[str, Any],
]:
    package = _read_json(package_path, "source package")
    if package.get("task_count") != expected_count:
        raise ValueError("source package.task_count disagrees with wrapper package")
    member_ids = _member_list(
        package.get("task_spec_ids"), "source package.task_spec_ids", expected_count
    )
    task_ref = package.get("task_specs")
    runtime = package.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("source package.runtime must be an object")
    task_specs_path = _verified_ref(package_path, task_ref, "source package.task_specs")
    catalog_path = _verified_ref(
        package_path,
        runtime.get("registration_catalog"),
        "source package.runtime.registration_catalog",
    )
    task_rows = _validate_member_records(
        _read_jsonl(task_specs_path, "source package TaskSpecs"),
        member_ids,
        "source package TaskSpecs",
    )
    catalog = _read_json(catalog_path, "source package runtime catalog")
    if catalog.get("task_count") != expected_count:
        raise ValueError("source package runtime catalog.task_count disagrees with package")
    if catalog.get("loader_ready") is not True:
        raise ValueError("source package runtime catalog is not loader_ready")
    catalog_rows = _validate_member_records(
        catalog.get("tasks") if isinstance(catalog.get("tasks"), list) else [],
        member_ids,
        "source package runtime catalog.tasks",
    )
    return member_ids, task_rows, catalog_rows, task_specs_path, catalog_path, package


def load_oracle_preflight_candidates(package_path: Path) -> dict[str, Mapping[str, Any]]:
    """Load and validate static preflight candidates without changing package state."""

    package_path = Path(package_path).resolve()
    package = _read_json(package_path, "benchmark package")
    preflight = package.get("oracle_preflight")
    if preflight is None:
        return {}
    if not isinstance(preflight, Mapping):
        raise ValueError("benchmark package.oracle_preflight must be an object")
    candidates_path = _verified_ref(
        package_path,
        preflight.get("candidates"),
        "oracle_preflight.candidates",
    )
    source_package_path = _verified_ref(
        package_path,
        preflight.get("source_package"),
        "oracle_preflight.source_package",
    )
    expected_count = package.get("task_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
    ):
        raise ValueError("benchmark package.task_count must be a positive integer")
    package_ids = _member_list(
        package.get("task_spec_ids"), "benchmark package.task_spec_ids", expected_count
    )
    (
        member_ids,
        task_rows,
        catalog_rows,
        source_task_specs_path,
        source_catalog_path,
        source_package,
    ) = _validate_source_package(source_package_path, expected_count)
    wrapper_runtime = package.get("runtime")
    source_runtime = source_package.get("runtime")
    if not isinstance(wrapper_runtime, Mapping) or not isinstance(source_runtime, Mapping):
        raise ValueError("wrapper/source runtime references are required")
    wrapper_task_specs_path = _verified_ref(
        package_path, package.get("task_specs"), "benchmark package.task_specs"
    )
    wrapper_catalog_path = _verified_ref(
        package_path,
        wrapper_runtime.get("registration_catalog"),
        "benchmark package.runtime.registration_catalog",
    )
    if wrapper_runtime.get("adapter") != source_runtime.get("adapter"):
        raise ValueError("wrapper/source runtime adapters disagree")
    if wrapper_task_specs_path != source_task_specs_path:
        raise ValueError("wrapper/source TaskSpecs references disagree")
    if wrapper_catalog_path != source_catalog_path:
        raise ValueError("wrapper/source runtime catalog references disagree")
    if package_ids != member_ids:
        raise ValueError("benchmark package membership does not match source package")
    records = _read_jsonl(candidates_path, "oracle_preflight candidates")
    if len(records) != expected_count:
        raise ValueError(
            f"oracle_preflight candidates must contain exactly {expected_count} records"
        )
    actual_ids = [
        _member_id(record, f"oracle_preflight candidates[{index}]")
        for index, record in enumerate(records)
    ]
    if actual_ids != member_ids:
        raise ValueError(
            "oracle_preflight candidate membership/order does not match source package"
        )

    family_poses: dict[
        str, dict[int, tuple[tuple[float, ...], tuple[float, ...]]]
    ] = defaultdict(dict)
    family_ranks: dict[str, set[int]] = {}
    result: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        task_id = actual_ids[index]
        label = f"oracle_preflight candidates[{index}] ({task_id})"
        _validate_static_status(record, label)
        family = record.get("source_family_id")
        if not isinstance(family, str) or not family:
            raise ValueError(f"{label}.source_family_id is required")
        source_family = task_rows[task_id].get("source_family_id")
        if source_family is not None and source_family != family:
            raise ValueError(f"{label}.source_family_id differs from source TaskSpec")
        catalog_family = catalog_rows[task_id].get("source_family_id")
        if catalog_family is not None and catalog_family != family:
            raise ValueError(f"{label}.source_family_id differs from runtime catalog")
        if not isinstance(record.get("complete_search"), bool):
            raise ValueError(f"{label}.complete_search is invalid")
        evaluated_cells = record.get("evaluated_cell_count")
        if (
            isinstance(evaluated_cells, bool)
            or not isinstance(evaluated_cells, int)
            or evaluated_cells < 0
        ):
            raise ValueError(f"{label}.evaluated_cell_count is invalid")
        if record.get("route_legs") is not None and not isinstance(record.get("route_legs"), list):
            raise ValueError(f"{label}.route_legs must be a list or null")
        if (
            record.get("egress_world_directions") is not None
            and not isinstance(record.get("egress_world_directions"), list)
        ):
            raise ValueError(f"{label}.egress_world_directions must be a list or null")
        fixed_pose = record.get("fixed_robot_pose")
        view = record.get("initial_camera_view_state")
        if fixed_pose is not None:
            _validate_pose(fixed_pose, f"{label}.fixed_robot_pose")
        if view is not None:
            _validate_view(view, f"{label}.initial_camera_view_state")
        poses = _validate_oracle_preflight(record, label)
        ranks = set(poses)
        previous_ranks = family_ranks.get(family)
        if previous_ranks is not None and ranks != previous_ranks:
            raise ValueError(f"{label} rank set disagrees with its family")
        family_ranks[family] = ranks
        for rank, (position, orientation) in poses.items():
            prior = family_poses[family].get(rank)
            if prior is not None and (
                not _same_float_vector(position, prior[0])
                or not _same_quaternion(orientation, prior[1])
            ):
                raise ValueError(f"{label} rank {rank} disagrees with its family pose")
            family_poses[family][rank] = (position, orientation)
        result[task_id] = record
    return result


__all__ = ["load_oracle_preflight_candidates"]
