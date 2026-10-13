"""Scene-affine task batches for one resident OmniGibson process.

The pure helpers in this module deliberately do not import OmniGibson.  They
build a scene superset before Kit starts and derive per-task restore documents
without mutating official task-instance files.  The live adapter is imported
only by the resident launcher after the official evaluator has constructed the
single scene.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import json

from behavior_hl.embodied.evaluator_bridge.full_scene_robot_filter import (
    serialized_generic_robot_model,
)
from behavior_hl.embodied.evaluator_bridge.resident_scene_state import (
    JOINT_STATE_KEYS,
)


RESIDENT_SCENE_BATCH_VERSION = "behavior2026.resident-scene-batch/v2"
MAX_TASKS_PER_RESIDENT_BATCH = 4
PARKING_BASE_Z_M = -20.0
PARKING_SPACING_M = 0.25


class ResidentSceneBatchError(RuntimeError):
    """Raised when a scene-affine batch cannot be represented safely."""


@dataclass(frozen=True)
class ResidentTaskSpec:
    task_name: str
    task_id: int
    scene_model: str
    split: str
    instance_id: int

    def __post_init__(self) -> None:
        if not self.task_name or not self.scene_model or not self.split:
            raise ValueError("resident task identity fields must be non-empty")
        if self.task_id < 0 or self.instance_id < 0:
            raise ValueError("resident task numeric identifiers must be non-negative")


def _resident_task_identity(task: ResidentTaskSpec) -> tuple[int, str, str, int]:
    """Return the stable identity used by sorting, duplicate checks and hashes."""

    if not isinstance(task, ResidentTaskSpec):
        raise ResidentSceneBatchError(
            "resident scene batches require ResidentTaskSpec tasks"
        )
    if (
        isinstance(task.task_id, bool)
        or not isinstance(task.task_id, int)
        or isinstance(task.instance_id, bool)
        or not isinstance(task.instance_id, int)
        or task.task_id < 0
        or task.instance_id < 0
    ):
        raise ResidentSceneBatchError(
            "resident task identifiers must be non-negative integers"
        )
    if (
        not isinstance(task.task_name, str)
        or not task.task_name.strip()
        or not isinstance(task.split, str)
        or not task.split.strip()
        or not isinstance(task.scene_model, str)
        or not task.scene_model.strip()
    ):
        raise ResidentSceneBatchError("resident task identity fields must be non-empty strings")
    return (task.task_id, task.task_name, task.split, task.instance_id)


def _resident_task_identity_record(task: ResidentTaskSpec) -> dict[str, Any]:
    task_id, task_name, split, instance_id = _resident_task_identity(task)
    return {
        "task_id": task_id,
        "task_name": task_name,
        "split": split,
        "instance_id": instance_id,
    }


def _resident_task_logical_identity(task: ResidentTaskSpec) -> tuple[str, str, int]:
    """Return the task identity independent of its numeric catalog ID."""

    _resident_task_identity(task)
    return (task.task_name, task.split, task.instance_id)


def _canonical_json_sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _filesystem_safe_token(value: str) -> str:
    """Make a human-readable path component without admitting path syntax."""

    token = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    token = token.strip(".-_")
    return token or "scene"


@dataclass(frozen=True)
class ResidentSceneShard:
    """One contiguous, scene-affine resident task shard.

    ``batch_id`` is derived only from the canonical scene and task identities,
    so it is stable across input ordering and process runs.  ``batch_path_id``
    adds a readable, filesystem-safe scene/shard prefix while retaining a
    digest suffix for collision resistance.
    """

    scene_model: str
    shard_index: int
    tasks: tuple[ResidentTaskSpec, ...]
    batch_id: str = field(init=False)
    batch_path_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.scene_model, str) or not self.scene_model.strip():
            raise ResidentSceneBatchError("resident shard scene model is empty")
        if (
            isinstance(self.shard_index, bool)
            or not isinstance(self.shard_index, int)
            or self.shard_index < 0
        ):
            raise ResidentSceneBatchError(
                "resident shard index must be a non-negative integer"
            )
        try:
            tasks = tuple(self.tasks)
        except TypeError as exc:
            raise ResidentSceneBatchError(
                "resident shard tasks must be iterable"
            ) from exc
        if not tasks:
            raise ResidentSceneBatchError("resident shard must contain tasks")

        scene_model = true_scene_model(self.scene_model)
        ordered_tasks = tuple(sorted(tasks, key=_resident_task_identity))
        identities = tuple(_resident_task_identity(task) for task in ordered_tasks)
        logical_identities = tuple(
            _resident_task_logical_identity(task) for task in ordered_tasks
        )
        if (
            len(set(identities)) != len(identities)
            or len(set(logical_identities)) != len(logical_identities)
        ):
            raise ResidentSceneBatchError(
                f"duplicate resident task identity in shard: {identities}"
            )
        for task in ordered_tasks:
            if true_scene_model(task.scene_model) != scene_model:
                raise ResidentSceneBatchError(
                    "resident shard task scene does not match shard scene: "
                    f"{task.task_name!r}"
                )

        object.__setattr__(self, "scene_model", scene_model)
        object.__setattr__(self, "tasks", ordered_tasks)
        canonical = {
            "schema_version": RESIDENT_SCENE_BATCH_VERSION,
            "scene_model": scene_model,
            "tasks": [
                _resident_task_identity_record(task) for task in ordered_tasks
            ],
        }
        batch_id = _canonical_json_sha256(canonical)
        object.__setattr__(self, "batch_id", batch_id)
        object.__setattr__(
            self,
            "batch_path_id",
            f"{_filesystem_safe_token(scene_model)}"
            f"__shard-{self.shard_index:03d}__{batch_id}",
        )

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    @property
    def task_identities(self) -> tuple[tuple[int, str, str, int], ...]:
        return tuple(_resident_task_identity(task) for task in self.tasks)

    @property
    def batch_index(self) -> int:
        """Compatibility alias for callers that call the local index batch index."""

        return self.shard_index


def _balanced_shard_sizes(
    task_count: int,
    maximum_tasks_per_resident_batch: int,
) -> tuple[int, ...]:
    """Return max-bounded sizes with the remainder spread from the front."""

    shard_count = (task_count + maximum_tasks_per_resident_batch - 1) // (
        maximum_tasks_per_resident_batch
    )
    base, remainder = divmod(task_count, shard_count)
    return tuple(
        base + 1 if index < remainder else base
        for index in range(shard_count)
    )


def partition_scene_groups(
    groups: Iterable[tuple[str, Sequence[ResidentTaskSpec]]],
    maximum_tasks_per_resident_batch: int = MAX_TASKS_PER_RESIDENT_BATCH,
) -> tuple[ResidentSceneShard, ...]:
    """Partition scene groups into deterministic, contiguous resident shards.

    The configured resident contract caps a shard at four tasks.  The number
    of shards is the minimum needed under that cap, then tasks are distributed
    as evenly as possible.  This avoids a one-task tail whenever a balanced
    arrangement can do so (for example, ``5 -> 3 + 2`` and ``21 ->
    4 + 4 + 4 + 3 + 3 + 3``).
    """

    if (
        isinstance(maximum_tasks_per_resident_batch, bool)
        or not isinstance(maximum_tasks_per_resident_batch, int)
        or not 1 <= maximum_tasks_per_resident_batch <= MAX_TASKS_PER_RESIDENT_BATCH
    ):
        raise ResidentSceneBatchError(
            "maximum_tasks_per_resident_batch must be an integer in [1, 4]"
        )
    if isinstance(groups, (str, bytes)):
        raise ResidentSceneBatchError("resident scene groups must be iterable groups")

    try:
        raw_groups = tuple(groups.items()) if isinstance(groups, Mapping) else tuple(groups)
    except TypeError as exc:
        raise ResidentSceneBatchError("resident scene groups must be iterable") from exc

    def _group_sort_key(group: Any) -> str:
        if (
            isinstance(group, (tuple, list))
            and len(group) == 2
            and isinstance(group[0], str)
        ):
            return true_scene_model(group[0])
        return ""

    raw_groups = tuple(sorted(raw_groups, key=_group_sort_key))

    shards: list[ResidentSceneShard] = []
    seen_scenes: set[str] = set()
    seen_identities: set[tuple[int, str, str, int]] = set()
    seen_logical_identities: set[tuple[str, str, int]] = set()
    for group_index, group in enumerate(raw_groups):
        if not isinstance(group, (tuple, list)) or len(group) != 2:
            raise ResidentSceneBatchError(
                f"resident scene group {group_index} must be (scene_model, tasks)"
            )
        raw_scene_model, raw_tasks = group
        if not isinstance(raw_scene_model, str) or not raw_scene_model.strip():
            raise ResidentSceneBatchError(
                f"resident scene group {group_index} has an empty scene model"
            )
        scene_model = true_scene_model(raw_scene_model)
        if scene_model in seen_scenes:
            raise ResidentSceneBatchError(
                f"duplicate resident scene group: {scene_model!r}"
            )
        seen_scenes.add(scene_model)
        if isinstance(raw_tasks, (str, bytes)):
            raise ResidentSceneBatchError(
                f"resident scene group {group_index} tasks are not iterable"
            )
        try:
            tasks = tuple(raw_tasks)
        except TypeError as exc:
            raise ResidentSceneBatchError(
                f"resident scene group {group_index} tasks are not iterable"
            ) from exc
        if not tasks:
            raise ResidentSceneBatchError(
                f"resident scene group {group_index} must contain tasks"
            )
        ordered_tasks = tuple(sorted(tasks, key=_resident_task_identity))
        for task in ordered_tasks:
            identity = _resident_task_identity(task)
            if identity in seen_identities:
                raise ResidentSceneBatchError(
                    f"duplicate resident task identity: {identity}"
                )
            seen_identities.add(identity)
            logical_identity = _resident_task_logical_identity(task)
            if logical_identity in seen_logical_identities:
                raise ResidentSceneBatchError(
                    f"duplicate resident task identity: {logical_identity}"
                )
            seen_logical_identities.add(logical_identity)
            if true_scene_model(task.scene_model) != scene_model:
                raise ResidentSceneBatchError(
                    "resident task scene does not match scene group: "
                    f"{identity} -> {scene_model!r}"
                )

        offset = 0
        for shard_index, shard_size in enumerate(
            _balanced_shard_sizes(
                len(ordered_tasks), maximum_tasks_per_resident_batch
            )
        ):
            shard_tasks = ordered_tasks[offset : offset + shard_size]
            if len(shard_tasks) != shard_size:
                raise ResidentSceneBatchError(
                    "resident scene partition lost tasks while slicing"
                )
            shards.append(
                ResidentSceneShard(
                    scene_model=scene_model,
                    shard_index=shard_index,
                    tasks=shard_tasks,
                )
            )
            offset += shard_size
        if offset != len(ordered_tasks):
            raise ResidentSceneBatchError(
                "resident scene partition did not consume every task"
            )

    # This is intentionally checked independently of the slicing above: a
    # future change to the shard-size policy must not silently drop/duplicate
    # a task.
    flattened = tuple(
        identity
        for shard in shards
        for identity in shard.task_identities
    )
    if len(flattened) != len(seen_identities) or set(flattened) != seen_identities:
        raise ResidentSceneBatchError(
            "resident scene partition failed exactly-once task coverage"
        )
    return tuple(shards)


def true_scene_model(scene_identity: str) -> str:
    """Return the actual scene model from either old or normalized manifests."""

    value = str(scene_identity).strip()
    if not value:
        raise ValueError("scene identity must be non-empty")
    return value.split("::", 1)[0]


def group_tasks_by_scene(
    tasks: Iterable[ResidentTaskSpec],
) -> tuple[tuple[str, tuple[ResidentTaskSpec, ...]], ...]:
    """Group tasks deterministically by the real scene model."""

    groups: dict[str, list[ResidentTaskSpec]] = {}
    seen: set[tuple[int, str, str, int]] = set()
    seen_logical: set[tuple[str, str, int]] = set()
    for task in tasks:
        identity = _resident_task_identity(task)
        if identity in seen:
            raise ResidentSceneBatchError(f"duplicate resident task: {identity}")
        seen.add(identity)
        logical_identity = _resident_task_logical_identity(task)
        if logical_identity in seen_logical:
            raise ResidentSceneBatchError(
                f"duplicate resident task: {logical_identity}"
            )
        seen_logical.add(logical_identity)
        groups.setdefault(true_scene_model(task.scene_model), []).append(task)
    return tuple(
        (scene, tuple(sorted(items, key=_resident_task_identity)))
        for scene, items in sorted(groups.items())
    )


def load_scene_document(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    _validate_scene_document(value, source=str(path))
    return value


def _validate_scene_document(value: Any, *, source: str) -> None:
    if not isinstance(value, Mapping):
        raise ResidentSceneBatchError(f"scene document is not a mapping: {source}")
    for key in ("init_info", "objects_info", "state"):
        if key not in value:
            raise ResidentSceneBatchError(f"scene document misses {key}: {source}")
    init_info = value.get("objects_info", {}).get("init_info")
    object_state = value.get("state", {}).get("registry", {}).get("object_registry")
    if not isinstance(init_info, Mapping) or not isinstance(object_state, Mapping):
        raise ResidentSceneBatchError(f"scene object registries are invalid: {source}")
    if set(init_info) != set(object_state):
        raise ResidentSceneBatchError(f"scene init/state object names disagree: {source}")


def _object_signature(init_info: Mapping[str, Any]) -> tuple[Any, ...]:
    args = init_info.get("args", {})
    return (
        init_info.get("class_module"),
        init_info.get("class_name"),
        args.get("name"),
        args.get("category"),
        args.get("model"),
        args.get("prim_path"),
    )


def build_union_scene_document(
    documents: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge task templates without changing common-object definitions.

    The first document remains the primary task metadata/state.  Missing
    movable task objects from later documents are added before Kit starts,
    parked off-scene until their own task restore activates them.
    Duplicate names must describe the same object asset.
    """

    if not documents:
        raise ResidentSceneBatchError("at least one scene document is required")
    for index, document in enumerate(documents):
        _validate_scene_document(document, source=f"documents[{index}]")
    union = deepcopy(dict(documents[0]))
    union_init = union["objects_info"]["init_info"]
    union_state = union["state"]["registry"]["object_registry"]
    additions: list[str] = []
    common: set[str] = set(union_init)
    for document in documents[1:]:
        source_init = document["objects_info"]["init_info"]
        source_state = document["state"]["registry"]["object_registry"]
        common &= set(source_init)
        for name in sorted(source_init):
            if name in union_init:
                if _object_signature(union_init[name]) != _object_signature(source_init[name]):
                    raise ResidentSceneBatchError(
                        f"incompatible duplicate object definition: {name}"
                    )
                continue
            union_init[name] = deepcopy(source_init[name])
            # Task zero skips cross-task restore; enforce its isolation here too.
            union_state[name] = _park_object_state(source_state[name], index=len(additions))
            additions.append(name)
    _validate_scene_document(union, source="union")
    serialized = json.dumps(union, sort_keys=True, separators=(",", ":"))
    report = {
        "schema_version": RESIDENT_SCENE_BATCH_VERSION,
        "document_count": len(documents),
        "object_count": len(union_init),
        "common_object_count": len(common),
        "added_object_count": len(additions),
        "added_objects": sorted(additions),
        "union_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }
    return union, report


def _park_object_state(state: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    parked = deepcopy(dict(state))
    root = parked.get("root_link")
    if not isinstance(root, Mapping):
        raise ResidentSceneBatchError("union-only object has no root_link state")
    root = dict(root)
    pos = list(root.get("pos", ()))
    if len(pos) != 3:
        raise ResidentSceneBatchError("union-only object has invalid root pose")
    pos[0] = PARKING_SPACING_M * float(index % 16)
    pos[1] = PARKING_SPACING_M * float(index // 16)
    pos[2] = PARKING_BASE_Z_M - PARKING_SPACING_M * float(index // 256)
    root["pos"] = pos
    root["lin_vel"] = [0.0, 0.0, 0.0]
    root["ang_vel"] = [0.0, 0.0, 0.0]
    parked["root_link"] = root
    if "joint_vel" in parked:
        parked["joint_vel"] = [0.0 for _ in parked["joint_vel"]]
    return parked


def _preserve_missing_joint_state_fields(
    state: Mapping[str, Any],
    fallback: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    """Carry only absent modern joint fields from the completed live state.

    Authored task templates are allowed to use the native legacy ``joints``
    representation, which the OG loader already handles.  For the modern
    representation, an authored template can be missing one or both fields;
    the only safe source is the initial live state captured after native
    construction.  Never replace an authored field or synthesize a value.
    """

    merged = deepcopy(dict(state))
    if "joints" in merged or not isinstance(fallback, Mapping) or "joints" in fallback:
        return merged, []
    filled: list[str] = []
    for key in JOINT_STATE_KEYS:
        if key not in merged and key in fallback:
            merged[key] = deepcopy(fallback[key])
            filled.append(key)
    return merged, filled


def build_runtime_task_restore_document(
    *,
    runtime_superset: Mapping[str, Any],
    union_template: Mapping[str, Any],
    target_template: Mapping[str, Any],
    filter_embedded_scene_agents: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create an object-stable task restore document for a live resident scene.

    ``runtime_superset`` is captured after the evaluator adds its robot.  Its
    init registry is therefore preserved verbatim.  Target task objects receive
    official template state; union-only task objects are parked below the scene;
    runtime-only persistent objects such as the robot keep their captured state.
    """

    for name, document in (
        ("runtime_superset", runtime_superset),
        ("union_template", union_template),
        ("target_template", target_template),
    ):
        _validate_scene_document(document, source=name)
    runtime = deepcopy(dict(runtime_superset))
    runtime_init = runtime["objects_info"]["init_info"]
    runtime_state = runtime["state"]["registry"]["object_registry"]
    union_init = union_template["objects_info"]["init_info"]
    union_names = set(union_init)
    target_names = set(target_template["objects_info"]["init_info"])
    filtered_union_names = (
        {
            name
            for name, init_info in union_init.items()
            if serialized_generic_robot_model(init_info) is not None
        }
        if filter_embedded_scene_agents
        else set()
    )
    unexpectedly_live_filtered_names = filtered_union_names & set(runtime_init)
    if unexpectedly_live_filtered_names:
        names = sorted(unexpectedly_live_filtered_names)
        raise ResidentSceneBatchError(
            f"filtered union robots remain live: {names[:10]}"
        )
    missing_union_names = union_names - set(runtime_init)
    unexpected_missing_names = missing_union_names - filtered_union_names
    if unexpected_missing_names:
        missing = sorted(unexpected_missing_names)
        raise ResidentSceneBatchError(
            f"live runtime misses union objects: {missing[:10]}"
        )
    if not target_names.issubset(union_names):
        missing = sorted(target_names - union_names)
        raise ResidentSceneBatchError(f"target template escaped union: {missing[:10]}")
    target_state = target_template["state"]["registry"]["object_registry"]
    union_state = union_template["state"]["registry"]["object_registry"]
    joint_state_fallbacks: list[dict[str, Any]] = []
    restored_target_names = target_names - filtered_union_names
    for name in sorted(restored_target_names):
        restored, filled = _preserve_missing_joint_state_fields(
            target_state[name], runtime_state.get(name)
        )
        runtime_state[name] = restored
        if filled:
            joint_state_fallbacks.append(
                {"object": name, "fields": filled, "source": "runtime_superset"}
            )
    inactive = sorted(union_names - target_names - filtered_union_names)
    for index, name in enumerate(inactive):
        parked_source, filled = _preserve_missing_joint_state_fields(
            union_state[name], runtime_state.get(name)
        )
        runtime_state[name] = _park_object_state(parked_source, index=index)
        if filled:
            joint_state_fallbacks.append(
                {"object": name, "fields": filled, "source": "runtime_superset"}
            )
    runtime["state"]["pos"] = deepcopy(target_template["state"]["pos"])
    runtime["state"]["ori"] = deepcopy(target_template["state"]["ori"])
    runtime["state"]["registry"]["system_registry"] = deepcopy(
        target_template["state"]["registry"]["system_registry"]
    )
    runtime["metadata"] = deepcopy(target_template.get("metadata", {}))
    _validate_scene_document(runtime, source="runtime_target")
    persistent = sorted(set(runtime_init) - union_names)
    report = {
        "schema_version": RESIDENT_SCENE_BATCH_VERSION,
        "target_object_count": len(target_names),
        "restored_target_object_count": len(restored_target_names),
        "filtered_union_object_count": len(filtered_union_names),
        "filtered_union_objects": sorted(filtered_union_names),
        "filter_embedded_scene_agents": bool(filter_embedded_scene_agents),
        "inactive_union_object_count": len(inactive),
        "inactive_union_objects": inactive,
        "persistent_runtime_object_count": len(persistent),
        "persistent_runtime_objects": persistent,
        "object_add_remove_required": False,
        "parking_base_z_m": PARKING_BASE_Z_M,
        "runtime_superset_joint_state_fallback_object_count": len(
            joint_state_fallbacks
        ),
        "runtime_superset_joint_state_fallback_field_count": sum(
            len(item["fields"]) for item in joint_state_fallbacks
        ),
        "runtime_superset_joint_state_fallbacks": joint_state_fallbacks,
    }
    return runtime, report


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    from behavior_hl.embodied.evaluator_bridge.oracle_progress_journal import json_safe

    value = json_safe(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)
