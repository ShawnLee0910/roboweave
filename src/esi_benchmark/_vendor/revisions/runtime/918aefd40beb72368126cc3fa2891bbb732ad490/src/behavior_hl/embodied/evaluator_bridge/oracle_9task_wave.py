"""Pure validation utilities for the frozen nine-task Oracle wave."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from behavior_hl.embodied.evaluator_bridge.inside_task_profiles import (
    INSIDE_TASK_PROFILES,
    get_inside_task_profile,
)


WAVE_SCHEMA_VERSION = "behavior2026.oracle-9task-wave/v1"
APPDATA_MARKER_SCHEMA_VERSION = "behavior2026.appdata-slot/v1"
NINE_TASK_WAVE_PROFILES = frozenset(
    {
        "picking_up_toys",
        "composting_waste",
        "picking_up_trash",
        "putting_dirty_dishes_in_sink",
        "collecting_aluminum_cans",
        "putting_away_toys",
        "boxing_books_up_for_storage",
        "dispose_of_glass",
        "clean_up_broken_glass",
    }
)


def load_wave_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("wave manifest must be a JSON object")
    return value


def task_instance_path(data_root: Path, task_name: str) -> Path:
    profile = get_inside_task_profile(task_name)
    split_root = {
        "train": Path("scenes"),
        "public_test": Path("scene_test/public"),
        "hidden_test": Path("scene_test/private"),
    }.get(profile.split)
    if split_root is None:
        raise ValueError(f"unsupported task instance split: {profile.split!r}")
    stem = (
        f"{profile.scene_model}_task_{profile.task_name}_0_"
        f"{profile.instance_id}_template-tro_state.json"
    )
    return (
        Path(data_root)
        / "2026-challenge-task-instances"
        / split_root
        / profile.scene_model
        / "json"
        / f"{profile.scene_model}_task_{profile.task_name}_instances"
        / stem
    )


def critical_source_fingerprint(
    *, project_root: Path, relative_paths: list[str]
) -> str:
    root = Path(project_root).resolve()
    aggregate = hashlib.sha256()
    for relative in sorted(relative_paths):
        path = (root / relative).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"critical source escaped project root: {path}") from exc
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return aggregate.hexdigest()


def appdata_source_identity(source_root: Path, stack_id: str) -> str:
    root = Path(source_root).resolve(strict=True)
    aggregate = hashlib.sha256()
    aggregate.update(stack_id.encode("utf-8"))
    aggregate.update(b"\0")
    aggregate.update(str(root).encode("utf-8"))
    aggregate.update(b"\n")
    for relative in ("global/cache", "global/data", "local/cache", "local/data"):
        directory = root / relative
        if not directory.is_dir():
            raise FileNotFoundError(f"AppData seed directory is missing: {directory}")
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0dir\n")
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            rel = path.relative_to(root).as_posix()
            aggregate.update(rel.encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
            aggregate.update(b"\n")
    return aggregate.hexdigest()


def build_appdata_marker(
    *, source_root: Path, stack_id: str, slot_index: int
) -> dict[str, Any]:
    return {
        "schema_version": APPDATA_MARKER_SCHEMA_VERSION,
        "stack_id": stack_id,
        "slot_index": int(slot_index),
        "source_root": str(Path(source_root).resolve(strict=True)),
        "source_identity_sha256": appdata_source_identity(source_root, stack_id),
        "copy_mode": "lazy_cache_seed",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_appdata_marker(
    *, marker_path: Path, source_root: Path, stack_id: str, slot_index: int
) -> dict[str, Any]:
    marker = json.loads(Path(marker_path).read_text(encoding="utf-8"))
    expected = {
        "schema_version": APPDATA_MARKER_SCHEMA_VERSION,
        "stack_id": stack_id,
        "slot_index": int(slot_index),
        "source_root": str(Path(source_root).resolve(strict=True)),
        "source_identity_sha256": appdata_source_identity(source_root, stack_id),
        "copy_mode": "lazy_cache_seed",
    }
    mismatch = {
        key: {"actual": marker.get(key), "expected": value}
        for key, value in expected.items()
        if marker.get(key) != value
    }
    if mismatch:
        raise ValueError(f"AppData ready marker mismatch: {mismatch}")
    return marker


def validate_wave_manifest(
    manifest: Mapping[str, Any],
    *,
    require_task_states: bool = True,
    require_source_fingerprint: bool = True,
) -> dict[str, Any]:
    if manifest.get("schema_version") != WAVE_SCHEMA_VERSION:
        raise ValueError("unexpected nine-task wave schema")
    execution = manifest.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("execution contract must be a mapping")
    expected_execution = {
        "attempts_per_task": 1,
        "automatic_retry": False,
        "oracle_only": True,
        "scene_load_budget_per_task": 1,
        "route_planning": "online_official_robot_eroded_astar_per_leg",
        "complete_route_precompiled_before_execution": False,
        "leg_route_planned_before_motion": True,
        "local_interaction_reach_m": 1.3,
        "execution_fidelity": "kinematic_research",
        "official_challenge_eligible": False,
    }
    mismatched_execution = {
        key: {"actual": execution.get(key), "expected": expected}
        for key, expected in expected_execution.items()
        if execution.get(key) != expected
    }
    if mismatched_execution:
        raise ValueError(f"execution contract mismatch: {mismatched_execution}")

    catalog = manifest.get("task_catalog")
    if not isinstance(catalog, Mapping) or set(catalog) != NINE_TASK_WAVE_PROFILES:
        raise ValueError("task catalog must cover the nine frozen profiles exactly")
    for task_name, record in catalog.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"task catalog entry is not a mapping: {task_name}")
        profile = get_inside_task_profile(task_name)
        expected = {
            "task_id": profile.task_id,
            "scene_model": profile.scene_model,
            "split": profile.split,
            "instance_id": profile.instance_id,
            "goal_leaves": profile.expected_goal_leaf_count,
            "containers": profile.expected_container_count,
        }
        if dict(record) != expected:
            raise ValueError(
                f"task catalog drift for {task_name}: actual={dict(record)}, "
                f"expected={expected}"
            )

    agents = manifest.get("agents")
    if not isinstance(agents, list) or len(agents) != 3:
        raise ValueError("wave requires exactly three agents")
    agent_ids: list[str] = []
    flattened_tasks: list[str] = []
    task_locations: dict[str, tuple[str, str]] = {}
    for agent in agents:
        if not isinstance(agent, Mapping):
            raise ValueError("agent entry must be a mapping")
        agent_id = agent.get("agent_id")
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        agent_ids.append(agent_id)
        lanes = agent.get("lanes")
        if not isinstance(lanes, list) or len(lanes) != 2:
            raise ValueError(f"{agent_id} must own exactly two lanes")
        if sorted(len(lane.get("tasks", ())) for lane in lanes) != [1, 2]:
            raise ValueError(f"{agent_id} lane lengths must be one and two")
        lane_ids = [lane.get("lane_id") for lane in lanes]
        if set(lane_ids) != {"a", "b"}:
            raise ValueError(f"{agent_id} lane IDs must be a and b")
        if len({lane.get("server_port") for lane in lanes}) != 2:
            raise ValueError(f"{agent_id} lane ports must be unique")
        if len({lane.get("appdata_slot") for lane in lanes}) != 2:
            raise ValueError(f"{agent_id} appdata slots must be unique")
        agent_tasks: list[str] = []
        for lane in lanes:
            lane_id = str(lane["lane_id"])
            tasks = lane.get("tasks")
            if not isinstance(tasks, list) or not all(
                isinstance(task, str) for task in tasks
            ):
                raise ValueError(f"{agent_id}/{lane_id} tasks must be strings")
            for task in tasks:
                task_locations[task] = (agent_id, lane_id)
            agent_tasks.extend(tasks)
        if len(agent_tasks) != 3:
            raise ValueError(f"{agent_id} must own exactly three tasks")
        flattened_tasks.extend(agent_tasks)
    if len(set(agent_ids)) != len(agent_ids):
        raise ValueError("agent IDs must be unique")
    counts = Counter(flattened_tasks)
    if counts != Counter({name: 1 for name in NINE_TASK_WAVE_PROFILES}):
        raise ValueError(f"wave task assignment is not exactly once: {counts}")

    scenes: dict[str, list[str]] = defaultdict(list)
    for task_name in flattened_tasks:
        scenes[get_inside_task_profile(task_name).scene_model].append(task_name)
    for scene_tasks in scenes.values():
        if len(scene_tasks) < 2:
            continue
        locations = {task_locations[task] for task in scene_tasks}
        if len(locations) != 1:
            raise ValueError(
                f"same-scene tasks must share one serial lane: {scene_tasks}"
            )

    data_root = Path(str(manifest["data_root"])).resolve()
    if require_task_states:
        missing = [
            str(task_instance_path(data_root, task))
            for task in flattened_tasks
            if not task_instance_path(data_root, task).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"missing frozen TRO states: {missing}")

    source_identity = manifest.get("source_identity")
    if not isinstance(source_identity, Mapping):
        raise ValueError("source_identity must be a mapping")
    critical_files = source_identity.get("critical_files")
    if not isinstance(critical_files, list) or not critical_files:
        raise ValueError("critical source file list must be non-empty")
    fingerprint = critical_source_fingerprint(
        project_root=Path(str(manifest["source_root"])),
        relative_paths=list(critical_files),
    )
    expected_fingerprint = source_identity.get("critical_aggregate_sha256")
    if require_source_fingerprint and fingerprint != expected_fingerprint:
        raise ValueError(
            "critical source fingerprint mismatch: "
            f"actual={fingerprint}, expected={expected_fingerprint}"
        )
    return {
        "schema_version": WAVE_SCHEMA_VERSION,
        "campaign_id": manifest["campaign_id"],
        "agent_count": len(agents),
        "task_count": len(flattened_tasks),
        "critical_aggregate_sha256": fingerprint,
        "task_states_checked": bool(require_task_states),
        "same_scene_lane_serial": True,
        "cross_task_resident_scene_reuse": False,
    }


def agent_spec(manifest: Mapping[str, Any], agent_id: str) -> Mapping[str, Any]:
    matches = [
        agent
        for agent in manifest.get("agents", ())
        if isinstance(agent, Mapping) and agent.get("agent_id") == agent_id
    ]
    if len(matches) != 1:
        raise KeyError(f"unknown or duplicate agent_id: {agent_id!r}")
    return matches[0]
