"""Interactive JSONL runner for the five-task high-level pilot."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any

from behavior_hl.actions import HighLevelAction
from behavior_hl.controller import EpisodeController
from behavior_hl.esi_backup_actions import (
    ActionDisposition,
    BackupPose,
    ESIBackupActionError,
    NavigationGuards,
    RestrictedESIBackupActions,
    RestrictedMotionController,
    supported_esi_backup_actions,
)
from behavior_hl.hard_trajectories import describe_hard_trajectory
from behavior_hl.omnigibson_backend import BackendActionError, OmniGibsonBackend
from behavior_hl.navigation import PlanarPose
from behavior_hl.project_identity import BUNDLE_ROOT, PROJECT_ROOT
from behavior_hl.tasks import FIVE_TASKS, TaskSpec, get_task_spec


DEFAULT_BUNDLE_ROOT = BUNDLE_ROOT
DEFAULT_BEHAVIOR_ROOT = DEFAULT_BUNDLE_ROOT / "BEHAVIOR-1K-2026"
DEFAULT_DATA_ROOT = DEFAULT_BUNDLE_ROOT / "BEHAVIOR-1K-2026-data"
REQUIRED_ISAAC_SIM_SERIES = "5.1."


class RuntimePreflightError(RuntimeError):
    """Raised before OmniGibson import when the isolated runtime is incompatible."""


def _require_isaac_runtime() -> str:
    """Fail quickly before importing OmniGibson with an incompatible runtime."""

    if sys.version_info[:2] != (3, 11):
        current_python = f"{sys.version_info[0]}.{sys.version_info[1]}"
        raise RuntimePreflightError(
            "BEHAVIOR-1K 3.9.0 requires Python 3.11 with Isaac Sim 5.1; "
            f"current Python is {current_python}"
        )
    if importlib.util.find_spec("isaacsim") is None:
        raise RuntimePreflightError(
            "Isaac Sim 5.1 is not installed in this isolated Python environment; "
            "the runner will not fall back to ESI-Bench Isaac Sim 4.5 runtime"
        )
    try:
        version = metadata.version("isaacsim")
    except metadata.PackageNotFoundError as exc:
        raise RuntimePreflightError(
            "the isaacsim module is visible but its distribution metadata is missing; "
            "refusing an unverifiable runtime"
        ) from exc
    if not version.startswith(REQUIRED_ISAAC_SIM_SERIES):
        required_series = REQUIRED_ISAAC_SIM_SERIES.rstrip(".")
        raise RuntimePreflightError(
            f"Isaac Sim {version} is incompatible; BEHAVIOR-1K 3.9.0 requires "
            f"the {required_series} series"
        )
    return version


def _default_appdata_root(*, pid: int | None = None) -> Path:
    """Return a per-process runtime directory without mutating global state."""

    process_id = os.getpid() if pid is None else pid
    return (PROJECT_ROOT / ".runtime" / "appdata" / f"pid-{process_id}").resolve()


def _configure_isolated_process(
    behavior_root: Path,
    data_root: Path,
    *,
    headless: bool,
    appdata_root: Path | None = None,
) -> Path:
    if not behavior_root.is_dir():
        raise FileNotFoundError(f"BEHAVIOR source root not found: {behavior_root}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"BEHAVIOR data root not found: {data_root}")

    resolved_appdata_root = (
        _default_appdata_root()
        if appdata_root is None
        else appdata_root.expanduser().resolve()
    )
    resolved_appdata_root.mkdir(parents=True, exist_ok=True)

    # Override rather than inherit any ESI process settings. These variables are
    # process-local and never alter the user's shell or the ESI-Bench checkout.
    os.environ["OMNIGIBSON_DATA_PATH"] = str(data_root.resolve())
    os.environ["OMNIGIBSON_APPDATA_PATH"] = str(resolved_appdata_root)
    os.environ["OMNIGIBSON_HEADLESS"] = "1" if headless else "0"

    component_roots = (
        behavior_root / "OmniGibson",
        behavior_root / "bddl3",
        behavior_root / "joylo",
    )
    for component_root in reversed(component_roots):
        if component_root.is_dir():
            sys.path.insert(0, str(component_root.resolve()))
    return resolved_appdata_root


def _build_environment(
    task: TaskSpec,
    behavior_root: Path,
    data_root: Path,
    *,
    headless: bool,
    appdata_root: Path | None = None,
) -> tuple[Any, Any, Path]:
    _require_isaac_runtime()
    resolved_appdata_root = _configure_isolated_process(
        behavior_root,
        data_root,
        headless=headless,
        appdata_root=appdata_root,
    )

    import yaml

    import omnigibson as og
    from omnigibson.macros import gm

    with gm.unlocked():
        gm.ENABLE_TRANSITION_RULES = True
        gm.USE_GPU_DYNAMICS = False

    config_path = behavior_root / "OmniGibson" / "omnigibson" / "configs" / "r1pro_primitives.yaml"
    available_path = data_root / "2026-challenge-task-instances" / "metadata" / "available_tasks.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    with available_path.open("r", encoding="utf-8") as stream:
        available_tasks = yaml.safe_load(stream)

    task_config = available_tasks[task.name][0]
    config["scene"]["scene_model"] = task_config["scene_model"]
    config["scene"]["load_room_types"] = None
    config["scene"]["load_room_instances"] = task_config.get("load_room_instances")
    config["scene"]["load_task_relevant_only"] = False
    config["robots"][0]["position"] = task_config["robot_start_position"]
    config["robots"][0]["orientation"] = task_config["robot_start_orientation"]
    config["task"] = {
        "type": "BehaviorTask",
        "activity_name": task.name,
        "activity_definition_id": 0,
        "activity_instance_id": 0,
        "online_object_sampling": False,
        "debug_object_sampling": False,
        "highlight_task_relevant_objects": False,
        "include_obs": False,
        "termination_config": {"max_steps": 100_000},
        "reward_config": {"r_potential": 1.0},
    }

    env = og.Environment(configs=config)
    env.reset()
    return og, env, resolved_appdata_root


def _load_task_instance(og: Any, env: Any, *, instance_id: int, split: str) -> Path:
    from omnigibson.utils.asset_utils import get_task_instance_path
    from omnigibson.utils.bddl_utils import is_system_bddl_inst
    from omnigibson.utils.python_utils import recursively_convert_to_torch

    scene_model = env.task.scene_name
    filename = env.task.get_cached_activity_scene_filename(
        scene_model=scene_model,
        activity_name=env.task.activity_name,
        activity_definition_id=env.task.activity_definition_id,
        activity_instance_id=instance_id,
    )
    instance_path = get_task_instance_path(
        scene_model,
        f"{scene_model}_task_{env.task.activity_name}_instances/{filename}-tro_state",
        mode=split,
    )
    if instance_path is None:
        raise FileNotFoundError(
            f"no {split} instance {instance_id} for {env.task.activity_name} in {scene_model}"
        )

    with Path(instance_path).open("r", encoding="utf-8") as stream:
        state = recursively_convert_to_torch(json.load(stream))

    robot = env.robots[0]
    for key, value in state.items():
        if key == "robot_poses":
            poses = {name.lower(): pose_list for name, pose_list in value.items()}
            if "robot" in poses:
                available_poses = poses["robot"]
            elif robot.model in poses:
                available_poses = poses[robot.model]
            else:
                raise KeyError(f"no generic or {robot.model!r} robot pose in {instance_path}")
            robot.set_position_orientation(available_poses[0]["position"], available_poses[0]["orientation"])
            env.scene.write_task_metadata(key=key, data=value)
        else:
            env.task.object_scope[key].load_state(value, serialized=False)

    og.sim.update_handles()
    for _ in range(25):
        og.sim.step_physics()
        for instance, entity in env.task.object_scope.items():
            if not is_system_bddl_inst(instance) and entity is not None:
                entity.keep_still()
    env.scene.update_initial_file()
    env.scene.reset()
    return Path(instance_path)


def _to_planar_pose(pose: BackupPose) -> PlanarPose:
    return PlanarPose(pose.x, pose.y, pose.z, pose.yaw)


def _current_backup_pose(backend: OmniGibsonBackend) -> BackupPose:
    pose = backend.current_planar_pose()
    return BackupPose(pose.x, pose.y, pose.z, pose.yaw, pitch=0.0)


def _build_esi_backup_adapter(backend: OmniGibsonBackend) -> RestrictedESIBackupActions:
    def same_floor(start: BackupPose, target: BackupPose) -> bool:
        return backend.same_floor_motion(_to_planar_pose(start), _to_planar_pose(target))

    def guarded_motion(start: BackupPose, target: BackupPose) -> bool:
        return backend.guarded_local_motion(_to_planar_pose(start), _to_planar_pose(target))

    guards = NavigationGuards(
        same_floor=same_floor,
        traversable=guarded_motion,
        collision_free=guarded_motion,
    )
    return RestrictedESIBackupActions(
        motion=RestrictedMotionController(guards),
        resolve_object=backend.resolve_object_label,
        # Nearest-door commands remain fail-closed until a real segmentation-aware
        # visibility selector is available from the simulator observation wrapper.
        select_door=None,
    )


def _parse_esi_backup_line(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith(chr(34)):
        payload = json.loads(stripped)
        if not isinstance(payload, str):
            raise ESIBackupActionError("action_must_be_string", "ESI JSON input must contain one string")
        return payload
    if stripped.startswith("{") or stripped.startswith("["):
        raise ESIBackupActionError(
            "action_must_be_string",
            "ESI backup mode accepts a raw action line or a JSON string, not an object",
        )
    return stripped


def _describe() -> int:
    payload = []
    for spec in sorted(FIVE_TASKS.values(), key=lambda item: item.b100_index):
        payload.append(
            {
                "b100_index": spec.b100_index,
                "task": spec.name,
                "scene_model": spec.scene_model,
                "goal_leaf_templates": spec.goal_leaf_templates,
                "official_canonical_goal": spec.official_canonical_goal,
                "actions": sorted(action.value for action in spec.allowed_actions),
                "physics_status": spec.physics_status,
                "physics_note": spec.physics_note,
            }
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _describe_hard(task_name: str | None = None) -> int:
    if task_name is not None:
        payload: dict[str, Any] | list[dict[str, Any]] = describe_hard_trajectory(task_name)
    else:
        payload = [
            describe_hard_trajectory(spec.name)
            for spec in sorted(FIVE_TASKS.values(), key=lambda item: item.b100_index)
        ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(FIVE_TASKS))
    parser.add_argument("--mode", choices=("symbolic", "physics"), default="symbolic")
    parser.add_argument("--behavior-root", type=Path, default=DEFAULT_BEHAVIOR_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--appdata-root",
        type=Path,
        help="process-private OmniGibson appdata directory (default: .runtime/appdata/pid-<pid>)",
    )
    parser.add_argument("--split", choices=("train", "public_test", "hidden_test"), default="train")
    parser.add_argument("--instance-id", type=int, default=0)
    parser.add_argument("--max-macro-steps", type=int, default=200)
    parser.add_argument("--max-simulator-steps", type=int, default=100_000)
    parser.add_argument("--max-simulator-steps-per-macro", type=int, default=10_000)
    parser.add_argument("--max-input-lines", type=int, default=400)
    parser.add_argument(
        "--input-protocol",
        choices=("strict-json", "esi-backup"),
        default="strict-json",
        help="use guarded legacy ESI strings only when explicitly requested",
    )
    parser.add_argument("--gui", action="store_true", help="launch with a viewer instead of headless mode")
    describe_group = parser.add_mutually_exclusive_group()
    describe_group.add_argument(
        "--describe",
        action="store_true",
        help="print the five task specs without launching Isaac Sim",
    )
    describe_group.add_argument(
        "--describe-hard",
        action="store_true",
        help="print one or five public hard-v1 manifests without launching Isaac Sim",
    )
    args = parser.parse_args()

    if args.describe:
        return _describe()
    if args.describe_hard:
        return _describe_hard(args.task)
    if args.task is None:
        parser.error("--task is required unless --describe or --describe-hard is used")
    for argument_name in (
        "max_macro_steps",
        "max_simulator_steps",
        "max_simulator_steps_per_macro",
        "max_input_lines",
    ):
        if getattr(args, argument_name) <= 0:
            parser.error(f"--{argument_name.replace(chr(95), chr(45))} must be positive")

    spec = get_task_spec(args.task)
    if args.mode == "physics" and spec.physics_status == "blocked":
        parser.error(f"{spec.name} is blocked in physics mode: {spec.physics_note}")

    try:
        _require_isaac_runtime()
    except RuntimePreflightError as exc:
        print(
            json.dumps(
                {"event": "runtime_unavailable", "message": str(exc)},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 3

    og = None
    try:
        og, env, appdata_root = _build_environment(
            spec,
            args.behavior_root.resolve(),
            args.data_root.resolve(),
            headless=not args.gui,
            appdata_root=args.appdata_root,
        )
        instance_path = _load_task_instance(og, env, instance_id=args.instance_id, split=args.split)
        backend = OmniGibsonBackend(
            env,
            mode=args.mode,
            max_simulator_steps_per_macro=args.max_simulator_steps_per_macro,
            max_total_simulator_steps=args.max_simulator_steps,
        )
        episode = EpisodeController(
            spec,
            backend,
            max_macro_steps=args.max_macro_steps,
            max_simulator_steps=args.max_simulator_steps,
        )
        esi_adapter = _build_esi_backup_adapter(backend) if args.input_protocol == "esi-backup" else None

        ready = {
            "event": "ready",
            "task": spec.name,
            "b100_index": spec.b100_index,
            "mode": args.mode,
            "input_protocol": args.input_protocol,
            "budgets": {
                "max_macro_steps": args.max_macro_steps,
                "max_simulator_steps": args.max_simulator_steps,
                "max_simulator_steps_per_macro": args.max_simulator_steps_per_macro,
                "max_input_lines": args.max_input_lines,
            },
            "instance": str(instance_path),
            "appdata_root": str(appdata_root),
            "goal_leaf_templates": spec.goal_leaf_templates,
            "official_canonical_goal": spec.official_canonical_goal,
            "allowed_actions": sorted(action.value for action in spec.allowed_actions),
            "addressable_objects": backend.addressable_objects(),
            "success": backend.success(),
            "q_score": backend.q_score(),
            "goal_status": backend.goal_status(),
        }
        if esi_adapter is not None:
            unbound_actions = {
                "turn_up",
                "turn_down",
                "open nearest visible door",
                "close nearest visible door",
            }
            ready["esi_backup_actions"] = [
                action for action in supported_esi_backup_actions() if action not in unbound_actions
            ]
            ready["esi_backup_unbound_actions"] = sorted(unbound_actions)
            ready["esi_backup_limitations"] = [
                "move_up/down are forbidden",
                "turn_up/down are rejected for R1Pro because it has no active camera joint",
                "nearest-door commands remain disabled until visibility segmentation is bound",
                "map guards are rechecked by the core backend at execution time",
            ]
        print(json.dumps(ready, ensure_ascii=False), flush=True)

        loop_end_reason = "stdin_eof"
        for line_number, line in enumerate(sys.stdin, start=1):
            if line_number > args.max_input_lines:
                loop_end_reason = "input_line_limit"
                print(
                    json.dumps(
                        {
                            "event": "input_limit",
                            "line": line_number,
                            "max_input_lines": args.max_input_lines,
                        }
                    ),
                    flush=True,
                )
                break
            if not line.strip():
                continue
            try:
                prepared_payload: dict[str, Any] = {}
                if esi_adapter is None:
                    action = HighLevelAction.from_json(line)
                else:
                    legacy_action = _parse_esi_backup_line(line)
                    prepared = esi_adapter.prepare(
                        legacy_action,
                        _current_backup_pose(backend),
                        held_instance=backend.held_object(),
                    )
                    if prepared.disposition is ActionDisposition.VIEW_ONLY:
                        raise ESIBackupActionError(
                            "view_control_unbound",
                            "R1Pro has no active camera joint; view-only pitch is fail-closed",
                        )
                    if prepared.core_action is None:
                        raise ESIBackupActionError(
                            "missing_core_action",
                            "prepared ESI action has no executable core action",
                        )
                    action = HighLevelAction.from_dict(prepared.core_action)
                    prepared_payload = {
                        "esi_action": prepared.normalized_action,
                        "esi_disposition": prepared.disposition.value,
                        "esi_approved_pose": asdict(prepared.approved_pose),
                    }

                result = episode.step(action)
                step_event = {
                    "event": "step",
                    "input_protocol": args.input_protocol,
                    **prepared_payload,
                    **asdict(result),
                }
                print(json.dumps(step_event, ensure_ascii=False), flush=True)
                if result.terminated:
                    break
            except Exception as exc:
                error = {
                    "event": "error",
                    "line": line_number,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "episode_terminated": episode.terminated,
                }
                if isinstance(exc, BackendActionError):
                    error["reason"] = exc.reason
                    error["details"] = exc.details
                elif isinstance(exc, ESIBackupActionError):
                    error["reason"] = exc.code
                print(json.dumps(error, ensure_ascii=False), flush=True)
                if episode.terminated:
                    break

        if not episode.terminated:
            print(
                json.dumps(
                    {
                        "event": "incomplete",
                        "reason": loop_end_reason,
                        "macro_steps": episode.macro_steps,
                        "simulator_steps": episode.simulator_steps,
                        "success": backend.success(),
                        "q_score": backend.q_score(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return 2
        return 0
    finally:
        if og is not None:
            og.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
