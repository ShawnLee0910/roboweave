"""One-load Task25 Action20 pick / carry / floor-placement atomic canary.

Private setup locates the live workbench and moves the robot to an audited
interaction station.  Every subject turn then uses the existing public
Action20 registry and resident executor.  No goal truth, object identity, or
station geometry is copied into public feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from behavior_hl.embodied.evaluator_bridge.action20_task25_atomic_spec import (
    ACTION20_TASK25_ATOMIC_SPEC_VERSION,
    TASK25_CARRY_ATOMIC_SPEC,
    Task25CarryAtomicSpec,
    validate_task25_carry_atomic_spec,
)
from behavior_hl.embodied.evaluator_bridge.marker_symbolic_grasp_backend import (
    official_instance_bboxes,
)


ACTION20_TASK25_ATOMIC_RUNNER_VERSION = (
    "behavior2026.action20-task25-carry-atomic-runner/v1"
)
PHASE_NAME = "action20_task25_carry_atomic"
_MINIMUM_PRIVATE_VISIBLE_PIXELS = 8


@dataclass(frozen=True, slots=True)
class Task25WorkbenchFixture:
    """Private live handles for one common Task25 support surface."""

    support: Any = field(repr=False)
    station_target: Any = field(repr=False)
    eligible_targets: tuple[Any, ...] = field(repr=False)
    support_name_private: str
    station_target_name_private: str
    eligible_target_names_private: tuple[str, ...]
    support_aabb_private: tuple[tuple[float, ...], tuple[float, ...]]
    station_target_xy_private: tuple[float, float]

    def to_private_record(self) -> dict[str, Any]:
        return {
            "schema_version": "behavior2026.task25-workbench-fixture/v1",
            "support_name_private": self.support_name_private,
            "station_target_name_private": self.station_target_name_private,
            "eligible_target_names_private": list(
                self.eligible_target_names_private
            ),
            "support_aabb_private": [
                list(bound) for bound in self.support_aabb_private
            ],
            "station_target_xy_private": list(
                self.station_target_xy_private
            ),
            "selection_policy": (
                "largest common live OnTop support; stable-name tie-break; "
                "nearest eligible station target"
            ),
            "private_only": True,
        }


def _default_runtime_api() -> Any:
    return importlib.import_module(
        "behavior_hl.embodied.evaluator_bridge."
        "run_b1_marker_symbolic_rollout"
    )


def _default_role_provider(
    evaluator: Any,
    profile: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    module = importlib.import_module(
        "behavior_hl.embodied.evaluator_bridge.inside_task_profiles"
    )
    return module.private_inside_task_roles(evaluator, profile)


def _entity_name(entity: Any) -> str:
    name = str(getattr(entity, "name", ""))
    if not name:
        raise RuntimeError("Task25 atomic fixture has an unnamed entity")
    return name


def _finite_vector(value: Any, *, minimum_size: int, label: str) -> tuple[float, ...]:
    try:
        flattened = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Task25 {label} is not numeric") from exc
    if len(flattened) < minimum_size or not all(
        math.isfinite(item) for item in flattened
    ):
        raise RuntimeError(f"Task25 {label} is not finite")
    return flattened


def _entity_xy(entity: Any) -> tuple[float, float]:
    getter = getattr(entity, "get_position_orientation", None)
    if not callable(getter):
        raise RuntimeError("Task25 fixture entity lacks a live pose")
    position, _orientation = getter()
    values = _finite_vector(position, minimum_size=2, label="fixture pose")
    return values[0], values[1]


def _entity_aabb(
    entity: Any,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    aabb = getattr(entity, "aabb", None)
    if not isinstance(aabb, (tuple, list)) or len(aabb) != 2:
        raise RuntimeError("Task25 workbench lacks a live AABB")
    lower = _finite_vector(aabb[0], minimum_size=3, label="AABB lower")
    upper = _finite_vector(aabb[1], minimum_size=3, label="AABB upper")
    if any(low >= high for low, high in zip(lower[:3], upper[:3])):
        raise RuntimeError("Task25 workbench AABB is degenerate")
    return lower, upper


def _scene_objects(evaluator: Any) -> tuple[Any, ...]:
    scene = getattr(getattr(evaluator, "env", None), "scene", None)
    if scene is None:
        scene = getattr(evaluator, "scene", None)
    objects = getattr(scene, "objects", None)
    if isinstance(objects, Mapping):
        result = tuple(objects.values())
    else:
        result = tuple(objects or ())
    if not result:
        raise RuntimeError("Task25 atomic fixture scene has no objects")
    return result


def _named_state(entity: Any, state_name: str) -> Any | None:
    states = getattr(entity, "states", None)
    if not isinstance(states, Mapping):
        return None
    matches = [
        state
        for state_type, state in states.items()
        if getattr(state_type, "__name__", "") == state_name
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Task25 fixture has duplicate {state_name} state")
    return matches[0] if matches else None


def _robot_xy(evaluator: Any) -> tuple[float, float]:
    robot = getattr(evaluator, "robot", None)
    return _entity_xy(robot)


def _entity_xy_yaw(entity: Any) -> tuple[float, float, float]:
    getter = getattr(entity, "get_position_orientation", None)
    if not callable(getter):
        raise RuntimeError("Task25 private setup entity lacks a live pose")
    position, orientation = getter()
    xyz = _finite_vector(position, minimum_size=2, label="private setup pose")
    quaternion = _finite_vector(
        orientation,
        minimum_size=4,
        label="private setup orientation",
    )
    qx, qy, qz, qw = quaternion[:4]
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-9:
        raise RuntimeError("Task25 private setup orientation is degenerate")
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return xyz[0], xyz[1], yaw


def _wrapped_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _robot_base_positions(robot: Any) -> Any:
    getter = getattr(robot, "get_joint_positions", None)
    base_indices = getattr(robot, "base_idx", None)
    if not callable(getter) or base_indices is None:
        raise RuntimeError("Task25 private setup lacks floating-base joints")
    try:
        positions = getter()[base_indices]
    except Exception as exc:
        raise RuntimeError(
            "Task25 private setup cannot read floating-base joints"
        ) from exc
    if len(positions) < 5:
        raise RuntimeError("Task25 private setup requires a 6-DoF base")
    return positions


def _reset_robot_base_tilt(robot: Any) -> tuple[float, float]:
    positions = _robot_base_positions(robot)
    prior = _finite_vector(
        positions[3:5],
        minimum_size=2,
        label="private setup base roll/pitch",
    )[:2]
    try:
        zero_tilt = positions[3:5] * 0.0
    except Exception as exc:
        raise RuntimeError("Task25 private setup cannot zero base tilt") from exc
    set_positions = getattr(robot, "set_joint_positions", None)
    set_velocities = getattr(robot, "set_joint_velocities", None)
    if not callable(set_positions) or not callable(set_velocities):
        raise RuntimeError("Task25 private setup lacks base joint writers")
    base_indices = robot.base_idx
    set_positions(zero_tilt, indices=base_indices[3:5], drive=False)
    set_velocities(zero_tilt, indices=base_indices[3:5], drive=False)
    after = _finite_vector(
        _robot_base_positions(robot)[3:5],
        minimum_size=2,
        label="private setup reset roll/pitch",
    )[:2]
    if any(abs(value) > 1e-6 for value in after):
        raise RuntimeError("Task25 private setup base tilt reset failed")
    return prior


def _default_private_target_visibility(
    *,
    evaluator: Any,
    target: Any,
    observation: Any,
) -> Mapping[str, Any]:
    capture_module = importlib.import_module(
        "behavior_hl.embodied.evaluator_bridge.r1pro_b1_live_entrypoint"
    )
    marker_module = importlib.import_module(
        "behavior_hl.embodied.evaluator_bridge.private_visual_markers"
    )
    simulator_step = getattr(observation, "simulator_step", None)
    if (
        isinstance(simulator_step, bool)
        or not isinstance(simulator_step, int)
        or simulator_step < 0
    ):
        raise RuntimeError("Task25 private visibility lacks a simulator step")
    values = getattr(observation, "values", None)
    try:
        head_rgb = values["cameras"]["head"]["rgb"]
    except Exception as exc:
        raise RuntimeError("Task25 private visibility lacks head RGB") from exc
    frame = capture_module.capture_private_head_instance_segmentation(
        evaluator,
        simulator_step=simulator_step,
        expected_task_id=TASK25_CARRY_ATOMIC_SPEC.task_id,
        observation_profile=TASK25_CARRY_ATOMIC_SPEC.observation_profile,
    )
    official_bboxes = official_instance_bboxes(frame)
    snapshot = marker_module.build_private_marker_snapshot(
        frame=frame,
        public_head_rgb=head_rgb,
        eligible_objects_by_name={_entity_name(target): target},
        official_bboxes=official_bboxes,
        created_action_index=0,
        minimum_visible_pixels=_MINIMUM_PRIVATE_VISIBLE_PIXELS,
    )
    bindings = tuple(snapshot.bindings.values())
    matching = tuple(
        binding
        for binding in bindings
        if getattr(binding, "target_token", None) is target
        and getattr(binding, "binding_kind", "object") == "object"
    )
    visible_pixels = (
        0 if len(matching) != 1 else int(matching[0].visible_pixels)
    )
    return {
        "visible": bool(
            len(matching) == 1
            and visible_pixels >= _MINIMUM_PRIVATE_VISIBLE_PIXELS
        ),
        "binding_count": len(matching),
        "visible_pixels": visible_pixels,
        "minimum_visible_pixels": _MINIMUM_PRIVATE_VISIBLE_PIXELS,
        "camera": str(getattr(snapshot, "camera", "")),
        "simulator_step": simulator_step,
        "private_only": True,
    }


def _private_official_station_route_plan(
    *,
    evaluator: Any,
    target: Any,
    desired_distance_m: float,
    maximum_station_distance_m: float,
    runtime_api: Any | None = None,
) -> Any:
    """Plan one private station with the donor's official-map bindings.

    The mature donor fixes its maximum station distance to the Action20 local
    reach threshold (1.30 m).  This adapter delegates to that exact donor for
    the atomic 1.20 / 1.30 m contract.  A caller with a different audited hard
    limit, such as the distance/reset 1.37 / 1.40 m band, uses the same live
    scene bindings and official eroded-map planner while passing that hard
    limit explicitly.  No radial object-yaw fallback is permitted.
    """

    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in (desired_distance_m, maximum_station_distance_m)
    ):
        raise ValueError("Task25 official station distances must be finite")
    desired = float(desired_distance_m)
    maximum = float(maximum_station_distance_m)
    if desired <= 0.0 or maximum <= 0.0 or desired > maximum:
        raise ValueError("Task25 official station desired distance exceeds hard limit")

    api = runtime_api or _default_runtime_api()
    donor_maximum = float(getattr(api, "_LOCAL_REACH_M", 1.30))
    donor = getattr(api, "_private_official_oracle_route_plan", None)
    if math.isclose(maximum, donor_maximum, rel_tol=0.0, abs_tol=1e-9):
        if not callable(donor):
            raise RuntimeError("Task25 official station donor is unavailable")
        return donor(
            evaluator=evaluator,
            target=target,
            desired_distance_m=desired,
        )

    planner = getattr(api, "plan_official_traversability_route", None)
    pose_provider = getattr(api, "planar_base_pose_from_robot", None)
    object_xy_provider = getattr(api, "_object_xy", None)
    terminal_aabb_provider = getattr(
        api,
        "_private_interaction_terminal_aabbs_xy",
        None,
    )
    route_result_type = getattr(api, "OfficialTraversabilityRouteResult", None)
    if not all(
        callable(provider)
        for provider in (
            planner,
            pose_provider,
            object_xy_provider,
            terminal_aabb_provider,
            route_result_type,
        )
    ):
        raise RuntimeError("Task25 official station planner bindings are unavailable")

    robot = getattr(evaluator, "robot", None)
    scene = getattr(getattr(evaluator, "env", None), "scene", None)
    if scene is None:
        scene = getattr(robot, "scene", None)
    if scene is None:
        return route_result_type(
            status="unavailable",
            plan=None,
            private_report={
                "provider_version": (
                    "behavior2026.official-traversability-route-station-yaw/v2"
                ),
                "reason": "evaluator_scene_unavailable",
            },
        )

    official_cell_astar = None
    official_four_connected_astar = None
    trav_map = getattr(scene, "trav_map", None)
    if str(getattr(type(trav_map), "__module__", "")).startswith(
        "omnigibson."
    ):
        from omnigibson.utils.motion_planning_utils import astar

        official_cell_astar = astar
        official_four_connected_astar = (
            lambda search_map, start, goal: astar(
                search_map,
                start,
                goal,
                eight_connected=False,
            )
        )

    pose = pose_provider(robot)
    tx, ty = object_xy_provider(target)
    target_rooms = tuple(
        {
            str(room)
            for room in (getattr(target, "in_rooms", None) or ())
            if isinstance(room, str) and room
        }
    )
    portal_categories = getattr(api, "OPENABLE_ROOM_PORTAL_CATEGORIES", ())
    boundary_target = bool(
        str(getattr(target, "category", "")) in portal_categories
        and len(target_rooms) == 2
    )
    try:
        robot_position = robot.get_position_orientation()[0]
        robot_z = float(robot_position[2])
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
        robot_z = 0.0
    return planner(
        scene=scene,
        robot=robot,
        start_xy=(pose.x, pose.y),
        target_xy=(tx, ty),
        target_room_names=(target_rooms or None),
        structural_terminal_aabbs_xy=terminal_aabb_provider(
            target=target,
            scene=scene,
        ),
        robot_z=robot_z,
        desired_distance_m=desired,
        maximum_station_distance_m=maximum,
        official_cell_astar=official_cell_astar,
        official_four_connected_astar=official_four_connected_astar,
        allow_cross_room_station_when_target_room_known=boundary_target,
    )


def _private_workbench_front_view_setup(
    *,
    runtime: Any,
    evaluator: Any,
    target: Any,
    desired_distance_m: float,
    hard_distance_m: float,
    observation_provider: Callable[[Any], Any],
    visibility_provider: Callable[..., Mapping[str, Any]] | None = None,
    runtime_api: Any | None = None,
    private_probe_path: Path | None = None,
    private_route_provider: Callable[..., Any] | None = None,
    private_station_pose_provider: Callable[..., Any] | None = None,
) -> Mapping[str, Any]:
    """Teleport to an official-map private station without public actions."""

    probe: dict[str, Any] = {
        "schema_version": "behavior2026.task25-private-station-probe/v1",
        "status": "STARTED",
        "setup_kind": "official_eroded_map_station",
        "desired_distance_m_private": desired_distance_m,
        "maximum_station_distance_m_private": hard_distance_m,
        "setup_action_count": 0,
        "public_action_index_increment": 0,
        "simulator_steps": 0,
        "private_only": True,
    }

    def persist_probe(status: str, **fields: Any) -> None:
        probe.update(fields)
        probe["status"] = status
        if private_probe_path is not None:
            _write_json(Path(private_probe_path), probe)

    persist_probe("STARTED")

    try:
        navigation = getattr(
            runtime.symbolic_primitives,
            "_navigate_to_pose",
            None,
        )
        synchronize = getattr(
            runtime.symbolic_primitives,
            "sync_marker_kinematic_hold",
            None,
        )
        if not callable(navigation) or not callable(synchronize):
            raise RuntimeError("Task25 private Symbolic station API is unavailable")
        if not callable(observation_provider):
            raise RuntimeError(
                "Task25 private rendered observation API is unavailable"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (desired_distance_m, hard_distance_m)
        ) or not (0.0 < float(desired_distance_m) <= float(hard_distance_m)):
            raise RuntimeError("Task25 private station distances are invalid")

        api = (
            runtime_api
            or getattr(observation_provider, "__self__", None)
            or _default_runtime_api()
        )
        route_provider = private_route_provider or getattr(
            api,
            "_task25_private_station_route_plan",
            None,
        )
        if route_provider is None:
            route = _private_official_station_route_plan(
                evaluator=evaluator,
                target=target,
                desired_distance_m=float(desired_distance_m),
                maximum_station_distance_m=float(hard_distance_m),
                runtime_api=api,
            )
        else:
            route = route_provider(
                evaluator=evaluator,
                target=target,
                desired_distance_m=float(desired_distance_m),
                maximum_station_distance_m=float(hard_distance_m),
            )
        route_private = getattr(route, "private_report", None)
        if not isinstance(route_private, Mapping):
            raise RuntimeError("Task25 official station route report is invalid")
        persist_probe(
            "ROUTE_READY",
            route_status_private=str(getattr(route, "status", "")),
            route_plan_present_private=getattr(route, "plan", None) is not None,
            route_report_private=dict(route_private),
        )
        station_provider = (
            private_station_pose_provider
            or getattr(api, "_task25_private_station_pose", None)
            or getattr(api, "_privileged_symbolic_station_pose", None)
        )
        if not callable(station_provider):
            raise RuntimeError("Task25 official station pose provider is unavailable")
        tx, ty = _entity_xy(target)
        raw_target_pose, planned_distance, station_selection = station_provider(
            target_xy=(tx, ty),
            route=route,
        )
        target_pose = _finite_vector(
            raw_target_pose,
            minimum_size=3,
            label="official station pose",
        )[:3]
        if (
            isinstance(planned_distance, bool)
            or not isinstance(planned_distance, (int, float))
            or not math.isfinite(float(planned_distance))
            or float(planned_distance) <= 0.0
            or not isinstance(station_selection, str)
            or not station_selection
        ):
            raise RuntimeError("Task25 official station selection is invalid")
        persist_probe(
            "STATION_READY",
            station_pose_private=list(target_pose),
            station_target_distance_m_private=float(planned_distance),
            station_selection_private=station_selection,
        )

        robot = getattr(evaluator, "robot", None)
        prior_roll_pitch = _reset_robot_base_tilt(robot)
        clear = getattr(runtime.store, "clear", None)
        if not callable(clear):
            raise RuntimeError("Task25 private setup lacks marker-store reset")
        clear()
        generator = iter(navigation(target_pose))
        yielded = 0
        try:
            while True:
                try:
                    command = next(generator)
                except StopIteration:
                    break
                if command is not None:
                    raise RuntimeError(
                        "Task25 private Symbolic setup returned a public action"
                    )
                yielded += 1
                synchronize()
        finally:
            close = getattr(generator, "close", None)
            if callable(close):
                close()

        robot_x, robot_y, robot_yaw = _entity_xy_yaw(robot)
        distance_m = math.hypot(tx - robot_x, ty - robot_y)
        heading_error_rad = _wrapped_angle(
            math.atan2(ty - robot_y, tx - robot_x) - robot_yaw
        )
        after_roll_pitch = _finite_vector(
            _robot_base_positions(robot)[3:5],
            minimum_size=2,
            label="private setup final roll/pitch",
        )[:2]
        persist_probe(
            "GEOMETRY_READY",
            measured_distance_m_private=distance_m,
            measured_heading_error_rad_private=heading_error_rad,
            base_roll_pitch_before_private_reset_rad=list(prior_roll_pitch),
            base_roll_pitch_after_private_reset_rad=list(after_roll_pitch),
            symbolic_generator_yield_count=yielded,
        )
        if distance_m > hard_distance_m:
            raise RuntimeError("Task25 private station exceeds hard reach")
        if abs(heading_error_rad) > 0.05:
            raise RuntimeError("Task25 private station does not face the target")
        if any(abs(value) > 1e-6 for value in after_roll_pitch):
            raise RuntimeError("Task25 private station retained base tilt")

        observation = observation_provider(runtime)
        provider = visibility_provider or _default_private_target_visibility
        visibility = provider(
            evaluator=evaluator,
            target=target,
            observation=observation,
        )
        if not isinstance(visibility, Mapping):
            raise RuntimeError("Task25 private visibility report is invalid")
        visible_pixels = visibility.get("visible_pixels")
        visibility_pass = bool(
            visibility.get("visible") is True
            and not isinstance(visible_pixels, bool)
            and isinstance(visible_pixels, int)
            and visible_pixels >= _MINIMUM_PRIVATE_VISIBLE_PIXELS
        )
        persist_probe(
            "VISIBILITY_READY" if visibility_pass else "VISIBILITY_FAIL",
            visibility_private=dict(visibility),
            visibility_pass=visibility_pass,
        )
        if not visibility_pass:
            raise RuntimeError("Task25 workbench target is not privately visible")
        report = {
            "status": "PASS",
            "setup_kind": "official_eroded_map_station",
            "setup_action_count": 0,
            "public_action_index_increment": 0,
            "simulator_steps": 0,
            "symbolic_generator_yield_count": yielded,
            "target_pose_private": list(target_pose),
            "distance_m_private": distance_m,
            "heading_error_rad_private": heading_error_rad,
            "maximum_station_distance_m_private": float(hard_distance_m),
            "station_target_distance_m_private": float(planned_distance),
            "station_selection_private": station_selection,
            "route_status_private": str(getattr(route, "status", "")),
            "route_report_private": dict(route_private),
            "base_roll_pitch_before_private_reset_rad": list(prior_roll_pitch),
            "base_roll_pitch_after_private_reset_rad": list(after_roll_pitch),
            "base_tilt_reset_for_marker_view": True,
            "visibility_private": dict(visibility),
            "target_name_private": _entity_name(target),
            "subject_recording_started_after_setup": True,
            "navigation_under_test": False,
            "base_teleport": True,
            "official_challenge_eligible": False,
            "execution_fidelity": "kinematic_research",
        }
        persist_probe("PASS", station_report_private=report)
        return report
    except Exception as exc:
        persist_probe(
            "FAIL",
            failure_phase_private=str(probe.get("status", "STARTED")),
            error_type=type(exc).__name__,
            error=str(exc),
            public_actions_executed=0,
            public_action_index=0,
        )
        raise


def resolve_task25_workbench_fixture(
    evaluator: Any,
    profile: Any,
    *,
    role_provider: Callable[
        [Any, Any], tuple[Mapping[str, Any], Mapping[str, Any]]
    ]
    | None = None,
) -> Task25WorkbenchFixture:
    """Resolve the common live support holding the most Task25 food objects."""

    spec = TASK25_CARRY_ATOMIC_SPEC
    observed_identity = (
        getattr(profile, "task_id", None),
        getattr(profile, "task_name", None),
        getattr(profile, "scene_model", None),
    )
    required_identity = (spec.task_id, spec.task_name, spec.scene_model)
    if observed_identity != required_identity:
        raise RuntimeError("loaded task differs from Task25 atomic spec")
    provider = role_provider or _default_role_provider
    graspables, _containers = provider(evaluator, profile)
    targets = tuple(graspables.values())
    if not targets or len({id(target) for target in targets}) != len(targets):
        raise RuntimeError("Task25 atomic graspable fixtures are invalid")

    scene_objects = _scene_objects(evaluator)
    supported: dict[int, tuple[Any, list[Any]]] = {}
    for target in targets:
        _entity_name(target)
        _entity_xy(target)
        state = _named_state(target, "OnTop")
        get_value = getattr(state, "get_value", None)
        if not callable(get_value):
            continue
        for candidate in scene_objects:
            if candidate is target:
                continue
            try:
                on_top = bool(get_value(candidate))
            except Exception:
                continue
            if not on_top:
                continue
            _entity_name(candidate)
            _entity_aabb(candidate)
            bucket = supported.setdefault(id(candidate), (candidate, []))
            bucket[1].append(target)

    if not supported:
        raise RuntimeError("Task25 atomic workbench has no live OnTop support")
    ranked = sorted(
        supported.values(),
        key=lambda item: (-len(item[1]), _entity_name(item[0])),
    )
    support, support_targets_raw = ranked[0]
    support_targets = tuple(
        sorted(support_targets_raw, key=_entity_name)
    )
    if not support_targets:
        raise RuntimeError("Task25 atomic workbench has no eligible target")
    robot_xy = _robot_xy(evaluator)
    station_target = min(
        support_targets,
        key=lambda target: (
            math.dist(_entity_xy(target), robot_xy),
            _entity_name(target),
        ),
    )
    return Task25WorkbenchFixture(
        support=support,
        station_target=station_target,
        eligible_targets=support_targets,
        support_name_private=_entity_name(support),
        station_target_name_private=_entity_name(station_target),
        eligible_target_names_private=tuple(
            _entity_name(target) for target in support_targets
        ),
        support_aabb_private=_entity_aabb(support),
        station_target_xy_private=_entity_xy(station_target),
    )


def _safe_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_json(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _safe_json(item())
        except Exception:
            pass
    return str(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_safe_json(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _record_stage(api: Any, stages: Path | None, name: str, **fields: Any) -> None:
    if stages is None:
        return
    recorder = getattr(api, "_record_stage", None)
    if callable(recorder):
        recorder(stages, name, **fields)


def _held_object(runtime: Any) -> Any:
    getter = getattr(runtime.symbolic_primitives, "_get_obj_in_hand", None)
    if not callable(getter):
        raise RuntimeError("Task25 atomic runtime lacks internal carry state")
    return getter()


def _assert_carry_boundary(
    runtime: Any,
    *,
    expected_held: bool,
    expected_identity: Any | None,
) -> Any | None:
    held = _held_object(runtime)
    if (held is not None) is not expected_held:
        raise RuntimeError("Task25 atomic internal carry boundary failed")
    if expected_identity is not None and held is not expected_identity:
        raise RuntimeError("Task25 atomic held identity changed")
    snapshot = runtime.carry_tracker.snapshot()
    if not isinstance(snapshot, Mapping):
        raise RuntimeError("Task25 atomic public carry snapshot is invalid")
    if snapshot.get("likely") is not expected_held:
        raise RuntimeError("Task25 atomic public carry boundary failed")
    if expected_held:
        hand = str(getattr(runtime.symbolic_primitives, "arm", ""))
        if hand not in {"left", "right"} or snapshot.get("active_hand") != hand:
            raise RuntimeError("Task25 atomic active hand is inconsistent")
        private_held = getattr(runtime.carry_tracker, "held_object", None)
        if callable(private_held) and private_held() is not held:
            raise RuntimeError("Task25 atomic tracker identity is inconsistent")
        validator = getattr(
            runtime.symbolic_primitives,
            "is_marker_symbolic_hold_valid",
            None,
        )
        if not callable(validator) or validator(hand, held) is not True:
            raise RuntimeError("Task25 atomic symbolic hold is invalid")
    elif snapshot.get("active_hand") is not None:
        raise RuntimeError("Task25 atomic empty carry retained an active hand")
    return held


def _frame_boundary_record(frames: Sequence[Any]) -> dict[str, Any]:
    if len(frames) < 2:
        raise RuntimeError("Task25 atomic action lacks before/after observations")
    before = frames[0]
    after = frames[-1]
    return {
        "frame_count": len(frames),
        "before_simulator_step": getattr(before, "simulator_step", None),
        "after_simulator_step": getattr(after, "simulator_step", None),
        "first_physical_after_present": True,
    }


def _selected_marker_target(runtime: Any, marker: str) -> Any:
    snapshot = getattr(runtime.store, "active_snapshot", None)
    bindings = getattr(snapshot, "bindings", None)
    if not isinstance(bindings, Mapping) or marker not in bindings:
        raise RuntimeError("Task25 atomic selection lacks frozen C1 binding")
    binding = bindings[marker]
    target = getattr(binding, "target_token", None)
    if target is None or getattr(binding, "binding_kind", "object") != "object":
        raise RuntimeError("Task25 atomic C1 binding is not an object")
    return target


def run_action20_task25_atomic_canary(
    *,
    evaluator: Any,
    control: Any,
    profile: Any,
    run_dir: Path,
    stages: Path | None = None,
    spec: Task25CarryAtomicSpec = TASK25_CARRY_ATOMIC_SPEC,
    runtime_api: Any | None = None,
    role_provider: Callable[
        [Any, Any], tuple[Mapping[str, Any], Mapping[str, Any]]
    ]
    | None = None,
    motion_safety_starter: Any | None = None,
    private_visibility_provider: Callable[..., Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Run one private setup followed by the frozen public carry chain."""

    validate_task25_carry_atomic_spec(spec)
    api = runtime_api or _default_runtime_api()
    phase_dir = Path(run_dir) / "strategies" / PHASE_NAME
    phase_dir.mkdir(parents=True, exist_ok=False)
    _record_stage(api, stages, "ACTION20_TASK25_ATOMIC_START")
    result: dict[str, Any] = {
        "schema_version": ACTION20_TASK25_ATOMIC_SPEC_VERSION,
        "runner_version": ACTION20_TASK25_ATOMIC_RUNNER_VERSION,
        "status": "FAIL",
        "task_id": spec.task_id,
        "task_name": spec.task_name,
        "seed": spec.seed,
        "base_execution": spec.base_execution,
        "observation_profile": spec.observation_profile,
        "scene_load_count": 1,
        "public_turns": [],
        "official_challenge_eligible": False,
        "execution_fidelity": "kinematic_research",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(phase_dir / "atomic_spec.json", spec.to_record())
    runtime = None
    try:
        fixture = resolve_task25_workbench_fixture(
            evaluator,
            profile,
            role_provider=role_provider,
        )
        _write_json(
            phase_dir / "private" / "workbench_fixture.json",
            fixture.to_private_record(),
        )
        starter = motion_safety_starter
        if starter is None:
            starter = api._build_full_motion_safety_starter(evaluator)
        runtime = api._build_marker_runtime(
            evaluator=evaluator,
            control=control,
            profile=profile,
            full_actions=True,
            motion_safety_starter=starter,
            suppress_held_object_environment_collisions=False,
            logical_inventory_carry=True,
            base_execution=spec.base_execution,
            observation_profile=spec.observation_profile,
            action_profile="marker_symbolic_action20",
        )
        _assert_carry_boundary(
            runtime,
            expected_held=False,
            expected_identity=None,
        )
        station_report = _private_workbench_front_view_setup(
            runtime=runtime,
            evaluator=evaluator,
            target=fixture.station_target,
            desired_distance_m=spec.preferred_station_m,
            hard_distance_m=spec.hard_reach_m,
            observation_provider=api._capture_rendered_oracle_observation,
            visibility_provider=private_visibility_provider,
            runtime_api=api,
            private_probe_path=(
                phase_dir / "private" / "station_probe.json"
            ),
        )
        if station_report.get("status") != "PASS":
            raise RuntimeError("Task25 atomic station setup did not reach workbench")
        action_index = 0
        setup_steps = 0
        _write_json(
            phase_dir / "private" / "station_setup.json",
            {
                "status": "PASS",
                "simulator_steps": setup_steps,
                "setup_action_count": 0,
                "public_action_index_increment": 0,
                "distance_m_private": station_report["distance_m_private"],
                "hard_reach_m": spec.hard_reach_m,
                "station_report_private": dict(station_report),
                "subject_recording_started_after_setup": True,
            },
        )

        selection = runtime.registry.parse_payload(spec.selection_payload())
        selection_result, selection_frames, selection_steps = api._execute(
            runtime=runtime,
            action=selection,
            action_index=action_index,
            phase_dir=phase_dir,
        )
        if selection_result.status != spec.selection_status:
            raise RuntimeError("Task25 atomic pick selection was not available")
        _assert_carry_boundary(
            runtime,
            expected_held=False,
            expected_identity=None,
        )
        selected_target = _selected_marker_target(
            runtime,
            spec.selection_marker,
        )
        if not any(
            selected_target is candidate for candidate in fixture.eligible_targets
        ):
            raise RuntimeError("Task25 atomic C1 is outside the workbench fixture")
        result["public_turns"].append(
            {
                "kind": "selection",
                "action": spec.selection_action,
                "status": selection_result.status,
                "simulator_steps": selection_steps,
                "frames": _frame_boundary_record(selection_frames),
            }
        )
        action_index += 1

        held_identity = None
        total_subject_steps = 0
        for turn in spec.turns:
            before_identity = _assert_carry_boundary(
                runtime,
                expected_held=turn.held_before,
                expected_identity=(
                    held_identity if turn.preserve_held_identity else None
                ),
            )
            action = runtime.registry.parse_payload(turn.payload())
            action_result, frames, used_steps = api._execute(
                runtime=runtime,
                action=action,
                action_index=action_index,
                phase_dir=phase_dir,
            )
            if action_result.status not in turn.allowed_statuses:
                raise RuntimeError(
                    f"Task25 atomic {turn.action} returned a disallowed status"
                )
            expected_identity = None
            if turn.action == "pick":
                expected_identity = selected_target
            elif turn.preserve_held_identity:
                expected_identity = before_identity
            after_identity = _assert_carry_boundary(
                runtime,
                expected_held=turn.held_after,
                expected_identity=expected_identity,
            )
            if turn.held_after:
                held_identity = after_identity
            else:
                held_identity = None
            result["public_turns"].append(
                {
                    "kind": "subject",
                    "ordinal": turn.ordinal,
                    "action": turn.action,
                    "status": action_result.status,
                    "public_feedback": dict(action_result.public_feedback),
                    "simulator_steps": used_steps,
                    "held_after": turn.held_after,
                    "held_identity_preserved": bool(
                        turn.preserve_held_identity
                    ),
                    "frames": _frame_boundary_record(frames),
                }
            )
            total_subject_steps += used_steps
            action_index += 1

        if held_identity is not None or _held_object(runtime) is not None:
            raise RuntimeError(
                "Task25 atomic floor placement did not clear internal carry"
            )
        result.update(
            {
                "status": "PASS",
                "setup_simulator_steps": setup_steps,
                "setup_action_count": 0,
                "setup_action_index_increment": 0,
                "selection_simulator_steps": selection_steps,
                "subject_simulator_steps": total_subject_steps,
                "selected_marker": spec.selection_marker,
                "selected_target_private": _entity_name(selected_target),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _record_stage(api, stages, "ACTION20_TASK25_ATOMIC_PASS")
        _write_json(phase_dir / "summary.json", result)
        return result
    except Exception as exc:
        result.update(
            {
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _record_stage(
            api,
            stages,
            "ACTION20_TASK25_ATOMIC_FAIL",
            error_type=type(exc).__name__,
        )
        _write_json(phase_dir / "summary.json", result)
        raise
    finally:
        if runtime is not None:
            close = getattr(runtime.environment, "close", None)
            if callable(close):
                close()


def make_action20_task25_atomic_callback(
    *,
    profile: Any | None = None,
    spec: Task25CarryAtomicSpec = TASK25_CARRY_ATOMIC_SPEC,
    runtime_api: Any | None = None,
    role_provider: Callable[
        [Any, Any], tuple[Mapping[str, Any], Mapping[str, Any]]
    ]
    | None = None,
    private_visibility_provider: Callable[..., Mapping[str, Any]] | None = None,
) -> Callable[..., Mapping[str, Any]]:
    """Return a callback compatible with the existing one-load harness."""

    validate_task25_carry_atomic_spec(spec)
    resolved_profile = profile
    if resolved_profile is None:
        profiles = importlib.import_module(
            "behavior_hl.embodied.evaluator_bridge.inside_task_profiles"
        )
        resolved_profile = profiles.get_inside_task_profile(spec.task_name)

    def callback(
        *,
        evaluator: Any,
        control: Any,
        bindings: Any,
        run_dir: Path,
        stages: Path,
    ) -> Mapping[str, Any]:
        del bindings
        summary = run_action20_task25_atomic_canary(
            evaluator=evaluator,
            control=control,
            profile=resolved_profile,
            run_dir=run_dir,
            stages=stages,
            spec=spec,
            runtime_api=runtime_api,
            role_provider=role_provider,
            private_visibility_provider=private_visibility_provider,
        )
        success = summary.get("status") == "PASS"
        return {
            "schema_version": ACTION20_TASK25_ATOMIC_SPEC_VERSION,
            "strategies": [summary],
            "action20_task25_atomic_success": success,
            "oracle_success": success,
            "scene_load_count": 1,
            "model_lane_authorized": False,
            "qwen_attempted": False,
            "official_challenge_eligible": False,
            "execution_fidelity": "kinematic_research",
        }

    return callback


__all__ = [
    "ACTION20_TASK25_ATOMIC_RUNNER_VERSION",
    "PHASE_NAME",
    "Task25WorkbenchFixture",
    "_private_official_station_route_plan",
    "_private_workbench_front_view_setup",
    "make_action20_task25_atomic_callback",
    "resolve_task25_workbench_fixture",
    "run_action20_task25_atomic_canary",
]
