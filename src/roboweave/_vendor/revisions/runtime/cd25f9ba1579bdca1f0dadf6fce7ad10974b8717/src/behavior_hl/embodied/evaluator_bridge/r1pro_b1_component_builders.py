"""Concrete B1 component adapters for the resident composite factory.

All simulator-facing dependencies are injected.  The adapters select a
starter from the action's public hand argument, adapt a labelled controller
step to one resident ``step_once`` call, and return only service keys consumed
by the selected plugin.  No adapter invents a hand, pose, or target.
"""

from __future__ import annotations

import math
from time import perf_counter
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from behavior_hl.embodied.evaluator_bridge.r1pro_actuator_binding import (
    R1PRO_LOCAL_BASE_BINDING_VERSION,
    R1ProActuatorBindingError,
    build_r1pro_actuator_bindings,
    build_r1pro_local_base_binding,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_b1_service_factory import (
    R1PRO_B1_COMPONENT_VERSIONS,
    PublicCarryProxySource,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_grasp_binding import (
    R1PRO_GRASP_DRIVER_VERSION,
    RobotPose,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_joint_stall_detector import (
    notify_r1pro_command,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_grasp_service_factory import (
    build_resident_r1pro_grasp_service_bundle,
)
from behavior_hl.embodied.evaluator_bridge.private_head_segmentation_side_channel import (
    PrivateHeadSegmentationSideChannel,
    PrivateHeadSegmentationUnavailableError,
)
from behavior_hl.embodied.evaluator_bridge.private_observation_ingress import (
    HEAD_INSTANCE_SEGMENTATION_KEY,
    HEAD_RGB_KEY,
    capture_private_head_frame,
)
from behavior_hl.embodied.evaluator_bridge.marker_symbolic_grasp_backend import (
    AUTOMATIC_INTERACTION_APPROACH_OPERATIONS,
    MARKER_SYMBOLIC_BACKEND_VERSION,
    MarkerInputUnavailableError,
    MarkerSymbolicGraspBackendAdapter,
    SymbolicPreparationResult,
    official_instance_bboxes,
)
from behavior_hl.embodied.evaluator_bridge.action23_outcome_verifier import (
    Action23OutcomeVerifier,
)
from behavior_hl.embodied.evaluator_bridge.logical_inventory import (
    LogicalInventoryRelation,
)
from behavior_hl.embodied.evaluator_bridge.marker_symbolic_spatial_execution import (
    plan_live_under_supports,
)
from behavior_hl.embodied.evaluator_bridge.interaction_station_geometry import (
    nearest_interaction_target_xy,
)
from behavior_hl.embodied.evaluator_bridge.nested_container_operations import (
    resolve_openable_container_ancestry,
)
from behavior_hl.embodied.evaluator_bridge.private_visual_markers import (
    PickBundleTargetToken,
    PrivateVisualMarkerStore,
    SpatialMarkerTargetToken,
    current_target_visibility,
)
from behavior_hl.embodied.evaluator_bridge.private_drawer_parts import (
    ArticulatedPartTransactionBoundary,
    DrawerPartBlockedError,
    DrawerPartTargetToken,
    drawer_part_tokens,
    move_drawer_part_incrementally,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_visible_official_grasp_factory import (
    build_r1pro_visible_official_grasp_services,
    planar_base_pose_from_robot,
    private_picking_up_toys_eligible_objects,
    private_picking_up_toys_eligible_objects_by_category,
    private_picking_up_toys_marker_objects,
    private_picking_up_toys_marker_roles,
)
from behavior_hl.embodied.evaluator_bridge.visible_official_grasp_backend import (
    VISIBLE_OFFICIAL_GRASP_BACKEND_VERSION,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_place_services import (
    LocalRGBPatchAssessment,
    R1PRO_PLACE_SERVICES_VERSION,
    R1ProPlaceMotionLimits,
    build_r1pro_place_service_bundle,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_starter_grasp_driver import (
    PublicGripperWristEvidence,
    StarterGraspPoseConfig,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_torso_release_binding import (
    build_r1pro_torso_release_bindings,
)
from behavior_hl.embodied.evaluator_bridge.spatial_candidate_admission import (
    query_spatial_candidate_collision_overlap,
)
from behavior_hl.embodied.evaluator_bridge.spatial_symbolic_actions import (
    DEFAULT_MAX_ANGULAR_SPEED_RAD_S,
    DEFAULT_MAX_LINEAR_SPEED_M_S,
    DEFAULT_SOLID_OVERLAP_TOLERANCE_M,
    SpatialPoseCandidate,
    object_aabb,
    object_pose,
    placement_predicate_mode,
    read_action_relation,
    state_by_name,
    symbolic_on_top_relation,
)
from behavior_hl.embodied.policy.protocol import (
    ActionEnvelope,
    MarkerGraspArgs,
    MarkerPlaceInsideArgs,
    MarkerTargetOrRequestArgs,
    MarkerTargetArgs,
    NoArgs,
    PointActionArgs,
    ReleaseArgs,
    VisibleObjectGraspArgs,
)
from behavior_hl.embodied.policy.profiles import ACTION20_LOCAL_WORKSPACE_M
from behavior_hl.embodied.policy.hold_policy import (
    LEFT_ARM_QPOS,
    RIGHT_ARM_QPOS,
    TRUNK_QPOS,
)
from behavior_hl.embodied.policy.observations import (
    CAMERA_SENSOR_NAMES,
    PROPRIO_DIM,
)
from behavior_hl.embodied.policy.r1pro.action_layout import (
    R1PRO_ACTION_LAYOUT,
)
from behavior_hl.embodied.runtime.b1_dispatcher_factory import (
    CARRY_COMPENSATOR_SERVICE,
)
from behavior_hl.embodied.runtime.marker_symbolic_carry_contract import (
    MarkerSymbolicCarryContractError,
    MarkerSymbolicCarryPhase,
    require_marker_symbolic_carry_phase,
)
from behavior_hl.embodied.runtime.public_action_feedback import (
    EXECUTION_UNVERIFIED,
    OUT_OF_REACH,
    SELECTION_UNAVAILABLE,
)
from behavior_hl.embodied.runtime.plugins.grasp import (
    PHYSICAL_GRASP_BACKEND_SERVICE,
    PRIVATE_GRASP_AUDIT_SERVICE,
)
from behavior_hl.embodied.runtime.plugins.visible_official_grasp import (
    VISIBLE_OFFICIAL_GRASP_BACKEND_SERVICE,
)
from behavior_hl.embodied.runtime.plugins.marker_symbolic_grasp import (
    MARKER_SYMBOLIC_BACKEND_SERVICE,
)
from behavior_hl.embodied.runtime.plugins.local_base import (
    LOCAL_BASE_ACTUATOR_SERVICE,
)
from behavior_hl.embodied.runtime.plugins.place import (
    MOTION_SERVICE,
    RGBD_SOURCE_SERVICE,
    VISUAL_ASSESSOR_SERVICE,
)
from behavior_hl.embodied.runtime.plugins.release import (
    PHYSICAL_RELEASE_ACTUATOR_SERVICE,
)
from behavior_hl.embodied.runtime.plugins.torso import (
    PHYSICAL_TORSO_ACTUATOR_SERVICE,
)


CARRY_ACTUATOR_SERVICE = "carry_actuator"
CARRY_PRIVATE_AUDIT_SERVICE = "carry_private_audit"

_BASE_ACTIONS = frozenset(
    {
        "move_backward",
        "move_forward",
        "move_left",
        "move_right",
        "turn_left",
        "turn_right",
    }
)
_LOOK_ACTIONS = frozenset({"look_up", "look_center", "look_down"})
_VERTICAL_ACTIONS = frozenset({"move_up", "move_down"})
_WAIT_ACTIONS = frozenset({"wait"})
_MARKERLESS_SYMBOLIC_ACTIONS = frozenset(
    {"put_back", "release", "place_on_floor"}
)

# Shared logical-carry ceiling for single objects and relation bundles.
ACTION20_SYMBOLIC_PICK_MAX_MASS_KG = 10.0

_NON_OBJECT_SUPPORT_CATEGORIES = frozenset(
    {"ceiling", "floor", "floors", "ground", "wall", "walls"}
)


class R1ProB1ComponentBuilderError(RuntimeError):
    """Raised before a component can issue a simulator command."""


class R1ProB1ArticulatedPartBlockedError(R1ProB1ComponentBuilderError):
    """Raised after an articulated part returns to its last safe pose."""

    blocked_category = "articulated_part_blocked"


def _live_movable_objects(
    scene_objects: Sequence[Any],
) -> dict[str, Any]:
    """Return uniquely named live rigid objects that the carry can pose."""

    movable: dict[str, Any] = {}
    for target in scene_objects:
        fixed = getattr(target, "fixed_base", None)
        if fixed is None:
            fixed = getattr(target, "_fixed_base", False)
        name = getattr(target, "name", None)
        if (
            bool(fixed)
            or not isinstance(name, str)
            or not name
            or name in movable
            or not callable(getattr(target, "get_position_orientation", None))
            or not callable(getattr(target, "set_position_orientation", None))
        ):
            continue
        try:
            lower, upper = object_aabb(target)
        except Exception:
            continue
        extent = upper - lower
        if not np.isfinite(extent).all() or np.any(extent <= 0.0):
            continue
        movable[name] = target
    return movable


def _current_visible_scene_objects(
    scene_objects: Sequence[Any],
    frame: Any,
) -> tuple[Any, ...]:
    """Return objects with current object- or visual-mesh pixels."""

    return tuple(
        target
        for target in scene_objects
        if current_target_visibility(
            frame=frame,
            target=target,
            minimum_visible_pixels=1,
        )[0]
    )


def _symbolic_pick_capacity_eligible(target: Any) -> bool:
    """Apply the total-payload limit for logical carry."""

    try:
        mass = float(target.mass)
    except Exception:
        return False
    return bool(
        math.isfinite(mass)
        and 0.0 < mass <= ACTION20_SYMBOLIC_PICK_MAX_MASS_KG
    )


def _bundle_fits_symbolic_pick_capacity(members: Sequence[Any]) -> bool:
    try:
        mass = sum(float(member.mass) for member in members)
    except Exception:
        return False
    return bool(
        math.isfinite(mass)
        and 0.0 < mass <= ACTION20_SYMBOLIC_PICK_MAX_MASS_KG
    )


def _live_pickable_objects(
    scene_objects: Sequence[Any],
) -> dict[str, Any]:
    """Return movable objects accepted by the logical-carry capacity gate."""

    return {
        name: target
        for name, target in _live_movable_objects(scene_objects).items()
        if _symbolic_pick_capacity_eligible(target)
    }


def _live_accessible_pickable_objects(
    scene_objects: Sequence[Any],
    pickable_objects: Mapping[str, Any],
) -> dict[str, Any]:
    """Exclude targets whose live openable containment ancestry is closed."""

    objects_by_name: dict[str, Any] = {}
    for target in scene_objects:
        name = getattr(target, "name", None)
        if not isinstance(name, str) or not name:
            continue
        if name in objects_by_name and objects_by_name[name] is not target:
            return {}
        objects_by_name[name] = target

    def is_inside(child_name: str, parent_name: str) -> bool:
        state = state_by_name(objects_by_name[child_name], "Inside")
        if state is None:
            return False
        getter = getattr(state, "get_value", None)
        if not callable(getter):
            raise RuntimeError("official Inside state is unreadable")
        return getter(objects_by_name[parent_name])

    accessible: dict[str, Any] = {}
    for name, target in pickable_objects.items():
        if objects_by_name.get(name) is not target:
            continue
        ancestry = resolve_openable_container_ancestry(
            scope_names=objects_by_name,
            child_scope_name=name,
            is_openable=lambda candidate: (
                state_by_name(objects_by_name[candidate], "Open") is not None
            ),
            is_inside=is_inside,
        )
        if not ancestry.success:
            continue
        blocked = False
        for ancestor_name in ancestry.outer_to_inner:
            open_state = state_by_name(objects_by_name[ancestor_name], "Open")
            getter = getattr(open_state, "get_value", None)
            if not callable(getter):
                blocked = True
                break
            try:
                observed = getter()
            except Exception:
                blocked = True
                break
            if not isinstance(observed, (bool, np.bool_)) or not bool(observed):
                blocked = True
                break
        if not blocked:
            accessible[name] = target
    return accessible


def _live_relation(child: Any, kind: str, parent: Any) -> bool:
    state_name = {"on_top": "OnTop", "attached_to": "AttachedTo"}[kind]

    def one_way(first: Any, second: Any) -> bool:
        state = state_by_name(first, state_name)
        getter = getattr(state, "get_value", None)
        if not callable(getter):
            return False
        try:
            return bool(getter(second))
        except Exception:
            return False

    return one_way(child, parent) or (
        kind == "attached_to" and one_way(parent, child)
    )


def _stable_on_top(child: Any, parent: Any) -> bool:
    if not _live_relation(child, "on_top", parent):
        return False
    speeds = []
    for method_name in ("get_linear_velocity", "get_angular_velocity"):
        method = getattr(child, method_name, None)
        if not callable(method):
            return False
        try:
            value = method()
            for callback_name in ("detach", "cpu"):
                callback = getattr(value, callback_name, None)
                if callable(callback):
                    value = callback()
            callback = getattr(value, "numpy", None)
            if callable(callback):
                value = callback()
            speeds.append(float(np.linalg.norm(np.asarray(value, dtype=float))))
        except Exception:
            return False
    return bool(
        math.isfinite(speeds[0])
        and math.isfinite(speeds[1])
        and speeds[0] <= DEFAULT_MAX_LINEAR_SPEED_M_S
        and speeds[1] <= DEFAULT_MAX_ANGULAR_SPEED_RAD_S
    )


def _live_pick_bundle_targets(
    movable_objects: Mapping[str, Any],
    pickable_objects: Mapping[str, Any],
) -> tuple[tuple[PickBundleTargetToken, ...], frozenset[str]]:
    """Build AttachedTo assemblies and maximal stable-OnTop closures."""

    objects = tuple(movable_objects.values())
    if len(objects) < 2:
        return (), frozenset()
    for target in objects:
        clearer = getattr(target, "clear_states_cache", None)
        if callable(clearer):
            try:
                clearer()
            except Exception:
                return (), frozenset()

    parent = list(range(len(objects)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    attached_relations: list[LogicalInventoryRelation] = []
    for child_index, child in enumerate(objects):
        for parent_index in range(child_index + 1, len(objects)):
            other = objects[parent_index]
            if _live_relation(child, "attached_to", other):
                union(child_index, parent_index)
                attached_relations.append(
                    LogicalInventoryRelation("attached_to", child, other)
                )

    components: dict[int, list[Any]] = {}
    for index, target in enumerate(objects):
        components.setdefault(find(index), []).append(target)
    component_by_object = {
        id(target): component_id
        for component_id, members in components.items()
        for target in members
    }
    attached_by_component: dict[int, list[LogicalInventoryRelation]] = {}
    for relation in attached_relations:
        attached_by_component.setdefault(
            component_by_object[id(relation.child)],
            [],
        ).append(relation)
    eligible_components = {
        component_id
        for component_id, members in components.items()
        if all(
            pickable_objects.get(str(getattr(target, "name", ""))) is target
            for target in members
        )
    }
    roots = {
        component_id: min(
            members,
            key=lambda target: str(getattr(target, "name", "")),
        )
        for component_id, members in components.items()
        if component_id in eligible_components
    }

    targets: list[PickBundleTargetToken] = []
    blocked_names: set[str] = set()
    for component_id, members in components.items():
        if len(members) < 2:
            continue
        root = roots.get(component_id)
        if root is None or not _bundle_fits_symbolic_pick_capacity(members):
            blocked_names.update(
                str(getattr(member, "name", "")) for member in members
            )
            continue
        relations = tuple(attached_by_component[component_id])
        targets.append(PickBundleTargetToken(root, relations, exclusive=True))

    on_top_by_child_component: dict[
        int, dict[int, list[LogicalInventoryRelation]]
    ] = {}
    for child in objects:
        child_component = component_by_object[id(child)]
        if child_component not in eligible_components:
            continue
        for support in pickable_objects.values():
            support_component = component_by_object[id(support)]
            if (
                support_component not in eligible_components
                or child_component == support_component
                or not _stable_on_top(child, support)
            ):
                continue
            on_top_by_child_component.setdefault(child_component, {}).setdefault(
                support_component,
                [],
            ).append(LogicalInventoryRelation("on_top", child, support))

    parent_edge_by_child: dict[int, tuple[int, LogicalInventoryRelation]] = {}
    children_by_parent: dict[int, list[tuple[int, LogicalInventoryRelation]]] = {}
    for child_component, by_parent in on_top_by_child_component.items():
        if len(by_parent) != 1:
            continue
        parent_component, relations = next(iter(by_parent.items()))
        relation = min(
            relations,
            key=lambda edge: (
                str(getattr(edge.parent, "name", "")),
                str(getattr(edge.child, "name", "")),
            ),
        )
        parent_edge_by_child[child_component] = (parent_component, relation)
        children_by_parent.setdefault(parent_component, []).append(
            (child_component, relation)
        )

    for root_component in sorted(children_by_parent):
        root = roots.get(root_component)
        if root is None or root_component in parent_edge_by_child:
            continue
        relations: list[LogicalInventoryRelation] = []
        visiting: set[int] = set()

        def visit(component_id: int) -> None:
            if component_id in visiting:
                raise ValueError("cyclic stable OnTop bundle")
            visiting.add(component_id)
            relations.extend(attached_by_component.get(component_id, ()))
            for child_component, on_top in children_by_parent.get(component_id, ()):
                relations.append(on_top)
                visit(child_component)
            visiting.remove(component_id)

        try:
            visit(root_component)
            token = PickBundleTargetToken(root, tuple(relations))
        except ValueError:
            continue
        if _bundle_fits_symbolic_pick_capacity(token.member_tokens):
            targets.append(token)
    return (
        tuple(
            sorted(
                targets,
                key=lambda token: (
                    str(getattr(token.root_token, "name", "")),
                    not token.exclusive,
                    token.member_names,
                ),
            )
        ),
        frozenset(blocked_names),
    )


def _generic_on_top_support_objects(
    scene_objects: Sequence[Any],
    held: Any,
) -> dict[str, Any]:
    """Let live geometry, rather than an object-category whitelist, find supports."""

    supports: dict[str, Any] = {}
    for target in scene_objects:
        category = str(getattr(target, "category", "")).casefold()
        if (
            target is held
            or category in _NON_OBJECT_SUPPORT_CATEGORIES
        ):
            continue
        name = str(getattr(target, "name", ""))
        if not name:
            continue
        supports[name] = target
    return supports


def _under_marker_target_objects(
    candidates: Mapping[str, Any],
    *,
    held: Any,
    scene_objects: Sequence[Any],
    robot: Any,
    logical_snapshot: Any | None,
    task_fallback_names: Iterable[str] = (),
    symbolic_support: bool = False,
) -> dict[str, Any]:
    """Offer ray-verified targets plus task references; execution still checks fit."""

    if held is None:
        return {}
    held_ids = {id(held)} | {
        id(obj) for obj in getattr(logical_snapshot, "object_refs", ())
    }
    fallback_names = frozenset(task_fallback_names)
    fallback_objects = {
        str(obj.name): obj
        for obj in scene_objects
        if str(getattr(obj, "name", "")) in fallback_names
        and id(obj) not in held_ids
    }
    return {
        name: target
        for name, target in {**candidates, **fallback_objects}.items()
        if id(target) not in held_ids
        and (
            name in fallback_objects
            or any(
                plan.candidate_plan.candidates
                for plan in plan_live_under_supports(
                    held=held,
                    target=target,
                    supports=scene_objects,
                    robot=robot,
                    logical_snapshot=logical_snapshot,
                    symbolic_support=symbolic_support,
                    stop_after_first_usable=True,
                )
            )
        )
    }


def _resolve_spatial_marker_target(
    target: Any,
    operation: str,
) -> tuple[Any, str | None]:
    """Unwrap a current-frame selector without exposing it publicly."""

    if not isinstance(target, SpatialMarkerTargetToken):
        return target, None
    if target.operation != operation:
        raise R1ProB1ComponentBuilderError(
            "spatial marker operation does not match executor"
        )
    return target.target_token, target.selector


def _spatial_marker_region(
    target: Any,
    operation: str,
) -> tuple[
    tuple[float, float] | None,
    tuple[float, float] | None,
    float | None,
]:
    if not isinstance(target, SpatialMarkerTargetToken):
        return None, None, None
    if target.operation != operation:
        raise R1ProB1ComponentBuilderError(
            "spatial marker operation does not match executor"
        )
    return (
        target.region_center_xy,
        target.region_extent_xy,
        target.surface_z_m,
    )


def _resolve_marker_geometry_target(target: Any, operation: str) -> Any:
    """Resolve the live object that owns the world pose for reach checks."""

    token_type = type(target).__name__
    if isinstance(target, PickBundleTargetToken):
        target = target.target_token
    if isinstance(target, SpatialMarkerTargetToken):
        target, _ = _resolve_spatial_marker_target(target, operation)
    if isinstance(target, DrawerPartTargetToken):
        target = target.target_token
    if not callable(getattr(target, "get_position_orientation", None)):
        error = MarkerInputUnavailableError(
            "marker geometry target has no live world pose"
        )
        error.private_report = {
            "phase": "marker_geometry_resolution", "operation": operation,
            "marker_token_type_private": token_type,
            "resolved_object_type_private": type(target).__name__,
        }
        raise error
    return target


def _nearest_target_xy(
    target: Any,
    observer_xy: Any,
) -> tuple[np.ndarray, str]:
    """Return the point on the target footprint facing the observer."""

    observer = np.asarray(observer_xy, dtype=np.float64).reshape(-1)
    if observer.shape != (2,) or not np.isfinite(observer).all():
        raise R1ProB1ComponentBuilderError("marker observer XY is invalid")
    bounds = getattr(target, "aabb", None)
    if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
        lower = np.asarray(bounds[0], dtype=np.float64).reshape(-1)
        upper = np.asarray(bounds[1], dtype=np.float64).reshape(-1)
        if (
            lower.shape == (3,)
            and upper.shape == (3,)
            and np.isfinite(lower).all()
            and np.isfinite(upper).all()
            and np.all(lower <= upper)
        ):
            nearest, source = nearest_interaction_target_xy(
                observer,
                (float(lower[0]), float(lower[1])),
                target_aabb_xy=(
                    float(lower[0]),
                    float(lower[1]),
                    float(upper[0]),
                    float(upper[1]),
                ),
            )
            return np.asarray(nearest, dtype=np.float64), source
    position, _ = target.get_position_orientation()
    position = np.asarray(position, dtype=np.float64).reshape(-1)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise MarkerInputUnavailableError(
            "marker target position is unavailable"
        )
    nearest, source = nearest_interaction_target_xy(observer, position[:2])
    return np.asarray(nearest, dtype=np.float64), source


def _articulation_pose_diagnostic(target: DrawerPartTargetToken) -> Mapping[str, Any]:
    """Additional read-only evidence must not change an Action verdict."""
    try:
        return {"selected_open": target.is_open(), "parts": tuple(
            {"selector": part.selector, "position": part.current_position(), "open": part.is_open()}
            for part in drawer_part_tokens(target.target_token))}
    except Exception as exc:
        return {"selected_open": None, "parts": (), "error": f"{type(exc).__name__}: {exc}"}


def _read_marker_symbolic_canonical_state(
    obj: Any,
    state_name: str,
    other: Any | None,
    *,
    symbolic_on_top: bool = False,
) -> bool:
    """Read canonical state after unwrapping private Marker selectors.

    A drawer-part Marker denotes one live joint rather than a scene object
    with an OmniGibson ``states`` registry.  Spatial placement Markers likewise
    wrap the real relation target.  The executor keeps those wrappers private,
    while semantic verification reads the selected joint or underlying object.
    """

    if (
        isinstance(obj, DrawerPartTargetToken)
        and state_name == "Open"
        and other is None
    ):
        return bool(obj.is_open())
    while isinstance(
        obj,
        (SpatialMarkerTargetToken, DrawerPartTargetToken, PickBundleTargetToken),
    ):
        obj = obj.target_token
    while isinstance(
        other,
        (SpatialMarkerTargetToken, DrawerPartTargetToken, PickBundleTargetToken),
    ):
        other = other.target_token
    if symbolic_on_top:
        return read_action_relation(obj, state_name, other, predicate_mode="action20_symbolic")
    states = getattr(obj, "states", None)
    if not isinstance(states, Mapping):
        raise R1ProB1ComponentBuilderError(
            "action target has no official state registry"
        )
    for state_type, state in states.items():
        if getattr(state_type, "__name__", "") != state_name:
            continue
        get_value = getattr(state, "get_value", None)
        if not callable(get_value):
            raise R1ProB1ComponentBuilderError(
                f"official {state_name} state is unreadable"
            )
        return bool(get_value() if other is None else get_value(other))
    raise R1ProB1ComponentBuilderError(
        f"official {state_name} state is unavailable"
    )


def _single_articulated_part_token(target: Any) -> DrawerPartTargetToken:
    state = next(
        (
            value
            for state_type, value in getattr(target, "states", {}).items()
            if getattr(state_type, "__name__", "") == "Open"
        ),
        None,
    )
    info = getattr(state, "relevant_joints_info", None)
    if not isinstance(info, (tuple, list)) or len(info) != 3:
        raise R1ProB1ComponentBuilderError(
            "official Open joint metadata is unavailable"
        )
    _, joints, directions = info
    if len(joints) != 1 or len(directions) != 1:
        raise R1ProB1ComponentBuilderError(
            "whole-object open/close requires exactly one articulated part"
        )
    joint = joints[0]
    body1 = str(getattr(joint, "body1", ""))
    link = next(
        (
            value
            for value in getattr(target, "links", {}).values()
            if str(getattr(value, "prim_path", "")) == body1
        ),
        None,
    )
    if link is None:
        raise R1ProB1ComponentBuilderError(
            "official Open moving link is unavailable"
        )
    return DrawerPartTargetToken(
        object_token=target,
        joint_token=joint,
        link_token=link,
        joint_direction=int(directions[0]),
    )


def _entity_prim_roots(value: Any) -> frozenset[str]:
    roots = {
        str(getattr(link, "prim_path", "")).rstrip("/")
        for link in getattr(value, "links", {}).values()
    }
    prim_path = str(getattr(value, "prim_path", "")).rstrip("/")
    if prim_path:
        roots.add(prim_path)
    return frozenset(root for root in roots if root.startswith("/"))


def _articulated_part_external_contact_pairs(
    token: DrawerPartTargetToken,
    ignored_objects: Iterable[Any],
) -> frozenset[tuple[str, str]]:
    """Return current contacts from one moving link to external objects."""

    from omnigibson.utils.usd_utils import RigidContactAPI

    excluded = set(_entity_prim_roots(token.target_token))
    for obj in ignored_objects:
        excluded.update(_entity_prim_roots(obj))

    def excluded_path(path: str) -> bool:
        return any(path == root or path.startswith(f"{root}/") for root in excluded)

    pairs = RigidContactAPI.get_contact_pairs(
        scene_idx=token.target_token.scene.idx,
        query_set=[token.link_token],
        with_set=None,
        current_only=False,
    )
    return frozenset(
        (str(query_path), str(other_path))
        for query_path, other_path in pairs
        if not excluded_path(str(other_path))
    )


def _articulated_obstruction_detail(report, *, robot, target, scene_objects):
    """Classify known live identities, never object names or exception prose."""
    target = getattr(target, "target_token", target)
    try:
        robot_roots = _entity_prim_roots(robot)
        external_roots = frozenset(
            path for obj in scene_objects if obj is not robot and obj is not target
            for path in _entity_prim_roots(obj)
        ) - robot_roots
    except Exception:
        return None  # Missing diagnostic identities must not change execution.
    causes = set()
    if report.get("obstructing_objects_private"):
        causes.add("contents_obstruction")
    for pair in report.get("blocking_contact_pairs_private", ()) or ():
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            return None
        def matches(roots):
            return any(isinstance(value, str) and (value == root or value.startswith(root + "/"))
                       for value in pair for root in roots)
        if matches(robot_roots):
            causes.add("robot_obstruction")
        elif matches(external_roots):
            causes.add("external_obstruction")
        else:
            return None
    return "multiple_obstructions" if len(causes) > 1 else next(iter(causes), None)


def _articulated_part_symbolic_overlap_pairs(
    token: DrawerPartTargetToken,
    ignored_objects: Iterable[Any],
) -> frozenset[tuple[str, str]]:
    """Return live overlap blockers for the current zero-physics link pose."""

    import omnigibson as og

    raw_aabb = getattr(token.link_token, "visual_aabb", None)
    raw_aabb = raw_aabb() if callable(raw_aabb) else raw_aabb
    if not isinstance(raw_aabb, (tuple, list)) or len(raw_aabb) != 2:
        raise RuntimeError("articulated link visual AABB is unavailable")
    position, orientation = object_pose(token.link_token)
    ignored = tuple(ignored_objects)
    result = query_spatial_candidate_collision_overlap(
        omnigibson=og,
        held=token.link_token,
        closure_objects=(token.target_token, *ignored),
        candidate=SpatialPoseCandidate(
            name=f"{token.selector}:symbolic_sweep",
            position=position,
            orientation_xyzw=orientation,
        ),
        candidate_aabb_world=(raw_aabb[0], raw_aabb[1]),
        xy_margin_m=0.0,
        xy_inset_m=DEFAULT_SOLID_OVERLAP_TOLERANCE_M,
        z_inset_m=DEFAULT_SOLID_OVERLAP_TOLERANCE_M,
    )
    if result.query_error is not None or result.query_box is None:
        raise RuntimeError(result.query_error or result.reason)
    pairs = set(
        (
            str(hit.rigid_body_path or ""),
            str(hit.collision_path or ""),
        )
        for hit in result.blocking_hits
    )
    for obj in ignored:
        position, orientation = object_pose(obj)
        content_report = query_spatial_candidate_collision_overlap(
            omnigibson=og, held=obj,
            closure_objects=(token.target_token, *ignored),
            candidate=SpatialPoseCandidate(
                name=f"{token.selector}:carried_content",
                position=position, orientation_xyzw=orientation,
            ),
            xy_margin_m=0.0,
            xy_inset_m=DEFAULT_SOLID_OVERLAP_TOLERANCE_M,
            z_inset_m=DEFAULT_SOLID_OVERLAP_TOLERANCE_M,
        )
        if content_report.query_error is not None or content_report.query_box is None:
            raise RuntimeError(content_report.query_error or content_report.reason)
        pairs.update(
            (str(hit.rigid_body_path or ""), str(hit.collision_path or ""))
            for hit in content_report.blocking_hits
        )
    return frozenset(pairs)


def _open_marker_symbolic_target(
    *,
    target: Any,
    scene_objects: Iterable[Any],
    external_contact_pairs: Callable[
        [DrawerPartTargetToken, Iterable[Any]],
        frozenset[tuple[str, str]],
    ] = _articulated_part_external_contact_pairs,
    physics_step_once: Callable[[], int] | None = None,
    zero_physics: bool = False,
    render_once: Callable[[], Any] | None = None,
):
    """Move one official Open joint to its farthest collision-free position."""

    token = _single_articulated_part_token(target)
    return (
        yield from move_drawer_part_incrementally(
            token,
            opened=True,
            scene_objects=scene_objects,
            external_contact_pairs=lambda contents: external_contact_pairs(
                token,
                contents,
            ),
            physics_step_once=physics_step_once,
            zero_physics=zero_physics,
            render_once=render_once,
        )
    )


class _LabeledStepOnce:
    """Convert one labelled binding step into exactly one resident step."""

    def __init__(self, step_once: Callable[[], int]) -> None:
        if not callable(step_once):
            raise R1ProB1ComponentBuilderError(
                "step_once must be callable"
            )
        self._step_once = step_once

    def __call__(self, label: str) -> int:
        if not isinstance(label, str) or not label:
            raise R1ProB1ComponentBuilderError(
                "component step label must be non-empty"
            )
        result = self._step_once()
        if isinstance(result, bool) or not isinstance(result, int) or result != 1:
            raise R1ProB1ComponentBuilderError(
                "step_once must report exactly one simulator step"
            )
        return 1


class _MotionDependencies:
    def __init__(
        self,
        *,
        public_stability: Callable[[], Mapping[str, Any]],
        private_safety: Callable[[], Mapping[str, Any]],
        command_frequency_hz: float,
    ) -> None:
        _require_callable(public_stability, "public_stability")
        _require_callable(private_safety, "private_safety")
        if (
            isinstance(command_frequency_hz, bool)
            or not isinstance(command_frequency_hz, (int, float))
            or float(command_frequency_hz) <= 0.0
        ):
            raise ValueError("command_frequency_hz must be positive")
        self.public_stability = public_stability
        self.private_safety = private_safety
        self.command_frequency_hz = float(command_frequency_hz)

    def base_kwargs(
        self,
        *,
        control: Any,
        public_snapshot_provider: Callable[[], Mapping[str, Any]],
        step_once: Callable[[], int],
    ) -> dict[str, Any]:
        return {
            "control": control,
            "evaluator_step": _LabeledStepOnce(step_once),
            "public_proprio": _public_proprio(public_snapshot_provider),
            "public_stability": self.public_stability,
            "private_safety": self.private_safety,
            "command_frequency_hz": self.command_frequency_hz,
        }

    def proxy_kwargs(
        self,
        *,
        control: Any,
        public_snapshot_provider: Callable[[], Mapping[str, Any]],
        step_once: Callable[[], int],
        public_grasp_proxy_source: PublicCarryProxySource,
    ) -> dict[str, Any]:
        return {
            **self.base_kwargs(
                control=control,
                public_snapshot_provider=public_snapshot_provider,
                step_once=step_once,
            ),
            "public_grasp_proxy": (
                public_grasp_proxy_source.grasp_proxy_likely
            ),
        }


class R1ProBaseComponentBuilder:
    """Empty-hand base adapter backed by the minimal base-only binding."""

    api_contract_version = R1PRO_LOCAL_BASE_BINDING_VERSION

    def __init__(
        self,
        *,
        public_stability: Callable[[], Mapping[str, Any]],
        private_safety: Callable[[], Mapping[str, Any]],
        command_frequency_hz: float,
        local_base_binding_builder: Callable[..., Any] = (
            build_r1pro_local_base_binding
        ),
    ) -> None:
        self._motion = _MotionDependencies(
            public_stability=public_stability,
            private_safety=private_safety,
            command_frequency_hz=command_frequency_hz,
        )
        _require_callable(
            local_base_binding_builder,
            "local_base_binding_builder",
        )
        self._local_base_binding_builder = local_base_binding_builder

    def build(self, **kwargs: Any) -> Mapping[str, Any]:
        action, active_hand = _common_action_context(
            kwargs,
            expected_actions=_BASE_ACTIONS | _WAIT_ACTIONS,
        )
        del action
        if active_hand is not None:
            raise R1ProB1ComponentBuilderError(
                "empty-hand base builder received an active hand"
            )
        binding = _build_local_base_binding(
            self._local_base_binding_builder,
            motion=self._motion,
            kwargs=kwargs,
        )
        local_base = getattr(binding, "local_base", None)
        if local_base is None:
            raise R1ProB1ComponentBuilderError(
                "actuator binding did not provide local-base actuator"
            )
        return MappingProxyType(
            {LOCAL_BASE_ACTUATOR_SERVICE: local_base}
        )


class R1ProCarryComponentBuilder:
    """Carry base/look adapter backed by the combined actuator binding."""

    api_contract_version = R1PRO_B1_COMPONENT_VERSIONS["carry"]

    def __init__(
        self,
        *,
        public_stability: Callable[[], Mapping[str, Any]],
        private_safety: Callable[[], Mapping[str, Any]],
        command_frequency_hz: float,
        compensator: Any,
        private_audit: Any = None,
        actuator_binding_builder: Callable[..., Any] = (
            build_r1pro_actuator_bindings
        ),
    ) -> None:
        self._motion = _MotionDependencies(
            public_stability=public_stability,
            private_safety=private_safety,
            command_frequency_hz=command_frequency_hz,
        )
        if not callable(
            getattr(compensator, "target_for_torso_step", None)
        ):
            raise R1ProB1ComponentBuilderError(
                "compensator must implement target_for_torso_step"
            )
        if private_audit is not None and not callable(
            getattr(private_audit, "capture_endpoint", None)
        ):
            raise R1ProB1ComponentBuilderError(
                "private_audit must implement capture_endpoint"
            )
        _require_callable(
            actuator_binding_builder,
            "actuator_binding_builder",
        )
        self._compensator = compensator
        self._private_audit = private_audit
        self._actuator_binding_builder = actuator_binding_builder

    def build(self, **kwargs: Any) -> Mapping[str, Any]:
        _, active_hand = _common_action_context(
            kwargs,
            expected_actions=(
                _BASE_ACTIONS | _LOOK_ACTIONS | _VERTICAL_ACTIONS | _WAIT_ACTIONS
            ),
        )
        if active_hand not in {"left", "right"}:
            raise R1ProB1ComponentBuilderError(
                "carry builder requires public active_hand"
            )
        bindings = _build_actuator_bindings(
            self._actuator_binding_builder,
            motion=self._motion,
            kwargs=kwargs,
            active_hand=active_hand,
            component="carry actuator",
        )
        carry = getattr(bindings, "carry", None)
        if carry is None:
            raise R1ProB1ComponentBuilderError(
                "actuator binding did not provide carry actuator"
            )
        services = {
            CARRY_ACTUATOR_SERVICE: carry,
            CARRY_COMPENSATOR_SERVICE: self._compensator,
        }
        if self._private_audit is not None:
            services[CARRY_PRIVATE_AUDIT_SERVICE] = self._private_audit
        return MappingProxyType(services)


class R1ProTorsoComponentBuilder:
    """Empty-hand torso adapter backed by the torso/release binding."""

    api_contract_version = R1PRO_B1_COMPONENT_VERSIONS["torso"]

    def __init__(
        self,
        *,
        public_stability: Callable[[], Mapping[str, Any]],
        private_safety: Callable[[], Mapping[str, Any]],
        command_frequency_hz: float,
        max_base_z_drift_m: float = 0.03,
        max_abs_roll_pitch_deg: float = 5.0,
        torso_release_binding_builder: Callable[..., Any] = (
            build_r1pro_torso_release_bindings
        ),
    ) -> None:
        self._motion = _MotionDependencies(
            public_stability=public_stability,
            private_safety=private_safety,
            command_frequency_hz=command_frequency_hz,
        )
        self._max_base_z_drift_m = _positive(
            max_base_z_drift_m,
            "max_base_z_drift_m",
        )
        self._max_abs_roll_pitch_deg = _positive(
            max_abs_roll_pitch_deg,
            "max_abs_roll_pitch_deg",
        )
        _require_callable(
            torso_release_binding_builder,
            "torso_release_binding_builder",
        )
        self._binding_builder = torso_release_binding_builder

    def build(self, **kwargs: Any) -> Mapping[str, Any]:
        _, active_hand = _common_action_context(
            kwargs,
            expected_actions=_LOOK_ACTIONS | _VERTICAL_ACTIONS,
        )
        if active_hand is not None:
            raise R1ProB1ComponentBuilderError(
                "empty torso builder received an active hand"
            )
        bindings = _build_torso_release_bindings(
            self._binding_builder,
            motion=self._motion,
            kwargs=kwargs,
            max_base_z_drift_m=self._max_base_z_drift_m,
            max_abs_roll_pitch_deg=self._max_abs_roll_pitch_deg,
        )
        torso = getattr(bindings, "torso", None)
        if torso is None:
            raise R1ProB1ComponentBuilderError(
                "torso/release binding did not provide torso actuator"
            )
        return MappingProxyType(
            {PHYSICAL_TORSO_ACTUATOR_SERVICE: torso}
        )


class R1ProReleaseComponentBuilder:
    """Physical release adapter backed by the torso/release binding."""

    api_contract_version = R1PRO_B1_COMPONENT_VERSIONS["release"]

    def __init__(
        self,
        *,
        public_stability: Callable[[], Mapping[str, Any]],
        private_safety: Callable[[], Mapping[str, Any]],
        command_frequency_hz: float,
        max_base_z_drift_m: float = 0.03,
        max_abs_roll_pitch_deg: float = 5.0,
        torso_release_binding_builder: Callable[..., Any] = (
            build_r1pro_torso_release_bindings
        ),
    ) -> None:
        self._motion = _MotionDependencies(
            public_stability=public_stability,
            private_safety=private_safety,
            command_frequency_hz=command_frequency_hz,
        )
        self._max_base_z_drift_m = _positive(
            max_base_z_drift_m,
            "max_base_z_drift_m",
        )
        self._max_abs_roll_pitch_deg = _positive(
            max_abs_roll_pitch_deg,
            "max_abs_roll_pitch_deg",
        )
        _require_callable(
            torso_release_binding_builder,
            "torso_release_binding_builder",
        )
        self._binding_builder = torso_release_binding_builder

    def build(self, **kwargs: Any) -> Mapping[str, Any]:
        action, active_hand = _common_action_context(
            kwargs,
            expected_actions=frozenset({"release"}),
        )
        if (
            not isinstance(action.args, ReleaseArgs)
            or active_hand != action.args.hand
        ):
            raise R1ProB1ComponentBuilderError(
                "release requires matching public active_hand"
            )
        bindings = _build_torso_release_bindings(
            self._binding_builder,
            motion=self._motion,
            kwargs=kwargs,
            max_base_z_drift_m=self._max_base_z_drift_m,
            max_abs_roll_pitch_deg=self._max_abs_roll_pitch_deg,
        )
        release = getattr(bindings, "release", None)
        if release is None:
            raise R1ProB1ComponentBuilderError(
                "torso/release binding did not provide release actuator"
            )
        return MappingProxyType(
            {PHYSICAL_RELEASE_ACTUATOR_SERVICE: release}
        )


class R1ProGraspComponentBuilder:
    """Physical grasp adapter with explicit public/private providers."""

    api_contract_version = R1PRO_GRASP_DRIVER_VERSION

    def __init__(
        self,
        *,
        starter_by_hand: Mapping[str, Any],
        allowed_hands: Sequence[str] = ("left", "right"),
        camera_intrinsics: Mapping[str, Any],
        private_robot_pose_provider: Callable[[Any], RobotPose],
        public_evidence_provider: Callable[
            [str], PublicGripperWristEvidence
        ],
        private_contact_provider: Callable[[str], bool],
        private_assisted_provider: Callable[[str], bool],
        private_held_provider: Callable[[str], Any],
        private_lift_provider: Callable[[str], float],
        private_audit_writer: Callable[[Mapping[str, Any]], None],
        safe_carry_pose_provider: Callable[..., Any],
        binding_config_overrides: Mapping[str, Any] | None = None,
        pose_config: StarterGraspPoseConfig | None = None,
        grasp_bundle_builder: Callable[..., Any] = (
            build_resident_r1pro_grasp_service_bundle
        ),
    ) -> None:
        self._allowed_hands = _hand_set(allowed_hands)
        self._starter_by_hand = _starter_map(
            starter_by_hand,
            allowed_hands=self._allowed_hands,
        )
        self._camera_intrinsics = _mapping(
            camera_intrinsics,
            "camera_intrinsics",
        )
        for callback, name in (
            (private_robot_pose_provider, "private_robot_pose_provider"),
            (public_evidence_provider, "public_evidence_provider"),
            (private_contact_provider, "private_contact_provider"),
            (private_assisted_provider, "private_assisted_provider"),
            (private_held_provider, "private_held_provider"),
            (private_lift_provider, "private_lift_provider"),
            (private_audit_writer, "private_audit_writer"),
            (safe_carry_pose_provider, "safe_carry_pose_provider"),
            (grasp_bundle_builder, "grasp_bundle_builder"),
        ):
            _require_callable(callback, name)
        self._private_robot_pose_provider = private_robot_pose_provider
        self._public_evidence_provider = public_evidence_provider
        self._private_contact_provider = private_contact_provider
        self._private_assisted_provider = private_assisted_provider
        self._private_held_provider = private_held_provider
        self._private_lift_provider = private_lift_provider
        self._private_audit_writer = private_audit_writer
        self._safe_carry_pose_provider = safe_carry_pose_provider
        self._binding_config_overrides = _optional_mapping(
            binding_config_overrides,
            "binding_config_overrides",
        )
        self._pose_config = pose_config
        self._bundle_builder = grasp_bundle_builder

    def build(self, **kwargs: Any) -> Mapping[str, Any]:
        action, active_hand = _common_action_context(
            kwargs,
            expected_actions=frozenset({"grasp_at"}),
        )
        if not isinstance(action.args, PointActionArgs):
            raise R1ProB1ComponentBuilderError(
                "grasp_at requires parsed PointActionArgs"
            )
        if action.args.hand not in self._allowed_hands:
            raise R1ProB1ComponentBuilderError(
                "grasp hand is not admitted by live calibration"
            )
        if active_hand is not None:
            raise R1ProB1ComponentBuilderError(
                "grasp builder refuses an existing public carry state"
            )
        starter = self._starter_by_hand[action.args.hand]
        frozen_snapshot = _snapshot(kwargs["public_snapshot_provider"])
        pose = self._private_robot_pose_provider(kwargs["evaluator"])
        if not isinstance(pose, RobotPose):
            raise R1ProB1ComponentBuilderError(
                "private_robot_pose_provider must return RobotPose"
            )
        try:
            bundle = self._bundle_builder(
                starter=starter,
                shared_control=kwargs["control"],
                one_step=_LabeledStepOnce(kwargs["step_once"]),
                frozen_public_snapshot=frozen_snapshot,
                camera_intrinsics=self._camera_intrinsics,
                private_robot_pose=pose,
                public_evidence_provider=self._public_evidence_provider,
                private_contact_provider=self._private_contact_provider,
                private_assisted_provider=self._private_assisted_provider,
                private_held_provider=self._private_held_provider,
                private_lift_provider=self._private_lift_provider,
                private_audit_writer=self._private_audit_writer,
                safe_carry_pose_provider=self._safe_carry_pose_provider,
                binding_config_overrides=self._binding_config_overrides,
                pose_config=self._pose_config,
            )
        except Exception as exc:
            raise _component_error("grasp", exc) from exc
        return _bundle_services(
            bundle,
            component="grasp",
            allowed_keys={
                PHYSICAL_GRASP_BACKEND_SERVICE,
                PRIVATE_GRASP_AUDIT_SERVICE,
            },
        )


class R1ProVisibleOfficialGraspComponentBuilder:
    """Opt-in evaluator-private visibility gate plus official GRASP."""

    api_contract_version = VISIBLE_OFFICIAL_GRASP_BACKEND_VERSION

    def __init__(
        self,
        *,
        starter_by_hand: Mapping[str, Any],
        private_held_provider: Callable[[str], Any],
        private_audit_writer: Callable[[Mapping[str, Any]], None],
        allowed_hands: Sequence[str] = ("left", "right"),
        grasp_config: Any = None,
        visibility_config: Any = None,
        service_builder: Callable[..., Any] = (
            build_r1pro_visible_official_grasp_services
        ),
    ) -> None:
        self._allowed_hands = _hand_set(allowed_hands)
        self._starter_by_hand = _starter_map(
            starter_by_hand,
            allowed_hands=self._allowed_hands,
        )
        for callback, name in (
            (private_held_provider, "private_held_provider"),
            (private_audit_writer, "private_audit_writer"),
            (service_builder, "service_builder"),
        ):
            _require_callable(callback, name)
        self._private_held_provider = private_held_provider
        self._private_audit_writer = private_audit_writer
        self._grasp_config = grasp_config
        self._visibility_config = visibility_config
        self._service_builder = service_builder

    def build(self, **kwargs: Any) -> Mapping[str, Any]:
        action, active_hand = _common_action_context(
            kwargs,
            expected_actions=frozenset({"grasp_at", "grasp"}),
        )
        if action.action == "grasp_at":
            if (
                not isinstance(action.args, PointActionArgs)
                or action.args.camera != "head"
                or action.args.hand not in self._allowed_hands
            ):
                raise R1ProB1ComponentBuilderError(
                    "visible official grasp requires an admitted head point"
                )
            hand = action.args.hand
        else:
            if not isinstance(action.args, VisibleObjectGraspArgs):
                raise R1ProB1ComponentBuilderError(
                    "visible-object grasp requires parsed category arguments"
                )
            if len(self._allowed_hands) != 1:
                raise R1ProB1ComponentBuilderError(
                    "visible-object grasp requires one calibrated hand"
                )
            hand = next(iter(self._allowed_hands))
        if active_hand is not None:
            raise R1ProB1ComponentBuilderError(
                "visible official grasp refuses an existing carry state"
            )
        evaluator = kwargs["evaluator"]
        control = kwargs["control"]
        snapshot = _snapshot(kwargs["public_snapshot_provider"])
        try:
            head_sensor = evaluator.robot.sensors[
                CAMERA_SENSOR_NAMES["head"]
            ]
            instance_registry = getattr(
                head_sensor,
                "INSTANCE_REGISTRY",
                None,
            )
            if not isinstance(instance_registry, Mapping):
                raise R1ProB1ComponentBuilderError(
                    "head sensor has no instance registry"
                )
            frozen_frame = capture_private_head_frame(
                raw=getattr(evaluator, "obs", None),
                public_head_rgb=snapshot["cameras"]["head"]["rgb"],
                simulator_step=_private_observation_count(control),
                instance_registry={
                    int(key): str(value)
                    for key, value in instance_registry.items()
                },
            )
            unique_ids = {
                int(value) for value in np.unique(frozen_frame.segmentation)
            }
            known_ids = unique_ids.intersection(
                frozen_frame.instance_id_to_name
            )
            self._private_audit_writer(
                {
                    "schema_version": (
                        "behavior2026.private-segmentation-coverage/v1"
                    ),
                    "event": "private_segmentation_registry_coverage",
                    "details": {
                        "registry_size": len(
                            frozen_frame.instance_id_to_name
                        ),
                        "segmentation_unique_id_count": len(unique_ids),
                        "known_unique_id_count": len(known_ids),
                        "coverage_ratio": (
                            len(known_ids) / len(unique_ids)
                            if unique_ids
                            else 1.0
                        ),
                    },
                }
            )
            eligible = private_picking_up_toys_eligible_objects(evaluator)
            eligible_by_category = (
                private_picking_up_toys_eligible_objects_by_category(evaluator)
            )
            starter = self._starter_by_hand[hand]
            labelled_step = _LabeledStepOnce(kwargs["step_once"])

            def command_step(command: Any, label: str) -> int:
                control.command_direct_action(command)
                return labelled_step(label)

            bundle = self._service_builder(
                starter_by_hand={hand: starter},
                private_frame_provider=lambda: frozen_frame,
                eligible_objects_provider=lambda: eligible,
                eligible_objects_by_category_provider=(
                    lambda: eligible_by_category
                ),
                command_step=command_step,
                base_pose_provider=lambda: planar_base_pose_from_robot(
                    evaluator.robot
                ),
                held_object_provider_by_hand={
                    hand: lambda: self._private_held_provider(hand)
                },
                hold_safe_by_hand={hand: control.hold},
                private_audit_writer=self._private_audit_writer,
                grasp_config=self._grasp_config,
                visibility_config=self._visibility_config,
            )
        except Exception as exc:
            raise _component_error("visible_official_grasp", exc) from exc
        return _bundle_services(
            bundle,
            component="visible_official_grasp",
            allowed_keys={
                VISIBLE_OFFICIAL_GRASP_BACKEND_SERVICE,
                PRIVATE_GRASP_AUDIT_SERVICE,
            },
        )


class R1ProVisibleObjectOfficialGraspComponentBuilder(
    R1ProVisibleOfficialGraspComponentBuilder
):
    """Object-only variant with one explicit executor-selected hand."""

    def __init__(
        self,
        *,
        starter_by_hand: Mapping[str, Any],
        private_held_provider: Callable[[str], Any],
        private_audit_writer: Callable[[Mapping[str, Any]], None],
        allowed_hands: Sequence[str],
        grasp_config: Any = None,
        visibility_config: Any = None,
        service_builder: Callable[..., Any] = (
            build_r1pro_visible_official_grasp_services
        ),
    ) -> None:
        hands = tuple(allowed_hands)
        if len(hands) != 1 or hands[0] not in {"left", "right"}:
            raise ValueError(
                "visible-object official grasp requires exactly one hand"
            )
        self.implicit_object_grasp_hand = hands[0]
        super().__init__(
            starter_by_hand=starter_by_hand,
            private_held_provider=private_held_provider,
            private_audit_writer=private_audit_writer,
            allowed_hands=hands,
            grasp_config=grasp_config,
            visibility_config=visibility_config,
            service_builder=service_builder,
        )

    def build(self, **kwargs: Any) -> Mapping[str, Any]:
        action = kwargs.get("action")
        if not isinstance(action, ActionEnvelope) or action.action != "grasp":
            raise R1ProB1ComponentBuilderError(
                "visible-object component accepts grasp only"
            )
        return super().build(**kwargs)


class R1ProMarkerSymbolicComponentBuilder:
    """Shared explicit-mark / Symbolic GRASP / PLACE_INSIDE service."""

    api_contract_version = MARKER_SYMBOLIC_BACKEND_VERSION

    def __init__(
        self,
        *,
        symbolic_primitives: Any,
        grasp_primitive: Any,
        place_inside_primitive: Any,
        open_primitive: Any | None = None,
        close_primitive: Any | None = None,
        toggle_on_primitive: Any | None = None,
        toggle_off_primitive: Any | None = None,
        release_primitive: Any | None = None,
        spatial_generator_factory: Callable[..., Any] | None = None,
        marker_store: PrivateVisualMarkerStore,
        implicit_hand: str,
        maximum_local_reach_m: float = ACTION20_LOCAL_WORKSPACE_M,
        maximum_local_approach_m: float = 0.40,
        local_approach_enabled: bool = True,
        local_approach_stop_m: float = 0.86,
        local_approach_maximum_steps: int = 32,
        local_approach_hard_stop_m: float = 0.45,
        on_top_station_prepare_factory: Callable[..., Any] | None = None,
        interaction_approach_prepare_factory: Callable[..., Any] | None = None,
        automatic_interaction_approach_enabled: bool = False,
        placement_physical_consequences_enabled: bool = False,
        symbolic_zero_physics: bool = False,
        render_without_physics: Callable[[], Any] | None = None,
        public_stability: Callable[[], Mapping[str, Any]] | None = None,
        private_safety: Callable[[], Mapping[str, Any]] | None = None,
        marker_objects_provider: Callable[[Any], Mapping[str, Any]] | None = None,
        marker_roles_provider: Callable[
            [Any], tuple[Sequence[str], Sequence[str]]
        ] | None = None,
        operation_marker_objects_provider: Callable[
            [Any, str], Mapping[str, Any]
        ] | None = None,
        under_task_fallback_names_provider: Callable[[], Iterable[str]] | None = None,
        task_target_fallback_names_provider: Callable[[str], Iterable[str]] | None = None,
        on_top_snapshot_provider: Callable[..., Any] | None = None,
        next_to_snapshot_provider: Callable[..., Any] | None = None,
        private_head_segmentation_side_channel: (
            PrivateHeadSegmentationSideChannel | None
        ) = None,
        private_head_segmentation_side_channel_provider: (
            Callable[[Any], PrivateHeadSegmentationSideChannel] | None
        ) = None,
    ) -> None:
        if not callable(getattr(symbolic_primitives, "apply_ref", None)):
            raise TypeError("symbolic_primitives must provide apply_ref")
        if grasp_primitive is None or place_inside_primitive is None:
            raise ValueError("symbolic primitive enum values must be supplied")
        if not isinstance(marker_store, PrivateVisualMarkerStore):
            raise TypeError("marker_store must be PrivateVisualMarkerStore")
        if implicit_hand not in {"left", "right"}:
            raise ValueError("implicit_hand must be left or right")
        if marker_objects_provider is None:
            marker_objects_provider = private_picking_up_toys_marker_objects
        if marker_roles_provider is None:
            marker_roles_provider = private_picking_up_toys_marker_roles
        operation_marker_provider_custom = (
            operation_marker_objects_provider is not None
        )
        if operation_marker_objects_provider is None:
            operation_marker_objects_provider = (
                lambda evaluator, _operation: marker_objects_provider(evaluator)
            )
        if not callable(marker_objects_provider):
            raise TypeError("marker_objects_provider must be callable")
        if not callable(marker_roles_provider):
            raise TypeError("marker_roles_provider must be callable")
        if not callable(operation_marker_objects_provider):
            raise TypeError("operation_marker_objects_provider must be callable")
        if under_task_fallback_names_provider is not None and not callable(
            under_task_fallback_names_provider
        ):
            raise TypeError("under_task_fallback_names_provider must be callable")
        if task_target_fallback_names_provider is not None and not callable(
            task_target_fallback_names_provider
        ):
            raise TypeError("task_target_fallback_names_provider must be callable")
        if on_top_snapshot_provider is not None and not callable(
            on_top_snapshot_provider
        ):
            raise TypeError("on_top_snapshot_provider must be callable")
        if next_to_snapshot_provider is not None and not callable(
            next_to_snapshot_provider
        ):
            raise TypeError("next_to_snapshot_provider must be callable")
        if (
            private_head_segmentation_side_channel is not None
            and not isinstance(
                private_head_segmentation_side_channel,
                PrivateHeadSegmentationSideChannel,
            )
        ):
            raise TypeError(
                "private_head_segmentation_side_channel has an invalid type"
            )
        if (
            private_head_segmentation_side_channel_provider is not None
            and not callable(private_head_segmentation_side_channel_provider)
        ):
            raise TypeError(
                "private_head_segmentation_side_channel_provider must be callable"
            )
        if (
            private_head_segmentation_side_channel is not None
            and private_head_segmentation_side_channel_provider is not None
        ):
            raise ValueError(
                "provide a fixed segmentation channel or a late-bound provider"
            )
        if (
            isinstance(maximum_local_reach_m, bool)
            or not isinstance(maximum_local_reach_m, (int, float))
            or not math.isfinite(float(maximum_local_reach_m))
            or float(maximum_local_reach_m) <= 0.0
        ):
            raise ValueError("maximum_local_reach_m must be finite and positive")
        if not isinstance(local_approach_enabled, bool):
            raise TypeError("local_approach_enabled must be boolean")
        if not isinstance(automatic_interaction_approach_enabled, bool):
            raise TypeError(
                "automatic_interaction_approach_enabled must be boolean"
            )
        if on_top_station_prepare_factory is not None and not callable(
            on_top_station_prepare_factory
        ):
            raise TypeError("on_top_station_prepare_factory must be callable")
        if interaction_approach_prepare_factory is not None and not callable(
            interaction_approach_prepare_factory
        ):
            raise TypeError(
                "interaction_approach_prepare_factory must be callable"
            )
        if (
            automatic_interaction_approach_enabled
            and interaction_approach_prepare_factory is None
        ):
            raise ValueError(
                "automatic interaction approach requires a prepare factory"
            )
        if not isinstance(placement_physical_consequences_enabled, bool):
            raise TypeError(
                "placement_physical_consequences_enabled must be boolean"
            )
        if not isinstance(symbolic_zero_physics, bool):
            raise TypeError("symbolic_zero_physics must be boolean")
        if symbolic_zero_physics and not callable(render_without_physics):
            raise TypeError(
                "zero-physics symbolic actions require render_without_physics"
            )
        self._primitives = symbolic_primitives
        self._grasp_primitive = grasp_primitive
        self._place_inside_primitive = place_inside_primitive
        optional = (open_primitive, close_primitive, release_primitive)
        if any(value is not None for value in optional) and any(
            value is None for value in optional
        ):
            raise ValueError(
                "open, close, and release primitives must be supplied together"
            )
        self._open_primitive = open_primitive
        self._close_primitive = close_primitive
        self._release_primitive = release_primitive
        self.v34_symbolic_actions = all(value is not None for value in optional)
        toggle_optional = (toggle_on_primitive, toggle_off_primitive)
        if any(value is not None for value in toggle_optional) and any(
            value is None for value in toggle_optional
        ):
            raise ValueError("toggle-on and toggle-off primitives must be supplied together")
        self._toggle_on_primitive = toggle_on_primitive
        self._toggle_off_primitive = toggle_off_primitive
        self.toggle_symbolic_actions = all(
            value is not None for value in toggle_optional
        )
        if spatial_generator_factory is not None and not callable(
            spatial_generator_factory
        ):
            raise TypeError("spatial_generator_factory must be callable")
        self._spatial_generator_factory = spatial_generator_factory
        self.action23_symbolic_actions = bool(
            self.v34_symbolic_actions
            and self.toggle_symbolic_actions
            and spatial_generator_factory is not None
        )
        self._marker_store = marker_store
        self.implicit_object_grasp_hand = implicit_hand
        self._maximum_local_reach_m = float(maximum_local_reach_m)
        for value, name in (
            (maximum_local_approach_m, "maximum_local_approach_m"),
            (local_approach_stop_m, "local_approach_stop_m"),
            (local_approach_hard_stop_m, "local_approach_hard_stop_m"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if local_approach_stop_m >= maximum_local_reach_m:
            raise ValueError(
                "local_approach_stop_m must leave margin inside local reach"
            )
        if (
            isinstance(local_approach_maximum_steps, bool)
            or not isinstance(local_approach_maximum_steps, int)
            or local_approach_maximum_steps <= 0
        ):
            raise ValueError("local_approach_maximum_steps must be positive")
        if (public_stability is None) != (private_safety is None):
            raise ValueError(
                "public_stability and private_safety must be supplied together"
            )
        if public_stability is not None and not callable(public_stability):
            raise TypeError("public_stability must be callable")
        if private_safety is not None and not callable(private_safety):
            raise TypeError("private_safety must be callable")
        self._maximum_local_approach_m = float(maximum_local_approach_m)
        self._local_approach_enabled = local_approach_enabled
        self._local_approach_stop_m = float(local_approach_stop_m)
        self._local_approach_maximum_steps = local_approach_maximum_steps
        self._local_approach_hard_stop_m = float(local_approach_hard_stop_m)
        self._on_top_station_prepare_factory = on_top_station_prepare_factory
        self._interaction_approach_prepare_factory = (
            interaction_approach_prepare_factory
        )
        self._automatic_interaction_approach_enabled = (
            automatic_interaction_approach_enabled
        )
        self._placement_physical_consequences_enabled = (
            placement_physical_consequences_enabled
        )
        self._symbolic_zero_physics = symbolic_zero_physics
        self._render_without_physics = render_without_physics
        self._public_stability = public_stability
        self._private_safety = private_safety
        self._marker_objects_provider = marker_objects_provider
        self._marker_roles_provider = marker_roles_provider
        self._operation_marker_objects_provider = operation_marker_objects_provider
        self._under_task_fallback_names_provider = (
            under_task_fallback_names_provider or (lambda: ())
        )
        self._task_target_fallback_names_provider = (
            task_target_fallback_names_provider or (lambda _operation: ())
        )
        self._on_top_snapshot_provider = on_top_snapshot_provider
        self._next_to_snapshot_provider = next_to_snapshot_provider
        self._operation_marker_provider_custom = operation_marker_provider_custom
        self._private_head_segmentation_side_channel = (
            private_head_segmentation_side_channel
        )
        self._private_head_segmentation_side_channel_provider = (
            private_head_segmentation_side_channel_provider
        )
        self._pre_pick: dict[str, Any] = {}

    def build(self, **kwargs: Any) -> Mapping[str, Any]:
        expected_actions = (
            frozenset(
                {
                    "mark_visible_objects",
                    "pick",
                    "place_inside",
                    "put_back",
                    "open",
                    "close",
                    "release",
                    "place_on_floor",
                    *(
                        ("toggle_on", "toggle_off")
                        if self.toggle_symbolic_actions
                        else ()
                    ),
                    *(
                        (
                            "place_on_top",
                            "place_next_to",
                            "place_under",
                            "place_touching",
                        )
                        if self.action23_symbolic_actions
                        else ()
                    ),
                }
            )
            if self.v34_symbolic_actions
            else frozenset(
                {"mark_visible_objects", "grasp", "place_inside"}
            )
        )
        action, active_hand = _common_action_context(
            kwargs,
            expected_actions=expected_actions,
        )
        if action.action in {"grasp", "pick"}:
            expected_type = (
                (MarkerTargetArgs, MarkerTargetOrRequestArgs)
                if action.action == "pick"
                else MarkerGraspArgs
            )
            if not isinstance(action.args, expected_type) or active_hand is not None:
                raise R1ProB1ComponentBuilderError(
                    "marker grasp requires an empty public carry state"
                )
        elif action.action in {
            "place_inside",
            "place_on_top",
            "place_next_to",
            "place_under",
            "place_touching",
        }:
            expected_type = (
                (MarkerTargetArgs, MarkerTargetOrRequestArgs)
                if self.v34_symbolic_actions
                else MarkerPlaceInsideArgs
            )
            if (
                not isinstance(action.args, expected_type)
                or active_hand != self.implicit_object_grasp_hand
            ):
                raise R1ProB1ComponentBuilderError(
                    "marker place requires the implicit-hand carry state"
                )
        elif action.action in {"open", "close", "toggle_on", "toggle_off"}:
            if (
                not isinstance(
                    action.args,
                    (MarkerTargetArgs, MarkerTargetOrRequestArgs),
                )
                or active_hand is not None
            ):
                raise R1ProB1ComponentBuilderError(
                    "marker open/close/toggle requires an empty public carry state"
                )
        elif action.action in {"put_back", "release", "place_on_floor"}:
            if (
                not isinstance(action.args, NoArgs)
                or active_hand != self.implicit_object_grasp_hand
            ):
                raise R1ProB1ComponentBuilderError(
                    f"{action.action} requires the implicit-hand carry state"
                )
        evaluator = kwargs["evaluator"]
        control = kwargs["control"]
        visual_binding_required = action.action not in _MARKERLESS_SYMBOLIC_ACTIONS
        public_snapshot = _snapshot(kwargs["public_snapshot_provider"])
        simulator_step = public_snapshot.get("simulator_step")
        if (
            isinstance(simulator_step, bool)
            or not isinstance(simulator_step, int)
            or simulator_step < 0
        ):
            raise R1ProB1ComponentBuilderError(
                "marker symbolic component requires public simulator_step"
            )

        def raw_head_rgb() -> np.ndarray:
            raw = getattr(evaluator, "obs", None)
            if not isinstance(raw, Mapping) or HEAD_RGB_KEY not in raw:
                raise R1ProB1ComponentBuilderError(
                    "raw official head RGB is unavailable"
                )
            value = raw[HEAD_RGB_KEY]
            for method in ("detach", "cpu"):
                callback = getattr(value, method, None)
                if callable(callback):
                    value = callback()
            callback = getattr(value, "numpy", None)
            if callable(callback):
                value = callback()
            array = np.ascontiguousarray(np.asarray(value)[..., :3], dtype=np.uint8)
            return array

        try:
            def live_private_frame() -> Any:
                current = _snapshot(kwargs["public_snapshot_provider"])
                current_step = current.get("simulator_step")
                if (
                    isinstance(current_step, bool)
                    or not isinstance(current_step, int)
                    or current_step < 0
                ):
                    raise R1ProB1ComponentBuilderError(
                        "live marker frame lacks simulator_step"
                    )
                side_channel = self._private_head_segmentation_side_channel
                if side_channel is None:
                    provider = self._private_head_segmentation_side_channel_provider
                    if provider is not None:
                        side_channel = provider(evaluator)
                        if not isinstance(
                            side_channel,
                            PrivateHeadSegmentationSideChannel,
                        ):
                            raise R1ProB1ComponentBuilderError(
                                "private segmentation provider returned an invalid channel"
                            )
                        self._private_head_segmentation_side_channel = side_channel
                if side_channel is not None:
                    if side_channel.evaluator is not evaluator or not side_channel.started:
                        raise R1ProB1ComponentBuilderError(
                            "private segmentation side channel is not ready"
                        )
                    return side_channel.capture(
                        simulator_step=current_step,
                        public_head_rgb=raw_head_rgb(),
                    )
                head_sensor = evaluator.robot.sensors[CAMERA_SENSOR_NAMES["head"]]
                registry = getattr(head_sensor, "INSTANCE_REGISTRY", None)
                if not isinstance(registry, Mapping):
                    raise R1ProB1ComponentBuilderError(
                        "head sensor has no instance registry"
                    )
                return capture_private_head_frame(
                    raw=evaluator.obs,
                    public_head_rgb=raw_head_rgb(),
                    simulator_step=current_step,
                    instance_registry={
                        int(key): str(value) for key, value in registry.items()
                    },
                )

            # A transient private read cannot produce a valid C-label, but it
            # also has not changed physics.  Use one same-RGB empty frame so the
            # existing Marker path returns ``markers_unavailable`` instead of
            # escalating an unnecessary scene recovery.
            marker_frame_unavailable = False
            try:
                initial_private_frame = (
                    live_private_frame() if visual_binding_required else None
                )
            except PrivateHeadSegmentationUnavailableError:
                rgb = raw_head_rgb()
                initial_private_frame = capture_private_head_frame(
                    raw={
                        HEAD_RGB_KEY: rgb,
                        HEAD_INSTANCE_SEGMENTATION_KEY: np.zeros(
                            rgb.shape[:2], dtype=np.int64
                        ),
                    },
                    public_head_rgb=rgb,
                    simulator_step=simulator_step,
                    instance_registry={},
                )
                marker_frame_unavailable = True
            marker_objects = dict(self._marker_objects_provider(evaluator))
            graspable_names, container_names = self._marker_roles_provider(
                evaluator
            )
            pickable_names = frozenset(graspable_names)
            pick_bundle_targets: tuple[PickBundleTargetToken, ...] = ()
            if action.action == "pick":
                assert initial_private_frame is not None
                scene_objects = tuple(evaluator.env.scene.objects)
                visible_scene_objects = _current_visible_scene_objects(
                    scene_objects,
                    initial_private_frame,
                )
                movable_objects = _live_movable_objects(
                    visible_scene_objects,
                )
                visible_pickable_objects = _live_accessible_pickable_objects(
                    scene_objects,
                    _live_pickable_objects(visible_scene_objects),
                )
                pick_bundle_targets, blocked_pick_names = (
                    _live_pick_bundle_targets(
                        movable_objects,
                        visible_pickable_objects,
                    )
                )
                # Role admission must not freeze the earlier capture's
                # visibility. Marker publication takes a fresh frame and is
                # the single authority for current object / mesh pixels.
                pickable_objects = _live_accessible_pickable_objects(
                    scene_objects,
                    _live_pickable_objects(scene_objects),
                )
                for name in blocked_pick_names:
                    pickable_objects.pop(name, None)
                marker_objects.update(pickable_objects)
                pickable_names = frozenset(pickable_objects)
            operation_marker_objects: dict[str, Any] = {}
            if action.action in {
                "place_inside",
                "place_on_top",
                "place_next_to",
                "place_under",
                "place_touching",
            }:
                operation_marker_objects = dict(
                    self._operation_marker_objects_provider(evaluator, action.action)
                )
                for name, target in operation_marker_objects.items():
                    existing = marker_objects.get(name)
                    if existing is not None and existing is not target:
                        raise R1ProB1ComponentBuilderError(
                            "operation marker object name is ambiguous"
                        )
                    marker_objects[name] = target
            openable_names: frozenset[str] = frozenset()
            if self.v34_symbolic_actions:
                openable_objects = {
                    str(obj.name): obj
                    for obj in evaluator.env.scene.objects
                    if any(
                        getattr(state_type, "__name__", "") == "Open"
                        for state_type in getattr(obj, "states", {})
                    )
                }
                marker_objects.update(openable_objects)
                openable_names = frozenset(openable_objects)
            toggleable_names: frozenset[str] = frozenset()
            if self.toggle_symbolic_actions:
                toggleable_names = frozenset(
                    str(obj.name)
                    for obj in evaluator.env.scene.objects
                    if any(
                        getattr(state_type, "__name__", "") == "ToggledOn"
                        for state_type in getattr(obj, "states", {})
                    )
                )
            support_names: frozenset[str] = frozenset()
            if self.action23_symbolic_actions and action.action == "place_on_top":
                held = self._primitives._get_obj_in_hand()
                if held is None:
                    raise R1ProB1ComponentBuilderError(
                        "place_on_top requires one privately held object"
                    )
                support_objects = _generic_on_top_support_objects(
                    evaluator.env.scene.objects,
                    held,
                )
                marker_objects.update(support_objects)
                support_names = frozenset(support_objects)

            def operation_marker_objects_for(operation: str) -> dict[str, Any]:
                diagnostic = {
                    "operation": operation, "filters_private": [],
                    "candidate_source": "custom_provider" if self._operation_marker_provider_custom else "marker_objects",
                }
                operation_marker_objects_for.publication_diagnostic = diagnostic
                candidates = dict(
                    self._operation_marker_objects_provider(evaluator, operation)
                    if self._operation_marker_provider_custom
                    else marker_objects
                )
                fallback_names = frozenset(
                    self._task_target_fallback_names_provider(operation)
                )
                if fallback_names:
                    scene_by_name = {
                        str(getattr(obj, "name", "")): obj
                        for obj in evaluator.env.scene.objects
                        if str(getattr(obj, "name", ""))
                    }
                    for name in sorted(fallback_names):
                        target = scene_by_name.get(name)
                        if target is not None and name not in candidates:
                            candidates[name] = target
                            diagnostic.setdefault(
                                "task_fallback_targets_private", []
                            ).append(name)
                diagnostic["input_candidate_count"] = len(candidates)
                role_names = (
                    set(pickable_names)
                    if operation == "pick"
                    else set(container_names)
                    if operation == "place_inside"
                    else set(openable_names)
                    if operation in {"open", "close"}
                    else set(toggleable_names)
                    if operation in {"toggle_on", "toggle_off"}
                    else set(support_names)
                    if operation == "place_on_top"
                    else set(candidates)
                )
                diagnostic["filters_private"].extend(
                    {"target_private": name, "reason": "ROLE_NOT_ADMITTED"}
                    for name in candidates if name not in role_names
                )
                candidates = {
                    name: target
                    for name, target in candidates.items()
                    if name in role_names
                }
                if operation == "place_under":
                    before_under = set(candidates)
                    candidates = _under_marker_target_objects(
                        candidates,
                        held=self._primitives._get_obj_in_hand(),
                        scene_objects=tuple(evaluator.env.scene.objects),
                        robot=evaluator.robot,
                        logical_snapshot=getattr(
                            self._primitives,
                            "_logical_inventory_snapshot",
                            None,
                        ),
                        task_fallback_names=self._under_task_fallback_names_provider(),
                        symbolic_support=(
                            self._symbolic_zero_physics
                            and not self._placement_physical_consequences_enabled
                        ),
                    )
                    diagnostic["filters_private"].extend(
                        {"target_private": name, "reason": "UNDER_ADMISSION_REJECTED", "detail": "UNKNOWN"}
                        for name in sorted(before_under - set(candidates))
                    )
                desired_state = {
                    "place_inside": ("Open", True),
                    "open": ("Open", False),
                    "close": ("Open", True),
                    "toggle_on": ("ToggledOn", False),
                    "toggle_off": ("ToggledOn", True),
                }.get(operation)
                if desired_state is None:
                    diagnostic["output_candidate_count"] = len(candidates)
                    return candidates
                state_name, desired_value = desired_state
                filtered: dict[str, Any] = {}
                for name, target in candidates.items():
                    # Parent Open=True means "any part is open".  Let the
                    # per-part filter below keep closed siblings selectable.
                    if operation in {"open", "close"} and drawer_part_tokens(target):
                        filtered[name] = target
                        continue
                    state = next(
                        (
                            value
                            for state_type, value in getattr(target, "states", {}).items()
                            if getattr(state_type, "__name__", "") == state_name
                        ),
                        None,
                    )
                    if state is None:
                        if operation == "place_inside":
                            filtered[name] = target
                        else:
                            diagnostic["filters_private"].append({"target_private": name, "reason": "STATE_UNAVAILABLE", "state": state_name})
                        continue
                    try:
                        current_value = bool(state.get_value())
                    except Exception as exc:
                        diagnostic["filters_private"].append({"target_private": name, "reason": "STATE_QUERY_ERROR", "retained": True, "error_type": type(exc).__name__})
                        filtered[name] = target
                        continue
                    if current_value is desired_value:
                        filtered[name] = target
                    else:
                        diagnostic["filters_private"].append({"target_private": name, "reason": "STATE_PRECONDITION_MISMATCH", "state": state_name, "observed": current_value, "required": desired_value})
                diagnostic["output_candidate_count"] = len(filtered)
                return filtered
            labelled_step = _LabeledStepOnce(kwargs["step_once"])

            physics_step_once = kwargs.get("physics_step_once")
            if physics_step_once is not None and not callable(physics_step_once):
                raise R1ProB1ComponentBuilderError(
                    "physics_step_once must be callable"
                )
            mark_boundary_already_checkpointed = kwargs.get(
                "mark_boundary_already_checkpointed"
            )
            if physics_step_once is not None and not callable(
                mark_boundary_already_checkpointed
            ):
                raise R1ProB1ComponentBuilderError(
                    "articulated physics transaction boundary marker is unavailable"
                )

            def command_step(command: Any) -> int:
                if isinstance(command, ArticulatedPartTransactionBoundary):
                    control.hold()
                    boundary_steps = labelled_step(
                        "marker_symbolic_articulated_transaction_boundary"
                    )
                    mark_boundary_already_checkpointed()
                    return command.simulator_steps + boundary_steps
                if command is None:
                    # The official Symbolic primitive mutates the semantic
                    # state before yielding settle commands.  Its inherited
                    # Starter settle path calls robot.q_to_action(), which is
                    # incompatible with the evaluator's mixed R1Pro controller
                    # layout.  A None sentinel means: keep the public robot
                    # command at the safe hold target and advance exactly one
                    # real simulator step.
                    get_obj_in_hand = getattr(
                        self._primitives,
                        "_get_obj_in_hand",
                        None,
                    )
                    carrying = (
                        callable(get_obj_in_hand)
                        and get_obj_in_hand() is not None
                    )
                    if carrying:
                        # SharedCanaryControl.hold() intentionally opens both
                        # grippers.  That is safe for navigation, but the
                        # marker-bound carry contract requires a closed active
                        # hand. Keep only the actual symbolic hand closed while
                        # using fresh public proprioception for every joint hold
                        # target. The held-object query stays inside the private
                        # executor and is never returned to the model.
                        expected_obj = get_obj_in_hand()
                        _assert_marker_symbolic_grasp_held(
                            self._primitives,
                            hand=self.implicit_object_grasp_hand,
                            expected_obj=expected_obj,
                            phase=MarkerSymbolicCarryPhase.BEFORE_SETTLE_STEP,
                        )
                        current = _snapshot(
                            kwargs["public_snapshot_provider"]
                        )
                        hand = self.implicit_object_grasp_hand
                        action = _marker_symbolic_closed_hold_action(
                            current,
                            hand=hand,
                        )
                        control.command_carry_action(
                            action,
                            active_hand=hand,
                        )
                        steps = labelled_step(
                            "marker_symbolic_closed_hold_settle"
                        )
                        _assert_marker_symbolic_grasp_held(
                            self._primitives,
                            hand=hand,
                            expected_obj=expected_obj,
                            phase=MarkerSymbolicCarryPhase.AFTER_SETTLE_STEP,
                        )
                        return steps
                    control.hold()
                    return labelled_step("marker_symbolic_open_hold_settle")
                control.command_direct_action(command)
                return labelled_step("marker_symbolic")

            def symbolic_hold_safe() -> None:
                get_obj_in_hand = getattr(
                    self._primitives, "_get_obj_in_hand", None
                )
                held = get_obj_in_hand() if callable(get_obj_in_hand) else None
                if held is None:
                    control.hold()
                    return
                current = _snapshot(kwargs["public_snapshot_provider"])
                control.command_carry_action(
                    _marker_symbolic_closed_hold_action(
                        current,
                        hand=self.implicit_object_grasp_hand,
                    ),
                    active_hand=self.implicit_object_grasp_hand,
                )

            def symbolic_generator(target: Any):
                bundle_relations: tuple[LogicalInventoryRelation, ...] = ()
                if isinstance(target, PickBundleTargetToken):
                    bundle_relations = target.relations
                    target = target.target_token
                if self.v34_symbolic_actions:
                    position, orientation = target.get_position_orientation()
                    self._pre_pick.clear()
                    self._pre_pick.update(
                        {
                            "object": target,
                            "position": position.clone()
                            if callable(getattr(position, "clone", None))
                            else np.array(position, copy=True),
                            "orientation": orientation.clone()
                            if callable(getattr(orientation, "clone", None))
                            else np.array(orientation, copy=True),
                        }
                    )
                if bundle_relations:
                    generator = getattr(
                        self._primitives,
                        "marker_symbolic_pick_generator",
                        None,
                    )
                    if not callable(generator):
                        raise R1ProB1ComponentBuilderError(
                            "logical bundle Pick executor is unavailable"
                        )
                    return generator(
                        self._grasp_primitive,
                        target,
                        relations=bundle_relations,
                    )
                return self._primitives.apply_ref(
                    self._grasp_primitive,
                    target,
                    attempts=1,
                )

            def place_generator(target: Any):
                if isinstance(target, SpatialMarkerTargetToken):
                    target, selector = _resolve_spatial_marker_target(
                        target,
                        "place_inside",
                    )
                    if isinstance(target, DrawerPartTargetToken):
                        generator = getattr(
                            self._primitives,
                            "marker_symbolic_inside_compartment_generator",
                            None,
                        )
                        if not callable(generator):
                            raise R1ProB1ComponentBuilderError(
                                "selected compartment executor is unavailable"
                            )
                        yield from generator(target, selected_selector=selector)
                        self._pre_pick.clear()
                        return
                    if selector and selector.startswith(
                        (
                            "visible_container_cell_",
                            "visible_shelf_cell_",
                            "visible_shelf_level_",
                        )
                    ):
                        generator = getattr(
                            self._primitives,
                            "marker_symbolic_container_cell_generator",
                            None,
                        )
                        if not callable(generator):
                            raise R1ProB1ComponentBuilderError(
                                "selected container-cell executor is unavailable"
                            )
                        yield from generator(target, selected_selector=selector)
                        self._pre_pick.clear()
                        return
                yield from self._primitives.apply_ref(
                    self._place_inside_primitive,
                    target,
                    attempts=1,
                )
                self._pre_pick.clear()

            articulated_part_collision_pairs = (
                _articulated_part_symbolic_overlap_pairs
                if self._symbolic_zero_physics
                else _articulated_part_external_contact_pairs
            )

            def drawer_part_generator(target: DrawerPartTargetToken, *, opened: bool):
                symbolic_hold_safe()
                before_diagnostic = _articulation_pose_diagnostic(target)
                try:
                    report = yield from move_drawer_part_incrementally(
                        target,
                        opened=opened,
                        scene_objects=evaluator.env.scene.objects,
                        external_contact_pairs=lambda contents: (
                            articulated_part_collision_pairs(
                                target,
                                contents,
                            )
                        ),
                        physics_step_once=(
                            None
                            if self._symbolic_zero_physics
                            else physics_step_once
                        ),
                        **(
                            {
                                "zero_physics": True,
                                "render_once": self._render_without_physics,
                            }
                            if self._symbolic_zero_physics
                            else {}
                        ),
                    )
                except DrawerPartBlockedError as exc:
                    setattr(
                        self._primitives,
                        "_last_marker_symbolic_drawer_part_report",
                        dict(exc.report),
                    )
                    error = R1ProB1ArticulatedPartBlockedError(
                        "articulated part motion is blocked"
                    )
                    error.blocked_category = exc.blocked_category
                    error.private_report = dict(exc.report)
                    if self._symbolic_zero_physics and exc.blocked_category == "articulated_part_blocked":
                        error.blocked_detail = _articulated_obstruction_detail(
                            exc.report, robot=getattr(evaluator, "robot", None), target=target,
                            scene_objects=evaluator.env.scene.objects,
                        )
                    raise error from exc
                setattr(
                    self._primitives,
                    "_last_marker_symbolic_drawer_part_report",
                    dict(report),
                )
                after_diagnostic = _articulation_pose_diagnostic(target)
                return {**dict(report), "selected_open_before": before_diagnostic['selected_open'],
                        "selected_open_after": after_diagnostic['selected_open'],
                        "parts_before_private": before_diagnostic['parts'],
                        "parts_after_private": after_diagnostic['parts'],
                        "joint_readback_errors_private": [row['error'] for row in (before_diagnostic, after_diagnostic) if 'error' in row]}

            def open_generator(target: Any):
                if isinstance(target, DrawerPartTargetToken):
                    return (yield from drawer_part_generator(target, opened=True))
                symbolic_hold_safe()
                try:
                    report = yield from _open_marker_symbolic_target(
                        target=target,
                        scene_objects=evaluator.env.scene.objects,
                        external_contact_pairs=articulated_part_collision_pairs,
                        physics_step_once=(
                            None
                            if self._symbolic_zero_physics
                            else physics_step_once
                        ),
                        **(
                            {
                                "zero_physics": True,
                                "render_once": self._render_without_physics,
                            }
                            if self._symbolic_zero_physics
                            else {}
                        ),
                    )
                except DrawerPartBlockedError as exc:
                    setattr(
                        self._primitives,
                        "_last_marker_symbolic_drawer_part_report",
                        dict(exc.report),
                    )
                    error = R1ProB1ArticulatedPartBlockedError(
                        "articulated part motion is blocked"
                    )
                    error.blocked_category = exc.blocked_category
                    error.private_report = dict(exc.report)
                    if self._symbolic_zero_physics and exc.blocked_category == "articulated_part_blocked":
                        error.blocked_detail = _articulated_obstruction_detail(
                            exc.report, robot=getattr(evaluator, "robot", None), target=target,
                            scene_objects=evaluator.env.scene.objects,
                        )
                    raise error from exc
                setattr(
                    self._primitives,
                    "_last_marker_symbolic_drawer_part_report",
                    dict(report),
                )

                return report

            def close_generator(target: Any):
                if not isinstance(target, DrawerPartTargetToken):
                    target = _single_articulated_part_token(target)
                return (yield from drawer_part_generator(target, opened=False))

            def toggle_on_generator(target: Any):
                return self._primitives.apply_ref(
                    self._toggle_on_primitive,
                    target,
                    attempts=1,
                )

            def toggle_off_generator(target: Any):
                return self._primitives.apply_ref(
                    self._toggle_off_primitive,
                    target,
                    attempts=1,
                )

            def spatial_generator(target: Any, operation: str):
                if self._spatial_generator_factory is None:
                    raise R1ProB1ComponentBuilderError(
                        "Action23 spatial kernel is unavailable"
                    )
                center_xy, extent_xy, surface_z_m = _spatial_marker_region(
                    target,
                    operation,
                )
                target, selector = _resolve_spatial_marker_target(target, operation)
                yield from self._spatial_generator_factory(
                    target,
                    operation,
                    **(
                        {"selected_selector": selector}
                        if selector is not None
                        else {}
                    ),
                    **(
                        {
                            "placement_region_center_xy": center_xy,
                            "placement_region_extent_xy": extent_xy,
                        }
                        if center_xy is not None
                        else {}
                    ),
                    **(
                        {"placement_surface_z_m": surface_z_m}
                        if surface_z_m is not None
                        else {}
                    ),
                )
                self._pre_pick.clear()

            def place_on_top_generator(target: Any):
                yield from spatial_generator(target, "place_on_top")

            def place_next_to_generator(target: Any):
                yield from spatial_generator(target, "place_next_to")

            def place_under_generator(target: Any):
                yield from spatial_generator(target, "place_under")

            def place_touching_generator(target: Any):
                yield from spatial_generator(target, "place_touching")

            def release_generator(_target: Any):
                yield from self._primitives.apply_ref(
                    self._release_primitive,
                    attempts=1,
                )
                self._pre_pick.clear()

            def place_on_floor_generator(_target: Any):
                generator = getattr(
                    self._primitives,
                    "marker_symbolic_place_on_floor_generator",
                    None,
                )
                if not callable(generator):
                    raise R1ProB1ComponentBuilderError(
                        "place_on_floor executor is unavailable"
                    )
                yield from generator()
                self._pre_pick.clear()

            def put_back_generator(_target: Any):
                held = self._primitives._get_obj_in_hand()
                if (
                    held is None
                    or self._pre_pick.get("object") is not held
                    or "position" not in self._pre_pick
                    or "orientation" not in self._pre_pick
                ):
                    raise R1ProB1ComponentBuilderError(
                        "put_back has no matching private pre-pick pose"
                    )
                logical_snapshot = getattr(
                    self._primitives,
                    "_logical_inventory_snapshot",
                    None,
                )
                yield from self._primitives.apply_ref(
                    self._release_primitive,
                    attempts=1,
                )
                if logical_snapshot is None:
                    held.set_position_orientation(
                        position=self._pre_pick["position"],
                        orientation=self._pre_pick["orientation"],
                    )
                else:
                    report = logical_snapshot.restore(
                        set_pose=self._primitives._logical_inventory_set_pose,
                        copy_pose=self._primitives._logical_inventory_copy_pose,
                    )
                    if not report.ok:
                        raise R1ProB1ComponentBuilderError(
                            "put_back could not restore the complete held bundle"
                        ) from report.first_error
                yield None
                if logical_snapshot is not None:
                    preserved, _ = (
                        self._primitives._logical_inventory_relations_preserved(
                            logical_snapshot
                        )
                    )
                    if not preserved:
                        raise R1ProB1ComponentBuilderError(
                            "put_back did not restore the selected bundle relations"
                        )
                self._pre_pick.clear()

            def local_reach_admission(
                target: Any,
                operation: str,
            ) -> tuple[bool, Mapping[str, Any]]:
                base_position, _ = evaluator.robot.get_position_orientation()
                base_xy = np.asarray(base_position, dtype=np.float64)[:2]
                target = _resolve_marker_geometry_target(target, operation)
                nearest_xy, distance_reference = _nearest_target_xy(
                    target, base_xy
                )
                horizontal_distance_m = float(
                    np.linalg.norm(nearest_xy - base_xy)
                )
                finite = math.isfinite(horizontal_distance_m)
                maximum_local_workspace_m = self._maximum_local_reach_m
                if (
                    self._local_approach_enabled
                    and
                    self._private_safety is not None
                    and operation
                    in {
                        "grasp",
                        "pick",
                        "place_inside",
                        "place_on_top",
                        "place_next_to",
                        "place_under",
                        "place_touching",
                    }
                ):
                    maximum_local_workspace_m += self._maximum_local_approach_m
                return (
                    finite
                    and horizontal_distance_m <= maximum_local_workspace_m,
                    {
                        "horizontal_distance_m": horizontal_distance_m,
                        "maximum_local_reach_m": self._maximum_local_reach_m,
                        "maximum_local_workspace_m": maximum_local_workspace_m,
                        "target_name": str(getattr(target, "name", "")),
                        "distance_reference": distance_reference,
                        "nearest_target_xy_m": nearest_xy.tolist(),
                    },
                )

            def local_prepare(target: Any, operation: str):
                (
                    region_center_xy,
                    region_extent_xy,
                    surface_z_m,
                ) = _spatial_marker_region(target, operation)
                geometry_target = _resolve_marker_geometry_target(
                    target,
                    operation,
                )
                consume_oracle = getattr(
                    self._primitives,
                    "consume_private_oracle_symbolic_action",
                    None,
                )
                if callable(consume_oracle) and consume_oracle(target, operation):
                    return
                if (
                    operation == "place_on_top"
                    and self._on_top_station_prepare_factory is not None
                ):
                    result = yield from self._on_top_station_prepare_factory(
                        geometry_target,
                        step_once=kwargs["step_once"],
                        public_snapshot_provider=kwargs[
                            "public_snapshot_provider"
                        ],
                        placement_region_center_xy=region_center_xy,
                        placement_region_extent_xy=region_extent_xy,
                        **(
                            {"placement_surface_z_m": surface_z_m}
                            if surface_z_m is not None
                            else {}
                        ),
                    )
                    if not isinstance(result, SymbolicPreparationResult):
                        raise R1ProB1ComponentBuilderError(
                            "OnTop station preparation returned an invalid result"
                        )
                    visible_pixels = None
                    mask_source = None
                    handed_off = False
                    rolled_back = False
                    try:
                        try:
                            visible_pixels, mask_source = (
                                current_target_visibility(
                                    frame=live_private_frame(),
                                    target=geometry_target,
                                )
                            )
                            if visible_pixels < 8:
                                raise R1ProB1ComponentBuilderError(
                                    "same marked support is no longer visible "
                                    "after interaction-station navigation"
                                )
                        except Exception as exc:
                            rollback_report = None
                            rolled_back = True
                            if result.finalize is not None:
                                rollback_report = result.finalize(False)
                            failure_report = {
                                **dict(result.report),
                                "status": "blocked",
                                "reason": "target_visibility_recheck_failed",
                                "target_visibility_recheck_private": {
                                    "visible_pixels": visible_pixels,
                                    "mask_source": mask_source,
                                    "accepted": False,
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                },
                            }
                            if rollback_report is not None:
                                failure_report["rollback_private"] = dict(
                                    rollback_report
                                )
                            error = R1ProB1ComponentBuilderError(
                                "same marked support failed visibility "
                                "recheck"
                            )
                            error.private_report = failure_report
                            raise error from exc
                        prepared = SymbolicPreparationResult(
                            {
                                **dict(result.report),
                                "target_visibility_recheck_private": {
                                    "visible_pixels": int(visible_pixels),
                                    "mask_source": mask_source,
                                    "accepted": True,
                                },
                            },
                            result.finalize,
                        )
                        handed_off = True
                        return prepared
                    finally:
                        # Generator close and other non-Exception control flow
                        # must roll back, but must never be converted to a
                        # public blocked result.
                        if (
                            not handed_off
                            and not rolled_back
                            and result.finalize is not None
                        ):
                            result.finalize(False)
                if (
                    self._automatic_interaction_approach_enabled
                    and operation
                    in AUTOMATIC_INTERACTION_APPROACH_OPERATIONS
                ):
                    automatic_prepare_started = perf_counter()
                    base_position, _ = evaluator.robot.get_position_orientation()
                    base_xy = np.asarray(base_position, dtype=np.float64)[:2]
                    nearest_xy, distance_reference = _nearest_target_xy(
                        geometry_target,
                        base_xy,
                    )
                    initial_distance = float(
                        np.linalg.norm(nearest_xy - base_xy)
                    )
                    if initial_distance <= self._maximum_local_reach_m:
                        return
                    prepare_factory = self._interaction_approach_prepare_factory
                    if prepare_factory is None:
                        raise R1ProB1ComponentBuilderError(
                            "automatic interaction approach is unavailable"
                        )
                    approach_factory_started = perf_counter()
                    result = yield from prepare_factory(
                        geometry_target,
                        operation=operation,
                        interaction_target_xy=nearest_xy,
                        step_once=kwargs["step_once"],
                        public_snapshot_provider=kwargs[
                            "public_snapshot_provider"
                        ],
                    )
                    approach_factory_seconds = (
                        perf_counter() - approach_factory_started
                    )
                    if not isinstance(result, SymbolicPreparationResult):
                        raise R1ProB1ComponentBuilderError(
                            "interaction approach returned an invalid result"
                        )
                    arrival_revalidation_started = perf_counter()
                    arrival_failure_category = EXECUTION_UNVERIFIED
                    try:
                        arrived_position, _ = (
                            evaluator.robot.get_position_orientation()
                        )
                        arrived_xy = np.asarray(
                            arrived_position,
                            dtype=np.float64,
                        )[:2]
                        arrived_nearest_xy, arrived_distance_reference = (
                            _nearest_target_xy(
                                geometry_target,
                                arrived_xy,
                            )
                        )
                        arrived_distance = float(
                            np.linalg.norm(arrived_nearest_xy - arrived_xy)
                        )
                        live_candidates = operation_marker_objects_for(
                            operation
                        )
                        target_name = str(
                            getattr(geometry_target, "name", "")
                        )
                        same_target_legal = bool(
                            target_name
                            and live_candidates.get(target_name)
                            is geometry_target
                        )
                        visible_pixels, mask_source = current_target_visibility(
                            frame=live_private_frame(),
                            target=geometry_target,
                        )
                        if (
                            arrived_distance > self._maximum_local_reach_m
                            or not same_target_legal
                            or visible_pixels < 8
                        ):
                            arrival_failure_category = (
                                OUT_OF_REACH
                                if arrived_distance > self._maximum_local_reach_m
                                else SELECTION_UNAVAILABLE
                            )
                            raise R1ProB1ComponentBuilderError(
                                "same marked target is not executable after "
                                "interaction approach"
                            )
                    except BaseException as exc:
                        phase_timing = dict(
                            result.report.get("phase_timing_private", {})
                        )
                        phase_timing.update(
                            {
                                "approach_factory_seconds": (
                                    approach_factory_seconds
                                ),
                                "arrival_revalidation_seconds": (
                                    perf_counter()
                                    - arrival_revalidation_started
                                ),
                                "local_prepare_total_seconds": (
                                    perf_counter()
                                    - automatic_prepare_started
                                ),
                            }
                        )
                        rollback_report = (
                            None
                            if result.finalize is None
                            else result.finalize(False)
                        )
                        if not isinstance(exc, Exception):
                            raise
                        failure_report = {
                            **dict(result.report),
                            "status": "blocked",
                            "reason": "arrival_target_invalid",
                            "phase_timing_private": phase_timing,
                            "arrival_revalidation_private": {
                                "accepted": False,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            },
                        }
                        if rollback_report is not None:
                            failure_report["rollback_private"] = dict(
                                rollback_report
                            )
                        error = R1ProB1ComponentBuilderError(
                            "automatic interaction approach arrival "
                            "revalidation failed"
                        )
                        error.private_report = failure_report
                        error.blocked_category = arrival_failure_category
                        raise error from exc
                    phase_timing = dict(
                        result.report.get("phase_timing_private", {})
                    )
                    phase_timing.update(
                        {
                            "approach_factory_seconds": (
                                approach_factory_seconds
                            ),
                            "arrival_revalidation_seconds": (
                                perf_counter() - arrival_revalidation_started
                            ),
                            "local_prepare_total_seconds": (
                                perf_counter() - automatic_prepare_started
                            ),
                        }
                    )
                    return SymbolicPreparationResult(
                        {
                            **dict(result.report),
                            "phase_timing_private": phase_timing,
                            "initial_target_distance_private": {
                                "horizontal_distance_m": initial_distance,
                                "distance_reference": distance_reference,
                            },
                            "arrival_revalidation_private": {
                                "accepted": True,
                                "same_target_legal": True,
                                "visible_pixels": int(visible_pixels),
                                "mask_source": mask_source,
                                "horizontal_distance_m": arrived_distance,
                                "distance_reference": (
                                    arrived_distance_reference
                                ),
                            },
                        },
                        result.finalize,
                    )
                if (
                    self._private_safety is None
                    or not self._local_approach_enabled
                ):
                    return
                if operation not in {
                    "grasp",
                    "place_inside",
                    "place_on_top",
                    "place_next_to",
                    "place_under",
                    "place_touching",
                }:
                    # Open/close/toggle actions do not move the base as a
                    # hidden side effect.  Their admission above is the
                    # complete configured local-workspace contract.
                    return

                def planar_distance() -> tuple[float, np.ndarray, np.ndarray]:
                    base_position, base_quaternion = (
                        evaluator.robot.get_position_orientation()
                    )
                    base = np.asarray(base_position, dtype=np.float64)
                    quat = np.asarray(base_quaternion, dtype=np.float64)
                    if (
                        base.shape != (3,)
                        or quat.shape != (4,)
                        or not np.isfinite(base).all()
                        or not np.isfinite(quat).all()
                    ):
                        raise R1ProB1ComponentBuilderError(
                            "marker local approach received invalid poses"
                        )
                    nearest_xy, _ = _nearest_target_xy(
                        geometry_target, base[:2]
                    )
                    return float(np.linalg.norm(nearest_xy - base[:2])), base, quat

                initial_distance, initial_base, _ = planar_distance()
                if initial_distance > (
                    self._maximum_local_reach_m
                    + self._maximum_local_approach_m
                ):
                    raise R1ProB1ComponentBuilderError(
                        "marked target is outside bounded local approach"
                    )
                previous_distance = initial_distance
                low_progress_streak = 0
                step_count = 0
                while previous_distance > self._local_approach_stop_m:
                    if step_count >= self._local_approach_maximum_steps:
                        raise R1ProB1ComponentBuilderError(
                            "marker local approach exhausted its step budget"
                        )
                    distance, base, quat = planar_distance()
                    travelled = float(
                        np.linalg.norm(base[:2] - initial_base[:2])
                    )
                    if travelled >= self._maximum_local_approach_m:
                        raise R1ProB1ComponentBuilderError(
                            "marker local approach exhausted its distance budget"
                        )
                    nearest_xy, _ = _nearest_target_xy(
                        geometry_target, base[:2]
                    )
                    delta = nearest_xy - base[:2]
                    x, y, z, w = (float(value) for value in quat)
                    yaw = math.atan2(
                        2.0 * (w * z + x * y),
                        1.0 - 2.0 * (y * y + z * z),
                    )
                    cy, sy = math.cos(yaw), math.sin(yaw)
                    local = np.asarray(
                        (cy * delta[0] + sy * delta[1],
                         -sy * delta[0] + cy * delta[1]),
                        dtype=np.float64,
                    )
                    norm = float(np.linalg.norm(local))
                    if not math.isfinite(norm) or norm <= 1e-6:
                        raise R1ProB1ComponentBuilderError(
                            "marker local approach direction is invalid"
                        )
                    speed = 1.0 if operation == "grasp" else 0.30
                    base_command = np.asarray(
                        (speed * local[0] / norm, speed * local[1] / norm, 0.0),
                        dtype=np.float32,
                    )
                    public_state = control.public_state()
                    clearance = float(
                        public_state.get("head_forward_clearance_m", float("nan"))
                    )
                    if (
                        base_command[0] > 0.2
                        and (
                            not math.isfinite(clearance)
                            or clearance <= self._local_approach_hard_stop_m
                        )
                    ):
                        raise R1ProB1ComponentBuilderError(
                            "marker local approach encountered a forward obstacle"
                        )
                    current = _snapshot(kwargs["public_snapshot_provider"])
                    action = _marker_symbolic_closed_hold_action(
                        current,
                        hand=self.implicit_object_grasp_hand,
                    )
                    if operation == "grasp":
                        # Local approach must preserve the base component at
                        # the canary server boundary.  MODE_DIRECT_ACTION is
                        # intentionally arm-local and zeros action[:3]; use
                        # the carry channel here as a bounded, closed-hand
                        # approach before the symbolic carry state exists.
                        # This keeps the action semantic (no object pose is
                        # exposed) while making the requested base motion real.
                        expected_obj = None
                    else:
                        expected_obj = self._primitives._get_obj_in_hand()
                        _assert_marker_symbolic_grasp_held(
                            self._primitives,
                            hand=self.implicit_object_grasp_hand,
                            expected_obj=expected_obj,
                            phase=MarkerSymbolicCarryPhase.BEFORE_SETTLE_STEP,
                        )
                    action[R1PRO_ACTION_LAYOUT.slice_for("base")] = base_command
                    notify_r1pro_command(self._private_safety, action)
                    control.command_carry_action(
                        action,
                        active_hand=self.implicit_object_grasp_hand,
                    )
                    labelled_step("marker_symbolic_local_approach")
                    if operation in {
                        "place_inside",
                        "place_on_top",
                        "place_next_to",
                        "place_under",
                        "place_touching",
                    }:
                        _assert_marker_symbolic_grasp_held(
                            self._primitives,
                            hand=self.implicit_object_grasp_hand,
                            expected_obj=expected_obj,
                            phase=MarkerSymbolicCarryPhase.AFTER_SETTLE_STEP,
                        )
                    safety = dict(self._private_safety())
                    stability = dict(self._public_stability())
                    if (
                        bool(safety.get("base_contact"))
                        or bool(safety.get("predicted_self_collision"))
                        or bool(safety.get("joint_stalled"))
                        or abs(float(stability.get("base_z_drift_m", 0.0))) > 0.03
                        or abs(float(stability.get("roll_deg", 0.0))) > 5.0
                        or abs(float(stability.get("pitch_deg", 0.0))) > 5.0
                    ):
                        raise R1ProB1ComponentBuilderError(
                            "marker local approach was physically blocked"
                        )
                    new_distance, _, _ = planar_distance()
                    progress = previous_distance - new_distance
                    low_progress_streak = (
                        low_progress_streak + 1 if progress < 0.002 else 0
                    )
                    if low_progress_streak >= 3:
                        raise R1ProB1ComponentBuilderError(
                            "marker local approach stalled"
                        )
                    previous_distance = new_distance
                    step_count += 1
                    yield 1

                fresh = live_private_frame()
                visible_pixels, _ = current_target_visibility(
                    frame=fresh,
                    target=geometry_target,
                )
                if visible_pixels < 8:
                    raise R1ProB1ComponentBuilderError(
                        "same marked instance is no longer visible after local approach"
                    )

            held_object_provider = getattr(
                self._primitives,
                "_get_obj_in_hand",
                None,
            )
            if not callable(held_object_provider):
                raise R1ProB1ComponentBuilderError(
                    "symbolic primitive lacks a held-object provider"
                )

            private_outcome_verifier = None
            if self.action23_symbolic_actions:
                predicate_mode = placement_predicate_mode(
                    symbolic_zero_physics=self._symbolic_zero_physics,
                    physical_consequences_enabled=(
                        self._placement_physical_consequences_enabled
                    ),
                )

                def read_action_outcome_state(
                    obj: Any,
                    state_name: str,
                    other: Any | None,
                ) -> bool:
                    return _read_marker_symbolic_canonical_state(
                        obj,
                        state_name,
                        other,
                        symbolic_on_top=predicate_mode == "action20_symbolic",
                    )

                action23_verifier = Action23OutcomeVerifier(
                    read_state=read_action_outcome_state,
                    read_held=held_object_provider,
                )

                def verify_action23_outcome(
                    operation: str,
                    target: Any | None,
                    acted_object: Any | None,
                    simulator_steps: int,
                ) -> Mapping[str, Any]:
                    if isinstance(target, PickBundleTargetToken):
                        target = target.target_token
                    execution_report = getattr(
                        self._primitives,
                        "_last_private_exact_report",
                        {},
                    )
                    if operation == "place_on_floor":
                        support_name = (
                            execution_report.get("selected_support_private")
                            if isinstance(execution_report, Mapping)
                            else None
                        )
                        target = self._primitives.env.scene.object_registry(
                            "name", support_name
                        )
                    report = dict(
                        action23_verifier.verify(
                            operation=operation,
                            target=target,
                            acted_object=acted_object,
                            simulator_steps=simulator_steps,
                        )
                    )
                    report["predicate_mode"] = predicate_mode
                    if (
                        predicate_mode == "action20_symbolic"
                        and operation in {"place_on_floor", "place_on_top"}
                        and target is not None
                        and acted_object is not None
                    ):
                        try:
                            raw_official = _read_marker_symbolic_canonical_state(
                                acted_object,
                                "OnTop",
                                target,
                            )
                            raw_error = None
                        except Exception as exc:
                            raw_official = None
                            raw_error = f"{type(exc).__name__}: {exc}"
                        report.update(
                            {
                                "predicate_readback_policy": (
                                    "action20_symbolic_geometry"
                                ),
                                "raw_official_observed_private": raw_official,
                                "raw_official_error_private": raw_error,
                            }
                        )
                    if (
                        isinstance(execution_report, Mapping)
                        and execution_report.get("operation") == operation
                        and execution_report.get(
                            "physical_consequence_committed"
                        )
                        is True
                    ):
                        report["physical_consequence_committed"] = True
                    return report

                private_outcome_verifier = verify_action23_outcome

            backend = MarkerSymbolicGraspBackendAdapter(
                marker_store=self._marker_store,
                private_frame_provider=(
                    (lambda: initial_private_frame)
                    if marker_frame_unavailable
                    else live_private_frame
                ),
                raw_head_rgb_provider=raw_head_rgb,
                eligible_objects_provider=lambda: marker_objects,
                graspable_object_names_provider=lambda: pickable_names,
                place_container_names_provider=lambda: container_names,
                openable_object_names_provider=lambda: openable_names,
                support_object_names_provider=lambda: support_names,
                operation_marker_objects_provider=operation_marker_objects_for,
                on_top_snapshot_provider=self._on_top_snapshot_provider,
                next_to_snapshot_provider=self._next_to_snapshot_provider,
                toggleable_object_names_provider=lambda: toggleable_names,
                official_bbox_provider=official_instance_bboxes,
                symbolic_generator_factory=symbolic_generator,
                symbolic_place_generator_factory=place_generator,
                symbolic_operation_factories=(
                    {
                        "open": open_generator,
                        "close": close_generator,
                        "release": release_generator,
                        "place_on_floor": place_on_floor_generator,
                        **(
                            {
                                "toggle_on": toggle_on_generator,
                                "toggle_off": toggle_off_generator,
                                **(
                                    {
                                        "place_on_top": place_on_top_generator,
                                        "place_next_to": place_next_to_generator,
                                        "place_under": place_under_generator,
                                        "place_touching": place_touching_generator,
                                    }
                                    if self.action23_symbolic_actions
                                    else {}
                                ),
                            }
                            if self.toggle_symbolic_actions
                            else {}
                        ),
                        "put_back": put_back_generator,
                    }
                    if self.v34_symbolic_actions
                    else None
                ),
                command_step=command_step,
                hold_safe=symbolic_hold_safe,
                execution_integrity_state=evaluator,
                placement_rollback_factory=(
                    lambda target: self._primitives._capture_symbolic_placement_rollback(
                        # Snapshot the live owner, not an intermediate compartment
                        # token. Keep the original token for selected-cell execution.
                        _resolve_marker_geometry_target(
                            target,
                            target.operation if isinstance(target, SpatialMarkerTargetToken)
                            else "placement_checkpoint",
                        ) if target is not None else None,
                        None,
                    )
                ) if self._symbolic_zero_physics and not self._placement_physical_consequences_enabled else None,
                # Always reject targets beyond the complete local workspace
                # before a symbolic token is created.  With live safety
                # enabled, the workspace includes the bounded physical
                # approach distance; without it, only direct arm reach is
                # admitted.
                local_reach_admission=local_reach_admission,
                local_prepare_factory=(
                    local_prepare
                    if self._private_safety is not None
                    or self._on_top_station_prepare_factory is not None
                    or self._interaction_approach_prepare_factory is not None
                    else None
                ),
                prepare_place_on_top=(
                    self._on_top_station_prepare_factory is not None
                ),
                automatic_interaction_approach_enabled=(
                    self._automatic_interaction_approach_enabled
                ),
                placement_physical_consequences_enabled=(
                    self._placement_physical_consequences_enabled
                ),
                held_object_provider=held_object_provider,
                private_outcome_verifier=private_outcome_verifier,
                pick_bundle_targets_provider=lambda: pick_bundle_targets,
            )
        except Exception as exc:
            raise _component_error("marker_symbolic", exc) from exc
        return MappingProxyType({MARKER_SYMBOLIC_BACKEND_SERVICE: backend})


class R1ProPlaceComponentBuilder:
    """Observation-only place adapter with explicit target-pose construction."""

    api_contract_version = R1PRO_PLACE_SERVICES_VERSION

    def __init__(
        self,
        *,
        starter_by_hand: Mapping[str, Any],
        allowed_hands: Sequence[str] = ("left", "right"),
        camera_intrinsics: Mapping[str, Any],
        target_pose_builder: Callable[
            [tuple[float, float, float], str], tuple[Any, Any]
        ],
        safe_hold_action: Callable[[str, bool], Sequence[float]],
        limits: R1ProPlaceMotionLimits | None = None,
        visual_config: LocalRGBPatchAssessment | None = None,
        place_bundle_builder: Callable[..., Any] = (
            build_r1pro_place_service_bundle
        ),
    ) -> None:
        self._allowed_hands = _hand_set(allowed_hands)
        self._starter_by_hand = _starter_map(
            starter_by_hand,
            allowed_hands=self._allowed_hands,
        )
        self._camera_intrinsics = _mapping(
            camera_intrinsics,
            "camera_intrinsics",
        )
        _require_callable(target_pose_builder, "target_pose_builder")
        _require_callable(safe_hold_action, "safe_hold_action")
        _require_callable(place_bundle_builder, "place_bundle_builder")
        self._target_pose_builder = target_pose_builder
        self._safe_hold_action = safe_hold_action
        self._limits = limits
        self._visual_config = visual_config
        self._bundle_builder = place_bundle_builder

    def build(self, **kwargs: Any) -> Mapping[str, Any]:
        action, active_hand = _common_action_context(
            kwargs,
            expected_actions=frozenset({"place_inside_at"}),
        )
        if (
            not isinstance(action.args, PointActionArgs)
            or active_hand != action.args.hand
        ):
            raise R1ProB1ComponentBuilderError(
                "place_inside_at requires matching public active_hand"
            )
        if action.args.hand not in self._allowed_hands:
            raise R1ProB1ComponentBuilderError(
                "place hand is not admitted by live calibration"
            )
        starter = self._starter_by_hand[action.args.hand]
        frozen_snapshot = _snapshot(kwargs["public_snapshot_provider"])
        try:
            bundle = self._bundle_builder(
                starter=starter,
                shared_control=kwargs["control"],
                evaluator_step=_LabeledStepOnce(kwargs["step_once"]),
                target_pose_builder=self._target_pose_builder,
                safe_hold_action=self._safe_hold_action,
                initial_frozen_public_snapshot=frozen_snapshot,
                snapshot_provider=kwargs["public_snapshot_provider"],
                camera_intrinsics=self._camera_intrinsics,
                limits=self._limits,
                visual_config=self._visual_config,
            )
        except Exception as exc:
            raise _component_error("place", exc) from exc
        return _bundle_services(
            bundle,
            component="place",
            allowed_keys={
                RGBD_SOURCE_SERVICE,
                MOTION_SERVICE,
                VISUAL_ASSESSOR_SERVICE,
            },
        )


def _build_actuator_bindings(
    builder: Callable[..., Any],
    *,
    motion: _MotionDependencies,
    kwargs: Mapping[str, Any],
    active_hand: str | None,
    component: str,
) -> Any:
    try:
        return builder(
            **motion.proxy_kwargs(
                control=kwargs["control"],
                public_snapshot_provider=kwargs[
                    "public_snapshot_provider"
                ],
                step_once=kwargs["step_once"],
                public_grasp_proxy_source=kwargs[
                    "public_grasp_proxy_source"
                ],
            ),
            active_hand=active_hand,
        )
    except R1ProActuatorBindingError as exc:
        raise R1ProB1ComponentBuilderError(
            f"{component} construction failed closed: {exc}"
        ) from exc
    except Exception as exc:
        raise _component_error(component, exc) from exc


def _build_local_base_binding(
    builder: Callable[..., Any],
    *,
    motion: _MotionDependencies,
    kwargs: Mapping[str, Any],
) -> Any:
    try:
        return builder(
            **motion.base_kwargs(
                control=kwargs["control"],
                public_snapshot_provider=kwargs[
                    "public_snapshot_provider"
                ],
                step_once=kwargs["step_once"],
            )
        )
    except R1ProActuatorBindingError as exc:
        raise R1ProB1ComponentBuilderError(
            f"local-base construction failed closed: {exc}"
        ) from exc
    except Exception as exc:
        raise _component_error("local-base", exc) from exc


def _build_torso_release_bindings(
    builder: Callable[..., Any],
    *,
    motion: _MotionDependencies,
    kwargs: Mapping[str, Any],
    max_base_z_drift_m: float,
    max_abs_roll_pitch_deg: float,
) -> Any:
    try:
        return builder(
            **motion.proxy_kwargs(
                control=kwargs["control"],
                public_snapshot_provider=kwargs[
                    "public_snapshot_provider"
                ],
                step_once=kwargs["step_once"],
                public_grasp_proxy_source=kwargs[
                    "public_grasp_proxy_source"
                ],
            ),
            max_base_z_drift_m=max_base_z_drift_m,
            max_abs_roll_pitch_deg=max_abs_roll_pitch_deg,
        )
    except Exception as exc:
        raise _component_error("torso/release actuator", exc) from exc


def _common_action_context(
    kwargs: Mapping[str, Any],
    *,
    expected_actions: frozenset[str],
) -> tuple[ActionEnvelope, str | None]:
    required = {
        "action_index",
        "action",
        "evaluator",
        "control",
        "public_snapshot_provider",
        "step_once",
        "public_carry_snapshot",
        "public_grasp_proxy_source",
        "active_hand",
    }
    optional = {
        "physics_step_once",
        "mark_boundary_already_checkpointed",
    }
    missing = required.difference(kwargs)
    unknown = set(kwargs).difference(required | optional)
    if missing or unknown:
        raise R1ProB1ComponentBuilderError(
            f"component build kwargs mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    action_index = kwargs["action_index"]
    if (
        isinstance(action_index, bool)
        or not isinstance(action_index, int)
        or action_index < 0
    ):
        raise R1ProB1ComponentBuilderError(
            "action_index must be non-negative"
        )
    action = kwargs["action"]
    if not isinstance(action, ActionEnvelope):
        raise R1ProB1ComponentBuilderError(
            "action must be ActionEnvelope"
        )
    if action.action not in expected_actions:
        raise R1ProB1ComponentBuilderError(
            f"component does not support action {action.action!r}"
        )
    for name in (
        "public_snapshot_provider",
        "step_once",
        "public_carry_snapshot",
    ):
        _require_callable(kwargs[name], name)
    source = kwargs["public_grasp_proxy_source"]
    if not callable(getattr(source, "grasp_proxy_likely", None)):
        raise R1ProB1ComponentBuilderError(
            "public_grasp_proxy_source is invalid"
        )
    active_hand = kwargs["active_hand"]
    if active_hand not in {None, "left", "right"}:
        raise R1ProB1ComponentBuilderError("active_hand is invalid")
    return action, active_hand


def _public_proprio(
    provider: Callable[[], Mapping[str, Any]],
) -> Callable[[], Sequence[float]]:
    _require_callable(provider, "public_snapshot_provider")

    def read() -> Sequence[float]:
        snapshot = _snapshot(provider)
        if "proprio" not in snapshot:
            raise R1ProB1ComponentBuilderError(
                "public snapshot lacks proprio"
            )
        return snapshot["proprio"]

    return read


def _snapshot(
    provider: Callable[[], Mapping[str, Any]],
) -> Mapping[str, Any]:
    _require_callable(provider, "public_snapshot_provider")
    value = provider()
    if not isinstance(value, Mapping):
        raise R1ProB1ComponentBuilderError(
            "public_snapshot_provider must return a mapping"
        )
    return value


def _marker_symbolic_closed_hold_action(
    snapshot: Mapping[str, Any],
    *,
    hand: str,
) -> np.ndarray:
    """Build a closed-hand hold from the resident public snapshot subset."""

    if hand not in {"left", "right"}:
        raise R1ProB1ComponentBuilderError(
            "marker symbolic hold hand must be left or right"
        )
    if "proprio" not in snapshot:
        raise R1ProB1ComponentBuilderError(
            "marker symbolic hold snapshot lacks proprio"
        )
    proprio = np.asarray(snapshot["proprio"], dtype=np.float32)
    if proprio.shape != (PROPRIO_DIM,) or not np.isfinite(proprio).all():
        raise R1ProB1ComponentBuilderError(
            "marker symbolic hold proprio must be finite and 61D"
        )
    action = np.zeros(R1PRO_ACTION_LAYOUT.dimension, dtype=np.float32)
    action[R1PRO_ACTION_LAYOUT.slice_for("torso")] = proprio[TRUNK_QPOS]
    action[R1PRO_ACTION_LAYOUT.slice_for("left_arm")] = proprio[
        LEFT_ARM_QPOS
    ]
    action[R1PRO_ACTION_LAYOUT.slice_for("right_arm")] = proprio[
        RIGHT_ARM_QPOS
    ]
    action[R1PRO_ACTION_LAYOUT.slice_for("left_gripper")] = (
        -1.0 if hand == "left" else 1.0
    )
    action[R1PRO_ACTION_LAYOUT.slice_for("right_gripper")] = (
        -1.0 if hand == "right" else 1.0
    )
    return action


def _assert_marker_symbolic_grasp_held(
    primitives: Any,
    *,
    hand: str,
    expected_obj: Any,
    phase: MarkerSymbolicCarryPhase,
) -> None:
    """Fail closed if a marker-bound Symbolic hold is lost during settle."""

    if hand not in {"left", "right"}:
        raise R1ProB1ComponentBuilderError(
            "marker symbolic grasp assertion hand must be left or right"
        )
    if expected_obj is None:
        raise R1ProB1ComponentBuilderError(
            "marker symbolic grasp assertion lacks expected object"
        )
    try:
        phase = require_marker_symbolic_carry_phase(phase)
    except MarkerSymbolicCarryContractError as exc:
        raise R1ProB1ComponentBuilderError(str(exc)) from exc
    kinematic_validator = getattr(
        primitives,
        "is_marker_symbolic_hold_valid",
        None,
    )
    if callable(kinematic_validator):
        valid = kinematic_validator(hand, expected_obj)
        if not isinstance(valid, bool):
            raise R1ProB1ComponentBuilderError(
                "marker symbolic hold validator must return boolean"
            )
        if not valid:
            raise R1ProB1ComponentBuilderError(
                f"marker symbolic kinematic hold was lost {phase.value}"
            )
        return

    robot = getattr(primitives, "robot", None)
    held_by_arm = getattr(robot, "_ag_obj_in_hand", None)
    constraints = getattr(robot, "_ag_obj_constraints", None)
    if not isinstance(held_by_arm, Mapping) or not isinstance(
        constraints, Mapping
    ):
        raise R1ProB1ComponentBuilderError(
            "official symbolic grasp state mappings are unavailable"
        )
    if held_by_arm.get(hand) is not expected_obj:
        raise R1ProB1ComponentBuilderError(
            f"official symbolic grasp object was lost {phase.value}"
        )
    if constraints.get(hand) is None:
        raise R1ProB1ComponentBuilderError(
            f"official symbolic grasp constraint was lost {phase.value}"
        )


def _private_observation_count(control: Any) -> int:
    """Read a private monotonic frame identity without exposing it."""

    public_state = getattr(control, "public_state", None)
    if not callable(public_state):
        raise R1ProB1ComponentBuilderError(
            "control lacks public_state() observation counter"
        )
    state = public_state()
    count = state.get("observation_count") if isinstance(state, Mapping) else None
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise R1ProB1ComponentBuilderError(
            "control observation_count is invalid"
        )
    return count


def _bundle_services(
    bundle: Any,
    *,
    component: str,
    allowed_keys: set[str],
) -> Mapping[str, Any]:
    services = getattr(bundle, "services", None)
    if not isinstance(services, Mapping):
        raise R1ProB1ComponentBuilderError(
            f"{component} bundle services must be a mapping"
        )
    keys = set(services)
    if not keys or not keys.issubset(allowed_keys):
        raise R1ProB1ComponentBuilderError(
            f"{component} bundle returned unexpected service keys: "
            f"{sorted(keys.difference(allowed_keys))}"
        )
    if any(not isinstance(key, str) or not key for key in services):
        raise R1ProB1ComponentBuilderError(
            f"{component} bundle returned an invalid service key"
        )
    return MappingProxyType(dict(services))


def _hand_set(value: Sequence[str]) -> frozenset[str]:
    hands = tuple(value)
    if (
        not hands
        or len(set(hands)) != len(hands)
        or not set(hands).issubset({"left", "right"})
    ):
        raise R1ProB1ComponentBuilderError(
            "allowed_hands must be unique left/right hand names"
        )
    return frozenset(hands)


def _starter_map(
    value: Mapping[str, Any],
    *,
    allowed_hands: frozenset[str],
) -> Mapping[str, Any]:
    mapping = _mapping(value, "starter_by_hand")
    if set(mapping) != set(allowed_hands):
        raise R1ProB1ComponentBuilderError(
            "starter_by_hand must define exactly the allowed_hands"
        )
    if any(starter is None for starter in mapping.values()):
        raise R1ProB1ComponentBuilderError(
            "starter_by_hand cannot contain placeholders"
        )
    return mapping


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) or not key for key in value):
        raise ValueError(f"{name} keys must be non-empty strings")
    return MappingProxyType(dict(value))


def _optional_mapping(
    value: Mapping[str, Any] | None,
    name: str,
) -> Mapping[str, Any] | None:
    return None if value is None else _mapping(value, name)


def _require_callable(value: Any, name: str) -> None:
    if not callable(value):
        raise TypeError(f"{name} must be callable")


def _positive(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be positive")
    return float(value)


def _component_error(component: str, exc: Exception) -> Exception:
    if isinstance(exc, R1ProB1ComponentBuilderError):
        return exc
    return R1ProB1ComponentBuilderError(
        f"{component} component construction failed: "
        f"{type(exc).__name__}: {exc}"
    )
