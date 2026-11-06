"""Live marker-symbolic spatial placement execution."""

from __future__ import annotations

import math
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from behavior_hl.embodied.evaluator_bridge.marker_symbolic_inside_execution import (
    _refresh_inside_readback_without_physics,
)
from behavior_hl.embodied.evaluator_bridge.on_top_support_layers import (
    DEFAULT_PROTECTED_ROTATION_RAD,
    plan_on_top_support_layer_candidates,
    plan_support_plane_lateral_bindings,
    quaternion_angle_rad,
    restrict_and_rank_on_top_plan,
    summarize_original_destination_on_top,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_visible_official_grasp_factory import (
    planar_base_pose_from_robot,
)
from behavior_hl.embodied.evaluator_bridge.spatial_candidate_admission import (
    execute_admitted_spatial_candidate,
    query_spatial_candidate_collision_overlap,
    query_symbolic_placement_final_collision,
    spatial_object_link_paths,
)
from behavior_hl.embodied.evaluator_bridge.spatial_symbolic_actions import (
    DEFAULT_MAX_ANGULAR_SPEED_RAD_S,
    DEFAULT_MAX_LINEAR_SPEED_M_S,
    DEFAULT_SETTLE_STEPS,
    DEFAULT_STABILITY_WINDOW,
    DEFAULT_SYMBOLIC_CANDIDATE_TRIAL_LIMIT,
    DownwardSupportRayEvidence,
    SPATIAL_ACTION_TO_PREDICATE,
    SpatialPoseCandidate,
    VerticalPlacementEvidence,
    build_downward_support_ray_requests,
    build_spatial_partial_report,
    candidate_aabb,
    candidate_blockers,
    expanded_support_agnostic_lateral_candidates,
    object_aabb,
    placement_predicate_mode,
    plan_default_under_supports,
    rebase_candidate_to_local_support,
    sample_spatial_stability,
    select_next_to_screen_candidate,
    select_touching_screen_candidate,
    summarize_relation_authoritative_postcondition,
    summarize_spatial_stability,
    summarize_supported_lateral_postcondition,
    summarize_symbolic_spatial_postcondition,
)
from behavior_hl.embodied.policy.profiles import ACTION20_LOCAL_WORKSPACE_M
from behavior_hl.embodied.runtime.public_action_feedback import (
    ACTION20_BLOCKED_CATEGORY_METADATA_KEY,
    EXECUTION_UNVERIFIED,
    GEOMETRIC_FIT_BLOCKED,
    OUT_OF_REACH,
    PLACEMENT_COLLISION_BLOCKED,
    PLACEMENT_SUPPORT_UNAVAILABLE,
    RELATION_NOT_REALIZED,
    placement_failure_category,
    REQUIRES_OBJECT_IN_LEFT_HAND,
)


class _PrivateVerifiedSpatialTerminalSuccess(BaseException):
    """Private runner signal that preserves a verifier-confirmed terminal pose."""


def _support_rebase_failure_category(report):
    # Only positively identified geometric rejection is a deterministic failure.
    return (
        PLACEMENT_SUPPORT_UNAVAILABLE
        if report.reason in {
            "support_ray_no_hit", "support_ray_hit_is_not_horizontal",
            "support_ray_hit_forbidden_entity_role",
            "support_ray_hits_lack_coplanar_majority",
        }
        else EXECUTION_UNVERIFIED
    )


def _spatial_collision_failure_category(report: Any) -> str | None:
    if report is None or bool(getattr(report, "accepted", False)):
        return None
    object_reports = tuple(getattr(report, "object_reports", ()))
    reports = (
        tuple(item for _name, item in object_reports)
        if object_reports
        else (report,)
    )
    if any(getattr(item, "query_error", None) for item in reports):
        return EXECUTION_UNVERIFIED
    if any(tuple(getattr(item, "blocking_hits", ())) for item in reports):
        return PLACEMENT_COLLISION_BLOCKED
    return EXECUTION_UNVERIFIED


def _empty_on_top_plan_category(plan: Any) -> str:
    metadata = plan.planning_metadata_private or {}
    if metadata.get("original_plane_candidate_count", 0) > 0 and metadata.get("reachable_candidate_count") == 0:
        return OUT_OF_REACH
    return GEOMETRIC_FIT_BLOCKED


def _empty_under_plans_category(plans: Sequence[Any]) -> str | None:
    if not plans:
        return PLACEMENT_SUPPORT_UNAVAILABLE
    categories = set()
    for support_plan in plans:
        plan = support_plan.candidate_plan
        metadata = plan.planning_metadata_private or {}
        if metadata.get("geometry_query_failed") or metadata.get("vertical_probe_available") is False:
            categories.add(EXECUTION_UNVERIFIED)
        elif metadata.get("footprint_fits") is False:
            categories.add(GEOMETRIC_FIT_BLOCKED)
        elif not plan.admission_reports:
            categories.add(None)
        for report in plan.admission_reports:
            evidence = report.vertical_evidence
            if report.blockers:
                categories.add(PLACEMENT_COLLISION_BLOCKED)
            elif evidence is None:
                categories.add(None)
            elif evidence.error is not None:
                categories.add(EXECUTION_UNVERIFIED)
            elif not evidence.target_first_above or not evidence.support_first_below:
                categories.add(PLACEMENT_SUPPORT_UNAVAILABLE)
            elif (
                evidence.clear_underside and evidence.vertical_clearance_m is not None
                and math.isfinite(evidence.vertical_clearance_m)
                and evidence.vertical_clearance_m < metadata.get("required_clearance_m", -math.inf)
            ):
                categories.add(GEOMETRIC_FIT_BLOCKED)
            else:
                categories.add(None)
    return placement_failure_category(categories)


def _rollback_spatial_candidate_trial(
    *,
    omnigibson: Any,
    checkpoint: Any | None,
    reproject_carry: Callable[[], Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Restore an applied trial before returning the object to its carry pose."""

    checkpoint_error = None
    if checkpoint is not None:
        try:
            omnigibson.sim.load_state(checkpoint, serialized=False)
        except BaseException as exc:
            checkpoint_error = exc
    try:
        report = reproject_carry()
    except BaseException as exc:
        if checkpoint_error is not None:
            raise exc from checkpoint_error
        raise
    if checkpoint_error is not None:
        raise checkpoint_error
    return report


def _candidate_from_spatial_execution(execution: Any) -> Any | None:
    """Return the applied candidate while enforcing fail-closed admission."""

    decision = execution.decision
    if not decision.accepted_for_execution:
        if execution.applied:
            raise RuntimeError("rejected spatial admission applied its candidate")
        return None
    if not execution.applied:
        raise RuntimeError("accepted spatial admission did not apply its candidate")
    if decision.candidate is None:
        raise RuntimeError("accepted spatial admission lacks a candidate")
    return decision.candidate


def _ordered_floor_candidates(bindings: Any, region_center_xy: Any):
    center = np.asarray(region_center_xy, dtype=np.float64)
    return tuple(
        binding.candidate
        for binding in sorted(
            bindings,
            key=lambda item: (
                item.orientation_quarter_turn != -1,
                float(
                    np.linalg.norm(
                        (item.future_aabb_min_world[:2] + item.future_aabb_max_world[:2])
                        / 2.0
                        - center
                    )
                ),
                not item.broad_footprint_free,
                item.candidate.name,
            ),
        )
    )


def _relation_target_role(operation: str) -> str:
    return "declared_support" if operation == "place_on_floor" else "relation_target"


def _placement_relation_target(operation: str, target: Any, support: Any):
    return support if operation == "place_on_floor" else target


def _candidate_geometry_for_aabb(
    held: Any,
    candidate: SpatialPoseCandidate,
    candidate_aabb_world: tuple[Any, Any] | None,
):
    if candidate_aabb_world is None:
        return held
    return SimpleNamespace(
        aabb=candidate_aabb_world,
        get_position_orientation=lambda: (
            candidate.position,
            candidate.orientation_xyzw,
        ),
    )


def _keep_logical_inventory_closure_still(logical_snapshot: Any | None) -> int:
    """Drop carry-only velocity before a nested closure starts settling."""

    if logical_snapshot is None:
        return 0
    stopped = 0
    for obj in logical_snapshot.object_refs:
        keep_still = getattr(obj, "keep_still", None)
        if not callable(keep_still):
            raise RuntimeError(
                "logical inventory object lacks the official keep-still API"
            )
        keep_still()
        stopped += 1
    return stopped


def _logical_inventory_descendants_stable(
    logical_snapshot: Any | None,
    *,
    vector_reader: Callable[..., np.ndarray],
    velocity_diagnostic_only: bool = False,
) -> tuple[bool, tuple[dict[str, Any], ...]]:
    """Require every nested object to be slow, not merely still Inside."""

    reports = []
    for entry in () if logical_snapshot is None else logical_snapshot.entries:
        obj = entry.object_ref
        linear = vector_reader(
            obj.get_linear_velocity(), length=3, name="nested linear velocity"
        )
        angular = vector_reader(
            obj.get_angular_velocity(), length=3, name="nested angular velocity"
        )
        linear_speed = float(np.linalg.norm(linear))
        angular_speed = float(np.linalg.norm(angular))
        stable = bool(
            math.isfinite(linear_speed)
            and math.isfinite(angular_speed)
            and (velocity_diagnostic_only or (
                linear_speed <= DEFAULT_MAX_LINEAR_SPEED_M_S
                and angular_speed <= DEFAULT_MAX_ANGULAR_SPEED_RAD_S
            ))
        )
        reports.append(
            {
                "object_private": str(getattr(obj, "name", "")) or None,
                "linear_speed_m_s": linear_speed,
                "angular_speed_rad_s": angular_speed,
                "stable": stable,
                "velocity_diagnostic_only": velocity_diagnostic_only,
            }
        )
    return all(report["stable"] for report in reports), tuple(reports)


def _fresh_official_inside_containment(
    logical_snapshot: Any | None,
    inside_state_type: Any,
) -> tuple[bool, int]:
    """Read captured ``Inside`` edges after bounded official cache clears."""

    if logical_snapshot is None:
        return True, 0
    seen: set[int] = set()
    for obj in logical_snapshot.object_refs:
        identity = id(obj)
        if identity in seen:
            continue
        seen.add(identity)
        clearer = getattr(obj, "clear_states_cache", None)
        if not callable(clearer):
            raise RuntimeError(
                "logical inventory object lacks the official state cache "
                "invalidation API"
            )
        clearer()
    return (
        all(
            bool(child.states[inside_state_type].get_value(parent))
            for child, parent in logical_snapshot.containment_edges
        ),
        len(seen),
    )


def _ray_hits_object(result: Mapping[str, Any], obj: Any) -> bool:
    if not result or not bool(result.get("hit", False)):
        return False
    body = str(result.get("rigidBody", ""))
    return any(
        body == path or body.startswith(f"{path}/")
        for path in spatial_object_link_paths(obj)
    )


def plan_live_under_supports(
    *,
    held: Any,
    target: Any,
    supports: Sequence[Any],
    robot: Any,
    logical_snapshot: Any | None = None,
    raytest_batch: Callable[..., Sequence[Mapping[str, Any]]] | None = None,
    tensor_module: Any | None = None,
    symbolic_support: bool = False,
):
    """Return the same ray-certified Under plans used by markers and execution."""

    if held is None or target is None or target is held:
        return ()
    if raytest_batch is None:
        from omnigibson.utils.sampling_utils import raytest_batch as live_raytest_batch

        raytest_batch = live_raytest_batch
    closure_objects = tuple(
        getattr(logical_snapshot, "object_refs", (held,))
        if logical_snapshot is not None
        else (held,)
    )
    ignored = tuple(
        dict.fromkeys(
            path
            for item in (*closure_objects, robot)
            if item is not None
            for path in spatial_object_link_paths(item)
        )
    )

    def points(values):
        if tensor_module is None:
            return values
        return tensor_module.as_tensor(values, dtype=tensor_module.float32)

    def under_probe(candidate, probe_held, probe_target, probe_support):
        candidate_min, candidate_max = candidate_aabb(probe_held, candidate)
        _, target_max = object_aabb(probe_target)
        support_min, support_max = object_aabb(probe_support)
        center = (candidate_min[:2] + candidate_max[:2]) / 2.0
        lo = candidate_min[:2] + 0.2 * (candidate_max[:2] - candidate_min[:2])
        hi = candidate_max[:2] - 0.2 * (candidate_max[:2] - candidate_min[:2])
        footprint = (
            center,
            np.asarray((lo[0], lo[1])),
            np.asarray((hi[0], lo[1])),
            np.asarray((lo[0], hi[1])),
            np.asarray((hi[0], hi[1])),
        )
        up_z = float(support_max[2]) + 0.001
        up_end = max(
            float(target_max[2]) + 0.05,
            float(candidate_max[2]) + 0.10,
        )
        down_z = float(candidate_min[2]) + 0.01
        down_end = min(
            float(support_min[2]) - 0.05,
            float(candidate_min[2]) - 0.10,
        )
        kwargs = {
            "only_closest": True,
            "ignore_bodies": ignored,
            "ignore_collisions": ignored,
        }
        upward = raytest_batch(
            points([(float(x), float(y), up_z) for x, y in footprint]),
            points([(float(x), float(y), up_end) for x, y in footprint]),
            **kwargs,
        )
        downward = raytest_batch(
            points([(float(x), float(y), down_z) for x, y in footprint]),
            points([(float(x), float(y), down_end) for x, y in footprint]),
            **kwargs,
        )
        target_above = len(upward) == len(footprint) and all(
            _ray_hits_object(item, probe_target) for item in upward
        )
        support_hits = downward[:1] if symbolic_support else downward
        support_below = len(downward) == len(footprint) and all(
            _ray_hits_object(item, probe_support) for item in support_hits
        )
        clearance = None
        if target_above and support_below:
            above_z = np.asarray([float(item["position"][2]) for item in upward])
            below_z = np.asarray([float(item["position"][2]) for item in support_hits])
            if not (np.isfinite(above_z).all() and np.isfinite(below_z).all()):
                raise ValueError("Under ray hit heights must be finite")
            # The held box must fit between the lowest underside and highest
            # support hit across its footprint, not between whole-object tops.
            clearance = float(np.min(above_z) - np.max(below_z))
        return VerticalPlacementEvidence(
            target_first_above=target_above,
            support_first_below=support_below,
            clear_underside=target_above and support_below,
            vertical_clearance_m=clearance,
            error=(
                "incomplete_ray_batch"
                if len(upward) != len(footprint) or len(downward) != len(footprint)
                else None
            ),
        )

    return plan_default_under_supports(
        held,
        target,
        supports,
        vertical_probe=under_probe,
        scene_objects=(),
        allow_support_overhang=symbolic_support,
    )


def marker_symbolic_spatial_generator(
    owner: Any,
    target: Any,
    operation: str,
    *,
    selected_selector: str | None = None,
    placement_region_center_xy: Any | None = None,
    placement_region_extent_xy: Any | None = None,
    placement_surface_z_m: float | None = None,
    future_reference_feasibility_query: Callable[
        [Any, SpatialPoseCandidate], Mapping[str, Any]
    ]
    | None = None,
    action20_enabled: bool,
    action_primitive_error: Any,
    capture_carry_integrity_token: Callable[..., Mapping[str, Any]],
    capture_carry_projection_reference_factory: Callable[
        ..., Mapping[str, Any]
    ],
    cpu_vector: Callable[..., np.ndarray],
    evaluate_future_reference_candidate_gate: Callable[
        ..., Mapping[str, Any]
    ],
    object_state_types: Any,
    omnigibson: Any,
    record_rejected_candidate_reprojection_telemetry: Callable[..., None],
    reproject_rejected_candidate_from_carry_ssot: Callable[
        ..., Mapping[str, Any]
    ],
    reprojection_error_type: type[BaseException],
    reprojection_version: str,
    tensor_module: Any,
    transform_utils: Any,
):
    """Execute one marker-bound P5 spatial placement.

    Candidate generation and canonical readback come from the
    validated Action23 spatial module.  This adapter only owns the
    mainline carry transaction.  A failed placement is recoverable:
    Symbolic restores the action-start world and logical carry state;
    Physical retains its existing release / recovery policy.
    """

    from omnigibson.utils.sampling_utils import raytest_batch

    if operation == 'place_on_top' and owner._placement_physical_consequences_enabled:
        if (placement_region_center_xy is None) != (placement_region_extent_xy is None):
            raise ValueError('Physical placement region is incomplete')
        if selected_selector is not None and placement_region_center_xy is None:
            raise ValueError('Physical selected region is missing geometry')
        from .physical_ontop_drop import execute_drop
        yield from execute_drop(owner, target, center_xy=placement_region_center_xy,
                                extent_xy=placement_region_extent_xy,
                                selector=selected_selector, published_surface_z_m=placement_surface_z_m,
                                og=omnigibson, tensor=tensor_module, transforms=transform_utils)
        return

    relation = SPATIAL_ACTION_TO_PREDICATE.get(operation)
    if relation is None:
        raise action_primitive_error(
            action_primitive_error.Reason.PRE_CONDITION_ERROR,
            "Unsupported Action23 spatial operation",
            {"operation": operation},
        )
    if selected_selector is not None and operation not in {
        "place_on_top",
        "place_next_to",
        "place_touching",
    }:
        raise action_primitive_error(
            action_primitive_error.Reason.PRE_CONDITION_ERROR,
            "This spatial operation does not accept a visual selector",
            {"operation": operation},
        )
    if (placement_region_center_xy is None) != (
        placement_region_extent_xy is None
    ):
        raise action_primitive_error(
            action_primitive_error.Reason.PRE_CONDITION_ERROR,
            "Spatial placement region is incomplete",
            {"operation": operation},
        )
    if (
        placement_region_center_xy is not None
        and operation not in {
            "place_on_floor",
            "place_on_top",
            "place_next_to",
        }
    ):
        raise action_primitive_error(
            action_primitive_error.Reason.PRE_CONDITION_ERROR,
            "This spatial operation does not accept a placement region",
            {"operation": operation},
        )
    if placement_surface_z_m is not None and (
        operation != "place_on_top" or placement_region_center_xy is None
    ):
        raise action_primitive_error(
            action_primitive_error.Reason.PRE_CONDITION_ERROR,
            "Only a selected OnTop surface may supply a surface height",
            {"operation": operation},
        )
    symbolic_direct_readback = bool(
        placement_predicate_mode(
            symbolic_zero_physics=action20_enabled,
            physical_consequences_enabled=owner._placement_physical_consequences_enabled,
        ) == "action20_symbolic"
        and operation
        in {
            "place_on_floor",
            "place_on_top",
            "place_next_to",
            "place_under",
        }
    )
    physical_on_top = (
        operation == "place_on_top" and owner._placement_physical_consequences_enabled
    )
    held = owner._get_obj_in_hand()
    if held is None:
        metadata = {"operation": operation}
        if symbolic_direct_readback:
            metadata[ACTION20_BLOCKED_CATEGORY_METADATA_KEY] = (
                REQUIRES_OBJECT_IN_LEFT_HAND
            )
        raise action_primitive_error(
            action_primitive_error.Reason.PRE_CONDITION_ERROR,
            "Spatial placement requires a held object",
            metadata,
        )
    if target is None or target is held:
        raise action_primitive_error(
            action_primitive_error.Reason.PRE_CONDITION_ERROR,
            "Spatial placement requires a distinct target",
        )
    logical_snapshot = owner._logical_inventory_snapshot
    if owner._logical_inventory_carry and logical_snapshot is None:
        raise RuntimeError("spatial placement lacks its carry snapshot")
    rollback_symbolically = (
        owner._capture_symbolic_placement_rollback(target, getattr(object_state_types, relation))
        if symbolic_direct_readback and owner._logical_inventory_carry else None
    )

    scene_objects = tuple(getattr(owner.env.scene, "objects", ()))
    closure_ids = {
        id(item)
        for item in (
            logical_snapshot.object_refs
            if logical_snapshot is not None
            else (held,)
        )
    }
    blockers = tuple(
        item for item in scene_objects if id(item) not in closure_ids
    )
    candidate_plan = None
    candidate_reports: list[dict[str, Any]] = []
    candidate_failure_categories: set[str | None] = set()
    applied_candidate_attempt_count = 0
    physics_candidate_attempt_count = 0
    selected_candidate = None
    selected_support = None
    selected_stability = None
    placement_committed = False
    physical_consequence_committed = False
    verified_terminal_success = False
    failure: BaseException | None = None
    carry_projection_reference = None
    rejected_candidate_reprojection_count = 0
    action_recovery_reprojection_private: dict[str, Any] = {}
    fresh_candidate_generation_count = 0
    future_reference_candidate_rejection_count = 0
    last_rejected_candidate_reprojection_valid = False
    candidate_recovery_attempted = False
    terminal_interrupted_candidate = None
    physical_release_started = False
    active_trial_checkpoint = None
    on_top_plan = None
    next_to_physical_bindings = None
    next_to_preferred_orientation = None
    if operation == "place_next_to" and placement_region_center_xy is not None:
        if selected_selector is None:
            raise action_primitive_error(
                action_primitive_error.Reason.PRE_CONDITION_ERROR,
                "Physical NextTo placement requires its visual selector",
            )
        if owner._logical_inventory_origin_pose is not None:
            next_to_preferred_orientation = cpu_vector(
                transform_utils.mat2pose(
                    owner._logical_inventory_origin_pose
                )[1],
                length=4,
                name="pre-pick object orientation",
            )
    owner._placing_with_predicate = True

    def physical_next_to_bindings():
        pose = planar_base_pose_from_robot(owner.robot)
        return tuple(
            item
            for item in plan_support_plane_lateral_bindings(
                held,
                target,
                observer_xy=(pose.x, pose.y),
                preferred_orientation_xyzw=(
                    next_to_preferred_orientation
                ),
            )
            if item.selector == selected_selector
        )

    def downward_support_evidence(
        candidate,
        candidate_aabb_world=None,
    ):
        """Measure one candidate's actual local support without labels.

        The geometry kernel owns all acceptance thresholds.  This
        live adapter only performs the five official raycasts and
        assigns identity-derived roles.  An unbound hit is treated as
        a structural obstacle and therefore fails closed.
        """

        candidate_geometry = _candidate_geometry_for_aabb(
            held,
            candidate,
            candidate_aabb_world,
        )
        requests = build_downward_support_ray_requests(
            candidate_geometry,
            candidate,
        )
        closure_objects = (
            logical_snapshot.object_refs
            if logical_snapshot is not None
            else (held,)
        )
        ignored = tuple(
            path
            for item in closure_objects
            for path in spatial_object_link_paths(item)
        )
        results = raytest_batch(
            tensor_module.as_tensor(
                [request.start_xyz for request in requests],
                dtype=tensor_module.float32,
            ),
            tensor_module.as_tensor(
                [request.end_xyz for request in requests],
                dtype=tensor_module.float32,
            ),
            only_closest=True,
            ignore_bodies=ignored,
            ignore_collisions=ignored,
        )

        def hit_identity(result: Mapping[str, Any]):
            if not result or not bool(result.get("hit", False)):
                return None, None, None
            if _ray_hits_object(result, target):
                return _relation_target_role(operation), str(
                    getattr(target, "name", "")
                ) or str(result.get("rigidBody", "")), target
            if _ray_hits_object(result, owner.robot):
                return "robot", str(
                    getattr(owner.robot, "name", "")
                ) or str(result.get("rigidBody", "")), owner.robot
            for item in closure_objects:
                if _ray_hits_object(result, item):
                    return "held_object", str(
                        getattr(item, "name", "")
                    ) or str(result.get("rigidBody", "")), item
            if (
                selected_support is not None
                and _ray_hits_object(result, selected_support)
            ):
                return "declared_support", str(
                    getattr(selected_support, "name", "")
                ) or str(result.get("rigidBody", "")), selected_support
            for item in scene_objects:
                if _ray_hits_object(result, item):
                    return "support_candidate", str(
                        getattr(item, "name", "")
                    ) or str(result.get("rigidBody", "")), item
            return "structural_obstacle", str(
                result.get("rigidBody", "")
            ) or None, None

        evidence = []
        support_refs: list[Any] = []
        for request, result in zip(requests, results, strict=True):
            role, entity_key, hit_ref = hit_identity(result)
            hit = bool(result and result.get("hit", False))
            if role in {"support_candidate", "declared_support"}:
                support_refs.append(hit_ref)
            evidence.append(
                DownwardSupportRayEvidence(
                    sample_name=request.sample_name,
                    start_xyz=request.start_xyz,
                    end_xyz=request.end_xyz,
                    hit_position_xyz=(
                        result.get("position") if hit else None
                    ),
                    hit_normal_xyz=(
                        result.get("normal") if hit else None
                    ),
                    hit_entity_key=entity_key,
                    entity_role=role,
                    error=None,
                )
            )
        report = rebase_candidate_to_local_support(
            candidate_geometry,
            candidate,
            evidence,
            allow_overhang=symbolic_direct_readback,
        )
        local_support = None
        if report.accepted:
            support_counts: dict[int, tuple[int, Any]] = {}
            for item in support_refs:
                if item is None:
                    continue
                count, _ = support_counts.get(id(item), (0, item))
                support_counts[id(item)] = (count + 1, item)
            if symbolic_direct_readback:
                # Rebasing uses the centre hit, not the majority of corners
                # that may now overhang a different, lower support.
                local_support = hit_identity(results[0])[2]
            elif support_counts:
                local_support = sorted(
                    support_counts.values(),
                    key=lambda entry: (
                        -entry[0],
                        str(getattr(entry[1], "name", "")),
                    ),
                )[0][1]
        return report, local_support

    def contact_targets(support):
        return (
            (target, support)
            if relation == "Touching"
            else (support,)
        )

    def collision_shape_admission(candidate, candidate_aabb_world=None):
        closure_objects = (
            logical_snapshot.object_refs
            if logical_snapshot is not None
            else (held,)
        )
        return query_spatial_candidate_collision_overlap(
            omnigibson=omnigibson,
            held=held,
            closure_objects=closure_objects,
            contact_objects=contact_targets(selected_support) if symbolic_direct_readback else (),
            geometry_certified_objects=() if symbolic_direct_readback else contact_targets(selected_support),
            candidate=candidate,
            candidate_aabb_world=candidate_aabb_world,
        )

    def apply_candidate(candidate) -> None:
        nonlocal candidate_recovery_attempted
        nonlocal last_rejected_candidate_reprojection_valid
        nonlocal physical_release_started
        # Once mutation starts, every exit must restore this trial or commit it.
        candidate_recovery_attempted = False
        last_rejected_candidate_reprojection_valid = False
        pose = transform_utils.pose2mat(
            (
                tensor_module.as_tensor(candidate.position, dtype=tensor_module.float32),
                tensor_module.as_tensor(candidate.orientation_xyzw, dtype=tensor_module.float32),
            )
        )
        physical_release_started = physical_on_top
        try:
            if owner._logical_inventory_carry:
                owner._apply_logical_inventory_root_pose(pose)
                _keep_logical_inventory_closure_still(
                    owner._logical_inventory_snapshot
                )
            else:
                held.set_position_orientation(*transform_utils.mat2pose(pose))
                keep_still = getattr(held, "keep_still", None)
                if callable(keep_still):
                    keep_still()
        finally:
            if physical_release_started:
                # The pose write may partially succeed before raising. Never let
                # later carry synchronization restore an applied Physical trial.
                owner._clear_logical_inventory_state()

    def project_from_frozen_carry_ssot():
        """Project the same held closure from the frozen EEF transform.

        This path deliberately bypasses the generic logical-inventory
        synchronization cleanup because a rejected-candidate recovery
        failure is a hard transaction failure.  It must never restore
        the pre-grasp world pose or clear held state as a side effect.
        """

        eef = transform_utils.pose2mat(
            (
                owner.robot.get_eef_position(owner.arm),
                owner.robot.get_eef_orientation(owner.arm),
            )
        )
        expected_root_pose = owner._fixed_kinematic_world_pose(eef)
        if owner._logical_inventory_carry:
            snapshot = owner._logical_inventory_snapshot
            if snapshot is None:
                raise reprojection_error_type(
                    "logical inventory snapshot vanished before reprojection"
                )
            updated = snapshot.apply_root_pose(
                expected_root_pose,
                set_pose=owner._logical_inventory_set_pose,
                copy_pose=owner._logical_inventory_copy_pose,
                compose_pose=owner._logical_inventory_compose_pose,
                valid_pose=owner._logical_inventory_valid_pose,
            )
            owner._logical_inventory_snapshot = updated
            owner._logical_inventory_pose = updated.root_private_pose
            owner._logical_inventory_last_descendant_count = (
                updated.descendant_count
            )
            owner._logical_inventory_last_sync_count = updated.sync_count
        else:
            held.set_position_orientation(
                *transform_utils.mat2pose(expected_root_pose)
            )
        keep_still = getattr(held, "keep_still", None)
        if callable(keep_still):
            keep_still()
        return expected_root_pose

    def capture_carry_projection_reference():
        return capture_carry_projection_reference_factory(
            capture_integrity=lambda: (
                capture_carry_integrity_token(owner)
            ),
            read_root_pose=lambda: (
                owner._logical_inventory_pose_for_object(held)
            ),
            read_live_aabb=lambda: object_aabb(held),
            read_live_collision_points=lambda: (
                held.collision_points_world
            ),
            # Establish a self-consistent Fabric root / collision
            # geometry snapshot before any candidate mutates the
            # object.  This is a render-only app update: it advances
            # neither physics nor the evaluator observation.
            refresh_readback=lambda: omnigibson.sim.render(),
        )

    def reproject_rejected_candidate():
        if carry_projection_reference is None:
            raise reprojection_error_type(
                "candidate recovery lacks its carry reference"
            )
        return reproject_rejected_candidate_from_carry_ssot(
            reference=carry_projection_reference,
            project_from_carry_ssot=project_from_frozen_carry_ssot,
            capture_integrity=lambda: (
                capture_carry_integrity_token(owner)
            ),
            read_root_pose=lambda: (
                owner._logical_inventory_pose_for_object(held)
            ),
            read_live_aabb=lambda: object_aabb(held),
            record_private_telemetry=lambda report: (
                record_rejected_candidate_reprojection_telemetry(
                    candidate_reports=candidate_reports,
                    action_recovery_report=(
                        action_recovery_reprojection_private
                    ),
                    report=report,
                )
            ),
            # Kinematic pose writers update the carry SSoT
            # immediately, while Fabric-backed collision transforms
            # can lag for more than one app update.  Perform at most
            # three non-physics renders, stopping on the first strict
            # root-local-geometry readback match.  No observation is
            # requested or yielded by this private transaction.
            refresh_readback=lambda: omnigibson.sim.render(),
            maximum_readback_refreshes=3,
        )

    def rollback_candidate_trial():
        nonlocal active_trial_checkpoint
        nonlocal candidate_recovery_attempted
        nonlocal last_rejected_candidate_reprojection_valid
        nonlocal rejected_candidate_reprojection_count

        if candidate_recovery_attempted:
            raise RuntimeError("spatial candidate recovery was already attempted")
        candidate_recovery_attempted = True
        checkpoint = active_trial_checkpoint
        # A failed recovery is terminal.  Clearing first prevents an outer
        # abort handler from replaying the same simulator rollback.
        active_trial_checkpoint = None
        report = _rollback_spatial_candidate_trial(
            omnigibson=omnigibson,
            checkpoint=checkpoint,
            reproject_carry=reproject_rejected_candidate,
        )
        rejected_candidate_reprojection_count += 1
        last_rejected_candidate_reprojection_valid = True
        return report

    def candidate_is_accepted(
        candidate,
        support,
        *,
        support_rebase_report=None,
        collision_shape_report=None,
        admission_decision=None,
        broad_blocked_by=(),
        on_top_binding=None,
    ):
        nonlocal selected_stability
        sample_count = 1 if symbolic_direct_readback else DEFAULT_STABILITY_WINDOW
        settle_started = perf_counter()
        if not symbolic_direct_readback:
            # Stability samples are real physics steps.  Use the final samples
            # of the existing settle budget instead of adding another budget.
            for _ in range(
                max(0, DEFAULT_SETTLE_STEPS - DEFAULT_STABILITY_WINDOW)
            ):
                yield None
        settle_only_seconds = perf_counter() - settle_started
        samples = []
        errors = []
        relation_window: list[bool] = []
        nested_motion_window: list[bool] = []
        nested_motion_reports: list[tuple[dict[str, Any], ...]] = []
        relation_state_cache_clear_call_count = 0
        readback_refresh = None
        final_collision_report = None
        expected_room = None
        room_membership_window: list[bool | None] = []
        if relation in {"NextTo", "Touching"}:
            seg_map = getattr(owner.env.scene, "_seg_map", None)
            room_lookup = getattr(
                seg_map, "get_room_instance_by_point", None
            )
            if callable(room_lookup):
                try:
                    raw_room = room_lookup(
                        tuple(float(value) for value in candidate.position[:2])
                    )
                    if isinstance(raw_room, str) and raw_room:
                        expected_room = raw_room
                except Exception:
                    expected_room = None
        stability_started = perf_counter()
        if symbolic_direct_readback:
            protected = [held, target, support]
            if logical_snapshot is not None:
                protected.extend(logical_snapshot.object_refs)
            readback_refresh = _refresh_inside_readback_without_physics(
                owner,
                *protected,
            )
            final_collision_report = query_symbolic_placement_final_collision(
                omnigibson=omnigibson,
                moving_objects=(
                    logical_snapshot.object_refs
                    if logical_snapshot is not None
                    else (held,)
                ),
                contact_objects=contact_targets(
                    support
                ),
            )
        for _ in range(sample_count):
            if not symbolic_direct_readback:
                yield None
            try:
                if physical_on_top:
                    for obj in {id(obj): obj for obj in (held, target, support)}.values():
                        obj.clear_states_cache()
                relations_preserved_sample = True
                if logical_snapshot is not None:
                    (
                        relations_preserved_sample,
                        state_cache_clear_call_count,
                    ) = owner._logical_inventory_relations_preserved(
                        logical_snapshot
                    )
                    relation_state_cache_clear_call_count += (
                        state_cache_clear_call_count
                    )
                    if not symbolic_direct_readback:
                        (
                            nested_motion_stable,
                            nested_motion_report,
                        ) = _logical_inventory_descendants_stable(
                            logical_snapshot,
                            vector_reader=cpu_vector,
                            velocity_diagnostic_only=physical_on_top,
                        )
                sample = sample_spatial_stability(
                    held,
                    _placement_relation_target(
                        operation,
                        target,
                        support,
                    ),
                    relation=relation,
                    support=support,
                    symbolic_on_top=symbolic_direct_readback,
                    support_surface_z_m=(
                        placement_surface_z_m
                        if operation == "place_on_top"
                        else None
                    ),
                )
                samples.append(sample)
                if logical_snapshot is not None:
                    relation_window.append(relations_preserved_sample)
                    if not symbolic_direct_readback:
                        nested_motion_window.append(nested_motion_stable)
                        nested_motion_reports.append(nested_motion_report)
                if relation in {"NextTo", "Touching"}:
                    measured_room = None
                    if callable(room_lookup):
                        try:
                            raw_room = room_lookup(sample.position[:2])
                            if isinstance(raw_room, str) and raw_room:
                                measured_room = raw_room
                        except Exception:
                            measured_room = None
                    room_membership_window.append(
                        None
                        if expected_room is None or measured_room is None
                        else measured_room == expected_room
                    )
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        stability_seconds = perf_counter() - stability_started
        if symbolic_direct_readback:
            post_support_report = None
            post_support_error = None
            report = summarize_relation_authoritative_postcondition(
                samples,
                required_samples=1,
                require_geometric_support=True,
            )
            report.update(
                {
                    "verification_policy": (
                        "behavior2026.symbolic-spatial-single-readback/v1"
                    ),
                    "settle_steps_required": 0,
                    "physics_stability_window_required": False,
                }
            )
        elif relation in {"NextTo", "Touching"}:
            post_support_report = None
            post_support_error = None
            if samples:
                final_sample = samples[-1]
                final_candidate = SpatialPoseCandidate(
                    name=f"{candidate.name}:post_settle_support_probe",
                    position=np.asarray(
                        final_sample.position, dtype=np.float64
                    ),
                    orientation_xyzw=np.asarray(
                        final_sample.orientation_xyzw, dtype=np.float64
                    ),
                )
                try:
                    post_support_report, _ = downward_support_evidence(
                        final_candidate
                    )
                except Exception as exc:
                    post_support_error = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    errors.append(post_support_error)
            report = summarize_supported_lateral_postcondition(
                samples,
                expected_support_bottom_z_m=(
                    None
                    if support_rebase_report is None
                    else support_rebase_report.rebased_bottom_z_m
                ),
                expected_candidate_xy=candidate.position[:2],
                room_membership_window=room_membership_window,
                post_settle_support_valid=(
                    None
                    if post_support_report is None
                    else bool(post_support_report.accepted)
                ),
            )
        elif relation == "OnTop":
            post_support_report = None
            post_support_error = None
            report = summarize_original_destination_on_top(
                samples, velocity_diagnostic_only=physical_on_top,
            )
            if operation == "place_on_floor":
                report.update(
                    {
                        "verification_policy": (
                            "behavior2026.local-ground-support-on-top/v1"
                        ),
                        "official_on_top_original_destination_required": False,
                        "official_touching_original_destination_required": False,
                        "official_on_top_actual_support_required": True,
                        "official_touching_actual_support_required": True,
                    }
                )
        else:
            post_support_report = None
            post_support_error = None
            report = (
                summarize_symbolic_spatial_postcondition(samples)
                if action20_enabled
                else summarize_spatial_stability(samples)
            )
        try:
            orientation_errors = [
                quaternion_angle_rad(
                    sample.orientation_xyzw,
                    candidate.orientation_xyzw,
                )
                for sample in samples
            ]
        except Exception as exc:
            orientation_errors = []
            errors.append(
                f"candidate orientation verification failed: {type(exc).__name__}: {exc}"
            )
        orientation_preserved = bool(
            len(orientation_errors) == sample_count
            and all(
                error <= DEFAULT_PROTECTED_ROTATION_RAD
                for error in orientation_errors
            )
        )
        report.update(
            {
                "candidate_orientation_error_rad_private": orientation_errors,
                "candidate_orientation_preserved": orientation_preserved,
                "candidate_orientation_is_rejection_gate": (
                    not symbolic_direct_readback
                ),
                "maximum_candidate_orientation_error_rad": (
                    DEFAULT_PROTECTED_ROTATION_RAD
                ),
            }
        )
        relations_preserved = bool(
            logical_snapshot is None
            or (
                len(relation_window) == sample_count
                and all(relation_window)
            )
        )
        nested_motion_stable = bool(
            symbolic_direct_readback
            or logical_snapshot is None
            or (
                len(nested_motion_window) == DEFAULT_STABILITY_WINDOW
                and all(nested_motion_window)
            )
        )
        if (
            final_collision_report is not None
            and not final_collision_report.accepted
        ):
            failure_reasons = list(report.get("failure_reasons", ()))
            failure_reasons.append("final_state_solid_overlap")
            report["failure_reasons"] = failure_reasons
        accepted = bool(
            report.get("accepted")
            and (
                final_collision_report is None
                or final_collision_report.accepted
            )
            and (symbolic_direct_readback or orientation_preserved)
            and relations_preserved
            and nested_motion_stable
            and not errors
        )
        if symbolic_direct_readback and not accepted:
            reasons = set(report.get("failure_reasons", ()))
            if (
                errors
                or not relations_preserved
                or not nested_motion_stable
                or reasons
                & {
                    "incomplete_stability_window",
                    "non_finite_stability_sample",
                }
            ):
                candidate_failure_categories.add(EXECUTION_UNVERIFIED)
            elif (
                final_collision_report is not None
                and not final_collision_report.accepted
            ):
                candidate_failure_categories.add(
                    _spatial_collision_failure_category(
                        final_collision_report
                    )
                )
            elif reasons == {"canonical_relation_not_stable"}:
                candidate_failure_categories.add(RELATION_NOT_REALIZED)
            elif "canonical_support_not_stable" in reasons:
                candidate_failure_categories.add(PLACEMENT_SUPPORT_UNAVAILABLE)
            else:
                candidate_failure_categories.add(None)
        physical_consequence_commit_safe = bool(
            len(samples) == DEFAULT_STABILITY_WINDOW
            and not errors
            and all(
                np.all(np.isfinite(sample.position))
                and np.all(np.isfinite(sample.orientation_xyzw))
                and np.isfinite(sample.bottom_to_support_gap_m)
                and np.isfinite(sample.linear_speed_m_s)
                and np.isfinite(sample.angular_speed_rad_s)
                and np.isfinite(sample.tilt_rad)
                and (
                    sample.bottom_z_m is None
                    or np.isfinite(sample.bottom_z_m)
                )
                for sample in samples
            )
        )
        candidate_reports.append(
            {
                "candidate": candidate.name,
                "on_top_candidate_binding_private": (
                    None
                    if on_top_binding is None
                    else on_top_binding.to_private_dict()
                ),
                "local_support_rebase_private": (
                    None
                    if support_rebase_report is None
                    else support_rebase_report.to_private_dict()
                ),
                "post_settle_local_support_private": (
                    None
                    if post_support_report is None
                    else post_support_report.to_private_dict()
                ),
                "post_settle_local_support_error_private": (
                    post_support_error
                ),
                "candidate_room_private": expected_room,
                "broad_blockers_private": list(broad_blocked_by),
                "collision_shape_admission_private": (
                    None
                    if collision_shape_report is None
                    else collision_shape_report.to_private_dict()
                ),
                "candidate_admission_policy_private": (
                    None
                    if admission_decision is None
                    else admission_decision.to_private_dict()
                ),
                "final_state_collision_admission_private": (
                    None
                    if final_collision_report is None
                    else final_collision_report.to_private_dict()
                ),
                **report,
                "logical_inventory_relation_window_private": (
                    relation_window
                ),
                "logical_inventory_relations_preserved": (
                    relations_preserved
                ),
                # Backward-compatible aliases retained for existing audit readers.
                "logical_inventory_containment_window_private": relation_window,
                "logical_inventory_containment_preserved": relations_preserved,
                "logical_inventory_nested_motion_window_private": (
                    nested_motion_window
                ),
                "logical_inventory_nested_motion_samples_private": (
                    nested_motion_reports
                ),
                "logical_inventory_nested_motion_stable": (
                    nested_motion_stable
                ),
                "logical_inventory_state_cache_clear_call_count_private": (
                    relation_state_cache_clear_call_count
                ),
                "readback_errors_private": errors,
                "zero_physics_readback_refresh_private": (
                    None
                    if readback_refresh is None
                    else dict(readback_refresh)
                ),
                "physical_consequence_commit_safe_private": (
                    physical_consequence_commit_safe
                ),
                "phase_timing_private": {
                    "physics_settle_seconds": settle_only_seconds,
                    "stability_verification_seconds": stability_seconds,
                    "settle_steps_total": (
                        0 if symbolic_direct_readback else DEFAULT_SETTLE_STEPS
                    ),
                    "stability_sample_steps": (
                        0
                        if symbolic_direct_readback
                        else DEFAULT_STABILITY_WINDOW
                    ),
                    "readback_samples": len(samples),
                },
                "accepted": accepted,
            }
        )
        if accepted:
            selected_stability = report
        return accepted

    try:
        yield from owner._detach_held_object_at_current_pose()
        owner._robot_object_collision_guard.restore(
            held, reason="before_action23_spatial_placement"
        )
        # Establish one fresh EEF projection before candidate geometry
        # is generated.  All later rejected candidates return to this
        # same frozen-transform reference, never to their failed pose
        # or the object's pre-grasp world pose.
        try:
            project_from_frozen_carry_ssot()
        except reprojection_error_type:
            raise
        except BaseException as exc:
            raise reprojection_error_type(
                "initial carry SSoT projection failed"
            ) from exc
        carry_projection_reference = (
            capture_carry_projection_reference()
        )
        if relation == "OnTop":
            selected_support = target
            # Whole-object scene AABBs are intentionally omitted from
            # candidate culling.  They remain private diagnostics in
            # the per-candidate report; the live PhysX collision-shape
            # query below is the execution admission authority.
            on_top_plan = plan_on_top_support_layer_candidates(
                held,
                target,
                broad_diagnostic_objects=blockers,
                preferred_orientation_xyzw=(
                    None
                    if owner._logical_inventory_origin_pose is None
                    else cpu_vector(
                        transform_utils.mat2pose(owner._logical_inventory_origin_pose)[1],
                        length=4,
                        name="pre-pick object orientation",
                    )
                ),
                region_center_xy=placement_region_center_xy,
                region_extent_xy=placement_region_extent_xy,
                surface_z_m=placement_surface_z_m,
                allow_overhang=(
                    symbolic_direct_readback
                    or operation != "place_on_top"
                    or owner._placement_physical_consequences_enabled
                ),
                include_support_center=(
                    symbolic_direct_readback and operation == "place_on_top"
                ),
            )
            if operation == "place_on_top" and action20_enabled:
                base_pose = planar_base_pose_from_robot(owner.robot)
                on_top_plan = restrict_and_rank_on_top_plan(
                    on_top_plan,
                    (base_pose.x, base_pose.y),
                    ACTION20_LOCAL_WORKSPACE_M,
                )
            candidate_plan = on_top_plan.candidate_plan
            candidates = candidate_plan.candidates
            if (
                operation == "place_on_floor"
                and placement_region_center_xy is not None
            ):
                candidates = _ordered_floor_candidates(
                    on_top_plan.bindings,
                    placement_region_center_xy,
                )
        elif relation == "Under":
            plans = plan_live_under_supports(
                held=held,
                target=target,
                supports=scene_objects,
                robot=owner.robot,
                logical_snapshot=logical_snapshot,
                raytest_batch=raytest_batch,
                tensor_module=tensor_module,
                symbolic_support=symbolic_direct_readback,
            )
            usable = tuple(plan for plan in plans if plan.candidate_plan.candidates)
            if not usable:
                metadata = {
                    "support plan count": len(plans),
                    "support plans private": [
                        plan.to_private_dict() for plan in plans
                    ],
                }
                if symbolic_direct_readback:
                    category = _empty_under_plans_category(plans)
                    if category is not None:
                        metadata[ACTION20_BLOCKED_CATEGORY_METADATA_KEY] = category
                raise action_primitive_error(
                    action_primitive_error.Reason.SAMPLING_ERROR,
                    "Under placement has no ray-verified candidate",
                    metadata,
                )
            selected_support = usable[0].support
            candidate_plan = usable[0].candidate_plan
            candidates = candidate_plan.candidates
        else:
            # Lateral relations do not require the reference object
            # to have one globally unique OnTop support.  Generate XY
            # poses first; each candidate discovers and verifies its
            # own local support with five downward rays.
            selected_support = None
            if operation == "place_next_to" and placement_region_center_xy is not None:
                next_to_physical_bindings = physical_next_to_bindings()
                if not next_to_physical_bindings:
                    metadata = {}
                    if symbolic_direct_readback:
                        metadata[
                            ACTION20_BLOCKED_CATEGORY_METADATA_KEY
                        ] = PLACEMENT_SUPPORT_UNAVAILABLE
                    raise action_primitive_error(
                        action_primitive_error.Reason.SAMPLING_ERROR,
                        "Selected NextTo support point is unavailable",
                        metadata,
                    )
                candidates = tuple(
                    item.candidate for item in next_to_physical_bindings
                )
            else:
                candidates = expanded_support_agnostic_lateral_candidates(
                    held,
                    target,
                    relation=relation,
                )
                if selected_selector is not None:
                    pose = planar_base_pose_from_robot(owner.robot)
                    selector_fn = (
                        select_next_to_screen_candidate
                        if relation == "NextTo"
                        else select_touching_screen_candidate
                    )
                    candidates = selector_fn(
                        candidates,
                        held=held,
                        target=target,
                        observer_xy=(pose.x, pose.y),
                        selector=selected_selector,
                    )
        candidate_count = len(candidates)
        if symbolic_direct_readback and candidate_count == 0:
            candidate_failure_categories.add(
                _empty_on_top_plan_category(candidate_plan)
                if relation == "OnTop"
                else PLACEMENT_SUPPORT_UNAVAILABLE
            )
        for candidate_index in range(candidate_count):
            active_lateral_binding = None
            if relation in {"NextTo", "Touching"}:
                # Generate this candidate only after the prior applied
                # candidate (if any) has completed its carry
                # reprojection transaction and fresh-AABB readback.
                if next_to_physical_bindings is not None:
                    live_bindings = physical_next_to_bindings()
                    live_candidates = tuple(
                        item.candidate for item in live_bindings
                    )
                else:
                    live_bindings = ()
                    live_candidates = expanded_support_agnostic_lateral_candidates(
                        held,
                        target,
                        relation=relation,
                    )
                if selected_selector is not None and next_to_physical_bindings is None:
                    pose = planar_base_pose_from_robot(owner.robot)
                    selector_fn = (
                        select_next_to_screen_candidate
                        if relation == "NextTo"
                        else select_touching_screen_candidate
                    )
                    live_candidates = selector_fn(
                        live_candidates,
                        held=held,
                        target=target,
                        observer_xy=(pose.x, pose.y),
                        selector=selected_selector,
                    )
                if (
                    len(live_candidates) != candidate_count
                    or tuple(item.name for item in live_candidates)
                    != tuple(item.name for item in candidates)
                ):
                    raise reprojection_error_type(
                        "fresh carry candidate keys changed after reprojection"
                    )
                candidate = live_candidates[candidate_index]
                if live_bindings:
                    active_lateral_binding = live_bindings[candidate_index]
                fresh_candidate_generation_count += 1
            else:
                candidate = candidates[candidate_index]
            support_rebase_report = None
            collision_admission = None
            admission_decision = None
            candidate_execution = None
            broad_blocked_by = ()
            on_top_binding = None
            if active_trial_checkpoint is not None:
                raise RuntimeError("prior OnTop trial checkpoint is still active")
            if relation == "OnTop":
                if on_top_plan is None:
                    raise RuntimeError("OnTop candidate plan is unavailable")
                on_top_binding = on_top_plan.binding_for(
                    candidate.name
                )
                collision_bounds = (
                    on_top_binding.future_aabb_min_world,
                    on_top_binding.future_aabb_max_world,
                )
                local_support_required = False
                blocker_candidate = candidate
                if operation == "place_on_floor":
                    selected_support = target
                    (
                        support_rebase_report,
                        candidate_local_support,
                    ) = downward_support_evidence(
                        candidate,
                        (
                            on_top_binding.future_aabb_min_world,
                            on_top_binding.future_aabb_max_world,
                        ),
                    )
                    if not support_rebase_report.accepted:
                        if symbolic_direct_readback:
                            candidate_failure_categories.add(
                                _support_rebase_failure_category(support_rebase_report)
                            )
                        candidate_reports.append(
                            {
                                "candidate": candidate.name,
                                "local_support_rebase_private": (
                                    support_rebase_report.to_private_dict()
                                ),
                                "accepted": False,
                            }
                        )
                        continue
                    if candidate_local_support is None:
                        if symbolic_direct_readback:
                            candidate_failure_categories.add(
                                PLACEMENT_SUPPORT_UNAVAILABLE
                            )
                        candidate_reports.append(
                            {
                                "candidate": candidate.name,
                                "local_support_rebase_private": (
                                    support_rebase_report.to_private_dict()
                                ),
                                "candidate_admission_policy_private": {
                                    "accepted_for_execution": False,
                                    "reason": (
                                        "local_support_identity_unavailable"
                                    ),
                                    "official_relation_declared": False,
                                },
                                "accepted": False,
                            }
                        )
                        continue
                    selected_support = candidate_local_support
                    blocker_candidate = (
                        support_rebase_report.rebased_candidate
                    )
                    if blocker_candidate is None:
                        raise RuntimeError(
                            "accepted local support rebase lacks a candidate"
                        )
                    delta = blocker_candidate.position - candidate.position
                    collision_bounds = (
                        collision_bounds[0] + delta,
                        collision_bounds[1] + delta,
                    )
                    local_support_required = True
                broad_blocked_by = candidate_blockers(
                    held=held,
                    target=target,
                    support=selected_support,
                    candidate=blocker_candidate,
                    scene_objects=blockers,
                )

                def apply_on_top_candidate(item):
                    nonlocal active_trial_checkpoint
                    # The narrow query runs before this callback.  Only
                    # an admitted pose pays for a simulator checkpoint.
                    active_trial_checkpoint = omnigibson.sim.dump_state(
                        serialized=False
                    )
                    apply_candidate(item)

                candidate_execution = execute_admitted_spatial_candidate(
                    candidate=candidate,
                    broad_blockers=broad_blocked_by,
                    collision_shape_query=(
                        lambda item, _bounds=collision_bounds: collision_shape_admission(
                            item,
                            _bounds,
                        )
                    ),
                    apply_candidate=apply_on_top_candidate,
                    local_support_report=support_rebase_report,
                    local_support_required=local_support_required,
                )
                collision_admission = (
                    candidate_execution.collision_shape_report
                )
                admission_decision = candidate_execution.decision
            elif relation == "Under":
                broad_blocked_by = candidate_blockers(
                    held=held,
                    target=target,
                    support=selected_support,
                    candidate=candidate,
                    scene_objects=blockers,
                )
                candidate_execution = execute_admitted_spatial_candidate(
                    candidate=candidate,
                    broad_blockers=broad_blocked_by,
                    collision_shape_query=collision_shape_admission,
                    apply_candidate=apply_candidate,
                )
                collision_admission = (
                    candidate_execution.collision_shape_report
                )
                admission_decision = candidate_execution.decision
            elif relation in {"NextTo", "Touching"}:
                lateral_candidate_bounds = (
                    None
                    if active_lateral_binding is None
                    else (
                        active_lateral_binding.future_aabb_min_world,
                        active_lateral_binding.future_aabb_max_world,
                    )
                )
                (
                    support_rebase_report,
                    candidate_local_support,
                ) = downward_support_evidence(
                    candidate,
                    lateral_candidate_bounds,
                )
                if not support_rebase_report.accepted:
                    if symbolic_direct_readback:
                        candidate_failure_categories.add(
                        _support_rebase_failure_category(support_rebase_report)
                        )
                    candidate_reports.append(
                        {
                            "candidate": candidate.name,
                            "local_support_rebase_private": (
                                support_rebase_report.to_private_dict()
                            ),
                            "accepted": False,
                        }
                    )
                    continue
                if candidate_local_support is None:
                    if symbolic_direct_readback:
                        candidate_failure_categories.add(
                            PLACEMENT_SUPPORT_UNAVAILABLE
                        )
                    candidate_reports.append(
                        {
                            "candidate": candidate.name,
                            "local_support_rebase_private": (
                                support_rebase_report.to_private_dict()
                            ),
                            "candidate_admission_policy_private": {
                                "accepted_for_execution": False,
                                "reason": (
                                    "local_support_identity_unavailable"
                                ),
                                "official_relation_declared": False,
                            },
                            "accepted": False,
                        }
                    )
                    continue
                selected_support = candidate_local_support
                rebased_candidate = (
                    support_rebase_report.rebased_candidate
                )
                if rebased_candidate is None:
                    raise RuntimeError(
                        "accepted local support rebase lacks a candidate"
                    )
                broad_blocked_by = candidate_blockers(
                    held=held,
                    target=target,
                    support=selected_support,
                    candidate=rebased_candidate,
                    scene_objects=blockers,
                )

                def lateral_collision_shape_admission(item):
                    if lateral_candidate_bounds is None:
                        return collision_shape_admission(item)
                    delta = item.position - candidate.position
                    return collision_shape_admission(
                        item,
                        (
                            lateral_candidate_bounds[0] + delta,
                            lateral_candidate_bounds[1] + delta,
                        ),
                    )

                candidate_execution = execute_admitted_spatial_candidate(
                    candidate=candidate,
                    broad_blockers=broad_blocked_by,
                    collision_shape_query=lateral_collision_shape_admission,
                    apply_candidate=apply_candidate,
                    local_support_report=support_rebase_report,
                    local_support_required=True,
                )
                collision_admission = (
                    candidate_execution.collision_shape_report
                )
                admission_decision = candidate_execution.decision
            if candidate_execution is None or admission_decision is None:
                raise RuntimeError("spatial relation has no candidate executor")
            admitted_candidate = _candidate_from_spatial_execution(
                candidate_execution
            )
            if admitted_candidate is None:
                if symbolic_direct_readback:
                    if admission_decision.reason in {
                        "local_support_evidence_required",
                        "local_support_rejected",
                    }:
                        category = PLACEMENT_SUPPORT_UNAVAILABLE
                    elif admission_decision.reason == (
                        "collision_shape_overlap_rejected"
                    ):
                        category = _spatial_collision_failure_category(
                            collision_admission
                        )
                    else:
                        category = None
                    candidate_failure_categories.add(category)
                candidate_reports.append(
                    {
                        "candidate": candidate.name,
                        "on_top_candidate_binding_private": (
                            None
                            if on_top_binding is None
                            else on_top_binding.to_private_dict()
                        ),
                        "local_support_rebase_private": (
                            None
                            if support_rebase_report is None
                            else support_rebase_report.to_private_dict()
                        ),
                        "broad_blockers_private": list(
                            broad_blocked_by
                        ),
                        "collision_shape_admission_private": (
                            None
                            if collision_admission is None
                            else collision_admission.to_private_dict()
                        ),
                        "candidate_admission_policy_private": (
                            admission_decision.to_private_dict()
                        ),
                        "accepted": False,
                    }
                )
                continue
            candidate = admitted_candidate
            terminal_interrupted_candidate = {
                "candidate": candidate.name,
                "status": "settle_in_progress",
                "candidate_applied": True,
                "local_support_rebase_private": (
                    None
                    if support_rebase_report is None
                    else support_rebase_report.to_private_dict()
                ),
                "broad_blockers_private": list(
                    broad_blocked_by
                ),
                "collision_shape_admission_private": (
                    None
                    if collision_admission is None
                    else collision_admission.to_private_dict()
                ),
                "candidate_admission_policy_private": (
                    None
                    if admission_decision is None
                    else admission_decision.to_private_dict()
                ),
                "accepted": None,
                "official_postcondition_readback_complete": False,
            }
            applied_candidate_attempt_count += 1
            if not symbolic_direct_readback:
                physics_candidate_attempt_count += 1
            spatial_postcondition_accepted = bool(
                (
                    yield from candidate_is_accepted(
                        candidate,
                        selected_support,
                        support_rebase_report=support_rebase_report,
                        collision_shape_report=collision_admission,
                        admission_decision=admission_decision,
                        broad_blocked_by=broad_blocked_by,
                        on_top_binding=on_top_binding,
                    )
                )
            )
            terminal_interrupted_candidate = None
            if (
                not spatial_postcondition_accepted
                and owner._placement_physical_consequences_enabled
                and candidate_reports[-1].get(
                    "physical_consequence_commit_safe_private"
                )
                is True
            ):
                selected_candidate = candidate.name
                physical_consequence_committed = True
                active_trial_checkpoint = None
                owner._clear_logical_inventory_state()
                break
            if spatial_postcondition_accepted:
                future_reference_report = None
                if future_reference_feasibility_query is not None:
                    future_reference_report = dict(
                        evaluate_future_reference_candidate_gate(
                            query=(
                                future_reference_feasibility_query
                            ),
                            held=held,
                            candidate=candidate,
                        )
                    )
                    gate_accepted = (
                        future_reference_report.get("accepted") is True
                    )
                    candidate_reports[-1][
                        "future_reference_interaction_feasibility_private"
                    ] = future_reference_report
                    if not gate_accepted:
                        if symbolic_direct_readback:
                            candidate_failure_categories.add(None)
                        candidate_reports[-1]["accepted"] = False
                        failure_reasons = list(
                            candidate_reports[-1].get(
                                "failure_reasons", ()
                            )
                        )
                        if (
                            "future_reference_interaction_unavailable"
                            not in failure_reasons
                        ):
                            failure_reasons.append(
                                "future_reference_interaction_unavailable"
                            )
                        candidate_reports[-1][
                            "failure_reasons"
                        ] = failure_reasons
                        selected_stability = None
                        future_reference_candidate_rejection_count += 1
                        spatial_postcondition_accepted = False
                if spatial_postcondition_accepted:
                    selected_candidate = candidate.name
                    placement_committed = True
                    active_trial_checkpoint = None
                    break
            if relation == "OnTop" and active_trial_checkpoint is None:
                raise RuntimeError("OnTop rejected trial lacks its checkpoint")
            if owner._placement_physical_consequences_enabled:
                # Physical actions never rewind an applied candidate.  Stop at
                # the observed failure and return its real carry/world state;
                # retrying another candidate would silently mutate a failed
                # episode into a transaction search.
                active_trial_checkpoint = None
                raise action_primitive_error(
                    action_primitive_error.Reason.SAMPLING_ERROR,
                    "Physical placement failed after execution began",
                    {"execution_started": True,
                     "rollback_performed": False},
                )
            try:
                reprojection_report = rollback_candidate_trial()
            except reprojection_error_type as exc:
                if candidate_reports:
                    private_report = getattr(
                        exc,
                        "private_report",
                        None,
                    )
                    if isinstance(private_report, Mapping):
                        candidate_reports[-1][
                            "rejected_candidate_carry_reprojection_private"
                        ] = dict(private_report)
                    else:
                        candidate_reports[-1][
                            "rejected_candidate_carry_reprojection_private"
                        ] = {
                            "version": reprojection_version,
                            "status": "FAIL",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                raise
            candidate_reports[-1][
                "rejected_candidate_carry_reprojection_private"
            ] = dict(reprojection_report)
            # Symbolic may try more poses in the same selected region, but
            # query faults are not evidence that another pose is appropriate.
            if symbolic_direct_readback:
                if (applied_candidate_attempt_count >= DEFAULT_SYMBOLIC_CANDIDATE_TRIAL_LIMIT
                    or EXECUTION_UNVERIFIED in candidate_failure_categories):
                    break
            elif operation != "place_next_to" or applied_candidate_attempt_count >= 2:
                break
        if not placement_committed and not physical_consequence_committed:
            metadata = {
                "operation": operation,
                "candidate_attempt_count_private": len(candidate_reports),
                "applied_candidate_attempt_count_private": (
                    applied_candidate_attempt_count
                ),
                "physics_candidate_attempt_count_private": (
                    physics_candidate_attempt_count
                ),
                "canonical_failure_reasons_private": sorted(
                    {
                        reason
                        for report in candidate_reports
                        for reason in report.get("failure_reasons", ())
                    }
                ),
            }
            category = placement_failure_category(candidate_failure_categories)
            if symbolic_direct_readback and category is not None:
                metadata[ACTION20_BLOCKED_CATEGORY_METADATA_KEY] = category
            raise action_primitive_error(
                action_primitive_error.Reason.SAMPLING_ERROR,
                (
                    candidate_plan.planning_error
                    if operation == "place_on_top"
                    and not candidates
                    and candidate_plan is not None
                    and candidate_plan.planning_error is not None
                    else "Action23 spatial candidates failed canonical readback"
                ),
                metadata,
            )
        if placement_committed:
            owner._clear_logical_inventory_state()
    except _PrivateVerifiedSpatialTerminalSuccess:
        verified_terminal_success = True
        # Only the private exact runner may preserve a partially observed
        # placement, after its live verifier has confirmed global success.
        if terminal_interrupted_candidate is not None:
            terminal_interrupted_candidate = {
                **dict(terminal_interrupted_candidate),
                "status": "official_global_terminal_during_settle",
                "generator_close_structured": True,
                "verified_terminal_signal": True,
            }
            candidate_reports.append(
                dict(terminal_interrupted_candidate)
            )
        raise
    except BaseException as exc:
        failure = exc
        if terminal_interrupted_candidate is not None:
            terminal_interrupted_candidate = {
                **dict(terminal_interrupted_candidate),
                "status": "interrupted_during_settle",
                "generator_close_structured": isinstance(exc, GeneratorExit),
                "accepted": False,
            }
            candidate_reports.append(dict(terminal_interrupted_candidate))
        # Restore the active OnTop trial, then reattach to the current EEF.
        # No pre-pick rollback is performed.
        if (
            not owner._placement_physical_consequences_enabled
            and not isinstance(exc, reprojection_error_type)
            and carry_projection_reference is not None
            and not last_rejected_candidate_reprojection_valid
            and not candidate_recovery_attempted
        ):
            try:
                rollback_candidate_trial()
            except BaseException as recovery_exc:
                failure = recovery_exc
                raise
        raise
    finally:
        cleanup_completed = False
        try:
            owner._placing_with_predicate = False
            if owner._robot_object_collision_guard.snapshot()["active"]:
                owner._robot_object_collision_guard.restore(
                    held, reason="action23_spatial_final_cleanup"
                )
            owner._last_private_exact_report = build_spatial_partial_report(
                operation=operation,
                candidate_plan=candidate_plan,
                candidate_attempts=candidate_reports,
                selected_candidate=selected_candidate,
                selected_support=selected_support,
                stability_report=selected_stability,
                error=(
                    failure
                    if not placement_committed
                    and not physical_consequence_committed
                    else None
                ),
            )
            owner._last_private_exact_report.update(
                {
                    "placement_committed": placement_committed,
                    "physical_consequence_committed": (
                        physical_consequence_committed
                    ),
                    "physical_release_started": physical_release_started,
                    "physical_release_state_private": (
                        {
                            "logical_object_held": owner._get_obj_in_hand() is not None,
                            "logical_snapshot_present": owner._logical_inventory_snapshot is not None,
                            "visual_projection_active": getattr(owner, "_logical_inventory_visual_projection_active", None),
                            "official_object_held": (
                                owner.robot._ag_obj_in_hand.get(owner.arm) is not None
                                if isinstance(getattr(owner.robot, "_ag_obj_in_hand", None), Mapping) else None
                            ),
                            "official_constraint_present": (
                                owner.robot._ag_obj_constraints.get(owner.arm) is not None
                                if isinstance(getattr(owner.robot, "_ag_obj_constraints", None), Mapping) else None
                            ),
                            "collision_guard": dict(owner._robot_object_collision_guard.snapshot()),
                        } if physical_release_started else None
                    ),
                    "recoverable_executor_noop": False,
                    "held_object_retained": bool(
                        failure is not None and owner._get_obj_in_hand() is held
                    ),
                    "rejected_candidate_reprojection_count": (
                        rejected_candidate_reprojection_count
                    ),
                    "action_recovery_carry_reprojection_private": (
                        None
                        if not action_recovery_reprojection_private
                        else dict(action_recovery_reprojection_private)
                    ),
                    "rejected_candidate_reprojection_version": (
                        reprojection_version
                    ),
                    "fresh_candidate_generation_count": (
                        fresh_candidate_generation_count
                    ),
                    "future_reference_gate_required": bool(
                        future_reference_feasibility_query is not None
                    ),
                    "future_reference_candidate_rejection_count": (
                        future_reference_candidate_rejection_count
                    ),
                    "future_reference_gate_version": (
                        "behavior2026.future-reference-interaction-feasibility/v1"
                    ),
                    "terminal_interrupted_candidate_private": (
                        None
                        if terminal_interrupted_candidate is None
                        else dict(terminal_interrupted_candidate)
                    ),
                    "execution_fidelity": "kinematic_research",
                    "official_challenge_eligible": False,
                }
            )
            cleanup_completed = True
        finally:
            if (rollback_symbolically is not None and not physical_consequence_committed
                and not verified_terminal_success
                and (not placement_committed or failure is not None or not cleanup_completed)):
                rollback_symbolically()


def marker_symbolic_place_on_floor_generator(
    owner: Any,
    *,
    action_primitive_error: Any,
):
    """Place the held object on a nearby floor through the OnTop transaction."""

    pose = planar_base_pose_from_robot(owner.robot)
    desired_xy = np.asarray(
        (
            pose.x + 0.75 * math.cos(pose.yaw_rad),
            pose.y + 0.75 * math.sin(pose.yaw_rad),
        ),
        dtype=np.float64,
    )
    robot_position, _ = owner.robot.get_position_orientation()
    base_z = float(robot_position[2])
    candidates = []
    unreadable_floor_geometry_count = 0
    for floor in tuple(getattr(owner.env.scene, "objects", ())):
        if str(getattr(floor, "category", "")).casefold() not in {
            "floor",
            "floors",
            "ground",
        }:
            continue
        try:
            lower, upper = object_aabb(floor)
        except Exception:
            unreadable_floor_geometry_count += 1
            continue
        if np.any(upper[:2] - lower[:2] <= 0.02):
            continue
        if float(upper[2]) > base_z + 0.30:
            continue
        local_xy = np.clip(
            desired_xy,
            lower[:2] + 0.01,
            upper[:2] - 0.01,
        )
        candidates.append(
            (
                float(np.linalg.norm(local_xy - desired_xy)),
                abs(float(upper[2]) - base_z),
                str(getattr(floor, "name", "")),
                floor,
                local_xy,
            )
        )
    if not candidates:
        metadata = {"unreadable_floor_geometry_count": unreadable_floor_geometry_count}
        if not owner._placement_physical_consequences_enabled:
            metadata[ACTION20_BLOCKED_CATEGORY_METADATA_KEY] = (
                EXECUTION_UNVERIFIED if unreadable_floor_geometry_count else PLACEMENT_SUPPORT_UNAVAILABLE
            )
        raise action_primitive_error(
            action_primitive_error.Reason.SAMPLING_ERROR,
            "No nearby floor support is available",
            metadata,
        )
    _distance, _height, _name, floor, center_xy = min(candidates)
    yield from owner.marker_symbolic_spatial_generator(
        floor,
        "place_on_floor",
        placement_region_center_xy=center_xy,
        placement_region_extent_xy=(0.60, 0.60),
    )
