"""Inside-placement execution for the live marker-symbolic primitive."""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping

import numpy as np

from behavior_hl.embodied.evaluator_bridge.deterministic_inside_placement import (
    Aabb3,
    DEFAULT_MAX_BLOCKED_FACE_PENETRATION_RATIO,
    DEFAULT_MAX_CONTENT_OVERLAP_RATIO,
    plan_symbolic_inside_candidates,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_visible_official_grasp_factory import (
    planar_base_pose_from_robot,
)
from behavior_hl.embodied.evaluator_bridge.logical_inventory import (
    LogicalInventorySnapshot,
)
from behavior_hl.embodied.evaluator_bridge.physical_inside_execution import (
    PhysicalInsideExecution,
)
from behavior_hl.embodied.evaluator_bridge.spatial_candidate_admission import (
    query_symbolic_placement_final_collision,
)
from behavior_hl.embodied.evaluator_bridge.spatial_symbolic_actions import (
    SpatialPoseCandidate,
    shelf_screen_cell_aabb,
)
from behavior_hl.embodied.runtime.public_action_feedback import (
    ACTION20_BLOCKED_CATEGORY_METADATA_KEY,
    EXECUTION_UNVERIFIED,
    GEOMETRIC_FIT_BLOCKED,
    PLACEMENT_COLLISION_BLOCKED,
    RELATION_NOT_REALIZED,
    REQUIRES_OBJECT_IN_LEFT_HAND,
    placement_failure_category,
)


def _preferred_inside_orientation(owner, physical, transform_utils, cpu_vector):
    """Use the pre-pick rotation, not the current hand presentation pose."""
    origin = getattr(owner, "_logical_inventory_origin_pose", None)
    if physical is not None or origin is None:
        return None
    return cpu_vector(transform_utils.mat2pose(origin)[1], length=4,
                      name="pre-pick Inside orientation")


def _empty_inside_plan_category(plan: Any) -> str | None:
    """Classify only an exclusive, structured empty-plan cause."""

    overlap = int(plan.rejected_initial_overlap)
    geometry = sum(
        int(value)
        for value in (
            plan.rejected_outside_volume,
            plan.rejected_non_fitting_orientation,
            plan.rejected_gross_intrusion,
        )
    )
    if overlap > 0 and geometry == 0:
        return PLACEMENT_COLLISION_BLOCKED
    if geometry > 0 and overlap == 0:
        return GEOMETRIC_FIT_BLOCKED
    return None


def _final_inside_collision_category(report: Any) -> str | None:
    if report is None or bool(getattr(report, "accepted", False)):
        return None
    object_reports = tuple(getattr(report, "object_reports", ()))
    if any(item.query_error for _name, item in object_reports):
        return EXECUTION_UNVERIFIED
    if any(item.blocking_hits for _name, item in object_reports):
        return PLACEMENT_COLLISION_BLOCKED
    return EXECUTION_UNVERIFIED


def _refresh_inside_readback_without_physics(
    owner: Any,
    *objects: Any,
) -> Mapping[str, Any]:
    """Refresh live geometry and state caches without advancing physics."""

    refresh = getattr(
        owner,
        "refresh_private_inside_selected_option_readback",
        None,
    )
    if not callable(refresh):
        raise RuntimeError("zero-physics Inside readback is unavailable")
    report = refresh(protected_objects=objects)
    if (
        not isinstance(report, Mapping)
        or report.get("accepted") is not True
        or report.get("physics_step_count") != 0
        or report.get("predicate_setter_called") is not False
    ):
        raise RuntimeError("zero-physics Inside readback contract failed")
    return report


def _final_inside_collision_admission(
    owner: Any,
    held: Any,
    container: Any,
):
    """Check closure against external solids and the container's real surfaces."""

    import omnigibson as og

    snapshot = owner._logical_inventory_snapshot
    return query_symbolic_placement_final_collision(
        omnigibson=og,
        moving_objects=(
            snapshot.object_refs if snapshot is not None else (held,)
        ),
        contact_objects=(container,),
    )


def _is_closeable_container(container: Any) -> bool:
    """Return whether a container exposes an official Open state."""

    states = getattr(container, "states", None)
    return isinstance(states, Mapping) and any(
        getattr(state_type, "__name__", "") == "Open"
        for state_type in states
    )


def _inside_existing_content_aabbs(
    held: Any,
    container: Any,
    *,
    live_inside_value: Callable[[Any, Any], bool],
) -> dict[str, Aabb3]:
    """Return live contents that the shared Inside planner must avoid."""

    scene_objects = getattr(getattr(held, "scene", None), "objects", ())
    if isinstance(scene_objects, Mapping):
        scene_objects = tuple(scene_objects.values())
    existing: dict[str, Aabb3] = {}
    for candidate_obj in tuple(scene_objects or ()):
        if candidate_obj is held or candidate_obj is container:
            continue
        if not live_inside_value(candidate_obj, container):
            continue
        bounds = getattr(candidate_obj, "aabb", None)
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            continue
        lower = np.asarray(bounds[0], dtype=np.float64).reshape(-1)
        upper = np.asarray(bounds[1], dtype=np.float64).reshape(-1)
        if lower.size != 3 or upper.size != 3:
            continue
        existing[str(getattr(candidate_obj, "name", id(candidate_obj)))] = (
            Aabb3(tuple(lower.tolist()), tuple(upper.tolist()))
        )
    return existing


def _physical_inside_execution(
    owner: Any,
) -> PhysicalInsideExecution | None:
    if not owner._placement_physical_consequences_enabled:
        return None
    return PhysicalInsideExecution(owner)


def _begin_inside_placement(
    owner: Any,
    physical: PhysicalInsideExecution | None,
):
    if physical is not None:
        return (yield from physical.begin())
    return (yield from owner._detach_held_object_at_current_pose())


def _inside_candidate_for_execution(
    physical: PhysicalInsideExecution | None,
    held: Any,
    container: Any,
    desired: SpatialPoseCandidate,
    *,
    candidate_aabb: Aabb3,
    fillable_aabb: Aabb3,
    opening_frame: Any | None = None,
    point_in_fillable_volume: Callable[[tuple[float, float, float]], bool]
    | None = None,
) -> tuple[SpatialPoseCandidate | None, tuple[Mapping[str, Any], ...], bool]:
    """Route one candidate without embedding physical release mechanics."""

    if physical is None:
        return desired, (), False
    return physical.admit_candidate(
        held,
        container,
        desired,
        candidate_aabb=candidate_aabb,
        fillable_aabb=fillable_aabb,
        opening_frame=opening_frame,
        point_in_fillable_volume=point_in_fillable_volume,
    )


def _symbolic_inside_readback(
    owner: Any,
    held: Any,
    container: Any,
    inside_state_type: Any,
    logical_snapshot: Any,
    *,
    cpu_vector: Callable[..., np.ndarray],
) -> Mapping[str, Any]:
    """Read one Inside result without advancing physics."""

    active = owner._logical_inventory_snapshot or logical_snapshot
    protected = [held, container]
    if active is not None:
        protected.extend(active.object_refs)
    refresh = _refresh_inside_readback_without_physics(owner, *protected)
    position, orientation = held.get_position_orientation()
    nested_preserved, _ = owner._logical_inventory_relations_preserved(active)
    official_inside = bool(held.states[inside_state_type].get_value(container))
    pose_finite = bool(
        np.isfinite(cpu_vector(position, length=3, name="Inside position")).all()
        and np.isfinite(
            cpu_vector(orientation, length=4, name="Inside orientation")
        ).all()
    )
    failures = []
    if not official_inside:
        failures.append("official_inside_not_satisfied")
    if not nested_preserved:
        failures.append("nested_relations_not_preserved")
    if not pose_finite:
        failures.append("non_finite_pose")
    return {
        "accepted": not failures,
        "readback_samples": 1,
        "physics_step_count": 0,
        "settle_steps_total": 0,
        "stability_sample_steps": 0,
        "official_inside_window": [official_inside],
        "nested_relations_window": [bool(nested_preserved)],
        "pose_finite_window": [pose_finite],
        "linear_speed_m_s": [],
        "angular_speed_rad_s": [],
        "failure_reasons": failures,
        "refresh_private": dict(refresh),
    }


def _inside_release_readback(
    physical: PhysicalInsideExecution | None,
    owner: Any,
    held: Any,
    container: Any,
    inside_state_type: Any,
    logical_snapshot: Any,
    *,
    cpu_vector: Callable[..., np.ndarray],
):
    """Dispatch to the isolated Symbolic or physical readback."""

    if physical is not None:
        return (
            yield from physical.release_readback(
                held,
                container,
                inside_state_type,
                logical_snapshot,
                cpu_vector=cpu_vector,
                refresh_without_physics=(
                    _refresh_inside_readback_without_physics
                ),
            )
        )
    return _symbolic_inside_readback(
        owner,
        held,
        container,
        inside_state_type,
        logical_snapshot,
        cpu_vector=cpu_vector,
    )


def _marker_symbolic_selected_inside_band_generator(
    owner: Any,
    container: Any,
    *,
    selected_selector: str,
    band_low: Any,
    band_high: Any,
    selection_granularity: str,
    cpu_vector: Callable[..., np.ndarray],
):
    """Place into one privately selected live compartment/Z band."""

    import torch as th
    from omnigibson import object_states
    from omnigibson.action_primitives.action_primitive_set_base import (
        ActionPrimitiveError,
    )
    from omnigibson.utils import transform_utils as T

    held = owner._get_obj_in_hand()
    if held is None:
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
            "Selected compartment placement requires a held object",
            {
                ACTION20_BLOCKED_CATEGORY_METADATA_KEY: (
                    REQUIRES_OBJECT_IN_LEFT_HAND
                )
            },
        )
    logical_snapshot = owner._logical_inventory_snapshot
    if owner._logical_inventory_carry and logical_snapshot is None:
        raise RuntimeError(
            "selected compartment placement lacks its carry snapshot"
        )
    low = (
        band_low.clone()
        if callable(getattr(band_low, "clone", None))
        else th.as_tensor(band_low)
    )
    high = (
        band_high.clone()
        if callable(getattr(band_high, "clone", None))
        else th.as_tensor(band_high)
    )
    root_position, root_orientation = held.get_position_orientation()
    if (
        owner._kinematic_object_extent_m is None
        or owner._kinematic_object_bbox_center_local_m is None
    ):
        raise RuntimeError("grasp-time held object geometry is unavailable")
    held_extent = cpu_vector(
        owner._kinematic_object_extent_m,
        length=3,
        name="held object grasp-time base-aligned bbox extent",
    )
    local_origin_to_center = th.as_tensor(
        cpu_vector(
            owner._kinematic_object_bbox_center_local_m,
            length=3,
            name="held object grasp-time bbox center in object frame",
        ),
        dtype=root_position.dtype,
        device=root_position.device,
    )
    selected_fillable = Aabb3(
        tuple(cpu_vector(low, length=3, name="selected band low")),
        tuple(cpu_vector(high, length=3, name="selected band high")),
    )

    def live_inside(candidate_obj: Any, target: Any) -> bool:
        states = getattr(candidate_obj, "states", None)
        if not isinstance(states, Mapping):
            return False
        state = states.get(object_states.Inside)
        return bool(state is not None and state.get_value(target))

    existing = _inside_existing_content_aabbs(
        held,
        container,
        live_inside_value=live_inside,
    )
    physical = _physical_inside_execution(owner)
    penetration_limit, content_overlap_limit, opening_face_is_bounded = (
        physical.plan_options()
        if physical is not None
        else (
            DEFAULT_MAX_BLOCKED_FACE_PENETRATION_RATIO,
            DEFAULT_MAX_CONTENT_OVERLAP_RATIO,
            True,
        )
    )
    plan = plan_symbolic_inside_candidates(
        held_extent_m=held_extent,
        preferred_orientation_xyzw=_preferred_inside_orientation(owner, physical, T, cpu_vector),
        fillable_aabb=selected_fillable,
        existing_content_aabbs=existing,
        max_candidates=48,
        maximum_blocked_face_penetration_ratio=penetration_limit,
        maximum_content_overlap_ratio=content_overlap_limit,
        include_opening_face_penetration=opening_face_is_bounded,
    )
    candidates = plan.candidates
    if not candidates:
        metadata = {"candidate plan": dict(plan.private_record())}
        if physical is None:
            category = _empty_inside_plan_category(plan)
            if category is not None:
                metadata[ACTION20_BLOCKED_CATEGORY_METADATA_KEY] = category
        owner._last_private_exact_report = {
            "operation": "place_inside",
            "selection_granularity": selection_granularity,
            "selected_selector_private": selected_selector,
            "candidate_plan_private": dict(plan.private_record()),
            "placement_committed": False,
            "physical_consequence_committed": False,
            "recoverable_executor_noop": True,
            "held_object_retained": owner._get_obj_in_hand() is held,
        }
        error = ActionPrimitiveError(
            ActionPrimitiveError.Reason.SAMPLING_ERROR,
            "No collision-free pose fits in the selected compartment",
            metadata,
        )
        error.private_report = dict(owner._last_private_exact_report)
        raise error
    selected_candidate = None
    selected_candidate_id = None
    selected_orientation = None
    official_inside = False
    band_preserved = False
    inside_stability: Mapping[str, Any] = {}
    placement_committed = False
    physical_consequence_committed = False
    physical_entry_path_used = False
    candidate_collision_reports: list[Mapping[str, Any]] = []
    candidate_attempts: list[dict[str, Any]] = []
    candidate_failure_categories: set[str | None] = set()
    failure: BaseException | None = None
    cleanup: Mapping[str, Any] = {}
    rollback_symbolically = (
        owner._capture_symbolic_inside_rollback(container, object_states.Inside)
        if physical is None and owner._logical_inventory_carry else None
    )
    owner._placing_with_predicate = True
    try:
        yield from _begin_inside_placement(owner, physical)
        owner._robot_object_collision_guard.restore(
            held,
            reason="before_selected_compartment_placement",
        )
        for index, planned in enumerate(candidates):
            rotation_index = int(planned.candidate_id.split(":")[1][1:])
            candidate_rotation = th.as_tensor(
                planned.orientation_matrix,
                dtype=root_position.dtype,
                device=root_position.device,
            )
            orientation = T.mat2quat(candidate_rotation)
            candidate = (
                th.as_tensor(
                    planned.position_m,
                    dtype=root_position.dtype,
                    device=root_position.device,
                )
                - candidate_rotation @ local_origin_to_center
            )
            desired = SpatialPoseCandidate(
                name=f"selected_inside:{planned.candidate_id}",
                position=cpu_vector(
                    candidate, length=3, name="Inside candidate position"
                ),
                orientation_xyzw=cpu_vector(
                    orientation,
                    length=4,
                    name="Inside candidate orientation",
                ),
            )
            attempt: dict[str, Any] = {
                "candidate_id_private": planned.candidate_id,
                "position_m_private": desired.position.tolist(),
                "orientation_xyzw_private": desired.orientation_xyzw.tolist(),
            }
            candidate_attempts.append(attempt)
            admitted, collision_reports, used_entry_path = (
                _inside_candidate_for_execution(
                    physical,
                    held,
                    container,
                    desired,
                    candidate_aabb=planned.initial_aabb,
                    fillable_aabb=selected_fillable,
                )
            )
            candidate_collision_reports.extend(collision_reports)
            attempt["collision_admission_private"] = list(collision_reports)
            if admitted is None:
                continue
            physical_entry_path_used = used_entry_path
            candidate = th.as_tensor(
                admitted.position,
                dtype=low.dtype,
                device=low.device,
            )
            orientation = th.as_tensor(
                admitted.orientation_xyzw,
                dtype=root_orientation.dtype,
                device=root_orientation.device,
            )
            held.set_position_orientation(
                position=candidate,
                orientation=orientation,
            )
            keep_still = getattr(held, "keep_still", None)
            if callable(keep_still):
                keep_still()
            if owner._logical_inventory_carry:
                active = owner._logical_inventory_snapshot or logical_snapshot
                pose = owner._logical_inventory_pose_for_object(held)
                owner._logical_inventory_snapshot = active.apply_root_pose(
                    pose,
                    set_pose=owner._logical_inventory_set_pose,
                    copy_pose=owner._logical_inventory_copy_pose,
                    compose_pose=owner._logical_inventory_compose_pose,
                    valid_pose=owner._logical_inventory_valid_pose,
                )
                owner._logical_inventory_pose = pose
            active = owner._logical_inventory_snapshot
            protected = [held, container]
            if active is not None:
                protected.extend(active.object_refs)
            readback_refresh = _refresh_inside_readback_without_physics(
                owner,
                *protected,
            )
            final_collision_report = (
                None
                if physical is not None
                else _final_inside_collision_admission(
                    owner,
                    held,
                    container,
                )
            )
            if final_collision_report is not None:
                candidate_collision_reports.append(
                    {
                        "candidate": desired.name,
                        "phase": "final_symbolic_pose",
                        **final_collision_report.to_private_dict(),
                    }
                )
            position, orientation = held.get_position_orientation()
            settled_low, settled_high = held.states[
                object_states.AABB
            ].get_value()
            official_inside = bool(
                held.states[object_states.Inside].get_value(container)
            )
            band_preserved = bool(
                th.all(settled_low >= low - 0.001).item()
                and th.all(settled_high <= high + 0.001).item()
            )
            nested_preserved, _ = owner._logical_inventory_relations_preserved(
                active
            )
            pose_finite = bool(
                th.isfinite(position).all().item()
                and th.isfinite(orientation).all().item()
            )
            inside_stability = {
                "accepted": bool(
                    official_inside
                    and band_preserved
                    and nested_preserved
                    and pose_finite
                    and (
                        final_collision_report is None
                        or final_collision_report.accepted
                    )
                ),
                "readback_samples": 1,
                "physics_step_count": 0,
                "final_state_collision_admission_private": (
                    None
                    if final_collision_report is None
                    else final_collision_report.to_private_dict()
                ),
                "refresh_private": dict(readback_refresh),
            }
            attempt["initial_readback_private"] = dict(inside_stability)
            if (
                physical is None
                and not inside_stability["accepted"]
            ):
                if not nested_preserved or not pose_finite:
                    candidate_failure_categories.add(EXECUTION_UNVERIFIED)
                elif (
                    final_collision_report is not None
                    and not final_collision_report.accepted
                ):
                    candidate_failure_categories.add(
                        _final_inside_collision_category(
                            final_collision_report
                        )
                    )
                elif not official_inside or not band_preserved:
                    candidate_failure_categories.add(
                        RELATION_NOT_REALIZED
                    )
                else:
                    candidate_failure_categories.add(None)
                if EXECUTION_UNVERIFIED in candidate_failure_categories:
                    break
            if inside_stability["accepted"] or (
                physical is not None
                and physical.pre_release_ready(
                    pose_finite=pose_finite,
                    nested_preserved=nested_preserved,
                )
            ):
                inside_stability = yield from _inside_release_readback(
                    physical,
                    owner,
                    held,
                    container,
                    object_states.Inside,
                    logical_snapshot,
                    cpu_vector=cpu_vector,
                )
                settled_low, settled_high = held.states[
                    object_states.AABB
                ].get_value()
                band_preserved = bool(
                    th.all(settled_low >= low - 0.001).item()
                    and th.all(settled_high <= high + 0.001).item()
                )
                inside_stability["selected_band_preserved"] = band_preserved
                if not band_preserved:
                    inside_stability["accepted"] = False
                    inside_stability["failure_reasons"].append(
                        "selected_band_not_preserved"
                    )
                official_inside = bool(
                    inside_stability["official_inside_window"]
                    and all(inside_stability["official_inside_window"])
                )
                nested_preserved = bool(
                    inside_stability["nested_relations_window"]
                    and all(inside_stability["nested_relations_window"])
                )
                attempt["release_readback_private"] = dict(inside_stability)
                if (
                    physical is None
                    and not inside_stability["accepted"]
                ):
                    failures = set(
                        inside_stability.get("failure_reasons", ())
                    )
                    candidate_failure_categories.add(
                        EXECUTION_UNVERIFIED
                        if failures
                        & {
                            "nested_relations_not_preserved",
                            "non_finite_pose",
                        }
                        else RELATION_NOT_REALIZED
                    )
                if inside_stability["accepted"]:
                    selected_candidate = index
                    selected_candidate_id = planned.candidate_id
                    selected_orientation = rotation_index
                elif (
                    physical is not None
                    and physical.consequence_commit_safe(inside_stability)
                ):
                    selected_candidate = index
                    selected_candidate_id = planned.candidate_id
                    selected_orientation = rotation_index
                    physical_consequence_committed = True
                    owner._clear_logical_inventory_state()
                elif (
                    physical is None
                    and rollback_symbolically is not None
                    and index + 1 < len(candidates)
                    and inside_stability.get("physics_step_count") == 0
                    and nested_preserved
                    and set(inside_stability.get("failure_reasons", ()))
                    and set(inside_stability["failure_reasons"]) <= {
                        "official_inside_not_satisfied",
                        "selected_band_not_preserved",
                    }
                ):
                    rollback_report = rollback_symbolically()
                    attempt["rollback_private"] = dict(rollback_report)
                    if not (
                        rollback_report.get("accepted") is True
                        and rollback_report.get("physics_step_count") == 0
                        and rollback_report.get("nested_relations_restored") is True
                        and rollback_report.get("protected_pose_restore_private", {}).get("accepted") is True
                        and rollback_report.get("logical_restore_private", {}).get("held_identity_restored") is True
                        and rollback_report.get("logical_restore_private", {}).get("logical_inventory_identity_restored") is True
                        and owner._get_obj_in_hand() is held
                    ):
                        raise RuntimeError("Inside candidate rollback was not verified")
                    owner._placing_with_predicate = True
                    continue
                break
        if (
            not inside_stability.get("accepted")
            and not physical_consequence_committed
        ):
            metadata = {"candidate count": len(candidates)}
            category = placement_failure_category(candidate_failure_categories)
            if category is not None:
                metadata[ACTION20_BLOCKED_CATEGORY_METADATA_KEY] = category
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.SAMPLING_ERROR,
                "Selected compartment did not satisfy official Inside and its selected band",
                metadata,
            )
        if not nested_preserved and not physical_consequence_committed:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "Selected compartment placement broke the held closure",
                {
                    ACTION20_BLOCKED_CATEGORY_METADATA_KEY: (
                        EXECUTION_UNVERIFIED
                    )
                },
            )
        if not physical_consequence_committed:
            placement_committed = True
            owner._clear_logical_inventory_state()
    except BaseException as exc:
        failure = exc
        raise
    finally:
        cleanup_completed = False
        try:
            owner._placing_with_predicate = False
            if (
                not placement_committed
                and not physical_consequence_committed
                and owner._logical_inventory_carry
                and owner._logical_inventory_snapshot is not None
                and rollback_symbolically is None
            ):
                eef_pose = T.pose2mat((
                    owner.robot.get_eef_position(owner.arm),
                    owner.robot.get_eef_orientation(owner.arm),
                ))
                owner._apply_logical_inventory_root_pose(
                    owner._fixed_kinematic_world_pose(eef_pose)
                )
                keep_still = getattr(held, "keep_still", None)
                if callable(keep_still):
                    keep_still()
                cleanup = {
                    "held_object_retained": owner._get_obj_in_hand() is held,
                    "logical_inventory_reprojected_to_carry": True,
                }
            if owner._robot_object_collision_guard.snapshot()["active"]:
                owner._robot_object_collision_guard.restore(
                    held,
                    reason="selected_compartment_final_cleanup",
                )
            owner._last_private_exact_report = {
                "operation": "place_inside",
                "selection_granularity": selection_granularity,
                "selected_selector_private": selected_selector,
                "selected_candidate_index_private": selected_candidate,
                "selected_candidate_id_private": selected_candidate_id,
                "selected_orientation_index_private": selected_orientation,
                "candidate_plan_private": dict(plan.private_record()),
                "candidate_attempts_private": candidate_attempts,
                "official_inside": official_inside,
                "selected_band_preserved": band_preserved,
                "inside_stability_private": dict(inside_stability),
                "placement_committed": placement_committed,
                "physical_consequence_committed": (
                    physical_consequence_committed
                ),
                "recoverable_executor_noop": False,
                "held_object_retained": owner._get_obj_in_hand() is held,
                "collision_free_entry_path_used_private": (
                    physical_entry_path_used
                ),
                "candidate_collision_admission_private": list(
                    candidate_collision_reports
                ),
                "predicate_setter_called": False,
                "error_private": (
                    None if failure is None else f"{type(failure).__name__}: {failure}"
                ),
                **dict(cleanup),
            }
            cleanup_completed = True
        finally:
            if (rollback_symbolically is not None and not physical_consequence_committed
                and (not placement_committed or failure is not None or not cleanup_completed)):
                rollback_symbolically()
            if failure is not None:
                failure.private_report = dict(owner._last_private_exact_report)


def marker_symbolic_inside_compartment_generator(
    owner: Any,
    target: Any,
    *,
    selected_selector: str | None = None,
):
    from omnigibson.action_primitives.action_primitive_set_base import (
        ActionPrimitiveError,
    )
    from behavior_hl.embodied.evaluator_bridge.private_drawer_parts import (
        DrawerPartTargetToken,
        drawer_fillable_aabbs,
    )

    if not isinstance(target, DrawerPartTargetToken) or not target.is_open():
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
            "Selected drawer compartment must be live and open",
        )
    regions = drawer_fillable_aabbs(target)
    low, high = max(
        regions,
        key=lambda bounds: float(np.prod(bounds[1] - bounds[0])),
    )
    yield from owner._marker_symbolic_selected_inside_band_generator(
        target.target_token,
        selected_selector=selected_selector or target.selector,
        band_low=low,
        band_high=high,
        selection_granularity="visible_open_drawer_compartment",
    )


def marker_symbolic_container_cell_generator(
    owner: Any,
    container: Any,
    *,
    selected_selector: str,
    cpu_vector: Callable[..., np.ndarray],
):
    import torch as th
    from omnigibson import object_states
    from omnigibson.action_primitives.action_primitive_set_base import (
        ActionPrimitiveError,
    )

    cell_match = re.fullmatch(
        r"visible_(?:container|shelf)_cell_r(\d+)_of_(\d+)_c(\d+)_of_(\d+)",
        selected_selector,
    )
    level_match = re.fullmatch(
        r"visible_shelf_level_(\d+)_of_(\d+)",
        selected_selector,
    )
    if cell_match is None and level_match is None:
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
            "Container placement requires a visible-cell selector",
        )
    low, high = container.states[object_states.AABB].get_value()
    low = low.clone() if callable(getattr(low, "clone", None)) else th.as_tensor(low)
    high = high.clone() if callable(getattr(high, "clone", None)) else th.as_tensor(high)
    if cell_match is not None:
        row_index, row_count, column_index, column_count = map(
            int,
            cell_match.groups(),
        )
        pose = planar_base_pose_from_robot(owner.robot)
        cell_low, cell_high = shelf_screen_cell_aabb(
            cpu_vector(low, length=3, name="shelf AABB low"),
            cpu_vector(high, length=3, name="shelf AABB high"),
            observer_xy=(pose.x, pose.y),
            row_index=row_index,
            row_count=row_count,
            column_index=column_index,
            column_count=column_count,
        )
        band_low = th.as_tensor(
            cell_low,
            dtype=low.dtype,
            device=low.device,
        )
        band_high = th.as_tensor(
            cell_high,
            dtype=high.dtype,
            device=high.device,
        )
        granularity = "visible_container_compartment"
    else:
        level_index, level_count = map(int, level_match.groups())
        if not 1 <= level_index <= level_count:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "Shelf selector is outside its visible level count",
            )
        band_height = (high[2] - low[2]) / level_count
        band_low = low.clone()
        band_high = high.clone()
        band_low[2] = low[2] + (level_count - level_index) * band_height
        band_high[2] = band_low[2] + band_height
        granularity = "visible_shelf_level"
    yield from owner._marker_symbolic_selected_inside_band_generator(
        container,
        selected_selector=selected_selector,
        band_low=band_low,
        band_high=band_high,
        selection_granularity=granularity,
    )


def _try_high_level_symbolic_inside_candidates(
    owner: Any,
    *,
    held: Any,
    container: Any,
    predicate: Any,
    physical: PhysicalInsideExecution | None,
    cpu_vector: Callable[..., np.ndarray],
    private_fillable_regions: Callable[[Any], Any],
    live_inside_value: Callable[[Any, Any], bool],
) -> tuple[bool, Mapping[str, Any]]:
    """Commit the first finite centre-in-fillable candidate without physics."""

    import torch as th
    from omnigibson.utils import transform_utils as T

    fillable_regions = private_fillable_regions(container)
    existing = _inside_existing_content_aabbs(
        held,
        container,
        live_inside_value=live_inside_value,
    )
    root_position, _root_orientation = held.get_position_orientation()
    if (
        owner._kinematic_object_extent_m is None
        or owner._kinematic_object_bbox_center_local_m is None
    ):
        raise RuntimeError("grasp-time held object geometry is unavailable")
    held_extent = cpu_vector(
        owner._kinematic_object_extent_m,
        length=3,
        name="held object grasp-time base-aligned bbox extent",
    )
    local_origin_to_center = th.as_tensor(
        cpu_vector(
            owner._kinematic_object_bbox_center_local_m,
            length=3,
            name="held object grasp-time bbox center in object frame",
        ),
        dtype=root_position.dtype,
        device=root_position.device,
    )

    reports: list[dict[str, Any]] = []
    plan_reports: list[Mapping[str, Any]] = []
    # Keep partial evidence if a query raises before this helper can return.
    owner._last_private_exact_report = {
        "operation": "place_inside",
        "symbolic_inside_candidate_report_private": {
            "fillable_region_count_private": len(fillable_regions),
            "fillable_region_plans_private": plan_reports,
            "candidate_attempts_private": reports,
            "existing_content_aabbs_private": {
                name: {"lower": list(bounds.lower), "upper": list(bounds.upper)}
                for name, bounds in existing.items()
            },
        },
    }
    failure_categories: set[str | None] = set()
    selected: str | None = None
    physical_entry_path_used = False
    closeable_container = _is_closeable_container(container)
    penetration_limit, content_overlap_limit, opening_face_is_bounded = (
        physical.plan_options()
        if physical is not None
        else (
            DEFAULT_MAX_BLOCKED_FACE_PENETRATION_RATIO,
            DEFAULT_MAX_CONTENT_OVERLAP_RATIO,
            bool(closeable_container or len(fillable_regions) > 1),
        )
    )
    for fillable_index, (fillable_link, fillable) in enumerate(fillable_regions):

        def point_in_volume(
            position: tuple[float, float, float],
            *,
            _link=fillable_link,
        ) -> bool:
            points = th.as_tensor([position], dtype=th.float32)
            return bool(_link.check_points_in_volume(points).item())

        plan = plan_symbolic_inside_candidates(
            held_extent_m=held_extent,
            preferred_orientation_xyzw=_preferred_inside_orientation(owner, physical, T, cpu_vector),
            fillable_aabb=fillable,
            existing_content_aabbs=existing,
            point_in_fillable_volume=point_in_volume,
            max_candidates=48,
            maximum_blocked_face_penetration_ratio=penetration_limit,
            maximum_content_overlap_ratio=content_overlap_limit,
            include_opening_face_penetration=opening_face_is_bounded,
        )
        plan_reports.append(
            {
                "fillable_region_index_private": fillable_index,
                "fillable_meta_link_name_private": str(
                    getattr(fillable_link, "name", fillable_index)
                ),
                **dict(plan.private_record()),
            }
        )
        if not plan.candidates and physical is None:
            failure_categories.add(_empty_inside_plan_category(plan))
        for candidate in plan.candidates:
            candidate_id = (
                f"fillable_{fillable_index}:{candidate.candidate_id}"
            )
            candidate_rotation = th.as_tensor(
                candidate.orientation_matrix,
                dtype=root_position.dtype,
                device=root_position.device,
            )
            orientation = T.mat2quat(candidate_rotation)
            candidate_root_position = (
                th.as_tensor(
                    candidate.position_m,
                    dtype=root_position.dtype,
                    device=root_position.device,
                )
                - candidate_rotation @ local_origin_to_center
            )
            pose = T.pose2mat((candidate_root_position, orientation))
            pose_finite = owner._logical_inventory_valid_pose(pose)
            attempt: dict[str, Any] = {
                "candidate_id_private": candidate_id,
                "fillable_region_index_private": fillable_index,
                "position_m_private": cpu_vector(candidate_root_position, length=3, name="Inside position").tolist(),
                "orientation_xyzw_private": cpu_vector(orientation, length=4, name="Inside orientation").tolist(),
                "initial_aabb_private": {
                    "lower": list(candidate.initial_aabb.lower),
                    "upper": list(candidate.initial_aabb.upper),
                },
                "candidate_center_in_fillable": True,
                "pose_finite": bool(pose_finite),
                "blocked_face_penetration_ratio_private": (
                    candidate.blocked_face_penetration_ratio
                ),
                "maximum_blocked_face_penetration_ratio_private": (
                    penetration_limit
                ),
                "intrusion_within_limit": bool(
                    candidate.blocked_face_penetration_ratio
                    <= penetration_limit
                ),
                "content_overlap_ratio_private": (
                    candidate.content_overlap_ratio
                ),
                "maximum_content_overlap_ratio_private": (
                    content_overlap_limit
                ),
                "content_overlap_within_limit": bool(
                    candidate.content_overlap_ratio <= content_overlap_limit
                ),
                "physics_step_count": 0,
                "accepted": False,
            }
            reports.append(attempt)
            if not pose_finite:
                if physical is None:
                    failure_categories.add(EXECUTION_UNVERIFIED)
                if physical is None:
                    break
                continue
            desired = SpatialPoseCandidate(
                name=candidate_id,
                position=cpu_vector(
                    candidate_root_position,
                    length=3,
                    name="Inside candidate position",
                ),
                orientation_xyzw=cpu_vector(
                    orientation,
                    length=4,
                    name="Inside candidate orientation",
                ),
            )
            admitted, collision_reports, used_entry_path = (
                _inside_candidate_for_execution(
                    physical,
                    held,
                    container,
                    desired,
                    candidate_aabb=candidate.initial_aabb,
                    fillable_aabb=fillable,
                    opening_frame=fillable_link,
                    point_in_fillable_volume=point_in_volume,
                )
            )
            attempt["collision_admission_private"] = list(collision_reports)
            if admitted is None:
                continue
            physical_entry_path_used = used_entry_path
            pose = T.pose2mat((
                th.as_tensor(
                    admitted.position,
                    dtype=root_position.dtype,
                    device=root_position.device,
                ),
                th.as_tensor(
                    admitted.orientation_xyzw,
                    dtype=root_position.dtype,
                    device=root_position.device,
                ),
            ))
            owner._apply_logical_inventory_root_pose(pose)
            active_snapshot = owner._logical_inventory_snapshot
            for carried in (
                (held,)
                if active_snapshot is None
                else active_snapshot.object_refs
            ):
                keep_still = getattr(carried, "keep_still", None)
                if callable(keep_still):
                    keep_still()

            protected = (
                (held, container)
                if active_snapshot is None
                else (*active_snapshot.object_refs, container)
            )
            try:
                refresh = dict(
                    _refresh_inside_readback_without_physics(
                        owner,
                        *protected,
                    )
                )
                refresh_error = None
            except Exception as exc:
                refresh = {}
                refresh_error = f"{type(exc).__name__}: {exc}"
            final_collision_report = (
                None
                if physical is not None or refresh_error is not None
                else _final_inside_collision_admission(
                    owner,
                    held,
                    container,
                )
            )
            try:
                official_inside = bool(
                    held.states[predicate].get_value(container)
                )
                official_inside_error = None
            except Exception as exc:
                official_inside = None
                official_inside_error = f"{type(exc).__name__}: {exc}"

            candidate_accepted = bool(
                physical is not None
                or (
                    refresh_error is None
                    and official_inside is True
                    and final_collision_report is not None
                    and final_collision_report.accepted
                )
            )
            attempt.update(
                {
                    "official_inside_private": official_inside,
                    "official_inside_error_private": official_inside_error,
                    "refresh_private": refresh,
                    "refresh_error_private": refresh_error,
                    "final_state_collision_admission_private": (
                        None
                        if final_collision_report is None
                        else final_collision_report.to_private_dict()
                    ),
                    "accepted": candidate_accepted,
                }
            )
            if not candidate_accepted:
                if physical is None:
                    if (
                        refresh_error is not None
                        or official_inside_error is not None
                    ):
                        failure_categories.add(EXECUTION_UNVERIFIED)
                    elif (
                        final_collision_report is not None
                        and not final_collision_report.accepted
                    ):
                        failure_categories.add(
                            _final_inside_collision_category(
                                final_collision_report
                            )
                        )
                    elif official_inside is not True:
                        failure_categories.add(RELATION_NOT_REALIZED)
                    else:
                        failure_categories.add(None)
                    if EXECUTION_UNVERIFIED in failure_categories:
                        break
                continue
            selected = candidate_id
            break
        if selected is not None or (physical is None and EXECUTION_UNVERIFIED in failure_categories):
            break

    if (
        selected is None
        and not failure_categories
        and physical is None
    ):
        failure_categories.add(EXECUTION_UNVERIFIED)

    report_policy = (
        dict(physical.candidate_report_policy())
        if physical is not None
        else {
            "semantic_admission_private": (
                "zero_solid_overlap_and_official_inside"
            ),
            "official_inside_readback_required": True,
            "blocked_face_penetration_is_rejection_gate": True,
            "final_state_collision_readback_required": True,
            "content_overlap_is_rejection_gate": True,
        }
    )
    return selected is not None, {
        "fillable_region_count_private": len(fillable_regions),
        "fillable_region_plans_private": plan_reports,
        "selected_candidate_id_private": selected,
        "candidate_attempts_private": reports,
        "existing_content_aabbs_private": {
            name: {"lower": list(bounds.lower), "upper": list(bounds.upper)}
            for name, bounds in existing.items()
        },
        "deterministic_failure_categories_private": sorted(
            category
            for category in failure_categories
            if category is not None
        ),
        "deterministic_failure_unclassified_private": (
            None in failure_categories
        ),
        "semantic_admission_private": report_policy[
            "semantic_admission_private"
        ],
        "collision_free_entry_path_used_private": (
            physical_entry_path_used
        ),
        "official_inside_state_setter_called": False,
        "official_inside_readback_required": report_policy[
            "official_inside_readback_required"
        ],
        "selected_option_goal_preservation_required": False,
        "complete_aabb_containment_required": False,
        "closeable_container_private": closeable_container,
        "maximum_blocked_face_penetration_ratio_private": (
            penetration_limit
        ),
        "blocked_face_penetration_is_rejection_gate": report_policy[
            "blocked_face_penetration_is_rejection_gate"
        ],
        "opening_face_penetration_is_rejection_gate": (
            opening_face_is_bounded
        ),
        "collision_shape_overlap_is_rejection_gate": True,
        "final_state_collision_readback_required": report_policy[
            "final_state_collision_readback_required"
        ],
        "maximum_content_overlap_ratio_private": content_overlap_limit,
        "content_overlap_is_rejection_gate": report_policy[
            "content_overlap_is_rejection_gate"
        ],
        "container_motion_thresholds_required": False,
        "container_motion_probe_steps": 0,
        "free_settle_steps": 0,
        "stability_sample_steps": 0,
        "gravity_preserved": True,
        "environment_collision_preserved": True,
    }


def place_inside_with_predicate(
    owner: Any,
    obj: Any,
    predicate: Any,
    *,
    cpu_vector: Callable[..., np.ndarray],
    live_inside_container_ready: Callable[[Any], bool],
):
    """Execute the logical-carry Inside transaction."""

    from omnigibson.action_primitives.action_primitive_set_base import (
        ActionPrimitiveError,
    )

    obj_in_hand = owner._get_obj_in_hand()
    if obj_in_hand is None:
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
            "You need to be grasping an object first to place it somewhere.",
            {
                ACTION20_BLOCKED_CATEGORY_METADATA_KEY: (
                    REQUIRES_OBJECT_IN_LEFT_HAND
                )
            },
        )
    if not live_inside_container_ready(obj):
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
            "Open the container before placing an object inside it.",
        )
    semantic_private_fallback = bool(
        owner._private_exact_operation_active == (obj, "place_inside")
        and owner._semantic_only_place_inside
    )
    relation = (str(obj_in_hand.name), str(obj.name))
    logical_snapshot = owner._logical_inventory_snapshot
    physical = _physical_inside_execution(owner) if not semantic_private_fallback else None
    rollback_symbolically = (
        owner._capture_symbolic_inside_rollback(obj, predicate)
        if not semantic_private_fallback
        and physical is None
        else None
    )
    owner._placing_with_predicate = True
    logical_placement_succeeded = False
    physical_consequence_committed = False
    failure: BaseException | None = None
    owner._last_private_exact_report = {"operation": "place_inside"}
    try:
        if semantic_private_fallback:
            yield from owner._detach_held_object_at_current_pose()
            owner._semantic_inside_ledger.commit(
                object_name=relation[0],
                container_name=relation[1],
            )
            placement_index = owner._private_exact_inside_fallback_count
            owner._private_exact_inside_fallback_count += 1
            if logical_snapshot is not None:
                owner._logical_inventory_last_descendant_count = (
                    logical_snapshot.descendant_count
                )
                owner._logical_inventory_last_sync_count = (
                    logical_snapshot.sync_count
                )
                owner._logical_inventory_nested_preserved = True
            logical_placement_succeeded = True
            owner._last_private_exact_report = {
                "operation": "place_inside",
                "placement_fallback_used": True,
                "placement_fallback": "private_semantic_inside_ledger",
                "semantic_only_place_inside": True,
                "semantic_relation_private": {
                    "object": relation[0],
                    "container": relation[1],
                },
                "semantic_relation_count_private": (
                    owner._semantic_inside_ledger.relation_count
                ),
                "placement_sequence_index": placement_index,
                "official_inside_state_setter": False,
                "official_predicate_verified": False,
                "physical_placement_attempted": False,
                "physics_stability_window_required": False,
                "logical_inventory_carry": True,
                "logical_inventory_closure_version": LogicalInventorySnapshot.VERSION,
                "logical_inventory_descendant_count": (
                    0
                    if logical_snapshot is None
                    else logical_snapshot.descendant_count
                ),
                "logical_inventory_sync_count": (
                    0
                    if logical_snapshot is None
                    else logical_snapshot.sync_count
                ),
                "logical_inventory_restored_count": 0,
                "logical_inventory_restore_error": None,
                "logical_inventory_nested_preserved": True,
            }
            owner._clear_logical_inventory_state()
        else:
            yield from _begin_inside_placement(owner, physical)
            owner._robot_object_collision_guard.restore(
                obj_in_hand,
                reason="before_high_level_symbolic_inside",
            )
            placement_index = owner._private_exact_inside_fallback_count
            placement_succeeded, symbolic_inside_report = (
                owner._try_high_level_symbolic_inside_candidates(
                    held=obj_in_hand,
                    container=obj,
                    predicate=predicate,
                    physical=physical,
                )
            )
            owner._last_private_exact_report = {
                "operation": "place_inside",
                "placement_fallback_used": True,
                "placement_fallback": "high_level_symbolic_inside_candidates",
                "official_inside_state_setter": False,
                "official_inside_get_value_required": False,
                "symbolic_inside_candidate_report_private": dict(
                    symbolic_inside_report
                ),
                "placement_sequence_index": placement_index,
                "placed_object_environment_collisions_preserved": True,
                "placed_object_gravity_preserved": True,
                "placed_object_inertial_load_preserved": True,
            }
            if not placement_succeeded:
                metadata = {
                    "held object": obj_in_hand.name,
                    "container": obj.name,
                }
                categories = set(
                    symbolic_inside_report.get(
                        "deterministic_failure_categories_private",
                        (),
                    )
                )
                if symbolic_inside_report.get("deterministic_failure_unclassified_private", False):
                    categories.add(None)
                category = placement_failure_category(categories)
                if physical is None and category is not None:
                    metadata[
                        ACTION20_BLOCKED_CATEGORY_METADATA_KEY
                    ] = category
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.SAMPLING_ERROR,
                    "No size- and capacity-admissible Inside candidate was available",
                    metadata,
                )
            inside_stability = yield from _inside_release_readback(
                physical,
                owner,
                obj_in_hand,
                obj,
                predicate,
                logical_snapshot,
                cpu_vector=cpu_vector,
            )
            owner._last_private_exact_report.update(
                {
                    "official_inside_readback_required": True,
                    "physics_stability_window_required": False,
                    "free_settle_steps": int(
                        inside_stability["physics_step_count"]
                    ),
                    "stability_sample_steps": 0,
                    "inside_stability_private": dict(inside_stability),
                }
            )
            physical_consequence_committed = bool(
                not inside_stability["accepted"]
                and physical is not None
                and physical.consequence_commit_safe(inside_stability)
            )
            if physical_consequence_committed:
                owner._last_private_exact_report.update(
                    {
                        "logical_inventory_carry": True,
                        "placement_committed": False,
                        "physical_consequence_committed": True,
                        "held_object_retained": False,
                        "recoverable_executor_noop": False,
                    }
                )
                owner._clear_logical_inventory_state()
                owner._robot_object_collision_guard.restore(
                    obj_in_hand,
                    reason="placement_pose_applied_before_settle",
                )
                return
            if not inside_stability["accepted"]:
                failure_reasons = set(
                    inside_stability["failure_reasons"]
                )
                category = (
                    EXECUTION_UNVERIFIED
                    if failure_reasons
                    & {
                        "nested_relations_not_preserved",
                        "non_finite_pose",
                    }
                    else RELATION_NOT_REALIZED
                )
                metadata = {
                    "failure reasons": inside_stability["failure_reasons"]
                }
                if physical is None:
                    metadata[
                        ACTION20_BLOCKED_CATEGORY_METADATA_KEY
                    ] = category
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                    "Inside placement did not satisfy official readback",
                    metadata,
                )
            owner._semantic_inside_ledger.commit(
                object_name=relation[0],
                container_name=relation[1],
            )
            owner._private_exact_inside_fallback_count += 1
            # Preserve the experimental Physical commit boundary. Symbolic
            # remains rollbackable until release cleanup has also succeeded.
            logical_placement_succeeded = physical is not None
            snapshot_after = owner._logical_inventory_snapshot
            if snapshot_after is not None:
                owner._logical_inventory_last_descendant_count = (
                    snapshot_after.descendant_count
                )
                owner._logical_inventory_last_sync_count = (
                    snapshot_after.sync_count
                )
            owner._logical_inventory_nested_preserved = True
            owner._last_private_exact_report.update(
                {
                    "logical_inventory_carry": True,
                    "placement_committed": True,
                    "semantic_relation_private": {
                        "object": relation[0],
                        "container": relation[1],
                    },
                    "semantic_relation_count_private": (
                        owner._semantic_inside_ledger.relation_count
                    ),
                    "zero_physics_semantic_commit": (
                        physical is None
                    ),
                    "logical_inventory_nested_preserved": True,
                }
            )
            owner._clear_logical_inventory_state()
            owner._robot_object_collision_guard.restore(
                obj_in_hand,
                reason="placement_pose_applied_before_settle",
            )
            logical_placement_succeeded = True
    except BaseException as exc:
        failure = exc
        raise
    finally:
        owner._placing_with_predicate = False
        try:
            if (
                logical_snapshot is not None
                and not logical_placement_succeeded
                and not physical_consequence_committed
            ):
                failure_report = dict(owner._last_private_exact_report or {})
                rollback_report = None
                if rollback_symbolically is not None:
                    rollback_report = rollback_symbolically()
                else:
                    owner.sync_marker_kinematic_hold()
                owner._last_private_exact_report = {
                    **failure_report,
                    "operation": "place_inside",
                    "logical_inventory_carry": True,
                    "placement_committed": False,
                    "recoverable_executor_noop": bool(rollback_report is not None and rollback_report["accepted"]),
                    "held_object_retained": owner._get_obj_in_hand() is obj_in_hand,
                    "logical_inventory_restored_count": (
                        0 if rollback_report is None
                        else rollback_report["logical_inventory_restored_count"]
                    ),
                    "logical_inventory_restore_error": None,
                    "logical_inventory_nested_preserved": (
                        None if rollback_report is None else rollback_report["nested_relations_restored"]
                    ),
                    "rollback_private": rollback_report,
                }
        finally:
            owner._placing_with_predicate = False
            if owner._robot_object_collision_guard.snapshot()["active"]:
                owner._robot_object_collision_guard.restore(
                    obj_in_hand,
                    reason="placement_exception_cleanup",
                )
            if failure is not None:
                failure.private_report = dict(owner._last_private_exact_report)
    if not semantic_private_fallback:
        try:
            official_inside_readback = bool(
                obj_in_hand.states[predicate].get_value(obj)
            )
            official_inside_readback_error = None
        except Exception as exc:  # private diagnostics cannot undo a commit
            official_inside_readback = None
            official_inside_readback_error = (
                f"{type(exc).__name__}: {exc}"
            )
        owner._last_private_exact_report.update(
            {
                "official_inside_readback_private": official_inside_readback,
                "official_inside_readback_error_private": (
                    official_inside_readback_error
                ),
                "official_inside_readback_is_rejection_gate": True,
            }
        )
    owner._skip_next_symbolic_settle = True
