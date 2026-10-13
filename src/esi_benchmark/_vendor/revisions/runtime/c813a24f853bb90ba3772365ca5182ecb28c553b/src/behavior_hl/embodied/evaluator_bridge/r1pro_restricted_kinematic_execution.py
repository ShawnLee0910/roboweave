"""Live restricted-kinematic navigation adapter for R1Pro."""

from __future__ import annotations

from dataclasses import asdict
import math
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from behavior_hl.embodied.evaluator_bridge.interaction_station_geometry import (
    supercover_line_cells,
)
from behavior_hl.embodied.evaluator_bridge.private_traversability_route import (
    _load_structural_open_grid,
    check_official_local_segment,
    official_world_to_map_cell,
)
from behavior_hl.embodied.evaluator_bridge.private_door_navigation import (
    OPENABLE_ROOM_PORTAL_CATEGORIES,
)
from behavior_hl.embodied.evaluator_bridge.private_drawer_parts import (
    ARTICULATED_ENDPOINT_TOLERANCE_RATIO,
    _is_revolute,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_b1_component_builders import (
    _assert_marker_symbolic_grasp_held,
    _marker_symbolic_closed_hold_action,
    _single_articulated_part_token,
)
from behavior_hl.embodied.runtime.marker_symbolic_carry_contract import (
    MarkerSymbolicCarryPhase,
)
from behavior_hl.embodied.runtime.plugins.restricted_kinematic_base import (
    RESTRICTED_KINEMATIC_INTERNAL_SAMPLE_M,
    RESTRICTED_KINEMATIC_PUBLIC_MOVE_M,
    FixedStepKinematicBaseExecutor,
)
from behavior_hl.navigation import PlanarPose


RESTRICTED_KINEMATIC_POST_APPLY_STEPS = 4
# ponytail: add portal normals only if a scene has overlapping nearby doors.
PORTAL_SNAP_CAPTURE_RANGE_M = 1.0
PORTAL_SNAP_MIN_LATERAL_OVERLAP = 0.50
PORTAL_SNAP_MAX_TRANSLATION_M = 2.0


def _portal_object_state(
    obj: Any,
) -> tuple[bool, tuple[float, float]] | None:
    """Return whether one declared portal is at its fully-open endpoint."""

    try:
        position, _ = obj.get_position_orientation()
        token = _single_articulated_part_token(obj)
        if not _is_revolute(token.joint_token):
            return None
        open_end, closed_end = token.endpoints()
        tolerance = max(
            1.0e-5,
            abs(open_end - closed_end)
            * ARTICULATED_ENDPOINT_TOLERANCE_RATIO,
        )
        return (
            abs(token.current_position() - open_end) <= tolerance,
            (float(position[0]), float(position[1])),
        )
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
        return None


def _portal_snap_target(
    scene: Any,
    robot: Any,
    start: PlanarPose,
    requested: PlanarPose,
    *,
    local_report: Mapping[str, Any],
    local_segment_checker: Callable[..., Any],
) -> tuple[PlanarPose | None, Mapping[str, Any] | None]:
    """Resolve one forward step to the nearest fully-cleared portal pose."""

    dx, dy = requested.x - start.x, requested.y - start.y
    requested_m = math.hypot(dx, dy)
    forward = (math.cos(start.yaw), math.sin(start.yaw))
    if (
        requested_m <= 1.0e-9
        or requested.yaw != start.yaw
        or (dx * forward[0] + dy * forward[1]) / requested_m < 1.0 - 1.0e-6
    ):
        return None, None
    try:
        trav_map = scene.trav_map
        resolution = float(trav_map.map_resolution)
        map_size = int(trav_map.map_size)
        floor = int(local_report["floor"])
        source = tuple(map(int, local_report["source_official_cell"]))
        target = tuple(map(int, local_report["target_official_cell"]))
        x_min, y_min, x_max, y_max = map(
            float, local_report["base_collision_bounds_xy_m"]
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return None, None
    if (
        local_report.get("map_with_objects") is not True
        or local_report.get("base_collision_geometry_available") is not True
        or local_report.get("static_map_policy")
        != "oriented_base_collision_rectangle"
        or resolution <= 0.0
        or source == target
        or x_max <= x_min
        or y_max <= y_min
    ):
        return None, None
    structural = _load_structural_open_grid(
        str(getattr(scene, "scene_dir", "")), floor, map_size
    )
    if structural is None or structural.shape != (map_size, map_size):
        return None, None

    forward_probe = official_world_to_map_cell(
        trav_map=trav_map,
        point=(
            start.x + PORTAL_SNAP_CAPTURE_RANGE_M * forward[0],
            start.y + PORTAL_SNAP_CAPTURE_RANGE_M * forward[1],
        ),
    )
    direction_cell = target if forward_probe is None else forward_probe
    delta = (
        direction_cell[0] - source[0],
        direction_cell[1] - source[1],
    )
    norm = math.hypot(*delta)
    along = (delta[0] / norm, delta[1] / norm)
    lateral = (-along[1], along[0])
    radius = max(1, math.ceil(PORTAL_SNAP_CAPTURE_RANGE_M / resolution))
    cells = supercover_line_cells(
        (
            round(source[0] - along[0] * radius),
            round(source[1] - along[1] * radius),
        ),
        (
            round(source[0] + along[0] * radius),
            round(source[1] + along[1] * radius),
        ),
    )
    sections: list[tuple[tuple[int, int], tuple[float, float] | None]] = []
    for center in cells:
        row, col = center
        if not (0 <= row < map_size and 0 <= col < map_size) or not structural[
            row, col
        ]:
            sections.append((center, None))
            continue
        offsets: list[float] = []
        for sign in (-1.0, 1.0):
            visited = {center}
            for offset in range(1, radius + 1):
                side = (
                    round(row + sign * lateral[0] * offset),
                    round(col + sign * lateral[1] * offset),
                )
                if side in visited:
                    continue
                visited.add(side)
                if not (0 <= side[0] < map_size and 0 <= side[1] < map_size):
                    break
                if not structural[side]:
                    offsets.append(math.dist(center, side))
                    break
        sections.append((center, tuple(offsets) if len(offsets) == 2 else ()))

    captured: list[
        tuple[float, float, float, float, tuple[int, int]]
    ] = []
    base_width = y_max - y_min
    source_index = min(
        range(len(sections)),
        key=lambda i: math.dist(sections[i][0], source),
    )
    broad_section_seen = any(section[1] == () for section in sections)
    for index, (cell, walls) in enumerate(sections):
        if not walls or (index > 0 and sections[index - 1][1]):
            continue
        end = index
        while end + 1 < len(sections) and sections[end + 1][1]:
            end += 1
        if not broad_section_seen or any(
            sections[i][1] is None
            for i in range(min(source_index, index), max(source_index, end) + 1)
        ):
            continue
        near_m = (
            (cell[0] - source[0]) * along[0]
            + (cell[1] - source[1]) * along[1]
        ) * resolution
        far_cell = sections[end][0]
        far_m = (
            (far_cell[0] - source[0]) * along[0]
            + (far_cell[1] - source[1]) * along[1]
        ) * resolution
        landing_m = far_m + max(0.0, -x_min) + resolution
        run_walls = [sections[i][1] for i in range(index, end + 1)]
        longitudinal_distance_m = max(0.0, near_m, -far_m)
        lateral_overlap_ratio = min(
            max(
                0.0,
                min(y_max, (pair[1] - 0.5) * resolution)
                - max(y_min, (-pair[0] + 0.5) * resolution),
            )
            / base_width
            for pair in run_walls
        )
        if (
            longitudinal_distance_m
            <= PORTAL_SNAP_CAPTURE_RANGE_M + 1.0e-9
            and lateral_overlap_ratio + 1.0e-9
            >= PORTAL_SNAP_MIN_LATERAL_OVERLAP
            and landing_m > requested_m + 1.0e-9
        ):
            captured.append(
                (
                    landing_m,
                    longitudinal_distance_m,
                    lateral_overlap_ratio,
                    min((sum(pair) - 1.0) * resolution for pair in run_walls),
                    sections[(index + end) // 2][0],
                )
            )
    if not captured:
        return None, None
    (
        landing_m,
        longitudinal_distance_m,
        lateral_overlap_ratio,
        gap_width_m,
        portal_cell,
    ) = min(captured)
    audit: dict[str, Any] = {
        "attempted": True,
        "longitudinal_capture_range_m": PORTAL_SNAP_CAPTURE_RANGE_M,
        "longitudinal_distance_m": float(longitudinal_distance_m),
        "lateral_footprint_overlap_ratio": float(lateral_overlap_ratio),
        "minimum_lateral_footprint_overlap_ratio": (
            PORTAL_SNAP_MIN_LATERAL_OVERLAP
        ),
        "structural_portal_cell_private": list(portal_cell),
        "structural_gap_width_m": float(gap_width_m),
        "required_gap_width_m": float(base_width + resolution),
        "requested_translation_m": float(requested_m),
        "snap_translation_m": float(landing_m),
    }
    if (
        gap_width_m + 1.0e-9 < base_width + resolution
        or landing_m > PORTAL_SNAP_MAX_TRANSLATION_M
    ):
        return None, {**audit, "status": "blocked", "reason": "portal_geometry_blocked"}

    portal_m = (
        (portal_cell[0] - source[0]) * along[0]
        + (portal_cell[1] - source[1]) * along[1]
    ) * resolution
    portal_xy = (
        start.x + portal_m * forward[0],
        start.y + portal_m * forward[1],
    )
    nearby_doors: list[tuple[Any, tuple[bool, tuple[float, float]] | None]] = []
    for obj in tuple(getattr(scene, "objects", ()) or ()):
        if str(getattr(obj, "category", "")) not in OPENABLE_ROOM_PORTAL_CATEGORIES:
            continue
        state = _portal_object_state(obj)
        if state is None:
            try:
                position, _ = obj.get_position_orientation()
                object_xy = (float(position[0]), float(position[1]))
            except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
                continue
        else:
            object_xy = state[1]
        if math.dist(object_xy, portal_xy) <= PORTAL_SNAP_CAPTURE_RANGE_M:
            nearby_doors.append((obj, state))
    if nearby_doors and any(state is None or not state[0] for _, state in nearby_doors):
        return None, {
            **audit,
            "status": "blocked",
            "reason": "declared_portal_not_fully_open",
            "door_names_private": [
                str(getattr(obj, "name", "")) for obj, _ in nearby_doors
            ],
        }

    minimum_landing = PlanarPose(
        start.x + landing_m * forward[0],
        start.y + landing_m * forward[1],
        start.z,
        start.yaw,
    )
    left = (-forward[1], forward[0])
    scan_step_m = max(resolution, RESTRICTED_KINEMATIC_INTERNAL_SAMPLE_M)
    scan_steps = int(PORTAL_SNAP_CAPTURE_RANGE_M / scan_step_m + 1.0e-9)
    offsets = sorted(
        (
            math.hypot(forward_step * scan_step_m, lateral_step * scan_step_m),
            abs(lateral_step),
            forward_step,
            lateral_step,
        )
        for forward_step in range(scan_steps + 1)
        for lateral_step in range(-scan_steps, scan_steps + 1)
        if math.hypot(forward_step, lateral_step) <= scan_steps + 1.0e-9
    )
    selected: tuple[
        PlanarPose,
        PlanarPose,
        dict[str, Any],
        dict[str, Any],
        float,
        float,
    ] | None = None
    candidates_checked = 0

    def segment_open(
        source_pose: PlanarPose,
        target_pose: PlanarPose,
    ) -> tuple[bool, dict[str, Any]]:
        report = local_segment_checker(
            scene=scene,
            robot=robot,
            start_xy=(source_pose.x, source_pose.y),
            target_xy=(target_pose.x, target_pose.y),
            robot_z=start.z,
            start_yaw_rad=start.yaw,
            target_yaw_rad=start.yaw,
            allow_bounded_source_snap=False,
        )
        private = dict(report.private_report)
        return (
            bool(
                report.traversable
                and private.get("map_with_objects") is True
                and private.get("source_open") is True
                and private.get("base_collision_geometry_available") is True
            ),
            private,
        )

    try:
        minimum_open, minimum_private = segment_open(
            minimum_landing,
            minimum_landing,
        )
        if not minimum_open:
            return None, {
                **audit,
                "status": "blocked",
                "reason": "portal_landing_blocked",
                "landing_guard_private": dict(minimum_private),
            }
        for _, _, forward_step, lateral_step in offsets:
            forward_offset_m = forward_step * scan_step_m
            lateral_offset_m = lateral_step * scan_step_m
            candidate = PlanarPose(
                minimum_landing.x
                + forward_offset_m * forward[0]
                + lateral_offset_m * left[0],
                minimum_landing.y
                + forward_offset_m * forward[1]
                + lateral_offset_m * left[1],
                start.z,
                start.yaw,
            )
            if (
                math.hypot(candidate.x - start.x, candidate.y - start.y)
                > PORTAL_SNAP_MAX_TRANSLATION_M + 1.0e-9
            ):
                continue
            candidates_checked += 1
            candidate_open, candidate_private = (
                (True, minimum_private)
                if candidate == minimum_landing
                else segment_open(minimum_landing, candidate)
            )
            if not candidate_open:
                continue
            egress = PlanarPose(
                candidate.x
                + RESTRICTED_KINEMATIC_PUBLIC_MOVE_M * forward[0],
                candidate.y
                + RESTRICTED_KINEMATIC_PUBLIC_MOVE_M * forward[1],
                start.z,
                start.yaw,
            )
            egress_open, egress_private = segment_open(candidate, egress)
            if egress_open:
                selected = (
                    candidate,
                    egress,
                    candidate_private,
                    egress_private,
                    forward_offset_m,
                    lateral_offset_m,
                )
                break
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        return None, {
            **audit,
            "status": "blocked",
            "reason": "portal_landing_check_failed",
            "error_type": type(exc).__name__,
        }
    if selected is None:
        return None, {
            **audit,
            "status": "blocked",
            "reason": "portal_forward_egress_unavailable",
            "landing_search_radius_m": PORTAL_SNAP_CAPTURE_RANGE_M,
            "landing_search_step_m": float(scan_step_m),
            "landing_candidates_checked": candidates_checked,
        }
    (
        landing,
        forward_egress,
        landing_private,
        egress_private,
        selected_forward_offset_m,
        selected_lateral_offset_m,
    ) = selected
    final_audit = {
        **audit,
        "status": "completed",
        "reason": "structural_portal_snap",
        "portal_kind_private": (
            "open_declared_door" if nearby_doors else "doorless_structural_opening"
        ),
        "door_names_private": [
            str(getattr(obj, "name", "")) for obj, _ in nearby_doors
        ],
        "landing_pose_private": {
            "x": float(landing.x),
            "y": float(landing.y),
            "yaw_rad": float(landing.yaw),
        },
        "minimum_landing_pose_private": {
            "x": float(minimum_landing.x),
            "y": float(minimum_landing.y),
            "yaw_rad": float(minimum_landing.yaw),
        },
        "forward_egress_pose_private": {
            "x": float(forward_egress.x),
            "y": float(forward_egress.y),
            "yaw_rad": float(forward_egress.yaw),
        },
        "minimum_clear_landing_translation_m": float(landing_m),
        "snap_translation_m": float(
            math.hypot(landing.x - start.x, landing.y - start.y)
        ),
        "selected_forward_offset_m": float(selected_forward_offset_m),
        "selected_lateral_offset_m": float(selected_lateral_offset_m),
        "landing_search_radius_m": PORTAL_SNAP_CAPTURE_RANGE_M,
        "landing_search_step_m": float(scan_step_m),
        "landing_candidates_checked": candidates_checked,
        "landing_guard_private": dict(landing_private),
        "forward_egress_guard_private": dict(egress_private),
    }
    return landing, final_audit


def _symbolic_open_portal_crossing(
    scene: Any,
    start: PlanarPose,
    target: PlanarPose,
) -> Mapping[str, Any] | None:
    """Recognize one fully-open room-door boundary crossing."""

    if abs(target.yaw - start.yaw) > 1.0e-12:
        return None
    segmentation = getattr(scene, "_seg_map", None)
    room_at = getattr(segmentation, "get_room_instance_by_point", None)
    if not callable(room_at):
        return None
    try:
        start_room = room_at((float(start.x), float(start.y)))
        target_room = room_at((float(target.x), float(target.y)))
    except (RuntimeError, TypeError, ValueError):
        return None
    if (
        not isinstance(start_room, str)
        or not isinstance(target_room, str)
        or not start_room
        or not target_room
        or start_room == target_room
    ):
        return None

    for door in tuple(getattr(scene, "objects", ()) or ()):
        if str(getattr(door, "category", "")) not in OPENABLE_ROOM_PORTAL_CATEGORIES:
            continue
        rooms = {
            str(room)
            for room in (getattr(door, "in_rooms", None) or ())
            if isinstance(room, str) and room
        }
        if rooms != {start_room, target_room}:
            continue
        state = _portal_object_state(door)
        if state is None or not state[0]:
            continue
        center_x, center_y = state[1]
        portal_distance = min(
            math.hypot(center_x - start.x, center_y - start.y),
            math.hypot(center_x - target.x, center_y - target.y),
        )
        if portal_distance <= PORTAL_SNAP_CAPTURE_RANGE_M:
            return {
                "door_name_private": str(getattr(door, "name", "")),
                "room_transition_private": [start_room, target_room],
                "portal_distance_m": float(portal_distance),
                "full_body_sweep_ignored": True,
            }
    return None


def read_restricted_planar_pose(
    robot: Any,
    *,
    quaternion_to_euler: Callable[[Any], Any],
) -> PlanarPose:
    """Read the robot pose through the live simulator adapter."""

    position, orientation = robot.get_position_orientation()
    yaw = float(quaternion_to_euler(orientation)[2].item())
    return PlanarPose(
        float(position[0]),
        float(position[1]),
        float(position[2]),
        yaw,
    )


def restricted_local_guard(
    start: PlanarPose,
    target: PlanarPose,
    *,
    diagnostic: dict[str, Any],
    robot: Any,
    scene: Any,
    endpoint_only: bool = False,
    local_segment_checker: Callable[..., Any] = check_official_local_segment,
) -> bool | PlanarPose:
    """Apply the configured static traversability guard to one local step."""

    guard_started = perf_counter()
    diagnostic.clear()
    if not isinstance(endpoint_only, bool):
        raise TypeError("endpoint_only must be boolean")
    if abs(float(target.z) - float(start.z)) > 1.0e-9:
        diagnostic.update(reason="requested_z_change", traversable=False)
        return False
    checked_start = target if endpoint_only else start
    report = local_segment_checker(
        scene=scene,
        robot=robot,
        start_xy=(checked_start.x, checked_start.y),
        target_xy=(target.x, target.y),
        robot_z=target.z,
        start_yaw_rad=checked_start.yaw,
        target_yaw_rad=target.yaw,
        allow_bounded_source_snap=not endpoint_only,
    )
    diagnostic.update(report.private_report)
    diagnostic["traversable"] = bool(report.traversable)
    diagnostic["collision_scope"] = (
        "endpoint_pose" if endpoint_only else "swept_segment"
    )
    diagnostic["timing_private"] = {
        "official_local_segment_seconds": perf_counter() - guard_started,
    }
    if endpoint_only:
        return target if report.traversable else False
    snap_target, snap_audit = _portal_snap_target(
        scene,
        robot,
        start,
        target,
        local_report=diagnostic,
        local_segment_checker=local_segment_checker,
    )
    if snap_audit is not None:
        diagnostic["portal_snap_private"] = dict(snap_audit)
        if snap_audit.get("status") == "blocked":
            return False
    if snap_target is not None:
        return snap_target
    if report.traversable:
        return target
    requested_m = math.hypot(target.x - start.x, target.y - start.y)
    if requested_m <= 1e-9 or target.yaw != start.yaw:
        return False
    if report.private_report.get("source_open") is False:
        diagnostic.update(
            reason="source_blocked_deferred_to_live_collision_guard",
            source_escape_live_collision_required=True,
        )
        return target
    try:
        safe_prefix_m = float(report.private_report["safe_prefix_length_m"])
    except (KeyError, TypeError, ValueError):
        return False
    if (
        not math.isfinite(safe_prefix_m)
        or safe_prefix_m < RESTRICTED_KINEMATIC_INTERNAL_SAMPLE_M - 1e-9
    ):
        return False
    fraction = min(1.0, safe_prefix_m / requested_m)
    return PlanarPose(
        start.x + fraction * (target.x - start.x),
        start.y + fraction * (target.y - start.y),
        start.z,
        target.yaw,
    )


def build_restricted_kinematic_executor(
    *,
    arm_presentation_ladder: Any | None,
    candidate_collision_samples: Callable[
        [PlanarPose, PlanarPose], Sequence[PlanarPose]
    ],
    candidate_pose_collision_provider: Any,
    collision_report: dict[str, Any],
    control: Any,
    evaluator: Any,
    guard: Callable[[PlanarPose, PlanarPose], bool | PlanarPose],
    guard_report: dict[str, Any],
    public_snapshot_provider: Any,
    read_pose: Callable[[], PlanarPose],
    step_once: Any,
    symbolic: Any,
    turn_angle_deg: float,
    endpoint_only_collision: bool = False,
    reuse_preflighted_empty_hand_world: bool = False,
    reuse_current_collision_world: bool = False,
    post_apply_steps: int = RESTRICTED_KINEMATIC_POST_APPLY_STEPS,
) -> FixedStepKinematicBaseExecutor:
    """Build the live collision-checked base executor for one public action."""

    if not callable(step_once) or not callable(public_snapshot_provider):
        raise RuntimeError("restricted kinematic executor lacks resident callbacks")
    if candidate_pose_collision_provider is None:
        raise RuntimeError(
            "restricted kinematic candidate collision provider is unavailable"
        )
    if not isinstance(endpoint_only_collision, bool):
        raise TypeError("endpoint_only_collision must be boolean")
    if not isinstance(reuse_preflighted_empty_hand_world, bool):
        raise TypeError("collision world reuse flag must be boolean")
    if not isinstance(reuse_current_collision_world, bool):
        raise TypeError("current collision world reuse flag must be boolean")
    if reuse_preflighted_empty_hand_world and reuse_current_collision_world:
        raise ValueError("collision world reuse modes are mutually exclusive")
    if (
        isinstance(post_apply_steps, bool)
        or not isinstance(post_apply_steps, int)
        or not 0 <= post_apply_steps <= RESTRICTED_KINEMATIC_POST_APPLY_STEPS
    ):
        raise ValueError("post_apply_steps must be an integer from 0 to 4")
    guard_report.clear()
    collision_report.clear()

    def collision_guard(
        start: PlanarPose,
        target: PlanarPose,
    ) -> bool | PlanarPose:
        if arm_presentation_ladder is not None:
            arm_presentation_ladder.stow_for_base_motion()
        collision_report.clear()
        world_reused = False
        if reuse_preflighted_empty_hand_world:
            reuse_world = getattr(
                candidate_pose_collision_provider,
                "reuse_preflighted_empty_hand_world",
                None,
            )
            if not callable(reuse_world):
                raise RuntimeError(
                    "preflighted collision world reuse is unavailable"
                )
            world_reused = bool(reuse_world())
        elif reuse_current_collision_world:
            reuse_world = getattr(
                candidate_pose_collision_provider,
                "reuse_current_world",
                None,
            )
            if not callable(reuse_world):
                raise RuntimeError("current collision world reuse is unavailable")
            world_reused = bool(reuse_world())
        if not world_reused:
            candidate_pose_collision_provider.invalidate_world()
        portal_snap = guard_report.get("portal_snap_private")
        portal_snap_active = bool(
            not endpoint_only_collision
            and isinstance(portal_snap, Mapping)
            and portal_snap.get("status") == "completed"
        )
        collision_check_endpoint = target
        if endpoint_only_collision:
            candidates = (target,)
        elif portal_snap_active:
            try:
                minimum_record = portal_snap["minimum_landing_pose_private"]
                minimum_landing = PlanarPose(
                    float(minimum_record["x"]),
                    float(minimum_record["y"]),
                    target.z,
                    float(minimum_record["yaw_rad"]),
                )
                egress_record = portal_snap["forward_egress_pose_private"]
                forward_egress = PlanarPose(
                    float(egress_record["x"]),
                    float(egress_record["y"]),
                    target.z,
                    float(egress_record["yaw_rad"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("portal forward-egress audit is invalid") from exc
            # Skip only the doorway interior.  The landing and the complete
            # next public forward sweep use the ordinary full-body guard.
            landing_transition = (
                ()
                if minimum_landing == target
                else tuple(candidate_collision_samples(minimum_landing, target))
            )
            candidates = (
                minimum_landing,
                *landing_transition,
                *tuple(candidate_collision_samples(target, forward_egress)),
            )
            collision_check_endpoint = forward_egress
        else:
            candidates = tuple(candidate_collision_samples(start, target))
        audits: list[dict[str, Any]] = []
        predicted_collision = False
        last_safe = start
        first_collision: PlanarPose | None = None
        check_many = getattr(candidate_pose_collision_provider, "check_many", None)
        batch_timing_private: Mapping[str, Any]
        batch_evaluated_candidate_count: int
        if callable(check_many):
            batch_result = check_many(candidates)
            collisions = tuple(bool(value) for value in batch_result.predicted_collisions)
            batch_audits = tuple(batch_result.audits)
            if len(collisions) != len(candidates) or len(batch_audits) != len(candidates):
                raise RuntimeError("candidate collision batch size changed")
            batch_timing_private = dict(batch_result.timing_private)
            batch_evaluated_candidate_count = len(collisions)
            for candidate, collision, audit in zip(
                candidates,
                collisions,
                batch_audits,
                strict=True,
            ):
                predicted_collision = collision
                audits.append(asdict(audit))
                if predicted_collision:
                    first_collision = candidate
                    break
                last_safe = candidate
            batch_query_used = True
        else:
            sequential_started = perf_counter()
            for candidate in candidates:
                predicted_collision = bool(candidate_pose_collision_provider(candidate))
                audit = candidate_pose_collision_provider.last_audit
                if audit is None:
                    raise RuntimeError(
                        "candidate collision provider produced no private audit"
                    )
                audits.append(asdict(audit))
                if predicted_collision:
                    first_collision = candidate
                    break
                last_safe = candidate
            batch_timing_private = {
                "total_seconds": perf_counter() - sequential_started,
                "candidate_count": len(audits),
                "compatibility_fallback": True,
            }
            batch_evaluated_candidate_count = len(audits)
            batch_query_used = False
        translation_m = math.hypot(target.x - start.x, target.y - start.y)
        safe_prefix_m = math.hypot(last_safe.x - start.x, last_safe.y - start.y)
        partial_prefix_available = bool(
            not portal_snap_active
            and predicted_collision
            and translation_m > 1.0e-12
            and safe_prefix_m
            >= RESTRICTED_KINEMATIC_INTERNAL_SAMPLE_M - 1.0e-9
        )
        scene = getattr(getattr(evaluator, "env", None), "scene", None)
        portal_crossing = (
            _symbolic_open_portal_crossing(scene, start, target)
            if predicted_collision
            and scene is not None
            and not endpoint_only_collision
            and not portal_snap_active
            else None
        )
        collision_report.update(
            guard_version="behavior2026.restricted-kinematic-live-curobo-guard/v8",
            reason=(
                "structural_portal_snap"
                if portal_snap_active and not predicted_collision
                else (
                    "open_portal_symbolic_crossing"
                    if portal_crossing is not None
                    else (
                        "navigation_collision_blocked"
                        if predicted_collision
                        else "navigation_collision_clear"
                    )
                )
            ),
            collision_free=not predicted_collision,
            open_portal_symbolic_crossing_private=portal_crossing,
            portal_snap_private=(dict(portal_snap) if portal_snap_active else None),
            collision_scope=(
                "endpoint_pose"
                if endpoint_only_collision
                else "portal_landing_and_egress"
                if portal_snap_active
                else "swept_segment"
            ),
            intermediate_collision_check_skipped=(
                endpoint_only_collision or portal_snap_active
            ),
            portal_forward_egress_checked=portal_snap_active,
            partial_safe_prefix_available=partial_prefix_available,
            safe_prefix_length_m=float(safe_prefix_m),
            safe_prefix_endpoint_world_x=float(last_safe.x),
            safe_prefix_endpoint_world_y=float(last_safe.y),
            safe_prefix_endpoint_world_yaw_rad=float(last_safe.yaw),
            first_collision_world_x=(
                None if first_collision is None else float(first_collision.x)
            ),
            first_collision_world_y=(
                None if first_collision is None else float(first_collision.y)
            ),
            first_collision_world_yaw_rad=(
                None if first_collision is None else float(first_collision.yaw)
            ),
            collision_checked_endpoint_world_x=float(collision_check_endpoint.x),
            collision_checked_endpoint_world_y=float(collision_check_endpoint.y),
            collision_checked_endpoint_world_yaw_rad=float(
                collision_check_endpoint.yaw
            ),
            candidate_samples_private=audits,
            candidate_sample_count=len(audits),
            batch_query_used=batch_query_used,
            batch_evaluated_candidate_count=batch_evaluated_candidate_count,
            preflighted_empty_hand_world_reused=(
                world_reused and reuse_preflighted_empty_hand_world
            ),
            current_collision_world_reused=(
                world_reused and reuse_current_collision_world
            ),
            preflighted_empty_hand_world_refresh_edges=(
                0 if reuse_preflighted_empty_hand_world else None
            ),
            curobo_timing_private=dict(batch_timing_private),
            segment_sample_fractions=tuple(
                (index + 1) / len(candidates) for index in range(len(candidates))
            ),
            maximum_translation_sample_spacing_m=(
                None
                if endpoint_only_collision
                else RESTRICTED_KINEMATIC_INTERNAL_SAMPLE_M
            ),
            simulator_steps=0,
            robot_moved=False,
            physics_stepped=False,
            task_truth_read=False,
        )
        if not predicted_collision or portal_crossing is not None:
            return True
        if partial_prefix_available:
            return last_safe
        return False

    def rollback_pose(last_good: PlanarPose) -> int:
        robot_pose = symbolic._get_robot_pose_from_2d_pose(
            (float(last_good.x), float(last_good.y), float(last_good.yaw))
        )
        # The legacy helper's holonomic z is root-relative; setters use world z.
        robot_pose[0][2] = float(last_good.z)
        evaluator.robot.set_position_orientation(*robot_pose)
        symbolic.sync_marker_kinematic_hold()
        return 0

    def apply_pose(target: PlanarPose) -> int:
        apply_started = perf_counter()
        held = symbolic._get_obj_in_hand()
        carrying = held is not None
        hand = str(getattr(symbolic, "arm", "")) if carrying else None
        if carrying and hand not in {"left", "right"}:
            raise RuntimeError("internal symbolic carry hand is unavailable")
        if carrying:
            _assert_marker_symbolic_grasp_held(
                symbolic,
                hand=hand,
                expected_obj=held,
                phase=MarkerSymbolicCarryPhase.BEFORE_RESTRICTED_KINEMATIC_BASE_STEP,
            )
        steps = 0
        settle_started = perf_counter()
        if post_apply_steps == 0:
            robot_pose = symbolic._get_robot_pose_from_2d_pose(
                (float(target.x), float(target.y), float(target.yaw))
            )
            # Apply the same world-height endpoint that the guards checked.
            robot_pose[0][2] = float(target.z)
            evaluator.robot.set_position_orientation(*robot_pose)
            symbolic.sync_marker_kinematic_hold()
        else:
            navigation = getattr(symbolic, "_navigate_to_pose", None)
            if not callable(navigation):
                raise RuntimeError(
                    "official Symbolic _navigate_to_pose is unavailable"
                )
            generator = iter(
                navigation((float(target.x), float(target.y), float(target.yaw)))
            )
            try:
                for _ in range(post_apply_steps):
                    try:
                        command = next(generator)
                    except StopIteration as exc:
                        raise RuntimeError(
                            "official Symbolic navigation ended before the bounded "
                            "settle window"
                        ) from exc
                    if command is not None:
                        raise RuntimeError(
                            "official Symbolic navigation returned an unexpected action"
                        )
                    symbolic.sync_marker_kinematic_hold()
                    if carrying:
                        hold = _marker_symbolic_closed_hold_action(
                            public_snapshot_provider(),
                            hand=hand,
                        )
                        control.command_carry_action(hold, active_hand=hand)
                    else:
                        control.hold()
                    steps += int(step_once())
                    symbolic.sync_marker_kinematic_hold()
                    if carrying:
                        _assert_marker_symbolic_grasp_held(
                            symbolic,
                            hand=hand,
                            expected_obj=held,
                            phase=(
                                MarkerSymbolicCarryPhase
                                .DURING_RESTRICTED_KINEMATIC_BASE_STEP
                            ),
                        )
            finally:
                close = getattr(generator, "close", None)
                if callable(close):
                    close()
        settle_seconds = perf_counter() - settle_started
        if carrying:
            _assert_marker_symbolic_grasp_held(
                symbolic,
                hand=hand,
                expected_obj=held,
                phase=MarkerSymbolicCarryPhase.AFTER_RESTRICTED_KINEMATIC_BASE_STEP,
            )
        collision_report["pose_application_timing_private"] = {
            "official_symbolic_bounded_settle_seconds": settle_seconds,
            "restricted_post_apply_steps": steps,
            "generic_symbolic_settle_steps_skipped": (
                8 - post_apply_steps
            ),
            "total_seconds": perf_counter() - apply_started,
        }
        return steps

    return FixedStepKinematicBaseExecutor(
        read_pose=read_pose,
        guard=guard,
        apply_pose=apply_pose,
        guard_diagnostic=lambda: dict(guard_report),
        collision_guard=collision_guard,
        collision_guard_diagnostic=lambda: dict(collision_report),
        rollback_pose=rollback_pose,
        move_distance_m=RESTRICTED_KINEMATIC_PUBLIC_MOVE_M,
        turn_angle_deg=turn_angle_deg,
        maximum_guard_translation_m=PORTAL_SNAP_MAX_TRANSLATION_M,
        allow_guard_translation_redirect=True,
    )


# Compatibility names retained for existing private probes.
_read_restricted_planar_pose = read_restricted_planar_pose
_restricted_local_guard = restricted_local_guard
_build_restricted_kinematic_executor = build_restricted_kinematic_executor
