"""Bounded, opt-in base placement before Oracle; never a navigation recovery."""

import math
import time
from copy import deepcopy
from itertools import chain, islice
from collections import Counter
from collections.abc import Mapping


def is_cuda_oom(exc):
    return (type(exc).__name__ in {"OutOfMemoryError", "CudaOutOfMemoryError"}
            or "cuda out of memory" in str(exc).casefold())


def presentation_failure_scope(exc):
    evidence = getattr(exc, "private_report", {})
    if is_cuda_oom(exc):
        return "RESOURCE_EXHAUSTED", "FRESH_RESOURCE_ATTEMPT"
    if evidence.get("budget_exhausted"):
        return "PREPARATION_DEADLINE_EXHAUSTED", "NONE"
    if evidence.get("xy_invariant_geometry_rejection") is True:
        return "PRESENTATION_GEOMETRY_INVARIANT", "NONE"
    return "NATIVE_PLANNER_FAILED", "DIFFERENT_BASE_POSE"


def candidate_admission(candidate, original_xy, *, room_lookup, static_check, expected_room=None):
    """Keep room evidence separate from Action map admission; unknown rejects."""
    row = {"accepted": False, "reason": None}
    if expected_room is not None and (not isinstance(expected_room, str) or not expected_room):
        raise ValueError('expected preparation room must be a nonempty string')
    if candidate != original_xy or expected_room is not None:
        lookups = {}
        # One missing label must not hide the other endpoint's query evidence.
        for role, point in (("origin", original_xy), ("candidate", candidate)):
            lookup = {"xy": list(point), "room": None, "status": "ROOM_LOOKUP_UNKNOWN"}
            try:
                room = room_lookup(point)
                if isinstance(room, str) and room:
                    lookup.update(room=room, status="RESOLVED")
            except Exception as exc:
                if is_cuda_oom(exc):
                    raise
                if str(exc) != "private position does not resolve to an official room":
                    lookup["status"] = "ROOM_QUERY_FAILED"
                lookup["error"] = f"{type(exc).__name__}: {exc}"
            lookups[role] = lookup
        origin_room, target_room = (lookups[role]["room"] for role in ("origin", "candidate"))
        row.update(origin_room=origin_room, candidate_room=target_room,
                   room_lookup_private=lookups)
        for reason in ("ROOM_QUERY_FAILED", "ROOM_LOOKUP_UNKNOWN"):
            if any(lookup["status"] == reason for lookup in lookups.values()):
                return {**row, "reason": reason}
        if expected_room is not None and origin_room != expected_room:
            return {**row, 'expected_room':expected_room, 'reason':'ROOM_ORIGIN_CONTRACT_MISMATCH'}
        if (expected_room or origin_room) != target_room:
            return {**row, "reason": "ROOM_MISMATCH"}
    try:
        guard = dict(static_check(candidate))
    except Exception as exc:
        if is_cuda_oom(exc):
            raise
        return {**row, "reason": "STATIC_MAP_QUERY_FAILED", "error": f"{type(exc).__name__}: {exc}"}
    accepted = guard.get("traversable") is True
    return {**row, "accepted": accepted, "local_guard_private": guard,
            "reason": "ACCEPTED" if accepted else "STATIC_MAP_BLOCKED" if guard.get("reason") == "navigation_map_blocked" else "STATIC_MAP_QUERY_FAILED"}


def pose_readback_drift(before, after):
    """Validate numeric state; pose differences are Symbolic diagnostics, not gates."""
    vectors = [tuple(float(x) for x in vector) for pose in (before, after) for vector in pose]
    for vector, length in zip(vectors, (3, 4, 3, 4)):
        if len(vector) != length or not all(math.isfinite(x) for x in vector):
            raise RuntimeError("invalid/nonfinite pose readback")
    p0, q0, p1, q1 = vectors
    norms = math.hypot(*q0), math.hypot(*q1)
    if not all(math.isfinite(norm) and norm > 0 for norm in norms):
        raise RuntimeError("zero quaternion in pose readback")
    cosine = abs(sum((a / norms[0]) * (b / norms[1]) for a, b in zip(q0, q1)))
    delta = [b - a for a, b in zip(p0, p1)]
    return dict(position_delta_xyz_m=delta, position_error_m=math.hypot(*delta),
                orientation_error_rad=2 * math.acos(min(1.0, cosine)), diagnostic_only=True)


def nearby_positions(xy, original_yaw, *, admissible=None):
    """Near first on the body axis; at each distance prefer the admissible side."""
    positions = tuple(
        (xy[0] + radius * math.cos(original_yaw + angle),
         xy[1] + radius * math.sin(original_yaw + angle))
        for radius in (0.25, 0.5, 1.0, 1.5, 2.0)
        for angles in ((math.pi, 0.0), (math.pi / 2, -math.pi / 2))
        for angle in angles
    )
    if admissible is None:
        return positions
    # Stable pair ordering preserves backward-first only when evidence is tied.
    return tuple(position for index in range(0, len(positions), 2)
                 for position in sorted(positions[index:index + 2], key=lambda xy: not admissible(xy)))


def measured_base_admission(scene, robot):
    """Use the public movement guard at the actual pose, without source snapping."""
    from .private_traversability_route import check_official_local_segment

    position, orientation = robot.get_position_orientation()
    x, y, z, w = (float(value) for value in orientation)
    yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    xy = tuple(float(value) for value in position[:2])
    guard = check_official_local_segment(
        scene=scene, robot=robot, start_xy=xy, target_xy=xy,
        robot_z=float(position[2]), start_yaw_rad=yaw, target_yaw_rad=yaw,
        allow_bounded_source_snap=False, allow_test_fallback=False,
    )
    return {**guard.private_report, "traversable": bool(guard.traversable),
            "measured_xy": list(xy), "measured_yaw_rad": yaw}


def search_initial_pose(*, original_xy, candidates, admissible, restore,
                        place, prepare, verify, deadline, clock=time.monotonic,
                        initial_state_valid=lambda: True, measured_base_admission=None):
    """Restore before every trial and on failure; preserve the accepted state."""
    attempts = []
    report = dict(status="FAIL", attempts=attempts, initial_pose_search=True,
                  maximum_alternatives=20, deadline_exhausted=False)
    try:
        for index, xy in enumerate(chain((original_xy,), islice(candidates, 20))):
            if clock() >= deadline:
                report["deadline_exhausted"] = True
                break
            restore()
            attempt = dict(index=index, xy=list(xy))
            attempts.append(attempt)
            admission = admissible(xy)
            accepted = admission.get("accepted") is True if isinstance(admission, Mapping) else bool(admission)
            if isinstance(admission, Mapping):
                attempt["candidate_admission_private"] = dict(admission)
            if not accepted:
                reason = admission.get("reason", "STATIC_MAP_BLOCKED") if isinstance(admission, Mapping) else "STATIC_MAP_BLOCKED"
                attempt.update(status="room_rejected" if reason.startswith("ROOM_") else "static_rejected",
                               failure_type=reason, retry_scope="DIFFERENT_BASE_POSE")
                continue
            place(xy)
            def base_is_open(stage):
                if measured_base_admission is None:
                    return True
                admission = dict(measured_base_admission())
                attempt.setdefault("base_admission_private", {})[stage] = admission
                if admission.get("traversable") is not True:
                    attempt.update(status="static_rejected", rejection_stage=stage,
                                   failure_type="STATIC_MAP_BLOCKED" if admission.get("reason") == "navigation_map_blocked" else "STATIC_MAP_QUERY_FAILED",
                                   retry_scope="DIFFERENT_BASE_POSE")
                    return False
                return True
            if not base_is_open("before_presentation"):
                continue
            if not initial_state_valid():
                attempt.update(status="initial_semantics_rejected", rejection_stage="before_presentation")
                continue
            try:
                result = dict(prepare(deadline))
            except Exception as exc:
                attempt.update(status="presentation_failed", error=str(exc),
                               presentation=getattr(exc, "private_report", {}))
                failure_type, retry_scope = presentation_failure_scope(exc)
                attempt.update(failure_type=failure_type, retry_scope=retry_scope)
                if failure_type == "RESOURCE_EXHAUSTED" or failure_type == "PRESENTATION_GEOMETRY_INVARIANT":
                    report.update(failure_type=failure_type, retry_scope=retry_scope)
                    exc.private_report = report
                    raise
                if failure_type == "PREPARATION_DEADLINE_EXHAUSTED" or clock() >= deadline:
                    report["deadline_exhausted"] = True
                    attempt["status"] = "budget_exhausted"
                    break
                continue
            # Verification failure is not another pose candidate: stop and restore.
            if clock() >= deadline:
                report["deadline_exhausted"] = True
                attempt.update(status="budget_exhausted", failure_type="PREPARATION_DEADLINE_EXHAUSTED", retry_scope="NONE")
                break
            verify()
            if not base_is_open("after_presentation"):
                continue
            if not initial_state_valid():
                attempt.update(status="initial_semantics_rejected", rejection_stage="after_presentation")
                continue
            attempt["status"] = "accepted"
            report.update(status="PASS", selected_xy=list(xy), selected_index=index,
                          presentation=result)
            return report
        if report["deadline_exhausted"]:
            report.update(failure_type="PREPARATION_DEADLINE_EXHAUSTED", retry_scope="NONE")
            message = "nearby initial presentation deadline exhausted"
        else:
            alternatives = attempts[1:]
            room_failed = bool(alternatives) and all(
                a.get("failure_type") in {"ROOM_QUERY_FAILED", "ROOM_LOOKUP_UNKNOWN"}
                for a in alternatives
            )
            report.update(failure_type="ROOM_LOOKUP_UNAVAILABLE" if room_failed else "CANDIDATES_EXHAUSTED", retry_scope="NONE")
            message = "nearby room lookup unavailable" if room_failed else "nearby initial presentation candidates exhausted"
        error = RuntimeError(message)
        error.private_report = report
        raise error
    except BaseException as exc:
        restore()
        report["candidate_failure_counts_private"] = dict(Counter(
            a.get("failure_type", a.get("status", "UNKNOWN")) for a in attempts
        ))
        if is_cuda_oom(exc):
            report.update(failure_type="RESOURCE_EXHAUSTED", retry_scope="FRESH_RESOURCE_ATTEMPT")
            exc.private_report = report
        if not hasattr(exc, "private_report"):
            exc.private_report = report
        raise


def prepare_nearby_initial_pose(evaluator, *, initial_state_valid=lambda: True,
                                enable_nearby_search=True, deadline_monotonic=None, expected_room=None):
    import omnigibson as og
    import torch
    from .moderate_inward_gripper_presentation import apply_untucked_inward_gripper_presentation
    from .private_traversability_route import check_official_local_segment
    from .run_b1_marker_symbolic_rollout import _private_room_at_xy

    robot = evaluator.robot
    scene = evaluator.env.scene
    position, orientation = (value.clone() for value in robot.get_position_orientation())
    requested_position = position.clone()
    xy = tuple(float(value) for value in position[:2])
    qx, qy, qz, qw = (float(value) for value in orientation)
    yaw = math.atan2(2 * (qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))
    checkpoint = deepcopy(og.sim.dump_state(serialized=False))
    protected = []
    for obj in (scene.objects.values() if isinstance(scene.objects, Mapping) else scene.objects):
        if obj is robot:
            continue
        pose = tuple(value.clone() for value in obj.get_position_orientation())
        joints = obj.get_joint_positions().clone() if obj.n_joints else None
        protected.append((obj, pose, joints))
    pose_readbacks = {}

    def restore():
        try:
            og.sim.load_state(deepcopy(checkpoint), serialized=False)
            og.sim.render()
        except Exception as exc:
            exc.session_invalid = True
            raise

    def place(candidate):
        nonlocal requested_position
        target = position.clone()
        target[:2] = torch.as_tensor(candidate, device=target.device, dtype=target.dtype)
        requested_position = target.clone()
        robot.set_position_orientation(target, orientation)
        og.sim.render()

    def verify():
        actual_position, actual_orientation = robot.get_position_orientation()
        pose_readbacks["robot"] = pose_readback_drift((requested_position, orientation), (actual_position, actual_orientation))
        current_objects = scene.objects.values() if isinstance(scene.objects, Mapping) else scene.objects
        identities = {id(obj) for obj in current_objects}
        object_reports = []
        for obj, before, joints in protected:
            if id(obj) not in identities:
                raise RuntimeError(f"initial search lost object: {obj.name}")
            drift = pose_readback_drift(before, obj.get_position_orientation())
            joint_error = 0.0
            if joints is not None:
                actual_joints = obj.get_joint_positions()
                if joints.shape != actual_joints.shape or not torch.isfinite(actual_joints).all() or not torch.isfinite(joints).all():
                    raise RuntimeError(f"invalid/nonfinite joints: {obj.name}")
                joint_error = max((abs(float(a-b)) for a, b in zip(joints, actual_joints)), default=0.0)
            if drift["position_error_m"] or drift["orientation_error_rad"] or joint_error:
                object_reports.append(dict(object_name=obj.name, **drift, max_joint_error_rad=joint_error))
        pose_readbacks["objects_with_drift"] = object_reports
        pose_readbacks["objects_checked"] = len(protected)

    static_guard_by_xy = {}

    def open_base(candidate):
        guard = check_official_local_segment(
            scene=scene, robot=robot, start_xy=candidate, target_xy=candidate,
            robot_z=float(position[2]), start_yaw_rad=yaw, target_yaw_rad=yaw,
            allow_bounded_source_snap=False, allow_test_fallback=False,
        )
        static_guard_by_xy[candidate] = dict(guard.private_report)
        return {**guard.private_report, "traversable": bool(guard.traversable)}

    admission_by_xy = {}
    room_by_xy = {}

    def room_lookup(point):
        if point not in room_by_xy:
            try:
                room_by_xy[point] = _private_room_at_xy(scene, point)
            except Exception as exc:
                room_by_xy[point] = exc
        value = room_by_xy[point]
        if isinstance(value, Exception):
            raise value
        return value

    def candidates():
        yield from nearby_positions(xy, yaw, admissible=lambda point: admissible(point)["accepted"])

    def admissible(candidate):
        if candidate in admission_by_xy:
            return admission_by_xy[candidate]
        row = candidate_admission(candidate, xy, room_lookup=room_lookup, static_check=open_base, expected_room=expected_room)
        admission_by_xy[candidate] = row
        return row

    def static_admission_reports():
        return [{"xy": list(candidate), **row} for candidate, row in admission_by_xy.items()]

    try:
        report = search_initial_pose(
            original_xy=xy, candidates=candidates() if enable_nearby_search else (),
            admissible=admissible, restore=restore, place=place,
            prepare=lambda deadline: apply_untucked_inward_gripper_presentation(
                evaluator, deadline_monotonic=deadline), verify=verify,
            deadline=time.monotonic() + 600.0 if deadline_monotonic is None else deadline_monotonic,
            initial_state_valid=initial_state_valid,
            measured_base_admission=lambda: measured_base_admission(scene, robot),
        )
    except BaseException as exc:
        if hasattr(exc, "private_report"):
            exc.private_report["candidate_static_admission_private"] = static_admission_reports()
        raise
    report.update(original_position=position.tolist(), original_orientation=orientation.tolist(),
                  actual_position=robot.get_position_orientation()[0].tolist(),
                  actual_orientation=robot.get_position_orientation()[1].tolist(),
                  actual_joint_positions=robot.get_joint_positions().tolist(),
                  direction_source="radius_first_body_axes_static_admission_v3",
                  candidate_static_admission_private=static_admission_reports(),
                  original_yaw_rad=yaw,
                  time_budget_seconds=600,
                  pose_admission_policy="finite_readback_required_pose_drift_diagnostic_only",
                  pose_readback_private=pose_readbacks,
                  public_navigation_actions=0)
    presentation = report.pop("presentation")
    return {**presentation, "nearby_initial_pose": report}
