"""Collision-aware, presentation-only inward pose for both R1Pro grippers.

The live end-effector position is shifted 12--14 cm toward the robot centre in
the body frame while preserving its orientation.  Official Starter / CuRobo
IK chooses the feasible target.  This changes neither the Symbolic grasp
primitive nor its official two-finger midpoint rule.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from time import monotonic
from typing import Any, Mapping

import numpy as np

from behavior_hl.embodied.evaluator_bridge.r1pro_live_safe_carry_binding import (
    read_r1pro_live_robot_carry_state,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_live_safe_carry_candidate import (
    LIVE_POSE_FAMILY_SOURCE,
    LiveEEFPoseCandidate,
    LiveRobotCarryState,
)


MODERATE_INWARD_GRIPPER_PRESENTATION_VERSION = (
    "behavior2026.moderate-inward-gripper-presentation/v4"
)
INWARD_SHIFT_CANDIDATES_M = (0.12, 0.13, 0.14)
PLANNER_SHIFT_ORDER_M = (0.12, 0.13, 0.14)
MINIMUM_CENTER_CLEARANCE_M = 0.12
MINIMUM_EEF_SEPARATION_M = 0.24
MAXIMUM_TORSO_JOINT_DELTA_RAD = 0.50


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    item = value
    for method in ("detach", "cpu"):
        callback = getattr(item, method, None)
        if callable(callback):
            item = callback()
    numpy = getattr(item, "numpy", None)
    if callable(numpy):
        item = numpy()
    result = np.asarray(item, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise RuntimeError(f"{name} must contain {size} finite values")
    return result


def _rotation_matrix(quaternion_xyzw: Any) -> np.ndarray:
    x, y, z, w = _vector(quaternion_xyzw, 4, "body quaternion")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-8:
        raise RuntimeError("body quaternion has zero norm")
    x, y, z, w = (value / norm for value in (x, y, z, w))
    return np.asarray(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        ),
        dtype=np.float64,
    )


def eef_position_in_body_frame(state: LiveRobotCarryState) -> np.ndarray:
    """Return one live EEF position relative to the body."""

    rotation = _rotation_matrix(state.body_world_quaternion_xyzw)
    return rotation.T @ (
        np.asarray(state.eef_world_position, dtype=np.float64)
        - np.asarray(state.body_world_position, dtype=np.float64)
    )


class PresentationGeometryInvariantError(RuntimeError):
    """Body-frame target geometry unchanged by base XY relocation."""


def moderate_inward_pose_family(
    state: LiveRobotCarryState,
    *,
    shifts: tuple[float, ...] = INWARD_SHIFT_CANDIDATES_M,
) -> tuple[LiveEEFPoseCandidate, ...]:
    """Build modest inward candidates from the current live pose."""

    if not isinstance(state, LiveRobotCarryState):
        raise TypeError("state must be LiveRobotCarryState")
    if any(shift not in INWARD_SHIFT_CANDIDATES_M for shift in shifts):
        raise ValueError("shifts must be drawn from INWARD_SHIFT_CANDIDATES_M")
    relative = eef_position_in_body_frame(state)
    side = float(np.sign(relative[1]))
    if side == 0.0:
        raise RuntimeError("live EEF lies on the body centre line")
    rotation = _rotation_matrix(state.body_world_quaternion_xyzw)
    body = np.asarray(state.body_world_position, dtype=np.float64)
    candidates: list[LiveEEFPoseCandidate] = []
    # Evidence-only detail so a hard failure records the exact live geometry
    # it rejected; thresholds and rejection behavior are unchanged.
    live_detail = (
        f"live eef_body_y={float(relative[1]):+.4f} m, side={side:+.0f}, "
        f"arm_qpos_rad={[round(float(v), 4) for v in state.arm_qpos]}, "
        f"state_token={state.state_token[:20]}"
    )
    for shift in shifts:
        target_body = relative.copy()
        target_body[1] -= side * shift
        if float(np.sign(target_body[1])) != side:
            raise PresentationGeometryInvariantError(
                "inward candidate crossed the body centre line "
                f"(shift={shift} m; {live_detail})"
            )
        if abs(float(target_body[1])) < MINIMUM_CENTER_CLEARANCE_M:
            raise PresentationGeometryInvariantError(
                "inward candidate would obstruct the centre view "
                f"(shift={shift} m -> target_y={float(target_body[1]):+.4f} m "
                f"< clearance {MINIMUM_CENTER_CLEARANCE_M} m; {live_detail})"
            )
        target_world = body + rotation @ target_body
        candidates.append(
            LiveEEFPoseCandidate(
                candidate_id=(
                    f"{state.state_token[:12]}:inward_{round(shift * 1000):03d}mm"
                ),
                state_token=state.state_token,
                eef_world_position=tuple(float(value) for value in target_world),
                eef_world_quaternion_xyzw=state.eef_world_quaternion_xyzw,
                source=LIVE_POSE_FAMILY_SOURCE,
            )
        )
    return tuple(candidates)


def _state_report(state: LiveRobotCarryState) -> dict[str, Any]:
    return {
        "state_token": state.state_token,
        "body_world_position_m": list(state.body_world_position),
        "body_world_orientation_xyzw": list(state.body_world_quaternion_xyzw),
        "eef_world_position_m": list(state.eef_world_position),
        "eef_world_orientation_xyzw": list(state.eef_world_quaternion_xyzw),
        "eef_body_position_m": eef_position_in_body_frame(state).tolist(),
        "arm_qpos_rad": list(state.arm_qpos),
    }


@contextmanager
def _observe_plan_batch(generator: Any, records: list[dict[str, Any]]):
    """Observe the official aggregate without changing calls or return objects."""
    try:
        original = generator.plan_batch
        had_override = "plan_batch" in vars(generator)
        override = vars(generator).get("plan_batch")

        def observed(*args, **kwargs):
            returned = original(*args, **kwargs)
            record = {"scope": "official_batch_aggregate",
                      "start_state_collision_checked": False,
                      **dict.fromkeys(("status", "valid_query", "success", "attempts",
                                       "trajopt_attempts", "solve_time", "ik_time", "trajopt_time"))}
            try:
                result = returned[0]
                status = getattr(result, "status", None)
                status = getattr(status, "name", status)
                record["status"] = status[:200] if isinstance(status, str) else None
                valid = getattr(result, "valid_query", None)
                record["valid_query"] = valid if isinstance(valid, bool) else None
                success = getattr(result, "success", None)
                record["success"] = None
                if success is not None and int(success.numel()) <= 16:
                    record["success"] = _vector(
                        success, int(success.numel()), "planner success"
                    ).astype(bool).tolist()
                for name in ("attempts", "trajopt_attempts", "solve_time",
                             "ik_time", "trajopt_time"):
                    value = getattr(result, name, None)
                    record[name] = value if isinstance(value, (int, float)) and math.isfinite(value) else None
            except Exception as exc:
                record["diagnostic_error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
            return returned

        generator.plan_batch = observed
    except Exception as exc:
        records.append({"diagnostic_error": f"{type(exc).__name__}: {exc}"})
        yield
        return
    try:
        yield
    finally:
        try:
            if had_override:
                generator.plan_batch = override
            else:
                del generator.plan_batch
        except Exception as exc:
            records.append({"diagnostic_restore_error": f"{type(exc).__name__}: {exc}"})


def apply_untucked_inward_gripper_presentation(
    evaluator: Any,
    deadline_monotonic: float | None = None,
) -> Mapping[str, Any]:
    """Apply the official untucked preamble before inward arm presentation."""

    import omnigibson as og
    import torch as th

    from behavior_hl.embodied.evaluator_bridge.predeclared_camera_config import (
        R1PRO_OFFICIAL_UNTUCKED_LEFT_ARM,
        R1PRO_OFFICIAL_UNTUCKED_RIGHT_ARM,
    )

    robot = evaluator.robot
    posed = robot.get_joint_positions().clone()
    # Write the untucked constants through the exact positional slices the
    # load-time use_wrist_evidence_reset_pose flag uses on the 28-joint
    # config vector ([10:17] left, [17:24] right).  arm_control_idx orders
    # joints by arm_joint_names, which does NOT match the config slice order;
    # writing the constants through it scrambles joints past their limits
    # (first flight attempt 8, 2026-08-13 17:46).
    if int(posed.numel()) != 28:
        raise RuntimeError("R1Pro joint vector length drifted")
    posed[10:17] = th.as_tensor(
        R1PRO_OFFICIAL_UNTUCKED_LEFT_ARM, dtype=posed.dtype, device=posed.device
    )
    posed[17:24] = th.as_tensor(
        R1PRO_OFFICIAL_UNTUCKED_RIGHT_ARM, dtype=posed.dtype, device=posed.device
    )
    robot.set_joint_positions(posed, drive=False)
    set_velocities = getattr(robot, "set_joint_velocities", None)
    if callable(set_velocities):
        set_velocities(th.zeros_like(posed), drive=False)
    og.sim.render()
    if deadline_monotonic is None:
        return apply_moderate_inward_gripper_presentation(evaluator)
    return apply_moderate_inward_gripper_presentation(
        evaluator, deadline_monotonic=deadline_monotonic
    )


def apply_moderate_inward_gripper_presentation(
    evaluator: Any,
    deadline_monotonic: float | None = None,
) -> Mapping[str, Any]:
    diagnostic: dict[str, Any] = {
        "status": "FAIL", "before_by_hand": {}, "full_joint_reference": None,
        "planner_results": [], "valid_query_is_not_start_collision_evidence": True,
    }
    try:
        if deadline_monotonic is None:
            return _apply_moderate_inward_gripper_presentation(
                evaluator, diagnostic
            )
        return _apply_moderate_inward_gripper_presentation(
            evaluator, diagnostic, deadline_monotonic=deadline_monotonic
        )
    except Exception as exc:
        exc.private_report = diagnostic
        raise


def _apply_moderate_inward_gripper_presentation(
    evaluator: Any,
    diagnostic: dict[str, Any],
    deadline_monotonic: float | None = None,
) -> Mapping[str, Any]:
    """Plan both inward wrist targets together and apply one coherent state."""

    import omnigibson as og
    import torch as th
    from omnigibson.action_primitives.curobo import (
        CuRoboEmbodimentSelection,
    )

    from behavior_hl.embodied.evaluator_bridge.run_b1_marker_symbolic_rollout import (
        _build_full_motion_safety_starter,
    )

    robot = evaluator.robot
    starter = _build_full_motion_safety_starter(evaluator)
    before_by_hand: dict[str, LiveRobotCarryState] = {}
    for hand in ("left", "right"):
        starter._selected_arm = hand
        before = read_r1pro_live_robot_carry_state(starter, hand)
        before_by_hand[hand] = before
        diagnostic["before_by_hand"][hand] = _state_report(before)

    joint_reference = robot.get_joint_positions()
    reference_cpu = _vector(
        joint_reference,
        int(joint_reference.numel()),
        "full joint reference",
    )
    diagnostic["full_joint_reference"] = reference_cpu.tolist()
    trunk_indices = _vector(
        robot.trunk_control_idx,
        4,
        "trunk control indices",
    ).astype(np.int64)
    planner_attempts: list[dict[str, Any]] = []
    diagnostic["planner_attempts"] = planner_attempts
    selected_shift: float | None = None
    selected_target: Any | None = None
    selected_waypoint_count = 0
    planner_call_count = 0
    for shift in PLANNER_SHIFT_ORDER_M:
        target_pos: dict[str, Any] = {}
        target_quat: dict[str, Any] = {}
        candidate_rejection: tuple[str, Exception] | None = None
        for hand in ("left", "right"):
            try:
                candidate = moderate_inward_pose_family(
                    before_by_hand[hand], shifts=(shift,)
                )[0]
            except Exception as exc:
                from .nearby_initial_pose import is_cuda_oom
                if is_cuda_oom(exc):
                    diagnostic.update(failure_type="RESOURCE_EXHAUSTED", retry_scope="FRESH_RESOURCE_ATTEMPT")
                    raise
                candidate_rejection = (hand, exc)
                break
            position_reference, quaternion_reference = robot.get_eef_pose(
                arm=hand
            )
            link_name = robot.eef_link_names[hand]
            target_pos[link_name] = position_reference.new_tensor(
                candidate.eef_world_position
            )
            target_quat[link_name] = quaternion_reference.new_tensor(
                candidate.eef_world_quaternion_xyzw
            )
        if candidate_rejection is not None:
            hand, exc = candidate_rejection
            planner_attempts.append(
                {
                    "shift_m": shift,
                    "status": "rejected",
                    "rejection_stage": "candidate_geometry",
                    "xy_invariant": isinstance(exc, PresentationGeometryInvariantError),
                    "hand": hand,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        starter._selected_arm = "left"
        skip_obstacle_update = planner_call_count > 0
        results: list[dict[str, Any]] = []
        diagnostic["planner_results"].append({"shift_m": shift, "results": results})
        planner_started_monotonic = None
        if deadline_monotonic is not None:
            planner_started_monotonic = monotonic()
            if planner_started_monotonic >= deadline_monotonic:
                diagnostic["budget_exhausted"] = True
                diagnostic["budget_exhausted_reason"] = (
                    "presentation_planner_deadline_exhausted"
                )
                diagnostic["budget_exhausted_detail"] = {
                    "deadline_monotonic": deadline_monotonic,
                    "observed_monotonic": planner_started_monotonic,
                    "remaining_s": max(0.0, deadline_monotonic - planner_started_monotonic),
                    "shift_m": shift,
                    "planner_call_count": planner_call_count,
                    "native_planner_called": False,
                }
                raise RuntimeError("presentation_planner_budget_exhausted")
        planner_finished_monotonic = None
        try:
            with _observe_plan_batch(getattr(starter, "_motion_generator", None), results):
                try:
                    planner_call_count += 1
                    trajectory = starter._plan_joint_motion(
                        target_pos=target_pos,
                        target_quat=target_quat,
                        embodiment_selection=CuRoboEmbodimentSelection.ARM,
                        skip_obstacle_update=skip_obstacle_update,
                    )
                finally:
                    if deadline_monotonic is not None:
                        planner_finished_monotonic = monotonic()
                        if planner_finished_monotonic > deadline_monotonic:
                            overrun_s = (
                                planner_finished_monotonic - deadline_monotonic
                            )
                            diagnostic["planner_results"][-1].update(
                                {
                                    "deadline_overrun_s": overrun_s,
                                    "native_planner_completed_after_deadline": True,
                                }
                            )
                            diagnostic.setdefault(
                                "planner_deadline_overruns", []
                            ).append(
                                {"shift_m": shift, "overrun_s": overrun_s}
                            )
            if len(trajectory) == 0:
                raise RuntimeError("official planner returned no waypoints")
            candidate_target = trajectory[-1]
            candidate_cpu = _vector(
                candidate_target,
                len(reference_cpu),
                "official paired planner target",
            )
            maximum_torso_delta = float(
                np.max(
                    np.abs(
                        candidate_cpu[trunk_indices]
                        - reference_cpu[trunk_indices]
                    )
                )
            )
            if maximum_torso_delta > MAXIMUM_TORSO_JOINT_DELTA_RAD:
                raise RuntimeError(
                    "paired target changed torso by "
                    f"{maximum_torso_delta:.4f} rad"
                )
        except Exception as exc:
            from .nearby_initial_pose import is_cuda_oom
            if is_cuda_oom(exc):
                diagnostic.update(failure_type="RESOURCE_EXHAUSTED", retry_scope="FRESH_RESOURCE_ATTEMPT")
                raise
            planner_attempts.append(
                {
                    "shift_m": shift,
                    "status": "rejected",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        selected_shift = shift
        selected_target = candidate_target
        selected_waypoint_count = len(trajectory)
        planner_attempts.append(
            {
                "shift_m": shift,
                "status": "selected",
                "waypoint_count": selected_waypoint_count,
                "maximum_torso_joint_delta_rad": maximum_torso_delta,
            }
        )
        break
    if selected_target is None or selected_shift is None:
        if (len(planner_attempts) == len(PLANNER_SHIFT_ORDER_M)
                and all(row.get("xy_invariant") is True for row in planner_attempts)):
            diagnostic.update(xy_invariant_geometry_rejection=True,
                              failure_type="PRESENTATION_GEOMETRY_INVARIANT", retry_scope="NONE")
        raise RuntimeError(f"all paired inward plans failed: {planner_attempts}")

    target = th.as_tensor(
        selected_target,
        dtype=joint_reference.dtype,
        device=joint_reference.device,
    )
    robot.set_joint_positions(target, drive=False)
    set_velocities = getattr(robot, "set_joint_velocities", None)
    if callable(set_velocities):
        set_velocities(th.zeros_like(target), drive=False)
    og.sim.render()

    hands: list[dict[str, Any]] = []
    for hand in ("left", "right"):
        starter._selected_arm = hand
        before = before_by_hand[hand]
        after = read_r1pro_live_robot_carry_state(starter, hand)
        before_body = eef_position_in_body_frame(before)
        after_body = eef_position_in_body_frame(after)
        measured_shift = abs(float(before_body[1])) - abs(float(after_body[1]))
        if not 0.105 <= measured_shift <= 0.155:
            raise RuntimeError(
                f"{hand} gripper inward displacement was {measured_shift:.4f} m"
            )
        if float(np.sign(before_body[1])) != float(np.sign(after_body[1])):
            raise RuntimeError(f"{hand} gripper crossed the body centre line")
        if abs(float(after_body[1])) < MINIMUM_CENTER_CLEARANCE_M:
            raise RuntimeError(f"{hand} gripper obstructs the centre view")
        hands.append(
            {
                "hand": hand,
                "before": _state_report(before),
                "after": _state_report(after),
                "selected_target_displacement_m": selected_shift,
                "measured_inward_displacement_m": measured_shift,
            }
        )

    left = np.asarray(hands[0]["after"]["eef_world_position_m"])
    right = np.asarray(hands[1]["after"]["eef_world_position_m"])
    separation = float(np.linalg.norm(left - right))
    if separation < MINIMUM_EEF_SEPARATION_M:
        raise RuntimeError("inward grippers lack conservative separation")
    report = {
        "version": MODERATE_INWARD_GRIPPER_PRESENTATION_VERSION,
        "status": "PASS",
        "presentation_only": True,
        "direct_kinematic_joint_application": True,
        "official_collision_aware_paired_motion_plan": True,
        "official_planner_method": "_plan_joint_motion",
        "official_planner_embodiment": "arm",
        "camera_conditioned": False,
        "symbolic_rule_changed": False,
        "official_two_finger_midpoint_preserved": True,
        "candidate_shifts_m": list(INWARD_SHIFT_CANDIDATES_M),
        "planner_shift_order_m": list(PLANNER_SHIFT_ORDER_M),
        "selected_shift_m": selected_shift,
        "selected_waypoint_count": selected_waypoint_count,
        "planner_attempts": planner_attempts,
        "minimum_center_clearance_m": MINIMUM_CENTER_CLEARANCE_M,
        "maximum_torso_joint_delta_rad": MAXIMUM_TORSO_JOINT_DELTA_RAD,
        "eef_separation_m": separation,
        "hands": hands,
    }
    if deadline_monotonic is not None:
        report["deadline_monotonic"] = deadline_monotonic
        report["planner_deadline_overruns"] = list(
            diagnostic.get("planner_deadline_overruns", [])
        )
    return report


__all__ = [
    "INWARD_SHIFT_CANDIDATES_M",
    "MINIMUM_CENTER_CLEARANCE_M",
    "MINIMUM_EEF_SEPARATION_M",
    "MAXIMUM_TORSO_JOINT_DELTA_RAD",
    "PLANNER_SHIFT_ORDER_M",
    "MODERATE_INWARD_GRIPPER_PRESENTATION_VERSION",
    "apply_untucked_inward_gripper_presentation",
    "apply_moderate_inward_gripper_presentation",
    "eef_position_in_body_frame",
    "moderate_inward_pose_family",
]
