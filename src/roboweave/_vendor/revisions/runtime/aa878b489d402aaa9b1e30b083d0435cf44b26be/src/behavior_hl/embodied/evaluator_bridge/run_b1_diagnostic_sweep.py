"""Single-scene diagnostic sweep for the next B1 grasp experiments.

This is an isolated research runner.  It adds a private, no-step candidate IK
probe and then reuses the production Oracle/Qwen phases with one in-memory
reset between phases.  The public policy still receives only RGB-D/proprio
and the same high-level action registry.
"""

from __future__ import annotations

from dataclasses import asdict
import math
import json
import os
from pathlib import Path
import random
import traceback
from typing import Any, Callable, Mapping

import numpy as np

from behavior_hl.embodied.evaluator_bridge.run_b1_single_load import _record_stage
from behavior_hl.embodied.evaluator_bridge.r1pro_private_oracle_geometry import (
    RadialStation,
    local_support_placements_toward_robot,
    local_planar_error,
    rank_radial_stations_by_measured_cost,
    radial_station_candidates,
    table_edge_placements,
    tennis_ball_center_grasp_variants,
    wrap_angle,
    yaw_from_xy_quaternion,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_live_safe_carry_binding import (
    build_r1pro_live_safe_carry_proposal_bindings,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_grasp_candidates import (
    GraspCandidate,
    generate_grasp_candidates,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_strategy_sweep import (
    _oracle_phase,
    _preposition_variant,
    _qwen_phase,
    _reset_same_scene,
    _run_official_primitive,
    _scope_object,
    _write_json,
)


def _candidate_probe(
    *,
    evaluator: Any,
    bindings: Any,
    run_dir: Path,
    stages: Path,
    phase: str = "ik_probe",
) -> dict[str, Any]:
    """Call the official collision-aware IK once per live carry candidate.

    The probe performs no simulator step and never reads task/object scope.  It
    deliberately records per-candidate outcomes instead of collapsing them to
    ``all live collision-aware IK candidates failed``.
    """

    _record_stage(stages, "IK_PROBE_START", phase=phase)
    output_dir = run_dir / "strategies" / phase
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "method": "private_live_safe_carry_candidate_ik_probe",
        "phase": phase,
        "status": "FAIL",
        "simulator_steps": 0,
        "hand": "right",
        "candidates": [],
    }
    try:
        starter = bindings._starter_for(evaluator)
        callbacks = build_r1pro_live_safe_carry_proposal_bindings(
            starter=starter,
            hand="right",
        )
        state = callbacks.live_state_provider(starter, "right")
        poses = tuple(callbacks.candidate_pose_family(state))
        solver = starter._ik_solver_cartesian_to_joint_space
        lower = np.asarray(state.arm_lower_limits, dtype=np.float64)
        upper = np.asarray(state.arm_upper_limits, dtype=np.float64)
        current = np.asarray(state.arm_qpos, dtype=np.float64)
        span = upper - lower
        for index, pose in enumerate(poses):
            item: dict[str, Any] = {
                "index": index,
                "candidate_id": pose.candidate_id,
                "eef_world_position": list(pose.eef_world_position),
                "eef_world_quaternion_xyzw": list(
                    pose.eef_world_quaternion_xyzw
                ),
                "eef_displacement_from_live_m": float(
                    np.linalg.norm(
                        np.asarray(pose.eef_world_position)
                        - np.asarray(state.eef_world_position)
                    )
                ),
                "body_to_candidate_reach_m": float(
                    np.linalg.norm(
                        np.asarray(pose.eef_world_position)
                        - np.asarray(state.body_world_position)
                    )
                ),
            }
            try:
                official_pose = callbacks.official_pose_adapter(
                    starter, pose
                )
                solution = solver(
                    official_pose,
                    initial_joint_pos=state.official_initial_joint_pos,
                    skip_obstacle_update=False,
                )
                item["ik_solution_present"] = solution is not None
                if solution is not None:
                    target = np.asarray(
                        callbacks.arm_target_from_official_ik(
                            starter, solution, "right"
                        ),
                        dtype=np.float64,
                    )
                    normalized = (target - lower) / span
                    margin = float(
                        np.min(np.minimum(normalized, 1.0 - normalized))
                    )
                    item["arm_target"] = target.tolist()
                    item["joint_margin_fraction"] = margin
                    item["normalized_motion"] = float(
                        np.linalg.norm((target - current) / span)
                    )
            except Exception as exc:
                item.update(
                    {
                        "ik_solution_present": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            result["candidates"].append(item)
        result["state_token"] = state.state_token
        result["attempted_candidates"] = len(poses)
        result["feasible_candidates"] = sum(
            bool(item.get("ik_solution_present"))
            for item in result["candidates"]
        )
        result["status"] = (
            "PASS" if result["feasible_candidates"] else "ALL_IK_NONE"
        )
    except Exception as exc:
        result.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    _write_json(output_dir / "per_candidate.json", result)
    _record_stage(
        stages,
        "IK_PROBE_COMPLETE",
        phase=phase,
        status=result.get("status"),
        attempted_candidates=result.get("attempted_candidates", 0),
        feasible_candidates=result.get("feasible_candidates", 0),
    )
    return result


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")


def _native_vector(value: Any, size: int, name: str) -> np.ndarray:
    item = value
    for method in ("detach", "cpu"):
        callback = getattr(item, method, None)
        if callable(callback):
            item = callback()
    numpy_method = getattr(item, "numpy", None)
    if callable(numpy_method):
        item = numpy_method()
    array = np.asarray(item, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise RuntimeError(f"{name} must be a finite {size}D vector")
    return array


def _oracle_local_displacements(raw: str | None) -> tuple[float, ...]:
    """Parse the explicitly private local-placement sweep distances."""

    text = "0.08,0.12,0.16,0.20" if raw is None else raw
    try:
        values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise RuntimeError(
            "BEHAVIOR_HL_ORACLE_LOCAL_DISPLACEMENTS_M must be comma-separated meters"
        ) from exc
    if not values or any(not math.isfinite(item) or item <= 0.0 for item in values):
        raise RuntimeError(
            "BEHAVIOR_HL_ORACLE_LOCAL_DISPLACEMENTS_M must contain positive finite meters"
        )
    if any(right <= left for left, right in zip(values, values[1:])):
        raise RuntimeError(
            "BEHAVIOR_HL_ORACLE_LOCAL_DISPLACEMENTS_M must be strictly increasing"
        )
    return values


def _private_robot_pose(robot: Any) -> tuple[np.ndarray, np.ndarray]:
    getter = getattr(robot, "get_position_orientation", None)
    if not callable(getter):
        raise RuntimeError("private Oracle robot pose API is unavailable")
    try:
        position, quaternion = getter(frame="world", clone=True)
    except TypeError:
        position, quaternion = getter()
    return (
        _native_vector(position, 3, "robot position"),
        _native_vector(quaternion, 4, "robot quaternion"),
    )


def _private_object_position(obj: Any) -> np.ndarray:
    getter = getattr(obj, "get_position_orientation", None)
    if not callable(getter):
        raise RuntimeError("private Oracle object pose API is unavailable")
    position, _ = getter()
    return _native_vector(position, 3, "object position")


def _private_object_aabb_extent(obj: Any) -> np.ndarray:
    """Read an object's private world AABB extent for Oracle contact probes."""

    extent = getattr(obj, "aabb_extent", None)
    if extent is None:
        raise RuntimeError("private object AABB extent API is unavailable")
    return _native_vector(extent, 3, "object AABB extent")


def _private_object_pose(obj: Any) -> tuple[np.ndarray, np.ndarray]:
    getter = getattr(obj, "get_position_orientation", None)
    if not callable(getter):
        raise RuntimeError("private Oracle object pose API is unavailable")
    position, quaternion = getter()
    return (
        _native_vector(position, 3, "object position"),
        _native_vector(quaternion, 4, "object quaternion"),
    )


def _private_object_aabb(obj: Any) -> tuple[np.ndarray, np.ndarray]:
    bounds = getattr(obj, "aabb", None)
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        raise RuntimeError("private object AABB API is unavailable")
    return (
        _native_vector(bounds[0], 3, "object AABB minimum"),
        _native_vector(bounds[1], 3, "object AABB maximum"),
    )


def _append_private_trace(path: Path, record: Mapping[str, Any]) -> None:
    _append_jsonl(path, record)


def _oracle_navigate_to_station(
    *,
    evaluator: Any,
    control: Any,
    bindings: Any,
    station: RadialStation,
    trace_path: Path,
    maximum_steps: int = 650,
) -> dict[str, Any]:
    """Navigate to a private station using only real velocity commands."""

    from omnigibson.utils.usd_utils import RigidContactAPI
    from behavior_hl.embodied.evaluator_bridge.run_g2_g3a_suite import (
        _base_contact,
    )

    robot = evaluator.robot
    position_tolerance = 0.08
    yaw_tolerance = 0.10
    stall_window = 12
    previous_error = float("inf")
    no_progress = 0
    steps = 0
    status = "FAIL"
    reason = "step_budget_exhausted"
    for step in range(maximum_steps):
        position, quaternion = _private_robot_pose(robot)
        yaw = yaw_from_xy_quaternion(quaternion)
        local_x, local_y, yaw_error = local_planar_error(
            current_base_xy=position[:2],
            current_yaw_rad=yaw,
            station=station,
        )
        position_error = float(math.hypot(local_x, local_y))
        if position_error <= position_tolerance and abs(yaw_error) <= yaw_tolerance:
            control.hold()
            status = "PASS"
            reason = "station_reached"
            break
        base_contact_before = bool(_base_contact(evaluator, RigidContactAPI))
        if base_contact_before:
            control.hold()
            reason = "base_contact"
            break

        if abs(yaw_error) > 0.12:
            command = (0.0, 0.0, 0.18 if yaw_error > 0.0 else -0.18)
        else:
            # Holonomic base command is expressed in the robot frame.  Keep
            # each component bounded and do not issue a blind teleport-like
            # target; the next iteration reads the measured pose again.
            # Keep a bounded physical velocity, but do not divide by the
            # full error in metres.  The previous ``/ 0.10`` rule reduced a
            # 1-metre approach to roughly 0.1 m/s and exhausted the 30 Hz
            # budget before the free-space station.  A 0.20 m/s norm matches
            # the audited public base command while still re-reading the
            # measured pose after every simulator step.
            scale = max(1.0, position_error / 0.20)
            command = (
                float(np.clip(local_x / scale, -0.20, 0.20)),
                float(np.clip(local_y / scale, -0.20, 0.20)),
                0.0,
            )
        control.command_base(command)
        bindings._raw_step(evaluator, f"oracle.station.{station.station_id}.{step:03d}")
        steps += 1
        new_position, new_quaternion = _private_robot_pose(robot)
        new_yaw = yaw_from_xy_quaternion(new_quaternion)
        new_x, new_y, new_yaw_error = local_planar_error(
            current_base_xy=new_position[:2],
            current_yaw_rad=new_yaw,
            station=station,
        )
        new_error = float(math.hypot(new_x, new_y)) + abs(new_yaw_error)
        base_contact_after = bool(_base_contact(evaluator, RigidContactAPI))
        if new_error >= previous_error - 1e-4:
            no_progress += 1
        else:
            no_progress = 0
        previous_error = new_error
        _append_private_trace(
            trace_path,
            {
                "phase": "station_navigation",
                "station_id": station.station_id,
                "step": steps,
                "command": list(command),
                "position": position.tolist(),
                "yaw_rad": yaw,
                "local_error": [local_x, local_y, yaw_error],
                "position_error_m": position_error,
                "base_contact_before_step": base_contact_before,
                "base_contact_after_step": base_contact_after,
            },
        )
        if base_contact_after:
            control.hold()
            reason = "base_contact"
            break
        if no_progress >= stall_window:
            control.hold()
            reason = "navigation_stalled"
            break
    control.hold()
    position, quaternion = _private_robot_pose(robot)
    yaw = yaw_from_xy_quaternion(quaternion)
    local_x, local_y, yaw_error = local_planar_error(
        current_base_xy=position[:2],
        current_yaw_rad=yaw,
        station=station,
    )
    return {
        "station_id": station.station_id,
        "status": status,
        "reason": reason,
        "simulator_steps": steps,
        "final_position": position.tolist(),
        "final_yaw_rad": yaw,
        "final_local_error": [local_x, local_y, yaw_error],
        "final_position_error_m": float(math.hypot(local_x, local_y)),
    }


def _oracle_targeted_grasp(
    *,
    evaluator: Any,
    control: Any,
    bindings: Any,
    ball: Any,
    run_dir: Path,
    station: RadialStation,
    trial_index: int,
    target_points_override: tuple[np.ndarray, ...] | None = None,
    quaternions_override: tuple[tuple[float, float, float, float], ...] | None = None,
    pregrasp_offset_override: tuple[float, float, float] | None = None,
    lift_offset_override: tuple[float, float, float] | None = None,
    candidate_id_override: str | None = None,
) -> dict[str, Any]:
    """Try target-directed official IK and physical reach/close/lift."""

    from behavior_hl.embodied.evaluator_bridge.r1pro_b1_live_entrypoint import (
        _OfficialPrivateGraspProbe,
    )
    from behavior_hl.embodied.evaluator_bridge.r1pro_starter_grasp_driver import (
        PrivateGraspEvidence,
        PublicGripperWristEvidence,
        R1ProStarterGraspDriver,
        StarterGraspPoseConfig,
    )
    from behavior_hl.embodied.runtime.plugins.grasp import GraspMotionPhase
    from behavior_hl.embodied.evaluator_bridge.r1pro_torso_release_binding import (
        RIGHT_GRIPPER_QPOS,
        RIGHT_GRIPPER_QVEL,
    )

    starter = bindings._starter_for(evaluator)
    if getattr(starter, "arm", None) != "right":
        raise RuntimeError("Oracle targeted grasp requires the right Starter")
    probe = _OfficialPrivateGraspProbe(starter)
    probe.begin_target("right", ball)
    trace_path = run_dir / "private" / "oracle_targeted_grasp_trace.jsonl"
    initial_eef_position, initial_eef_quaternion = starter.robot.get_eef_pose(
        arm="right"
    )
    initial_eef_position = _native_vector(initial_eef_position, 3, "initial EEF position")
    initial_eef_quaternion = _native_vector(
        initial_eef_quaternion, 4, "initial EEF quaternion"
    )
    physical_executions = 0
    current_execution_started = False

    def public_evidence(hand: str) -> PublicGripperWristEvidence:
        if hand != "right":
            raise RuntimeError("Oracle public evidence hand changed")
        snapshot = bindings._snapshot(evaluator)
        proprio = _native_vector(snapshot["proprio"], 61, "Oracle proprio")
        return PublicGripperWristEvidence(
            qpos=tuple(float(value) for value in proprio[RIGHT_GRIPPER_QPOS]),
            qvel=tuple(float(value) for value in proprio[RIGHT_GRIPPER_QVEL]),
            # This is an inert type adapter for the private Oracle.  Physical
            # success is decided only by private contact/assisted/lift state.
            wrist_visual_consistent=True,
        )

    def private_audit(hand: str, phase: GraspMotionPhase) -> PrivateGraspEvidence:
        held = probe.held(hand)
        lift = probe.lift(hand) if phase is GraspMotionPhase.LIFT else 0.0
        witness = probe.contact_witness(hand, target=ball)
        target_contact = bool(
            witness.get("calculated_in_hand_is_target")
            or (
                witness.get("target_in_finger_contacts")
                and witness.get("target_in_raycast")
            )
        )
        snapshot = bindings._snapshot(evaluator)
        proprio = _native_vector(snapshot["proprio"], 61, "Oracle proprio")
        _append_private_trace(
            trace_path,
            {
                "phase": phase.value,
                "candidate_id": candidate_id_override,
                "qualified_contact": target_contact,
                "assisted_grasp_active": bool(probe.assisted(hand)),
                "held_object": getattr(held, "name", None),
                "payload_lift_m": float(lift),
                "right_gripper_qpos": [
                    float(value) for value in proprio[RIGHT_GRIPPER_QPOS]
                ],
                "right_gripper_qvel": [
                    float(value) for value in proprio[RIGHT_GRIPPER_QVEL]
                ],
                "official_contact_witness": witness,
            },
        )
        return PrivateGraspEvidence(
            qualified_contact=target_contact,
            assisted_grasp_active=probe.assisted(hand),
            held_token=held,
            payload_lift_m=lift,
        )

    def command_step(action: np.ndarray, label: str) -> int:
        nonlocal physical_executions, current_execution_started
        if not current_execution_started:
            physical_executions += 1
            current_execution_started = True
        control.command_direct_action(action)
        bindings._raw_step(evaluator, f"oracle.targeted.{label}")
        return 1

    def safe_hold(hand: str) -> int:
        control.hold()
        bindings._raw_step(evaluator, f"oracle.targeted.{hand}.safe_hold")
        return 1

    def safe_carry_pose_provider(**kwargs: Any) -> tuple[Any, Any]:
        del kwargs
        return initial_eef_position, initial_eef_quaternion

    target_center = _private_object_position(ball)
    target_extent = _private_object_aabb_extent(ball)
    # G4A approaches a public RGB-D surface point, then adds the robot's
    # measured fingertip-to-EEF clearance.  The old Oracle used the sphere
    # center plus a fixed 12 cm pregrasp, which was not the official geometry
    # and made the IK result uninterpretable.  Keep all of this private: Qwen
    # still receives only its onboard observation and its own prior actions.
    top_surface_offset = max(0.005, float(target_extent[2]) * 0.5)
    target_points = target_points_override or (
        target_center + np.asarray((0.0, 0.0, top_surface_offset)),
        target_center + np.asarray((0.0, 0.0, top_surface_offset * 0.5)),
        target_center,
        target_center + np.asarray((0.0, 0.0, -min(0.03, top_surface_offset))),
        target_center + np.asarray((0.02, 0.0, 0.0)),
    )
    fingertip_lengths = getattr(starter.robot, "eef_to_fingertip_lengths", None)
    if not isinstance(fingertip_lengths, Mapping):
        raise RuntimeError("private fingertip-length calibration is unavailable")
    right_lengths = fingertip_lengths.get("right")
    if not isinstance(right_lengths, Mapping) or not right_lengths:
        raise RuntimeError("private right fingertip-length calibration is unavailable")
    fingertip_offset = float(
        np.mean([float(value) for value in right_lengths.values()])
    )
    pregrasp_offset = fingertip_offset + 0.01
    if not math.isfinite(pregrasp_offset) or pregrasp_offset <= 0.0:
        raise RuntimeError("private fingertip pregrasp offset is invalid")
    geometry_audit = {
        "target_center_private": target_center.tolist(),
        "target_aabb_extent_private": target_extent.tolist(),
        "top_surface_offset_m": top_surface_offset,
        "fingertip_offset_m": fingertip_offset,
        "pregrasp_offset_m": pregrasp_offset,
        "target_point_count": len(target_points),
        "contact_geometry_source": "private_object_aabb_top_surface",
    }
    _write_json(run_dir / "private" / "oracle_target_geometry.json", geometry_audit)
    quaternions = quaternions_override or (
        (0.0, 0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0, 0.0),
    )
    attempts: list[dict[str, Any]] = []
    for point_index, point in enumerate(target_points):
        for quaternion_index, quaternion in enumerate(quaternions):
            current_execution_started = False
            driver = R1ProStarterGraspDriver(
                starter=starter,
                command_step=command_step,
                public_evidence_provider=public_evidence,
                private_audit_provider=private_audit,
                safe_hold=safe_hold,
                safe_carry_pose_provider=safe_carry_pose_provider,
                config=StarterGraspPoseConfig(
                    top_down_quaternion_xyzw=quaternion,
                    pregrasp_offset_world_m=(
                        pregrasp_offset_override
                        if pregrasp_offset_override is not None
                        else (0.0, 0.0, pregrasp_offset)
                    ),
                    lift_offset_world_m=(
                        lift_offset_override
                        if lift_offset_override is not None
                        else (0.0, 0.0, 0.10)
                    ),
                    # The official assisted-grasp primitive intentionally
                    # approaches the object with a constrained contact motion.
                    # Requiring the final contact pose to be collision-free
                    # before that motion rejects the very contact we need.
                    preflight_contact_pose_ik=False,
                ),
            )
            record: dict[str, Any] = {
                "trial_index": trial_index,
                "station_id": station.station_id,
                "point_index": point_index,
                "quaternion_index": quaternion_index,
                "candidate_id": candidate_id_override,
                "ik": "NONE",
                "physical_status": "NOT_STARTED",
            }
            try:
                plan = driver.preflight_ik(
                    world_point=tuple(float(value) for value in point),
                    hand="right",
                )
                record["preflight_diagnostics"] = list(
                    driver.last_preflight_diagnostics
                )
                record["ik"] = "PASS" if plan is not None else "NONE"
                if plan is None:
                    attempts.append(record)
                    continue
                record["physical_status"] = "RUNNING"
                assisted_authorized = False
                phase_budgets = {
                    GraspMotionPhase.REACH: 1200,
                    GraspMotionPhase.CLOSE: 360,
                    GraspMotionPhase.LIFT: 360,
                }
                for phase, budget in phase_budgets.items():
                    completed = False
                    for _ in range(budget):
                        sample = driver.tick_motion(
                            phase=phase,
                            plan_token=plan,
                            hand="right",
                        )
                        if phase is GraspMotionPhase.CLOSE and sample.contact_event:
                            assisted_authorized = driver.enable_assisted_grasp(
                                plan_token=plan,
                                hand="right",
                            )
                        if phase is GraspMotionPhase.LIFT and sample.payload_lift_m >= 0.05:
                            record["observed_lift_m"] = float(sample.payload_lift_m)
                        if sample.completed:
                            completed = True
                            break
                    if not completed:
                        raise RuntimeError(f"{phase.value}_budget_exhausted")
                    if phase is GraspMotionPhase.CLOSE and not assisted_authorized:
                        raise RuntimeError("no_contact_gated_assisted_grasp")
                held = probe.held("right")
                lift = probe.lift("right")
                record.update(
                    {
                        "physical_status": (
                            "PASS"
                            if held is ball and lift >= 0.05
                            else "FAIL_POSTCONDITION"
                        ),
                        "held_object": getattr(held, "name", None),
                        "assisted_authorized": bool(assisted_authorized),
                        "observed_lift_m": float(lift),
                    }
                )
                attempts.append(record)
                if record["physical_status"] == "PASS":
                    return {
                        "status": "PASS",
                        "station_id": station.station_id,
                        "target_center_private": target_center.tolist(),
                        "target_geometry": geometry_audit,
                        "attempts": attempts,
                        "physical_grasp_executions": 1,
                        "held_object": getattr(held, "name", None),
                        "observed_lift_m": float(lift),
                    }
            except Exception as exc:
                record.update(
                    {
                        "physical_status": "FAIL",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "motion_convergence_scopes": list(
                            getattr(
                                starter,
                                "_behavior_hl_motion_convergence_scopes",
                                (),
                            )
                        ),
                        "motion_convergence_samples": list(
                            getattr(
                                starter.robot,
                                "_behavior_hl_motion_convergence_samples",
                                (),
                            )
                        ),
                    }
                )
                attempts.append(record)
                control.hold()
                return {
                    "status": "FAIL",
                    "station_id": station.station_id,
                    "target_center_private": target_center.tolist(),
                    "attempts": attempts,
                    "physical_grasp_executions": physical_executions,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
    return {
        "status": "NO_TARGETED_IK",
        "station_id": station.station_id,
        "target_center_private": target_center.tolist(),
        "target_geometry": geometry_audit,
        "attempts": attempts,
        "physical_grasp_executions": physical_executions,
    }


def _oracle_targeted_phase(
    *,
    evaluator: Any,
    control: Any,
    bindings: Any,
    build_bindings: Callable[[], Any],
    run_dir: Path,
    stages: Path,
) -> tuple[dict[str, Any], int]:
    """Private station search followed by object-directed physical grasp."""

    _record_stage(stages, "ORACLE_TARGETED_START")
    task = evaluator.env.task
    ball = _scope_object(task, "tennis_ball.n.01_1")
    object_position = _private_object_position(ball)
    start_position, _ = _private_robot_pose(evaluator.robot)
    stations = radial_station_candidates(
        object_xy=object_position[:2],
        current_base_xy=start_position[:2],
        radii_m=(0.62, 0.72, 0.82),
        # In this fixed canary instance the +0.55 route is the first route
        # observed to remain free of the table; retain the other routes as
        # bounded fallbacks and record every collision outcome privately.
        angular_offsets_rad=(0.55, 0.0, -0.55, 1.10, -1.10),
    )
    result: dict[str, Any] = {
        "method": "private_object_directed_station_search_then_r1pro_grasp",
        "status": "FAIL",
        "target_category": str(getattr(ball, "category", "")),
        "station_candidates": [station.station_id for station in stations],
        "station_attempts": [],
        "physical_grasp_executions": 0,
        "private_truth_scope": "private_only",
    }
    reset_index = 0
    cumulative_navigation_steps = 0
    cumulative_navigation_budget = 1500
    attempted_stations = stations[:3]
    for index, station in enumerate(attempted_stations):
        remaining_navigation_steps = (
            cumulative_navigation_budget - cumulative_navigation_steps
        )
        if remaining_navigation_steps <= 0:
            result["station_navigation_budget_exhausted"] = True
            break
        trial_bindings = bindings if index == 0 else build_bindings()
        nav_path = run_dir / "private" / "oracle_station_navigation.jsonl"
        navigation = _oracle_navigate_to_station(
            evaluator=evaluator,
            control=control,
            bindings=trial_bindings,
            station=station,
            trace_path=nav_path,
            maximum_steps=min(650, remaining_navigation_steps),
        )
        cumulative_navigation_steps += int(navigation["simulator_steps"])
        attempt: dict[str, Any] = {
            "station": station.station_id,
            "navigation": navigation,
        }
        if navigation["status"] == "PASS":
            grasp = _oracle_targeted_grasp(
                evaluator=evaluator,
                control=control,
                bindings=trial_bindings,
                ball=ball,
                run_dir=run_dir,
                station=station,
                trial_index=index,
            )
            attempt["targeted_grasp"] = grasp
            result["station_attempts"].append(attempt)
            if grasp.get("status") == "PASS":
                result.update(
                    {
                        "status": "PASS",
                        "successful_station": station.station_id,
                        "physical_grasp_executions": 1,
                        "observed_lift_m": grasp.get("observed_lift_m"),
                        "held_object": grasp.get("held_object"),
                    }
                )
                break
        else:
            result["station_attempts"].append(attempt)
        if index + 1 < len(attempted_stations):
            reset_index = _reset_same_scene(
                evaluator=evaluator,
                control=control,
                stages=stages,
                phase=f"after_oracle_station_{index:02d}",
                run_dir=run_dir,
                reset_index=reset_index,
            )
    result["cumulative_navigation_steps"] = cumulative_navigation_steps
    result["cumulative_navigation_budget"] = cumulative_navigation_budget
    _write_json(run_dir / "private" / "oracle_targeted_result.json", result)
    _record_stage(
        stages,
        "ORACLE_TARGETED_COMPLETE",
        status=result.get("status"),
        physical_grasp_executions=result.get("physical_grasp_executions"),
        station_attempts=len(result.get("station_attempts", ())),
    )
    return result, reset_index


def _oracle_official_station_author(
    *,
    starter: Any,
    target: Any,
    target_position: np.ndarray,
    seeds: tuple[int, ...] = (11, 29, 47),
) -> tuple[tuple[RadialStation, ...], list[dict[str, Any]], tuple[Any, Any]]:
    """Ask the official Starter for private, collision-aware base stations."""

    import torch as th

    python_state = random.getstate()
    torch_state = th.random.get_rng_state()
    cuda_states = th.cuda.get_rng_state_all() if th.cuda.is_available() else None
    records: list[dict[str, Any]] = []
    stations: list[RadialStation] = []
    official_pregrasp: tuple[Any, Any] | None = None
    try:
        starter._motion_generator.update_obstacles()
        for seed in seeds:
            random.seed(seed)
            th.manual_seed(seed)
            if th.cuda.is_available():
                th.cuda.manual_seed_all(seed)
            record: dict[str, Any] = {"seed": seed, "source": "official_starter"}
            try:
                pregrasp, grasp = starter._sample_grasp_pose(target)
                if official_pregrasp is None:
                    official_pregrasp = pregrasp
                record["official_pregrasp_position"] = _native_vector(
                    pregrasp[0], 3, "official pregrasp position"
                ).tolist()
                record["official_pregrasp_quaternion_xyzw"] = _native_vector(
                    pregrasp[1], 4, "official pregrasp quaternion"
                ).tolist()
                record["official_grasp_position"] = _native_vector(
                    grasp[0], 3, "official grasp position"
                ).tolist()
                record["official_grasp_quaternion_xyzw"] = _native_vector(
                    grasp[1], 4, "official grasp quaternion"
                ).tolist()
                sampled = starter._sample_pose_near_object(
                    target,
                    eef_pose=pregrasp,
                    sampling_attempts=200,
                    skip_obstacle_update=True,
                )
                record["last_world_root_validation"] = dict(
                    getattr(
                        starter,
                        "_behavior_hl_last_world_root_validation",
                        {},
                    )
                )
                if sampled is None:
                    record.update({"status": "NONE", "reason": "official_sampling_failed"})
                else:
                    pose = _native_vector(sampled, 3, "official sampled station")
                    radius = float(np.linalg.norm(pose[:2] - target_position[:2]))
                    station = RadialStation(
                        station_id=f"official_seed_{seed}",
                        object_xy=tuple(float(value) for value in target_position[:2]),
                        base_xy=tuple(float(value) for value in pose[:2]),
                        radius_m=radius,
                        yaw_rad=float(pose[2]),
                    )
                    duplicate = any(
                        np.linalg.norm(
                            np.asarray(existing.base_xy) - np.asarray(station.base_xy)
                        ) < 0.02
                        and abs(wrap_angle(existing.yaw_rad - station.yaw_rad)) < 0.05
                        for existing in stations
                    )
                    record.update(
                        {
                            "status": "DUPLICATE" if duplicate else "PASS",
                            "sampled_base_pose": pose.tolist(),
                            "radius_m": radius,
                        }
                    )
                    if not duplicate:
                        stations.append(station)
            except Exception as exc:
                record.update(
                    {
                        "status": "ERROR",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            records.append(record)
    finally:
        random.setstate(python_state)
        th.random.set_rng_state(torch_state)
        if cuda_states is not None:
            th.cuda.set_rng_state_all(cuda_states)
    if official_pregrasp is None:
        raise RuntimeError("official Starter could not author a pregrasp pose")
    return tuple(stations), records, official_pregrasp


def _oracle_station_fallback(
    *,
    starter: Any,
    target_position: np.ndarray,
    current_base_xy: np.ndarray,
    official_pregrasp: tuple[Any, Any],
) -> tuple[tuple[RadialStation, ...], list[dict[str, Any]]]:
    """Bounded deterministic fallback using the official pose validator."""

    import torch as th
    import omnigibson.utils.transform_utils as T

    workspace = float(th.mean(starter.robot.arm_workspace_range["right"]).item())
    candidates = radial_station_candidates(
        object_xy=target_position[:2],
        current_base_xy=current_base_xy,
        radii_m=(0.55, 0.65, 0.75, 0.85),
        angular_offsets_rad=(0.0, 0.35, -0.35, 0.70, -0.70),
        arm_workspace_yaw_rad=workspace,
    )
    starter._motion_generator.update_obstacles()
    records: list[dict[str, Any]] = []
    feasible: list[RadialStation] = []
    reference = starter.robot.get_joint_positions()
    base_world_position, base_world_orientation = (
        starter.robot.get_position_orientation(frame="world")
    )
    root_world_position, root_world_orientation = (
        starter.robot.root_link.get_position_orientation()
    )
    base_idx = [
        int(value) for value in starter.robot.base_idx.detach().cpu().tolist()
    ]
    base_control_idx = [
        int(value)
        for value in starter.robot.base_control_idx.detach().cpu().tolist()
    ]
    object_in_hand = starter._get_obj_in_hand()
    attached_obj = (
        {starter.robot.eef_link_names[starter.arm]: object_in_hand.root_link}
        if object_in_hand is not None
        else None
    )
    for station in candidates:
        pose = reference.new_tensor(
            [station.base_xy[0], station.base_xy[1], station.yaw_rad]
        )
        candidate_joint_position = (
            starter._behavior_hl_world_base_candidate_joint_position(
                pose,
                reference,
            )
        )
        desired_world_quaternion = T.euler2quat(
            reference.new_tensor([0.0, 0.0, station.yaw_rad])
        )
        base_collision = bool(
            starter._motion_generator.check_collisions(
                candidate_joint_position.unsqueeze(0),
                self_collision_check=False,
                skip_obstacle_update=True,
                attached_obj=attached_obj,
            )[0].item()
        )
        pregrasp_reachable = False
        if not base_collision:
            pregrasp_reachable = bool(
                starter._target_in_reach_of_robot(
                    official_pregrasp,
                    initial_joint_pos=candidate_joint_position,
                    skip_obstacle_update=True,
                )
            )
        valid = not base_collision and pregrasp_reachable
        records.append(
            {
                "station_id": station.station_id,
                "source": "bounded_official_validate_poses_fallback",
                "candidate_pose_frame": "world",
                "base_pose": [*station.base_xy, station.yaw_rad],
                "desired_world_position": [
                    float(station.base_xy[0]),
                    float(station.base_xy[1]),
                    float(base_world_position[2].detach().cpu()),
                ],
                "desired_world_quaternion_xyzw": [
                    float(value)
                    for value in desired_world_quaternion.detach().cpu()
                ],
                "base_world_position_before_validation": [
                    float(value) for value in base_world_position.detach().cpu()
                ],
                "base_world_quaternion_xyzw_before_validation": [
                    float(value) for value in base_world_orientation.detach().cpu()
                ],
                "root_world_position_before_validation": [
                    float(value) for value in root_world_position.detach().cpu()
                ],
                "root_world_quaternion_xyzw_before_validation": [
                    float(value) for value in root_world_orientation.detach().cpu()
                ],
                "base_idx": base_idx,
                "base_control_idx": base_control_idx,
                "attached_object_present": object_in_hand is not None,
                "root_relative_base_joints": [
                    float(value)
                    for value in candidate_joint_position[
                        starter.robot.base_idx
                    ].detach().cpu()
                ],
                "base_collision": base_collision,
                "pregrasp_reachable": pregrasp_reachable,
                "validation_failure_stage": (
                    "base_collision"
                    if base_collision
                    else (
                        None
                        if pregrasp_reachable
                        else "official_pregrasp_reachability"
                    )
                ),
                "valid": valid,
            }
        )
        if valid:
            feasible.append(station)
            if len(feasible) == 3:
                break
    return tuple(feasible), records


def _candidate_offsets(
    candidate: GraspCandidate,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    contact = np.asarray(candidate.contact_position_xyz, dtype=np.float64)
    return (
        tuple(
            float(value)
            for value in np.asarray(candidate.pregrasp_position_xyz) - contact
        ),
        tuple(
            float(value)
            for value in np.asarray(candidate.lift_position_xyz) - contact
        ),
    )


def _oracle_restore_open_safe_pose(
    *,
    evaluator: Any,
    control: Any,
    bindings: Any,
    starter: Any,
    safe_pose: tuple[Any, Any],
    trace_path: Path,
    retry_index: int,
) -> dict[str, Any]:
    """Physically open and retract through official Starter generators."""

    record: dict[str, Any] = {
        "retry_index": retry_index,
        "method": "official_execute_release_then_move_hand",
        "simulator_steps": 0,
        "status": "FAIL",
    }
    try:
        phases = [("release", starter._execute_release(), 360)]
        for label, generator, budget in phases:
            for step_index, raw_action in enumerate(generator):
                if step_index >= budget:
                    raise RuntimeError(f"{label}_budget_exhausted")
                action = np.asarray(raw_action, dtype=np.float64)
                if action.shape != (23,) or not np.isfinite(action).all():
                    raise RuntimeError(f"{label}_invalid_official_action")
                control.command_direct_action(action)
                bindings._raw_step(
                    evaluator,
                    f"oracle.generic.retry_{retry_index}.{label}.{step_index:04d}",
                )
                record["simulator_steps"] += 1
            record[f"{label}_completed"] = True
        starter._motion_generator.update_obstacles()
        safe_pose_reachable = bool(
            starter._target_in_reach_of_robot(
                safe_pose,
                initial_joint_pos=None,
                skip_obstacle_update=True,
            )
        )
        record["safe_pose_reachable_after_release"] = safe_pose_reachable
        if not safe_pose_reachable:
            raise RuntimeError("safe retry retract pose failed official IK")
        for step_index, raw_action in enumerate(
            starter._move_hand(
                safe_pose,
                lock_auxiliary_arm=True,
                # Safe retry is a collision-checked free-space retract, not a
                # contact approach.  Use Starter's official low-precision
                # convergence mode so R1Pro torso steady-state error cannot
                # poison recovery at the strict 0.005 threshold.
                low_precision=True,
            )
        ):
            if step_index >= 1200:
                raise RuntimeError("safe_pose_budget_exhausted")
            action = np.asarray(raw_action, dtype=np.float64)
            if action.shape != (23,) or not np.isfinite(action).all():
                raise RuntimeError("safe_pose_invalid_official_action")
            control.command_direct_action(action)
            bindings._raw_step(
                evaluator,
                f"oracle.generic.retry_{retry_index}.safe_pose.{step_index:04d}",
            )
            record["simulator_steps"] += 1
        record["safe_pose_completed"] = True
        assisted_state = getattr(starter.robot, "_ag_obj_in_hand", None)
        assisted_empty = bool(
            isinstance(assisted_state, Mapping)
            and assisted_state.get("right") is None
        )
        held_empty = starter._get_obj_in_hand() is None
        record.update(
            {
                "held_empty": held_empty,
                "assisted_empty": assisted_empty,
                "status": "PASS" if held_empty and assisted_empty else "FAIL",
            }
        )
    except Exception as exc:
        control.hold()
        record.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    _append_private_trace(trace_path, {"phase": "retry_safe_restore", **record})
    return record


def _oracle_generic_grasp_phase(
    *,
    evaluator: Any,
    control: Any,
    bindings: Any,
    run_dir: Path,
    stages: Path,
) -> tuple[dict[str, Any], int]:
    """One-load official-station A/B/C physical tennis-ball canary."""

    import omnigibson as og
    from omnigibson import object_states
    from omnigibson.action_primitives.starter_semantic_action_primitives import (
        StarterSemanticActionPrimitiveSet,
    )
    import torch as th
    import omnigibson.utils.transform_utils as T

    _record_stage(stages, "ORACLE_GENERIC_GRASP_START")
    if (
        os.environ.get("BEHAVIOR_HL_ORACLE_GENERIC_GRASP") != "1"
        or os.environ.get("BEHAVIOR_HL_ORACLE_ONLY") != "1"
    ):
        raise RuntimeError(
            "private generic grasp state restore requires the explicit "
            "Oracle-only kinematic-research gate"
        )
    task = evaluator.env.task
    target = _scope_object(task, "tennis_ball.n.01_1")
    support = _scope_object(task, "table.n.02_1")
    receptacle = _scope_object(task, "toy_box.n.01_1")
    starter = bindings._starter_for(evaluator)
    if getattr(starter.robot, "grasping_mode", None) != "assisted":
        raise RuntimeError("generic physical grasp requires assisted grasping mode")
    trace_path = run_dir / "private" / "oracle_generic_grasp_trace.jsonl"

    support_minimum, support_maximum = _private_object_aabb(support)
    extent = _private_object_aabb_extent(target)
    robot_position, _ = _private_robot_pose(evaluator.robot)
    matching_placements = [
        item
        for item in table_edge_placements(
            table_aabb_min=support_minimum,
            table_aabb_max=support_maximum,
            object_extent_xyz=extent,
            current_base_xy=robot_position[:2],
            inset_m=0.08,
            clearance_m=0.004,
            lateral_fractions=(-0.35,),
        )
        if item.placement_id == "y_min_f-0.35"
    ]
    if len(matching_placements) != 1:
        raise RuntimeError("private canary placement y_min_f-0.35 is unavailable")
    placement = matching_placements[0]
    original_position, original_quaternion = target.get_position_orientation()

    def restore_target(label: str) -> np.ndarray:
        tensor_factory = getattr(original_position, "new_tensor", None)
        if callable(tensor_factory):
            restored_position = tensor_factory(placement.object_center_xyz)
            zero_velocity = th.zeros_like(restored_position)
        else:
            restored_position = th.as_tensor(
                placement.object_center_xyz, dtype=th.float32
            )
            zero_velocity = th.zeros(3, dtype=th.float32)
        target.set_position_orientation(
            position=restored_position,
            orientation=original_quaternion,
            frame="world",
        )
        for velocity_name in ("set_linear_velocity", "set_angular_velocity"):
            setter = getattr(target, velocity_name, None)
            if callable(setter):
                setter(zero_velocity)
        for settle_index in range(30):
            control.hold()
            bindings._raw_step(
                evaluator, f"oracle.generic.{label}.settle.{settle_index:03d}"
            )
        if not bool(target.states[object_states.OnTop].get_value(support)):
            raise RuntimeError("generic grasp target is not OnTop(support)")
        return _private_object_position(target)

    settled_center = restore_target("initial")
    fingertip_lengths = starter.robot.eef_to_fingertip_lengths["right"]
    fingertip_offset = float(np.mean(tuple(fingertip_lengths.values())))
    candidates = generate_grasp_candidates(
        "sphere",
        object_center_xyz=settled_center,
        object_extent_xyz=extent,
        fingertip_offset_m=fingertip_offset,
    )
    validation_joint_positions_before = starter.robot.get_joint_positions().clone()
    validation_base_position_before, validation_base_quaternion_before = (
        starter.robot.get_position_orientation(frame="world")
    )
    runtime_branch_attestation = {
        "is_holonomic_base": bool(starter.robot.is_holonomic_base),
        "robot_initialized": bool(starter.robot.initialized),
        "sim_is_playing": bool(og.sim.is_playing()),
        "transform_probe_yaw_rad": 0.7,
        "transform_probe_quaternion_xyzw": [
            float(value)
            for value in T.euler2quat(
                validation_joint_positions_before.new_tensor([0.0, 0.0, 0.7])
            ).detach().cpu()
        ],
    }
    if not all(
        runtime_branch_attestation[key]
        for key in ("is_holonomic_base", "robot_initialized", "sim_is_playing")
    ):
        raise RuntimeError(
            "R1Pro runtime does not match official playing holonomic branch"
        )
    stations, station_records, official_pregrasp = _oracle_official_station_author(
        starter=starter,
        target=target,
        target_position=settled_center,
    )
    fallback_records: list[dict[str, Any]] = []
    if not stations:
        robot_position, _ = _private_robot_pose(evaluator.robot)
        stations, fallback_records = _oracle_station_fallback(
            starter=starter,
            target_position=settled_center,
            current_base_xy=robot_position[:2],
            official_pregrasp=official_pregrasp,
        )
    validation_joint_positions_after = starter.robot.get_joint_positions()
    validation_base_position_after, validation_base_quaternion_after = (
        starter.robot.get_position_orientation(frame="world")
    )
    validation_state_attestation = {
        "joint_positions_unchanged": bool(
            th.equal(
                validation_joint_positions_before,
                validation_joint_positions_after,
            )
        ),
        "base_world_position_unchanged": bool(
            th.equal(
                validation_base_position_before,
                validation_base_position_after,
            )
        ),
        "base_world_quaternion_unchanged": bool(
            th.equal(
                validation_base_quaternion_before,
                validation_base_quaternion_after,
            )
        ),
    }
    station_audit = {
        "runtime_branch_attestation": runtime_branch_attestation,
        "validation_state_attestation": validation_state_attestation,
        "official_author_records": station_records,
        "fallback_records": fallback_records,
        "accepted_stations": [asdict(station) for station in stations],
        "official_pregrasp_position": _native_vector(
            official_pregrasp[0], 3, "official pregrasp"
        ).tolist(),
        "official_pregrasp_quaternion_xyzw": _native_vector(
            official_pregrasp[1], 4, "official pregrasp quaternion"
        ).tolist(),
    }
    if not all(validation_state_attestation.values()):
        raise RuntimeError("official station validation mutated live robot state")
    measured_rank_position, measured_rank_quaternion = _private_robot_pose(
        evaluator.robot
    )
    measured_rank_yaw = yaw_from_xy_quaternion(measured_rank_quaternion)
    ranked_station_costs = rank_radial_stations_by_measured_cost(
        stations,
        current_base_xy=measured_rank_position[:2],
        current_yaw_rad=measured_rank_yaw,
    )
    stations = tuple(item[0] for item in ranked_station_costs)
    station_audit["measured_pose_ranking"] = {
        "base_xy": measured_rank_position[:2].tolist(),
        "yaw_rad": measured_rank_yaw,
        "cost_formula": "planar_distance/0.20 + abs(wrapped_yaw_error)/0.18",
        "ranked": [
            {"station_id": station.station_id, "estimated_cost_s": cost}
            for station, cost in ranked_station_costs
        ],
    }
    _write_json(run_dir / "private" / "oracle_station_author.json", station_audit)
    _write_json(
        run_dir / "private" / "oracle_grasp_candidates.json",
        {"geometry_family": "sphere", "candidates": [asdict(item) for item in candidates]},
    )

    result: dict[str, Any] = {
        "method": "official_station_author_generic_sphere_abc",
        "status": "FAIL",
        "placement_id": placement.placement_id,
        "station_count": len(stations),
        "station_navigation_attempts": [],
        "physical_attempts": [],
        "physical_grasp_executions": 0,
        "physical_attempt_budget": 3,
        "direct_state_modification": True,
        "direct_state_modification_scope": "private_supported_canary_restore_only",
        "kinematic_research": True,
        "official_challenge_eligible": False,
        "strict_production_eligible": False,
        "private_truth_scope": "private_oracle_only",
    }
    reached_station: RadialStation | None = None
    cumulative_navigation_steps = 0
    cumulative_navigation_budget = 1500
    for station in stations[:3]:
        remaining_navigation_steps = (
            cumulative_navigation_budget - cumulative_navigation_steps
        )
        if remaining_navigation_steps <= 0:
            result["station_navigation_budget_exhausted"] = True
            break
        navigation = _oracle_navigate_to_station(
            evaluator=evaluator,
            control=control,
            bindings=bindings,
            station=station,
            trace_path=trace_path,
            maximum_steps=min(650, remaining_navigation_steps),
        )
        result["station_navigation_attempts"].append(navigation)
        cumulative_navigation_steps += int(navigation["simulator_steps"])
        if navigation.get("status") == "PASS":
            reached_station = station
            break
    result["cumulative_navigation_steps"] = cumulative_navigation_steps
    result["cumulative_navigation_budget"] = cumulative_navigation_budget
    if reached_station is None:
        result["failure_stage"] = "official_station_navigation"
    else:
        settled_center = _private_object_position(target)
        safe_retry_pose = starter.robot.get_eef_pose(arm="right")
        for index in range(len(candidates)):
            if index > 0:
                if starter._get_obj_in_hand() is not None:
                    result["stopped_early_reason"] = "hand_not_empty_after_failed_attempt"
                    break
                restore = _oracle_restore_open_safe_pose(
                    evaluator=evaluator,
                    control=control,
                    bindings=bindings,
                    starter=starter,
                    safe_pose=safe_retry_pose,
                    trace_path=trace_path,
                    retry_index=index,
                )
                result.setdefault("retry_restores", []).append(restore)
                if restore.get("status") != "PASS":
                    result["stopped_early_reason"] = "official_retry_restore_failed"
                    break
                settled_center = restore_target(f"retry_{index}")
            refreshed_candidates = generate_grasp_candidates(
                "sphere",
                object_center_xyz=settled_center,
                object_extent_xyz=_private_object_aabb_extent(target),
                fingertip_offset_m=fingertip_offset,
            )
            candidate = refreshed_candidates[index]
            pregrasp_offset, lift_offset = _candidate_offsets(candidate)
            starter._motion_generator.update_obstacles()
            grasp = _oracle_targeted_grasp(
                evaluator=evaluator,
                control=control,
                bindings=bindings,
                ball=target,
                run_dir=run_dir,
                station=reached_station,
                trial_index=index,
                target_points_override=(
                    np.asarray(candidate.contact_position_xyz, dtype=np.float64),
                ),
                quaternions_override=(candidate.top_down_quaternion_xyzw,),
                pregrasp_offset_override=pregrasp_offset,
                lift_offset_override=lift_offset,
                candidate_id_override=candidate.candidate_id,
            )
            attempt: dict[str, Any] = {
                "index": index,
                "candidate": asdict(candidate),
                "grasp": grasp,
                "status": "FAIL",
            }
            result["physical_attempts"].append(attempt)
            result["physical_grasp_executions"] += int(
                grasp.get("physical_grasp_executions", 0)
            )
            if result["physical_grasp_executions"] > 3:
                raise RuntimeError("generic physical grasp budget exceeded")
            if grasp.get("status") != "PASS":
                continue
            place = _run_official_primitive(
                evaluator=evaluator,
                starter=starter,
                primitive=StarterSemanticActionPrimitiveSet.PLACE_INSIDE,
                target=receptacle,
                phase="oracle.generic.place_inside",
                trace_path=trace_path,
            )
            held_after_place = starter._get_obj_in_hand()
            assisted_state = getattr(starter.robot, "_ag_obj_in_hand", None)
            assisted_empty_after_place = bool(
                isinstance(assisted_state, Mapping)
                and assisted_state.get("right") is None
            )
            inside_window: list[bool] = []
            for stability_index in range(30):
                control.hold()
                bindings._raw_step(
                    evaluator,
                    f"oracle.generic.inside_stability.{stability_index:03d}",
                )
                inside_window.append(
                    bool(target.states[object_states.Inside].get_value(receptacle))
                )
            attempt.update(
                {
                    "place": place,
                    "held_after_place": getattr(held_after_place, "name", None),
                    "assisted_empty_after_place": assisted_empty_after_place,
                    "inside_stability_window": inside_window,
                    "inside_stable": (
                        held_after_place is None
                        and assisted_empty_after_place
                        and bool(inside_window)
                        and all(inside_window)
                    ),
                }
            )
            if attempt["inside_stable"]:
                attempt["status"] = "PASS"
                result.update(
                    {
                        "status": "PASS",
                        "successful_station": reached_station.station_id,
                        "successful_candidate": candidate.candidate_id,
                        "observed_lift_m": grasp.get("observed_lift_m"),
                        "inside_stable": True,
                    }
                )
                break
    _write_json(run_dir / "private" / "oracle_generic_grasp_result.json", result)
    _record_stage(
        stages,
        "ORACLE_GENERIC_GRASP_COMPLETE",
        status=result["status"],
        station_count=len(stations),
        physical_attempt_count=len(result["physical_attempts"]),
    )
    return result, 0


def _oracle_screened_center_grasp_phase(
    *,
    evaluator: Any,
    control: Any,
    bindings: Any,
    run_dir: Path,
    stages: Path,
) -> tuple[dict[str, Any], int]:
    """Deterministically screen stations, then physically center-grasp a ball."""

    from omnigibson import object_states
    from omnigibson.action_primitives.starter_semantic_action_primitives import (
        StarterSemanticActionPrimitiveSet,
    )
    import torch as th

    _record_stage(stages, "ORACLE_SCREENED_CENTER_GRASP_START")
    task = evaluator.env.task
    ball = _scope_object(task, "tennis_ball.n.01_1")
    table = _scope_object(task, "table.n.02_1")
    toy_box = _scope_object(task, "toy_box.n.01_1")
    starter = bindings._starter_for(evaluator)
    trace_path = run_dir / "private" / "oracle_screened_center_trace.jsonl"
    table_minimum, table_maximum = _private_object_aabb(table)
    ball_extent = _private_object_aabb_extent(ball)
    robot_position, _ = _private_robot_pose(evaluator.robot)
    placements = table_edge_placements(
        table_aabb_min=table_minimum,
        table_aabb_max=table_maximum,
        object_extent_xyz=ball_extent,
        current_base_xy=robot_position[:2],
        inset_m=0.08,
        clearance_m=0.004,
        lateral_fractions=(-0.35,),
    )
    placement = next(item for item in placements if item.placement_id == "y_min_f-0.35")
    raw_position, raw_quaternion = ball.get_position_orientation()
    ball.set_position_orientation(
        position=th.as_tensor(
            placement.object_center_xyz,
            dtype=raw_position.dtype,
            device=raw_position.device,
        ),
        orientation=raw_quaternion,
        frame="world",
    )
    for velocity_name in ("set_linear_velocity", "set_angular_velocity"):
        setter = getattr(ball, velocity_name, None)
        if callable(setter):
            setter(th.zeros_like(raw_position))
    for settle_index in range(30):
        control.hold()
        bindings._raw_step(evaluator, f"oracle.screened.settle.{settle_index:03d}")
    settled_center, _ = _private_object_pose(ball)
    if not bool(ball.states[object_states.OnTop].get_value(table)):
        raise RuntimeError("screened center-grasp ball is not OnTop(table)")

    variants = tennis_ball_center_grasp_variants(
        object_center_xyz=settled_center,
        object_extent_xyz=ball_extent,
        table_edge=placement.table_edge,
    )
    workspace_range = starter.robot.arm_workspace_range["right"]
    workspace_yaw = float(th.mean(workspace_range).item())
    robot_position, _ = _private_robot_pose(evaluator.robot)
    stations = radial_station_candidates(
        object_xy=settled_center[:2],
        current_base_xy=robot_position[:2],
        radii_m=(1.00, 1.20, 1.40),
        angular_offsets_rad=(0.0, 0.35, -0.35, 0.70, -0.70),
        arm_workspace_yaw_rad=workspace_yaw,
    )
    screened_variants = (
        next(item for item in variants if item.variant_id == "center_official_top_down"),
        next(item for item in variants if item.variant_id == "center_edge_aligned"),
    )
    # One base collision query per station plus at most one pregrasp IK query
    # per station / grasp variant: 15 + 30 = 45, below the hard cap of 48.
    if len(stations) * (1 + len(screened_variants)) > 48:
        raise RuntimeError("screened center-grasp budget exceeds 48 IK checks")
    fingertip_lengths = starter.robot.eef_to_fingertip_lengths["right"]
    pregrasp_offset = float(np.mean(tuple(fingertip_lengths.values()))) + 0.01
    eef_position, _ = starter.robot.get_eef_pose(arm="right")
    joint_reference = starter.robot.get_joint_positions()
    starter._motion_generator.update_obstacles()
    screen: list[dict[str, Any]] = []
    feasible: list[tuple[RadialStation, Any, int]] = []
    for station_index, station in enumerate(stations):
        candidate_pose = joint_reference.new_tensor(
            [station.base_xy[0], station.base_xy[1], station.yaw_rad]
        )
        candidate_joint_position = joint_reference.clone()
        candidate_joint_position[starter.robot.base_control_idx] = candidate_pose
        base_collision = bool(
            starter._motion_generator.check_collisions(
                candidate_joint_position.unsqueeze(0),
                self_collision_check=False,
                skip_obstacle_update=True,
                attached_obj=None,
            )[0].item()
        )
        for variant_index, variant in enumerate(screened_variants):
            close_position = eef_position.new_tensor(variant.close_position_xyz)
            pregrasp_position = close_position + eef_position.new_tensor(
                [0.0, 0.0, pregrasp_offset]
            )
            quaternion = eef_position.new_tensor(
                variant.top_down_quaternion_xyzw
            )
            pregrasp_ok = False
            if not base_collision:
                pregrasp_ok = bool(
                    starter._target_in_reach_of_robot(
                        (pregrasp_position, quaternion),
                        initial_joint_pos=candidate_joint_position,
                        skip_obstacle_update=True,
                    )
                )
            record = {
                "station_id": station.station_id,
                "variant_id": variant.variant_id,
                "station_index": station_index,
                "variant_index": variant_index,
                "base_collision": base_collision,
                "pregrasp_ok": pregrasp_ok,
                "failure_stage": (
                    "base_collision"
                    if base_collision
                    else ("pregrasp_ik" if not pregrasp_ok else None)
                ),
                "feasible": pregrasp_ok,
            }
            screen.append(record)
            _append_private_trace(trace_path, {"phase": "motionless_screen", **record})
            if record["feasible"]:
                feasible.append((station, variant, variant_index))
    current_radius = float(np.linalg.norm(robot_position[:2] - settled_center[:2]))
    feasible.sort(key=lambda item: (abs(item[0].radius_m - current_radius), item[2]))
    diversified: list[tuple[RadialStation, Any, int]] = []
    seen_station_ids: set[str] = set()
    for item in feasible:
        if item[0].station_id not in seen_station_ids:
            diversified.append(item)
            seen_station_ids.add(item[0].station_id)
    diversified.extend(item for item in feasible if item not in diversified)
    result: dict[str, Any] = {
        "method": "private_deterministic_station_screen_center_grasp",
        "status": "FAIL",
        "placement_id": placement.placement_id,
        "settled_ball_position_private": settled_center.tolist(),
        "screen_checks": len(stations) + sum(
            1 for record in screen if not record["base_collision"]
        ),
        "screen_check_budget": 48,
        "workspace_yaw_rad": workspace_yaw,
        "screen_records": screen,
        "feasible_combination_count": len(feasible),
        "physical_attempts": [],
        "physical_attempt_budget": 3,
        "direct_state_modification": True,
        "direct_state_modification_scope": "tennis_ball_edge_calibration",
        "kinematic_research": True,
        "official_challenge_eligible": False,
        "strict_production_eligible": False,
    }
    for physical_index, (station, variant, _) in enumerate(diversified[:3]):
        attempt: dict[str, Any] = {
            "physical_index": physical_index,
            "station_id": station.station_id,
            "variant_id": variant.variant_id,
            "status": "FAIL",
        }
        navigation = _oracle_navigate_to_station(
            evaluator=evaluator,
            control=control,
            bindings=bindings,
            station=station,
            trace_path=trace_path,
        )
        attempt["navigation"] = navigation
        if navigation.get("status") != "PASS":
            result["physical_attempts"].append(attempt)
            continue
        starter._motion_generator.update_obstacles()
        grasp = _oracle_targeted_grasp(
            evaluator=evaluator,
            control=control,
            bindings=bindings,
            ball=ball,
            run_dir=run_dir,
            station=station,
            trial_index=physical_index,
            target_points_override=(
                np.asarray(variant.close_position_xyz, dtype=np.float64),
            ),
            quaternions_override=(variant.top_down_quaternion_xyzw,),
        )
        attempt["grasp"] = grasp
        if grasp.get("status") == "PASS":
            place = _run_official_primitive(
                evaluator=evaluator,
                starter=starter,
                primitive=StarterSemanticActionPrimitiveSet.PLACE_INSIDE,
                target=toy_box,
                phase="oracle.screened.place_inside",
                trace_path=trace_path,
            )
            inside = bool(ball.states[object_states.Inside].get_value(toy_box))
            attempt.update({"place": place, "inside_after_place": inside})
            if inside:
                attempt["status"] = "PASS"
                result["physical_attempts"].append(attempt)
                result.update(
                    {
                        "status": "PASS",
                        "successful_station": station.station_id,
                        "successful_variant": variant.variant_id,
                        "observed_lift_m": grasp.get("observed_lift_m"),
                        "inside_after_place": True,
                    }
                )
                break
        result["physical_attempts"].append(attempt)
        if starter._get_obj_in_hand() is not None:
            result["stopped_early_reason"] = "hand_not_empty_after_failure"
            break
        # Restore the same supported canary placement before the next real try.
        ball.set_position_orientation(
            position=raw_position.new_tensor(placement.object_center_xyz),
            orientation=raw_quaternion,
            frame="world",
        )
        for cleanup_index in range(30):
            control.hold()
            bindings._raw_step(
                evaluator,
                f"oracle.screened.retry_settle.{physical_index:02d}.{cleanup_index:03d}",
            )
    _write_json(run_dir / "private" / "oracle_screened_center_result.json", result)
    _record_stage(
        stages,
        "ORACLE_SCREENED_CENTER_GRASP_COMPLETE",
        status=result["status"],
        feasible_combination_count=len(feasible),
        physical_attempt_count=len(result["physical_attempts"]),
    )
    return result, 0


def _oracle_remaining_toys_phase(
    *,
    evaluator: Any,
    control: Any,
    bindings: Any,
    starter: Any,
    toy_box: Any,
    run_dir: Path,
    stages: Path,
    displacement_m: float,
) -> dict[str, Any]:
    """Continue the private Oracle over every toy after the ball canary.

    This deliberately keeps the scene resident and never resets between toys,
    because a reset would erase already-established ``Inside`` leaves.  Each
    toy is moved locally on its declared support, settled, and required to
    retain canonical ``OnTop`` before the official physical primitives run.
    """

    from omnigibson import object_states
    from omnigibson.action_primitives.starter_semantic_action_primitives import (
        StarterSemanticActionPrimitiveSet,
    )
    import torch as th

    task = evaluator.env.task
    sequence = (
        ("board_game.n.01_3", "table.n.02_1"),
        ("jigsaw_puzzle.n.01_1", "table.n.02_1"),
        ("jigsaw_puzzle.n.01_2", "table.n.02_1"),
        ("board_game.n.01_1", "bed.n.01_1"),
        ("board_game.n.01_2", "bed.n.01_1"),
    )
    trace_path = run_dir / "private" / "oracle_remaining_toys_trace.jsonl"
    result: dict[str, Any] = {
        "method": "private_resident_all_remaining_toys_official_grasp_place",
        "status": "FAIL",
        "requested_displacement_m": displacement_m,
        "target_count": len(sequence),
        "attempts": [],
        "inside_success_count": 0,
        "scene_reset_between_toys": False,
        "direct_state_modification": True,
        "direct_state_modification_scope": "five_remaining_goal_toys",
        "kinematic_research": True,
        "official_challenge_eligible": False,
        "strict_production_eligible": False,
    }
    _record_stage(stages, "ORACLE_REMAINING_TOYS_START", target_count=len(sequence))
    for index, (object_key, support_key) in enumerate(sequence):
        obj = _scope_object(task, object_key)
        support = _scope_object(task, support_key)
        raw_position, raw_quaternion = obj.get_position_orientation()
        original_position = _native_vector(raw_position, 3, "remaining toy position")
        robot_position, _ = _private_robot_pose(evaluator.robot)
        placement = local_support_placements_toward_robot(
            object_center_xyz=original_position,
            current_base_xy=robot_position[:2],
            displacements_m=(displacement_m,),
        )[0]
        attempt: dict[str, Any] = {
            "index": index,
            "object_scope_key": object_key,
            "object_name_private": getattr(obj, "name", None),
            "support_scope_key": support_key,
            "original_position_private": original_position.tolist(),
            "requested_position_private": list(placement.object_center_xyz),
            "status": "FAIL",
        }
        try:
            obj.set_position_orientation(
                position=th.as_tensor(
                    placement.object_center_xyz,
                    dtype=raw_position.dtype,
                    device=raw_position.device,
                ),
                orientation=raw_quaternion,
                frame="world",
            )
            zero_velocity = th.zeros_like(raw_position)
            for velocity_name in ("set_linear_velocity", "set_angular_velocity"):
                velocity_setter = getattr(obj, velocity_name, None)
                if callable(velocity_setter):
                    velocity_setter(zero_velocity)
            _append_private_trace(
                trace_path,
                {
                    "phase": "direct_state_modification",
                    "object_scope_key": object_key,
                    "support_scope_key": support_key,
                    "requested_position_private": list(placement.object_center_xyz),
                    "official_challenge_eligible": False,
                },
            )
            for settle_index in range(30):
                control.hold()
                bindings._raw_step(
                    evaluator,
                    f"oracle.remaining.{index:02d}.settle.{settle_index:03d}",
                )
            settled_position, _ = _private_object_pose(obj)
            attempt["settled_position_private"] = settled_position.tolist()
            attempt["ontop_support_after_settle"] = bool(
                obj.states[object_states.OnTop].get_value(support)
            )
            if not attempt["ontop_support_after_settle"]:
                raise RuntimeError("remaining toy lost its declared support after settle")

            attempt["grasp"] = _run_official_primitive(
                evaluator=evaluator,
                starter=starter,
                primitive=StarterSemanticActionPrimitiveSet.GRASP,
                target=obj,
                phase=f"oracle.remaining.{index:02d}.grasp",
                trace_path=trace_path,
            )
            held = starter._get_obj_in_hand()
            after_grasp_position, _ = _private_object_pose(obj)
            lift_m = float(after_grasp_position[2] - settled_position[2])
            attempt["held_after_grasp"] = getattr(held, "name", None)
            attempt["observed_lift_m"] = lift_m
            if held is not obj:
                raise RuntimeError("official grasp did not hold the expected remaining toy")
            if lift_m < 0.05:
                raise RuntimeError("official grasp did not lift the remaining toy by 5 cm")

            attempt["place"] = _run_official_primitive(
                evaluator=evaluator,
                starter=starter,
                primitive=StarterSemanticActionPrimitiveSet.PLACE_INSIDE,
                target=toy_box,
                phase=f"oracle.remaining.{index:02d}.place_inside",
                trace_path=trace_path,
            )
            attempt["inside_after_place"] = bool(
                obj.states[object_states.Inside].get_value(toy_box)
            )
            if not attempt["inside_after_place"]:
                raise RuntimeError("official place did not establish Inside(toy, toy_box)")
            attempt["status"] = "PASS"
            result["inside_success_count"] += 1
        except Exception as exc:
            held_after_failure = starter._get_obj_in_hand()
            attempt.update(
                {
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "held_after_failure": getattr(held_after_failure, "name", None),
                }
            )
            if held_after_failure is None:
                try:
                    obj.set_position_orientation(
                        position=raw_position,
                        orientation=raw_quaternion,
                        frame="world",
                    )
                    for velocity_name in ("set_linear_velocity", "set_angular_velocity"):
                        velocity_setter = getattr(obj, velocity_name, None)
                        if callable(velocity_setter):
                            velocity_setter(th.zeros_like(raw_position))
                    for cleanup_index in range(30):
                        control.hold()
                        bindings._raw_step(
                            evaluator,
                            f"oracle.remaining.{index:02d}.failure_cleanup.{cleanup_index:03d}",
                        )
                    attempt["failure_cleanup"] = "restored_original_pose_and_settled"
                except Exception as cleanup_exc:
                    attempt["failure_cleanup"] = "FAIL"
                    attempt["failure_cleanup_error"] = str(cleanup_exc)
        result["attempts"].append(attempt)
        _write_json(run_dir / "private" / "oracle_remaining_toys_result.json", result)
        if starter._get_obj_in_hand() is not None:
            result["stopped_early_reason"] = "hand_not_empty_after_failed_toy"
            break

    all_goal_keys = ("tennis_ball.n.01_1",) + tuple(item[0] for item in sequence)
    result["inside_leaf_truth"] = {
        key: bool(
            _scope_object(task, key).states[object_states.Inside].get_value(toy_box)
        )
        for key in all_goal_keys
    }
    result["full_bddl_inside_success"] = all(result["inside_leaf_truth"].values())
    result["status"] = "PASS" if result["full_bddl_inside_success"] else "PARTIAL"
    _write_json(run_dir / "private" / "oracle_remaining_toys_result.json", result)
    _record_stage(
        stages,
        "ORACLE_REMAINING_TOYS_COMPLETE",
        status=result["status"],
        inside_success_count=result["inside_success_count"],
        full_bddl_inside_success=result["full_bddl_inside_success"],
    )
    return result


def _oracle_edge_calibration_phase(
    *,
    evaluator: Any,
    control: Any,
    bindings: Any,
    build_bindings: Callable[[], Any],
    run_dir: Path,
    stages: Path,
) -> tuple[dict[str, Any], int]:
    """Prove the official grasp/place chain on a private tabletop-edge variant.

    This phase deliberately changes the tennis-ball initial pose and therefore
    can never be used for an official score.  It exists only to distinguish an
    unreachable original placement from a broken grasp executor.  Every state
    write is recorded and each candidate starts from ``Evaluator.reset()`` in
    the same resident scene.
    """

    from omnigibson import object_states
    from omnigibson.action_primitives.starter_semantic_action_primitives import (
        StarterSemanticActionPrimitiveSet,
    )
    import torch as th

    _record_stage(stages, "ORACLE_EDGE_CALIBRATION_START")
    task = evaluator.env.task
    ball = _scope_object(task, "tennis_ball.n.01_1")
    table = _scope_object(task, "table.n.02_1")
    toy_box = _scope_object(task, "toy_box.n.01_1")
    table_minimum, table_maximum = _private_object_aabb(table)
    ball_extent = _private_object_aabb_extent(ball)
    original_ball_position, _ = _private_object_pose(ball)
    robot_position, _ = _private_robot_pose(evaluator.robot)
    placement_mode = os.environ.get(
        "BEHAVIOR_HL_ORACLE_CALIBRATION_PLACEMENT", "local_toward_robot"
    )
    if placement_mode == "local_toward_robot":
        local_displacements = _oracle_local_displacements(
            os.environ.get("BEHAVIOR_HL_ORACLE_LOCAL_DISPLACEMENTS_M")
        )
        candidates = local_support_placements_toward_robot(
            object_center_xyz=original_ball_position,
            current_base_xy=robot_position[:2],
            displacements_m=local_displacements,
        )
    elif placement_mode == "table_aabb_edge":
        candidates = table_edge_placements(
            table_aabb_min=table_minimum,
            table_aabb_max=table_maximum,
            object_extent_xyz=ball_extent,
            current_base_xy=robot_position[:2],
            inset_m=0.08,
            clearance_m=0.004,
            lateral_fractions=(0.0, -0.35, 0.35),
        )
    else:
        raise RuntimeError(
            "BEHAVIOR_HL_ORACLE_CALIBRATION_PLACEMENT must be "
            "local_toward_robot or table_aabb_edge"
        )
    maximum_candidates = int(
        os.environ.get("BEHAVIOR_HL_ORACLE_EDGE_MAX_CANDIDATES", "4")
    )
    if maximum_candidates <= 0:
        raise RuntimeError("BEHAVIOR_HL_ORACLE_EDGE_MAX_CANDIDATES must be positive")
    candidates = candidates[:maximum_candidates]
    result: dict[str, Any] = {
        "method": "private_local_support_variant_official_grasp_place_inside",
        "status": "FAIL",
        "task": "picking_up_toys",
        "target_category": str(getattr(ball, "category", "")),
        "container_category": str(getattr(toy_box, "category", "")),
        "candidate_ids": [item.placement_id for item in candidates],
        "placement_mode": placement_mode,
        "original_ball_position_private": original_ball_position.tolist(),
        "robot_position_private": robot_position.tolist(),
        "table_aabb_min_private": table_minimum.tolist(),
        "table_aabb_max_private": table_maximum.tolist(),
        "candidate_attempts": [],
        "physical_grasp_executions": 0,
        "physical_grasp_successes": 0,
        "physical_place_executions": 0,
        "direct_state_modification": True,
        "direct_state_modification_scope": (
            "all_six_goal_toys"
            if os.environ.get("BEHAVIOR_HL_ORACLE_CONTINUE_REMAINING_TOYS") == "1"
            else "tennis_ball_position_only"
        ),
        "kinematic_research": True,
        "official_challenge_eligible": False,
        "strict_production_eligible": False,
    }
    reset_index = 0
    for index, placement in enumerate(candidates):
        trial_bindings = bindings if index == 0 else build_bindings()
        starter = trial_bindings._starter_for(evaluator)
        raw_original_position, raw_original_quaternion = ball.get_position_orientation()
        original_position = _native_vector(
            raw_original_position, 3, "original object position"
        )
        original_quaternion = _native_vector(
            raw_original_quaternion, 4, "original object quaternion"
        )
        attempt: dict[str, Any] = {
            "candidate_index": index,
            "placement_id": placement.placement_id,
            "table_edge": getattr(placement, "table_edge", None),
            "inset_m": getattr(placement, "inset_m", None),
            "requested_displacement_m": getattr(
                placement, "displacement_m", None
            ),
            "original_position_private": original_position.tolist(),
            "requested_position_private": list(placement.object_center_xyz),
            "state_modification_applied": False,
            "status": "FAIL",
        }
        trace_path = run_dir / "private" / "oracle_edge_calibration_trace.jsonl"
        try:
            setter = getattr(ball, "set_position_orientation", None)
            if not callable(setter):
                raise RuntimeError("private Oracle object pose setter is unavailable")
            setter(
                position=th.as_tensor(
                    placement.object_center_xyz,
                    dtype=raw_original_position.dtype,
                    device=raw_original_position.device,
                ),
                orientation=raw_original_quaternion,
                frame="world",
            )
            zero_velocity = th.zeros_like(raw_original_position)
            for velocity_name in ("set_linear_velocity", "set_angular_velocity"):
                velocity_setter = getattr(ball, velocity_name, None)
                if callable(velocity_setter):
                    velocity_setter(zero_velocity)
            attempt["state_modification_applied"] = True
            _append_private_trace(
                trace_path,
                {
                    "phase": "direct_state_modification",
                    "placement_id": placement.placement_id,
                    "object": getattr(ball, "name", None),
                    "original_position_private": original_position.tolist(),
                    "requested_position_private": list(placement.object_center_xyz),
                    "official_challenge_eligible": False,
                },
            )
            settle_steps = 30
            for settle_index in range(settle_steps):
                control.hold()
                trial_bindings._raw_step(
                    evaluator,
                    f"oracle.edge.{placement.placement_id}.settle.{settle_index:03d}",
                )
            settled_position, _ = _private_object_pose(ball)
            attempt["settled_position_private"] = settled_position.tolist()
            attempt["settle_displacement_m"] = float(
                np.linalg.norm(
                    settled_position
                    - np.asarray(placement.object_center_xyz, dtype=np.float64)
                )
            )
            attempt["ontop_table_after_settle"] = bool(
                ball.states[object_states.OnTop].get_value(table)
            )
            if not attempt["ontop_table_after_settle"]:
                raise RuntimeError("edge calibration ball is not OnTop(table) after settle")

            attempt["physical_grasp_attempted"] = True
            result["physical_grasp_executions"] += 1
            grasp = _run_official_primitive(
                evaluator=evaluator,
                starter=starter,
                primitive=StarterSemanticActionPrimitiveSet.GRASP,
                target=ball,
                phase=f"oracle.edge.{placement.placement_id}.grasp",
                trace_path=trace_path,
            )
            attempt["grasp"] = grasp
            held = starter._get_obj_in_hand()
            after_grasp_position, _ = _private_object_pose(ball)
            lift_m = float(after_grasp_position[2] - settled_position[2])
            attempt.update(
                {
                    "held_after_grasp": getattr(held, "name", None),
                    "after_grasp_position_private": after_grasp_position.tolist(),
                    "observed_lift_m": lift_m,
                }
            )
            if held is not ball:
                raise RuntimeError("official grasp did not hold the calibrated tennis ball")
            if lift_m < 0.05:
                raise RuntimeError("official grasp did not lift the calibrated tennis ball by 5 cm")
            result["physical_grasp_successes"] += 1

            place = _run_official_primitive(
                evaluator=evaluator,
                starter=starter,
                primitive=StarterSemanticActionPrimitiveSet.PLACE_INSIDE,
                target=toy_box,
                phase=f"oracle.edge.{placement.placement_id}.place_inside",
                trace_path=trace_path,
            )
            result["physical_place_executions"] += 1
            attempt["place"] = place
            attempt["inside_after_place"] = bool(
                ball.states[object_states.Inside].get_value(toy_box)
            )
            if not attempt["inside_after_place"]:
                raise RuntimeError("official place did not establish Inside(ball, toy_box)")
            attempt["status"] = "PASS"
            result["candidate_attempts"].append(attempt)
            result.update(
                {
                    "status": "PASS",
                    "successful_placement": placement.placement_id,
                    "observed_lift_m": lift_m,
                    "inside_after_place": True,
                }
            )
            if os.environ.get("BEHAVIOR_HL_ORACLE_CONTINUE_REMAINING_TOYS") == "1":
                result["remaining_toys"] = _oracle_remaining_toys_phase(
                    evaluator=evaluator,
                    control=control,
                    bindings=trial_bindings,
                    starter=starter,
                    toy_box=toy_box,
                    run_dir=run_dir,
                    stages=stages,
                    displacement_m=_oracle_local_displacements(
                        os.environ.get(
                            "BEHAVIOR_HL_ORACLE_REMAINING_TOY_DISPLACEMENT_M",
                            "0.20",
                        )
                    )[0],
                )
                result["full_bddl_inside_success"] = result["remaining_toys"].get(
                    "full_bddl_inside_success", False
                )
            break
        except Exception as exc:
            attempt.update(
                {
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            result["candidate_attempts"].append(attempt)
        if index + 1 < len(candidates):
            reset_index = _reset_same_scene(
                evaluator=evaluator,
                control=control,
                stages=stages,
                phase=f"after_oracle_edge_{index:02d}",
                run_dir=run_dir,
                reset_index=reset_index,
            )
    _write_json(run_dir / "private" / "oracle_edge_calibration_result.json", result)
    _record_stage(
        stages,
        "ORACLE_EDGE_CALIBRATION_COMPLETE",
        status=result.get("status"),
        successful_placement=result.get("successful_placement"),
        physical_grasp_successes=result.get("physical_grasp_successes"),
        inside_after_place=result.get("inside_after_place"),
    )
    return result, reset_index


def make_diagnostic_sweep_callback(
    *,
    build_bindings: Callable[[], Any],
) -> Callable[..., Mapping[str, Any]]:
    if not callable(build_bindings):
        raise TypeError("build_bindings must be callable")

    def callback(
        *,
        evaluator: Any,
        control: Any,
        bindings: Any,
        run_dir: Path,
        stages: Path,
    ) -> Mapping[str, Any]:
        reset_index = 0
        strategies: list[dict[str, Any]] = []

        if os.environ.get("BEHAVIOR_HL_ORACLE_GENERIC_GRASP") == "1":
            oracle, oracle_resets = _oracle_generic_grasp_phase(
                evaluator=evaluator,
                control=control,
                bindings=bindings,
                run_dir=run_dir,
                stages=stages,
            )
        elif os.environ.get("BEHAVIOR_HL_ORACLE_SCREENED_CENTER_GRASP") == "1":
            oracle, oracle_resets = _oracle_screened_center_grasp_phase(
                evaluator=evaluator,
                control=control,
                bindings=bindings,
                run_dir=run_dir,
                stages=stages,
            )
        elif os.environ.get("BEHAVIOR_HL_ORACLE_EDGE_CALIBRATION") == "1":
            oracle, oracle_resets = _oracle_edge_calibration_phase(
                evaluator=evaluator,
                control=control,
                bindings=bindings,
                build_bindings=build_bindings,
                run_dir=run_dir,
                stages=stages,
            )
        else:
            oracle, oracle_resets = _oracle_targeted_phase(
                evaluator=evaluator,
                control=control,
                bindings=bindings,
                build_bindings=build_bindings,
                run_dir=run_dir,
                stages=stages,
            )
        strategies.append(oracle)
        reset_index += oracle_resets

        if os.environ.get("BEHAVIOR_HL_ORACLE_ONLY") == "1":
            reset_index = _reset_same_scene(
                evaluator=evaluator,
                control=control,
                stages=stages,
                phase="after_oracle_only",
                run_dir=run_dir,
                reset_index=reset_index,
            )
            _record_stage(
                stages,
                "ORACLE_ONLY_SWEEP_PASS",
                strategy_count=1,
                physical_grasp_executions=oracle.get("physical_grasp_executions"),
            )
            return {
                "schema_version": "behavior2026.b1-diagnostic-sweep/v1",
                "task": "picking_up_toys",
                "split": "train",
                "instance_id": 0,
                "target_probe": "tennis_ball",
                "model": "qwen3.8-max-preview",
                "scene_loads_attempted": 1,
                "successful_scene_loads": 1,
                "in_memory_resets": reset_index,
                "strategies": strategies,
                "oracle_only": True,
                "official_challenge_eligible": False,
                "kinematic_research": True,
                "privileged_policy_inputs": False,
                "teleport_enabled": False,
                "viewer_camera_control": False,
            }
        reset_index = _reset_same_scene(
            evaluator=evaluator,
            control=control,
            stages=stages,
            phase="after_oracle",
            run_dir=run_dir,
            reset_index=reset_index,
        )

        probe_bindings = build_bindings()
        probe = _candidate_probe(
            evaluator=evaluator,
            bindings=probe_bindings,
            run_dir=run_dir,
            stages=stages,
        )
        strategies.append(probe)
        reset_index = _reset_same_scene(
            evaluator=evaluator,
            control=control,
            stages=stages,
            phase="after_ik_probe",
            run_dir=run_dir,
            reset_index=reset_index,
        )

        qwen_default = _qwen_phase(
            evaluator=evaluator,
            control=control,
            bindings=build_bindings(),
            run_dir=run_dir,
            stages=stages,
            phase="qwen_relaxed_gate",
            maximum_model_calls=32,
        )
        strategies.append(qwen_default)
        reset_index = _reset_same_scene(
            evaluator=evaluator,
            control=control,
            stages=stages,
            phase="after_qwen_relaxed_gate",
            run_dir=run_dir,
            reset_index=reset_index,
        )

        preposition_bindings = build_bindings()
        preposition = _preposition_variant(
            evaluator=evaluator,
            control=control,
            bindings=preposition_bindings,
            run_dir=run_dir,
            phase="qwen_relaxed_prepositioned",
        )
        qwen_prepositioned = _qwen_phase(
            evaluator=evaluator,
            control=control,
            bindings=preposition_bindings,
            run_dir=run_dir,
            stages=stages,
            phase="qwen_relaxed_prepositioned",
            maximum_model_calls=24,
        )
        qwen_prepositioned = {**preposition, **qwen_prepositioned}
        strategies.append(qwen_prepositioned)
        reset_index = _reset_same_scene(
            evaluator=evaluator,
            control=control,
            stages=stages,
            phase="after_qwen_relaxed_prepositioned",
            run_dir=run_dir,
            reset_index=reset_index,
        )

        return {
            "schema_version": "behavior2026.b1-diagnostic-sweep/v1",
            "task": "picking_up_toys",
            "split": "train",
            "instance_id": 0,
            "target_probe": "tennis_ball",
            "model": "qwen3.8-max-preview",
            "scene_loads_attempted": 1,
            "successful_scene_loads": 1,
            "in_memory_resets": reset_index,
            "strategies": strategies,
            "official_challenge_eligible": False,
            "kinematic_research": True,
            "privileged_policy_inputs": False,
            "teleport_enabled": False,
            "viewer_camera_control": False,
        }

    return callback
