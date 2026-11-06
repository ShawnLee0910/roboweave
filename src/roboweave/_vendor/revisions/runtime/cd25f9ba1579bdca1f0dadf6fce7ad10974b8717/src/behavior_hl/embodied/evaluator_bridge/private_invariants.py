"""Evaluator-private reset snapshots and comparisons.

This module may inspect simulator truth for audit artifacts. Its outputs must
never be sent to the policy process.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value, dtype=np.float64)


def _pose(entity: Any) -> dict[str, list[float]]:
    position, orientation = entity.get_position_orientation()
    return {
        "position": _array(position).tolist(),
        "orientation_xyzw": _array(orientation).tolist(),
    }


def capture_reset_snapshot(evaluator: Any) -> dict[str, Any]:
    """Capture robot and task-scope state after an official reset."""

    robot = evaluator.robot
    snapshot: dict[str, Any] = {
        "robot": {
            **_pose(robot),
            "joint_positions": _array(
                robot.get_joint_positions()
            ).tolist(),
        },
        "task_scope": {},
        "skipped_scope": [],
    }
    for scope_name, entity in sorted(
        evaluator.env.task.object_scope.items()
    ):
        if entity is None or scope_name == "agent.n.01_1":
            snapshot["skipped_scope"].append(scope_name)
            continue
        if not hasattr(entity, "get_position_orientation"):
            snapshot["skipped_scope"].append(scope_name)
            continue
        try:
            snapshot["task_scope"][scope_name] = _pose(entity)
        except (AttributeError, RuntimeError):
            snapshot["skipped_scope"].append(scope_name)
    return snapshot


def quaternion_angle_rad(first: Any, second: Any) -> float:
    """Shortest angular distance for xyzw quaternions."""

    left = _array(first)
    right = _array(second)
    if left.shape != (4,) or right.shape != (4,):
        raise ValueError("quaternions must have shape (4,)")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("quaternions must be non-zero")
    cosine = abs(float(np.dot(left / left_norm, right / right_norm)))
    return 2.0 * math.acos(float(np.clip(cosine, -1.0, 1.0)))


def compare_reset_snapshots(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    joint_tolerance_rad: float,
) -> dict[str, Any]:
    """Compare two snapshots and return a private fail-closed report."""

    for name, value in (
        ("position_tolerance_m", position_tolerance_m),
        ("orientation_tolerance_rad", orientation_tolerance_rad),
        ("joint_tolerance_rad", joint_tolerance_rad),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    failures: list[dict[str, Any]] = []
    expected_names = set(expected["task_scope"])
    actual_names = set(actual["task_scope"])
    if expected_names != actual_names:
        failures.append(
            {
                "field": "task_scope.keys",
                "missing": sorted(expected_names - actual_names),
                "unknown": sorted(actual_names - expected_names),
            }
        )

    def compare_pose(label: str, left: dict, right: dict) -> None:
        position_error = float(
            np.linalg.norm(
                _array(left["position"]) - _array(right["position"])
            )
        )
        orientation_error = quaternion_angle_rad(
            left["orientation_xyzw"],
            right["orientation_xyzw"],
        )
        if position_error > position_tolerance_m:
            failures.append(
                {
                    "field": f"{label}.position",
                    "error_m": position_error,
                    "tolerance_m": position_tolerance_m,
                }
            )
        if orientation_error > orientation_tolerance_rad:
            failures.append(
                {
                    "field": f"{label}.orientation",
                    "error_rad": orientation_error,
                    "tolerance_rad": orientation_tolerance_rad,
                }
            )

    compare_pose("robot", expected["robot"], actual["robot"])
    expected_joints = _array(expected["robot"]["joint_positions"])
    actual_joints = _array(actual["robot"]["joint_positions"])
    if expected_joints.shape != actual_joints.shape:
        failures.append(
            {
                "field": "robot.joint_positions.shape",
                "expected": list(expected_joints.shape),
                "actual": list(actual_joints.shape),
            }
        )
    else:
        joint_error = float(np.max(np.abs(expected_joints - actual_joints)))
        if joint_error > joint_tolerance_rad:
            failures.append(
                {
                    "field": "robot.joint_positions",
                    "max_error_rad": joint_error,
                    "tolerance_rad": joint_tolerance_rad,
                }
            )
    for scope_name in sorted(expected_names & actual_names):
        compare_pose(
            f"task_scope.{scope_name}",
            expected["task_scope"][scope_name],
            actual["task_scope"][scope_name],
        )
    return {
        "passed": not failures,
        "failures": failures,
        "expected_task_scope_count": len(expected_names),
        "actual_task_scope_count": len(actual_names),
        "expected_skipped_scope": expected.get("skipped_scope", []),
        "actual_skipped_scope": actual.get("skipped_scope", []),
    }

