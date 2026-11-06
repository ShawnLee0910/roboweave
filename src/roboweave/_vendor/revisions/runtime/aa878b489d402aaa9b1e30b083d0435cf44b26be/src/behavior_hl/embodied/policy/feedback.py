"""Sensor-derived public feedback with fixed, non-leaking messages."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


class PublicStatus(str, Enum):
    COMPLETED = "completed"
    MOTION_PARTIAL = "motion_partial"
    BLOCKED = "blocked"
    INVALID_POINT = "invalid_point"
    INVALID_DEPTH = "invalid_depth"
    REPOSITION_REQUIRED = "reposition_required"
    OUT_OF_REACH = "out_of_reach"
    IK_FAILED = "ik_failed"
    GRASP_LIKELY = "grasp_likely"
    GRASP_UNCERTAIN = "grasp_uncertain"
    CAVITY_UNCERTAIN = "cavity_uncertain"
    PLACE_UNSTABLE = "place_unstable"
    UNSAFE_WHILE_CARRYING = "unsafe_while_carrying"
    JOINT_STALLED = "joint_stalled"


PUBLIC_STATUS_MESSAGES = {
    PublicStatus.COMPLETED: "The requested local action completed.",
    PublicStatus.MOTION_PARTIAL: "The base moved only part of the requested local distance.",
    PublicStatus.BLOCKED: "Motion stalled before the requested local distance.",
    PublicStatus.INVALID_POINT: "The selected image point is invalid.",
    PublicStatus.INVALID_DEPTH: "No reliable onboard depth was available at the selected point.",
    PublicStatus.REPOSITION_REQUIRED: (
        "Move the base with effective translation before requesting another reach query."
    ),
    PublicStatus.OUT_OF_REACH: "The selected point is outside the current arm workspace.",
    PublicStatus.IK_FAILED: "A safe local arm solution was not found.",
    PublicStatus.GRASP_LIKELY: (
        "The gripper stopped before the calibrated empty-close limit; inspect the wrist view."
    ),
    PublicStatus.GRASP_UNCERTAIN: "The sensor evidence for grasping is uncertain.",
    PublicStatus.CAVITY_UNCERTAIN: "The visible opening does not provide a reliable safe release patch.",
    PublicStatus.PLACE_UNSTABLE: "The visible placement appears unstable; inspect from another view.",
    PublicStatus.UNSAFE_WHILE_CARRYING: "This view adjustment is outside the validated carry-safe interval.",
    PublicStatus.JOINT_STALLED: "A commanded robot joint stopped before the local target.",
}


@dataclass(frozen=True, slots=True)
class BodyOdometry:
    forward_m: float
    left_m: float
    yaw_deg: float

    def to_dict(self) -> dict[str, float]:
        return {
            "forward_m": _finite(self.forward_m, "forward_m"),
            "left_m": _finite(self.left_m, "left_m"),
            "yaw_deg": _finite(self.yaw_deg, "yaw_deg"),
        }


@dataclass(frozen=True, slots=True)
class PublicActionResult:
    status: PublicStatus
    simulator_steps: int
    hand: str | None = None
    body_odometry: BodyOdometry | None = None

    def __post_init__(self) -> None:
        if isinstance(self.simulator_steps, bool) or self.simulator_steps < 0:
            raise ValueError("simulator_steps must be a non-negative integer")
        if self.hand not in {None, "left", "right"}:
            raise ValueError("hand must be left, right, or None")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status.value,
            "simulator_steps": self.simulator_steps,
            "message": PUBLIC_STATUS_MESSAGES[self.status],
        }
        if self.hand is not None:
            payload["hand"] = self.hand
        if self.body_odometry is not None:
            payload["body_odometry"] = self.body_odometry.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class GraspProxyThresholds:
    empty_close_qpos: tuple[float, float]
    minimum_qpos_gap: float
    qvel_epsilon: float
    stable_steps: int

    def __post_init__(self) -> None:
        if self.minimum_qpos_gap <= 0.0:
            raise ValueError("minimum_qpos_gap must be positive")
        if self.qvel_epsilon <= 0.0:
            raise ValueError("qvel_epsilon must be positive")
        if self.stable_steps <= 0:
            raise ValueError("stable_steps must be positive")
        for index, value in enumerate(self.empty_close_qpos):
            _finite(value, f"empty_close_qpos[{index}]")


def grasp_proxy_result(
    *,
    hand: str,
    qpos: tuple[float, float],
    qvel: tuple[float, float],
    consecutive_stable_steps: int,
    wrist_visual_consistent: bool,
    simulator_steps: int,
    thresholds: GraspProxyThresholds,
) -> PublicActionResult:
    """Classify only proprioceptive and wrist-image evidence."""

    if hand not in {"left", "right"}:
        raise ValueError("hand must be left or right")
    qpos_gap = max(
        abs(_finite(value, f"qpos[{index}]") - thresholds.empty_close_qpos[index])
        for index, value in enumerate(qpos)
    )
    max_qvel = max(
        abs(_finite(value, f"qvel[{index}]"))
        for index, value in enumerate(qvel)
    )
    likely = (
        qpos_gap >= thresholds.minimum_qpos_gap
        and max_qvel <= thresholds.qvel_epsilon
        and consecutive_stable_steps >= thresholds.stable_steps
        and wrist_visual_consistent
    )
    return PublicActionResult(
        status=(
            PublicStatus.GRASP_LIKELY
            if likely
            else PublicStatus.GRASP_UNCERTAIN
        ),
        simulator_steps=simulator_steps,
        hand=hand,
    )


def movement_result(
    *,
    status: PublicStatus,
    simulator_steps: int,
    forward_m: float,
    left_m: float,
    yaw_deg: float,
) -> PublicActionResult:
    if status not in {
        PublicStatus.COMPLETED,
        PublicStatus.MOTION_PARTIAL,
        PublicStatus.BLOCKED,
    }:
        raise ValueError("movement_result received a non-movement status")
    return PublicActionResult(
        status=status,
        simulator_steps=simulator_steps,
        body_odometry=BodyOdometry(forward_m, left_m, yaw_deg),
    )


def _finite(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result
