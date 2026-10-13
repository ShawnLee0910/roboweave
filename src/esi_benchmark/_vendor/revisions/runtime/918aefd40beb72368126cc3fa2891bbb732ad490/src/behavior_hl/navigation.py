"""Planar ego navigation shared by the pure tests and the OmniGibson adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from behavior_hl.actions import ActionName, EGO_NAVIGATION_ACTIONS


class NavigationBlocked(RuntimeError):
    """Raised before a base move that violates a navigation guard."""


@dataclass(frozen=True, slots=True)
class PlanarPose:
    x: float
    y: float
    z: float
    yaw: float


NavigationGuard = Callable[[PlanarPose, PlanarPose], bool]


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class PlanarNavigator:
    """Proposes fixed-size robot-base moves while locking vertical position."""

    def __init__(
        self,
        *,
        move_distance: float = 0.25,
        turn_angle_degrees: float = 15.0,
        guard: NavigationGuard | None = None,
    ) -> None:
        if move_distance <= 0:
            raise ValueError("move_distance must be positive")
        if not 0 < turn_angle_degrees <= 180:
            raise ValueError("turn_angle_degrees must be in (0, 180]")
        self.move_distance = float(move_distance)
        self.turn_angle = math.radians(turn_angle_degrees)
        self.guard = guard or (lambda _start, _target: True)

    def transition(self, pose: PlanarPose, action: ActionName) -> PlanarPose:
        if action not in EGO_NAVIGATION_ACTIONS:
            raise ValueError(f"{action.value!r} is not an ego-navigation action")

        if action is ActionName.TURN_LEFT:
            target = PlanarPose(pose.x, pose.y, pose.z, normalize_angle(pose.yaw + self.turn_angle))
        elif action is ActionName.TURN_RIGHT:
            target = PlanarPose(pose.x, pose.y, pose.z, normalize_angle(pose.yaw - self.turn_angle))
        else:
            forward_x, forward_y = math.cos(pose.yaw), math.sin(pose.yaw)
            left_x, left_y = -forward_y, forward_x
            directions = {
                ActionName.MOVE_FORWARD: (forward_x, forward_y),
                ActionName.MOVE_BACK: (-forward_x, -forward_y),
                ActionName.MOVE_LEFT: (left_x, left_y),
                ActionName.MOVE_RIGHT: (-left_x, -left_y),
            }
            dx, dy = directions[action]
            target = PlanarPose(
                pose.x + self.move_distance * dx,
                pose.y + self.move_distance * dy,
                pose.z,
                pose.yaw,
            )

        # This is an invariant, not merely a convention.
        if target.z != pose.z:
            raise AssertionError("planar navigation changed z")
        if not self.guard(pose, target):
            raise NavigationBlocked(
                f"{action.value} rejected: target ({target.x:.3f}, {target.y:.3f}) is not locally traversable"
            )
        return target

