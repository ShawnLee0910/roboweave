"""Fail-closed R1Pro live binding for the physical-grasp runtime plugin.

This module has no OmniGibson imports.  The resident evaluator must provide a
small, explicitly versioned motion driver around the official runtime APIs.
That keeps importing this module CPU-only and prevents the current private
``_move_hand`` generator from being mistaken for a non-mutating IK preflight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from behavior_hl.embodied.evaluator_bridge.pixel_manipulation import (
    robot_base_point_to_world,
    validate_local_manipulation_target,
)
from behavior_hl.embodied.policy.rgbd_grounding import (
    RGBDGroundingError,
    GroundedPoint,
    ground_image_point,
)
from behavior_hl.embodied.runtime.plugins.grasp import (
    PHYSICAL_GRASP_BACKEND_SERVICE,
    PRIVATE_GRASP_AUDIT_SERVICE,
    GraspMotionPhase,
    GraspPhaseSample,
    IKResolution,
    PrivateGraspAuditEvent,
    RGBDPointResolution,
    RGBDPointStatus,
    ReachResolution,
)


R1PRO_GRASP_DRIVER_VERSION = "behavior2026.r1pro-grasp-driver/v1"


class R1ProGraspBindingError(RuntimeError):
    """Raised before action execution when a live capability is unavailable."""


@dataclass(frozen=True, slots=True)
class RobotPose:
    """Private robot world pose captured with the selected RGB-D frame."""

    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position",
            _finite_vector(self.position, 3, "position"),
        )
        object.__setattr__(
            self,
            "orientation_xyzw",
            _finite_vector(
                self.orientation_xyzw,
                4,
                "orientation_xyzw",
            ),
        )


@dataclass(frozen=True, slots=True)
class R1ProGraspBindingConfig:
    """Frozen public grounding and local-workspace limits."""

    intrinsic_by_camera: Mapping[str, Any]
    depth_radius_px: int = 1
    minimum_depth_m: float = 0.05
    maximum_depth_m: float = 5.0
    maximum_depth_spread_m: float = 0.08
    maximum_horizontal_reach_m: float = 1.30
    minimum_target_height_m: float = 0.05
    maximum_target_height_m: float = 1.80

    def __post_init__(self) -> None:
        if (
            isinstance(self.depth_radius_px, bool)
            or not isinstance(self.depth_radius_px, int)
            or not 0 <= self.depth_radius_px <= 3
        ):
            raise ValueError("depth_radius_px must be an integer in [0, 3]")
        frozen_intrinsics: dict[str, np.ndarray] = {}
        for camera, value in self.intrinsic_by_camera.items():
            if not isinstance(camera, str) or not camera:
                raise ValueError("camera roles must be non-empty strings")
            matrix = np.asarray(value, dtype=np.float64)
            if (
                matrix.shape != (3, 3)
                or not np.isfinite(matrix).all()
                or matrix[0, 0] <= 0.0
                or matrix[1, 1] <= 0.0
            ):
                raise ValueError(
                    f"intrinsic for {camera!r} must be finite and 3x3"
                )
            matrix = np.array(matrix, copy=True)
            matrix.setflags(write=False)
            frozen_intrinsics[camera] = matrix
        if not frozen_intrinsics:
            raise ValueError("at least one onboard camera intrinsic is required")
        object.__setattr__(
            self,
            "intrinsic_by_camera",
            MappingProxyType(frozen_intrinsics),
        )
        for name in (
            "minimum_depth_m",
            "maximum_depth_m",
            "maximum_depth_spread_m",
            "maximum_horizontal_reach_m",
            "minimum_target_height_m",
            "maximum_target_height_m",
        ):
            value = _finite_scalar(getattr(self, name), name)
            object.__setattr__(self, name, value)
        if not 0.0 < self.minimum_depth_m < self.maximum_depth_m:
            raise ValueError("depth interval must be positive and ordered")
        if self.maximum_depth_spread_m <= 0.0:
            raise ValueError("maximum_depth_spread_m must be positive")
        if self.maximum_horizontal_reach_m <= 0.0:
            raise ValueError("maximum_horizontal_reach_m must be positive")
        if self.minimum_target_height_m >= self.maximum_target_height_m:
            raise ValueError("target-height interval must be ordered")


class R1ProGraspMotionDriver(Protocol):
    """Versioned resident wrapper around official R1Pro/primitive APIs."""

    api_contract_version: str

    def preflight_ik(
        self,
        *,
        world_point: tuple[float, float, float],
        hand: str,
    ) -> Any | None:
        """Return an opaque plan only after non-mutating IK validation."""

    def tick_motion(
        self,
        *,
        phase: GraspMotionPhase,
        plan_token: Any,
        hand: str,
    ) -> GraspPhaseSample:
        """Advance one official 23D action and return bounded sensor evidence."""

    def enable_assisted_grasp(
        self,
        *,
        plan_token: Any,
        hand: str,
    ) -> bool:
        """Confirm the official contact-gated assisted grasp."""

    def hold_safe(self, *, hand: str) -> None:
        """Hold the current safe command without teleporting."""


@dataclass(frozen=True, slots=True)
class GraspBindingCapabilityReport:
    available: bool
    missing_capabilities: tuple[str, ...]
    evidence: Mapping[str, bool]
    official_api_notes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "missing_capabilities",
            tuple(self.missing_capabilities),
        )
        object.__setattr__(
            self,
            "evidence",
            MappingProxyType(dict(self.evidence)),
        )
        object.__setattr__(
            self,
            "official_api_notes",
            tuple(self.official_api_notes),
        )

    def require_available(self) -> None:
        if not self.available:
            missing = ", ".join(self.missing_capabilities)
            raise R1ProGraspBindingError(
                f"R1Pro physical-grasp binding unavailable: {missing}"
            )


@dataclass(frozen=True, slots=True)
class _TargetToken:
    grounded: GroundedPoint = field(repr=False)
    robot_pose: RobotPose = field(repr=False)


class PrivateGraspArtifactSink:
    """Write-only adapter for the evaluator's private JSONL appender."""

    def __init__(
        self,
        append_event: Callable[[Mapping[str, Any]], None],
    ) -> None:
        if not callable(append_event):
            raise TypeError("append_event must be callable")
        self._append_event = append_event

    def record(self, event: PrivateGraspAuditEvent) -> None:
        if not isinstance(event, PrivateGraspAuditEvent):
            raise TypeError("event must be PrivateGraspAuditEvent")
        self._append_event(
            {
                "schema_version": "behavior2026.private-grasp-audit/v1",
                "event": event.event,
                "details": dict(event.details),
            }
        )


def preflight_r1pro_grasp_binding(
    *,
    snapshot_provider: Any,
    robot_pose_provider: Any,
    motion_driver: Any,
    private_audit_sink: Any = None,
) -> GraspBindingCapabilityReport:
    """Inspect callable capabilities without invoking simulator state."""

    evidence = {
        "public_rgbd_snapshot_provider": callable(snapshot_provider),
        "private_robot_pose_provider": callable(robot_pose_provider),
        "versioned_motion_driver": (
            getattr(motion_driver, "api_contract_version", None)
            == R1PRO_GRASP_DRIVER_VERSION
        ),
        "non_mutating_ik_preflight": callable(
            getattr(motion_driver, "preflight_ik", None)
        ),
        "bounded_motion_tick": callable(
            getattr(motion_driver, "tick_motion", None)
        ),
        "contact_gated_assisted_grasp": callable(
            getattr(motion_driver, "enable_assisted_grasp", None)
        ),
        "safe_hold": callable(
            getattr(motion_driver, "hold_safe", None)
        ),
        "write_only_private_audit": (
            private_audit_sink is None
            or callable(getattr(private_audit_sink, "record", None))
        ),
    }
    missing = tuple(
        name for name, present in evidence.items() if not present
    )
    return GraspBindingCapabilityReport(
        available=not missing,
        missing_capabilities=missing,
        evidence=evidence,
        official_api_notes=(
            "Robot._get_proprioception_dict exposes gripper_{arm}_qpos/qvel.",
            "Assisted grasp requires official finger contacts and raycast intersection.",
            "StarterSemanticActionPrimitives._execute_grasp closes the physical gripper.",
            "Current G4a _move_hand is a mutating generator, not an IK preflight.",
            "Current G4a runner has no carry-safe retract phase.",
        ),
    )


def build_r1pro_grasp_services(
    *,
    config: R1ProGraspBindingConfig,
    snapshot_provider: Callable[[], Mapping[str, Any]],
    robot_pose_provider: Callable[[], RobotPose],
    motion_driver: R1ProGraspMotionDriver,
    private_audit_sink: Any = None,
) -> Mapping[str, Any]:
    """Build the exact services inserted into ``ActionExecutionContext``."""

    report = preflight_r1pro_grasp_binding(
        snapshot_provider=snapshot_provider,
        robot_pose_provider=robot_pose_provider,
        motion_driver=motion_driver,
        private_audit_sink=private_audit_sink,
    )
    report.require_available()
    backend = R1ProPhysicalGraspBackend(
        config=config,
        snapshot_provider=snapshot_provider,
        robot_pose_provider=robot_pose_provider,
        motion_driver=motion_driver,
    )
    services: dict[str, Any] = {
        PHYSICAL_GRASP_BACKEND_SERVICE: backend,
    }
    if private_audit_sink is not None:
        services[PRIVATE_GRASP_AUDIT_SERVICE] = private_audit_sink
    return MappingProxyType(services)


class R1ProPhysicalGraspBackend:
    """Live backend satisfying ``PhysicalGraspBackend`` without task truth."""

    def __init__(
        self,
        *,
        config: R1ProGraspBindingConfig,
        snapshot_provider: Callable[[], Mapping[str, Any]],
        robot_pose_provider: Callable[[], RobotPose],
        motion_driver: R1ProGraspMotionDriver,
    ) -> None:
        self._config = config
        self._snapshot_provider = snapshot_provider
        self._robot_pose_provider = robot_pose_provider
        self._motion_driver = motion_driver

    def resolve_rgbd_point(
        self,
        *,
        camera: str,
        point: tuple[float, float],
        hand: str,
    ) -> RGBDPointResolution:
        del hand
        if camera not in self._config.intrinsic_by_camera:
            return RGBDPointResolution(RGBDPointStatus.INVALID_POINT)
        try:
            grounded = ground_image_point(
                self._snapshot_provider(),
                camera=camera,
                point=point,
                intrinsic_matrix=self._config.intrinsic_by_camera[camera],
                depth_radius_px=self._config.depth_radius_px,
                minimum_depth_m=self._config.minimum_depth_m,
                maximum_depth_m=self._config.maximum_depth_m,
                maximum_depth_spread_m=(
                    self._config.maximum_depth_spread_m
                ),
            )
        except (RGBDGroundingError, KeyError, TypeError, ValueError):
            return RGBDPointResolution(RGBDPointStatus.INVALID_DEPTH)
        pose = self._robot_pose_provider()
        if not isinstance(pose, RobotPose):
            raise R1ProGraspBindingError(
                "robot_pose_provider must return RobotPose"
            )
        return RGBDPointResolution(
            RGBDPointStatus.READY,
            _TargetToken(grounded=grounded, robot_pose=pose),
        )

    def check_reach(
        self,
        *,
        target_token: Any,
        hand: str,
    ) -> ReachResolution:
        del hand
        target = _require_target(target_token)
        try:
            validate_local_manipulation_target(
                target.grounded.robot_base_xyz_m,
                maximum_horizontal_reach_m=(
                    self._config.maximum_horizontal_reach_m
                ),
                minimum_height_m=self._config.minimum_target_height_m,
                maximum_height_m=self._config.maximum_target_height_m,
            )
        except ValueError:
            return ReachResolution(False)
        return ReachResolution(True)

    def solve_ik(
        self,
        *,
        target_token: Any,
        hand: str,
    ) -> IKResolution:
        target = _require_target(target_token)
        world_point = robot_base_point_to_world(
            target.grounded.robot_base_xyz_m,
            robot_position=target.robot_pose.position,
            robot_orientation_xyzw=target.robot_pose.orientation_xyzw,
        )
        plan = self._motion_driver.preflight_ik(
            world_point=tuple(float(value) for value in world_point),
            hand=hand,
        )
        if plan is None:
            return IKResolution(False)
        return IKResolution(True, plan)

    def tick_motion(
        self,
        *,
        phase: GraspMotionPhase,
        plan_token: Any,
        hand: str,
    ) -> GraspPhaseSample:
        sample = self._motion_driver.tick_motion(
            phase=phase,
            plan_token=plan_token,
            hand=hand,
        )
        if not isinstance(sample, GraspPhaseSample):
            raise R1ProGraspBindingError(
                "motion driver returned an invalid GraspPhaseSample"
            )
        return sample

    def enable_assisted_grasp(
        self,
        *,
        plan_token: Any,
        hand: str,
    ) -> bool:
        enabled = self._motion_driver.enable_assisted_grasp(
            plan_token=plan_token,
            hand=hand,
        )
        if not isinstance(enabled, bool):
            raise R1ProGraspBindingError(
                "assisted-grasp result must be bool"
            )
        return enabled

    def hold_safe(self, *, hand: str) -> None:
        self._motion_driver.hold_safe(hand=hand)


def _require_target(value: Any) -> _TargetToken:
    if not isinstance(value, _TargetToken):
        raise R1ProGraspBindingError("invalid private RGB-D target token")
    return value


def _finite_vector(
    value: Sequence[float],
    size: int,
    name: str,
) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {size}D vector")
    return tuple(float(item) for item in array)


def _finite_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number
