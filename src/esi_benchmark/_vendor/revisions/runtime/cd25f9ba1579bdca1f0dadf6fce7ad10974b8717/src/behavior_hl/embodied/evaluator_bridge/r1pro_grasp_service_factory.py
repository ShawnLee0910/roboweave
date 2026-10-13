"""Narrow resident-evaluator factory for R1Pro physical-grasp services."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from behavior_hl.embodied.evaluator_bridge.r1pro_grasp_binding import (
    GraspBindingCapabilityReport,
    PrivateGraspArtifactSink,
    R1ProGraspBindingConfig,
    R1ProGraspBindingError,
    RobotPose,
    build_r1pro_grasp_services,
    preflight_r1pro_grasp_binding,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_starter_grasp_driver import (
    PrivateGraspEvidence,
    PublicGripperWristEvidence,
    R1ProStarterGraspDriver,
    StarterGraspDriverError,
    StarterGraspPoseConfig,
)
from behavior_hl.embodied.policy.observations import CAMERA_RESOLUTIONS
from behavior_hl.embodied.runtime.plugins.grasp import (
    PHYSICAL_GRASP_BACKEND_SERVICE,
    PRIVATE_GRASP_AUDIT_SERVICE,
    GraspMotionPhase,
)


_RESERVED_SERVICE_KEYS = frozenset(
    {
        PHYSICAL_GRASP_BACKEND_SERVICE,
        PRIVATE_GRASP_AUDIT_SERVICE,
    }
)
_FORBIDDEN_PUBLIC_FRAGMENTS = (
    "goal",
    "held",
    "object_scope",
    "robot_pose",
    "success",
    "task_state",
    "world_pose",
)


class R1ProGraspServiceFactoryError(RuntimeError):
    """Raised before any simulator command when factory preflight fails."""


@dataclass(frozen=True, slots=True)
class R1ProGraspServiceBundle:
    """Frozen services plus the capability evidence used to admit them."""

    services: Mapping[str, Any]
    capability_report: GraspBindingCapabilityReport
    frozen_snapshot: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "services",
            MappingProxyType(dict(self.services)),
        )


def build_resident_r1pro_grasp_service_bundle(
    *,
    starter: Any,
    shared_control: Any,
    one_step: Callable[[str], Any],
    frozen_public_snapshot: Mapping[str, Any],
    camera_intrinsics: Mapping[str, Any],
    private_robot_pose: RobotPose,
    public_evidence_provider: Callable[
        [str], PublicGripperWristEvidence
    ],
    private_contact_provider: Callable[[str], bool],
    private_assisted_provider: Callable[[str], bool],
    private_held_provider: Callable[[str], Any],
    private_lift_provider: Callable[[str], float],
    private_audit_writer: Callable[[Mapping[str, Any]], None],
    safe_carry_pose_provider: Any = None,
    base_services: Mapping[str, Any] | None = None,
    binding_config_overrides: Mapping[str, Any] | None = None,
    pose_config: StarterGraspPoseConfig | None = None,
) -> R1ProGraspServiceBundle:
    """Build live services only after a complete non-mutating preflight."""

    existing = dict(base_services or {})
    conflicts = _RESERVED_SERVICE_KEYS.intersection(existing)
    if conflicts:
        raise R1ProGraspServiceFactoryError(
            f"grasp service keys already exist: {sorted(conflicts)}"
        )
    if not isinstance(private_robot_pose, RobotPose):
        raise R1ProGraspServiceFactoryError(
            "private_robot_pose must be a validated RobotPose"
        )
    _require_callable(one_step, "one_step")
    _require_callable(
        safe_carry_pose_provider,
        "safe_carry_pose_provider",
    )
    _require_callable(
        public_evidence_provider,
        "public_evidence_provider",
    )
    for callback, name in (
        (private_contact_provider, "private_contact_provider"),
        (private_assisted_provider, "private_assisted_provider"),
        (private_held_provider, "private_held_provider"),
        (private_lift_provider, "private_lift_provider"),
        (private_audit_writer, "private_audit_writer"),
    ):
        _require_callable(callback, name)
    if not callable(getattr(shared_control, "command_direct_action", None)):
        raise R1ProGraspServiceFactoryError(
            "SharedCanaryControl lacks command_direct_action"
        )
    if not callable(getattr(shared_control, "hold", None)):
        raise R1ProGraspServiceFactoryError(
            "SharedCanaryControl lacks hold"
        )

    frozen_snapshot = _freeze_public_snapshot(frozen_public_snapshot)
    overrides = dict(binding_config_overrides or {})
    if "intrinsic_by_camera" in overrides:
        raise R1ProGraspServiceFactoryError(
            "intrinsic_by_camera must come from camera_intrinsics"
        )
    binding_config = R1ProGraspBindingConfig(
        intrinsic_by_camera=camera_intrinsics,
        **overrides,
    )
    _validate_snapshot_cameras(frozen_snapshot, binding_config)

    def command_step(action: np.ndarray, label: str) -> int:
        shared_control.command_direct_action(action)
        one_step(label)
        return 1

    def safe_hold(hand: str) -> int:
        del hand
        shared_control.hold()
        one_step("grasp.safe_hold")
        return 1

    def private_evidence(
        hand: str,
        phase: GraspMotionPhase,
    ) -> PrivateGraspEvidence:
        return PrivateGraspEvidence(
            qualified_contact=private_contact_provider(hand),
            assisted_grasp_active=private_assisted_provider(hand),
            held_token=private_held_provider(hand),
            payload_lift_m=(
                private_lift_provider(hand)
                if phase is GraspMotionPhase.LIFT
                else 0.0
            ),
        )

    try:
        driver = R1ProStarterGraspDriver(
            starter=starter,
            command_step=command_step,
            public_evidence_provider=public_evidence_provider,
            private_audit_provider=private_evidence,
            safe_hold=safe_hold,
            safe_carry_pose_provider=safe_carry_pose_provider,
            config=pose_config,
        )
    except StarterGraspDriverError as exc:
        raise R1ProGraspServiceFactoryError(str(exc)) from exc
    audit_sink = PrivateGraspArtifactSink(private_audit_writer)
    capability = preflight_r1pro_grasp_binding(
        snapshot_provider=lambda: frozen_snapshot,
        robot_pose_provider=lambda: private_robot_pose,
        motion_driver=driver,
        private_audit_sink=audit_sink,
    )
    try:
        capability.require_available()
        grasp_services = build_r1pro_grasp_services(
            config=binding_config,
            snapshot_provider=lambda: frozen_snapshot,
            robot_pose_provider=lambda: private_robot_pose,
            motion_driver=driver,
            private_audit_sink=audit_sink,
        )
    except R1ProGraspBindingError as exc:
        raise R1ProGraspServiceFactoryError(str(exc)) from exc
    combined = {**existing, **grasp_services}
    return R1ProGraspServiceBundle(
        services=combined,
        capability_report=capability,
        frozen_snapshot=frozen_snapshot,
    )


def _freeze_public_snapshot(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R1ProGraspServiceFactoryError(
            "frozen_public_snapshot must be a mapping"
        )
    for key in value:
        folded = str(key).casefold()
        if any(fragment in folded for fragment in _FORBIDDEN_PUBLIC_FRAGMENTS):
            raise R1ProGraspServiceFactoryError(
                f"public snapshot contains private field {key!r}"
            )
    if "cam_rel_poses" not in value or "cameras" not in value:
        raise R1ProGraspServiceFactoryError(
            "public snapshot lacks cam_rel_poses or cameras"
        )
    frozen = _freeze_value(value, "snapshot")
    cam_rel = np.asarray(frozen["cam_rel_poses"])
    if cam_rel.shape != (21,) or not np.isfinite(cam_rel).all():
        raise R1ProGraspServiceFactoryError(
            "cam_rel_poses must be finite and 21D"
        )
    if not isinstance(frozen["cameras"], Mapping):
        raise R1ProGraspServiceFactoryError("cameras must be a mapping")
    return frozen


def _freeze_value(value: Any, path: str) -> Any:
    if isinstance(value, Mapping):
        output = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise R1ProGraspServiceFactoryError(
                    f"{path} keys must be non-empty strings"
                )
            folded = key.casefold()
            if any(
                fragment in folded
                for fragment in _FORBIDDEN_PUBLIC_FRAGMENTS
            ):
                raise R1ProGraspServiceFactoryError(
                    f"{path}.{key} is private"
                )
            output[key] = _freeze_value(item, f"{path}.{key}")
        return MappingProxyType(output)
    if isinstance(value, np.ndarray):
        array = np.array(value, copy=True)
        if not np.issubdtype(array.dtype, np.number):
            raise R1ProGraspServiceFactoryError(
                f"{path} must be a numeric array"
            )
        array.setflags(write=False)
        return array
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise R1ProGraspServiceFactoryError(
        f"{path} contains unsupported {type(value).__name__}"
    )


def _require_callable(value: Any, name: str) -> None:
    if not callable(value):
        raise R1ProGraspServiceFactoryError(f"{name} must be callable")


def _validate_snapshot_cameras(
    snapshot: Mapping[str, Any],
    config: R1ProGraspBindingConfig,
) -> None:
    cameras = snapshot["cameras"]
    for camera in config.intrinsic_by_camera:
        if camera not in CAMERA_RESOLUTIONS:
            raise R1ProGraspServiceFactoryError(
                f"unsupported onboard camera role {camera!r}"
            )
        if camera not in cameras:
            raise R1ProGraspServiceFactoryError(
                f"frozen snapshot lacks camera {camera!r}"
            )
        camera_payload = cameras[camera]
        if (
            not isinstance(camera_payload, Mapping)
            or "depth_linear" not in camera_payload
        ):
            raise R1ProGraspServiceFactoryError(
                f"camera {camera!r} lacks depth_linear"
            )
        depth = np.asarray(camera_payload["depth_linear"]).squeeze()
        if depth.shape != CAMERA_RESOLUTIONS[camera]:
            raise R1ProGraspServiceFactoryError(
                f"camera {camera!r} depth shape is {depth.shape}, "
                f"expected {CAMERA_RESOLUTIONS[camera]}"
            )
