"""Concrete official-runtime dependencies for the R1Pro B1 pilot.

This module is import-safe without OmniGibson. Official imports occur only
inside callbacks after the live Evaluator exists. Policy-visible inputs come
only from the sanitized RGB-D/proprio snapshot provider; held-object,
collision, contact, and pose evidence stay in private executor callbacks.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from behavior_hl.embodied.evaluator_bridge.r1pro_b1_live_entrypoint import (
    _OfficialPrivateGraspProbe,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_b1_post_reset_plan import (
    R1ProB1PostResetLiveDependencies,
    SingleTennisBallPostResetConfig,
    build_single_tennis_ball_post_reset_plan,
)
from behavior_hl.embodied.evaluator_bridge.predeclared_camera_config import (
    FULL_CAMERA_PROFILE,
    HEAD_RGB_MINIMAL_CAMERA_PROFILE,
    THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
    THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_curobo_collision_provider import (
    R1ProCuRoboCollisionProviderFactory,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_grasp_binding import (
    RobotPose,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_joint_stall_detector import (
    R1ProCommandAwarePrivateSafety,
)
from behavior_hl.embodied.policy.hold_policy import (
    LEFT_ARM_QPOS,
    RIGHT_ARM_QPOS,
    TRUNK_QPOS,
)
from behavior_hl.embodied.policy.observations import (
    CAMERA_RESOLUTIONS,
    PROPRIO_DIM,
    TASK_ID,
    validate_head_rgb_minimal_observation,
    validate_public_observation,
    validate_three_rgb_head_eval_observation,
)
from behavior_hl.embodied.policy.r1pro.action_layout import (
    R1PRO_ACTION_LAYOUT,
)


R1PRO_B1_LIVE_DEPENDENCIES_VERSION = (
    "behavior2026.r1pro-b1-live-dependencies/v1"
)
_POST_RESET_SAFETY_FIELDS = frozenset(
    {"base_contact", "predicted_self_collision", "joint_stalled"}
)
_CALIBRATION_SAFETY_FIELDS = frozenset(
    {"unsafe_contact", "predicted_self_collision", "joint_stalled"}
)


class R1ProB1LiveDependenciesError(RuntimeError):
    """Fail-closed official runtime composition or sampling failure."""


def _profile_validator(
    observation_profile: str,
    *,
    expected_task_id: int,
) -> Callable[[Mapping[str, Any]], None]:
    """Bind one existing strict public validator to a frozen profile."""

    if (
        isinstance(expected_task_id, bool)
        or not isinstance(expected_task_id, int)
        or expected_task_id < 0
    ):
        raise ValueError("expected_task_id must be a non-negative integer")
    validators = {
        FULL_CAMERA_PROFILE: validate_public_observation,
        THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE: lambda snapshot, **kwargs: (
            validate_public_observation(snapshot, head_eval=True, **kwargs)
        ),
        HEAD_RGB_MINIMAL_CAMERA_PROFILE: (
            validate_head_rgb_minimal_observation
        ),
        THREE_RGB_HEAD_EVAL_CAMERA_PROFILE: (
            validate_three_rgb_head_eval_observation
        ),
    }
    try:
        validator = validators[observation_profile]
    except (KeyError, TypeError) as exc:
        raise ValueError("unknown motion-safety observation profile") from exc

    def validate(snapshot: Mapping[str, Any]) -> None:
        validator(snapshot, expected_task_id=expected_task_id)

    return validate


@dataclass(frozen=True, slots=True)
class R1ProMotionSafetyDependencies:
    """Measured motion dependencies shared by resident R1Pro runtimes."""

    public_stability: Callable[[], Mapping[str, float]]
    private_safety: R1ProCommandAwarePrivateSafety
    command_frequency_hz: float


def build_r1pro_motion_safety_dependencies(
    *,
    evaluator: Any,
    starter: Any,
    live_public_snapshot: Callable[[], Mapping[str, Any]],
    observation_profile: str = FULL_CAMERA_PROFILE,
    expected_task_id: int = TASK_ID,
    command_frequency_hz: float = 30.0,
    collision_provider_factory: Callable[[Any, Any], Callable[[], bool]]
    | None = None,
) -> R1ProMotionSafetyDependencies:
    """Bind real pose, contact, collision, and joint-stall measurements."""

    robot = getattr(evaluator, "robot", None)
    if (
        robot is None
        or getattr(starter, "robot", None) is not robot
        or getattr(starter, "arm", None) not in {"left", "right"}
    ):
        raise R1ProB1LiveDependenciesError(
            "motion safety requires one live evaluator/starter robot"
        )
    if not callable(live_public_snapshot):
        raise TypeError("live_public_snapshot must be callable")
    _profile_validator(
        observation_profile,
        expected_task_id=expected_task_id,
    )
    if (
        isinstance(command_frequency_hz, bool)
        or not isinstance(command_frequency_hz, (int, float))
        or not math.isfinite(float(command_frequency_hz))
        or float(command_frequency_hz) <= 0.0
    ):
        raise ValueError("command_frequency_hz must be finite and positive")

    baseline_position, baseline_quaternion = _robot_pose(robot)
    safety = R1ProOfficialPrivateSafetyFactory(
        expected_fields=_POST_RESET_SAFETY_FIELDS,
        collision_provider_factory=collision_provider_factory,
    )(
        evaluator,
        starter,
        live_public_snapshot=live_public_snapshot,
        observation_profile=observation_profile,
        expected_task_id=expected_task_id,
    )

    def public_stability() -> Mapping[str, float]:
        position, quaternion = _robot_pose(robot)
        relative = _quat_multiply(
            _quat_inverse(baseline_quaternion),
            quaternion,
        )
        roll, pitch = _roll_pitch_deg(relative)
        return {
            "base_z_drift_m": position[2] - baseline_position[2],
            "roll_deg": roll,
            "pitch_deg": pitch,
        }

    return R1ProMotionSafetyDependencies(
        public_stability=public_stability,
        private_safety=safety,
        command_frequency_hz=float(command_frequency_hz),
    )


class R1ProOfficialPrivateSafetyFactory:
    """Build anonymous contact/collision/stall safety for one live robot."""

    api_contract_version = R1PRO_B1_LIVE_DEPENDENCIES_VERSION

    def __init__(
        self,
        *,
        expected_fields: frozenset[str],
        collision_provider_factory: Callable[[Any, Any], Callable[[], bool]]
        | None = None,
    ) -> None:
        if expected_fields not in {
            _POST_RESET_SAFETY_FIELDS,
            _CALIBRATION_SAFETY_FIELDS,
        }:
            raise ValueError("private safety field set is unsupported")
        factory = (
            collision_provider_factory
            or R1ProCuRoboCollisionProviderFactory()
        )
        if not callable(factory):
            raise TypeError("collision_provider_factory must be callable")
        self._expected_fields = expected_fields
        self._collision_factory = factory

    def __call__(
        self,
        evaluator: Any,
        starter: Any,
        *,
        live_public_snapshot: Callable[[], Mapping[str, Any]],
        observation_profile: str = FULL_CAMERA_PROFILE,
        expected_task_id: int = TASK_ID,
    ) -> R1ProCommandAwarePrivateSafety:
        collision = self._collision_factory(evaluator, starter)
        if not callable(collision):
            raise R1ProB1LiveDependenciesError(
                "collision provider factory returned a non-callable"
            )

        validate_snapshot = _profile_validator(
            observation_profile,
            expected_task_id=expected_task_id,
        )

        def proprio() -> tuple[float, ...]:
            snapshot = live_public_snapshot()
            validate_snapshot(snapshot)
            value = _vector(
                snapshot["proprio"],
                PROPRIO_DIM,
                "public proprio",
            )
            return value

        def upstream() -> Mapping[str, bool]:
            unsafe_contact = (
                _official_base_contact(evaluator)
                or _official_trunk_arm_contact(evaluator)
            )
            values = {
                "predicted_self_collision": _strict_bool(
                    collision(),
                    "predicted collision",
                ),
                # The command-aware wrapper replaces this baseline with the
                # measured detector verdict after every registered command.
                "joint_stalled": False,
            }
            if self._expected_fields == _POST_RESET_SAFETY_FIELDS:
                values["base_contact"] = unsafe_contact
            else:
                values["unsafe_contact"] = unsafe_contact
            return values

        result = R1ProCommandAwarePrivateSafety(
            upstream_private_safety=upstream,
            public_proprio=proprio,
            expected_fields=self._expected_fields,
        )
        result.collision_provider = collision
        return result


class R1ProCalibrationPrivateSafetyProviderFactory:
    """Concrete factory consumed by safe-carry live sampling."""

    api_contract_version = R1PRO_B1_LIVE_DEPENDENCIES_VERSION

    def __init__(
        self,
        *,
        live_public_snapshot_factory: Callable[
            [Any], Callable[[], Mapping[str, Any]]
        ],
        collision_provider_factory: Callable[[Any, Any], Callable[[], bool]]
        | None = None,
    ) -> None:
        if not callable(live_public_snapshot_factory):
            raise TypeError(
                "live_public_snapshot_factory must be callable"
            )
        self._snapshot_factory = live_public_snapshot_factory
        self._safety_factory = R1ProOfficialPrivateSafetyFactory(
            expected_fields=_CALIBRATION_SAFETY_FIELDS,
            collision_provider_factory=collision_provider_factory,
        )

    def __call__(self, evaluator: Any, starter: Any) -> Any:
        snapshot = self._snapshot_factory(evaluator)
        if not callable(snapshot):
            raise R1ProB1LiveDependenciesError(
                "live snapshot factory returned a non-callable"
            )
        return self._safety_factory(
            evaluator,
            starter,
            live_public_snapshot=snapshot,
        )


class R1ProB1PostResetLiveDependencyFactory:
    """Create all post-reset dependencies from one evaluator and Starter."""

    api_contract_version = R1PRO_B1_LIVE_DEPENDENCIES_VERSION

    def __init__(
        self,
        *,
        command_frequency_hz: float = 30.0,
        collision_provider_factory: Callable[[Any, Any], Callable[[], bool]]
        | None = None,
    ) -> None:
        if (
            isinstance(command_frequency_hz, bool)
            or not isinstance(command_frequency_hz, (int, float))
            or not math.isfinite(float(command_frequency_hz))
            or float(command_frequency_hz) <= 0.0
        ):
            raise ValueError("command_frequency_hz must be finite and positive")
        self._frequency = float(command_frequency_hz)
        self._safety_factory = R1ProOfficialPrivateSafetyFactory(
            expected_fields=_POST_RESET_SAFETY_FIELDS,
            collision_provider_factory=collision_provider_factory,
        )

    def __call__(
        self,
        *,
        evaluator: Any,
        starter: Any,
        live_public_snapshot: Callable[[], Mapping[str, Any]],
    ) -> R1ProB1PostResetLiveDependencies:
        robot = getattr(evaluator, "robot", None)
        if (
            robot is None
            or getattr(starter, "robot", None) is not robot
            or getattr(starter, "arm", None) not in {"left", "right"}
        ):
            raise R1ProB1LiveDependenciesError(
                "dependencies require one live evaluator/starter robot"
            )
        if not callable(live_public_snapshot):
            raise TypeError("live_public_snapshot must be callable")
        intrinsics = _camera_intrinsics(evaluator)
        probe = _OfficialPrivateGraspProbe(starter)
        motion = build_r1pro_motion_safety_dependencies(
            evaluator=evaluator,
            starter=starter,
            live_public_snapshot=live_public_snapshot,
            command_frequency_hz=self._frequency,
            collision_provider_factory=self._safety_factory._collision_factory,
        )
        safety = motion.private_safety

        def private_robot_pose(value: Any) -> RobotPose:
            if value is not evaluator:
                raise R1ProB1LiveDependenciesError(
                    "private robot pose received a foreign evaluator"
                )
            position, quaternion = _robot_pose(robot)
            return RobotPose(position, quaternion)

        def target_pose_builder(
            base_xyz: tuple[float, float, float],
            hand: str,
        ) -> tuple[Any, Any]:
            if hand != starter.arm:
                raise R1ProB1LiveDependenciesError(
                    "target pose hand differs from selected Starter"
                )
            local = _vector(base_xyz, 3, "robot-base target")
            base_position, base_orientation = _raw_robot_pose(robot)
            from omnigibson.utils.transform_utils import pose_transform
            import torch as th

            device = getattr(base_position, "device", None)
            dtype = getattr(base_position, "dtype", th.float32)
            local_position = th.as_tensor(
                local,
                dtype=dtype,
                device=device,
            )
            local_orientation = th.as_tensor(
                (0.0, 0.0, 0.0, 1.0),
                dtype=dtype,
                device=device,
            )
            world_position, _ = pose_transform(
                base_position,
                base_orientation,
                local_position,
                local_orientation,
            )
            _, current_eef_orientation = robot.get_eef_pose(arm=hand)
            _normalized_quaternion(
                current_eef_orientation,
                "current EEF quaternion",
            )
            return world_position, current_eef_orientation

        def safe_hold_action(hand: str, released: bool) -> Sequence[float]:
            if hand != starter.arm or not isinstance(released, bool):
                raise R1ProB1LiveDependenciesError(
                    "safe hold hand/released state is invalid"
                )
            snapshot = live_public_snapshot()
            validate_public_observation(snapshot)
            proprio = np.asarray(snapshot["proprio"], dtype=np.float64)
            if proprio.shape != (PROPRIO_DIM,) or not np.isfinite(
                proprio
            ).all():
                raise R1ProB1LiveDependenciesError(
                    "safe hold public proprio is invalid"
                )
            action = np.zeros(
                R1PRO_ACTION_LAYOUT.dimension,
                dtype=np.float32,
            )
            action[R1PRO_ACTION_LAYOUT.slice_for("torso")] = proprio[
                TRUNK_QPOS
            ]
            action[R1PRO_ACTION_LAYOUT.slice_for("left_arm")] = proprio[
                LEFT_ARM_QPOS
            ]
            action[R1PRO_ACTION_LAYOUT.slice_for("right_arm")] = proprio[
                RIGHT_ARM_QPOS
            ]
            action[R1PRO_ACTION_LAYOUT.slice_for("left_gripper")] = (
                1.0 if released or hand != "left" else -1.0
            )
            action[R1PRO_ACTION_LAYOUT.slice_for("right_gripper")] = (
                1.0 if released or hand != "right" else -1.0
            )
            return action

        def capture_private_carry_endpoint(
            *,
            stage: str,
            episode_id: str,
            action_index: int,
        ) -> Mapping[str, Any]:
            if (
                stage not in {"start", "end"}
                or not isinstance(episode_id, str)
                or not episode_id
                or isinstance(action_index, bool)
                or not isinstance(action_index, int)
                or action_index < 0
            ):
                raise R1ProB1LiveDependenciesError(
                    "private carry endpoint identity is invalid"
                )
            held = probe.held(starter.arm)
            linear, angular = _payload_speeds(held)
            position, quaternion = _robot_pose(robot)
            eef_position, eef_quaternion = _eef_pose(robot, starter.arm)
            diagnostics = safety.detector.last_diagnostics
            collision_audit = getattr(
                safety.collision_provider,
                "last_audit",
                None,
            )
            return {
                "stage": stage,
                "episode_id": episode_id,
                "action_index": action_index,
                "hand": starter.arm,
                "official_held": held is not None,
                "assisted": probe.assisted(starter.arm),
                "qualified_contact": probe.qualified_contact(starter.arm),
                "lift_m": probe.lift(starter.arm),
                "payload_linear_speed_mps": linear,
                "payload_angular_speed_radps": angular,
                "robot_position": position,
                "robot_quaternion_xyzw": quaternion,
                "eef_position": eef_position,
                "eef_quaternion_xyzw": eef_quaternion,
                "base_contact": _official_base_contact(evaluator),
                "trunk_arm_contact": _official_trunk_arm_contact(evaluator),
                "joint_stall_diagnostics": (
                    None if diagnostics is None else asdict(diagnostics)
                ),
                "collision_audit": (
                    None
                    if collision_audit is None
                    else asdict(collision_audit)
                ),
            }

        return R1ProB1PostResetLiveDependencies(
            starter=starter,
            camera_intrinsics=intrinsics,
            private_robot_pose_provider=private_robot_pose,
            private_contact_provider=probe.qualified_contact,
            private_assisted_provider=probe.assisted,
            private_held_provider=probe.held,
            private_lift_provider=probe.lift,
            target_pose_builder=target_pose_builder,
            safe_hold_action=safe_hold_action,
            public_stability=motion.public_stability,
            private_safety=safety,
            capture_private_carry_endpoint=(
                capture_private_carry_endpoint
            ),
            command_frequency_hz=self._frequency,
        )


def make_concrete_single_tennis_ball_post_reset_callback(
    *,
    dependency_factory: R1ProB1PostResetLiveDependencyFactory,
    config: SingleTennisBallPostResetConfig,
) -> Callable[..., Any]:
    """Create dependencies after reset, then build the normal B1 plan."""

    if not isinstance(
        dependency_factory,
        R1ProB1PostResetLiveDependencyFactory,
    ):
        raise TypeError(
            "dependency_factory must be "
            "R1ProB1PostResetLiveDependencyFactory"
        )
    if not isinstance(config, SingleTennisBallPostResetConfig):
        raise TypeError("config must be SingleTennisBallPostResetConfig")

    def callback(**kwargs: Any) -> Any:
        expected = {
            "evaluator",
            "control",
            "formal_wrist",
            "safe_carry",
            "witness",
            "run_dir",
            "starter",
            "live_public_snapshot",
        }
        if set(kwargs) != expected:
            raise R1ProB1LiveDependenciesError(
                "concrete post-reset callback fields changed"
            )
        dependencies = dependency_factory(
            evaluator=kwargs["evaluator"],
            starter=kwargs["starter"],
            live_public_snapshot=kwargs["live_public_snapshot"],
        )
        return build_single_tennis_ball_post_reset_plan(
            evaluator=kwargs["evaluator"],
            control=kwargs["control"],
            formal_wrist=kwargs["formal_wrist"],
            safe_carry=kwargs["safe_carry"],
            witness=kwargs["witness"],
            run_dir=kwargs["run_dir"],
            dependencies=dependencies,
            config=config,
        )

    return callback


def _camera_intrinsics(evaluator: Any) -> dict[str, np.ndarray]:
    names = getattr(evaluator, "robot_camera_names", None)
    robot = getattr(evaluator, "robot", None)
    sensors = getattr(robot, "sensors", None)
    if not isinstance(names, Mapping) or not isinstance(sensors, Mapping):
        raise R1ProB1LiveDependenciesError(
            "official camera name/sensor mappings are unavailable"
        )
    result: dict[str, np.ndarray] = {}
    for role in CAMERA_RESOLUTIONS:
        name = names.get(role)
        if not isinstance(name, str) or "::" not in name:
            raise R1ProB1LiveDependenciesError(
                f"official camera mapping is invalid for {role}"
            )
        sensor_name = name.split("::", 1)[1]
        sensor = sensors.get(sensor_name)
        intrinsic = np.asarray(
            getattr(sensor, "intrinsic_matrix", None),
            dtype=np.float64,
        )
        if (
            intrinsic.shape != (3, 3)
            or not np.isfinite(intrinsic).all()
            or intrinsic[0, 0] <= 0.0
            or intrinsic[1, 1] <= 0.0
        ):
            raise R1ProB1LiveDependenciesError(
                f"official camera intrinsic is invalid for {role}"
            )
        result[role] = np.array(intrinsic, copy=True)
    return result


def _official_base_contact(evaluator: Any) -> bool:
    from omnigibson.utils.usd_utils import RigidContactAPI
    from behavior_hl.embodied.evaluator_bridge.run_g2_g3a_suite import (
        _base_contact,
    )

    return bool(_base_contact(evaluator, RigidContactAPI))


def _official_trunk_arm_contact(evaluator: Any) -> bool:
    from omnigibson.utils.usd_utils import RigidContactAPI
    from behavior_hl.embodied.evaluator_bridge.run_g2_g3a_suite import (
        _trunk_arm_contact,
    )

    return bool(_trunk_arm_contact(evaluator, RigidContactAPI))


def _raw_robot_pose(robot: Any) -> tuple[Any, Any]:
    getter = getattr(robot, "get_position_orientation", None)
    if not callable(getter):
        raise R1ProB1LiveDependenciesError("robot pose API is unavailable")
    try:
        return getter(frame="world", clone=True)
    except TypeError:
        return getter()


def _robot_pose(robot: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    position, quaternion = _raw_robot_pose(robot)
    return (
        _vector(position, 3, "robot position"),
        _normalized_quaternion(quaternion, "robot quaternion"),
    )


def _eef_pose(
    robot: Any,
    hand: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    getter = getattr(robot, "get_eef_pose", None)
    if not callable(getter):
        raise R1ProB1LiveDependenciesError("robot EEF pose API unavailable")
    position, quaternion = getter(arm=hand)
    return (
        _vector(position, 3, "EEF position"),
        _normalized_quaternion(quaternion, "EEF quaternion"),
    )


def _payload_speeds(held: Any) -> tuple[float, float]:
    if held is None:
        return 0.0, 0.0
    linear = getattr(held, "get_linear_velocity", None)
    angular = getattr(held, "get_angular_velocity", None)
    if not callable(linear) or not callable(angular):
        raise R1ProB1LiveDependenciesError(
            "held object velocity APIs are unavailable"
        )
    return (
        float(np.linalg.norm(_vector(linear(), 3, "payload linear velocity"))),
        float(
            np.linalg.norm(_vector(angular(), 3, "payload angular velocity"))
        ),
    )


def _roll_pitch_deg(value: Sequence[float]) -> tuple[float, float]:
    x, y, z, w = _normalized_quaternion(value, "relative quaternion")
    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch = math.asin(
        max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    )
    return math.degrees(roll), math.degrees(pitch)


def _quat_inverse(value: Sequence[float]) -> tuple[float, ...]:
    x, y, z, w = _normalized_quaternion(value, "quaternion")
    return -x, -y, -z, w


def _quat_multiply(
    first: Sequence[float],
    second: Sequence[float],
) -> tuple[float, ...]:
    x1, y1, z1, w1 = _normalized_quaternion(
        first,
        "first quaternion",
    )
    x2, y2, z2, w2 = _normalized_quaternion(
        second,
        "second quaternion",
    )
    return _normalized_quaternion(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        "quaternion product",
    )


def _normalized_quaternion(value: Any, name: str) -> tuple[float, ...]:
    result = np.asarray(_array(value), dtype=np.float64)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise R1ProB1LiveDependenciesError(f"{name} is invalid")
    norm = float(np.linalg.norm(result))
    if norm <= 1e-8:
        raise R1ProB1LiveDependenciesError(f"{name} has zero norm")
    return tuple(float(item) for item in result / norm)


def _vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    result = np.asarray(_array(value), dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise R1ProB1LiveDependenciesError(
            f"{name} must be a finite {size}-vector"
        )
    return tuple(float(item) for item in result)


def _array(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise R1ProB1LiveDependenciesError(f"{name} must be boolean")
    return bool(value)
