"""Concrete same-scene R1Pro safe-carry live sample provider.

Policy-visible grasp confidence is injected through a separately bound
public-only callback.  Official held identity, payload motion, robot pose,
EEF pose, and anonymous safety flags remain evaluator-private and are used
only to construct :class:`SafeCarryLiveSample`.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from behavior_hl.embodied.evaluator_bridge.safe_carry_calibration import (
    SafeCarryLiveSample,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_joint_stall_detector import (
    notify_r1pro_command,
)
from behavior_hl.embodied.policy.observations import (
    PROPRIO_DIM,
    validate_public_observation,
)


R1PRO_SAFE_CARRY_LIVE_SAMPLE_VERSION = (
    "behavior2026.r1pro-safe-carry-live-sample/v1"
)
_SAFETY_FIELDS = frozenset(
    {"unsafe_contact", "predicted_self_collision", "joint_stalled"}
)


class R1ProSafeCarryLiveSampleError(RuntimeError):
    """Fail-closed public/private live sampling error."""


class R1ProSafeCarryLiveSampleProviderFactory:
    """Concrete factory matching the live entrypoint callback contract."""

    api_contract_version = R1PRO_SAFE_CARRY_LIVE_SAMPLE_VERSION

    def __init__(
        self,
        *,
        private_safety_provider_factory: Callable[
            [Any, Any], Callable[[], Mapping[str, Any]]
        ],
    ) -> None:
        if not callable(private_safety_provider_factory):
            raise TypeError(
                "private_safety_provider_factory must be callable"
            )
        self._safety_factory = private_safety_provider_factory

    def __call__(
        self,
        evaluator: Any,
        starter: Any,
        *,
        live_public_snapshot: Callable[[], Mapping[str, Any]],
        private_lift_provider: Callable[[str], float],
    ) -> "R1ProSafeCarryLiveSampleProvider":
        safety = self._safety_factory(evaluator, starter)
        return R1ProSafeCarryLiveSampleProvider(
            evaluator=evaluator,
            starter=starter,
            live_public_snapshot=live_public_snapshot,
            private_lift_provider=private_lift_provider,
            private_safety_provider=safety,
        )


class R1ProSafeCarryLiveSampleProvider:
    """One fresh typed sample after each confirmed evaluator step."""

    api_contract_version = R1PRO_SAFE_CARRY_LIVE_SAMPLE_VERSION

    def __init__(
        self,
        *,
        evaluator: Any,
        starter: Any,
        live_public_snapshot: Callable[[], Mapping[str, Any]],
        private_lift_provider: Callable[[str], float],
        private_safety_provider: Callable[[], Mapping[str, Any]],
    ) -> None:
        robot = getattr(evaluator, "robot", None)
        if (
            robot is None
            or getattr(starter, "robot", None) is not robot
            or getattr(starter, "arm", None) not in {"left", "right"}
        ):
            raise R1ProSafeCarryLiveSampleError(
                "sample provider requires one live evaluator/starter robot"
            )
        for name, callback in (
            ("live_public_snapshot", live_public_snapshot),
            ("private_lift_provider", private_lift_provider),
            ("private_safety_provider", private_safety_provider),
        ):
            if not callable(callback):
                raise TypeError(f"{name} must be callable")
        if not callable(getattr(robot, "get_eef_pose", None)):
            raise R1ProSafeCarryLiveSampleError(
                "official robot EEF pose API is unavailable"
            )
        if not callable(getattr(starter, "_get_obj_in_hand", None)):
            raise R1ProSafeCarryLiveSampleError(
                "official Starter held-object API is unavailable"
            )
        self._evaluator = evaluator
        self._starter = starter
        self._robot = robot
        self._public_snapshot = live_public_snapshot
        self._private_lift = private_lift_provider
        self._private_safety = private_safety_provider
        self._hand = starter.arm
        self._public_grasp_proxy: Callable[[], bool] | None = None
        self._poisoned = False
        self._sample_count = 0
        self._initial_base_position, self._initial_base_quaternion = (
            _robot_pose(robot)
        )

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def register_command(self, action: Sequence[float]) -> None:
        """Forward the concrete 23D command to command-aware safety."""

        notify_r1pro_command(self._private_safety, action)

    def bind_public_grasp_proxy(
        self,
        callback: Callable[[], bool],
    ) -> None:
        """Bind the public-only wrist proxy once, after its positive window."""

        if self._public_grasp_proxy is not None:
            raise R1ProSafeCarryLiveSampleError(
                "public grasp proxy was already bound"
            )
        if not callable(callback):
            raise TypeError("public grasp proxy must be callable")
        self._public_grasp_proxy = callback

    def __call__(
        self,
        *,
        hand: str,
        validation_stage: str | None,
        phase: str,
        requested_arm_target: Sequence[float],
        commanded_arm_target: Sequence[float],
    ) -> SafeCarryLiveSample:
        if self._poisoned:
            raise R1ProSafeCarryLiveSampleError(
                "live sample provider is poisoned"
            )
        try:
            return self._sample(
                hand=hand,
                validation_stage=validation_stage,
                phase=phase,
                requested_arm_target=requested_arm_target,
                commanded_arm_target=commanded_arm_target,
            )
        except Exception:
            self._poisoned = True
            raise

    def _sample(
        self,
        *,
        hand: str,
        validation_stage: str | None,
        phase: str,
        requested_arm_target: Sequence[float],
        commanded_arm_target: Sequence[float],
    ) -> SafeCarryLiveSample:
        if (
            hand != self._hand
            or phase not in {"retract", "g3b"}
            or (phase == "retract") != (validation_stage is None)
        ):
            raise R1ProSafeCarryLiveSampleError(
                "safe-carry sample phase/hand changed"
            )
        if self._public_grasp_proxy is None:
            raise R1ProSafeCarryLiveSampleError(
                "public grasp proxy is not bound"
            )
        requested = _vector(requested_arm_target, 7, "requested arm target")
        commanded = _vector(commanded_arm_target, 7, "commanded arm target")

        public = self._public_snapshot()
        validate_public_observation(public)
        proprio = _vector(public["proprio"], PROPRIO_DIM, "public proprio")
        public_arm = (
            proprio[3:10] if hand == "left" else proprio[28:35]
        )
        public_torso = proprio[53:57]
        proxy = self._public_grasp_proxy()
        if not isinstance(proxy, bool):
            raise R1ProSafeCarryLiveSampleError(
                "public grasp proxy must return bool"
            )

        held_token = self._starter._get_obj_in_hand()
        assisted = getattr(self._robot, "_ag_obj_in_hand", None)
        if not isinstance(assisted, Mapping) or hand not in assisted:
            raise R1ProSafeCarryLiveSampleError(
                "official assisted held state is unavailable"
            )
        official_held = (
            held_token is not None and assisted[hand] is held_token
        )
        payload_linear, payload_angular = _payload_speeds(held_token)
        safety = self._private_safety()
        if not isinstance(safety, Mapping) or set(safety) != _SAFETY_FIELDS:
            raise R1ProSafeCarryLiveSampleError(
                "private safety audit fields changed"
            )
        if any(not isinstance(safety[name], bool) for name in _SAFETY_FIELDS):
            raise R1ProSafeCarryLiveSampleError(
                "private safety audit values must be bool"
            )

        eef_position, eef_quaternion = _eef_pose(self._robot, hand)
        base_position, base_quaternion = _robot_pose(self._robot)
        body_position, body_quaternion = _relative_pose(
            base_position,
            base_quaternion,
            eef_position,
            eef_quaternion,
        )
        relative_base_quaternion = _quat_multiply(
            _quat_inverse(self._initial_base_quaternion),
            base_quaternion,
        )
        roll, pitch = _roll_pitch_deg(relative_base_quaternion)
        lift = float(self._private_lift(hand))
        if not math.isfinite(lift):
            raise R1ProSafeCarryLiveSampleError(
                "private payload lift is not finite"
            )

        self._sample_count += 1
        return SafeCarryLiveSample(
            hand=hand,
            arm_qpos=tuple(public_arm),
            torso_qpos=tuple(public_torso),
            arm_tracking_error_rad=max(
                abs(actual - target)
                for actual, target in zip(public_arm, commanded)
            ),
            eef_world_position=eef_position,
            eef_world_quaternion_xyzw=eef_quaternion,
            eef_body_position=body_position,
            eef_body_quaternion_xyzw=body_quaternion,
            official_held=official_held,
            held_token=held_token,
            grasp_proxy_likely=proxy,
            object_lift_m=lift,
            payload_linear_speed_mps=payload_linear,
            payload_angular_speed_radps=payload_angular,
            base_z_drift_m=(
                base_position[2] - self._initial_base_position[2]
            ),
            roll_deg=roll,
            pitch_deg=pitch,
            unsafe_contact=safety["unsafe_contact"],
            predicted_self_collision=safety[
                "predicted_self_collision"
            ],
            joint_stalled=safety["joint_stalled"],
            validation_stage=validation_stage,
        )


def _payload_speeds(held: Any) -> tuple[float, float]:
    if held is None:
        return 0.0, 0.0
    linear_getter = getattr(held, "get_linear_velocity", None)
    angular_getter = getattr(held, "get_angular_velocity", None)
    if not callable(linear_getter) or not callable(angular_getter):
        raise R1ProSafeCarryLiveSampleError(
            "official held object velocity APIs are unavailable"
        )
    return (
        float(np.linalg.norm(_vector(linear_getter(), 3, "payload linear velocity"))),
        float(np.linalg.norm(_vector(angular_getter(), 3, "payload angular velocity"))),
    )


def _robot_pose(robot: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    getter = getattr(robot, "get_position_orientation", None)
    if not callable(getter):
        raise R1ProSafeCarryLiveSampleError("robot pose API is unavailable")
    try:
        position, quaternion = getter(frame="world", clone=True)
    except TypeError:
        position, quaternion = getter()
    return (
        _vector(position, 3, "robot position"),
        _normalized_quaternion(quaternion, "robot quaternion"),
    )


def _eef_pose(
    robot: Any,
    hand: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    position, quaternion = robot.get_eef_pose(arm=hand)
    return (
        _vector(position, 3, "EEF position"),
        _normalized_quaternion(quaternion, "EEF quaternion"),
    )


def _relative_pose(
    base_position: Sequence[float],
    base_quaternion: Sequence[float],
    world_position: Sequence[float],
    world_quaternion: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    inverse = _quat_inverse(base_quaternion)
    translated = tuple(
        float(world_position[index]) - float(base_position[index])
        for index in range(3)
    )
    body_position = _quat_rotate(inverse, translated)
    body_quaternion = _quat_multiply(inverse, world_quaternion)
    return body_position, body_quaternion


def _roll_pitch_deg(quaternion: Sequence[float]) -> tuple[float, float]:
    x, y, z, w = _normalized_quaternion(quaternion, "relative quaternion")
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
    return (-x, -y, -z, w)


def _quat_multiply(
    first: Sequence[float],
    second: Sequence[float],
) -> tuple[float, ...]:
    x1, y1, z1, w1 = _normalized_quaternion(first, "first quaternion")
    x2, y2, z2, w2 = _normalized_quaternion(second, "second quaternion")
    return _normalized_quaternion(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        "product quaternion",
    )


def _quat_rotate(
    quaternion: Sequence[float],
    vector: Sequence[float],
) -> tuple[float, ...]:
    x, y, z, w = _normalized_quaternion(quaternion, "rotation quaternion")
    q = np.asarray((x, y, z), dtype=np.float64)
    v = np.asarray(_vector(vector, 3, "rotation vector"), dtype=np.float64)
    rotated = v + 2.0 * np.cross(q, np.cross(q, v) + w * v)
    return tuple(float(item) for item in rotated)


def _normalized_quaternion(value: Any, name: str) -> tuple[float, ...]:
    result = np.asarray(_array(value), dtype=np.float64)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise R1ProSafeCarryLiveSampleError(f"{name} is invalid")
    norm = float(np.linalg.norm(result))
    if norm <= 1e-8:
        raise R1ProSafeCarryLiveSampleError(f"{name} has zero norm")
    return tuple(float(item) for item in result / norm)


def _vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    result = np.asarray(_array(value), dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise R1ProSafeCarryLiveSampleError(
            f"{name} must be finite shape ({size},)"
        )
    return tuple(float(item) for item in result)


def _array(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value
