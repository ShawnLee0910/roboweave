"""Versioned SharedCanaryControl bindings for torso look and release.

The bridge consumes public R1Pro proprioception and anonymous safety flags.
It imports no simulator package and never queries held objects, task scope, or
world poses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from behavior_hl.embodied.evaluator_bridge.r1pro_joint_stall_detector import (
    notify_r1pro_command,
)
from behavior_hl.embodied.policy.hold_policy import (
    LEFT_ARM_QPOS,
    RIGHT_ARM_QPOS,
    TRUNK_QPOS,
)
from behavior_hl.embodied.policy.observations import PROPRIO_DIM
from behavior_hl.embodied.policy.r1pro.action_layout import (
    R1PRO_ACTION_LAYOUT,
)
from behavior_hl.embodied.runtime.plugins.release import (
    PHYSICAL_RELEASE_ACTUATOR_VERSION,
    PhysicalReleaseCommand,
    PhysicalReleaseSample,
)
from behavior_hl.embodied.runtime.plugins.torso import (
    PHYSICAL_TORSO_ACTUATOR_VERSION,
    EmptyTorsoCommand,
    EmptyTorsoSample,
)


R1PRO_TORSO_RELEASE_BINDING_VERSION = (
    "r1pro-shared-canary-torso-release-v2"
)

LEFT_GRIPPER_QPOS = slice(24, 26)
LEFT_GRIPPER_QVEL = slice(26, 28)
RIGHT_GRIPPER_QPOS = slice(49, 51)
RIGHT_GRIPPER_QVEL = slice(51, 53)

_PUBLIC_STATE_FIELDS = frozenset(
    {
        "observation_count",
        "base_qvel",
        "trunk_qpos",
        "head_forward_clearance_m",
    }
)
_STABILITY_FIELDS = frozenset(
    {"base_z_drift_m", "roll_deg", "pitch_deg"}
)
_PRIVATE_SAFETY_FIELDS = frozenset(
    {"base_contact", "predicted_self_collision", "joint_stalled"}
)


class R1ProTorsoReleaseBindingError(RuntimeError):
    """Raised before an unsafe or unsupported command advances physics."""


@dataclass(frozen=True, slots=True)
class R1ProTorsoReleasePreflight:
    version: str
    proprio_dimension: int
    command_frequency_hz: float
    torso_actuator_version: str
    release_actuator_version: str
    public_inputs: tuple[str, ...]
    private_safety_inputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class R1ProTorsoReleaseBindings:
    torso: "SharedCanaryPhysicalTorsoActuator"
    release: "SharedCanaryPhysicalReleaseActuator"
    public_grasp_proxy: "CallbackPublicGraspProxySource"
    preflight: R1ProTorsoReleasePreflight


@dataclass(frozen=True, slots=True)
class _Inputs:
    control: Any
    evaluator_step: Callable[[str], Any]
    public_proprio: Callable[[], Sequence[float]]
    public_stability: Callable[[], Mapping[str, Any]]
    private_safety: Callable[[], Mapping[str, Any]]
    command_frequency_hz: float
    max_base_z_drift_m: float
    max_abs_roll_pitch_deg: float


class CallbackPublicGraspProxySource:
    """Narrow public proxy wrapper used by AdaptivePhysicalActionPlugin."""

    def __init__(self, callback: Callable[[], bool]) -> None:
        if not callable(callback):
            raise TypeError("public grasp proxy callback must be callable")
        self._callback = callback

    def grasp_proxy_likely(self) -> bool:
        value = self._callback()
        if not isinstance(value, bool):
            raise R1ProTorsoReleaseBindingError(
                "public grasp proxy callback must return bool"
            )
        return value


def preflight_r1pro_torso_release_binding(
    *,
    control: Any,
    evaluator_step: Callable[[str], Any],
    public_proprio: Callable[[], Sequence[float]],
    public_grasp_proxy: Callable[[], bool],
    public_stability: Callable[[], Mapping[str, Any]],
    private_safety: Callable[[], Mapping[str, Any]],
    command_frequency_hz: float,
    max_base_z_drift_m: float = 0.03,
    max_abs_roll_pitch_deg: float = 5.0,
) -> R1ProTorsoReleasePreflight:
    """Read public inputs only; never command, step, or read private safety."""

    for method in (
        "command_torso",
        "command_direct_action",
        "hold",
        "public_state",
    ):
        if not callable(getattr(control, method, None)):
            raise R1ProTorsoReleaseBindingError(
                f"SharedCanaryControl is missing {method}"
            )
    for name, callback in (
        ("evaluator_step", evaluator_step),
        ("public_proprio", public_proprio),
        ("public_grasp_proxy", public_grasp_proxy),
        ("public_stability", public_stability),
        ("private_safety", private_safety),
    ):
        if not callable(callback):
            raise R1ProTorsoReleaseBindingError(
                f"{name} must be callable"
            )
    frequency = _positive(command_frequency_hz, "command_frequency_hz")
    _positive(max_base_z_drift_m, "max_base_z_drift_m")
    _positive(max_abs_roll_pitch_deg, "max_abs_roll_pitch_deg")
    proprio = _proprio(public_proprio())
    state = _public_state(control)
    # The server caches obs_n while Evaluator exposes the post-step obs_n+1.
    # These are both public states but are deliberately one frame apart.
    # Validate shape / finiteness independently instead of requiring an
    # impossible exact cross-frame match.
    _vector(state["base_qvel"], 3, "base_qvel")
    _vector(state["trunk_qpos"], 4, "trunk_qpos")
    _stability(public_stability())
    proxy = public_grasp_proxy()
    if not isinstance(proxy, bool):
        raise R1ProTorsoReleaseBindingError(
            "public grasp proxy must be boolean"
        )
    # private_safety is deliberately not called during construction.
    return R1ProTorsoReleasePreflight(
        version=R1PRO_TORSO_RELEASE_BINDING_VERSION,
        proprio_dimension=PROPRIO_DIM,
        command_frequency_hz=frequency,
        torso_actuator_version=PHYSICAL_TORSO_ACTUATOR_VERSION,
        release_actuator_version=PHYSICAL_RELEASE_ACTUATOR_VERSION,
        public_inputs=(
            "base_qvel",
            "gripper_qpos",
            "gripper_qvel",
            "public_grasp_proxy",
            "r1pro_proprio",
            "stability_proxy",
            "trunk_qpos",
        ),
        private_safety_inputs=tuple(sorted(_PRIVATE_SAFETY_FIELDS)),
    )


def build_r1pro_torso_release_bindings(
    *,
    control: Any,
    evaluator_step: Callable[[str], Any],
    public_proprio: Callable[[], Sequence[float]],
    public_grasp_proxy: Callable[[], bool],
    public_stability: Callable[[], Mapping[str, Any]],
    private_safety: Callable[[], Mapping[str, Any]],
    command_frequency_hz: float,
    max_base_z_drift_m: float = 0.03,
    max_abs_roll_pitch_deg: float = 5.0,
) -> R1ProTorsoReleaseBindings:
    preflight = preflight_r1pro_torso_release_binding(
        control=control,
        evaluator_step=evaluator_step,
        public_proprio=public_proprio,
        public_grasp_proxy=public_grasp_proxy,
        public_stability=public_stability,
        private_safety=private_safety,
        command_frequency_hz=command_frequency_hz,
        max_base_z_drift_m=max_base_z_drift_m,
        max_abs_roll_pitch_deg=max_abs_roll_pitch_deg,
    )
    inputs = _Inputs(
        control=control,
        evaluator_step=evaluator_step,
        public_proprio=public_proprio,
        public_stability=public_stability,
        private_safety=private_safety,
        command_frequency_hz=preflight.command_frequency_hz,
        max_base_z_drift_m=float(max_base_z_drift_m),
        max_abs_roll_pitch_deg=float(max_abs_roll_pitch_deg),
    )
    return R1ProTorsoReleaseBindings(
        torso=SharedCanaryPhysicalTorsoActuator(inputs),
        release=SharedCanaryPhysicalReleaseActuator(inputs),
        public_grasp_proxy=CallbackPublicGraspProxySource(
            public_grasp_proxy
        ),
        preflight=preflight,
    )


class SharedCanaryPhysicalTorsoActuator:
    """PhysicalTorsoActuator backed by command_torso plus one live step."""

    api_contract_version = PHYSICAL_TORSO_ACTUATOR_VERSION

    def __init__(self, inputs: _Inputs) -> None:
        self._inputs = inputs

    def read(self) -> EmptyTorsoSample:
        return self._sample(stepped=False)

    def step(self, command: EmptyTorsoCommand) -> EmptyTorsoSample:
        if not isinstance(command, EmptyTorsoCommand):
            raise R1ProTorsoReleaseBindingError(
                "torso step requires EmptyTorsoCommand"
            )
        proprio = _proprio(self._inputs.public_proprio())
        tracking_action = _joint_hold_action(proprio)
        tracking_action[R1PRO_ACTION_LAYOUT.slice_for("torso")] = (
            command.torso_target
        )
        notify_r1pro_command(
            self._inputs.private_safety,
            tracking_action,
        )
        self._inputs.control.command_torso(
            command.torso_target,
            max_delta_per_step=0.025,
        )
        try:
            self._inputs.evaluator_step("runtime.empty_torso")
        except Exception:
            self._inputs.control.hold()
            raise
        return self._sample(stepped=True)

    def hold(self) -> None:
        self._inputs.control.hold()

    def _sample(self, *, stepped: bool) -> EmptyTorsoSample:
        proprio = _proprio(self._inputs.public_proprio())
        state = _public_state(self._inputs.control)
        qvel = _vector(state["base_qvel"], 3, "base_qvel")
        stability = _stability(self._inputs.public_stability())
        safety = (
            _private_safety(self._inputs.private_safety())
            if stepped
            else _empty_private_safety()
        )
        stability_unsafe = _stability_unsafe(
            stability,
            self._inputs,
        )
        dt = 1.0 / self._inputs.command_frequency_hz
        return EmptyTorsoSample(
            torso_qpos=tuple(
                float(value) for value in proprio[TRUNK_QPOS]
            ),
            forward_delta_m=qvel[0] * dt if stepped else 0.0,
            left_delta_m=qvel[1] * dt if stepped else 0.0,
            yaw_delta_deg=(
                math.degrees(qvel[2] * dt) if stepped else 0.0
            ),
            joint_stalled=safety["joint_stalled"],
            predicted_self_collision=(
                safety["predicted_self_collision"]
                or safety["base_contact"]
                or stability_unsafe
            ),
        )


class SharedCanaryPhysicalReleaseActuator:
    """PhysicalReleaseActuator using a finite 23D direct action."""

    api_contract_version = PHYSICAL_RELEASE_ACTUATOR_VERSION

    def __init__(self, inputs: _Inputs) -> None:
        self._inputs = inputs
        self._held_joint_targets: dict[
            str,
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ] = {}

    def read(self, *, hand: str) -> PhysicalReleaseSample:
        _require_hand(hand)
        proprio = _proprio(self._inputs.public_proprio())
        self._held_joint_targets[hand] = (
            np.array(proprio[TRUNK_QPOS], copy=True),
            np.array(proprio[LEFT_ARM_QPOS], copy=True),
            np.array(proprio[RIGHT_ARM_QPOS], copy=True),
        )
        return self._sample(
            hand=hand,
            stepped=False,
            held_targets=self._held_joint_targets[hand],
        )

    def step(
        self,
        command: PhysicalReleaseCommand,
    ) -> PhysicalReleaseSample:
        if not isinstance(command, PhysicalReleaseCommand):
            raise R1ProTorsoReleaseBindingError(
                "release step requires PhysicalReleaseCommand"
            )
        held_targets = self._held_joint_targets.get(command.hand)
        if held_targets is None:
            raise R1ProTorsoReleaseBindingError(
                "release read must run before the first step"
            )
        proprio = _proprio(self._inputs.public_proprio())
        action = _release_action(
            proprio,
            hand=command.hand,
            held_targets=held_targets,
        )
        notify_r1pro_command(self._inputs.private_safety, action)
        self._inputs.control.command_direct_action(action)
        try:
            self._inputs.evaluator_step("runtime.release")
        except Exception:
            self._inputs.control.hold()
            raise
        return self._sample(
            hand=command.hand,
            stepped=True,
            held_targets=held_targets,
        )

    def hold_open(self, *, hand: str) -> None:
        _require_hand(hand)
        held_targets = self._held_joint_targets.get(hand)
        if held_targets is None:
            raise R1ProTorsoReleaseBindingError(
                "release read must run before hold_open"
            )
        proprio = _proprio(self._inputs.public_proprio())
        self._inputs.control.command_direct_action(
            _release_action(
                proprio,
                hand=hand,
                held_targets=held_targets,
            )
        )

    def _sample(
        self,
        *,
        hand: str,
        stepped: bool,
        held_targets: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> PhysicalReleaseSample:
        proprio = _proprio(self._inputs.public_proprio())
        state = _public_state(self._inputs.control)
        qvel = _vector(state["base_qvel"], 3, "base_qvel")
        stability = _stability(self._inputs.public_stability())
        safety = (
            _private_safety(self._inputs.private_safety())
            if stepped
            else _empty_private_safety()
        )
        qpos_slice, qvel_slice = _gripper_slices(hand)
        trunk_target, left_target, right_target = held_targets
        arm_tracking_error = max(
            float(np.max(np.abs(proprio[TRUNK_QPOS] - trunk_target))),
            float(np.max(np.abs(proprio[LEFT_ARM_QPOS] - left_target))),
            float(np.max(np.abs(proprio[RIGHT_ARM_QPOS] - right_target))),
        )
        dt = 1.0 / self._inputs.command_frequency_hz
        return PhysicalReleaseSample(
            gripper_qpos=tuple(
                float(value) for value in proprio[qpos_slice]
            ),
            gripper_qvel=tuple(
                float(value) for value in proprio[qvel_slice]
            ),
            forward_delta_m=qvel[0] * dt if stepped else 0.0,
            left_delta_m=qvel[1] * dt if stepped else 0.0,
            yaw_delta_deg=(
                math.degrees(qvel[2] * dt) if stepped else 0.0
            ),
            arm_tracking_error_rad=arm_tracking_error,
            joint_stalled=(
                any(safety.values())
                or _stability_unsafe(stability, self._inputs)
            ),
        )


def _release_action(
    proprio: np.ndarray,
    *,
    hand: str,
    held_targets: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    _require_hand(hand)
    trunk_target, left_target, right_target = held_targets
    action = np.zeros(R1PRO_ACTION_LAYOUT.dimension, dtype=np.float32)
    action[R1PRO_ACTION_LAYOUT.slice_for("base")] = (0.0, 0.0, 0.0)
    action[R1PRO_ACTION_LAYOUT.slice_for("torso")] = trunk_target
    action[R1PRO_ACTION_LAYOUT.slice_for("left_arm")] = left_target
    action[R1PRO_ACTION_LAYOUT.slice_for("right_arm")] = right_target
    # The non-selected hand remains open. Only the selected hand changes from
    # its prior closed carry command to the official +1 open command.
    action[R1PRO_ACTION_LAYOUT.slice_for("left_gripper")] = 1.0
    action[R1PRO_ACTION_LAYOUT.slice_for("right_gripper")] = 1.0
    action[R1PRO_ACTION_LAYOUT.slice_for(f"{hand}_gripper")] = 1.0
    if action.shape != (23,) or not np.isfinite(action).all():
        raise R1ProTorsoReleaseBindingError(
            "release action must be finite and 23D"
        )
    return action


def _joint_hold_action(proprio: np.ndarray) -> np.ndarray:
    action = np.zeros(R1PRO_ACTION_LAYOUT.dimension, dtype=np.float32)
    action[R1PRO_ACTION_LAYOUT.slice_for("torso")] = proprio[TRUNK_QPOS]
    action[R1PRO_ACTION_LAYOUT.slice_for("left_arm")] = proprio[
        LEFT_ARM_QPOS
    ]
    action[R1PRO_ACTION_LAYOUT.slice_for("right_arm")] = proprio[
        RIGHT_ARM_QPOS
    ]
    return action


def _public_state(control: Any) -> Mapping[str, Any]:
    state = control.public_state()
    if not isinstance(state, Mapping):
        raise R1ProTorsoReleaseBindingError(
            "SharedCanaryControl public_state must be a mapping"
        )
    if set(state) != _PUBLIC_STATE_FIELDS:
        raise R1ProTorsoReleaseBindingError(
            "SharedCanaryControl public_state fields do not match"
        )
    count = state["observation_count"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
    ):
        raise R1ProTorsoReleaseBindingError(
            "SharedCanaryControl observation_count must be a non-negative integer"
        )
    return state


def _proprio(value: Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (PROPRIO_DIM,) or not np.isfinite(array).all():
        raise R1ProTorsoReleaseBindingError(
            f"public R1Pro proprio must be finite {PROPRIO_DIM}D"
        )
    return array


def _stability(value: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != _STABILITY_FIELDS:
        raise R1ProTorsoReleaseBindingError(
            "public stability proxy fields do not match"
        )
    return {
        field: _finite(value[field], field)
        for field in _STABILITY_FIELDS
    }


def _private_safety(value: Mapping[str, Any]) -> dict[str, bool]:
    if not isinstance(value, Mapping) or set(value) != _PRIVATE_SAFETY_FIELDS:
        raise R1ProTorsoReleaseBindingError(
            "private safety fields must be anonymous booleans only"
        )
    result: dict[str, bool] = {}
    for field in _PRIVATE_SAFETY_FIELDS:
        item = value[field]
        if not isinstance(item, bool):
            raise R1ProTorsoReleaseBindingError(
                f"private safety {field} must be boolean"
            )
        result[field] = item
    return result


def _empty_private_safety() -> dict[str, bool]:
    return {field: False for field in _PRIVATE_SAFETY_FIELDS}


def _stability_unsafe(
    stability: Mapping[str, float],
    inputs: _Inputs,
) -> bool:
    return (
        abs(stability["base_z_drift_m"])
        > inputs.max_base_z_drift_m
        or abs(stability["roll_deg"])
        > inputs.max_abs_roll_pitch_deg
        or abs(stability["pitch_deg"])
        > inputs.max_abs_roll_pitch_deg
    )


def _gripper_slices(hand: str) -> tuple[slice, slice]:
    _require_hand(hand)
    if hand == "left":
        return LEFT_GRIPPER_QPOS, LEFT_GRIPPER_QVEL
    return RIGHT_GRIPPER_QPOS, RIGHT_GRIPPER_QVEL


def _require_hand(hand: str) -> None:
    if hand not in {"left", "right"}:
        raise R1ProTorsoReleaseBindingError(
            "hand must be left or right"
        )


def _vector(value: Any, size: int, field: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise R1ProTorsoReleaseBindingError(
            f"{field} must be a finite {size}-vector"
        )
    return tuple(float(item) for item in array)


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R1ProTorsoReleaseBindingError(
            f"{field} must be finite"
        )
    result = float(value)
    if not math.isfinite(result):
        raise R1ProTorsoReleaseBindingError(
            f"{field} must be finite"
        )
    return result


def _positive(value: Any, field: str) -> float:
    result = _finite(value, field)
    if result <= 0.0:
        raise R1ProTorsoReleaseBindingError(
            f"{field} must be positive"
        )
    return result
