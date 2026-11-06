"""Versioned live bindings from SharedCanaryControl to runtime actuators.

The bridge consumes only the policy-visible R1Pro proprioception, public
clearance / grasp proxies, and anonymous boolean safety signals.  Its step
callback is the only operation allowed to advance Evaluator physics.
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
from behavior_hl.embodied.runtime.plugins.carry import (
    CarryCommand,
    CarrySample,
)
from behavior_hl.embodied.runtime.plugins.local_base import LocalBaseSample


R1PRO_ACTUATOR_BINDING_VERSION = "r1pro-shared-canary-actuators-v2"
R1PRO_LOCAL_BASE_BINDING_VERSION = (
    "r1pro-shared-canary-local-base-v2"
)
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


class R1ProActuatorBindingError(RuntimeError):
    """Raised before unsafe or unsupported live control can advance physics."""


@dataclass(frozen=True, slots=True)
class R1ProActuatorPreflight:
    version: str
    proprio_dimension: int
    command_frequency_hz: float
    local_base_supported: bool
    carry_look_supported: bool
    carry_base_supported: bool
    public_inputs: tuple[str, ...]
    private_safety_inputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class R1ProLocalBasePreflight:
    """Minimal base-only capability report with no carry capability."""

    version: str
    proprio_dimension: int
    command_frequency_hz: float
    public_inputs: tuple[str, ...]
    private_safety_inputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class R1ProLocalBaseBinding:
    """A base-only binding that cannot expose or construct carry control."""

    local_base: "SharedCanaryLocalBaseActuator"
    preflight: R1ProLocalBasePreflight


@dataclass(frozen=True, slots=True)
class R1ProActuatorBindings:
    local_base: "SharedCanaryLocalBaseActuator"
    carry: "SharedCanaryCarryActuator"
    preflight: R1ProActuatorPreflight


@dataclass(frozen=True, slots=True)
class _BaseInputs:
    control: Any
    evaluator_step: Callable[[str], Any]
    public_proprio: Callable[[], Sequence[float]]
    public_stability: Callable[[], Mapping[str, Any]]
    private_safety: Callable[[], Mapping[str, Any]]
    command_frequency_hz: float


@dataclass(frozen=True, slots=True)
class _CarryInputs(_BaseInputs):
    public_grasp_proxy: Callable[[], bool]


def preflight_r1pro_local_base_binding(
    *,
    control: Any,
    evaluator_step: Callable[[str], Any],
    public_proprio: Callable[[], Sequence[float]],
    public_stability: Callable[[], Mapping[str, Any]],
    private_safety: Callable[[], Mapping[str, Any]],
    command_frequency_hz: float,
) -> R1ProLocalBasePreflight:
    """Validate only capabilities consumed by local-base control."""

    for method in ("command_base", "hold", "public_state"):
        if not callable(getattr(control, method, None)):
            raise R1ProActuatorBindingError(
                f"SharedCanaryControl is missing {method}"
            )
    for name, callback in (
        ("evaluator_step", evaluator_step),
        ("public_proprio", public_proprio),
        ("public_stability", public_stability),
        ("private_safety", private_safety),
    ):
        if not callable(callback):
            raise R1ProActuatorBindingError(f"{name} must be callable")
    frequency = _positive(command_frequency_hz, "command_frequency_hz")
    _validate_base_public_inputs(
        control=control,
        public_proprio=public_proprio,
        public_stability=public_stability,
    )
    # private_safety remains post-step only and is not sampled by preflight.
    return R1ProLocalBasePreflight(
        version=R1PRO_LOCAL_BASE_BINDING_VERSION,
        proprio_dimension=PROPRIO_DIM,
        command_frequency_hz=frequency,
        public_inputs=(
            "base_qvel",
            "head_forward_clearance_m",
            "r1pro_proprio",
            "stability_proxy",
        ),
        private_safety_inputs=tuple(sorted(_PRIVATE_SAFETY_FIELDS)),
    )


def preflight_r1pro_actuator_binding(
    *,
    control: Any,
    evaluator_step: Callable[[str], Any],
    public_proprio: Callable[[], Sequence[float]],
    public_grasp_proxy: Callable[[], bool],
    public_stability: Callable[[], Mapping[str, Any]],
    private_safety: Callable[[], Mapping[str, Any]],
    command_frequency_hz: float,
) -> R1ProActuatorPreflight:
    """Read current public inputs without commanding or stepping Evaluator."""

    for method in (
        "command_base",
        "command_direct_action",
        "command_carry_action",
        "hold",
        "public_state",
    ):
        if not callable(getattr(control, method, None)):
            raise R1ProActuatorBindingError(
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
            raise R1ProActuatorBindingError(f"{name} must be callable")
    frequency = _positive(command_frequency_hz, "command_frequency_hz")

    _validate_base_public_inputs(
        control=control,
        public_proprio=public_proprio,
        public_stability=public_stability,
    )
    proxy = public_grasp_proxy()
    if not isinstance(proxy, bool):
        raise R1ProActuatorBindingError(
            "public grasp proxy must be boolean"
        )
    # private_safety is intentionally not called by preflight.  Contact APIs
    # are sampled only after a real physical step and never select a target.
    return R1ProActuatorPreflight(
        version=R1PRO_ACTUATOR_BINDING_VERSION,
        proprio_dimension=PROPRIO_DIM,
        command_frequency_hz=frequency,
        local_base_supported=True,
        carry_look_supported=True,
        # MODE_CARRY_ACTION preserves the explicit base command and the server
        # independently enforces that the active gripper stays closed.
        carry_base_supported=True,
        public_inputs=(
            "base_qvel",
            "head_forward_clearance_m",
            "public_grasp_proxy",
            "r1pro_proprio",
            "stability_proxy",
        ),
        private_safety_inputs=tuple(sorted(_PRIVATE_SAFETY_FIELDS)),
    )


def build_r1pro_local_base_binding(
    *,
    control: Any,
    evaluator_step: Callable[[str], Any],
    public_proprio: Callable[[], Sequence[float]],
    public_stability: Callable[[], Mapping[str, Any]],
    private_safety: Callable[[], Mapping[str, Any]],
    command_frequency_hz: float,
) -> R1ProLocalBaseBinding:
    """Construct only local-base control, with no hand or carry object."""

    preflight = preflight_r1pro_local_base_binding(
        control=control,
        evaluator_step=evaluator_step,
        public_proprio=public_proprio,
        public_stability=public_stability,
        private_safety=private_safety,
        command_frequency_hz=command_frequency_hz,
    )
    inputs = _BaseInputs(
        control=control,
        evaluator_step=evaluator_step,
        public_proprio=public_proprio,
        public_stability=public_stability,
        private_safety=private_safety,
        command_frequency_hz=preflight.command_frequency_hz,
    )
    return R1ProLocalBaseBinding(
        local_base=SharedCanaryLocalBaseActuator(inputs),
        preflight=preflight,
    )


def build_r1pro_actuator_bindings(
    *,
    control: Any,
    evaluator_step: Callable[[str], Any],
    public_proprio: Callable[[], Sequence[float]],
    public_grasp_proxy: Callable[[], bool],
    public_stability: Callable[[], Mapping[str, Any]],
    private_safety: Callable[[], Mapping[str, Any]],
    command_frequency_hz: float,
    active_hand: str,
) -> R1ProActuatorBindings:
    preflight = preflight_r1pro_actuator_binding(
        control=control,
        evaluator_step=evaluator_step,
        public_proprio=public_proprio,
        public_grasp_proxy=public_grasp_proxy,
        public_stability=public_stability,
        private_safety=private_safety,
        command_frequency_hz=command_frequency_hz,
    )
    if active_hand not in {"left", "right"}:
        raise R1ProActuatorBindingError(
            "active_hand must be left or right"
        )
    base_inputs = _BaseInputs(
        control=control,
        evaluator_step=evaluator_step,
        public_proprio=public_proprio,
        public_stability=public_stability,
        private_safety=private_safety,
        command_frequency_hz=preflight.command_frequency_hz,
    )
    carry_inputs = _CarryInputs(
        control=control,
        evaluator_step=evaluator_step,
        public_proprio=public_proprio,
        public_stability=public_stability,
        private_safety=private_safety,
        command_frequency_hz=preflight.command_frequency_hz,
        public_grasp_proxy=public_grasp_proxy,
    )
    return R1ProActuatorBindings(
        local_base=SharedCanaryLocalBaseActuator(base_inputs),
        carry=SharedCanaryCarryActuator(
            carry_inputs,
            active_hand=active_hand,
        ),
        preflight=preflight,
    )


class SharedCanaryLocalBaseActuator:
    """LocalBaseActuator backed by one command and one Evaluator step."""

    def __init__(self, inputs: _BaseInputs) -> None:
        self._inputs = inputs

    def read(self) -> LocalBaseSample:
        return self._sample(stepped=False)

    def step(
        self,
        command: tuple[float, float, float],
    ) -> LocalBaseSample:
        base = _vector(command, 3, "base_command")
        if any(abs(value) > 1.0 for value in base):
            raise R1ProActuatorBindingError(
                "base command exceeds normalized limits"
            )
        action = _joint_hold_action(
            _proprio(self._inputs.public_proprio())
        )
        action[R1PRO_ACTION_LAYOUT.slice_for("base")] = base
        notify_r1pro_command(self._inputs.private_safety, action)
        self._inputs.control.command_base(base)
        try:
            self._inputs.evaluator_step("runtime.local_base")
        except Exception:
            self._inputs.control.hold()
            raise
        return self._sample(stepped=True)

    def hold(self) -> None:
        self._inputs.control.hold()

    def _sample(self, *, stepped: bool) -> LocalBaseSample:
        state = _public_state(self._inputs.control)
        qvel = _vector(state["base_qvel"], 3, "base_qvel")
        stability = _stability(self._inputs.public_stability())
        safety = (
            _private_safety(self._inputs.private_safety())
            if stepped
            else _empty_private_safety()
        )
        dt = 1.0 / self._inputs.command_frequency_hz
        return LocalBaseSample(
            base_qvel=qvel,
            head_forward_clearance_m=_nonnegative_finite(
                state["head_forward_clearance_m"],
                "head_forward_clearance_m",
            ),
            forward_delta_m=qvel[0] * dt if stepped else 0.0,
            left_delta_m=qvel[1] * dt if stepped else 0.0,
            yaw_delta_deg=(
                math.degrees(qvel[2] * dt) if stepped else 0.0
            ),
            base_contact=safety["base_contact"],
            **stability,
        )


class SharedCanaryCarryActuator:
    """CarryActuator for direct torso/arm control with a closed active hand."""

    def __init__(
        self,
        inputs: _CarryInputs,
        *,
        active_hand: str,
    ) -> None:
        self._inputs = inputs
        self._active_hand = active_hand

    def read(self) -> CarrySample:
        return self._sample(stepped=False)

    def step(self, command: CarryCommand) -> CarrySample:
        if not isinstance(command, CarryCommand):
            raise R1ProActuatorBindingError(
                "carry step requires CarryCommand"
            )
        if command.active_hand != self._active_hand:
            raise R1ProActuatorBindingError(
                "CarryCommand changed the calibrated active hand"
            )
        proprio = _proprio(self._inputs.public_proprio())
        action = _carry_action(
            proprio,
            command=command,
            active_hand=self._active_hand,
        )
        notify_r1pro_command(self._inputs.private_safety, action)
        self._inputs.control.command_carry_action(
            action,
            active_hand=self._active_hand,
        )
        try:
            self._inputs.evaluator_step("runtime.carry")
        except Exception:
            self.hold()
            raise
        return self._sample(stepped=True)

    def hold(self) -> None:
        proprio = _proprio(self._inputs.public_proprio())
        action = _carry_hold_action(
            proprio,
            active_hand=self._active_hand,
        )
        self._inputs.control.command_carry_action(
            action,
            active_hand=self._active_hand,
        )

    def _sample(self, *, stepped: bool) -> CarrySample:
        proprio = _proprio(self._inputs.public_proprio())
        state = _public_state(self._inputs.control)
        qvel = _vector(state["base_qvel"], 3, "base_qvel")
        stability = _stability(self._inputs.public_stability())
        safety = (
            _private_safety(self._inputs.private_safety())
            if stepped
            else _empty_private_safety()
        )
        proxy = self._inputs.public_grasp_proxy()
        if not isinstance(proxy, bool):
            raise R1ProActuatorBindingError(
                "public grasp proxy must be boolean"
            )
        arm_slice = (
            LEFT_ARM_QPOS
            if self._active_hand == "left"
            else RIGHT_ARM_QPOS
        )
        dt = 1.0 / self._inputs.command_frequency_hz
        return CarrySample(
            active_hand=self._active_hand,
            grasp_proxy_likely=proxy,
            torso_qpos=tuple(float(value) for value in proprio[TRUNK_QPOS]),
            active_arm_qpos=tuple(
                float(value) for value in proprio[arm_slice]
            ),
            forward_delta_m=qvel[0] * dt if stepped else 0.0,
            left_delta_m=qvel[1] * dt if stepped else 0.0,
            yaw_delta_deg=(
                math.degrees(qvel[2] * dt) if stepped else 0.0
            ),
            base_contact=safety["base_contact"],
            predicted_self_collision=safety[
                "predicted_self_collision"
            ],
            joint_stalled=safety["joint_stalled"],
            **stability,
        )


def _carry_action(
    proprio: np.ndarray,
    *,
    command: CarryCommand,
    active_hand: str,
) -> np.ndarray:
    action = _carry_hold_action(proprio, active_hand=active_hand)
    action[R1PRO_ACTION_LAYOUT.slice_for("base")] = command.base_velocity
    action[R1PRO_ACTION_LAYOUT.slice_for("torso")] = command.torso_target
    action[
        R1PRO_ACTION_LAYOUT.slice_for(f"{active_hand}_arm")
    ] = command.active_arm_target
    return action


def _carry_hold_action(
    proprio: np.ndarray,
    *,
    active_hand: str,
) -> np.ndarray:
    action = np.zeros(R1PRO_ACTION_LAYOUT.dimension, dtype=np.float32)
    action[R1PRO_ACTION_LAYOUT.slice_for("torso")] = proprio[TRUNK_QPOS]
    action[R1PRO_ACTION_LAYOUT.slice_for("left_arm")] = proprio[
        LEFT_ARM_QPOS
    ]
    action[R1PRO_ACTION_LAYOUT.slice_for("right_arm")] = proprio[
        RIGHT_ARM_QPOS
    ]
    action[R1PRO_ACTION_LAYOUT.slice_for("left_gripper")] = (
        -1.0 if active_hand == "left" else 1.0
    )
    action[R1PRO_ACTION_LAYOUT.slice_for("right_gripper")] = (
        -1.0 if active_hand == "right" else 1.0
    )
    return action


def _joint_hold_action(proprio: np.ndarray) -> np.ndarray:
    """Build a tracking-only action; base and grippers are detector-irrelevant."""

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
        raise R1ProActuatorBindingError(
            "SharedCanaryControl public_state must be a mapping"
        )
    missing = _PUBLIC_STATE_FIELDS.difference(state)
    if missing:
        raise R1ProActuatorBindingError(
            f"public_state is missing fields: {sorted(missing)}"
        )
    # Ignore no unknown fields: rejecting additions prevents a future bridge
    # from silently starting to consume target coordinates or task truth.
    unknown = set(state).difference(_PUBLIC_STATE_FIELDS)
    if unknown:
        raise R1ProActuatorBindingError(
            f"public_state has undeclared fields: {sorted(unknown)}"
        )
    count = state["observation_count"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
    ):
        raise R1ProActuatorBindingError(
            "public_state observation_count must be a non-negative integer"
        )
    return state


def _validate_base_public_inputs(
    *,
    control: Any,
    public_proprio: Callable[[], Sequence[float]],
    public_stability: Callable[[], Mapping[str, Any]],
) -> None:
    state = _public_state(control)
    # Evaluator.step calls policy.forward(obs_n) before env.step produces
    # evaluator.obs_n+1.  The shared policy state is therefore intentionally
    # one public observation behind the current evaluator proprioception.
    # Validate both sources independently; exact cross-frame equality would
    # reject every physically moving or merely jittering robot.
    _proprio(public_proprio())
    _vector(state["base_qvel"], 3, "base_qvel")
    _nonnegative_finite(
        state["head_forward_clearance_m"],
        "head_forward_clearance_m",
    )
    _stability(public_stability())


def _proprio(value: Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (PROPRIO_DIM,) or not np.isfinite(array).all():
        raise R1ProActuatorBindingError(
            f"public R1Pro proprio must be finite {PROPRIO_DIM}D"
        )
    return array


def _stability(value: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != _STABILITY_FIELDS:
        raise R1ProActuatorBindingError(
            "public stability proxy fields do not match"
        )
    return {
        field: _finite(value[field], field)
        for field in _STABILITY_FIELDS
    }


def _private_safety(value: Mapping[str, Any]) -> dict[str, bool]:
    if not isinstance(value, Mapping) or set(value) != _PRIVATE_SAFETY_FIELDS:
        raise R1ProActuatorBindingError(
            "private safety fields must be anonymous booleans only"
        )
    result = {}
    for field in _PRIVATE_SAFETY_FIELDS:
        item = value[field]
        if not isinstance(item, bool):
            raise R1ProActuatorBindingError(
                f"private safety {field} must be boolean"
            )
        result[field] = item
    return result


def _empty_private_safety() -> dict[str, bool]:
    return {field: False for field in _PRIVATE_SAFETY_FIELDS}


def _vector(value: Any, size: int, field: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise R1ProActuatorBindingError(
            f"{field} must be a finite {size}-vector"
        )
    return tuple(float(item) for item in array)


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R1ProActuatorBindingError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise R1ProActuatorBindingError(f"{field} must be finite")
    return result


def _positive(value: Any, field: str) -> float:
    result = _finite(value, field)
    if result <= 0.0:
        raise R1ProActuatorBindingError(f"{field} must be positive")
    return result


def _nonnegative_finite(value: Any, field: str) -> float:
    result = _finite(value, field)
    if result < 0.0:
        raise R1ProActuatorBindingError(
            f"{field} must be non-negative"
        )
    return result
