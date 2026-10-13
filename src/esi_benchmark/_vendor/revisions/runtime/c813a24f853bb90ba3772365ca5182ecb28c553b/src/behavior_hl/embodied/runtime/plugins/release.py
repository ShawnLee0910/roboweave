"""Physical gripper release with public proprioceptive confirmation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from behavior_hl.embodied.policy.protocol import (
    ActionEnvelope,
    ReleaseArgs,
)
from behavior_hl.embodied.runtime.dispatch import (
    ActionExecutionContext,
    ActionPluginResult,
    ActionPluginTick,
)


PHYSICAL_RELEASE_ACTUATOR_SERVICE = "physical_release_actuator"
PHYSICAL_RELEASE_ACTUATOR_VERSION = (
    "behavior2026.r1pro-physical-release/v1"
)


class ReleaseContractError(RuntimeError):
    """Raised when a physical release binding violates the contract."""


@dataclass(frozen=True, slots=True)
class PhysicalReleaseConfig:
    open_qpos: tuple[float, float]
    qpos_tolerance: float = 0.01
    qvel_epsilon: float = 0.002
    stable_steps: int = 5
    max_base_translation_m: float = 0.01
    max_base_rotation_deg: float = 0.5
    max_arm_tracking_error_rad: float = 0.04

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "open_qpos",
            _pair(self.open_qpos, "open_qpos"),
        )
        for name in (
            "qpos_tolerance",
            "qvel_epsilon",
            "max_base_translation_m",
            "max_base_rotation_deg",
            "max_arm_tracking_error_rad",
        ):
            value = _finite(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if (
            isinstance(self.stable_steps, bool)
            or not isinstance(self.stable_steps, int)
            or self.stable_steps <= 0
        ):
            raise ValueError("stable_steps must be a positive integer")


@dataclass(frozen=True, slots=True)
class PhysicalReleaseCommand:
    """Open one gripper while explicitly holding base, torso, and arms."""

    hand: str
    open_command: float = 1.0
    base_velocity: tuple[float, ...] = (0.0, 0.0, 0.0)
    hold_torso: bool = True
    hold_arms: bool = True

    def __post_init__(self) -> None:
        if self.hand not in {"left", "right"}:
            raise ValueError("hand must be left or right")
        command = _finite(self.open_command, "open_command")
        if command != 1.0:
            raise ValueError("physical release must use the calibrated open command")
        object.__setattr__(self, "open_command", command)
        base = tuple(
            _finite(value, f"base_velocity[{index}]")
            for index, value in enumerate(self.base_velocity)
        )
        if len(base) != 3 or any(value != 0.0 for value in base):
            raise ValueError("physical release must command zero base velocity")
        object.__setattr__(self, "base_velocity", base)
        if self.hold_torso is not True or self.hold_arms is not True:
            raise ValueError("physical release must hold torso and arms")


@dataclass(frozen=True, slots=True)
class PhysicalReleaseSample:
    gripper_qpos: tuple[float, float]
    gripper_qvel: tuple[float, float]
    forward_delta_m: float = 0.0
    left_delta_m: float = 0.0
    yaw_delta_deg: float = 0.0
    arm_tracking_error_rad: float = 0.0
    joint_stalled: bool = False
    simulator_steps: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gripper_qpos",
            _pair(self.gripper_qpos, "gripper_qpos"),
        )
        object.__setattr__(
            self,
            "gripper_qvel",
            _pair(self.gripper_qvel, "gripper_qvel"),
        )
        for name in (
            "forward_delta_m",
            "left_delta_m",
            "yaw_delta_deg",
            "arm_tracking_error_rad",
        ):
            _finite(getattr(self, name), name)
        if self.arm_tracking_error_rad < 0.0:
            raise ValueError("arm_tracking_error_rad must be non-negative")
        if not isinstance(self.joint_stalled, bool):
            raise ValueError("joint_stalled must be boolean")
        if (
            isinstance(self.simulator_steps, bool)
            or not isinstance(self.simulator_steps, int)
            or self.simulator_steps != 1
        ):
            raise ValueError(
                "physical release safety sample must represent exactly one step"
            )


class PhysicalReleaseActuator(Protocol):
    api_contract_version: str

    def read(self, *, hand: str) -> PhysicalReleaseSample:
        """Read only public gripper/base/arm proprioceptive evidence."""

    def step(
        self,
        command: PhysicalReleaseCommand,
    ) -> PhysicalReleaseSample:
        """Advance one real gripper-open controller tick."""

    def hold_open(self, *, hand: str) -> None:
        """Keep the gripper open and hold all other robot joints/base."""


@dataclass(slots=True)
class _ReleaseState:
    actuator: PhysicalReleaseActuator = field(repr=False)
    hand: str
    sample: PhysicalReleaseSample
    config: PhysicalReleaseConfig
    stable_steps: int = 0
    terminal_status: str | None = None
    forward_m: float = 0.0
    left_m: float = 0.0
    yaw_deg: float = 0.0


@dataclass(slots=True)
class PhysicalReleasePlugin:
    controller_key: str
    config: PhysicalReleaseConfig
    actuator_service_key: str = PHYSICAL_RELEASE_ACTUATOR_SERVICE

    def __post_init__(self) -> None:
        if not isinstance(self.controller_key, str) or not self.controller_key:
            raise ValueError("controller_key must not be empty")

    def begin(
        self,
        action: ActionEnvelope,
        context: ActionExecutionContext,
    ) -> _ReleaseState:
        if not isinstance(action.args, ReleaseArgs):
            raise ReleaseContractError(
                "physical release requires parsed ReleaseArgs"
            )
        try:
            actuator = context.services[self.actuator_service_key]
        except KeyError as exc:
            raise ReleaseContractError(
                f"missing physical release actuator: "
                f"{self.actuator_service_key!r}"
            ) from exc
        if (
            getattr(actuator, "api_contract_version", None)
            != PHYSICAL_RELEASE_ACTUATOR_VERSION
            or not callable(getattr(actuator, "read", None))
            or not callable(getattr(actuator, "step", None))
            or not callable(getattr(actuator, "hold_open", None))
        ):
            raise ReleaseContractError(
                "physical release actuator contract/version mismatch"
            )
        sample = actuator.read(hand=action.args.hand)
        if not isinstance(sample, PhysicalReleaseSample):
            raise ReleaseContractError(
                "physical release actuator returned an invalid sample"
            )
        return _ReleaseState(
            actuator=actuator,
            hand=action.args.hand,
            sample=sample,
            config=self.config,
        )

    def tick(
        self,
        state: _ReleaseState,
        context: ActionExecutionContext,
    ) -> ActionPluginTick:
        del context
        if state.terminal_status is not None:
            return ActionPluginTick(done=True, simulator_steps=0)
        after = state.actuator.step(
            PhysicalReleaseCommand(hand=state.hand)
        )
        if not isinstance(after, PhysicalReleaseSample):
            raise ReleaseContractError(
                "physical release actuator returned an invalid step sample"
            )
        state.forward_m += after.forward_delta_m
        state.left_m += after.left_delta_m
        state.yaw_deg += after.yaw_delta_deg
        unsafe = (
            after.joint_stalled
            or after.arm_tracking_error_rad
            > self.config.max_arm_tracking_error_rad
            or math.hypot(state.forward_m, state.left_m)
            > self.config.max_base_translation_m
            or abs(state.yaw_deg) > self.config.max_base_rotation_deg
        )
        if unsafe:
            state.actuator.hold_open(hand=state.hand)
            state.terminal_status = "joint_stalled"
        elif _public_open(after, self.config):
            state.stable_steps += 1
            if state.stable_steps >= self.config.stable_steps:
                state.actuator.hold_open(hand=state.hand)
                state.terminal_status = "completed"
        else:
            state.stable_steps = 0
        state.sample = after
        return ActionPluginTick(
            done=state.terminal_status is not None,
            simulator_steps=after.simulator_steps,
            public_feedback=_release_feedback(state),
        )

    def finish(
        self,
        state: _ReleaseState,
        context: ActionExecutionContext,
    ) -> ActionPluginResult:
        del context
        return ActionPluginResult(
            state.terminal_status or "joint_stalled",
            _release_feedback(state),
        )

    def abort(
        self,
        state: _ReleaseState,
        context: ActionExecutionContext,
        reason: str,
    ) -> ActionPluginResult:
        del context, reason
        state.actuator.hold_open(hand=state.hand)
        state.terminal_status = "joint_stalled"
        return ActionPluginResult(
            "joint_stalled",
            _release_feedback(state),
        )


def _public_open(
    sample: PhysicalReleaseSample,
    config: PhysicalReleaseConfig,
) -> bool:
    return (
        all(
            abs(actual - expected) <= config.qpos_tolerance
            for actual, expected in zip(
                sample.gripper_qpos,
                config.open_qpos,
            )
        )
        and max(abs(value) for value in sample.gripper_qvel)
        <= config.qvel_epsilon
    )


def _release_feedback(state: _ReleaseState) -> dict[str, Any]:
    return {
        "hand": state.hand,
        "gripper_open_confirmed": (
            state.terminal_status == "completed"
        ),
        "public_stable_steps": state.stable_steps,
        "base_stable": (
            math.hypot(state.forward_m, state.left_m)
            <= state.config.max_base_translation_m
            and abs(state.yaw_deg)
            <= state.config.max_base_rotation_deg
        ),
    }


def _pair(
    value: Sequence[float],
    field: str,
) -> tuple[float, float]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must contain two finite numbers")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError(
            f"{field} must contain two finite numbers"
        ) from exc
    if len(items) != 2:
        raise ValueError(f"{field} must contain two finite numbers")
    return (
        _finite(items[0], f"{field}[0]"),
        _finite(items[1], f"{field}[1]"),
    )


def _finite(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result
