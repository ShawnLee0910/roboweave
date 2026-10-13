"""Physical local-base plugins with public odometry and collision feedback."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from behavior_hl.embodied.policy.approach_execution import (
    BASE_VELOCITY_INDEX,
    BaseMacroConfig,
    classify_macro_displacement,
)
from behavior_hl.embodied.policy.feedback import PublicStatus
from behavior_hl.embodied.policy.protocol import ActionEnvelope
from behavior_hl.embodied.runtime.dispatch import (
    ActionExecutionContext,
    ActionPluginResult,
    ActionPluginTick,
)


LOCAL_BASE_ACTUATOR_SERVICE = "local_base_actuator"
PUBLIC_GRASP_PROXY_SERVICE = "public_grasp_proxy_likely"


class LocalBaseContractError(RuntimeError):
    """Raised when the physical local-base service violates its contract."""


@dataclass(frozen=True, slots=True)
class LocalBaseSample:
    """One public robot-motion sample; no world pose or map is accepted."""

    base_qvel: tuple[float, float, float]
    head_forward_clearance_m: float
    forward_delta_m: float = 0.0
    left_delta_m: float = 0.0
    yaw_delta_deg: float = 0.0
    base_contact: bool = False
    base_z_drift_m: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_qvel",
            _vector(self.base_qvel, 3, "base_qvel"),
        )
        for name in (
            "head_forward_clearance_m",
            "forward_delta_m",
            "left_delta_m",
            "yaw_delta_deg",
            "base_z_drift_m",
            "roll_deg",
            "pitch_deg",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.head_forward_clearance_m < 0.0:
            raise ValueError("head_forward_clearance_m must be non-negative")
        if not isinstance(self.base_contact, bool):
            raise ValueError("base_contact must be boolean")


class LocalBaseActuator(Protocol):
    def read(self) -> LocalBaseSample:
        """Read the latest public physical-base sample."""

    def step(self, command: tuple[float, float, float]) -> LocalBaseSample:
        """Apply one normalized local velocity command and step physics once."""

    def hold(self) -> None:
        """Command zero local velocity."""


@dataclass(frozen=True, slots=True)
class LocalBaseSafetyLimits:
    maximum_z_drift_m: float = 0.03
    maximum_abs_roll_pitch_deg: float = 5.0

    def __post_init__(self) -> None:
        for name in ("maximum_z_drift_m", "maximum_abs_roll_pitch_deg"):
            value = _finite(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)


@dataclass(slots=True)
class _LocalBaseState:
    actuator: LocalBaseActuator = field(repr=False)
    sample: LocalBaseSample
    steps: int = 0
    low_speed_streak: int = 0
    slow_mode_seen: bool = False
    forward_m: float = 0.0
    left_m: float = 0.0
    yaw_deg: float = 0.0
    terminal_status: PublicStatus | None = None


@dataclass(slots=True)
class LocalBaseMotionPlugin:
    """Execute one ESI-style relative base macro on the physical R1Pro."""

    controller_key: str
    action_name: str
    config: BaseMacroConfig = field(default_factory=BaseMacroConfig)
    safety: LocalBaseSafetyLimits = field(
        default_factory=LocalBaseSafetyLimits
    )
    actuator_service_key: str = LOCAL_BASE_ACTUATOR_SERVICE

    def __post_init__(self) -> None:
        if not isinstance(self.controller_key, str) or not self.controller_key:
            raise ValueError("controller_key must be non-empty")
        # The config owns the canonical action whitelist and raises otherwise.
        self.config.maximum_step_budget(self.action_name)

    def begin(
        self,
        action: ActionEnvelope,
        context: ActionExecutionContext,
    ) -> _LocalBaseState:
        if action.action != self.action_name:
            raise LocalBaseContractError(
                f"{self.controller_key} received {action.action!r}"
            )
        try:
            actuator = context.services[self.actuator_service_key]
        except KeyError as exc:
            raise LocalBaseContractError(
                f"missing {self.actuator_service_key!r}"
            ) from exc
        if not all(
            callable(getattr(actuator, method, None))
            for method in ("read", "step", "hold")
        ):
            raise LocalBaseContractError("local base actuator is invalid")
        sample = actuator.read()
        if not isinstance(sample, LocalBaseSample):
            raise LocalBaseContractError(
                "local base actuator returned an invalid sample"
            )
        return _LocalBaseState(actuator=actuator, sample=sample)

    def tick(
        self,
        state: _LocalBaseState,
        context: ActionExecutionContext,
    ) -> ActionPluginTick:
        del context
        if state.terminal_status is not None:
            return ActionPluginTick(True, 0, self._feedback(state))
        decision = self.config.step_decision(
            self.action_name,
            head_forward_clearance_m=(
                state.sample.head_forward_clearance_m
            ),
        )
        if decision.hard_stop:
            state.actuator.hold()
            state.terminal_status = PublicStatus.BLOCKED
            return ActionPluginTick(True, 0, self._feedback(state))
        command = tuple(float(value) for value in decision.command)
        after = state.actuator.step(command)
        if not isinstance(after, LocalBaseSample):
            state.actuator.hold()
            raise LocalBaseContractError(
                "local base actuator step returned an invalid sample"
            )
        state.steps += 1
        state.slow_mode_seen = state.slow_mode_seen or decision.slow_mode
        state.forward_m += after.forward_delta_m
        state.left_m += after.left_delta_m
        state.yaw_deg += after.yaw_delta_deg

        if (
            after.base_contact
            or abs(after.base_z_drift_m) > self.safety.maximum_z_drift_m
            or abs(after.roll_deg)
            > self.safety.maximum_abs_roll_pitch_deg
            or abs(after.pitch_deg)
            > self.safety.maximum_abs_roll_pitch_deg
        ):
            state.terminal_status = PublicStatus.BLOCKED
        else:
            velocity_index, velocity_sign = BASE_VELOCITY_INDEX[
                self.action_name
            ]
            projected_speed = (
                velocity_sign * after.base_qvel[velocity_index]
            )
            if state.steps > self.config.stall_warmup_steps:
                if projected_speed < self.config.stall_speed_threshold:
                    state.low_speed_streak += 1
                else:
                    state.low_speed_streak = 0
            if state.low_speed_streak >= self.config.stall_window_steps:
                state.terminal_status = PublicStatus.BLOCKED

        budget = (
            self.config.forward_slow_steps
            if self.action_name == "move_forward"
            and state.slow_mode_seen
            else self.config.step_budget(self.action_name)
        )
        if state.terminal_status is None and state.steps >= budget:
            state.terminal_status = classify_macro_displacement(
                self.action_name,
                forward_m=state.forward_m,
                left_m=state.left_m,
                yaw_deg=state.yaw_deg,
                config=self.config,
            )
        state.sample = after
        if state.terminal_status is not None:
            state.actuator.hold()
        return ActionPluginTick(
            done=state.terminal_status is not None,
            simulator_steps=1,
            public_feedback=self._feedback(state),
        )

    def finish(
        self,
        state: _LocalBaseState,
        context: ActionExecutionContext,
    ) -> ActionPluginResult:
        del context
        if state.terminal_status is None:
            raise LocalBaseContractError("base action is not terminal")
        return ActionPluginResult(
            state.terminal_status.value,
            self._feedback(state),
        )

    def abort(
        self,
        state: _LocalBaseState,
        context: ActionExecutionContext,
        reason: str,
    ) -> ActionPluginResult:
        del context, reason
        state.actuator.hold()
        if state.terminal_status is None:
            state.terminal_status = classify_macro_displacement(
                self.action_name,
                forward_m=state.forward_m,
                left_m=state.left_m,
                yaw_deg=state.yaw_deg,
                config=self.config,
            )
        return ActionPluginResult(
            state.terminal_status.value,
            self._feedback(state),
        )

    def _feedback(self, state: _LocalBaseState) -> dict[str, Any]:
        return {
            "body_odometry": {
                "forward_m": float(state.forward_m),
                "left_m": float(state.left_m),
                "yaw_deg": float(state.yaw_deg),
            },
            "slow_mode_seen": state.slow_mode_seen,
        }


@dataclass(slots=True)
class _AdaptiveState:
    selected: Any = field(repr=False)
    inner_state: Any = field(repr=False)


@dataclass(slots=True)
class AdaptivePhysicalActionPlugin:
    """Select empty/carry controller only from the public grasp proxy."""

    controller_key: str
    empty_plugin: Any = field(repr=False)
    carrying_plugin: Any = field(repr=False)
    selector_service_key: str = PUBLIC_GRASP_PROXY_SERVICE

    def __post_init__(self) -> None:
        if not isinstance(self.controller_key, str) or not self.controller_key:
            raise ValueError("controller_key must be non-empty")
        for plugin in (self.empty_plugin, self.carrying_plugin):
            if getattr(plugin, "controller_key", None) != self.controller_key:
                raise ValueError(
                    "adaptive inner controller_key does not match"
                )

    def begin(
        self,
        action: ActionEnvelope,
        context: ActionExecutionContext,
    ) -> _AdaptiveState:
        try:
            value = context.services[self.selector_service_key]
        except KeyError as exc:
            raise LocalBaseContractError(
                f"missing public selector {self.selector_service_key!r}"
            ) from exc
        if callable(value):
            value = value()
        if not isinstance(value, bool):
            raise LocalBaseContractError(
                "public grasp proxy selector must be boolean"
            )
        selected = self.carrying_plugin if value else self.empty_plugin
        return _AdaptiveState(
            selected=selected,
            inner_state=selected.begin(action, context),
        )

    def tick(
        self,
        state: _AdaptiveState,
        context: ActionExecutionContext,
    ) -> ActionPluginTick:
        return state.selected.tick(state.inner_state, context)

    def finish(
        self,
        state: _AdaptiveState,
        context: ActionExecutionContext,
    ) -> ActionPluginResult:
        return state.selected.finish(state.inner_state, context)

    def abort(
        self,
        state: _AdaptiveState,
        context: ActionExecutionContext,
        reason: str,
    ) -> ActionPluginResult:
        return state.selected.abort(state.inner_state, context, reason)


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != size:
        raise ValueError(f"{name} must contain {size} values")
    return tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))
