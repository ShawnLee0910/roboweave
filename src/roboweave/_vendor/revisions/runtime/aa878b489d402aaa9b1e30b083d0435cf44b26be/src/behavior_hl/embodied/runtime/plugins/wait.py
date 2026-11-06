"""Fixed-duration physical hold plugin for the public ``wait`` action."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from behavior_hl.embodied.policy.action_specs.wait import WAIT_SIMULATOR_STEPS
from behavior_hl.embodied.policy.protocol import ActionEnvelope
from behavior_hl.embodied.runtime.dispatch import (
    ActionExecutionContext,
    ActionPluginResult,
    ActionPluginTick,
)
from behavior_hl.embodied.runtime.plugins.carry import CarryCommand, CarrySample
from behavior_hl.embodied.runtime.plugins.local_base import (
    LOCAL_BASE_ACTUATOR_SERVICE,
    LocalBaseSample,
)
from behavior_hl.embodied.runtime.plugins.torso import PUBLIC_GRASP_PROXY_SERVICE


CARRY_ACTUATOR_SERVICE = "carry_actuator"


class WaitContractError(RuntimeError):
    """Raised before wait can issue a malformed physical hold command."""


@dataclass(slots=True)
class _WaitState:
    actuator: Any = field(repr=False)
    carrying: bool
    sample: LocalBaseSample | CarrySample
    steps: int = 0
    terminal_status: str | None = None


@dataclass(slots=True)
class FixedDurationWaitPlugin:
    """Advance exactly 30 evaluator checkpoints using zero base commands."""

    controller_key: str = "temporal.wait"
    duration_steps: int = WAIT_SIMULATOR_STEPS
    proxy_service_key: str = PUBLIC_GRASP_PROXY_SERVICE

    def __post_init__(self) -> None:
        if not isinstance(self.controller_key, str) or not self.controller_key:
            raise ValueError("controller_key must not be empty")
        if self.duration_steps != WAIT_SIMULATOR_STEPS:
            raise ValueError("public wait duration is frozen at 30 steps")

    def begin(
        self,
        action: ActionEnvelope,
        context: ActionExecutionContext,
    ) -> _WaitState:
        if action.action != "wait":
            raise WaitContractError(
                f"{self.controller_key} received {action.action!r}"
            )
        source = context.services.get(self.proxy_service_key)
        read_proxy = getattr(source, "grasp_proxy_likely", None)
        if not callable(read_proxy):
            raise WaitContractError("wait requires the public carry proxy")
        carrying = read_proxy()
        if not isinstance(carrying, bool):
            raise WaitContractError("public carry proxy must return bool")
        service_key = (
            CARRY_ACTUATOR_SERVICE if carrying else LOCAL_BASE_ACTUATOR_SERVICE
        )
        actuator = context.services.get(service_key)
        if not all(
            callable(getattr(actuator, method, None))
            for method in ("read", "step", "hold")
        ):
            raise WaitContractError(
                f"wait actuator service {service_key!r} is invalid"
            )
        sample = actuator.read()
        expected = CarrySample if carrying else LocalBaseSample
        if not isinstance(sample, expected):
            raise WaitContractError("wait actuator returned an invalid sample")
        return _WaitState(
            actuator=actuator,
            carrying=carrying,
            sample=sample,
        )

    def tick(
        self,
        state: _WaitState,
        context: ActionExecutionContext,
    ) -> ActionPluginTick:
        del context
        if state.terminal_status is not None:
            return ActionPluginTick(True, 0, _feedback(state))
        if state.carrying:
            before = state.sample
            assert isinstance(before, CarrySample)
            if (
                not before.grasp_proxy_likely
                or before.active_hand not in {"left", "right"}
            ):
                state.actuator.hold()
                state.terminal_status = "unsafe_while_carrying"
                return ActionPluginTick(True, 0, _feedback(state))
            after = state.actuator.step(
                CarryCommand(
                    active_hand=before.active_hand,
                    base_velocity=(0.0, 0.0, 0.0),
                    torso_target=before.torso_qpos,
                    active_arm_target=before.active_arm_qpos,
                )
            )
            if not isinstance(after, CarrySample):
                raise WaitContractError("carry wait returned an invalid sample")
            if (
                not after.grasp_proxy_likely
                or after.active_hand != before.active_hand
            ):
                state.actuator.hold()
                state.terminal_status = "unsafe_while_carrying"
        else:
            after = state.actuator.step((0.0, 0.0, 0.0))
            if not isinstance(after, LocalBaseSample):
                raise WaitContractError("empty wait returned an invalid sample")
        state.sample = after
        state.steps += 1
        if state.terminal_status is None and state.steps == self.duration_steps:
            state.actuator.hold()
            state.terminal_status = "completed"
        return ActionPluginTick(
            done=state.terminal_status is not None,
            simulator_steps=1,
            public_feedback=_feedback(state),
            capture_observation=(state.steps == self.duration_steps),
        )

    def finish(
        self,
        state: _WaitState,
        context: ActionExecutionContext,
    ) -> ActionPluginResult:
        del context
        return ActionPluginResult(
            state.terminal_status or "aborted",
            _feedback(state),
        )

    def abort(
        self,
        state: _WaitState,
        context: ActionExecutionContext,
        reason: str,
    ) -> ActionPluginResult:
        del context, reason
        state.actuator.hold()
        if state.terminal_status is None:
            state.terminal_status = "aborted"
        return ActionPluginResult(state.terminal_status, _feedback(state))


def _feedback(state: _WaitState) -> dict[str, Any]:
    return {
        "elapsed_steps": state.steps,
        "duration_steps": WAIT_SIMULATOR_STEPS,
        "fresh_observation": state.steps == WAIT_SIMULATOR_STEPS,
    }


@dataclass(slots=True)
class SymbolicWaitPlugin:
    """Refresh the Action20 observation without advancing simulation time."""

    controller_key: str = "temporal.wait"

    def begin(
        self,
        action: ActionEnvelope,
        context: ActionExecutionContext,
    ) -> str:
        del context
        if action.action != "wait":
            raise WaitContractError(
                f"{self.controller_key} received {action.action!r}"
            )
        return "completed"

    def tick(
        self,
        state: str,
        context: ActionExecutionContext,
    ) -> ActionPluginTick:
        del state, context
        return ActionPluginTick(
            done=True,
            simulator_steps=0,
            public_feedback={
                "elapsed_steps": 0,
                "duration_steps": 0,
                "fresh_observation": True,
                "symbolic_refresh": True,
            },
            capture_observation=True,
        )

    def finish(
        self,
        state: str,
        context: ActionExecutionContext,
    ) -> ActionPluginResult:
        del context
        return ActionPluginResult(
            state,
            {
                "elapsed_steps": 0,
                "duration_steps": 0,
                "fresh_observation": True,
                "symbolic_refresh": True,
            },
        )

    def abort(
        self,
        state: str,
        context: ActionExecutionContext,
        reason: str,
    ) -> ActionPluginResult:
        del state, context, reason
        return ActionPluginResult("aborted", {"symbolic_refresh": False})
