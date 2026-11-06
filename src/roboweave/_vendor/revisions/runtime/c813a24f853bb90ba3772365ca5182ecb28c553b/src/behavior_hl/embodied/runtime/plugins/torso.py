"""Physical empty-hand torso look and public-proxy adaptive dispatch."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from behavior_hl.embodied.policy.protocol import ActionEnvelope
from behavior_hl.embodied.runtime.dispatch import (
    ActionExecutionContext,
    ActionPlugin,
    ActionPluginResult,
    ActionPluginTick,
)


PHYSICAL_TORSO_ACTUATOR_SERVICE = "physical_torso_actuator"
PUBLIC_GRASP_PROXY_SERVICE = "public_grasp_proxy"
PHYSICAL_TORSO_ACTUATOR_VERSION = "behavior2026.r1pro-physical-torso/v1"

# G3a-validated JoyLo interpolation points: 0.40 / 0.50 / 0.60 between
# official upright and downward profiles.
EMPTY_TORSO_PROFILES: Mapping[str, tuple[float, ...]] = MappingProxyType(
    {
        "up": (0.91, -1.24, -0.376, 0.0),
        "center": (1.025, -1.45, -0.47, 0.0),
        "down": (1.14, -1.66, -0.564, 0.0),
    }
)

# CPU-FK-derived configurations for the official R1Pro torso.  The onboard
# ZED optical axis is camera-local -Z, so the optical-pitch sign is opposite
# the torso terminal-link angle.
TORSO_HEIGHT_LEVELS = ("low", "neutral", "high")
TORSO_PITCH_LEVELS = ("down", "center", "up")
EMPTY_TORSO_GRID: Mapping[tuple[str, str], tuple[float, ...]] = MappingProxyType(
    {
        ("low", "down"): (0.825224718, -1.670691363, -1.151266032, 0.0),
        ("low", "center"): (1.14, -1.66, -0.564, 0.0),
        ("low", "up"): (1.256252266, -1.305627765, 0.168423888, 0.0),
        ("neutral", "down"): (0.736209854, -1.445350955, -1.015940488, 0.0),
        ("neutral", "center"): (1.025, -1.45, -0.47, 0.0),
        ("neutral", "up"): (1.112430662, -1.076911022, 0.252319028, 0.0),
        ("high", "down"): (0.637038864, -1.21639556, -0.887156084, 0.0),
        ("high", "center"): (0.91, -1.24, -0.376, 0.0),
        ("high", "up"): (0.964823571, -0.827795341, 0.352827617, 0.0),
    }
)
TORSO_GRID_ACTION_DELTAS: Mapping[str, tuple[str, int]] = MappingProxyType(
    {
        "move_up": ("height", 1),
        "move_down": ("height", -1),
        "look_up": ("pitch", 1),
        "look_down": ("pitch", -1),
    }
)


class TorsoContractError(RuntimeError):
    """Raised when a physical torso binding violates the controller contract."""


@dataclass(frozen=True, slots=True)
class EmptyTorsoLimits:
    max_joint_delta_per_tick: float = 0.025
    joint_goal_tolerance: float = 0.02
    joint_tracking_tolerance: float = 0.04
    max_base_translation_m: float = 0.01
    max_base_rotation_deg: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "max_joint_delta_per_tick",
            "joint_goal_tolerance",
            "joint_tracking_tolerance",
            "max_base_translation_m",
            "max_base_rotation_deg",
        ):
            value = _finite(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class EmptyTorsoCommand:
    """One bounded physical command with no viewer-camera or pose target."""

    torso_target: tuple[float, ...]
    base_velocity: tuple[float, ...] = (0.0, 0.0, 0.0)
    hold_arms: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "torso_target",
            _vector(self.torso_target, 4, "torso_target"),
        )
        base = _vector(self.base_velocity, 3, "base_velocity")
        if any(value != 0.0 for value in base):
            raise ValueError("empty torso look must command zero base velocity")
        object.__setattr__(self, "base_velocity", base)
        if self.hold_arms is not True:
            raise ValueError("empty torso look must hold both arms")


@dataclass(frozen=True, slots=True)
class EmptyTorsoSample:
    torso_qpos: tuple[float, ...]
    forward_delta_m: float = 0.0
    left_delta_m: float = 0.0
    yaw_delta_deg: float = 0.0
    joint_stalled: bool = False
    predicted_self_collision: bool = False
    simulator_steps: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "torso_qpos",
            _vector(self.torso_qpos, 4, "torso_qpos"),
        )
        for name in ("forward_delta_m", "left_delta_m", "yaw_delta_deg"):
            _finite(getattr(self, name), name)
        for name in ("joint_stalled", "predicted_self_collision"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if (
            isinstance(self.simulator_steps, bool)
            or not isinstance(self.simulator_steps, int)
            or self.simulator_steps != 1
        ):
            raise ValueError(
                "physical torso safety sample must represent exactly one step"
            )


class PhysicalTorsoActuator(Protocol):
    api_contract_version: str

    def read(self) -> EmptyTorsoSample:
        """Read approved torso proprioception and safety signals."""

    def step(self, command: EmptyTorsoCommand) -> EmptyTorsoSample:
        """Advance one bounded physical torso tick."""

    def hold(self) -> None:
        """Hold the latest safe torso/arm targets and stop the base."""


class PublicGraspProxySource(Protocol):
    def grasp_proxy_likely(self) -> bool:
        """Return only the calibrated public grasp proxy."""


@dataclass(slots=True)
class _EmptyTorsoState:
    actuator: PhysicalTorsoActuator = field(repr=False)
    target: tuple[float, ...]
    profile_name: str
    sample: EmptyTorsoSample
    limits: EmptyTorsoLimits
    terminal_status: str | None = None
    forward_m: float = 0.0
    left_m: float = 0.0
    yaw_deg: float = 0.0


@dataclass(slots=True)
class EmptyTorsoLookPlugin:
    """Move the physical torso to one frozen G3a profile."""

    controller_key: str
    target_profile: str
    limits: EmptyTorsoLimits = field(default_factory=EmptyTorsoLimits)
    actuator_service_key: str = PHYSICAL_TORSO_ACTUATOR_SERVICE

    def __post_init__(self) -> None:
        if not isinstance(self.controller_key, str) or not self.controller_key:
            raise ValueError("controller_key must not be empty")
        if self.target_profile not in EMPTY_TORSO_PROFILES:
            raise ValueError(
                "target_profile must be one of: up, center, down"
            )

    def begin(
        self,
        action: ActionEnvelope,
        context: ActionExecutionContext,
    ) -> _EmptyTorsoState:
        del action
        try:
            actuator = context.services[self.actuator_service_key]
        except KeyError as exc:
            raise TorsoContractError(
                f"missing physical torso actuator: "
                f"{self.actuator_service_key!r}"
            ) from exc
        if (
            getattr(actuator, "api_contract_version", None)
            != PHYSICAL_TORSO_ACTUATOR_VERSION
            or not callable(getattr(actuator, "read", None))
            or not callable(getattr(actuator, "step", None))
            or not callable(getattr(actuator, "hold", None))
        ):
            raise TorsoContractError(
                "physical torso actuator contract/version mismatch"
            )
        sample = actuator.read()
        if not isinstance(sample, EmptyTorsoSample):
            raise TorsoContractError(
                "physical torso actuator returned an invalid sample"
            )
        return _EmptyTorsoState(
            actuator=actuator,
            target=EMPTY_TORSO_PROFILES[self.target_profile],
            profile_name=self.target_profile,
            sample=sample,
            limits=self.limits,
        )

    def tick(
        self,
        state: _EmptyTorsoState,
        context: ActionExecutionContext,
    ) -> ActionPluginTick:
        del context
        if state.terminal_status is not None:
            return ActionPluginTick(done=True, simulator_steps=0)
        if _close(
            state.sample.torso_qpos,
            state.target,
            self.limits.joint_goal_tolerance,
        ):
            state.terminal_status = "completed"
            return ActionPluginTick(done=True, simulator_steps=0)
        next_target = _bounded_step(
            state.sample.torso_qpos,
            state.target,
            self.limits.max_joint_delta_per_tick,
        )
        after = state.actuator.step(
            EmptyTorsoCommand(torso_target=next_target)
        )
        if not isinstance(after, EmptyTorsoSample):
            raise TorsoContractError(
                "physical torso actuator returned an invalid step sample"
            )
        state.forward_m += after.forward_delta_m
        state.left_m += after.left_delta_m
        state.yaw_deg += after.yaw_delta_deg
        unsafe = (
            after.joint_stalled
            or after.predicted_self_collision
            or math.hypot(state.forward_m, state.left_m)
            > self.limits.max_base_translation_m
            or abs(state.yaw_deg) > self.limits.max_base_rotation_deg
            or not _close(
                after.torso_qpos,
                next_target,
                self.limits.joint_tracking_tolerance,
            )
        )
        if unsafe:
            state.actuator.hold()
            state.terminal_status = "joint_stalled"
        elif _close(
            after.torso_qpos,
            state.target,
            self.limits.joint_goal_tolerance,
        ):
            state.actuator.hold()
            state.terminal_status = "completed"
        state.sample = after
        return ActionPluginTick(
            done=state.terminal_status is not None,
            simulator_steps=after.simulator_steps,
            public_feedback=_empty_feedback(state),
        )

    def finish(
        self,
        state: _EmptyTorsoState,
        context: ActionExecutionContext,
    ) -> ActionPluginResult:
        del context
        return ActionPluginResult(
            state.terminal_status or "joint_stalled",
            _empty_feedback(state),
        )

    def abort(
        self,
        state: _EmptyTorsoState,
        context: ActionExecutionContext,
        reason: str,
    ) -> ActionPluginResult:
        del context, reason
        state.actuator.hold()
        state.terminal_status = "joint_stalled"
        return ActionPluginResult("joint_stalled", _empty_feedback(state))


@dataclass(slots=True)
class EmptyTorsoGridPlugin:
    """Move one relative axis step on the physical 3x3 torso grid."""

    controller_key: str
    action_name: str
    limits: EmptyTorsoLimits = field(default_factory=EmptyTorsoLimits)
    actuator_service_key: str = PHYSICAL_TORSO_ACTUATOR_SERVICE

    def __post_init__(self) -> None:
        if not isinstance(self.controller_key, str) or not self.controller_key:
            raise ValueError("controller_key must not be empty")
        if self.action_name not in TORSO_GRID_ACTION_DELTAS:
            raise ValueError("action_name is not a torso-grid action")

    def begin(
        self,
        action: ActionEnvelope,
        context: ActionExecutionContext,
    ) -> _EmptyTorsoState:
        if action.action != self.action_name:
            raise TorsoContractError(
                f"{self.controller_key} received {action.action!r}"
            )
        actuator = _torso_actuator(context, self.actuator_service_key)
        sample = actuator.read()
        if not isinstance(sample, EmptyTorsoSample):
            raise TorsoContractError(
                "physical torso actuator returned an invalid sample"
            )
        height, pitch = nearest_torso_grid_cell(sample.torso_qpos)
        axis, delta = TORSO_GRID_ACTION_DELTAS[self.action_name]
        next_height, next_pitch = height, pitch
        if axis == "height":
            next_index = TORSO_HEIGHT_LEVELS.index(height) + delta
            in_range = 0 <= next_index < len(TORSO_HEIGHT_LEVELS)
            if in_range:
                next_height = TORSO_HEIGHT_LEVELS[next_index]
        else:
            next_index = TORSO_PITCH_LEVELS.index(pitch) + delta
            in_range = 0 <= next_index < len(TORSO_PITCH_LEVELS)
            if in_range:
                next_pitch = TORSO_PITCH_LEVELS[next_index]
        state = _EmptyTorsoState(
            actuator=actuator,
            target=EMPTY_TORSO_GRID[(next_height, next_pitch)],
            profile_name=f"{next_height}.{next_pitch}",
            sample=sample,
            limits=self.limits,
        )
        if not in_range:
            state.terminal_status = "limit_reached"
        return state

    def tick(
        self,
        state: _EmptyTorsoState,
        context: ActionExecutionContext,
    ) -> ActionPluginTick:
        delegate = EmptyTorsoLookPlugin(
            controller_key=self.controller_key,
            target_profile="center",
            limits=self.limits,
            actuator_service_key=self.actuator_service_key,
        )
        return delegate.tick(state, context)

    def finish(
        self,
        state: _EmptyTorsoState,
        context: ActionExecutionContext,
    ) -> ActionPluginResult:
        del context
        return ActionPluginResult(
            state.terminal_status or "joint_stalled",
            _empty_feedback(state),
        )

    def abort(
        self,
        state: _EmptyTorsoState,
        context: ActionExecutionContext,
        reason: str,
    ) -> ActionPluginResult:
        del context, reason
        state.actuator.hold()
        state.terminal_status = "joint_stalled"
        return ActionPluginResult("joint_stalled", _empty_feedback(state))


@dataclass(slots=True)
class _SymbolicTorsoState:
    target: tuple[float, ...]
    profile_name: str
    qpos: tuple[float, ...]
    terminal_status: str | None = None


@dataclass(slots=True)
class SymbolicTorsoGridPlugin:
    """Write one Action20 torso-grid endpoint without advancing physics."""

    controller_key: str
    action_name: str
    read_qpos: Callable[[], Sequence[float]]
    write_qpos: Callable[[Sequence[float]], Sequence[float]]

    def __post_init__(self) -> None:
        if not isinstance(self.controller_key, str) or not self.controller_key:
            raise ValueError("controller_key must not be empty")
        if self.action_name not in TORSO_GRID_ACTION_DELTAS:
            raise ValueError("action_name is not a torso-grid action")
        if not callable(self.read_qpos) or not callable(self.write_qpos):
            raise TypeError("symbolic torso callbacks must be callable")

    def begin(
        self,
        action: ActionEnvelope,
        context: ActionExecutionContext,
    ) -> _SymbolicTorsoState:
        del context
        if action.action != self.action_name:
            raise TorsoContractError(
                f"{self.controller_key} received {action.action!r}"
            )
        qpos = _vector(self.read_qpos(), 4, "torso_qpos")
        height, pitch = nearest_torso_grid_cell(qpos)
        axis, delta = TORSO_GRID_ACTION_DELTAS[self.action_name]
        levels = TORSO_HEIGHT_LEVELS if axis == "height" else TORSO_PITCH_LEVELS
        current = height if axis == "height" else pitch
        next_index = levels.index(current) + delta
        in_range = 0 <= next_index < len(levels)
        if in_range:
            if axis == "height":
                height = levels[next_index]
            else:
                pitch = levels[next_index]
        return _SymbolicTorsoState(
            target=EMPTY_TORSO_GRID[(height, pitch)],
            profile_name=f"{height}.{pitch}",
            qpos=qpos,
            terminal_status=None if in_range else "limit_reached",
        )

    def tick(
        self,
        state: _SymbolicTorsoState,
        context: ActionExecutionContext,
    ) -> ActionPluginTick:
        del context
        if state.terminal_status is None:
            state.qpos = _vector(
                self.write_qpos(state.target),
                4,
                "written torso_qpos",
            )
            state.terminal_status = (
                "completed"
                if _close(state.qpos, state.target, 1e-6)
                else "joint_stalled"
            )
        return ActionPluginTick(
            done=True,
            simulator_steps=0,
            public_feedback=_symbolic_feedback(state),
            capture_observation=True,
        )

    def finish(
        self,
        state: _SymbolicTorsoState,
        context: ActionExecutionContext,
    ) -> ActionPluginResult:
        del context
        return ActionPluginResult(
            state.terminal_status or "joint_stalled",
            _symbolic_feedback(state),
        )

    def abort(
        self,
        state: _SymbolicTorsoState,
        context: ActionExecutionContext,
        reason: str,
    ) -> ActionPluginResult:
        del context, reason
        state.terminal_status = "joint_stalled"
        return ActionPluginResult("joint_stalled", _symbolic_feedback(state))


@dataclass(slots=True)
class RejectCarryingTorsoGridPlugin:
    """Reject grid motion while a physical carry proxy is active."""

    controller_key: str
    action_name: str

    def begin(
        self,
        action: ActionEnvelope,
        context: ActionExecutionContext,
    ) -> str:
        del context
        if action.action != self.action_name:
            raise TorsoContractError(
                f"{self.controller_key} received {action.action!r}"
            )
        return "unsafe_while_carrying"

    def tick(
        self,
        state: str,
        context: ActionExecutionContext,
    ) -> ActionPluginTick:
        del state, context
        return ActionPluginTick(done=True, simulator_steps=0)

    def finish(
        self,
        state: str,
        context: ActionExecutionContext,
    ) -> ActionPluginResult:
        del context
        return ActionPluginResult(state, {"torso_grid_motion": False})

    def abort(
        self,
        state: str,
        context: ActionExecutionContext,
        reason: str,
    ) -> ActionPluginResult:
        del context, reason
        return ActionPluginResult(state, {"torso_grid_motion": False})


def nearest_torso_grid_cell(qpos: Sequence[float]) -> tuple[str, str]:
    """Return the deterministic closest grid cell in joint space."""

    actual = _vector(qpos, 4, "torso_qpos")
    return min(
        EMPTY_TORSO_GRID,
        key=lambda cell: (
            sum(
                (left - right) ** 2
                for left, right in zip(actual, EMPTY_TORSO_GRID[cell])
            ),
            cell,
        ),
    )


def r1pro_head_optical_pitch_deg(qpos: Sequence[float]) -> float:
    """Return the corrected onboard optical-pitch proxy in degrees."""

    q1, q2, q3, _q4 = _vector(qpos, 4, "torso_qpos")
    return -math.degrees(q1 + q2 - q3)


def r1pro_head_link_height_m(qpos: Sequence[float]) -> float:
    """Return the official-URDF ZED-link height for grid verification."""

    q1, q2, q3, _q4 = _vector(qpos, 4, "torso_qpos")
    theta2 = q1 + q2
    theta3 = q1 + q2 - q3
    return (
        0.34265
        + 0.4 * math.cos(q1)
        + 0.3 * math.cos(theta2)
        - 0.065511 * math.sin(theta3)
        + 0.5756 * math.cos(theta3)
    )


def _torso_actuator(
    context: ActionExecutionContext,
    service_key: str,
) -> PhysicalTorsoActuator:
    try:
        actuator = context.services[service_key]
    except KeyError as exc:
        raise TorsoContractError(
            f"missing physical torso actuator: {service_key!r}"
        ) from exc
    if (
        getattr(actuator, "api_contract_version", None)
        != PHYSICAL_TORSO_ACTUATOR_VERSION
        or not callable(getattr(actuator, "read", None))
        or not callable(getattr(actuator, "step", None))
        or not callable(getattr(actuator, "hold", None))
    ):
        raise TorsoContractError(
            "physical torso actuator contract/version mismatch"
        )
    return actuator


@dataclass(slots=True)
class _AdaptiveState:
    delegate: ActionPlugin = field(repr=False)
    delegate_state: Any = field(repr=False)
    mode: str


@dataclass(slots=True)
class AdaptivePhysicalActionPlugin:
    """Select empty/carry implementation from one public proxy bit only."""

    controller_key: str
    empty_plugin: ActionPlugin
    carrying_plugin: ActionPlugin
    # Logical inventory follows the EEF through the private post-step sync;
    # reusing the empty-hand torso motion here does not weaken physical carry.
    logical_carry_plugin: ActionPlugin | None = None
    proxy_service_key: str = PUBLIC_GRASP_PROXY_SERVICE

    def __post_init__(self) -> None:
        if not isinstance(self.controller_key, str) or not self.controller_key:
            raise ValueError("controller_key must not be empty")
        plugins = [
            ("empty_plugin", self.empty_plugin),
            ("carrying_plugin", self.carrying_plugin),
        ]
        if self.logical_carry_plugin is not None:
            plugins.append(("logical_carry_plugin", self.logical_carry_plugin))
        for name, plugin in plugins:
            if not all(
                callable(getattr(plugin, method, None))
                for method in ("begin", "tick", "finish", "abort")
            ):
                raise TypeError(f"{name} does not satisfy ActionPlugin")

    def begin(
        self,
        action: ActionEnvelope,
        context: ActionExecutionContext,
    ) -> _AdaptiveState:
        try:
            source = context.services[self.proxy_service_key]
        except KeyError as exc:
            raise TorsoContractError(
                f"missing public grasp proxy source: "
                f"{self.proxy_service_key!r}"
            ) from exc
        read = getattr(source, "grasp_proxy_likely", None)
        if not callable(read):
            raise TorsoContractError(
                "public grasp proxy source is invalid"
            )
        likely = read()
        if not isinstance(likely, bool):
            raise TorsoContractError(
                "public grasp proxy source must return bool"
            )
        carry_mode = "physical" if likely else "empty"
        read_mode = getattr(source, "carry_mode", None)
        if callable(read_mode):
            carry_mode = read_mode()
            if carry_mode not in {"empty", "physical", "logical_inventory"}:
                raise TorsoContractError(
                    "public grasp proxy carry mode is invalid"
                )
            if (carry_mode == "empty") == likely:
                raise TorsoContractError(
                    "public grasp proxy carry mode conflicts with likely bit"
                )
        if (
            carry_mode == "logical_inventory"
            and self.logical_carry_plugin is not None
        ):
            delegate = self.logical_carry_plugin
        else:
            delegate = self.carrying_plugin if likely else self.empty_plugin
        return _AdaptiveState(
            delegate=delegate,
            delegate_state=delegate.begin(action, context),
            mode="carrying" if likely else "empty",
        )

    def tick(
        self,
        state: _AdaptiveState,
        context: ActionExecutionContext,
    ) -> ActionPluginTick:
        result = state.delegate.tick(state.delegate_state, context)
        if not isinstance(result, ActionPluginTick):
            raise TorsoContractError(
                "adaptive torso delegate returned invalid tick"
            )
        return result

    def finish(
        self,
        state: _AdaptiveState,
        context: ActionExecutionContext,
    ) -> ActionPluginResult:
        result = state.delegate.finish(state.delegate_state, context)
        if not isinstance(result, ActionPluginResult):
            raise TorsoContractError(
                "adaptive torso delegate returned invalid result"
            )
        feedback = dict(result.public_feedback)
        feedback["look_mode"] = state.mode
        return ActionPluginResult(result.status, feedback)

    def abort(
        self,
        state: _AdaptiveState,
        context: ActionExecutionContext,
        reason: str,
    ) -> ActionPluginResult:
        result = state.delegate.abort(
            state.delegate_state,
            context,
            reason,
        )
        if not isinstance(result, ActionPluginResult):
            raise TorsoContractError(
                "adaptive torso delegate returned invalid abort result"
            )
        feedback = dict(result.public_feedback)
        feedback["look_mode"] = state.mode
        return ActionPluginResult(result.status, feedback)


def _empty_feedback(state: _EmptyTorsoState) -> dict[str, Any]:
    height, pitch = nearest_torso_grid_cell(state.sample.torso_qpos)
    optical_pitch = r1pro_head_optical_pitch_deg(state.sample.torso_qpos)
    center_pitch = r1pro_head_optical_pitch_deg(
        EMPTY_TORSO_GRID[(height, "center")]
    )
    link_height = r1pro_head_link_height_m(state.sample.torso_qpos)
    neutral_height = r1pro_head_link_height_m(
        EMPTY_TORSO_GRID[("neutral", pitch)]
    )
    return {
        "profile": state.profile_name,
        "physical_camera_motion": True,
        "head_optical_pitch_delta_deg_from_center": optical_pitch - center_pitch,
        "head_link_height_delta_m_from_neutral": link_height - neutral_height,
        "base_stable": (
            math.hypot(state.forward_m, state.left_m)
            <= state.limits.max_base_translation_m
            and abs(state.yaw_deg)
            <= state.limits.max_base_rotation_deg
        ),
    }


def _symbolic_feedback(state: _SymbolicTorsoState) -> dict[str, Any]:
    height, pitch = nearest_torso_grid_cell(state.qpos)
    optical_pitch = r1pro_head_optical_pitch_deg(state.qpos)
    center_pitch = r1pro_head_optical_pitch_deg(
        EMPTY_TORSO_GRID[(height, "center")]
    )
    link_height = r1pro_head_link_height_m(state.qpos)
    neutral_height = r1pro_head_link_height_m(
        EMPTY_TORSO_GRID[("neutral", pitch)]
    )
    return {
        "profile": state.profile_name,
        "physical_camera_motion": False,
        "symbolic_endpoint": True,
        "head_optical_pitch_delta_deg_from_center": optical_pitch - center_pitch,
        "head_link_height_delta_m_from_neutral": link_height - neutral_height,
        "base_stable": True,
    }


def _bounded_step(
    current: Sequence[float],
    target: Sequence[float],
    maximum_delta: float,
) -> tuple[float, ...]:
    return tuple(
        before + max(-maximum_delta, min(maximum_delta, after - before))
        for before, after in zip(current, target)
    )


def _close(
    actual: Sequence[float],
    expected: Sequence[float],
    tolerance: float,
) -> bool:
    return all(
        abs(left - right) <= tolerance
        for left, right in zip(actual, expected)
    )


def _vector(
    value: Sequence[float],
    length: int,
    field: str,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must contain {length} finite numbers")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError(
            f"{field} must contain {length} finite numbers"
        ) from exc
    if len(items) != length:
        raise ValueError(f"{field} must contain {length} finite numbers")
    return tuple(
        _finite(item, f"{field}[{index}]")
        for index, item in enumerate(items)
    )


def _finite(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result
