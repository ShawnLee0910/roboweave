"""Simulator-independent contracts for safe actions while carrying.

The plugins in this module deliberately operate on relative odometry, public
proprioception, and a calibrated grasp proxy.  They never accept object poses,
object identities, teleport targets, maps, or task truth.  A private evaluator
may capture endpoint evidence through :class:`CarryPrivateAuditSink`, but that
API is write-only from the controller's point of view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

from behavior_hl.embodied.policy.protocol import ActionEnvelope
from behavior_hl.embodied.runtime.dispatch import (
    ActionExecutionContext,
    ActionPluginResult,
    ActionPluginTick,
)


class CarryContractError(RuntimeError):
    """Raised when a carry controller or service violates the frozen contract."""


class CarryMotionKind(str, Enum):
    """Kinds of relative base motion supported while carrying."""

    TRANSLATION = "translation"
    ROTATION = "rotation"


@dataclass(frozen=True, slots=True)
class CarrySafetyLimits:
    """Frozen controller limits, independent of simulator object state."""

    neutral_torso: tuple[float, ...] = (1.025, -1.45, -0.47, 0.0)
    downward_torso: tuple[float, ...] = (1.60, -2.50, -0.94, 0.0)
    torso_segment_tolerance: float = 0.04
    torso_goal_tolerance: float = 0.02
    # A velocity-driven base macro can briefly load the floating R1Pro torso
    # while the robot remains grasp-stable, contact-free, and well inside the
    # calibrated torso segment.  Keep that bounded in-motion transient
    # separate from the stricter action-boundary / settle tolerance.
    torso_motion_tracking_tolerance: float = 0.04
    max_torso_motion_delta_per_tick: float = 0.04
    max_torso_delta_per_tick: float = 0.025
    max_arm_delta_per_tick: float = 0.08
    arm_tracking_tolerance: float = 0.04
    max_look_base_translation_m: float = 0.01
    max_look_base_rotation_deg: float = 0.5
    max_base_z_drift_m: float = 0.03
    max_abs_roll_pitch_deg: float = 5.0
    max_translation_per_tick_m: float = 0.10
    max_rotation_per_tick_deg: float = 12.0
    # Physical manipulation lanes keep the aggregate robot joint-stall signal
    # terminal.  A marker-bound Symbolic lane may disable only this aggregate
    # bit because its arm / torso are deliberately frozen and are not part of
    # the task score.  Base contact, grasp continuity, discontinuous odometry,
    # height drift, and roll / pitch remain independent terminal gates.
    joint_stall_is_terminal: bool = True

    def __post_init__(self) -> None:
        neutral = _vector(self.neutral_torso, 4, "neutral_torso")
        downward = _vector(self.downward_torso, 4, "downward_torso")
        if _norm(tuple(b - a for a, b in zip(neutral, downward))) <= 0.0:
            raise ValueError("neutral_torso and downward_torso must differ")
        object.__setattr__(self, "neutral_torso", neutral)
        object.__setattr__(self, "downward_torso", downward)
        for name in (
            "torso_segment_tolerance",
            "torso_goal_tolerance",
            "torso_motion_tracking_tolerance",
            "max_torso_motion_delta_per_tick",
            "max_torso_delta_per_tick",
            "max_arm_delta_per_tick",
            "arm_tracking_tolerance",
            "max_look_base_translation_m",
            "max_look_base_rotation_deg",
            "max_base_z_drift_m",
            "max_abs_roll_pitch_deg",
            "max_translation_per_tick_m",
            "max_rotation_per_tick_deg",
        ):
            value = _finite(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.torso_motion_tracking_tolerance < self.torso_goal_tolerance:
            raise ValueError(
                "torso_motion_tracking_tolerance must be at least "
                "torso_goal_tolerance"
            )
        if self.torso_motion_tracking_tolerance > self.torso_segment_tolerance:
            raise ValueError(
                "torso_motion_tracking_tolerance must not exceed "
                "torso_segment_tolerance"
            )
        if (
            self.max_torso_motion_delta_per_tick
            > self.torso_motion_tracking_tolerance
        ):
            raise ValueError(
                "max_torso_motion_delta_per_tick must not exceed "
                "torso_motion_tracking_tolerance"
            )
        if not isinstance(self.joint_stall_is_terminal, bool):
            raise ValueError("joint_stall_is_terminal must be boolean")


@dataclass(frozen=True, slots=True)
class CarrySample:
    """One controller sample with no object identity or world-space target."""

    active_hand: str | None
    grasp_proxy_likely: bool
    torso_qpos: tuple[float, ...]
    active_arm_qpos: tuple[float, ...]
    forward_delta_m: float = 0.0
    left_delta_m: float = 0.0
    yaw_delta_deg: float = 0.0
    base_z_drift_m: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    base_contact: bool = False
    predicted_self_collision: bool = False
    joint_stalled: bool = False

    def __post_init__(self) -> None:
        if self.active_hand not in {None, "left", "right"}:
            raise ValueError("active_hand must be left, right, or None")
        if not isinstance(self.grasp_proxy_likely, bool):
            raise ValueError("grasp_proxy_likely must be boolean")
        object.__setattr__(
            self,
            "torso_qpos",
            _vector(self.torso_qpos, 4, "torso_qpos"),
        )
        object.__setattr__(
            self,
            "active_arm_qpos",
            _vector(self.active_arm_qpos, 7, "active_arm_qpos"),
        )
        for name in (
            "forward_delta_m",
            "left_delta_m",
            "yaw_delta_deg",
            "base_z_drift_m",
            "roll_deg",
            "pitch_deg",
        ):
            _finite(getattr(self, name), name)
        for name in (
            "base_contact",
            "predicted_self_collision",
            "joint_stalled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")


@dataclass(frozen=True, slots=True)
class CarryCommand:
    """A physical joint/velocity command; pose and teleport fields do not exist."""

    active_hand: str
    base_velocity: tuple[float, ...]
    torso_target: tuple[float, ...]
    active_arm_target: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.active_hand not in {"left", "right"}:
            raise ValueError("active_hand must be left or right")
        base = _vector(self.base_velocity, 3, "base_velocity")
        if any(abs(value) > 1.0 for value in base):
            raise ValueError("base_velocity exceeds normalized controller limits")
        object.__setattr__(self, "base_velocity", base)
        object.__setattr__(
            self,
            "torso_target",
            _vector(self.torso_target, 4, "torso_target"),
        )
        object.__setattr__(
            self,
            "active_arm_target",
            _vector(self.active_arm_target, 7, "active_arm_target"),
        )


class CarryActuator(Protocol):
    """Public-proprioception actuator used by carry-safe plugins."""

    def read(self) -> CarrySample:
        """Read the latest public proxy and executor safety signals."""

    def step(self, command: CarryCommand) -> CarrySample:
        """Apply one finite physical command and advance one simulator step."""

    def hold(self) -> None:
        """Stop the base and hold the latest safe joint targets."""


class ActiveArmCompensator(Protocol):
    """URDF/proprioception-only torso compensation for the active arm."""

    def target_for_torso_step(
        self,
        *,
        hand: str,
        torso_before: tuple[float, ...],
        torso_after: tuple[float, ...],
        arm_before: tuple[float, ...],
    ) -> Sequence[float]:
        """Return a bounded arm target without object pose or held truth."""


class CarryPrivateAuditSink(Protocol):
    """Write-only evaluator hook; its return value is always ignored."""

    def capture_endpoint(
        self,
        *,
        stage: str,
        episode_id: str,
        action_index: int,
    ) -> None:
        """Privately record held/contact/swing truth for later audit."""


@dataclass(slots=True)
class _CarryState:
    actuator: CarryActuator = field(repr=False)
    compensator: ActiveArmCompensator | None = field(default=None, repr=False)
    audit: CarryPrivateAuditSink | None = field(default=None, repr=False)
    sample: CarrySample | None = None
    hand: str | None = None
    terminal_status: str | None = None
    proxy_continuous: bool = True
    carry_pose_preserved: bool = True
    forward_m: float = 0.0
    left_m: float = 0.0
    yaw_deg: float = 0.0
    steps: int = 0
    endpoint_audited: bool = False
    # Evaluator-only terminal attribution. These fields are deliberately not
    # included by _public_feedback(), so a policy never receives the private
    # safety branch or numeric margins as executor text.
    failure_reason_private: str | None = None
    failure_snapshot_private: dict[str, Any] = field(default_factory=dict)
    tick_trace_private: list[dict[str, Any]] = field(default_factory=list)


def carry_torso_target(
    fraction: float,
    *,
    limits: CarrySafetyLimits,
) -> tuple[float, ...]:
    """Interpolate only inside the validated neutral-to-downward interval."""

    value = _finite(fraction, "fraction")
    if not 0.0 <= value <= 1.0:
        raise ValueError("carry torso fraction must be in [0, 1]")
    return tuple(
        neutral + value * (downward - neutral)
        for neutral, downward in zip(
            limits.neutral_torso,
            limits.downward_torso,
        )
    )


@dataclass(slots=True)
class CarrySafeLookPlugin:
    """Bounded torso look with synchronous active-arm compensation."""

    controller_key: str
    target_fraction: float
    limits: CarrySafetyLimits = field(default_factory=CarrySafetyLimits)
    actuator_service_key: str = "carry_actuator"
    compensator_service_key: str = "carry_arm_compensator"
    audit_service_key: str = "carry_private_audit"

    def __post_init__(self) -> None:
        if not self.controller_key:
            raise ValueError("controller_key must not be empty")
        carry_torso_target(self.target_fraction, limits=self.limits)

    def begin(
        self,
        action: ActionEnvelope,
        context: ActionExecutionContext,
    ) -> _CarryState:
        del action
        actuator = _service(context, self.actuator_service_key, "carry actuator")
        try:
            compensator = context.services[self.compensator_service_key]
        except KeyError as exc:
            raise CarryContractError(
                "missing active-arm compensator service: "
                f"{self.compensator_service_key!r}"
            ) from exc
        if not callable(getattr(compensator, "target_for_torso_step", None)):
            raise CarryContractError("active-arm compensator service is invalid")
        audit = context.services.get(self.audit_service_key)
        state = _CarryState(
            actuator=actuator,
            compensator=compensator,
            audit=audit,
        )
        state.sample = _read_sample(actuator)
        state.hand = state.sample.active_hand
        _audit(state, context, "start")
        if not _valid_carry_start(state.sample, self.limits):
            state.proxy_continuous = state.sample.grasp_proxy_likely
            state.carry_pose_preserved = _torso_fraction(
                state.sample.torso_qpos,
                self.limits,
            ) is not None
            _record_failure(
                state,
                _carry_start_failure_reason(state.sample, self.limits),
                state.sample,
                self.limits,
            )
            state.terminal_status = "unsafe_while_carrying"
        return state

    def tick(
        self,
        state: _CarryState,
        context: ActionExecutionContext,
    ) -> ActionPluginTick:
        del context
        if state.terminal_status is not None:
            return ActionPluginTick(done=True, simulator_steps=0)
        assert state.sample is not None and state.hand is not None
        target = carry_torso_target(self.target_fraction, limits=self.limits)
        if _close(state.sample.torso_qpos, target, self.limits.torso_goal_tolerance):
            state.terminal_status = "completed"
            return ActionPluginTick(done=True, simulator_steps=0)

        next_torso = _bounded_step(
            state.sample.torso_qpos,
            target,
            self.limits.max_torso_delta_per_tick,
        )
        assert state.compensator is not None
        proposed_arm = _vector(
            state.compensator.target_for_torso_step(
                hand=state.hand,
                torso_before=state.sample.torso_qpos,
                torso_after=next_torso,
                arm_before=state.sample.active_arm_qpos,
            ),
            7,
            "compensated active arm target",
        )
        if not _bounded_delta(
            state.sample.active_arm_qpos,
            proposed_arm,
            self.limits.max_arm_delta_per_tick,
        ):
            state.actuator.hold()
            state.carry_pose_preserved = False
            _record_failure(
                state,
                "arm_compensation_delta_too_large",
                state.sample,
                self.limits,
                safe_arm_target=proposed_arm,
            )
            state.terminal_status = "unsafe_while_carrying"
            return ActionPluginTick(done=True, simulator_steps=0)

        after = _step(
            state.actuator,
            CarryCommand(
                active_hand=state.hand,
                base_velocity=(0.0, 0.0, 0.0),
                torso_target=next_torso,
                active_arm_target=proposed_arm,
            ),
        )
        state.steps += 1
        _accumulate(state, after)
        failure = _common_failure(state, after, self.limits)
        if failure is None and (
            math.hypot(state.forward_m, state.left_m)
            > self.limits.max_look_base_translation_m
            or abs(state.yaw_deg) > self.limits.max_look_base_rotation_deg
        ):
            failure = "unsafe_while_carrying"
        if failure is None and not _close(
            after.active_arm_qpos,
            proposed_arm,
            self.limits.arm_tracking_tolerance,
        ):
            state.carry_pose_preserved = False
            _record_failure(
                state,
                "arm_tracking_error",
                after,
                self.limits,
                safe_arm_target=proposed_arm,
                previous_torso=state.sample.torso_qpos,
            )
            failure = "unsafe_while_carrying"
        if failure is None and not _close(
            after.torso_qpos,
            next_torso,
            self.limits.torso_goal_tolerance,
        ):
            state.carry_pose_preserved = False
            _record_failure(
                state,
                "torso_tracking_error",
                after,
                self.limits,
                safe_arm_target=proposed_arm,
                previous_torso=next_torso,
            )
            failure = "unsafe_while_carrying"
        if failure is not None:
            state.actuator.hold()
            state.terminal_status = failure
        elif _close(after.torso_qpos, target, self.limits.torso_goal_tolerance):
            state.terminal_status = "completed"
        state.sample = after
        return ActionPluginTick(
            done=state.terminal_status is not None,
            simulator_steps=1,
            public_feedback=_public_feedback(state),
        )

    def finish(
        self,
        state: _CarryState,
        context: ActionExecutionContext,
    ) -> ActionPluginResult:
        _audit(state, context, "end")
        return ActionPluginResult(
            state.terminal_status or "joint_stalled",
            _public_feedback(
                state,
                torso_interval="neutral_to_downward",
            ),
        )

    def abort(
        self,
        state: _CarryState,
        context: ActionExecutionContext,
        reason: str,
    ) -> ActionPluginResult:
        del reason
        state.actuator.hold()
        if state.terminal_status is None:
            state.terminal_status = "joint_stalled"
        _audit(state, context, "end")
        return ActionPluginResult(
            state.terminal_status,
            _public_feedback(
                state,
                torso_interval="neutral_to_downward",
            ),
        )


@dataclass(slots=True)
class CarryingBaseMotionPlugin:
    """Relative velocity motion that preserves a calibrated safe-carry pose."""

    controller_key: str
    base_velocity: tuple[float, float, float]
    target_odometry: tuple[float, float, float]
    safe_arm_targets: Mapping[str, Sequence[float]]
    safe_torso_targets: Mapping[str, Sequence[float]] = field(
        default_factory=dict
    )
    limits: CarrySafetyLimits = field(default_factory=CarrySafetyLimits)
    actuator_service_key: str = "carry_actuator"
    audit_service_key: str = "carry_private_audit"

    def __post_init__(self) -> None:
        if not self.controller_key:
            raise ValueError("controller_key must not be empty")
        self.base_velocity = _vector(self.base_velocity, 3, "base_velocity")
        if any(abs(value) > 1.0 for value in self.base_velocity):
            raise ValueError("base_velocity exceeds normalized controller limits")
        self.target_odometry = _vector(
            self.target_odometry,
            3,
            "target_odometry",
        )
        nonzero_translation = any(
            abs(value) > 0.0 for value in self.target_odometry[:2]
        )
        nonzero_rotation = abs(self.target_odometry[2]) > 0.0
        if nonzero_translation == nonzero_rotation:
            raise ValueError(
                "target_odometry must request either translation or rotation"
            )
        command_projection = sum(
            command * target
            for command, target in zip(
                self.base_velocity,
                self.target_odometry,
            )
        )
        if command_projection <= 0.0:
            raise ValueError(
                "base_velocity must point toward target_odometry"
            )
        hands = set(self.safe_arm_targets)
        if not hands or not hands.issubset({"left", "right"}):
            raise ValueError(
                "safe_arm_targets must define at least one calibrated hand"
            )
        self.safe_arm_targets = {
            hand: _vector(target, 7, f"safe_arm_targets[{hand}]")
            for hand, target in self.safe_arm_targets.items()
        }
        torso_hands = set(self.safe_torso_targets)
        if not torso_hands.issubset(hands):
            raise ValueError(
                "safe_torso_targets must be a subset of safe_arm_targets"
            )
        self.safe_torso_targets = {
            hand: _vector(target, 4, f"safe_torso_targets[{hand}]")
            for hand, target in self.safe_torso_targets.items()
        }

    def rebase_after_grasp(self, sample: CarrySample) -> Mapping[str, Any]:
        """Freeze one measured post-grasp carry pose without advancing physics.

        Symbolic GRASP is allowed to move the R1Pro arm and torso before it
        establishes the held-object constraint.  The pre-grasp calibration is
        therefore not a valid tracking target for subsequent base motion.  A
        resident runner calls this method exactly after a successful grasp,
        using the carry actuator's current sample.  Only robot proprioception
        and executor safety bits are admitted; object pose and task truth are
        neither accepted nor stored here.
        """

        if not isinstance(sample, CarrySample):
            raise CarryContractError("post-grasp carry reference is invalid")
        hand = sample.active_hand
        if hand not in self.safe_arm_targets:
            raise CarryContractError(
                "post-grasp active hand is not calibrated"
            )
        if not sample.grasp_proxy_likely:
            raise CarryContractError(
                "post-grasp carry proxy is not likely"
            )
        if sample.predicted_self_collision:
            raise CarryContractError(
                "post-grasp pose predicts self collision"
            )
        if sample.joint_stalled and self.limits.joint_stall_is_terminal:
            raise CarryContractError("post-grasp joints are stalled")
        if sample.base_contact:
            raise CarryContractError("post-grasp base is in contact")
        if (
            abs(sample.base_z_drift_m) > self.limits.max_base_z_drift_m
            or abs(sample.roll_deg) > self.limits.max_abs_roll_pitch_deg
            or abs(sample.pitch_deg) > self.limits.max_abs_roll_pitch_deg
        ):
            raise CarryContractError("post-grasp base stability is unsafe")

        self.safe_arm_targets = {
            **self.safe_arm_targets,
            hand: tuple(sample.active_arm_qpos),
        }
        self.safe_torso_targets = {
            **self.safe_torso_targets,
            hand: tuple(sample.torso_qpos),
        }
        return {
            "status": "rebased",
            "active_hand": hand,
            "source": "post_grasp_robot_proprioception",
            "grasp_proxy_likely": True,
            "predicted_self_collision": False,
            "joint_stalled": False,
            "base_contact": False,
            "torso_qpos": list(sample.torso_qpos),
            "active_arm_qpos": list(sample.active_arm_qpos),
        }

    @property
    def motion_kind(self) -> CarryMotionKind:
        if abs(self.target_odometry[2]) > 0.0:
            return CarryMotionKind.ROTATION
        return CarryMotionKind.TRANSLATION

    def begin(
        self,
        action: ActionEnvelope,
        context: ActionExecutionContext,
    ) -> _CarryState:
        del action
        actuator = _service(context, self.actuator_service_key, "carry actuator")
        audit = context.services.get(self.audit_service_key)
        state = _CarryState(actuator=actuator, audit=audit)
        state.sample = _read_sample(actuator)
        state.hand = state.sample.active_hand
        _audit(state, context, "start")
        if state.hand not in self.safe_arm_targets:
            _record_failure(
                state,
                "active_hand_not_calibrated",
                state.sample,
                self.limits,
            )
            state.terminal_status = "unsafe_while_carrying"
            return state
        safe_torso_target = self.safe_torso_targets.get(state.hand)
        if not _valid_carry_start(
            state.sample,
            self.limits,
            safe_torso_target=safe_torso_target,
        ):
            state.proxy_continuous = state.sample.grasp_proxy_likely
            _record_failure(
                state,
                _carry_start_failure_reason(
                    state.sample,
                    self.limits,
                    safe_torso_target=safe_torso_target,
                ),
                state.sample,
                self.limits,
                safe_arm_target=self.safe_arm_targets[state.hand],
                safe_torso_target=safe_torso_target,
            )
            state.terminal_status = "unsafe_while_carrying"
        elif not _close(
            state.sample.active_arm_qpos,
            self.safe_arm_targets[state.hand],
            self.limits.arm_tracking_tolerance,
        ):
            state.carry_pose_preserved = False
            _record_failure(
                state,
                "arm_tracking_error_at_start",
                state.sample,
                self.limits,
                safe_arm_target=self.safe_arm_targets[state.hand],
            )
            state.terminal_status = "unsafe_while_carrying"
        return state

    def tick(
        self,
        state: _CarryState,
        context: ActionExecutionContext,
    ) -> ActionPluginTick:
        del context
        if state.terminal_status is not None:
            return ActionPluginTick(done=True, simulator_steps=0)
        assert state.sample is not None and state.hand is not None
        safe_torso_target = self.safe_torso_targets.get(state.hand)
        commanded_torso_target = (
            state.sample.torso_qpos
            if safe_torso_target is None
            else safe_torso_target
        )
        after = _step(
            state.actuator,
            CarryCommand(
                active_hand=state.hand,
                base_velocity=self.base_velocity,
                torso_target=commanded_torso_target,
                active_arm_target=self.safe_arm_targets[state.hand],
            ),
        )
        state.steps += 1
        _accumulate(state, after)
        torso_delta = max(
            abs(actual - previous)
            for actual, previous in zip(
                after.torso_qpos,
                state.sample.torso_qpos,
            )
        )
        torso_reference_error = (
            None
            if safe_torso_target is None
            else max(
                abs(actual - expected)
                for actual, expected in zip(
                    after.torso_qpos,
                    safe_torso_target,
                )
            )
        )
        if torso_delta > self.limits.max_torso_motion_delta_per_tick:
            state.carry_pose_preserved = False
            _record_failure(
                state,
                "torso_motion_delta_too_large",
                after,
                self.limits,
                safe_arm_target=self.safe_arm_targets[state.hand],
                previous_torso=state.sample.torso_qpos,
                safe_torso_target=safe_torso_target,
                torso_tracking_tolerance=(
                    self.limits.max_torso_motion_delta_per_tick
                ),
            )
            failure = "unsafe_while_carrying"
        else:
            failure = _common_failure(
                state,
                after,
                self.limits,
                safe_torso_target=safe_torso_target,
                torso_reference_tolerance=(
                    self.limits.torso_motion_tracking_tolerance
                ),
            )
        if failure is None and not _close(
            after.active_arm_qpos,
            self.safe_arm_targets[state.hand],
            self.limits.arm_tracking_tolerance,
        ):
            state.carry_pose_preserved = False
            _record_failure(
                state,
                "arm_tracking_error",
                after,
                self.limits,
                safe_arm_target=self.safe_arm_targets[state.hand],
                previous_torso=state.sample.torso_qpos,
            )
            failure = "unsafe_while_carrying"
        if failure is None and not _close(
            after.torso_qpos,
            commanded_torso_target,
            self.limits.torso_motion_tracking_tolerance,
        ):
            state.carry_pose_preserved = False
            _record_failure(
                state,
                "torso_tracking_error",
                after,
                self.limits,
                safe_arm_target=self.safe_arm_targets[state.hand],
                previous_torso=commanded_torso_target,
                safe_torso_target=safe_torso_target,
                torso_tracking_tolerance=(
                    self.limits.torso_motion_tracking_tolerance
                ),
            )
            failure = "unsafe_while_carrying"
        if failure is None and _increment_too_large(after, self.limits):
            # Fail closed on discontinuous odometry.  Since CarryCommand has no
            # pose field, the plugin cannot cause or repair this via teleport.
            _record_failure(
                state,
                "odometry_increment_too_large",
                after,
                self.limits,
                safe_arm_target=self.safe_arm_targets[state.hand],
                previous_torso=state.sample.torso_qpos,
            )
            failure = "unsafe_while_carrying"
        if failure is None and after.base_contact:
            _record_failure(
                state,
                "base_contact",
                after,
                self.limits,
                safe_arm_target=self.safe_arm_targets[state.hand],
                previous_torso=state.sample.torso_qpos,
            )
            failure = "blocked"
        state.tick_trace_private.append(
            {
                "step": int(state.steps),
                "torso_qpos": list(after.torso_qpos),
                "safe_torso_target": (
                    None
                    if safe_torso_target is None
                    else list(safe_torso_target)
                ),
                "torso_reference_max_abs_error": torso_reference_error,
                "torso_reference_tolerance": float(
                    self.limits.torso_motion_tracking_tolerance
                ),
                "torso_delta_max_abs": float(torso_delta),
                "torso_delta_limit": float(
                    self.limits.max_torso_motion_delta_per_tick
                ),
                "grasp_proxy_likely": bool(after.grasp_proxy_likely),
                "predicted_self_collision": bool(
                    after.predicted_self_collision
                ),
                "joint_stalled": bool(after.joint_stalled),
                "base_contact": bool(after.base_contact),
                "base_z_drift_m": float(after.base_z_drift_m),
                "roll_deg": float(after.roll_deg),
                "pitch_deg": float(after.pitch_deg),
                "failure_reason": state.failure_reason_private,
            }
        )
        if failure is not None:
            state.actuator.hold()
            state.terminal_status = failure
        elif _target_reached(
            (state.forward_m, state.left_m, state.yaw_deg),
            self.target_odometry,
        ):
            state.actuator.hold()
            state.terminal_status = "completed"
        state.sample = after
        return ActionPluginTick(
            done=state.terminal_status is not None,
            simulator_steps=1,
            public_feedback=_public_feedback(
                state,
                motion_kind=self.motion_kind.value,
                include_odometry=True,
            ),
        )

    def finish(
        self,
        state: _CarryState,
        context: ActionExecutionContext,
    ) -> ActionPluginResult:
        _audit(state, context, "end")
        return ActionPluginResult(
            state.terminal_status or "motion_partial",
            _public_feedback(
                state,
                motion_kind=self.motion_kind.value,
                include_odometry=True,
            ),
        )

    def abort(
        self,
        state: _CarryState,
        context: ActionExecutionContext,
        reason: str,
    ) -> ActionPluginResult:
        del reason
        state.actuator.hold()
        if state.terminal_status is None:
            moved = math.hypot(state.forward_m, state.left_m) > 0.0 or abs(
                state.yaw_deg
            ) > 0.0
            state.terminal_status = "motion_partial" if moved else "blocked"
        _audit(state, context, "end")
        return ActionPluginResult(
            state.terminal_status,
            _public_feedback(
                state,
                motion_kind=self.motion_kind.value,
                include_odometry=True,
            ),
        )


def _service(
    context: ActionExecutionContext,
    key: str,
    label: str,
) -> Any:
    try:
        service = context.services[key]
    except KeyError as exc:
        raise CarryContractError(f"missing {label} service: {key!r}") from exc
    if not callable(getattr(service, "read", None)) or not callable(
        getattr(service, "step", None)
    ) or not callable(getattr(service, "hold", None)):
        raise CarryContractError(f"{label} service is invalid")
    return service


def _read_sample(actuator: CarryActuator) -> CarrySample:
    sample = actuator.read()
    if not isinstance(sample, CarrySample):
        raise CarryContractError("carry actuator returned an invalid sample")
    return sample


def _step(actuator: CarryActuator, command: CarryCommand) -> CarrySample:
    sample = actuator.step(command)
    if not isinstance(sample, CarrySample):
        raise CarryContractError("carry actuator returned an invalid step sample")
    return sample


def _valid_carry_start(
    sample: CarrySample,
    limits: CarrySafetyLimits,
    *,
    safe_torso_target: Sequence[float] | None = None,
) -> bool:
    torso_valid = (
        _torso_fraction(sample.torso_qpos, limits) is not None
        if safe_torso_target is None
        else _close(
            sample.torso_qpos,
            safe_torso_target,
            limits.torso_goal_tolerance,
        )
    )
    return (
        sample.active_hand in {"left", "right"}
        and sample.grasp_proxy_likely
        and torso_valid
        and not sample.predicted_self_collision
        and (
            not sample.joint_stalled
            or not limits.joint_stall_is_terminal
        )
    )


def _carry_start_failure_reason(
    sample: CarrySample,
    limits: CarrySafetyLimits,
    *,
    safe_torso_target: Sequence[float] | None = None,
) -> str:
    if sample.active_hand not in {"left", "right"}:
        return "invalid_active_hand"
    if not sample.grasp_proxy_likely:
        return "grasp_proxy_lost"
    if sample.predicted_self_collision:
        return "predicted_self_collision"
    if sample.joint_stalled and limits.joint_stall_is_terminal:
        return "joint_stalled"
    torso_valid = (
        _torso_fraction(sample.torso_qpos, limits) is not None
        if safe_torso_target is None
        else _close(
            sample.torso_qpos,
            safe_torso_target,
            limits.torso_goal_tolerance,
        )
    )
    if not torso_valid:
        if safe_torso_target is not None:
            return "torso_outside_post_grasp_reference"
        return "torso_outside_calibrated_segment"
    return "invalid_carry_start"


def _common_failure(
    state: _CarryState,
    sample: CarrySample,
    limits: CarrySafetyLimits,
    *,
    safe_torso_target: Sequence[float] | None = None,
    torso_reference_tolerance: float | None = None,
) -> str | None:
    if sample.active_hand != state.hand or not sample.grasp_proxy_likely:
        state.proxy_continuous = False
        _record_failure(
            state,
            (
                "active_hand_changed"
                if sample.active_hand != state.hand
                else "grasp_proxy_lost"
            ),
            sample,
            limits,
        )
        return "unsafe_while_carrying"
    if sample.predicted_self_collision:
        state.carry_pose_preserved = False
        _record_failure(
            state,
            "predicted_self_collision",
            sample,
            limits,
        )
        return "unsafe_while_carrying"
    if sample.joint_stalled and limits.joint_stall_is_terminal:
        state.carry_pose_preserved = False
        _record_failure(state, "joint_stalled", sample, limits)
        return "joint_stalled"
    if (
        abs(sample.base_z_drift_m) > limits.max_base_z_drift_m
        or abs(sample.roll_deg) > limits.max_abs_roll_pitch_deg
        or abs(sample.pitch_deg) > limits.max_abs_roll_pitch_deg
    ):
        state.carry_pose_preserved = False
        _record_failure(state, "base_stability_exceeded", sample, limits)
        return "unsafe_while_carrying"
    reference_tolerance = (
        limits.torso_goal_tolerance
        if torso_reference_tolerance is None
        else _finite(
            torso_reference_tolerance,
            "torso_reference_tolerance",
        )
    )
    if reference_tolerance <= 0.0:
        raise ValueError("torso_reference_tolerance must be positive")
    torso_valid = (
        _torso_fraction(sample.torso_qpos, limits) is not None
        if safe_torso_target is None
        else _close(
            sample.torso_qpos,
            safe_torso_target,
            reference_tolerance,
        )
    )
    if not torso_valid:
        state.carry_pose_preserved = False
        _record_failure(
            state,
            (
                "torso_outside_post_grasp_reference"
                if safe_torso_target is not None
                else "torso_outside_calibrated_segment"
            ),
            sample,
            limits,
            safe_torso_target=safe_torso_target,
            torso_tracking_tolerance=reference_tolerance,
        )
        return "unsafe_while_carrying"
    return None


def _record_failure(
    state: _CarryState,
    reason: str,
    sample: CarrySample,
    limits: CarrySafetyLimits,
    *,
    safe_arm_target: Sequence[float] | None = None,
    safe_torso_target: Sequence[float] | None = None,
    previous_torso: Sequence[float] | None = None,
    torso_tracking_tolerance: float | None = None,
) -> None:
    """Freeze one private, JSON-safe terminal safety attribution snapshot."""

    if state.failure_reason_private is not None:
        return
    arm_error = None
    if safe_arm_target is not None:
        target = _vector(safe_arm_target, 7, "safe_arm_target")
        arm_error = max(
            abs(actual - expected)
            for actual, expected in zip(sample.active_arm_qpos, target)
        )
    torso_error = None
    torso_reference = (
        previous_torso
        if previous_torso is not None
        else safe_torso_target
    )
    if torso_reference is not None:
        reference = _vector(torso_reference, 4, "torso_reference")
        torso_error = max(
            abs(actual - expected)
            for actual, expected in zip(sample.torso_qpos, reference)
        )
    resolved_torso_tracking_tolerance = (
        limits.torso_goal_tolerance
        if torso_tracking_tolerance is None
        else _finite(
            torso_tracking_tolerance,
            "torso_tracking_tolerance",
        )
    )
    state.failure_reason_private = str(reason)
    state.failure_snapshot_private = {
        "active_hand": sample.active_hand,
        "expected_hand": state.hand,
        "grasp_proxy_likely": sample.grasp_proxy_likely,
        "predicted_self_collision": sample.predicted_self_collision,
        "joint_stalled": sample.joint_stalled,
        "base_contact": sample.base_contact,
        "base_z_drift_m": float(sample.base_z_drift_m),
        "roll_deg": float(sample.roll_deg),
        "pitch_deg": float(sample.pitch_deg),
        "forward_delta_m": float(sample.forward_delta_m),
        "left_delta_m": float(sample.left_delta_m),
        "yaw_delta_deg": float(sample.yaw_delta_deg),
        "torso_fraction_valid": (
            _torso_fraction(sample.torso_qpos, limits) is not None
        ),
        "post_grasp_torso_reference_valid": (
            None
            if safe_torso_target is None
            else _close(
                sample.torso_qpos,
                safe_torso_target,
                limits.torso_goal_tolerance,
            )
        ),
        "post_grasp_torso_motion_reference_valid": (
            None
            if safe_torso_target is None
            else _close(
                sample.torso_qpos,
                safe_torso_target,
                resolved_torso_tracking_tolerance,
            )
        ),
        "active_arm_qpos": list(sample.active_arm_qpos),
        "safe_arm_target": (
            None
            if safe_arm_target is None
            else list(_vector(safe_arm_target, 7, "safe_arm_target"))
        ),
        "torso_qpos": list(sample.torso_qpos),
        "safe_torso_target": (
            None
            if safe_torso_target is None
            else list(_vector(safe_torso_target, 4, "safe_torso_target"))
        ),
        "previous_torso_target": (
            None
            if previous_torso is None
            else list(_vector(previous_torso, 4, "previous_torso"))
        ),
        "arm_tracking_max_abs_error": arm_error,
        "arm_tracking_tolerance": float(limits.arm_tracking_tolerance),
        "torso_tracking_max_abs_error": torso_error,
        "torso_tracking_tolerance": float(
            resolved_torso_tracking_tolerance
        ),
        "step": int(state.steps),
    }


def _increment_too_large(
    sample: CarrySample,
    limits: CarrySafetyLimits,
) -> bool:
    return (
        math.hypot(sample.forward_delta_m, sample.left_delta_m)
        > limits.max_translation_per_tick_m
        or abs(sample.yaw_delta_deg) > limits.max_rotation_per_tick_deg
    )


def _accumulate(state: _CarryState, sample: CarrySample) -> None:
    state.forward_m += sample.forward_delta_m
    state.left_m += sample.left_delta_m
    state.yaw_deg += sample.yaw_delta_deg


def _target_reached(
    actual: tuple[float, float, float],
    target: tuple[float, float, float],
) -> bool:
    target_translation_m = math.hypot(target[0], target[1])
    if target_translation_m > 0.0:
        # The R1Pro base qvel exposed by OmniGibson is not guaranteed to use
        # the same planar axes as the high-level body-frame command.  The
        # command direction is already validated in __post_init__, while all
        # collision, carry-pose, and per-tick discontinuity gates remain live.
        # Therefore a pure translation macro stops on measured planar path
        # length instead of projecting potentially world-frame deltas onto a
        # body-frame target.  No world pose or task GT crosses this boundary.
        return math.hypot(actual[0], actual[1]) >= target_translation_m

    target_yaw_deg = target[2]
    if abs(target_yaw_deg) <= 0.0:
        return False
    return actual[2] * target_yaw_deg >= target_yaw_deg * target_yaw_deg


def _public_feedback(
    state: _CarryState,
    *,
    torso_interval: str | None = None,
    motion_kind: str | None = None,
    include_odometry: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "grasp_proxy_continuous": state.proxy_continuous,
        "carry_pose_preserved": state.carry_pose_preserved,
    }
    if torso_interval is not None:
        payload["torso_interval"] = torso_interval
    if motion_kind is not None:
        payload["motion_kind"] = motion_kind
    if include_odometry:
        payload["body_odometry"] = {
            "forward_m": float(state.forward_m),
            "left_m": float(state.left_m),
            "yaw_deg": float(state.yaw_deg),
        }
    return payload


def _audit(
    state: _CarryState,
    context: ActionExecutionContext,
    stage: str,
) -> None:
    if state.audit is None:
        return
    if stage == "end" and state.endpoint_audited:
        return
    capture = getattr(state.audit, "capture_endpoint", None)
    if not callable(capture):
        raise CarryContractError("private carry audit service is invalid")
    # Deliberately ignore the return value.  Private held truth can be written
    # to evaluator artifacts but cannot influence controller execution.
    capture(
        stage=stage,
        episode_id=context.episode_id,
        action_index=context.action_index,
    )
    if stage == "end":
        state.endpoint_audited = True


def _torso_fraction(
    torso: Sequence[float],
    limits: CarrySafetyLimits,
) -> float | None:
    value = _vector(torso, 4, "torso")
    direction = tuple(
        downward - neutral
        for neutral, downward in zip(
            limits.neutral_torso,
            limits.downward_torso,
        )
    )
    denominator = sum(part * part for part in direction)
    fraction = sum(
        (joint - neutral) * part
        for joint, neutral, part in zip(
            value,
            limits.neutral_torso,
            direction,
        )
    ) / denominator
    # Controller settling can leave the measured joints just beyond either
    # endpoint of the validated segment.  Apply the existing geometric
    # tolerance to the endpoint as well as the segment interior instead of
    # rejecting solely because the unbounded projection is e.g. -0.002.
    # Values genuinely outside the safe corridor still fail the _close check.
    bounded_fraction = min(1.0, max(0.0, fraction))
    projected = carry_torso_target(bounded_fraction, limits=limits)
    if not _close(value, projected, limits.torso_segment_tolerance):
        return None
    return bounded_fraction


def _bounded_step(
    current: Sequence[float],
    target: Sequence[float],
    maximum_delta: float,
) -> tuple[float, ...]:
    return tuple(
        before + max(-maximum_delta, min(maximum_delta, after - before))
        for before, after in zip(current, target)
    )


def _bounded_delta(
    current: Sequence[float],
    target: Sequence[float],
    maximum_delta: float,
) -> bool:
    return all(
        abs(after - before) <= maximum_delta
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


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _vector(
    value: Sequence[float],
    length: int,
    field: str,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must contain {length} finite numbers")
    try:
        actual_length = len(value)
    except TypeError as exc:
        raise ValueError(
            f"{field} must contain {length} finite numbers"
        ) from exc
    if actual_length != length:
        raise ValueError(f"{field} must contain {length} finite numbers")
    return tuple(
        _finite(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    )


def _finite(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result
