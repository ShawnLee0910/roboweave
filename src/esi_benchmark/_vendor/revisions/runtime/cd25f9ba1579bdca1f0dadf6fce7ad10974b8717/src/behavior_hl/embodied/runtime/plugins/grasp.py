"""Physical RGB-D grasp plugin with a strict public/private boundary.

The plugin owns ordering and safety invariants, while a simulator-specific
``PhysicalGraspBackend`` binds those operations to R1Pro.  Opaque target and
IK-plan tokens, contact events, payload lift, and assisted-grasp state never
enter :class:`ActionPluginResult`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from behavior_hl.embodied.policy.feedback import (
    GraspProxyThresholds,
    PublicStatus,
    grasp_proxy_result,
)
from behavior_hl.embodied.policy.protocol import (
    ActionEnvelope,
    PointActionArgs,
)
from behavior_hl.embodied.runtime.dispatch import (
    ActionExecutionContext,
    ActionPluginResult,
    ActionPluginTick,
)


PHYSICAL_GRASP_BACKEND_SERVICE = "physical_grasp_backend"
PRIVATE_GRASP_AUDIT_SERVICE = "private_grasp_audit_sink"


class GraspContractError(RuntimeError):
    """Raised when a simulator binding violates the grasp plugin contract."""


class RGBDPointStatus(str, Enum):
    """Result of resolving a normalized onboard image point."""

    READY = "ready"
    INVALID_POINT = "invalid_point"
    INVALID_DEPTH = "invalid_depth"


class GraspMotionPhase(str, Enum):
    """Ordered physical phases executed after point/reach/IK preflight."""

    REACH = "reach"
    CLOSE = "physical_close"
    LIFT = "lift"
    SAFE_CARRY_RETRACT = "safe_carry_retract"


@dataclass(frozen=True, slots=True)
class RGBDPointResolution:
    """Private RGB-D resolution carrying an opaque backend target token."""

    status: RGBDPointStatus
    target_token: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.status is RGBDPointStatus.READY:
            if self.target_token is None:
                raise ValueError("ready RGB-D resolution requires target_token")
        elif self.target_token is not None:
            raise ValueError("failed RGB-D resolution cannot carry target_token")


@dataclass(frozen=True, slots=True)
class ReachResolution:
    """Private reachability check with no world coordinate disclosure."""

    reachable: bool

    def __post_init__(self) -> None:
        if not isinstance(self.reachable, bool):
            raise ValueError("reachable must be bool")


@dataclass(frozen=True, slots=True)
class IKResolution:
    """Private IK result carrying an opaque backend plan token."""

    succeeded: bool
    plan_token: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.succeeded, bool):
            raise ValueError("succeeded must be bool")
        if self.succeeded:
            if self.plan_token is None:
                raise ValueError("successful IK requires plan_token")
        elif self.plan_token is not None:
            raise ValueError("failed IK cannot carry plan_token")


@dataclass(frozen=True, slots=True)
class GraspPhaseSample:
    """One private physical sample returned by the live backend.

    ``contact_event`` and ``payload_lift_m`` are evaluator/executor-only
    evidence.  Gripper proprioception and the wrist-image consistency flag are
    the only fields eligible for the final public proxy.
    """

    completed: bool
    simulator_steps: int
    gripper_qpos: tuple[float, float]
    gripper_qvel: tuple[float, float]
    wrist_visual_consistent: bool
    contact_event: bool = field(default=False, repr=False)
    payload_lift_m: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.completed, bool):
            raise ValueError("completed must be bool")
        if (
            isinstance(self.simulator_steps, bool)
            or not isinstance(self.simulator_steps, int)
            or self.simulator_steps <= 0
        ):
            raise ValueError("simulator_steps must be a positive integer")
        object.__setattr__(
            self,
            "gripper_qpos",
            _finite_pair(self.gripper_qpos, "gripper_qpos"),
        )
        object.__setattr__(
            self,
            "gripper_qvel",
            _finite_pair(self.gripper_qvel, "gripper_qvel"),
        )
        if not isinstance(self.wrist_visual_consistent, bool):
            raise ValueError("wrist_visual_consistent must be bool")
        if not isinstance(self.contact_event, bool):
            raise ValueError("contact_event must be bool")
        object.__setattr__(
            self,
            "payload_lift_m",
            _finite(self.payload_lift_m, "payload_lift_m"),
        )


class PhysicalGraspBackend(Protocol):
    """Simulator binding consumed by :class:`PhysicalGraspPlugin`.

    Implementations may use private geometry, contacts, and assisted-grasp
    APIs internally.  Those values remain opaque to the policy runtime.
    """

    def resolve_rgbd_point(
        self,
        *,
        camera: str,
        point: tuple[float, float],
        hand: str,
    ) -> RGBDPointResolution:
        """Resolve the selected current-frame RGB-D point."""

    def check_reach(
        self,
        *,
        target_token: Any,
        hand: str,
    ) -> ReachResolution:
        """Check the frozen arm workspace without moving the robot."""

    def solve_ik(
        self,
        *,
        target_token: Any,
        hand: str,
    ) -> IKResolution:
        """Create a collision-aware local arm plan."""

    def tick_motion(
        self,
        *,
        phase: GraspMotionPhase,
        plan_token: Any,
        hand: str,
    ) -> GraspPhaseSample:
        """Advance exactly one physical phase by a bounded simulator tick."""

    def enable_assisted_grasp(
        self,
        *,
        plan_token: Any,
        hand: str,
    ) -> bool:
        """Enable assistance only after a real close-phase contact event."""

    def hold_safe(self, *, hand: str) -> None:
        """Stop local motion and hold a safe arm/gripper command."""


@dataclass(frozen=True, slots=True)
class PrivateGraspAuditEvent:
    """Private execution evidence that is write-only from the plugin."""

    event: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.event, str) or not self.event:
            raise ValueError("private audit event must be a non-empty string")
        object.__setattr__(
            self,
            "details",
            MappingProxyType(dict(self.details)),
        )


class PrivateGraspAuditSink(Protocol):
    """Evaluator-owned sink; it is never queried by policy-facing code."""

    def record(self, event: PrivateGraspAuditEvent) -> None:
        """Persist private contact, assistance, and lift evidence."""


@dataclass(frozen=True, slots=True)
class PhysicalGraspConfig:
    """Frozen thresholds for one physical-grasp plugin instance."""

    proxy_thresholds: GraspProxyThresholds
    minimum_payload_lift_m: float = 0.05

    def __post_init__(self) -> None:
        lift = _finite(
            self.minimum_payload_lift_m,
            "minimum_payload_lift_m",
        )
        if lift < 0.05:
            raise ValueError("minimum_payload_lift_m must be at least 0.05")
        object.__setattr__(self, "minimum_payload_lift_m", lift)


@dataclass(slots=True)
class _GraspLifecycle:
    backend: PhysicalGraspBackend = field(repr=False)
    args: PointActionArgs
    private_audit_sink: PrivateGraspAuditSink | None = field(
        default=None,
        repr=False,
    )
    stage: str = "rgbd"
    target_token: Any = field(default=None, repr=False)
    plan_token: Any = field(default=None, repr=False)
    physical_phase: GraspMotionPhase | None = None
    terminal_status: PublicStatus | None = None
    assisted_grasp_enabled: bool = field(default=False, repr=False)
    contact_observed: bool = field(default=False, repr=False)
    maximum_payload_lift_m: float = field(default=0.0, repr=False)
    last_qpos: tuple[float, float] | None = None
    last_qvel: tuple[float, float] | None = None
    wrist_visual_consistent: bool = False
    proxy_stable_steps: int = 0
    physical_steps: int = 0


class PhysicalGraspPlugin:
    """ActionPlugin for sensor-targeted physical grasp and carry-safe retract."""

    controller_key = "manipulation.grasp_at"

    def __init__(self, config: PhysicalGraspConfig) -> None:
        if not isinstance(config, PhysicalGraspConfig):
            raise TypeError("config must be PhysicalGraspConfig")
        self.config = config

    def begin(
        self,
        action: ActionEnvelope,
        context: ActionExecutionContext,
    ) -> _GraspLifecycle:
        if action.action != "grasp_at":
            raise GraspContractError(
                "physical grasp plugin accepts only grasp_at"
            )
        if not isinstance(action.args, PointActionArgs):
            raise GraspContractError(
                "grasp_at requires parsed PointActionArgs"
            )
        try:
            backend = context.services[PHYSICAL_GRASP_BACKEND_SERVICE]
        except KeyError as exc:
            raise GraspContractError(
                f"missing {PHYSICAL_GRASP_BACKEND_SERVICE!r} service"
            ) from exc
        _validate_backend(backend)
        private_audit_sink = context.services.get(
            PRIVATE_GRASP_AUDIT_SERVICE
        )
        if (
            private_audit_sink is not None
            and not callable(getattr(private_audit_sink, "record", None))
        ):
            raise GraspContractError(
                "private grasp audit sink must provide record(event)"
            )
        return _GraspLifecycle(
            backend=backend,
            private_audit_sink=private_audit_sink,
            args=action.args,
        )

    def tick(
        self,
        state: _GraspLifecycle,
        context: ActionExecutionContext,
    ) -> ActionPluginTick:
        del context
        if state.terminal_status is not None:
            raise GraspContractError("cannot tick a terminal grasp lifecycle")

        if state.stage == "rgbd":
            return self._tick_rgbd(state)
        if state.stage == "reach_check":
            return self._tick_reach_check(state)
        if state.stage == "ik":
            return self._tick_ik(state)
        if state.stage == "physical":
            return self._tick_physical(state)
        raise GraspContractError(f"unknown grasp lifecycle stage {state.stage!r}")

    def finish(
        self,
        state: _GraspLifecycle,
        context: ActionExecutionContext,
    ) -> ActionPluginResult:
        del context
        if state.terminal_status is None:
            raise GraspContractError("cannot finish a non-terminal grasp")

        status = state.terminal_status
        if status is PublicStatus.GRASP_LIKELY:
            if state.last_qpos is None or state.last_qvel is None:
                raise GraspContractError(
                    "completed grasp has no public gripper observation"
                )
            # The public classification reads only approved robot sensors.
            # Private contact, assisted-grasp, and payload-lift evidence gate
            # the physical lifecycle but are not classifier inputs.
            status = grasp_proxy_result(
                hand=state.args.hand,
                qpos=state.last_qpos,
                qvel=state.last_qvel,
                consecutive_stable_steps=state.proxy_stable_steps,
                wrist_visual_consistent=(
                    state.wrist_visual_consistent
                ),
                simulator_steps=state.physical_steps,
                thresholds=self.config.proxy_thresholds,
            ).status

        return ActionPluginResult(
            status=status.value,
            public_feedback=self._public_feedback(state),
        )

    def abort(
        self,
        state: _GraspLifecycle,
        context: ActionExecutionContext,
        reason: str,
    ) -> ActionPluginResult:
        del context, reason
        state.backend.hold_safe(hand=state.args.hand)
        state.terminal_status = PublicStatus.BLOCKED
        return ActionPluginResult(
            status=PublicStatus.BLOCKED.value,
            public_feedback={
                "hand": state.args.hand,
                "reason": "grasp_execution_aborted",
            },
        )

    def _tick_rgbd(self, state: _GraspLifecycle) -> ActionPluginTick:
        resolution = state.backend.resolve_rgbd_point(
            camera=state.args.camera,
            point=state.args.point,
            hand=state.args.hand,
        )
        if not isinstance(resolution, RGBDPointResolution):
            raise GraspContractError(
                "resolve_rgbd_point returned an invalid contract value"
            )
        if resolution.status is RGBDPointStatus.INVALID_POINT:
            return self._terminal(state, PublicStatus.INVALID_POINT, 0)
        if resolution.status is RGBDPointStatus.INVALID_DEPTH:
            return self._terminal(state, PublicStatus.INVALID_DEPTH, 0)
        state.target_token = resolution.target_token
        state.stage = "reach_check"
        return ActionPluginTick(done=False, simulator_steps=0)

    def _tick_reach_check(
        self,
        state: _GraspLifecycle,
    ) -> ActionPluginTick:
        resolution = state.backend.check_reach(
            target_token=state.target_token,
            hand=state.args.hand,
        )
        if not isinstance(resolution, ReachResolution):
            raise GraspContractError(
                "check_reach returned an invalid contract value"
            )
        if not resolution.reachable:
            return self._terminal(state, PublicStatus.OUT_OF_REACH, 0)
        state.stage = "ik"
        return ActionPluginTick(done=False, simulator_steps=0)

    def _tick_ik(self, state: _GraspLifecycle) -> ActionPluginTick:
        resolution = state.backend.solve_ik(
            target_token=state.target_token,
            hand=state.args.hand,
        )
        if not isinstance(resolution, IKResolution):
            raise GraspContractError(
                "solve_ik returned an invalid contract value"
            )
        if not resolution.succeeded:
            return self._terminal(state, PublicStatus.IK_FAILED, 0)
        state.plan_token = resolution.plan_token
        state.stage = "physical"
        state.physical_phase = GraspMotionPhase.REACH
        return ActionPluginTick(done=False, simulator_steps=0)

    def _tick_physical(
        self,
        state: _GraspLifecycle,
    ) -> ActionPluginTick:
        phase = state.physical_phase
        if phase is None:
            raise GraspContractError("physical grasp phase is missing")
        sample = state.backend.tick_motion(
            phase=phase,
            plan_token=state.plan_token,
            hand=state.args.hand,
        )
        if not isinstance(sample, GraspPhaseSample):
            raise GraspContractError(
                "tick_motion returned an invalid contract value"
            )
        if sample.contact_event and phase is not GraspMotionPhase.CLOSE:
            raise GraspContractError(
                "contact events are accepted only during physical close"
            )

        state.physical_steps += sample.simulator_steps
        state.last_qpos = sample.gripper_qpos
        state.last_qvel = sample.gripper_qvel
        state.wrist_visual_consistent = sample.wrist_visual_consistent

        if phase is GraspMotionPhase.CLOSE and sample.contact_event:
            state.contact_observed = True
            self._record_private(
                state,
                "physical_contact",
                {"phase": phase.value},
            )
            if not state.assisted_grasp_enabled:
                enabled = state.backend.enable_assisted_grasp(
                    plan_token=state.plan_token,
                    hand=state.args.hand,
                )
                if not isinstance(enabled, bool):
                    raise GraspContractError(
                        "enable_assisted_grasp must return bool"
                    )
                state.assisted_grasp_enabled = enabled
                self._record_private(
                    state,
                    "assisted_grasp",
                    {"enabled": enabled},
                )
                if not enabled:
                    state.backend.hold_safe(hand=state.args.hand)
                    return self._terminal(
                        state,
                        PublicStatus.GRASP_UNCERTAIN,
                        sample.simulator_steps,
                        capture_observation=True,
                    )

        self._update_proxy_window(state, sample)

        if phase is GraspMotionPhase.LIFT:
            state.maximum_payload_lift_m = max(
                state.maximum_payload_lift_m,
                sample.payload_lift_m,
            )
            self._record_private(
                state,
                "payload_lift",
                {
                    "observed_m": sample.payload_lift_m,
                    "maximum_m": state.maximum_payload_lift_m,
                },
            )

        if not sample.completed:
            return ActionPluginTick(
                done=False,
                simulator_steps=sample.simulator_steps,
                capture_observation=sample.contact_event,
            )

        if phase is GraspMotionPhase.REACH:
            state.physical_phase = GraspMotionPhase.CLOSE
        elif phase is GraspMotionPhase.CLOSE:
            if (
                not state.contact_observed
                or not state.assisted_grasp_enabled
            ):
                state.backend.hold_safe(hand=state.args.hand)
                return self._terminal(
                    state,
                    PublicStatus.GRASP_UNCERTAIN,
                    sample.simulator_steps,
                    capture_observation=True,
                )
            state.physical_phase = GraspMotionPhase.LIFT
        elif phase is GraspMotionPhase.LIFT:
            if (
                state.maximum_payload_lift_m
                < self.config.minimum_payload_lift_m
            ):
                state.backend.hold_safe(hand=state.args.hand)
                return self._terminal(
                    state,
                    PublicStatus.GRASP_UNCERTAIN,
                    sample.simulator_steps,
                    capture_observation=True,
                )
            state.physical_phase = GraspMotionPhase.SAFE_CARRY_RETRACT
        elif phase is GraspMotionPhase.SAFE_CARRY_RETRACT:
            return self._terminal(
                state,
                PublicStatus.GRASP_LIKELY,
                sample.simulator_steps,
                capture_observation=True,
            )
        return ActionPluginTick(
            done=False,
            simulator_steps=sample.simulator_steps,
            capture_observation=True,
        )

    def _update_proxy_window(
        self,
        state: _GraspLifecycle,
        sample: GraspPhaseSample,
    ) -> None:
        thresholds = self.config.proxy_thresholds
        # The public stability window is phase-gated by the commanded close,
        # never by private contact or held truth. Empty-close calibration is
        # represented by ``empty_close_qpos`` below.
        if state.physical_phase not in {
            GraspMotionPhase.CLOSE,
            GraspMotionPhase.LIFT,
            GraspMotionPhase.SAFE_CARRY_RETRACT,
        }:
            state.proxy_stable_steps = 0
            return
        qpos_gap = max(
            abs(sample.gripper_qpos[index] - thresholds.empty_close_qpos[index])
            for index in range(2)
        )
        max_qvel = max(abs(value) for value in sample.gripper_qvel)
        if (
            qpos_gap >= thresholds.minimum_qpos_gap
            and max_qvel <= thresholds.qvel_epsilon
            and sample.wrist_visual_consistent
        ):
            state.proxy_stable_steps += sample.simulator_steps
        else:
            state.proxy_stable_steps = 0

    @staticmethod
    def _terminal(
        state: _GraspLifecycle,
        status: PublicStatus,
        simulator_steps: int,
        *,
        capture_observation: bool = False,
    ) -> ActionPluginTick:
        state.terminal_status = status
        return ActionPluginTick(
            done=True,
            simulator_steps=simulator_steps,
            capture_observation=capture_observation,
        )

    @staticmethod
    def _public_feedback(state: _GraspLifecycle) -> dict[str, Any]:
        feedback: dict[str, Any] = {"hand": state.args.hand}
        if state.last_qpos is not None and state.last_qvel is not None:
            feedback.update(
                {
                    "gripper_qpos": list(state.last_qpos),
                    "gripper_qvel": list(state.last_qvel),
                    "wrist_visual_consistent": (
                        state.wrist_visual_consistent
                    ),
                    "proxy_stable_steps": state.proxy_stable_steps,
                }
            )
        return feedback

    @staticmethod
    def _record_private(
        state: _GraspLifecycle,
        event: str,
        details: Mapping[str, Any],
    ) -> None:
        sink = state.private_audit_sink
        if sink is not None:
            sink.record(PrivateGraspAuditEvent(event, details))


def _validate_backend(backend: Any) -> None:
    missing = [
        method_name
        for method_name in (
            "resolve_rgbd_point",
            "check_reach",
            "solve_ik",
            "tick_motion",
            "enable_assisted_grasp",
            "hold_safe",
        )
        if not callable(getattr(backend, method_name, None))
    ]
    if missing:
        raise GraspContractError(
            f"physical grasp backend is missing methods: {missing}"
        )


def _finite_pair(
    value: tuple[float, float],
    field_name: str,
) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly two values")
    return (
        _finite(value[0], f"{field_name}[0]"),
        _finite(value[1], f"{field_name}[1]"),
    )


def _finite(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number
