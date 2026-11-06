"""Shared fixed-step base plugin for the kinematic-research benchmark lane.

The plugin never chooses a destination and never accepts a world-space target
from a policy.  One registered ESI-style action can produce at most one
configured bounded translation or one configured bounded rotation.  A live
backend may block a requested landing pose and reports the actual odometry.
The active action profile chooses endpoint-only or swept collision admission
before the official local pose application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

from behavior_hl.embodied.policy.protocol import ActionEnvelope
from behavior_hl.embodied.runtime.dispatch import (
    ActionExecutionContext,
    ActionPluginResult,
    ActionPluginTick,
)
from behavior_hl.embodied.runtime.public_action_feedback import EXECUTION_UNVERIFIED
from behavior_hl.navigation import PlanarPose, normalize_angle


RESTRICTED_KINEMATIC_BASE_SERVICE = "restricted_kinematic_base_executor"
RESTRICTED_KINEMATIC_PUBLIC_MOVE_M = 0.15
RESTRICTED_KINEMATIC_PUBLIC_TURN_DEG = 15.0
ACTION20_RESTRICTED_KINEMATIC_PUBLIC_TURN_DEG = 10.0
RESTRICTED_KINEMATIC_INTERNAL_SAMPLE_M = 0.05
_GUARD_NUMERIC_RESIDUE_ABS = 1e-12
RESTRICTED_KINEMATIC_BASE_VERSION = (
    "behavior2026.restricted-kinematic-local-base/v9-endpoint-mode"
)
RESTRICTED_KINEMATIC_BASE_ACTIONS = frozenset(
    {
        "move_forward",
        "move_backward",
        "move_left",
        "move_right",
        "turn_left",
        "turn_right",
    }
)


class RestrictedKinematicBaseError(RuntimeError):
    """Raised when the bounded local-pose contract is violated."""


def _canonicalize_guard_pose(
    pose: PlanarPose,
    *,
    expected_z: float,
    expected_yaw: float,
    error_message: str,
) -> PlanarPose:
    """Snap representation-only residue while preserving strict safety.

    Collision guards may normalize an unchanged yaw through sin/cos and return
    a value that differs by one floating-point bit.  A 1e-12 boundary is many
    orders below simulator resolution; anything larger remains a hard error.
    """

    z_error = abs(float(pose.z) - float(expected_z))
    yaw_error = abs(
        normalize_angle(float(pose.yaw) - float(expected_yaw))
    )
    if (
        not math.isfinite(z_error)
        or not math.isfinite(yaw_error)
        or z_error > _GUARD_NUMERIC_RESIDUE_ABS
        or yaw_error > _GUARD_NUMERIC_RESIDUE_ABS
    ):
        raise RestrictedKinematicBaseError(error_message)
    return PlanarPose(
        float(pose.x),
        float(pose.y),
        float(expected_z),
        float(expected_yaw),
    )


@dataclass(frozen=True, slots=True)
class RestrictedKinematicExecution:
    status: str
    simulator_steps: int
    forward_m: float
    left_m: float
    yaw_deg: float
    private_report: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"completed", "blocked"}:
            raise ValueError("restricted kinematic status is invalid")
        if (
            isinstance(self.simulator_steps, bool)
            or not isinstance(self.simulator_steps, int)
            or self.simulator_steps < 0
        ):
            raise ValueError("simulator_steps must be non-negative")
        for name in ("forward_m", "left_m", "yaw_deg"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "private_report",
            MappingProxyType(dict(self.private_report)),
        )


class RestrictedKinematicExecutor(Protocol):
    def execute(self, action_name: str) -> RestrictedKinematicExecution:
        """Execute exactly one bounded registered local base action."""


def restricted_kinematic_action_target(
    pose: PlanarPose,
    action_name: str,
    *,
    move_distance_m: float,
    turn_angle_deg: float,
) -> PlanarPose:
    """Return the exact target owned by one registered local base action.

    Private Oracle search may inspect this geometry when choosing its next
    *public* action.  Keeping the calculation here prevents the authoring
    layer from drifting away from the executor that the model also uses.
    This helper does not check safety and does not apply a pose.
    """

    if action_name not in RESTRICTED_KINEMATIC_BASE_ACTIONS:
        raise RestrictedKinematicBaseError(
            f"unsupported restricted base action: {action_name!r}"
        )
    if not isinstance(pose, PlanarPose):
        raise RestrictedKinematicBaseError("target pose is invalid")
    _require_finite_pose(pose, "target source")
    distance = float(move_distance_m)
    turn_deg = float(turn_angle_deg)
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("move_distance_m must be finite and positive")
    if not math.isfinite(turn_deg) or turn_deg <= 0.0 or turn_deg > 180.0:
        raise ValueError("turn_angle_deg must be finite in (0, 180]")
    if action_name in {"turn_left", "turn_right"}:
        sign = 1.0 if action_name == "turn_left" else -1.0
        return PlanarPose(
            pose.x,
            pose.y,
            pose.z,
            normalize_angle(pose.yaw + sign * math.radians(turn_deg)),
        )
    forward = (math.cos(pose.yaw), math.sin(pose.yaw))
    left = (-forward[1], forward[0])
    directions = {
        "move_forward": forward,
        "move_backward": (-forward[0], -forward[1]),
        "move_left": left,
        "move_right": (-left[0], -left[1]),
    }
    dx, dy = directions[action_name]
    return PlanarPose(
        pose.x + distance * dx,
        pose.y + distance * dy,
        pose.z,
        pose.yaw,
    )


PoseReader = Callable[[], PlanarPose]
PoseGuard = Callable[[PlanarPose, PlanarPose], bool | PlanarPose]
CollisionGuard = Callable[[PlanarPose, PlanarPose], bool | PlanarPose]
PoseApplier = Callable[[PlanarPose], int]
GuardDiagnosticReader = Callable[[], Mapping[str, Any]]


@dataclass(slots=True)
class FixedStepKinematicBaseExecutor:
    """Pure fixed-step contract around a live pose reader/guard/applier."""

    read_pose: PoseReader = field(repr=False)
    guard: PoseGuard = field(repr=False)
    apply_pose: PoseApplier = field(repr=False)
    guard_diagnostic: GuardDiagnosticReader | None = field(
        default=None,
        repr=False,
    )
    collision_guard: CollisionGuard | None = field(default=None, repr=False)
    collision_guard_diagnostic: GuardDiagnosticReader | None = field(
        default=None,
        repr=False,
    )
    rollback_pose: PoseApplier | None = field(default=None, repr=False)
    move_distance_m: float = RESTRICTED_KINEMATIC_PUBLIC_MOVE_M
    turn_angle_deg: float = RESTRICTED_KINEMATIC_PUBLIC_TURN_DEG
    maximum_guard_translation_m: float | None = None
    allow_guard_translation_redirect: bool = False
    maximum_translation_error_m: float = 0.02
    maximum_yaw_error_deg: float = 2.0
    # Match the established live base safety envelope.  Millimetre-scale
    # PhysX readback noise is diagnostic for this kinematic-research lane;
    # centimetre-scale vertical motion is still rejected and rolled back.
    maximum_z_drift_m: float = 0.03

    def __post_init__(self) -> None:
        for callback_name in ("read_pose", "guard", "apply_pose"):
            if not callable(getattr(self, callback_name)):
                raise TypeError(f"{callback_name} must be callable")
        if self.guard_diagnostic is not None and not callable(
            self.guard_diagnostic
        ):
            raise TypeError("guard_diagnostic must be callable when provided")
        if self.collision_guard is not None and not callable(
            self.collision_guard
        ):
            raise TypeError("collision_guard must be callable when provided")
        if self.collision_guard_diagnostic is not None and not callable(
            self.collision_guard_diagnostic
        ):
            raise TypeError(
                "collision_guard_diagnostic must be callable when provided"
            )
        if self.rollback_pose is not None and not callable(self.rollback_pose):
            raise TypeError("rollback_pose must be callable when provided")
        if not isinstance(self.allow_guard_translation_redirect, bool):
            raise TypeError("allow_guard_translation_redirect must be boolean")
        if self.maximum_guard_translation_m is not None:
            self.maximum_guard_translation_m = float(
                self.maximum_guard_translation_m
            )
            if (
                not math.isfinite(self.maximum_guard_translation_m)
                or self.maximum_guard_translation_m <= 0.0
            ):
                raise ValueError(
                    "maximum_guard_translation_m must be finite and positive"
                )
        for name in (
            "move_distance_m",
            "turn_angle_deg",
            "maximum_translation_error_m",
            "maximum_yaw_error_deg",
            "maximum_z_drift_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, value)
        if self.turn_angle_deg > 180.0:
            raise ValueError("turn_angle_deg must not exceed 180")

    def execute(self, action_name: str) -> RestrictedKinematicExecution:
        if action_name not in RESTRICTED_KINEMATIC_BASE_ACTIONS:
            raise RestrictedKinematicBaseError(
                f"unsupported restricted base action: {action_name!r}"
            )
        start = self.read_pose()
        if not isinstance(start, PlanarPose):
            raise RestrictedKinematicBaseError("pose reader returned invalid pose")
        _require_finite_pose(start, "start")
        requested = self._target(start, action_name)
        if requested.z != start.z:
            raise RestrictedKinematicBaseError("local base target changed z")
        guard_result = self.guard(start, requested)
        if isinstance(guard_result, PlanarPose):
            _require_finite_pose(guard_result, "guard target")
            target = _canonicalize_guard_pose(
                guard_result,
                expected_z=start.z,
                expected_yaw=requested.yaw,
                error_message="guard changed z or requested yaw",
            )
            requested_distance = math.hypot(
                requested.x - start.x,
                requested.y - start.y,
            )
            applied_distance = math.hypot(
                target.x - start.x,
                target.y - start.y,
            )
            if applied_distance > requested_distance + 1e-9:
                requested_dx = requested.x - start.x
                requested_dy = requested.y - start.y
                applied_dx = target.x - start.x
                applied_dy = target.y - start.y
                if (
                    requested_distance <= 1e-9
                    or (
                        not self.allow_guard_translation_redirect
                        and abs(
                            requested_dx * applied_dy
                            - requested_dy * applied_dx
                        )
                        > 1e-8
                    )
                    or requested_dx * applied_dx + requested_dy * applied_dy <= 0.0
                    or self.maximum_guard_translation_m is None
                    or applied_distance > self.maximum_guard_translation_m + 1e-9
                ):
                    raise RestrictedKinematicBaseError(
                        "guard expanded the requested translation"
                    )
        elif isinstance(guard_result, bool):
            if not guard_result:
                return RestrictedKinematicExecution(
                    "blocked",
                    0,
                    0.0,
                    0.0,
                    0.0,
                    self._report(
                        start,
                        requested,
                        start,
                        start,
                        guard_pass=False,
                        collision_guard_attempted=False,
                        collision_guard_pass=None,
                    ),
                )
            target = requested
        else:
            raise RestrictedKinematicBaseError(
                "guard returned an invalid decision"
            )
        if target == start and requested != start:
            return RestrictedKinematicExecution(
                "blocked",
                0,
                0.0,
                0.0,
                0.0,
                self._report(
                    start,
                    requested,
                    start,
                    start,
                    guard_pass=False,
                    collision_guard_attempted=False,
                    collision_guard_pass=None,
                ),
            )

        collision_guard_attempted = self.collision_guard is not None
        collision_guard_pass: bool | None = None
        collision_guard_clipped = False
        if self.collision_guard is not None:
            collision_result = self.collision_guard(start, target)
            if isinstance(collision_result, PlanarPose):
                _require_finite_pose(
                    collision_result,
                    "collision-guard target",
                )
                collision_target = _canonicalize_guard_pose(
                    collision_result,
                    expected_z=start.z,
                    expected_yaw=target.yaw,
                    error_message=(
                        "collision guard changed z or requested yaw"
                    ),
                )
                accepted_distance = math.hypot(
                    target.x - start.x,
                    target.y - start.y,
                )
                collision_distance = math.hypot(
                    collision_target.x - start.x,
                    collision_target.y - start.y,
                )
                if collision_distance > accepted_distance + 1e-9:
                    raise RestrictedKinematicBaseError(
                        "collision guard expanded the accepted translation"
                    )
                target = collision_target
                collision_guard_pass = True
                collision_guard_clipped = True
            elif isinstance(collision_result, bool):
                collision_guard_pass = collision_result
            else:
                raise RestrictedKinematicBaseError(
                    "collision guard returned an invalid decision"
                )
            if not collision_guard_pass:
                return RestrictedKinematicExecution(
                    "blocked",
                    0,
                    0.0,
                    0.0,
                    0.0,
                    self._report(
                        start,
                        requested,
                        target,
                        start,
                        guard_pass=True,
                        collision_guard_attempted=True,
                        collision_guard_pass=False,
                        collision_guard_clipped=False,
                    ),
                )
            if target == start and requested != start:
                return RestrictedKinematicExecution(
                    "blocked",
                    0,
                    0.0,
                    0.0,
                    0.0,
                    self._report(
                        start,
                        requested,
                        target,
                        start,
                        guard_pass=True,
                        collision_guard_attempted=True,
                        collision_guard_pass=False,
                        collision_guard_clipped=True,
                    ),
                )

        simulator_steps = self.apply_pose(target)
        if (
            isinstance(simulator_steps, bool)
            or not isinstance(simulator_steps, int)
            or simulator_steps < 0
        ):
            raise RestrictedKinematicBaseError(
                "pose applier returned invalid simulator-step count"
            )
        end = self.read_pose()
        if not isinstance(end, PlanarPose):
            raise RestrictedKinematicBaseError("pose reader returned invalid end pose")
        _require_finite_pose(end, "end")
        translation_error = math.hypot(end.x - target.x, end.y - target.y)
        yaw_error = abs(math.degrees(normalize_angle(end.yaw - target.yaw)))
        z_drift = abs(end.z - start.z)
        if (
            translation_error > self.maximum_translation_error_m
            or yaw_error > self.maximum_yaw_error_deg
            or z_drift > self.maximum_z_drift_m
        ):
            rollback = "not_configured"
            if self.rollback_pose is not None:
                try:
                    rollback_steps = self.rollback_pose(start)
                    if (
                        isinstance(rollback_steps, bool)
                        or not isinstance(rollback_steps, int)
                        or rollback_steps < 0
                    ):
                        raise RestrictedKinematicBaseError(
                            "rollback pose applier returned invalid simulator-step count"
                        )
                    recovered = self.read_pose()
                    if not isinstance(recovered, PlanarPose):
                        raise RestrictedKinematicBaseError(
                            "pose reader returned invalid rollback pose"
                        )
                    _require_finite_pose(recovered, "rollback")
                    recovery_translation_error = math.hypot(
                        recovered.x - start.x,
                        recovered.y - start.y,
                    )
                    recovery_yaw_error = abs(
                        math.degrees(normalize_angle(recovered.yaw - start.yaw))
                    )
                    recovery_z_error = abs(recovered.z - start.z)
                    if (
                        recovery_translation_error
                        > self.maximum_translation_error_m
                        or recovery_yaw_error > self.maximum_yaw_error_deg
                        or recovery_z_error > self.maximum_z_drift_m
                    ):
                        raise RestrictedKinematicBaseError(
                            "last-good rollback postcondition failed"
                        )
                    rollback = "restored_last_good_pose"
                except Exception as exc:
                    raise RestrictedKinematicBaseError(
                        "restricted local pose postcondition failed and "
                        f"last-good rollback failed: {type(exc).__name__}: {exc}"
                    ) from exc
            if rollback == "restored_last_good_pose":
                return RestrictedKinematicExecution(
                    "blocked",
                    simulator_steps + rollback_steps,
                    0.0,
                    0.0,
                    0.0,
                    self._report(
                        start,
                        requested,
                        target,
                        recovered,
                        guard_pass=True,
                        collision_guard_attempted=collision_guard_attempted,
                        collision_guard_pass=collision_guard_pass,
                        collision_guard_clipped=collision_guard_clipped,
                        postcondition_private={
                            "status": "recovered_blocked",
                            "failed_end_pose_private": _pose_record(end),
                            "translation_error_m": translation_error,
                            "yaw_error_deg": yaw_error,
                            "z_drift_m": z_drift,
                            "rollback": rollback,
                            "rollback_simulator_steps": rollback_steps,
                        },
                    ),
                )
            raise RestrictedKinematicBaseError(
                "restricted local pose postcondition failed: "
                f"translation_error={translation_error:.6f}, "
                f"yaw_error_deg={yaw_error:.6f}, z_drift={z_drift:.6f}, "
                f"rollback={rollback}"
            )

        applied_distance = math.hypot(target.x - start.x, target.y - start.y)
        requested_distance = math.hypot(
            requested.x - start.x,
            requested.y - start.y,
        )
        odometry_pose = (
            requested
            if applied_distance > requested_distance + 1e-9
            else end
        )
        dx = odometry_pose.x - start.x
        dy = odometry_pose.y - start.y
        cosine = math.cos(start.yaw)
        sine = math.sin(start.yaw)
        forward = cosine * dx + sine * dy
        left = -sine * dx + cosine * dy
        yaw = math.degrees(normalize_angle(odometry_pose.yaw - start.yaw))
        return RestrictedKinematicExecution(
            "completed",
            simulator_steps,
            forward,
            left,
            yaw,
            self._report(
                start,
                requested,
                target,
                end,
                guard_pass=True,
                collision_guard_attempted=collision_guard_attempted,
                collision_guard_pass=collision_guard_pass,
                collision_guard_clipped=collision_guard_clipped,
            ),
        )

    def _target(self, pose: PlanarPose, action_name: str) -> PlanarPose:
        return restricted_kinematic_action_target(
            pose,
            action_name,
            move_distance_m=self.move_distance_m,
            turn_angle_deg=self.turn_angle_deg,
        )

    def _report(
        self,
        start: PlanarPose,
        requested: PlanarPose,
        applied_target: PlanarPose,
        end: PlanarPose,
        *,
        guard_pass: bool,
        collision_guard_attempted: bool,
        collision_guard_pass: bool | None,
        collision_guard_clipped: bool = False,
        postcondition_private: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        guard_private: Mapping[str, Any] | None = None
        if self.guard_diagnostic is not None:
            value = self.guard_diagnostic()
            if not isinstance(value, Mapping):
                raise RestrictedKinematicBaseError(
                    "guard diagnostic returned an invalid report"
                )
            guard_private = dict(value)
        collision_guard_private: Mapping[str, Any] | None = None
        if collision_guard_attempted and self.collision_guard_diagnostic is not None:
            value = self.collision_guard_diagnostic()
            if not isinstance(value, Mapping):
                raise RestrictedKinematicBaseError(
                    "collision guard diagnostic returned an invalid report"
                )
            collision_guard_private = dict(value)
        return {
            "schema_version": RESTRICTED_KINEMATIC_BASE_VERSION,
            "base_execution": "restricted_kinematic",
            "local_pose_update": True,
            "long_range_teleport": False,
            "physical_base_dynamics": False,
            "static_eroded_map_guard_pass": bool(guard_pass),
            "dynamic_collision_guard_claimed": bool(
                collision_guard_attempted
            ),
            "live_full_body_collision_guard_attempted": bool(
                collision_guard_attempted
            ),
            "live_full_body_collision_guard_pass": collision_guard_pass,
            "live_full_body_collision_guard_clipped": bool(
                collision_guard_clipped
            ),
            "local_guard_private": guard_private,
            "live_collision_guard_private": collision_guard_private,
            "postcondition_private": (
                None
                if postcondition_private is None
                else dict(postcondition_private)
            ),
            "start_pose_private": _pose_record(start),
            "requested_pose_private": _pose_record(requested),
            "applied_target_pose_private": _pose_record(applied_target),
            "end_pose_private": _pose_record(end),
            "translation_clipped": bool(
                math.hypot(
                    applied_target.x - start.x,
                    applied_target.y - start.y,
                )
                < math.hypot(
                    requested.x - start.x,
                    requested.y - start.y,
                )
                - 1e-9
            ),
            "translation_expanded": bool(
                math.hypot(
                    applied_target.x - start.x,
                    applied_target.y - start.y,
                )
                > math.hypot(
                    requested.x - start.x,
                    requested.y - start.y,
                )
                + 1e-9
            ),
            "requested_translation_m": math.hypot(
                requested.x - start.x,
                requested.y - start.y,
            ),
            "applied_translation_m": math.hypot(
                applied_target.x - start.x,
                applied_target.y - start.y,
            ),
        }


@dataclass(slots=True)
class _PluginState:
    executor: RestrictedKinematicExecutor = field(repr=False)
    action_name: str
    execution: RestrictedKinematicExecution | None = None

    def private_diagnostic(self) -> Mapping[str, Any] | None:
        return (
            None
            if self.execution is None
            else dict(self.execution.private_report)
        )


@dataclass(slots=True)
class RestrictedKinematicBasePlugin:
    """Dispatch one public base action through the shared local executor."""

    controller_key: str
    action_name: str
    implementation_version: str = field(
        default=RESTRICTED_KINEMATIC_BASE_VERSION,
        init=False,
    )
    service_key: str = RESTRICTED_KINEMATIC_BASE_SERVICE

    def __post_init__(self) -> None:
        if not isinstance(self.controller_key, str) or not self.controller_key:
            raise ValueError("controller_key must be non-empty")
        if self.action_name not in RESTRICTED_KINEMATIC_BASE_ACTIONS:
            raise ValueError("action_name is not a restricted base action")

    def begin(
        self,
        action: ActionEnvelope,
        context: ActionExecutionContext,
    ) -> _PluginState:
        if action.action != self.action_name:
            raise RestrictedKinematicBaseError(
                f"{self.controller_key} received {action.action!r}"
            )
        executor = context.services.get(self.service_key)
        if not callable(getattr(executor, "execute", None)):
            raise RestrictedKinematicBaseError(
                f"missing {self.service_key!r}"
            )
        return _PluginState(executor=executor, action_name=self.action_name)

    def tick(
        self,
        state: _PluginState,
        context: ActionExecutionContext,
    ) -> ActionPluginTick:
        del context
        if state.execution is not None:
            raise RestrictedKinematicBaseError(
                "restricted base action was ticked after completion"
            )
        execution = state.executor.execute(state.action_name)
        if not isinstance(execution, RestrictedKinematicExecution):
            raise RestrictedKinematicBaseError(
                "restricted base executor returned invalid result"
            )
        state.execution = execution
        return ActionPluginTick(
            done=True,
            simulator_steps=execution.simulator_steps,
            public_feedback=_restricted_kinematic_public_feedback(execution),
        )

    def finish(
        self,
        state: _PluginState,
        context: ActionExecutionContext,
    ) -> ActionPluginResult:
        del context
        if state.execution is None:
            raise RestrictedKinematicBaseError(
                "restricted base action is not terminal"
            )
        return ActionPluginResult(
            state.execution.status,
            _restricted_kinematic_public_feedback(state.execution),
        )

    def abort(
        self,
        state: _PluginState,
        context: ActionExecutionContext,
        reason: str,
    ) -> ActionPluginResult:
        del context, reason
        if state.execution is not None:
            return self.finish(state, ActionExecutionContext("abort", 0))
        return ActionPluginResult(
            "blocked",
            {
                "body_odometry": {"forward_m": 0.0, "left_m": 0.0, "yaw_deg": 0.0},
                "blocked_category": EXECUTION_UNVERIFIED,
            },
        )


def _restricted_kinematic_public_feedback(
    execution: RestrictedKinematicExecution,
) -> dict[str, Any]:
    feedback: dict[str, Any] = {
        "body_odometry": {
            "forward_m": execution.forward_m,
            "left_m": execution.left_m,
            "yaw_deg": execution.yaw_deg,
        }
    }
    report = execution.private_report
    if execution.status == "blocked":
        if report.get("static_eroded_map_guard_pass") is False:
            feedback["blocked_category"] = "navigation_path_blocked"
        elif (
            report.get("live_full_body_collision_guard_attempted") is True
            and report.get("live_full_body_collision_guard_pass") is False
        ):
            feedback["blocked_category"] = "navigation_collision_blocked"
        elif (report.get("postcondition_private") or {}).get("status") == "recovered_blocked":
            feedback["blocked_category"] = EXECUTION_UNVERIFIED
    return feedback


def _pose_record(pose: PlanarPose) -> dict[str, float]:
    return {
        "x": float(pose.x),
        "y": float(pose.y),
        "z": float(pose.z),
        "yaw_rad": float(pose.yaw),
    }


def _require_finite_pose(pose: PlanarPose, name: str) -> None:
    if not all(
        math.isfinite(float(value))
        for value in (pose.x, pose.y, pose.z, pose.yaw)
    ):
        raise RestrictedKinematicBaseError(f"{name} pose is non-finite")
