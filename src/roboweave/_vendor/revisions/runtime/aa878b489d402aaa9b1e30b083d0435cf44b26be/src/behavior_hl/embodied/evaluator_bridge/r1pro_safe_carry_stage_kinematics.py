"""R1Pro G3b stage commands and body-relative EEF compensation.

The module imports no simulator package.  It relies on an already initialized
Starter and its live R1Pro:

* ``robot.get_joint_positions()`` returns the full URDF joint vector;
* ``robot.trunk_control_idx`` contains four indices;
* ``robot.arm_control_idx[hand]`` contains seven indices;
* ``robot.joint_position_limits`` follows that same full-joint ordering;
* Starter's official ``_ik_solver_cartesian_to_joint_space`` returns
  ``[trunk(4), selected_arm(7)]`` because its
  ``_manipulation_control_idx`` concatenates those indices in that order.

The active-arm compensator keeps a frozen robot-body-relative EEF pose.  It
does not query a held object, object scope, BDDL, a map, or goal truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from behavior_hl.embodied.evaluator_bridge.r1pro_safe_carry_calibration_actuator import (
    R1ProSafeCarryStageCommand,
)
from behavior_hl.embodied.evaluator_bridge.safe_carry_calibration import (
    SafeCarryCalibrationRecord,
    SafeCarryCandidate,
    provider_from_validated_calibration,
)
from behavior_hl.embodied.runtime.plugins.carry import CarrySafetyLimits


R1PRO_SAFE_CARRY_STAGE_KINEMATICS_VERSION = (
    "behavior2026.r1pro-safe-carry-stage-kinematics/v1"
)
G3B_STAGE_SEQUENCE = ("neutral", "downward", "neutral_return")

_PUBLIC_JOINT_FIELDS = frozenset(
    {"torso_qpos", "left_arm_qpos", "right_arm_qpos"}
)


class R1ProSafeCarryStageKinematicsError(RuntimeError):
    """Fail-closed stage or kinematics error."""


@dataclass(frozen=True, slots=True)
class R1ProCarryKinematicsLimits:
    max_torso_delta_per_tick: float = 0.025
    max_arm_delta_per_tick: float = 0.08
    public_robot_sync_tolerance_rad: float = 0.01
    stage_tracking_tolerance_rad: float = 0.02
    ik_torso_tolerance_rad: float = 0.04
    candidate_arm_tolerance_rad: float = 0.08
    joint_limit_margin_rad: float = 1e-4

    def __post_init__(self) -> None:
        for name in (
            "max_torso_delta_per_tick",
            "max_arm_delta_per_tick",
            "public_robot_sync_tolerance_rad",
            "stage_tracking_tolerance_rad",
            "ik_torso_tolerance_rad",
            "candidate_arm_tolerance_rad",
            "joint_limit_margin_rad",
        ):
            value = _positive(getattr(self, name), name)
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class R1ProG3BReachabilityReport:
    """Zero-scene proof that every declared torso transition is bounded."""

    passed: bool
    samples_per_stage: int
    max_torso_delta_per_tick: float
    required_ticks_by_transition: Mapping[str, int]
    failures: tuple[str, ...]

    def require_reachable(self) -> None:
        if not self.passed:
            raise R1ProSafeCarryStageKinematicsError(
                "G3b torso stages are unreachable: "
                f"{list(self.failures)}"
            )


def audit_r1pro_g3b_stage_reachability(
    *,
    samples_per_stage: int,
    carry_limits: CarrySafetyLimits,
    kinematics_limits: R1ProCarryKinematicsLimits,
) -> R1ProG3BReachabilityReport:
    """Check declared G3b endpoints against both controller tick limits."""

    if (
        isinstance(samples_per_stage, bool)
        or not isinstance(samples_per_stage, int)
        or samples_per_stage <= 0
    ):
        raise ValueError("samples_per_stage must be a positive integer")
    if not isinstance(carry_limits, CarrySafetyLimits):
        raise TypeError("carry_limits must be CarrySafetyLimits")
    if not isinstance(kinematics_limits, R1ProCarryKinematicsLimits):
        raise TypeError(
            "kinematics_limits must be R1ProCarryKinematicsLimits"
        )
    failures: list[str] = []
    if not math.isclose(
        carry_limits.max_torso_delta_per_tick,
        kinematics_limits.max_torso_delta_per_tick,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        failures.append(
            "CarrySafetyLimits and R1ProCarryKinematicsLimits disagree "
            "on max_torso_delta_per_tick"
        )
    per_tick = min(
        carry_limits.max_torso_delta_per_tick,
        kinematics_limits.max_torso_delta_per_tick,
    )
    endpoints = (
        (
            "neutral_to_downward",
            carry_limits.neutral_torso,
            carry_limits.downward_torso,
        ),
        (
            "downward_to_neutral_return",
            carry_limits.downward_torso,
            carry_limits.neutral_torso,
        ),
    )
    required: dict[str, int] = {}
    for name, start, finish in endpoints:
        largest_delta = max(
            abs(target - current)
            for current, target in zip(start, finish)
        )
        ticks = int(math.ceil(largest_delta / per_tick - 1e-12))
        required[name] = ticks
        if ticks > samples_per_stage:
            failures.append(
                f"{name} requires {ticks} ticks but only "
                f"{samples_per_stage} samples are configured"
            )
    return R1ProG3BReachabilityReport(
        passed=not failures,
        samples_per_stage=samples_per_stage,
        max_torso_delta_per_tick=per_tick,
        required_ticks_by_transition=MappingProxyType(required),
        failures=tuple(failures),
    )


@dataclass(frozen=True, slots=True)
class _EEFAnchor:
    hand: str
    body_position: tuple[float, ...]
    body_quaternion_xyzw: tuple[float, ...]
    candidate_arm_target: tuple[float, ...]
    provenance: str
    validated: bool

    def __post_init__(self) -> None:
        if self.hand not in {"left", "right"}:
            raise ValueError("anchor hand must be left or right")
        object.__setattr__(
            self,
            "body_position",
            _vector(self.body_position, 3, "body_position"),
        )
        quaternion = _vector(
            self.body_quaternion_xyzw,
            4,
            "body_quaternion_xyzw",
        )
        if _norm(quaternion) <= 1e-8:
            raise ValueError("body EEF quaternion has zero norm")
        object.__setattr__(
            self,
            "body_quaternion_xyzw",
            quaternion,
        )
        object.__setattr__(
            self,
            "candidate_arm_target",
            _vector(
                self.candidate_arm_target,
                7,
                "candidate_arm_target",
            ),
        )
        if not isinstance(self.provenance, str) or not self.provenance:
            raise ValueError("anchor provenance must be non-empty")
        if not isinstance(self.validated, bool):
            raise ValueError("anchor validated flag must be bool")


class R1ProSafeCarryStageCommandBuilder:
    """Stateful, exact neutral/downward/neutral-return G3b builder."""

    api_contract_version = R1PRO_SAFE_CARRY_STAGE_KINEMATICS_VERSION

    def __init__(
        self,
        *,
        hand: str,
        public_joint_state: Callable[[], Mapping[str, Any]],
        samples_per_stage: int,
        inactive_gripper_command: float,
        carry_limits: CarrySafetyLimits | None = None,
        kinematics_limits: R1ProCarryKinematicsLimits | None = None,
    ) -> None:
        if hand not in {"left", "right"}:
            raise ValueError("hand must be left or right")
        if not callable(public_joint_state):
            raise TypeError("public_joint_state must be callable")
        if (
            isinstance(samples_per_stage, bool)
            or not isinstance(samples_per_stage, int)
            or samples_per_stage <= 0
        ):
            raise ValueError("samples_per_stage must be a positive integer")
        inactive_gripper = _scalar(
            inactive_gripper_command,
            "inactive_gripper_command",
        )
        if abs(inactive_gripper) > 1.0:
            raise ValueError(
                "inactive_gripper_command exceeds normalized limits"
            )
        self._hand = hand
        self._public_joint_state = public_joint_state
        self._samples_per_stage = samples_per_stage
        self._inactive_gripper_command = inactive_gripper
        self._carry_limits = carry_limits or CarrySafetyLimits()
        self._limits = kinematics_limits or R1ProCarryKinematicsLimits()
        self._expected_stage_index = 0
        self._stage_calls = 0
        self._completed_stage: str | None = None

    @property
    def stage_progress(self) -> tuple[str | None, int]:
        return self._completed_stage, self._stage_calls

    def __call__(
        self,
        *,
        hand: str,
        phase: str,
        validation_stage: str | None,
    ) -> R1ProSafeCarryStageCommand:
        self._require_hand(hand)
        state = _public_joint_state(self._public_joint_state())
        inactive_hand = "right" if hand == "left" else "left"
        inactive_arm = state[f"{inactive_hand}_arm_qpos"]
        current_torso = state["torso_qpos"]

        if phase == "retract":
            if validation_stage is not None or self._expected_stage_index:
                raise R1ProSafeCarryStageKinematicsError(
                    "retract is allowed only before the exact G3b sequence"
                )
            torso_target = current_torso
        elif phase == "g3b":
            torso_target = self._next_g3b_target(
                stage=validation_stage,
                current_torso=current_torso,
            )
        else:
            raise R1ProSafeCarryStageKinematicsError(
                f"unsupported safe-carry phase: {phase!r}"
            )
        return R1ProSafeCarryStageCommand(
            torso_target=torso_target,
            inactive_arm_target=inactive_arm,
            inactive_gripper_command=self._inactive_gripper_command,
        )

    def require_complete(self) -> None:
        if (
            self._expected_stage_index != len(G3B_STAGE_SEQUENCE) - 1
            or self._completed_stage != G3B_STAGE_SEQUENCE[-1]
            or self._stage_calls != self._samples_per_stage
        ):
            raise R1ProSafeCarryStageKinematicsError(
                "exact G3b stage sequence is incomplete"
            )
        state = _public_joint_state(self._public_joint_state())
        if not _close(
            state["torso_qpos"],
            self._carry_limits.neutral_torso,
            self._limits.stage_tracking_tolerance_rad,
        ):
            raise R1ProSafeCarryStageKinematicsError(
                "neutral_return did not physically reach neutral"
            )

    def _next_g3b_target(
        self,
        *,
        stage: str | None,
        current_torso: tuple[float, ...],
    ) -> tuple[float, ...]:
        if stage not in G3B_STAGE_SEQUENCE:
            raise R1ProSafeCarryStageKinematicsError(
                "G3b stage must be neutral, downward, or neutral_return"
            )
        expected = G3B_STAGE_SEQUENCE[self._expected_stage_index]
        if stage != expected:
            if (
                self._stage_calls == self._samples_per_stage
                and self._completed_stage == expected
                and self._expected_stage_index
                < len(G3B_STAGE_SEQUENCE) - 1
                and stage
                == G3B_STAGE_SEQUENCE[self._expected_stage_index + 1]
            ):
                previous_target = _stage_target(
                    expected,
                    self._carry_limits,
                )
                if not _close(
                    current_torso,
                    previous_target,
                    self._limits.stage_tracking_tolerance_rad,
                ):
                    raise R1ProSafeCarryStageKinematicsError(
                        f"G3b stage {expected!r} did not physically converge"
                    )
                self._expected_stage_index += 1
                self._stage_calls = 0
                expected = stage
            else:
                raise R1ProSafeCarryStageKinematicsError(
                    f"expected G3b stage {expected!r}, received {stage!r}"
                )
        if self._stage_calls >= self._samples_per_stage:
            raise R1ProSafeCarryStageKinematicsError(
                f"G3b stage {stage!r} exceeded its exact sample count"
            )

        target = _stage_target(stage, self._carry_limits)
        calls_remaining = self._samples_per_stage - self._stage_calls
        largest_delta = max(
            abs(goal - current)
            for current, goal in zip(current_torso, target)
        )
        reachable = (
            calls_remaining * self._limits.max_torso_delta_per_tick
            + self._limits.stage_tracking_tolerance_rad
        )
        if largest_delta > reachable:
            raise R1ProSafeCarryStageKinematicsError(
                f"G3b stage {stage!r} cannot reach its exact target within "
                "the remaining bounded ticks"
            )
        next_torso = _bounded_step(
            current_torso,
            target,
            self._limits.max_torso_delta_per_tick,
        )
        self._stage_calls += 1
        if self._stage_calls == self._samples_per_stage:
            if not _close(
                next_torso,
                target,
                self._limits.stage_tracking_tolerance_rad,
            ):
                raise R1ProSafeCarryStageKinematicsError(
                    f"G3b stage {stage!r} final command is not exact"
                )
            next_torso = target
            self._completed_stage = stage
        return next_torso

    def _require_hand(self, hand: str) -> None:
        if hand != self._hand:
            raise R1ProSafeCarryStageKinematicsError(
                "stage builder hand changed"
            )


class R1ProBodyRelativeEEFArmCompensator:
    """Official-IK compensation for one frozen body-relative EEF anchor."""

    api_contract_version = R1PRO_SAFE_CARRY_STAGE_KINEMATICS_VERSION

    def __init__(
        self,
        *,
        starter: Any,
        anchor: _EEFAnchor,
        kinematics_limits: R1ProCarryKinematicsLimits | None = None,
    ) -> None:
        _validate_starter_robot(starter, anchor.hand)
        self._starter = starter
        self._robot = starter.robot
        self._anchor = anchor
        self._limits = kinematics_limits or R1ProCarryKinematicsLimits()
        self._ik_calls = 0
        self._failed = False

    @property
    def anchor_provenance(self) -> tuple[str, bool]:
        return self._anchor.provenance, self._anchor.validated

    def target_for_torso_step(
        self,
        *,
        hand: str,
        torso_before: tuple[float, ...],
        torso_after: tuple[float, ...],
        arm_before: tuple[float, ...],
    ) -> tuple[float, ...]:
        """Return one bounded 7D target for ``CarrySafeLookPlugin``."""

        self._require_operational(hand)
        before_torso = _vector(torso_before, 4, "torso_before")
        after_torso = _vector(torso_after, 4, "torso_after")
        before_arm = _vector(arm_before, 7, "arm_before")
        if not _bounded_delta(
            before_torso,
            after_torso,
            self._limits.max_torso_delta_per_tick,
        ):
            raise R1ProSafeCarryStageKinematicsError(
                "torso compensation request exceeds per-tick bound"
            )
        try:
            full_joint_state, trunk_indices, arm_indices = (
                _live_robot_joint_state(self._robot, hand)
            )
            if not _close(
                full_joint_state[trunk_indices],
                before_torso,
                self._limits.public_robot_sync_tolerance_rad,
            ) or not _close(
                full_joint_state[arm_indices],
                before_arm,
                self._limits.public_robot_sync_tolerance_rad,
            ):
                raise R1ProSafeCarryStageKinematicsError(
                    "public proprioception is stale relative to live robot"
                )
            initial_joint_pos = _hypothetical_joint_state(
                self._robot.get_joint_positions(),
                trunk_indices=trunk_indices,
                torso_target=after_torso,
                arm_indices=arm_indices,
                arm_target=before_arm,
            )
            world_pose = self._world_eef_pose()
            solution = self._starter._ik_solver_cartesian_to_joint_space(
                world_pose,
                initial_joint_pos=initial_joint_pos,
                skip_obstacle_update=self._ik_calls > 0,
            )
            self._ik_calls += 1
            if solution is None:
                raise R1ProSafeCarryStageKinematicsError(
                    "official IK could not preserve body-relative EEF"
                )
            result = _array(solution, "official IK solution")
            if result.shape != (11,):
                raise R1ProSafeCarryStageKinematicsError(
                    "official R1Pro IK must return trunk(4)+arm(7)"
                )
            if not _close(
                result[:4],
                after_torso,
                self._limits.ik_torso_tolerance_rad,
            ):
                raise R1ProSafeCarryStageKinematicsError(
                    "official IK changed the requested torso target"
                )
            arm_target = tuple(float(item) for item in result[4:])
            if not _bounded_delta(
                before_arm,
                arm_target,
                self._limits.max_arm_delta_per_tick,
            ):
                raise R1ProSafeCarryStageKinematicsError(
                    "official IK active-arm delta exceeds per-tick bound"
                )
            _require_arm_within_urdf_limits(
                robot=self._robot,
                arm_indices=arm_indices,
                arm_target=arm_target,
                margin=self._limits.joint_limit_margin_rad,
            )
            return arm_target
        except Exception:
            self._failed = True
            raise

    def __call__(
        self,
        *,
        hand: str,
        requested_arm_target: tuple[float, ...],
        phase: str,
        validation_stage: str | None,
        stage_command: R1ProSafeCarryStageCommand,
    ) -> tuple[float, ...]:
        """Adapt the same IK implementation to the calibration actuator."""

        self._require_operational(hand)
        if not isinstance(stage_command, R1ProSafeCarryStageCommand):
            raise R1ProSafeCarryStageKinematicsError(
                "calibration stage command is invalid"
            )
        requested = _vector(
            requested_arm_target,
            7,
            "requested_arm_target",
        )
        if not _close(
            requested,
            self._anchor.candidate_arm_target,
            self._limits.candidate_arm_tolerance_rad,
        ):
            raise R1ProSafeCarryStageKinematicsError(
                "calibration arm target changed from the frozen candidate"
            )
        if phase == "retract":
            if validation_stage is not None:
                raise R1ProSafeCarryStageKinematicsError(
                    "retract cannot have a G3b validation stage"
                )
        elif phase == "g3b":
            if validation_stage not in G3B_STAGE_SEQUENCE:
                raise R1ProSafeCarryStageKinematicsError(
                    "calibration G3b stage is not in the exact sequence"
                )
        else:
            raise R1ProSafeCarryStageKinematicsError(
                f"unsupported calibration phase: {phase!r}"
            )
        full, trunk_indices, arm_indices = _live_robot_joint_state(
            self._robot,
            hand,
        )
        return self.target_for_torso_step(
            hand=hand,
            torso_before=tuple(float(item) for item in full[trunk_indices]),
            torso_after=stage_command.torso_target,
            arm_before=tuple(float(item) for item in full[arm_indices]),
        )

    def _world_eef_pose(self) -> tuple[Any, Any]:
        body_position, body_quaternion = (
            self._robot.get_position_orientation()
        )
        world_position = _transform_position(
            body_position,
            body_quaternion,
            self._anchor.body_position,
        )
        world_quaternion = _quat_multiply(
            body_quaternion,
            self._anchor.body_quaternion_xyzw,
        )
        position_reference, quaternion_reference = (
            self._robot.get_eef_pose(arm=self._anchor.hand)
        )
        return (
            _tensor_like(position_reference, world_position),
            _tensor_like(quaternion_reference, world_quaternion),
        )

    def _require_operational(self, hand: str) -> None:
        if hand != self._anchor.hand:
            raise R1ProSafeCarryStageKinematicsError(
                "active hand changed from the EEF anchor"
            )
        if self._failed:
            raise R1ProSafeCarryStageKinematicsError(
                "active-arm compensator is poisoned after IK failure"
            )


class ExactG3BCompletionCalibrationActuator:
    """Require the stage builder's physical endpoint on the final G3b sample."""

    def __init__(
        self,
        *,
        actuator: Any,
        stage_builder: Any,
        samples_per_stage: int,
    ) -> None:
        for name in (
            "ik_only_valid",
            "step_toward",
            "step_g3b",
            "safe_hold",
        ):
            if not callable(getattr(actuator, name, None)):
                raise TypeError(f"actuator is missing {name}")
        if not callable(getattr(stage_builder, "require_complete", None)):
            raise TypeError("stage_builder is missing require_complete")
        if (
            isinstance(samples_per_stage, bool)
            or not isinstance(samples_per_stage, int)
            or samples_per_stage <= 0
        ):
            raise ValueError("samples_per_stage must be positive")
        self._actuator = actuator
        self._stage_builder = stage_builder
        self._samples_per_stage = samples_per_stage
        self._stage_index = 0
        self._stage_samples = 0
        self._completion_verified = False
        self._poisoned = False

    def ik_only_valid(self, **kwargs: Any) -> bool:
        self._require_operational()
        return self._actuator.ik_only_valid(**kwargs)

    def step_toward(self, **kwargs: Any) -> Any:
        self._require_operational()
        return self._actuator.step_toward(**kwargs)

    def step_g3b(
        self,
        *,
        hand: str,
        arm_target: tuple[float, ...],
        stage: str,
    ) -> Any:
        self._require_operational()
        expected = G3B_STAGE_SEQUENCE[self._stage_index]
        if stage != expected:
            self._poisoned = True
            raise R1ProSafeCarryStageKinematicsError(
                f"expected completion stage {expected!r}, received {stage!r}"
            )
        if self._stage_samples >= self._samples_per_stage:
            self._poisoned = True
            raise R1ProSafeCarryStageKinematicsError(
                "completion wrapper received too many stage samples"
            )
        try:
            sample = self._actuator.step_g3b(
                hand=hand,
                arm_target=arm_target,
                stage=stage,
            )
        except Exception:
            self._poisoned = True
            raise
        self._stage_samples += 1
        if self._stage_samples == self._samples_per_stage:
            if self._stage_index == len(G3B_STAGE_SEQUENCE) - 1:
                try:
                    self._stage_builder.require_complete()
                except Exception:
                    self._poisoned = True
                    raise
                self._completion_verified = True
            else:
                self._stage_index += 1
                self._stage_samples = 0
        return sample

    def safe_hold(self, **kwargs: Any) -> None:
        self._actuator.safe_hold(**kwargs)
        if (
            self._stage_index == len(G3B_STAGE_SEQUENCE) - 1
            and self._stage_samples == self._samples_per_stage
            and not self._completion_verified
        ):
            self._poisoned = True
            raise R1ProSafeCarryStageKinematicsError(
                "final neutral_return endpoint was not verified"
            )

    def _require_operational(self) -> None:
        if self._poisoned:
            raise R1ProSafeCarryStageKinematicsError(
                "completion actuator is poisoned"
            )


def build_validated_r1pro_carry_compensator(
    *,
    starter: Any,
    calibration: SafeCarryCalibrationRecord,
    kinematics_limits: R1ProCarryKinematicsLimits | None = None,
) -> R1ProBodyRelativeEEFArmCompensator:
    """Build the scored/runtime compensator only from a passing record."""

    provider_from_validated_calibration(calibration)
    if tuple(item.stage for item in calibration.stage_records) != (
        G3B_STAGE_SEQUENCE
    ):
        raise R1ProSafeCarryStageKinematicsError(
            "validated calibration lacks the exact G3b stage sequence"
        )
    anchor = _EEFAnchor(
        hand=calibration.candidate.hand,
        body_position=calibration.body_eef_position,
        body_quaternion_xyzw=calibration.body_eef_quaternion_xyzw,
        candidate_arm_target=calibration.candidate.arm_target,
        provenance=f"validated:{calibration.sha256}",
        validated=True,
    )
    return R1ProBodyRelativeEEFArmCompensator(
        starter=starter,
        anchor=anchor,
        kinematics_limits=kinematics_limits,
    )


def build_candidate_r1pro_calibration_compensator(
    *,
    starter: Any,
    candidate: SafeCarryCandidate,
    kinematics_limits: R1ProCarryKinematicsLimits | None = None,
) -> R1ProBodyRelativeEEFArmCompensator:
    """Derive an unscored anchor for the bootstrap calibration segment only."""

    if not isinstance(candidate, SafeCarryCandidate):
        raise TypeError("candidate must be SafeCarryCandidate")
    _validate_starter_robot(starter, candidate.hand)
    body_position, body_quaternion = (
        starter.robot.get_position_orientation()
    )
    relative_position = _inverse_transform_position(
        body_position,
        body_quaternion,
        candidate.eef_world_position,
    )
    relative_quaternion = _quat_multiply(
        _quat_inverse(body_quaternion),
        candidate.eef_world_quaternion_xyzw,
    )
    anchor = _EEFAnchor(
        hand=candidate.hand,
        body_position=relative_position,
        body_quaternion_xyzw=relative_quaternion,
        candidate_arm_target=candidate.arm_target,
        provenance=f"candidate:{candidate.source}",
        validated=False,
    )
    return R1ProBodyRelativeEEFArmCompensator(
        starter=starter,
        anchor=anchor,
        kinematics_limits=kinematics_limits,
    )


def _stage_target(
    stage: str,
    limits: CarrySafetyLimits,
) -> tuple[float, ...]:
    if stage == "neutral":
        return limits.neutral_torso
    if stage == "downward":
        return limits.downward_torso
    if stage == "neutral_return":
        return limits.neutral_torso
    raise R1ProSafeCarryStageKinematicsError(
        f"unknown G3b stage: {stage!r}"
    )


def _public_joint_state(value: Any) -> dict[str, tuple[float, ...]]:
    if not isinstance(value, Mapping) or set(value) != _PUBLIC_JOINT_FIELDS:
        raise R1ProSafeCarryStageKinematicsError(
            "public joint state fields do not match the frozen contract"
        )
    return {
        "torso_qpos": _vector(value["torso_qpos"], 4, "torso_qpos"),
        "left_arm_qpos": _vector(
            value["left_arm_qpos"],
            7,
            "left_arm_qpos",
        ),
        "right_arm_qpos": _vector(
            value["right_arm_qpos"],
            7,
            "right_arm_qpos",
        ),
    }


def _validate_starter_robot(starter: Any, hand: str) -> None:
    if getattr(starter, "arm", None) != hand:
        raise R1ProSafeCarryStageKinematicsError(
            "Starter selected arm does not match carry hand"
        )
    if not callable(
        getattr(starter, "_ik_solver_cartesian_to_joint_space", None)
    ):
        raise R1ProSafeCarryStageKinematicsError(
            "Starter lacks official IK-only solver"
        )
    robot = getattr(starter, "robot", None)
    for name in (
        "get_joint_positions",
        "get_position_orientation",
        "get_eef_pose",
    ):
        if robot is None or not callable(getattr(robot, name, None)):
            raise R1ProSafeCarryStageKinematicsError(
                f"R1Pro lacks {name}"
            )
    if not hasattr(robot, "trunk_control_idx") or not hasattr(
        robot,
        "arm_control_idx",
    ):
        raise R1ProSafeCarryStageKinematicsError(
            "R1Pro lacks URDF control indices"
        )


def _live_robot_joint_state(
    robot: Any,
    hand: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    full = _array(robot.get_joint_positions(), "robot joint positions")
    if full.ndim != 1:
        raise R1ProSafeCarryStageKinematicsError(
            "robot joint positions must be one-dimensional"
        )
    trunk = _indices(robot.trunk_control_idx, 4, "trunk_control_idx")
    try:
        raw_arm = robot.arm_control_idx[hand]
    except Exception as exc:
        raise R1ProSafeCarryStageKinematicsError(
            "R1Pro lacks selected-arm control indices"
        ) from exc
    arm = _indices(raw_arm, 7, f"{hand}_arm_control_idx")
    combined = np.concatenate((trunk, arm))
    if (
        len(set(int(item) for item in combined)) != 11
        or np.any(combined < 0)
        or np.any(combined >= full.size)
    ):
        raise R1ProSafeCarryStageKinematicsError(
            "R1Pro URDF control indices are invalid or overlap"
        )
    return full, trunk, arm


def _hypothetical_joint_state(
    value: Any,
    *,
    trunk_indices: np.ndarray,
    torso_target: Sequence[float],
    arm_indices: np.ndarray,
    arm_target: Sequence[float],
) -> Any:
    if callable(getattr(value, "clone", None)):
        result = value.clone()
    else:
        result = np.array(value, dtype=np.float64, copy=True)
    _assign(result, trunk_indices, torso_target)
    _assign(result, arm_indices, arm_target)
    return result


def _assign(target: Any, indices: np.ndarray, values: Sequence[float]) -> None:
    if callable(getattr(target, "new_tensor", None)):
        target[indices.tolist()] = target.new_tensor(values)
    else:
        target[indices] = np.asarray(values, dtype=np.float64)


def _require_arm_within_urdf_limits(
    *,
    robot: Any,
    arm_indices: np.ndarray,
    arm_target: Sequence[float],
    margin: float,
) -> None:
    try:
        lower_raw, upper_raw = robot.joint_position_limits
    except Exception as exc:
        raise R1ProSafeCarryStageKinematicsError(
            "R1Pro lacks URDF joint position limits"
        ) from exc
    lower = _array(lower_raw, "joint lower limits")
    upper = _array(upper_raw, "joint upper limits")
    if lower.shape != upper.shape or lower.ndim != 1:
        raise R1ProSafeCarryStageKinematicsError(
            "URDF joint limit vectors are inconsistent"
        )
    if np.any(arm_indices >= lower.size):
        raise R1ProSafeCarryStageKinematicsError(
            "URDF joint limits do not cover active arm"
        )
    target = np.asarray(arm_target, dtype=np.float64)
    if np.any(target < lower[arm_indices] + margin) or np.any(
        target > upper[arm_indices] - margin
    ):
        raise R1ProSafeCarryStageKinematicsError(
            "official IK target violates active-arm URDF limits"
        )


def _transform_position(
    body_position: Sequence[float],
    body_quaternion: Sequence[float],
    relative_position: Sequence[float],
) -> np.ndarray:
    return np.asarray(_vector(body_position, 3, "body_position")) + (
        _quat_matrix(body_quaternion)
        @ np.asarray(relative_position, dtype=np.float64)
    )


def _inverse_transform_position(
    body_position: Sequence[float],
    body_quaternion: Sequence[float],
    world_position: Sequence[float],
) -> np.ndarray:
    rotation = _quat_matrix(body_quaternion)
    return rotation.T @ (
        np.asarray(_vector(world_position, 3, "world_position"))
        - np.asarray(_vector(body_position, 3, "body_position"))
    )


def _quat_matrix(value: Sequence[float]) -> np.ndarray:
    x, y, z, w = _normalized_quaternion(value)
    return np.asarray(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ]
    )


def _quat_multiply(
    first: Sequence[float],
    second: Sequence[float],
) -> np.ndarray:
    ax, ay, az, aw = _normalized_quaternion(first)
    bx, by, bz, bw = _normalized_quaternion(second)
    return np.asarray(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _quat_inverse(value: Sequence[float]) -> np.ndarray:
    x, y, z, w = _normalized_quaternion(value)
    return np.asarray((-x, -y, -z, w))


def _normalized_quaternion(value: Sequence[float]) -> np.ndarray:
    array = np.asarray(_vector(value, 4, "quaternion"), dtype=np.float64)
    return array / float(np.linalg.norm(array))


def _tensor_like(reference: Any, values: Sequence[float]) -> Any:
    if callable(getattr(reference, "new_tensor", None)):
        return reference.new_tensor(values)
    return np.asarray(values, dtype=np.float32)


def _bounded_step(
    current: Sequence[float],
    target: Sequence[float],
    maximum_delta: float,
) -> tuple[float, ...]:
    return tuple(
        now + max(-maximum_delta, min(maximum_delta, goal - now))
        for now, goal in zip(current, target)
    )


def _bounded_delta(
    before: Sequence[float],
    after: Sequence[float],
    maximum: float,
) -> bool:
    return max(abs(a - b) for a, b in zip(before, after)) <= maximum + 1e-9


def _close(
    first: Sequence[float],
    second: Sequence[float],
    tolerance: float,
) -> bool:
    return max(abs(a - b) for a, b in zip(first, second)) <= tolerance


def _indices(value: Any, size: int, field: str) -> np.ndarray:
    array = _array(value, field)
    if array.shape != (size,) or np.any(array != np.floor(array)):
        raise R1ProSafeCarryStageKinematicsError(
            f"{field} must contain exactly {size} integer indices"
        )
    return array.astype(np.int64)


def _array(value: Any, field: str) -> np.ndarray:
    candidate = value
    if callable(getattr(candidate, "detach", None)):
        candidate = candidate.detach()
    if callable(getattr(candidate, "cpu", None)):
        candidate = candidate.cpu()
    if callable(getattr(candidate, "numpy", None)):
        candidate = candidate.numpy()
    array = np.asarray(candidate)
    if not np.issubdtype(array.dtype, np.number):
        raise R1ProSafeCarryStageKinematicsError(
            f"{field} must be numeric"
        )
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise R1ProSafeCarryStageKinematicsError(
            f"{field} must be finite"
        )
    return result


def _vector(value: Any, size: int, field: str) -> tuple[float, ...]:
    array = _array(value, field)
    if array.shape != (size,):
        raise R1ProSafeCarryStageKinematicsError(
            f"{field} must be a finite {size}-vector"
        )
    return tuple(float(item) for item in array)


def _scalar(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite")
    array = np.asarray(value, dtype=np.float64)
    if array.shape != () or not np.isfinite(array):
        raise ValueError(f"{field} must be finite")
    return float(array)


def _positive(value: Any, field: str) -> float:
    result = _scalar(value, field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(item * item for item in value))
