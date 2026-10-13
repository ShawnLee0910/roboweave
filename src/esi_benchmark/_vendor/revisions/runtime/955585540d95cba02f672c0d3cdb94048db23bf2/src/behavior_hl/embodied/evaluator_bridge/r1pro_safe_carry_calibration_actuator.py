"""Concrete R1Pro actuator for resident safe-carry calibration.

This bridge is intentionally narrow.  Candidate generation and live evidence
collection are injected; it neither discovers objects nor reads task / BDDL
truth.  Each physical calibration sample is produced by exactly one
``SharedCanaryControl.command_carry_action`` call and one confirmed Evaluator
step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from behavior_hl.embodied.evaluator_bridge.r1pro_joint_stall_detector import (
    notify_r1pro_command,
)
from behavior_hl.embodied.evaluator_bridge.safe_carry_calibration import (
    SafeCarryLiveSample,
)
from behavior_hl.embodied.policy.r1pro.action_layout import (
    R1PRO_ACTION_LAYOUT,
)


R1PRO_SAFE_CARRY_CALIBRATION_ACTUATOR_VERSION = (
    "behavior2026.r1pro-safe-carry-calibration-actuator/v1"
)

_EXPECTED_SLICES = {
    "base": slice(0, 3),
    "torso": slice(3, 7),
    "left_arm": slice(7, 14),
    "left_gripper": slice(14, 15),
    "right_arm": slice(15, 22),
    "right_gripper": slice(22, 23),
}


class R1ProSafeCarryCalibrationActuatorError(RuntimeError):
    """Fail-closed live-binding or accounting error."""


@dataclass(frozen=True, slots=True)
class R1ProSafeCarryStageCommand:
    """Explicit non-active-body targets for one physical carry step.

    The active arm is supplied by the separately injected compensator.  The
    base is always forced to zero and the active gripper is always forced
    closed by this adapter.
    """

    torso_target: tuple[float, ...]
    inactive_arm_target: tuple[float, ...]
    inactive_gripper_command: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "torso_target",
            _vector(self.torso_target, 4, "torso_target"),
        )
        object.__setattr__(
            self,
            "inactive_arm_target",
            _vector(
                self.inactive_arm_target,
                7,
                "inactive_arm_target",
            ),
        )
        value = _scalar(
            self.inactive_gripper_command,
            "inactive_gripper_command",
        )
        if abs(value) > 1.0:
            raise ValueError(
                "inactive_gripper_command exceeds normalized limits"
            )
        object.__setattr__(
            self,
            "inactive_gripper_command",
            value,
        )


@dataclass(frozen=True, slots=True)
class R1ProSafeCarryCalibrationPreflight:
    version: str
    action_dimension: int
    selected_hand: str
    official_ik_only: bool
    carry_command_supported: bool
    exact_step_accounting: bool


def preflight_r1pro_safe_carry_calibration_actuator(
    *,
    starter: Any,
    control: Any,
    evaluator_step: Callable[[str], Any],
    stage_command_builder: Callable[..., R1ProSafeCarryStageCommand],
    active_arm_compensator: Callable[..., Sequence[float]],
    live_sample_provider: Callable[..., SafeCarryLiveSample],
) -> R1ProSafeCarryCalibrationPreflight:
    """Perform structural admission without IK, command, step, or sampling."""

    selected_hand = getattr(starter, "arm", None)
    if selected_hand not in {"left", "right"}:
        raise R1ProSafeCarryCalibrationActuatorError(
            "initialized Starter must select left or right arm"
        )
    if not callable(
        getattr(starter, "_ik_solver_cartesian_to_joint_space", None)
    ):
        raise R1ProSafeCarryCalibrationActuatorError(
            "initialized Starter lacks official IK-only solver"
        )
    robot = getattr(starter, "robot", None)
    if robot is None or not callable(
        getattr(robot, "get_eef_pose", None)
    ):
        raise R1ProSafeCarryCalibrationActuatorError(
            "initialized Starter lacks robot.get_eef_pose"
        )
    for method in ("command_carry_action", "hold"):
        if not callable(getattr(control, method, None)):
            raise R1ProSafeCarryCalibrationActuatorError(
                f"SharedCanaryControl is missing {method}"
            )
    for name, callback in (
        ("evaluator_step", evaluator_step),
        ("stage_command_builder", stage_command_builder),
        ("active_arm_compensator", active_arm_compensator),
        ("live_sample_provider", live_sample_provider),
    ):
        if not callable(callback):
            raise R1ProSafeCarryCalibrationActuatorError(
                f"{name} must be callable"
            )
    _validate_action_layout()
    return R1ProSafeCarryCalibrationPreflight(
        version=R1PRO_SAFE_CARRY_CALIBRATION_ACTUATOR_VERSION,
        action_dimension=R1PRO_ACTION_LAYOUT.dimension,
        selected_hand=selected_hand,
        official_ik_only=True,
        carry_command_supported=True,
        exact_step_accounting=True,
    )


def build_r1pro_safe_carry_calibration_actuator(
    *,
    starter: Any,
    control: Any,
    evaluator_step: Callable[[str], Any],
    stage_command_builder: Callable[..., R1ProSafeCarryStageCommand],
    active_arm_compensator: Callable[..., Sequence[float]],
    live_sample_provider: Callable[..., SafeCarryLiveSample],
) -> "R1ProSafeCarryCalibrationActuator":
    """Build a preflighted concrete calibration actuator."""

    preflight = preflight_r1pro_safe_carry_calibration_actuator(
        starter=starter,
        control=control,
        evaluator_step=evaluator_step,
        stage_command_builder=stage_command_builder,
        active_arm_compensator=active_arm_compensator,
        live_sample_provider=live_sample_provider,
    )
    return R1ProSafeCarryCalibrationActuator(
        starter=starter,
        control=control,
        evaluator_step=evaluator_step,
        stage_command_builder=stage_command_builder,
        active_arm_compensator=active_arm_compensator,
        live_sample_provider=live_sample_provider,
        preflight=preflight,
    )


class R1ProSafeCarryCalibrationActuator:
    """Official-IK / physical-carry implementation of the calibration protocol."""

    api_contract_version = R1PRO_SAFE_CARRY_CALIBRATION_ACTUATOR_VERSION

    def __init__(
        self,
        *,
        starter: Any,
        control: Any,
        evaluator_step: Callable[[str], Any],
        stage_command_builder: Callable[..., R1ProSafeCarryStageCommand],
        active_arm_compensator: Callable[..., Sequence[float]],
        live_sample_provider: Callable[..., SafeCarryLiveSample],
        preflight: R1ProSafeCarryCalibrationPreflight | None = None,
    ) -> None:
        self._starter = starter
        self._control = control
        self._evaluator_step = evaluator_step
        self._stage_command_builder = stage_command_builder
        self._active_arm_compensator = active_arm_compensator
        self._live_sample_provider = live_sample_provider
        self.preflight = preflight or (
            preflight_r1pro_safe_carry_calibration_actuator(
                starter=starter,
                control=control,
                evaluator_step=evaluator_step,
                stage_command_builder=stage_command_builder,
                active_arm_compensator=active_arm_compensator,
                live_sample_provider=live_sample_provider,
            )
        )
        self._carry_commands_issued = 0
        self._simulator_steps_confirmed = 0
        self._poisoned = False

    @property
    def step_accounting(self) -> tuple[int, int]:
        """Return ``(carry commands issued, simulator steps confirmed)``."""

        return (
            self._carry_commands_issued,
            self._simulator_steps_confirmed,
        )

    def ik_only_valid(
        self,
        *,
        hand: str,
        eef_pose: tuple[tuple[float, ...], tuple[float, ...]],
    ) -> bool:
        """Call official Starter IK exactly once without commanding physics."""

        self._require_hand(hand)
        self._require_operational()
        position, quaternion = _pose(eef_pose)
        position_reference, quaternion_reference = (
            self._starter.robot.get_eef_pose(arm=hand)
        )
        official_pose = (
            _tensor_like(position_reference, position),
            _tensor_like(quaternion_reference, quaternion),
        )
        solution = self._starter._ik_solver_cartesian_to_joint_space(
            official_pose,
            initial_joint_pos=None,
            skip_obstacle_update=False,
        )
        return solution is not None

    def step_toward(
        self,
        *,
        hand: str,
        arm_target: tuple[float, ...],
    ) -> SafeCarryLiveSample:
        return self._step(
            hand=hand,
            requested_arm_target=arm_target,
            phase="retract",
            validation_stage=None,
        )

    def step_g3b(
        self,
        *,
        hand: str,
        arm_target: tuple[float, ...],
        stage: str,
    ) -> SafeCarryLiveSample:
        if not isinstance(stage, str) or not stage:
            raise R1ProSafeCarryCalibrationActuatorError(
                "G3b stage must be a non-empty string"
            )
        return self._step(
            hand=hand,
            requested_arm_target=arm_target,
            phase="g3b",
            validation_stage=stage,
        )

    def safe_hold(self, *, hand: str) -> None:
        """Issue a non-stepping SharedCanaryControl hold, even if poisoned."""

        self._require_hand(hand)
        try:
            self._control.hold()
        except Exception as exc:
            self._poisoned = True
            raise R1ProSafeCarryCalibrationActuatorError(
                "SharedCanaryControl safe hold failed"
            ) from exc

    def _step(
        self,
        *,
        hand: str,
        requested_arm_target: Sequence[float],
        phase: str,
        validation_stage: str | None,
    ) -> SafeCarryLiveSample:
        self._require_hand(hand)
        self._require_operational()
        requested = _vector(
            requested_arm_target,
            7,
            "requested_arm_target",
        )
        stage_command = self._stage_command_builder(
            hand=hand,
            phase=phase,
            validation_stage=validation_stage,
        )
        if not isinstance(stage_command, R1ProSafeCarryStageCommand):
            raise R1ProSafeCarryCalibrationActuatorError(
                "stage command builder returned an invalid value"
            )
        compensated = _vector(
            self._active_arm_compensator(
                hand=hand,
                requested_arm_target=requested,
                phase=phase,
                validation_stage=validation_stage,
                stage_command=stage_command,
            ),
            7,
            "compensated_active_arm_target",
        )
        action = _carry_action(
            hand=hand,
            active_arm_target=compensated,
            stage_command=stage_command,
        )
        label = (
            "safe_carry_calibration.retract"
            if validation_stage is None
            else f"safe_carry_calibration.g3b.{validation_stage}"
        )
        notify_r1pro_command(self._live_sample_provider, action)
        self._issue_one_step(action=action, hand=hand, label=label)
        try:
            sample = self._live_sample_provider(
                hand=hand,
                validation_stage=validation_stage,
                phase=phase,
                requested_arm_target=requested,
                commanded_arm_target=compensated,
            )
            if not isinstance(sample, SafeCarryLiveSample):
                raise R1ProSafeCarryCalibrationActuatorError(
                    "live sample provider returned an invalid value"
                )
            if (
                sample.hand != hand
                or sample.validation_stage != validation_stage
            ):
                raise R1ProSafeCarryCalibrationActuatorError(
                    "live sample hand or validation stage changed"
                )
        except Exception:
            self._poisoned = True
            self._attempt_hold()
            raise
        return sample

    def _issue_one_step(
        self,
        *,
        action: np.ndarray,
        hand: str,
        label: str,
    ) -> None:
        try:
            self._control.command_carry_action(
                np.array(action, copy=True),
                active_hand=hand,
            )
        except Exception as exc:
            self._poisoned = True
            self._attempt_hold()
            raise R1ProSafeCarryCalibrationActuatorError(
                "carry command failed; physical step denied"
            ) from exc
        self._carry_commands_issued += 1
        try:
            result = self._evaluator_step(label)
        except Exception as exc:
            self._poisoned = True
            self._attempt_hold()
            raise R1ProSafeCarryCalibrationActuatorError(
                "Evaluator step failed after carry command"
            ) from exc
        if isinstance(result, bool) or not isinstance(result, int) or result != 1:
            self._poisoned = True
            self._attempt_hold()
            raise R1ProSafeCarryCalibrationActuatorError(
                "evaluator_step must confirm exactly one simulator step"
            )
        self._simulator_steps_confirmed += 1
        if self._carry_commands_issued != self._simulator_steps_confirmed:
            self._poisoned = True
            self._attempt_hold()
            raise R1ProSafeCarryCalibrationActuatorError(
                "safe-carry command/step accounting diverged"
            )

    def _require_hand(self, hand: str) -> None:
        if hand != self.preflight.selected_hand:
            raise R1ProSafeCarryCalibrationActuatorError(
                "calibration hand does not match initialized Starter arm"
            )

    def _require_operational(self) -> None:
        if self._poisoned:
            raise R1ProSafeCarryCalibrationActuatorError(
                "calibration actuator is poisoned after a live failure"
            )
        if self._carry_commands_issued != self._simulator_steps_confirmed:
            self._poisoned = True
            raise R1ProSafeCarryCalibrationActuatorError(
                "safe-carry command/step accounting is not balanced"
            )

    def _attempt_hold(self) -> None:
        try:
            self._control.hold()
        except Exception:
            pass


def _carry_action(
    *,
    hand: str,
    active_arm_target: Sequence[float],
    stage_command: R1ProSafeCarryStageCommand,
) -> np.ndarray:
    _validate_action_layout()
    inactive_hand = "right" if hand == "left" else "left"
    action = np.zeros(R1PRO_ACTION_LAYOUT.dimension, dtype=np.float32)
    action[R1PRO_ACTION_LAYOUT.slice_for("base")] = (0.0, 0.0, 0.0)
    action[R1PRO_ACTION_LAYOUT.slice_for("torso")] = (
        stage_command.torso_target
    )
    action[R1PRO_ACTION_LAYOUT.slice_for(f"{hand}_arm")] = _vector(
        active_arm_target,
        7,
        "active_arm_target",
    )
    action[
        R1PRO_ACTION_LAYOUT.slice_for(f"{inactive_hand}_arm")
    ] = stage_command.inactive_arm_target
    action[R1PRO_ACTION_LAYOUT.slice_for(f"{hand}_gripper")] = -1.0
    action[
        R1PRO_ACTION_LAYOUT.slice_for(f"{inactive_hand}_gripper")
    ] = stage_command.inactive_gripper_command
    if (
        action.shape != (23,)
        or not np.isfinite(action).all()
        or np.any(action[R1PRO_ACTION_LAYOUT.slice_for("base")] != 0.0)
        or action[
            R1PRO_ACTION_LAYOUT.slice_for(f"{hand}_gripper")
        ][0]
        != -1.0
    ):
        raise R1ProSafeCarryCalibrationActuatorError(
            "constructed carry action violates the frozen 23D contract"
        )
    return action


def _validate_action_layout() -> None:
    if R1PRO_ACTION_LAYOUT.dimension != 23:
        raise R1ProSafeCarryCalibrationActuatorError(
            "R1Pro action layout must be 23D"
        )
    actual = {
        name: R1PRO_ACTION_LAYOUT.slice_for(name)
        for name in _EXPECTED_SLICES
    }
    if actual != _EXPECTED_SLICES:
        raise R1ProSafeCarryCalibrationActuatorError(
            "R1Pro controller slices do not match the frozen contract"
        )


def _pose(
    value: Any,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise R1ProSafeCarryCalibrationActuatorError(
            "eef_pose must contain position and quaternion"
        )
    position = _vector(value[0], 3, "eef_world_position")
    quaternion = _vector(value[1], 4, "eef_world_quaternion_xyzw")
    if float(np.linalg.norm(quaternion)) <= 1e-8:
        raise R1ProSafeCarryCalibrationActuatorError(
            "eef quaternion has zero norm"
        )
    return position, quaternion


def _tensor_like(reference: Any, values: Sequence[float]) -> Any:
    if callable(getattr(reference, "new_tensor", None)):
        return reference.new_tensor(values)
    return np.asarray(values, dtype=np.float32)


def _vector(value: Any, size: int, field: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise R1ProSafeCarryCalibrationActuatorError(
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
