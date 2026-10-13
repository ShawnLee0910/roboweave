"""Stateful R1Pro command-versus-proprio joint-stall detection.

The detector is simulator-independent.  It compares the commanded torso and
arm targets in the frozen 23D R1Pro action layout with the corresponding
policy-visible joint positions in the 61D proprioception vector.  Base and
gripper commands are deliberately excluded: base obstruction is handled by
contact / progress checks and the two-finger grippers do not have a one-to-one
action-to-proprio mapping.

One command registration must be followed by exactly one post-step safety
sample.  Missing, duplicate, non-finite, or shape-changing evidence poisons
the detector and raises, so a caller cannot silently turn unavailable
tracking evidence into ``joint_stalled=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from behavior_hl.embodied.policy.hold_policy import (
    LEFT_ARM_QPOS,
    RIGHT_ARM_QPOS,
    TRUNK_QPOS,
)
from behavior_hl.embodied.policy.observations import PROPRIO_DIM
from behavior_hl.embodied.policy.r1pro.action_layout import (
    R1PRO_ACTION_LAYOUT,
)


R1PRO_JOINT_STALL_DETECTOR_VERSION = (
    "behavior2026.r1pro-command-proprio-joint-stall/v1"
)


class R1ProJointStallDetectorError(RuntimeError):
    """Fail-closed command / observation sequencing or schema error."""


@dataclass(frozen=True, slots=True)
class R1ProJointStallDiagnostics:
    """Private diagnostic state for the most recently consumed step."""

    commanded_motion: bool
    demanded_joint_count: int
    remaining_joint_count: int
    max_step_progress_rad: float
    max_remaining_error_rad: float
    consecutive_no_progress_steps: int
    joint_stalled: bool


@dataclass(frozen=True, slots=True)
class _PendingCommand:
    target: np.ndarray
    before: np.ndarray
    demanded: np.ndarray


class R1ProJointStallDetector:
    """Detect repeated no-progress torso / arm commands.

    ``stall_window_steps`` is intentionally greater than one by default so a
    single physics integration step with small movement does not immediately
    become a stall.  A step counts as no-progress only when at least one
    demanded joint remains outside ``tracking_tolerance_rad`` and *none* of
    the demanded joints moves by ``min_progress_rad`` toward its target.
    """

    api_contract_version = R1PRO_JOINT_STALL_DETECTOR_VERSION

    def __init__(
        self,
        *,
        command_deadband_rad: float = 0.01,
        tracking_tolerance_rad: float = 0.035,
        min_progress_rad: float = 0.001,
        stall_window_steps: int = 3,
    ) -> None:
        self.command_deadband_rad = _positive(
            command_deadband_rad,
            "command_deadband_rad",
        )
        self.tracking_tolerance_rad = _positive(
            tracking_tolerance_rad,
            "tracking_tolerance_rad",
        )
        self.min_progress_rad = _positive(
            min_progress_rad,
            "min_progress_rad",
        )
        if (
            isinstance(stall_window_steps, bool)
            or not isinstance(stall_window_steps, int)
            or stall_window_steps < 1
        ):
            raise ValueError("stall_window_steps must be a positive integer")
        self.stall_window_steps = stall_window_steps
        self._pending: _PendingCommand | None = None
        self._no_progress_steps = 0
        self._poisoned = False
        self._last_diagnostics: R1ProJointStallDiagnostics | None = None

    @property
    def last_diagnostics(self) -> R1ProJointStallDiagnostics | None:
        return self._last_diagnostics

    def register_command(
        self,
        action: Sequence[float],
        before_public_proprio: Sequence[float],
    ) -> None:
        """Freeze one pre-step command / proprio pair."""

        self._require_operational()
        if self._pending is not None:
            self._poison(
                "a second command was registered before post-step sampling"
            )
        try:
            action_array = _action(action)
            before = _joint_qpos(before_public_proprio)
            target = _joint_targets(action_array)
            demanded = (
                np.abs(target - before) >= self.command_deadband_rad
            )
            self._pending = _PendingCommand(
                target=np.array(target, copy=True),
                before=np.array(before, copy=True),
                demanded=np.array(demanded, copy=True),
            )
        except Exception:
            self._poisoned = True
            raise

    def observe_after_step(
        self,
        after_public_proprio: Sequence[float],
    ) -> bool:
        """Consume the pending command and return the current stall verdict."""

        self._require_operational()
        pending = self._pending
        if pending is None:
            self._poison(
                "post-step proprio was sampled without a registered command"
            )
        try:
            after = _joint_qpos(after_public_proprio)
            self._pending = None
            demanded_count = int(np.count_nonzero(pending.demanded))
            if demanded_count == 0:
                self._no_progress_steps = 0
                diagnostics = R1ProJointStallDiagnostics(
                    commanded_motion=False,
                    demanded_joint_count=0,
                    remaining_joint_count=0,
                    max_step_progress_rad=0.0,
                    max_remaining_error_rad=0.0,
                    consecutive_no_progress_steps=0,
                    joint_stalled=False,
                )
                self._last_diagnostics = diagnostics
                return False

            before_error = np.abs(pending.target - pending.before)
            after_error = np.abs(pending.target - after)
            toward_target_progress = before_error - after_error
            remaining = (
                pending.demanded
                & (after_error > self.tracking_tolerance_rad)
            )
            remaining_count = int(np.count_nonzero(remaining))
            max_progress = float(
                np.max(toward_target_progress[pending.demanded])
            )
            max_remaining = (
                float(np.max(after_error[remaining]))
                if remaining_count
                else 0.0
            )
            no_progress = (
                remaining_count > 0
                and max_progress < self.min_progress_rad
            )
            if no_progress:
                self._no_progress_steps += 1
            else:
                self._no_progress_steps = 0
            stalled = self._no_progress_steps >= self.stall_window_steps
            self._last_diagnostics = R1ProJointStallDiagnostics(
                commanded_motion=True,
                demanded_joint_count=demanded_count,
                remaining_joint_count=remaining_count,
                max_step_progress_rad=max_progress,
                max_remaining_error_rad=max_remaining,
                consecutive_no_progress_steps=self._no_progress_steps,
                joint_stalled=stalled,
            )
            return stalled
        except Exception:
            self._poisoned = True
            raise

    def _require_operational(self) -> None:
        if self._poisoned:
            raise R1ProJointStallDetectorError(
                "joint-stall detector is poisoned after invalid evidence"
            )

    def _poison(self, message: str) -> None:
        self._poisoned = True
        raise R1ProJointStallDetectorError(message)


class R1ProCommandAwarePrivateSafety:
    """Inject joint-stall evidence into an anonymous private-safety callback.

    The upstream callback remains responsible for contact and collision
    signals.  Its existing ``joint_stalled`` value is OR-ed with the detector
    verdict, never discarded.
    """

    api_contract_version = R1PRO_JOINT_STALL_DETECTOR_VERSION

    def __init__(
        self,
        *,
        upstream_private_safety: Callable[[], Mapping[str, Any]],
        public_proprio: Callable[[], Sequence[float]],
        expected_fields: frozenset[str],
        detector: R1ProJointStallDetector | None = None,
    ) -> None:
        if not callable(upstream_private_safety):
            raise TypeError("upstream_private_safety must be callable")
        if not callable(public_proprio):
            raise TypeError("public_proprio must be callable")
        if (
            not isinstance(expected_fields, frozenset)
            or "joint_stalled" not in expected_fields
            or not expected_fields
            or any(
                not isinstance(field, str) or not field
                for field in expected_fields
            )
        ):
            raise ValueError(
                "expected_fields must be a non-empty frozenset containing "
                "joint_stalled"
            )
        if detector is not None and not isinstance(
            detector,
            R1ProJointStallDetector,
        ):
            raise TypeError("detector has the wrong type")
        self._upstream = upstream_private_safety
        self._public_proprio = public_proprio
        self._expected_fields = expected_fields
        self.detector = detector or R1ProJointStallDetector()
        self._poisoned = False

    def register_command(self, action: Sequence[float]) -> None:
        if self._poisoned:
            raise R1ProJointStallDetectorError(
                "private-safety dependency is poisoned"
            )
        try:
            self.detector.register_command(
                action,
                self._public_proprio(),
            )
        except Exception:
            self._poisoned = True
            raise

    def __call__(self) -> Mapping[str, bool]:
        if self._poisoned:
            raise R1ProJointStallDetectorError(
                "private-safety dependency is poisoned"
            )
        try:
            upstream = self._upstream()
            if (
                not isinstance(upstream, Mapping)
                or set(upstream) != self._expected_fields
            ):
                raise R1ProJointStallDetectorError(
                    "upstream private-safety fields changed"
                )
            result: dict[str, bool] = {}
            for field in self._expected_fields:
                value = upstream[field]
                if not isinstance(value, bool):
                    raise R1ProJointStallDetectorError(
                        f"upstream private-safety {field} must be boolean"
                    )
                result[field] = value
            detected = self.detector.observe_after_step(
                self._public_proprio()
            )
            result["joint_stalled"] = (
                result["joint_stalled"] or detected
            )
            return result
        except Exception:
            self._poisoned = True
            raise


def notify_r1pro_command(
    private_safety_or_sample_provider: Any,
    action: Sequence[float],
) -> None:
    """Notify a command-aware dependency while preserving legacy callbacks."""

    callback = getattr(
        private_safety_or_sample_provider,
        "register_command",
        None,
    )
    if callback is not None:
        if not callable(callback):
            raise R1ProJointStallDetectorError(
                "register_command attribute is not callable"
            )
        callback(action)


def _action(value: Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.shape != (R1PRO_ACTION_LAYOUT.dimension,)
        or not np.isfinite(array).all()
    ):
        raise R1ProJointStallDetectorError(
            "R1Pro action must be a finite 23-vector"
        )
    return array


def _joint_qpos(value: Sequence[float]) -> np.ndarray:
    proprio = np.asarray(value, dtype=np.float64)
    if proprio.shape != (PROPRIO_DIM,) or not np.isfinite(proprio).all():
        raise R1ProJointStallDetectorError(
            "public R1Pro proprio must be a finite 61-vector"
        )
    return np.concatenate(
        (
            proprio[TRUNK_QPOS],
            proprio[LEFT_ARM_QPOS],
            proprio[RIGHT_ARM_QPOS],
        )
    )


def _joint_targets(action: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            action[R1PRO_ACTION_LAYOUT.slice_for("torso")],
            action[R1PRO_ACTION_LAYOUT.slice_for("left_arm")],
            action[R1PRO_ACTION_LAYOUT.slice_for("right_arm")],
        )
    )


def _positive(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a positive finite number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be a positive finite number")
    return result
