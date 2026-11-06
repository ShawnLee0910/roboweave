"""Public-sensor execution helpers for local approach macros."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from behavior_hl.embodied.policy.approach import (
    APPROACH_BASE_ACTIONS,
    APPROACH_LOOK_ACTIONS,
)
from behavior_hl.embodied.policy.feedback import PublicStatus


BASE_DIRECTIONS = {
    "move_forward": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "move_backward": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    "move_left": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "move_right": np.array([0.0, -1.0, 0.0], dtype=np.float32),
    "turn_left": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "turn_right": np.array([0.0, 0.0, -1.0], dtype=np.float32),
}
BASE_VELOCITY_INDEX = {
    "move_forward": (0, 1.0),
    "move_backward": (0, -1.0),
    "move_left": (1, 1.0),
    "move_right": (1, -1.0),
    "turn_left": (2, 1.0),
    "turn_right": (2, -1.0),
}
LOOK_TRANSLATES = {
    "look_up": 0.40,
    "look_center": 0.50,
    "look_down": 0.60,
}


@dataclass(frozen=True, slots=True)
class BaseMacroConfig:
    translation_command: float = 0.20
    rotation_command: float = 0.20
    forward_slow_command: float = 0.04
    forward_slow_clearance_m: float = 0.80
    forward_hard_stop_clearance_m: float = 0.45
    translation_steps: int = 30
    forward_slow_steps: int = 90
    rotation_steps: int = 20
    stall_warmup_steps: int = 6
    stall_window_steps: int = 6
    stall_speed_threshold: float = 0.02
    minimum_completed_translation_m: float = 0.05
    minimum_partial_translation_m: float = 0.005
    minimum_completed_rotation_deg: float = 3.0
    minimum_partial_rotation_deg: float = 0.5

    def __post_init__(self) -> None:
        for field in (
            "translation_command",
            "rotation_command",
            "forward_slow_command",
            "forward_slow_clearance_m",
            "forward_hard_stop_clearance_m",
            "stall_speed_threshold",
            "minimum_completed_translation_m",
            "minimum_partial_translation_m",
            "minimum_completed_rotation_deg",
            "minimum_partial_rotation_deg",
        ):
            value = getattr(self, field)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")
        if self.forward_slow_command >= self.translation_command:
            raise ValueError(
                "forward_slow_command must be below translation_command"
            )
        if (
            self.forward_hard_stop_clearance_m
            >= self.forward_slow_clearance_m
        ):
            raise ValueError(
                "forward hard-stop clearance must be below slow clearance"
            )
        if (
            self.minimum_partial_translation_m
            >= self.minimum_completed_translation_m
        ):
            raise ValueError(
                "minimum partial translation must be below completed translation"
            )
        if (
            self.minimum_partial_rotation_deg
            >= self.minimum_completed_rotation_deg
        ):
            raise ValueError(
                "minimum partial rotation must be below completed rotation"
            )
        for field in (
            "translation_steps",
            "forward_slow_steps",
            "rotation_steps",
            "stall_warmup_steps",
            "stall_window_steps",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.forward_slow_steps <= self.translation_steps:
            raise ValueError(
                "forward_slow_steps must exceed translation_steps"
            )

    def step_budget(
        self,
        action: str,
        *,
        head_forward_clearance_m: float | None = None,
    ) -> int:
        if action not in APPROACH_BASE_ACTIONS:
            raise ValueError("action is not a base approach action")
        if action.startswith("turn_"):
            return self.rotation_steps
        if action != "move_forward" or head_forward_clearance_m is None:
            return self.translation_steps
        _validate_clearance(head_forward_clearance_m)
        if head_forward_clearance_m < self.forward_slow_clearance_m:
            return self.forward_slow_steps
        return self.translation_steps

    def maximum_step_budget(self, action: str) -> int:
        """Return the safe outer loop bound before per-step depth checks.

        Forward execution can enter slow mode after the macro starts, so its
        outer loop must permit the slow budget.  This method does not authorize
        blind execution: callers must still obtain :meth:`step_decision` before
        every simulator step.
        """

        if action not in APPROACH_BASE_ACTIONS:
            raise ValueError("action is not a base approach action")
        return (
            self.forward_slow_steps
            if action == "move_forward"
            else self.step_budget(action)
        )

    def step_decision(
        self,
        action: str,
        *,
        head_forward_clearance_m: float,
    ) -> BaseStepDecision:
        """Select one physical command from the current public depth reading."""

        if action not in APPROACH_BASE_ACTIONS:
            raise ValueError("action is not a base approach action")
        _validate_clearance(head_forward_clearance_m)
        hard_stop = (
            action == "move_forward"
            and head_forward_clearance_m
            <= self.forward_hard_stop_clearance_m
        )
        slow_mode = (
            action == "move_forward"
            and not hard_stop
            and head_forward_clearance_m
            < self.forward_slow_clearance_m
        )
        if hard_stop:
            command = np.zeros(3, dtype=np.float32)
        elif action.startswith("turn_"):
            command = BASE_DIRECTIONS[action] * self.rotation_command
        elif slow_mode:
            command = BASE_DIRECTIONS[action] * self.forward_slow_command
        else:
            command = BASE_DIRECTIONS[action] * self.translation_command
        return BaseStepDecision(
            command=command,
            step_budget=(
                self.forward_slow_steps
                if slow_mode
                else self.step_budget(action)
            ),
            slow_mode=slow_mode,
            hard_stop=hard_stop,
        )

    def command(
        self,
        action: str,
        *,
        head_forward_clearance_m: float,
    ) -> np.ndarray:
        return self.step_decision(
            action,
            head_forward_clearance_m=head_forward_clearance_m,
        ).command


@dataclass(frozen=True, slots=True)
class BaseStepDecision:
    """One depth-gated base command; no simulator or private state is used."""

    command: np.ndarray
    step_budget: int
    slow_mode: bool
    hard_stop: bool

    def __post_init__(self) -> None:
        command = np.asarray(self.command)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("command must be a finite 3-vector")
        if (
            isinstance(self.step_budget, bool)
            or not isinstance(self.step_budget, int)
            or self.step_budget <= 0
        ):
            raise ValueError("step_budget must be a positive integer")
        if self.hard_stop and np.any(command):
            raise ValueError("a hard-stop decision cannot command base motion")
        if self.hard_stop and self.slow_mode:
            raise ValueError("hard-stop and slow mode are mutually exclusive")


def classify_macro_displacement(
    action: str,
    *,
    forward_m: float,
    left_m: float,
    yaw_deg: float,
    config: BaseMacroConfig,
    contact_detected: bool = False,
    clearance_hard_stop: bool = False,
    stall_detected: bool = False,
) -> PublicStatus:
    """Classify public ego-odometry without consulting simulator truth.

    Contact, velocity stall, and the forward-depth hard stop are terminal
    safety events and therefore take precedence over apparent displacement.
    Otherwise a macro is only ``completed`` after a conservative minimum
    signed displacement; smaller measurable progress is ``motion_partial``.
    """

    if action not in APPROACH_BASE_ACTIONS:
        raise ValueError("action is not a base approach action")
    values = np.asarray([forward_m, left_m, yaw_deg], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("ego odometry must be finite")
    if not isinstance(contact_detected, (bool, np.bool_)):
        raise ValueError("contact_detected must be boolean")
    if not isinstance(clearance_hard_stop, (bool, np.bool_)):
        raise ValueError("clearance_hard_stop must be boolean")
    if not isinstance(stall_detected, (bool, np.bool_)):
        raise ValueError("stall_detected must be boolean")
    if contact_detected or clearance_hard_stop or stall_detected:
        return PublicStatus.BLOCKED

    if action.startswith("turn_"):
        projected = float(
            yaw_deg if action == "turn_left" else -yaw_deg
        )
        completed = config.minimum_completed_rotation_deg
        partial = config.minimum_partial_rotation_deg
    else:
        index, sign = BASE_VELOCITY_INDEX[action]
        projected = float(sign * (forward_m, left_m)[index])
        completed = config.minimum_completed_translation_m
        partial = config.minimum_partial_translation_m
    if projected >= completed:
        return PublicStatus.COMPLETED
    if projected >= partial:
        return PublicStatus.MOTION_PARTIAL
    return PublicStatus.BLOCKED


class BaseMacroProgress:
    """Classify a bounded macro using only public base velocity."""

    def __init__(self, action: str, config: BaseMacroConfig) -> None:
        if action not in APPROACH_BASE_ACTIONS:
            raise ValueError("action is not a base approach action")
        self.action = action
        self.config = config
        self.steps = 0
        self.low_speed_streak = 0
        self.slow_mode_seen = False

    def observe(
        self,
        base_qvel,
        *,
        head_forward_clearance_m: float | None = None,
        contact_detected: bool = False,
    ) -> PublicStatus | None:
        velocity = np.asarray(base_qvel, dtype=np.float32)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError("base_qvel must be a finite 3-vector")
        if not isinstance(contact_detected, (bool, np.bool_)):
            raise ValueError("contact_detected must be boolean")
        if contact_detected:
            return PublicStatus.BLOCKED
        if head_forward_clearance_m is not None:
            decision = self.config.step_decision(
                self.action,
                head_forward_clearance_m=head_forward_clearance_m,
            )
            if decision.hard_stop:
                return PublicStatus.BLOCKED
            self.slow_mode_seen = self.slow_mode_seen or decision.slow_mode
        self.steps += 1
        index, sign = BASE_VELOCITY_INDEX[self.action]
        projected_speed = float(sign * velocity[index])
        if self.steps > self.config.stall_warmup_steps:
            if projected_speed < self.config.stall_speed_threshold:
                self.low_speed_streak += 1
            else:
                self.low_speed_streak = 0
            if self.low_speed_streak >= self.config.stall_window_steps:
                return PublicStatus.BLOCKED
        budget = (
            self.config.forward_slow_steps
            if self.action == "move_forward" and self.slow_mode_seen
            else self.config.step_budget(self.action)
        )
        if self.steps >= budget:
            return PublicStatus.COMPLETED
        return None


def _validate_clearance(value: float) -> None:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError("head forward clearance must be a finite number")
    if not np.isfinite(value):
        raise ValueError("head forward clearance must be finite")


def look_translate_for_action(action: str) -> float:
    if action not in APPROACH_LOOK_ACTIONS:
        raise ValueError("action is not a look approach action")
    return LOOK_TRANSLATES[action]
