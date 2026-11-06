"""Bounded stepwise adapter for BEHAVIOR 2026's official ``GRASP`` macro.

The official Starter primitive is allowed to perform local base alignment, but
this adapter stops it before that alignment can turn into model-bypassing
navigation.  It never restores the start pose after failure: the next RGB
observation therefore exposes any physical motion that actually occurred.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Iterator


BOUNDED_OFFICIAL_GRASP_VERSION = (
    "behavior2026.bounded-official-starter-grasp/v1"
)


class OfficialGraspStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PRIMITIVE_FAILED = "primitive_failed"
    ALIGNMENT_BUDGET_EXCEEDED = "alignment_budget_exceeded"
    STEP_BUDGET_EXCEEDED = "step_budget_exceeded"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class PlanarBasePose:
    x: float
    y: float
    yaw_rad: float

    def __post_init__(self) -> None:
        for name in ("x", "y", "yaw_rad"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, number)


@dataclass(frozen=True, slots=True)
class BoundedOfficialGraspConfig:
    """Limits for local automatic alignment inside one high-level grab."""

    maximum_base_translation_m: float = 0.35
    maximum_base_yaw_rad: float = math.radians(30.0)
    maximum_simulator_steps: int = 1800
    primitive_attempts: int = 3

    def __post_init__(self) -> None:
        for name in (
            "maximum_base_translation_m",
            "maximum_base_yaw_rad",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            number = float(value)
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, number)
        for name in ("maximum_simulator_steps", "primitive_attempts"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class OfficialGraspTick:
    status: OfficialGraspStatus
    simulator_steps: int
    base_translation_m: float
    base_yaw_rad: float
    error_type: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.simulator_steps not in (0, 1):
            raise ValueError("one macro tick must advance zero or one step")
        for name in ("base_translation_m", "base_yaw_rad"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def done(self) -> bool:
        return self.status is not OfficialGraspStatus.RUNNING


@dataclass(slots=True)
class _ActiveOfficialGrasp:
    target_token: Any = field(repr=False)
    hand: str
    generator: Iterator[Any] = field(repr=False)
    start_pose: PlanarBasePose
    simulator_steps: int = 0
    closed: bool = False


class BoundedOfficialGraspDriver:
    """Consume an official Starter ``GRASP`` generator one action at a time."""

    api_contract_version = BOUNDED_OFFICIAL_GRASP_VERSION

    def __init__(
        self,
        *,
        generator_factory: Callable[[Any, int], Iterable[Any]],
        command_step: Callable[[Any, str], int],
        base_pose_provider: Callable[[], PlanarBasePose],
        held_object_provider: Callable[[], Any],
        hold_safe: Callable[[], Any],
        config: BoundedOfficialGraspConfig | None = None,
    ) -> None:
        for value, name in (
            (generator_factory, "generator_factory"),
            (command_step, "command_step"),
            (base_pose_provider, "base_pose_provider"),
            (held_object_provider, "held_object_provider"),
            (hold_safe, "hold_safe"),
        ):
            if not callable(value):
                raise TypeError(f"{name} must be callable")
        active_config = config or BoundedOfficialGraspConfig()
        if not isinstance(active_config, BoundedOfficialGraspConfig):
            raise TypeError("config must be BoundedOfficialGraspConfig")
        self._generator_factory = generator_factory
        self._command_step = command_step
        self._base_pose_provider = base_pose_provider
        self._held_object_provider = held_object_provider
        self._hold_safe = hold_safe
        self.config = active_config

    def begin(self, *, target_token: Any, hand: str) -> _ActiveOfficialGrasp:
        if target_token is None:
            raise ValueError("target_token must not be None")
        if hand not in {"left", "right"}:
            raise ValueError("hand must be left or right")
        start = self._base_pose_provider()
        if not isinstance(start, PlanarBasePose):
            raise TypeError("base_pose_provider must return PlanarBasePose")
        generator = iter(
            self._generator_factory(
                target_token,
                self.config.primitive_attempts,
            )
        )
        return _ActiveOfficialGrasp(
            target_token=target_token,
            hand=hand,
            generator=generator,
            start_pose=start,
        )

    def tick(self, state: _ActiveOfficialGrasp) -> OfficialGraspTick:
        self._require_open(state)
        translation, yaw = self._displacement(state)
        if self._outside_alignment_budget(translation, yaw):
            return self._terminate(
                state,
                OfficialGraspStatus.ALIGNMENT_BUDGET_EXCEEDED,
                simulator_steps=0,
            )
        try:
            action = next(state.generator)
        except StopIteration:
            succeeded = _same_simulator_object(
                self._held_object_provider(),
                state.target_token,
            )
            return self._terminate(
                state,
                (
                    OfficialGraspStatus.SUCCEEDED
                    if succeeded
                    else OfficialGraspStatus.PRIMITIVE_FAILED
                ),
                simulator_steps=0,
                hold=False,
            )
        except Exception as exc:
            return self._terminate(
                state,
                OfficialGraspStatus.PRIMITIVE_FAILED,
                simulator_steps=0,
                error_type=type(exc).__name__,
            )

        # Pulling once more is mutation-free and lets a generator that used
        # exactly the admitted number of simulator steps report completion.
        # If another action is yielded, close it without executing that step.
        if state.simulator_steps >= self.config.maximum_simulator_steps:
            return self._terminate(
                state,
                OfficialGraspStatus.STEP_BUDGET_EXCEEDED,
                simulator_steps=0,
            )

        result = self._command_step(action, "official_grasp")
        if isinstance(result, bool) or result != 1:
            raise RuntimeError("command_step must advance exactly one step")
        state.simulator_steps += 1
        translation, yaw = self._displacement(state)
        if self._outside_alignment_budget(translation, yaw):
            return self._terminate(
                state,
                OfficialGraspStatus.ALIGNMENT_BUDGET_EXCEEDED,
                simulator_steps=1,
            )
        return OfficialGraspTick(
            status=OfficialGraspStatus.RUNNING,
            simulator_steps=1,
            base_translation_m=translation,
            base_yaw_rad=yaw,
        )

    def abort(self, state: _ActiveOfficialGrasp) -> OfficialGraspTick:
        self._require_open(state)
        return self._terminate(
            state,
            OfficialGraspStatus.ABORTED,
            simulator_steps=0,
        )

    def _terminate(
        self,
        state: _ActiveOfficialGrasp,
        status: OfficialGraspStatus,
        *,
        simulator_steps: int,
        hold: bool = True,
        error_type: str | None = None,
    ) -> OfficialGraspTick:
        if hold:
            self._hold_safe()
        close = getattr(state.generator, "close", None)
        if callable(close):
            close()
        state.closed = True
        translation, yaw = self._displacement(state)
        return OfficialGraspTick(
            status=status,
            simulator_steps=simulator_steps,
            base_translation_m=translation,
            base_yaw_rad=yaw,
            error_type=error_type,
        )

    def _displacement(
        self,
        state: _ActiveOfficialGrasp,
    ) -> tuple[float, float]:
        current = self._base_pose_provider()
        if not isinstance(current, PlanarBasePose):
            raise TypeError("base_pose_provider must return PlanarBasePose")
        translation = math.hypot(
            current.x - state.start_pose.x,
            current.y - state.start_pose.y,
        )
        yaw = abs(_wrapped_angle(current.yaw_rad - state.start_pose.yaw_rad))
        return translation, yaw

    def _outside_alignment_budget(
        self,
        translation: float,
        yaw: float,
    ) -> bool:
        return (
            translation > self.config.maximum_base_translation_m
            or yaw > self.config.maximum_base_yaw_rad
        )

    @staticmethod
    def _require_open(state: _ActiveOfficialGrasp) -> None:
        if not isinstance(state, _ActiveOfficialGrasp):
            raise TypeError("state is not an official grasp token")
        if state.closed:
            raise RuntimeError("official grasp token is already closed")


def _wrapped_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _same_simulator_object(held: Any, target: Any) -> bool:
    """Accept the same live object or a stable proxy for the same prim."""

    if held is target:
        return True
    if held is None or target is None:
        return False
    held_name = getattr(held, "name", None)
    target_name = getattr(target, "name", None)
    held_path = getattr(held, "prim_path", None)
    target_path = getattr(target, "prim_path", None)
    return (
        isinstance(held_name, str)
        and bool(held_name)
        and held_name == target_name
        and isinstance(held_path, str)
        and bool(held_path)
        and held_path == target_path
    )
