"""Bounded stepwise adapter for BEHAVIOR 2026's official PLACE_INSIDE.

The official Starter primitive may make a small local base adjustment, but
the driver prevents that adjustment from becoming hidden navigation.  It
never restores the starting pose: any partial physical motion remains visible
in the next onboard observation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Iterator

from behavior_hl.embodied.evaluator_bridge.bounded_official_grasp import (
    PlanarBasePose,
)


BOUNDED_OFFICIAL_PLACE_VERSION = (
    "behavior2026.bounded-official-starter-place-inside/v1"
)


class OfficialPlaceStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PRIMITIVE_FAILED = "primitive_failed"
    ALIGNMENT_BUDGET_EXCEEDED = "alignment_budget_exceeded"
    STEP_BUDGET_EXCEEDED = "step_budget_exceeded"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class BoundedOfficialPlaceConfig:
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
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class OfficialPlaceTick:
    status: OfficialPlaceStatus
    simulator_steps: int
    base_translation_m: float
    base_yaw_rad: float
    release_observed: bool = False
    error_type: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.simulator_steps not in (0, 1):
            raise ValueError("one macro tick must advance zero or one step")
        if not isinstance(self.release_observed, bool):
            raise ValueError("release_observed must be boolean")
        for name in ("base_translation_m", "base_yaw_rad"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def done(self) -> bool:
        return self.status is not OfficialPlaceStatus.RUNNING


@dataclass(slots=True)
class _ActiveOfficialPlace:
    container_token: Any = field(repr=False)
    payload_token: Any = field(repr=False)
    hand: str
    generator: Iterator[Any] = field(repr=False)
    start_pose: PlanarBasePose
    simulator_steps: int = 0
    release_observed: bool = False
    closed: bool = False


class BoundedOfficialPlaceDriver:
    """Consume an official PLACE_INSIDE generator one action at a time."""

    api_contract_version = BOUNDED_OFFICIAL_PLACE_VERSION

    def __init__(
        self,
        *,
        generator_factory: Callable[[Any, int], Iterable[Any]],
        command_step: Callable[[Any, str], int],
        base_pose_provider: Callable[[], PlanarBasePose],
        held_object_provider: Callable[[], Any],
        inside_checker: Callable[[Any, Any], bool],
        hold_safe: Callable[[], Any],
        config: BoundedOfficialPlaceConfig | None = None,
    ) -> None:
        for value, name in (
            (generator_factory, "generator_factory"),
            (command_step, "command_step"),
            (base_pose_provider, "base_pose_provider"),
            (held_object_provider, "held_object_provider"),
            (inside_checker, "inside_checker"),
            (hold_safe, "hold_safe"),
        ):
            if not callable(value):
                raise TypeError(f"{name} must be callable")
        active_config = config or BoundedOfficialPlaceConfig()
        if not isinstance(active_config, BoundedOfficialPlaceConfig):
            raise TypeError("config must be BoundedOfficialPlaceConfig")
        self._generator_factory = generator_factory
        self._command_step = command_step
        self._base_pose_provider = base_pose_provider
        self._held_object_provider = held_object_provider
        self._inside_checker = inside_checker
        self._hold_safe = hold_safe
        self.config = active_config

    def begin(self, *, container_token: Any, hand: str) -> _ActiveOfficialPlace:
        if container_token is None:
            raise ValueError("container_token must not be None")
        if hand not in {"left", "right"}:
            raise ValueError("hand must be left or right")
        payload = self._held_object_provider()
        if payload is None:
            raise ValueError("PLACE_INSIDE requires an object in hand")
        start = self._base_pose_provider()
        if not isinstance(start, PlanarBasePose):
            raise TypeError("base_pose_provider must return PlanarBasePose")
        generator = iter(
            self._generator_factory(
                container_token,
                self.config.primitive_attempts,
            )
        )
        return _ActiveOfficialPlace(
            container_token=container_token,
            payload_token=payload,
            hand=hand,
            generator=generator,
            start_pose=start,
        )

    def tick(self, state: _ActiveOfficialPlace) -> OfficialPlaceTick:
        self._require_open(state)
        translation, yaw = self._displacement(state)
        if self._outside_alignment_budget(translation, yaw):
            return self._terminate(
                state,
                OfficialPlaceStatus.ALIGNMENT_BUDGET_EXCEEDED,
                simulator_steps=0,
            )
        try:
            action = next(state.generator)
        except StopIteration:
            try:
                succeeded = (
                    self._held_object_provider() is None
                    and bool(
                        self._inside_checker(
                            state.payload_token,
                            state.container_token,
                        )
                    )
                )
            except Exception as exc:
                return self._terminate(
                    state,
                    OfficialPlaceStatus.PRIMITIVE_FAILED,
                    simulator_steps=0,
                    error_type=type(exc).__name__,
                )
            return self._terminate(
                state,
                (
                    OfficialPlaceStatus.SUCCEEDED
                    if succeeded
                    else OfficialPlaceStatus.PRIMITIVE_FAILED
                ),
                simulator_steps=0,
                hold=not succeeded,
            )
        except Exception as exc:
            return self._terminate(
                state,
                OfficialPlaceStatus.PRIMITIVE_FAILED,
                simulator_steps=0,
                error_type=type(exc).__name__,
            )

        if state.simulator_steps >= self.config.maximum_simulator_steps:
            return self._terminate(
                state,
                OfficialPlaceStatus.STEP_BUDGET_EXCEEDED,
                simulator_steps=0,
            )
        result = self._command_step(action, "official_place_inside")
        if isinstance(result, bool) or result != 1:
            raise RuntimeError("command_step must advance exactly one step")
        state.simulator_steps += 1
        if self._held_object_provider() is None:
            state.release_observed = True
        translation, yaw = self._displacement(state)
        if self._outside_alignment_budget(translation, yaw):
            return self._terminate(
                state,
                OfficialPlaceStatus.ALIGNMENT_BUDGET_EXCEEDED,
                simulator_steps=1,
            )
        return OfficialPlaceTick(
            status=OfficialPlaceStatus.RUNNING,
            simulator_steps=1,
            base_translation_m=translation,
            base_yaw_rad=yaw,
            release_observed=state.release_observed,
        )

    def abort(self, state: _ActiveOfficialPlace) -> OfficialPlaceTick:
        self._require_open(state)
        return self._terminate(
            state,
            OfficialPlaceStatus.ABORTED,
            simulator_steps=0,
        )

    def _terminate(
        self,
        state: _ActiveOfficialPlace,
        status: OfficialPlaceStatus,
        *,
        simulator_steps: int,
        hold: bool = True,
        error_type: str | None = None,
    ) -> OfficialPlaceTick:
        if hold:
            self._hold_safe()
        close = getattr(state.generator, "close", None)
        if callable(close):
            close()
        state.closed = True
        translation, yaw = self._displacement(state)
        return OfficialPlaceTick(
            status=status,
            simulator_steps=simulator_steps,
            base_translation_m=translation,
            base_yaw_rad=yaw,
            release_observed=state.release_observed,
            error_type=error_type,
        )

    def _displacement(self, state: _ActiveOfficialPlace) -> tuple[float, float]:
        current = self._base_pose_provider()
        if not isinstance(current, PlanarBasePose):
            raise TypeError("base_pose_provider must return PlanarBasePose")
        return (
            math.hypot(
                current.x - state.start_pose.x,
                current.y - state.start_pose.y,
            ),
            abs(_wrapped_angle(current.yaw_rad - state.start_pose.yaw_rad)),
        )

    def _outside_alignment_budget(self, translation: float, yaw: float) -> bool:
        return (
            translation > self.config.maximum_base_translation_m
            or yaw > self.config.maximum_base_yaw_rad
        )

    @staticmethod
    def _require_open(state: _ActiveOfficialPlace) -> None:
        if not isinstance(state, _ActiveOfficialPlace):
            raise TypeError("state is not an official place token")
        if state.closed:
            raise RuntimeError("official place token is already closed")


def _wrapped_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi
