"""Public robot-observation adapter boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


_FORBIDDEN_OBSERVATION_KEY_FRAGMENTS = (
    "addressable_object",
    "aabb",
    "bddl_scope",
    "goal_state",
    "goal_truth",
    "occupancy_map",
    "object_scope",
    "oracle_path",
    "oracle_route",
    "prim_path",
    "resolved_object",
    "seg_instance",
    "success_truth",
    "simulator_object",
    "task_scope",
    "trav_map",
    "world_pose",
    "verifier_result",
)


class ObservationBoundaryError(ValueError):
    """Raised when a robot adapter attempts to expose private state."""


@dataclass(frozen=True, slots=True)
class PublicRobotObservation:
    schema_version: str
    simulator_step: int
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or not self.schema_version:
            raise ValueError("schema_version must be a non-empty string")
        if (
            isinstance(self.simulator_step, bool)
            or not isinstance(self.simulator_step, int)
            or self.simulator_step < 0
        ):
            raise ValueError("simulator_step must be non-negative")
        object.__setattr__(
            self,
            "values",
            _freeze_public_value(self.values, path="observation"),
        )


class RobotObservationAdapter(ABC):
    """Convert simulator output into a frozen observation-only payload."""

    schema_version: str
    allowed_fields: frozenset[str]

    @abstractmethod
    def extract(self, simulator_observation: Any) -> Mapping[str, Any]:
        """Read only approved public sensors from the simulator observation."""

    def public_observation(
        self,
        simulator_observation: Any,
        *,
        simulator_step: int,
    ) -> PublicRobotObservation:
        values = self.extract(simulator_observation)
        if not isinstance(values, Mapping):
            raise ObservationBoundaryError(
                "observation adapter must return a mapping"
            )
        fields = set(values)
        unexpected = fields.difference(self.allowed_fields)
        if unexpected:
            raise ObservationBoundaryError(
                f"adapter exposed undeclared fields: {sorted(unexpected)}"
            )
        return PublicRobotObservation(
            schema_version=self.schema_version,
            simulator_step=simulator_step,
            values=values,
        )


def _freeze_public_value(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ObservationBoundaryError(
                    f"{path} keys must be non-empty strings"
                )
            folded = key.casefold()
            if any(
                fragment in folded
                for fragment in _FORBIDDEN_OBSERVATION_KEY_FRAGMENTS
            ):
                raise ObservationBoundaryError(
                    f"{path}.{key} is private evaluator information"
                )
            frozen[key] = _freeze_public_value(
                item,
                path=f"{path}.{key}",
            )
        return MappingProxyType(frozen)
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"O", "V", "c"}:
            raise ObservationBoundaryError(
                f"{path} has unsupported array dtype"
            )
        frozen_array = np.array(value, copy=True)
        frozen_array.setflags(write=False)
        return frozen_array
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze_public_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ObservationBoundaryError(
        f"{path} contains unsupported public value {type(value).__name__}"
    )
