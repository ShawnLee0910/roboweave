"""Public episode and task-adapter boundary.

This module deliberately has no OmniGibson imports.  A TaskAdapter may use
private simulator state internally, but the only value it can return to a
policy is the narrow, validated :class:`PublicTaskContext`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


_FORBIDDEN_PUBLIC_KEY_FRAGMENTS = (
    "addressable_object",
    "goal_state",
    "goal_status",
    "goal_truth",
    "aabb",
    "object_scope",
    "oracle_path",
    "oracle_route",
    "prim_path",
    "resolved_object",
    "simulator_object",
    "success_truth",
    "verifier_result",
)


class PublicBoundaryError(ValueError):
    """Raised when private evaluator data crosses the public task boundary."""


def _validated_public_metadata(
    value: Mapping[str, Any],
    *,
    path: str = "metadata",
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicBoundaryError(f"{path} must be a mapping")

    frozen: dict[str, Any] = {}
    for raw_key, item in value.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise PublicBoundaryError(f"{path} keys must be non-empty strings")
        key = raw_key.casefold()
        if any(fragment in key for fragment in _FORBIDDEN_PUBLIC_KEY_FRAGMENTS):
            raise PublicBoundaryError(
                f"{path}.{raw_key} is private evaluator information"
            )
        item_path = f"{path}.{raw_key}"
        if isinstance(item, Mapping):
            frozen[raw_key] = _validated_public_metadata(item, path=item_path)
        elif isinstance(item, (list, tuple)):
            frozen[raw_key] = tuple(
                _validated_public_metadata(element, path=f"{item_path}[{index}]")
                if isinstance(element, Mapping)
                else _public_scalar(element, f"{item_path}[{index}]")
                for index, element in enumerate(item)
            )
        else:
            frozen[raw_key] = _public_scalar(item, item_path)
    return MappingProxyType(frozen)


def _public_scalar(value: Any, path: str) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise PublicBoundaryError(
        f"{path} must contain only JSON scalar, sequence, or mapping values"
    )


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    """Frozen public identity and instruction for one episode."""

    episode_id: str
    task_id: str
    split: str
    instance_id: str
    seed: int
    instruction: str
    action_profile: str
    robot_profile: str = "r1pro"
    observation_profile: str = "r1pro_rgbd_fullres"
    model_profile: str = "qwen3_8_max"
    action_schema_version: str = "behavior-hl-r1pro-esi-v1"

    def __post_init__(self) -> None:
        for field_name in (
            "episode_id",
            "task_id",
            "split",
            "instance_id",
            "instruction",
            "action_profile",
            "robot_profile",
            "observation_profile",
            "model_profile",
            "action_schema_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


@dataclass(frozen=True, slots=True)
class PublicTaskContext:
    """The complete task information exposed to an observation-only policy."""

    episode: EpisodeSpec
    enabled_actions: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actions = tuple(self.enabled_actions)
        if not actions:
            raise ValueError("enabled_actions must not be empty")
        if any(not isinstance(name, str) or not name for name in actions):
            raise ValueError("enabled_actions must contain non-empty strings")
        if len(set(actions)) != len(actions):
            raise ValueError("enabled_actions must not contain duplicates")
        object.__setattr__(self, "enabled_actions", actions)
        object.__setattr__(
            self,
            "metadata",
            _validated_public_metadata(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the public prompt/manifest representation."""

        return {
            "episode_id": self.episode.episode_id,
            "task_id": self.episode.task_id,
            "split": self.episode.split,
            "instance_id": self.episode.instance_id,
            "seed": self.episode.seed,
            "instruction": self.episode.instruction,
            "action_profile": self.episode.action_profile,
            "robot_profile": self.episode.robot_profile,
            "observation_profile": self.episode.observation_profile,
            "model_profile": self.episode.model_profile,
            "action_schema_version": self.episode.action_schema_version,
            "enabled_actions": list(self.enabled_actions),
            "metadata": _thaw(self.metadata),
        }


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


class TaskAdapter(ABC):
    """Task-specific producer of a narrow public policy context.

    Resolved simulator objects, BDDL scope, goal truth, and verifier results
    must remain behind the evaluator boundary and must never be returned from
    :meth:`public_context`.
    """

    def __init__(self, episode: EpisodeSpec) -> None:
        self._episode = episode

    @property
    def episode(self) -> EpisodeSpec:
        return self._episode

    @abstractmethod
    def enabled_actions(self) -> tuple[str, ...]:
        """Return registry action names enabled for this episode."""

    def public_metadata(self) -> Mapping[str, Any]:
        """Return optional observation-independent public task metadata."""

        return {}

    def public_context(self) -> PublicTaskContext:
        """Build and validate the only task payload exposed to the policy."""

        return PublicTaskContext(
            episode=self.episode,
            enabled_actions=self.enabled_actions(),
            metadata=self.public_metadata(),
        )
