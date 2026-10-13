"""Public action availability derived only from public capability proxies."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol

from behavior_hl.embodied.runtime.episode import PublicTaskContext
from behavior_hl.embodied.runtime.observation import PublicRobotObservation


_CARRY_ONLY = frozenset(
    {
        "place_inside_at",
        "place_inside",
        "place_on_top",
        "place_next_to",
        "place_under",
        "place_touching",
        "release",
        "place_on_floor",
        "put_inside",
        "put_back",
    }
)
_EMPTY_HAND_ONLY = frozenset(
    {
        "grasp_at",
        "grasp",
        "pick",
        "open",
        "close",
        "toggle_on",
        "toggle_off",
    }
)


@dataclass(frozen=True, slots=True)
class ActionAvailability:
    available_actions: tuple[str, ...]
    unavailable_reasons: Mapping[str, str]

    def __post_init__(self) -> None:
        if len(set(self.available_actions)) != len(self.available_actions):
            raise ValueError("available actions must be unique")
        object.__setattr__(
            self,
            "unavailable_reasons",
            MappingProxyType(dict(self.unavailable_reasons)),
        )

    def require(self, action: str) -> None:
        if action not in self.available_actions:
            reason = self.unavailable_reasons.get(
                action,
                "action_not_available",
            )
            raise ValueError(f"action {action!r} unavailable: {reason}")

    def to_dict(self) -> dict[str, object]:
        return {
            "available_actions": list(self.available_actions),
            "unavailable_reasons": dict(self.unavailable_reasons),
        }


class ActionAvailabilityProvider(Protocol):
    def evaluate(
        self,
        task: PublicTaskContext,
        observation: PublicRobotObservation,
    ) -> ActionAvailability:
        """Return availability without reading private held or task truth."""


class EmbodimentActionAvailability:
    """Keep every configured high-level action callable on every turn.

    This is the BEHAVIOR-like policy boundary: an action remains selectable
    even when it is unlikely to work.  Reachability, contact, grasp state, and
    task truth stay inside the executor / verifier and are learned by the
    policy from subsequent onboard observations instead of an availability
    oracle.
    """

    def evaluate(
        self,
        task: PublicTaskContext,
        observation: PublicRobotObservation,
    ) -> ActionAvailability:
        if not isinstance(observation, PublicRobotObservation):
            raise TypeError("observation must be PublicRobotObservation")
        return ActionAvailability(tuple(task.enabled_actions), {})


class GraspProxyAvailability:
    """Switch grasp/place capabilities using only a calibrated public proxy."""

    proxy_field = "grasp_proxy_likely"

    def evaluate(
        self,
        task: PublicTaskContext,
        observation: PublicRobotObservation,
    ) -> ActionAvailability:
        proxy = observation.values.get(self.proxy_field, False)
        if not isinstance(proxy, bool):
            raise ValueError("grasp_proxy_likely must be boolean")
        available: list[str] = []
        unavailable: dict[str, str] = {}
        for action in task.enabled_actions:
            if proxy and action in _EMPTY_HAND_ONLY:
                unavailable[action] = "public_grasp_proxy_already_likely"
            elif not proxy and action in _CARRY_ONLY:
                unavailable[action] = "public_grasp_proxy_not_likely"
            else:
                available.append(action)
        return ActionAvailability(tuple(available), unavailable)
