"""Fail-closed zero-scene readiness checks for a resident episode."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from behavior_hl.embodied.policy.registry import ActionRegistry
from behavior_hl.embodied.runtime.dispatch import ActionPluginDispatcher


@dataclass(frozen=True, slots=True)
class ControllerCoverageReport:
    ready: bool
    enabled_action_to_controller: Mapping[str, str]
    intercepted_actions: tuple[str, ...]
    missing_actions: tuple[str, ...]
    missing_controller_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "enabled_action_to_controller",
            MappingProxyType(dict(self.enabled_action_to_controller)),
        )

    def require_ready(self) -> None:
        if not self.ready:
            raise RuntimeError(
                "resident controller coverage is incomplete: "
                f"actions={list(self.missing_actions)}, "
                f"controllers={list(self.missing_controller_keys)}"
            )


def audit_controller_coverage(
    *,
    registry: ActionRegistry,
    dispatcher: ActionPluginDispatcher,
    enabled_actions: Sequence[str],
    intercepted_actions: Sequence[str] = ("stop",),
) -> ControllerCoverageReport:
    """Prove every enabled non-intercepted action has one controller plugin."""

    enabled = tuple(enabled_actions)
    intercepted = tuple(intercepted_actions)
    if not enabled or len(set(enabled)) != len(enabled):
        raise ValueError("enabled_actions must be non-empty and unique")
    if len(set(intercepted)) != len(intercepted):
        raise ValueError("intercepted_actions must be unique")
    unknown = sorted(set(enabled).difference(registry.names))
    if unknown:
        raise ValueError(f"enabled actions are absent from registry: {unknown}")
    invalid_intercepts = sorted(set(intercepted).difference(enabled))
    if invalid_intercepts:
        raise ValueError(
            f"intercepted actions are not enabled: {invalid_intercepts}"
        )

    action_to_controller = {
        action: registry.spec(action).controller_key for action in enabled
    }
    available_keys = set(dispatcher.controller_keys)
    missing_actions = tuple(
        action
        for action in enabled
        if action not in intercepted
        and action_to_controller[action] not in available_keys
    )
    missing_keys = tuple(
        sorted({action_to_controller[action] for action in missing_actions})
    )
    return ControllerCoverageReport(
        ready=not missing_actions,
        enabled_action_to_controller=action_to_controller,
        intercepted_actions=intercepted,
        missing_actions=missing_actions,
        missing_controller_keys=missing_keys,
    )
