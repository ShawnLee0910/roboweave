"""Controller-key action plugin dispatcher with an explicit lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Protocol

from behavior_hl.embodied.policy.protocol import ActionEnvelope
from behavior_hl.embodied.policy.registry import ActionRegistry


class UnknownControllerError(LookupError):
    """Raised when an ActionSpec references an unregistered controller."""


class ControllerLifecycleError(RuntimeError):
    """Raised when a plugin violates the bounded action lifecycle."""


@dataclass(frozen=True, slots=True)
class ActionExecutionContext:
    """Minimal controller context; simulator handles stay outside public data."""

    episode_id: str
    action_index: int
    services: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if (
            isinstance(self.action_index, bool)
            or not isinstance(self.action_index, int)
            or self.action_index < 0
        ):
            raise ValueError("action_index must be non-negative")
        object.__setattr__(self, "services", MappingProxyType(dict(self.services)))


@dataclass(frozen=True, slots=True)
class ActionPluginTick:
    """One bounded simulator advance made by an action plugin."""

    done: bool
    simulator_steps: int
    public_feedback: Mapping[str, Any] = field(default_factory=dict)
    capture_observation: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.simulator_steps, bool)
            or not isinstance(self.simulator_steps, int)
            or self.simulator_steps < 0
        ):
            raise ValueError("simulator_steps must be a non-negative integer")
        object.__setattr__(
            self,
            "public_feedback",
            MappingProxyType(dict(self.public_feedback)),
        )
        if not isinstance(self.capture_observation, bool):
            raise ValueError("capture_observation must be boolean")


@dataclass(frozen=True, slots=True)
class ActionPluginResult:
    """Final public outcome of a plugin execution or abort."""

    status: str
    public_feedback: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("status must be a non-empty string")
        object.__setattr__(
            self,
            "public_feedback",
            MappingProxyType(dict(self.public_feedback)),
        )


class ActionPlugin(Protocol):
    """Controller implementation selected only through ``controller_key``."""

    controller_key: str

    def begin(
        self,
        action: ActionEnvelope,
        context: ActionExecutionContext,
    ) -> Any:
        """Create private execution state without advancing the simulator."""

    def tick(
        self,
        state: Any,
        context: ActionExecutionContext,
    ) -> ActionPluginTick:
        """Advance the bounded controller by one implementation-defined tick."""

    def finish(
        self,
        state: Any,
        context: ActionExecutionContext,
    ) -> ActionPluginResult:
        """Produce the final public result after a tick reports ``done``."""

    def abort(
        self,
        state: Any,
        context: ActionExecutionContext,
        reason: str,
    ) -> ActionPluginResult:
        """Hold/recover safely and return a typed public failure result."""


@dataclass(slots=True)
class ActiveAction:
    """Private lifecycle token returned by :meth:`ActionPluginDispatcher.begin`."""

    action: ActionEnvelope
    context: ActionExecutionContext
    plugin: ActionPlugin = field(repr=False)
    state: Any = field(repr=False)
    max_simulator_steps: int
    simulator_steps: int = 0
    capture_observation_requested: bool = False
    capture_observation_count: int = 0
    closed: bool = False


class ActionPluginDispatcher:
    """Dispatch registry actions without branching on the action name."""

    def __init__(self, plugins: Iterable[ActionPlugin]) -> None:
        by_key: dict[str, ActionPlugin] = {}
        for plugin in plugins:
            key = plugin.controller_key
            if not isinstance(key, str) or not key:
                raise ValueError("plugin controller_key must be a non-empty string")
            if key in by_key:
                raise ValueError(f"duplicate controller plugin: {key!r}")
            by_key[key] = plugin
        if not by_key:
            raise ValueError("at least one action plugin is required")
        self._plugins = by_key

    @property
    def controller_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))

    def plugin_for_controller(self, controller_key: str) -> ActionPlugin:
        """Return one registered plugin for private runtime maintenance.

        This is intentionally keyed by the registry's controller key rather
        than an action name.  Resident runners use it to refresh private
        controller references after a state-changing macro without reaching
        into the dispatcher's implementation dictionary.
        """

        if not isinstance(controller_key, str) or not controller_key:
            raise ValueError("controller_key must be a non-empty string")
        try:
            return self._plugins[controller_key]
        except KeyError as exc:
            raise UnknownControllerError(
                f"no plugin registered for controller_key={controller_key!r}"
            ) from exc

    def begin(
        self,
        registry: ActionRegistry,
        action: ActionEnvelope,
        context: ActionExecutionContext,
    ) -> ActiveAction:
        spec = registry.spec(action.action)
        try:
            plugin = self._plugins[spec.controller_key]
        except KeyError as exc:
            raise UnknownControllerError(
                f"no plugin registered for controller_key={spec.controller_key!r}"
            ) from exc
        state = plugin.begin(action, context)
        return ActiveAction(
            action=action,
            context=context,
            plugin=plugin,
            state=state,
            max_simulator_steps=spec.max_simulator_steps,
        )

    def tick(self, active: ActiveAction) -> ActionPluginResult | None:
        self._require_open(active)
        tick = active.plugin.tick(active.state, active.context)
        if not isinstance(tick, ActionPluginTick):
            raise ControllerLifecycleError("plugin tick returned an invalid value")
        active.simulator_steps += tick.simulator_steps
        active.capture_observation_requested = tick.capture_observation
        if tick.capture_observation:
            active.capture_observation_count += 1
        if active.simulator_steps > active.max_simulator_steps:
            return self.abort(active, "simulator_step_budget_exceeded")
        if not tick.done:
            return None

        result = active.plugin.finish(active.state, active.context)
        if not isinstance(result, ActionPluginResult):
            raise ControllerLifecycleError("plugin finish returned an invalid value")
        active.closed = True
        return result

    def abort(self, active: ActiveAction, reason: str) -> ActionPluginResult:
        self._require_open(active)
        if not isinstance(reason, str) or not reason:
            raise ValueError("abort reason must be a non-empty string")
        result = active.plugin.abort(
            active.state,
            active.context,
            reason,
        )
        if not isinstance(result, ActionPluginResult):
            raise ControllerLifecycleError("plugin abort returned an invalid value")
        active.closed = True
        return result

    @staticmethod
    def _require_open(active: ActiveAction) -> None:
        if not isinstance(active, ActiveAction):
            raise TypeError("active must be an ActiveAction")
        if active.closed:
            raise ControllerLifecycleError("action lifecycle is already closed")
