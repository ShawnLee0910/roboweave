"""Action-name-agnostic policy loop for one resident episode."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from behavior_hl.embodied.policy.protocol import ActionEnvelope
from behavior_hl.embodied.policy.registry import ActionRegistry
from behavior_hl.embodied.runtime.dispatch import (
    ActionExecutionContext,
    ActionPluginDispatcher,
    ActionPluginResult,
)
from behavior_hl.embodied.runtime.availability import (
    ActionAvailabilityProvider,
)
from behavior_hl.embodied.runtime.episode import (
    EpisodeSpec,
    PublicTaskContext,
)
from behavior_hl.embodied.runtime.observation import PublicRobotObservation
from behavior_hl.embodied.runtime.verifier import (
    PrivateVerifier,
    PrivateVerifierResult,
)


class PublicPolicy(Protocol):
    """Observation-only policy; private verifier output is never an argument."""

    def decide(
        self,
        task: PublicTaskContext,
        observation: PublicRobotObservation,
        history: tuple[Mapping[str, Any], ...],
    ) -> Mapping[str, Any] | ActionEnvelope:
        """Return exactly one action envelope or parseable action mapping."""


@dataclass(frozen=True, slots=True)
class PolicyLoopResult:
    action: ActionEnvelope
    public_action_result: ActionPluginResult
    simulator_steps: int
    private_verifier_result: PrivateVerifierResult
    official_success: bool


class PolicyLoop:
    """Execute one policy action without branching on its action name."""

    def __init__(
        self,
        *,
        episode: EpisodeSpec,
        task: PublicTaskContext,
        registry: ActionRegistry,
        dispatcher: ActionPluginDispatcher,
        policy: PublicPolicy,
        verifier: PrivateVerifier,
        availability_provider: ActionAvailabilityProvider | None = None,
        max_controller_ticks: int = 10_000,
    ) -> None:
        if task.episode != episode:
            raise ValueError("task context belongs to another episode")
        if set(task.enabled_actions).difference(registry.names):
            raise ValueError("task enables an action missing from registry")
        if (
            isinstance(max_controller_ticks, bool)
            or not isinstance(max_controller_ticks, int)
            or max_controller_ticks <= 0
        ):
            raise ValueError("max_controller_ticks must be positive")
        self.episode = episode
        self.task = task
        self.registry = registry
        self.dispatcher = dispatcher
        self.policy = policy
        self.verifier = verifier
        self.availability_provider = availability_provider
        self.max_controller_ticks = max_controller_ticks
        self._history: list[Mapping[str, Any]] = []

    @property
    def public_history(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._history)

    @property
    def model_history(self) -> tuple[Mapping[str, Any], ...]:
        """Expose only the policy's prior commands, never executor outcomes."""

        return tuple(
            _freeze_history_value(
                {
                    "sequence": item["sequence"],
                    "action": item["action"],
                }
            )
            for item in self._history
        )

    def execute_one(
        self,
        *,
        observation: PublicRobotObservation,
        evaluator_state: Any,
        services: Mapping[str, Any] | None = None,
    ) -> PolicyLoopResult:
        decision = self.policy.decide(
            self.task,
            observation,
            self.model_history,
        )
        action = (
            decision
            if isinstance(decision, ActionEnvelope)
            else self.registry.parse_payload(decision)
        )
        if action.action not in self.task.enabled_actions:
            raise ValueError("policy selected an action disabled for this task")
        if self.availability_provider is not None:
            self.availability_provider.evaluate(
                self.task,
                observation,
            ).require(action.action)
        context = ActionExecutionContext(
            episode_id=self.episode.episode_id,
            action_index=len(self._history),
            services=services or {},
        )
        active = self.dispatcher.begin(self.registry, action, context)
        result: ActionPluginResult | None = None
        for _ in range(self.max_controller_ticks):
            result = self.dispatcher.tick(active)
            if result is not None:
                break
        if result is None:
            result = self.dispatcher.abort(
                active,
                "controller_tick_budget_exceeded",
            )

        public_record = _freeze_history_value(
            {
                "sequence": len(self._history),
                "action": action.to_dict(),
                "result": {
                    "status": result.status,
                    **dict(result.public_feedback),
                },
                "simulator_steps": active.simulator_steps,
            },
        )
        self._history.append(public_record)
        try:
            verifier_result = self.verifier.evaluate(
                self.episode,
                evaluator_state,
            )
        except Exception as exc:
            verifier_result = PrivateVerifierResult(
                ok=False,
                official_success=False,
                official_q=None,
                error=f"{type(exc).__name__}: {exc}",
            )
        return PolicyLoopResult(
            action=action,
            public_action_result=result,
            simulator_steps=active.simulator_steps,
            private_verifier_result=verifier_result,
            official_success=(
                verifier_result.ok and verifier_result.official_success
            ),
        )


def _freeze_history_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _freeze_history_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_history_value(item) for item in value)
    return value
