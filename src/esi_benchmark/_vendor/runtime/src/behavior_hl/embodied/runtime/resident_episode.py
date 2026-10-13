"""Pure orchestration for one resident, observation-only task episode.

This module owns lifecycle order, budgets, verifier timing, termination, and
artifact completeness.  It has no simulator imports and does not recognize
individual controller action names except the protocol-level ``stop`` action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from behavior_hl.embodied.runtime.action_integrity import execution_integrity_failure

from behavior_hl.embodied.policy.protocol import (
    ActionEnvelope,
    ActionProtocolError,
)
from behavior_hl.embodied.policy.registry import ActionRegistry
from behavior_hl.embodied.runtime.artifact_contract import (
    EpisodeArtifactContract,
)
from behavior_hl.embodied.runtime.artifacts import (
    ArtifactInventory,
    ArtifactVisibility,
)
from behavior_hl.embodied.runtime.availability import (
    ActionAvailabilityProvider,
)
from behavior_hl.embodied.runtime.dispatch import (
    ActionExecutionContext,
    ActionPluginDispatcher,
    ActionPluginResult,
)
from behavior_hl.embodied.runtime.episode import EpisodeSpec, PublicTaskContext
from behavior_hl.embodied.runtime.observation import PublicRobotObservation
from behavior_hl.embodied.runtime.prompt import build_episode_prompt
from behavior_hl.embodied.runtime.termination import (
    TerminationDecision,
    TerminationFacts,
    classify_termination,
)
from behavior_hl.embodied.runtime.verification import (
    EpisodeVerificationController,
    PredicateVerifier,
    PrivatePredicateSnapshot,
    VerificationCheckpoint,
    safe_evaluate,
)


class ResidentModel(Protocol):
    """One strict model call over the exact prompt and public observation."""

    def complete(
        self,
        prompt: str,
        observation: PublicRobotObservation,
    ) -> str:
        """Return one raw strict-JSON action envelope."""


class ResidentEnvironment(Protocol):
    """Narrow resident environment boundary used by the orchestrator."""

    def observe(self) -> PublicRobotObservation:
        """Return the latest public onboard observation."""

    def frame_sha256(
        self,
        observation: PublicRobotObservation,
    ) -> Mapping[str, str]:
        """Return hashes for the exact onboard frames given to the model."""

    def frame_artifact(
        self,
        observation: PublicRobotObservation,
    ) -> Mapping[str, Any]:
        """Return exact encoded PNG evidence and its encoded-byte SHA-256."""

    def evaluator_state(self) -> Any:
        """Return private live state only for the evaluator-side verifier."""

    def controller_services(
        self,
        action_index: int,
        action: ActionEnvelope,
    ) -> Mapping[str, Any]:
        """Return private action-plugin services for this resident process."""

    def record_public_action_result(
        self,
        action: ActionEnvelope,
        result: ActionPluginResult,
    ) -> None:
        """Update executor state using only the parsed action and public result."""

    def settle(
        self,
        action: ActionEnvelope,
        result: ActionPluginResult,
        *,
        max_steps: int,
    ) -> int:
        """Perform action-aware settling without exceeding ``max_steps``."""


class ResidentArtifactSink(Protocol):
    """Artifact backend with one logical record namespace per episode."""

    def record(
        self,
        *,
        logical_name: str,
        visibility: ArtifactVisibility,
        payload: Any,
        append: bool = False,
    ) -> None:
        """Persist or append one required logical artifact."""

    def finalize(self, final_payload: Mapping[str, Any]) -> ArtifactInventory:
        """Finish video/manifest generation and return the final inventory."""


@dataclass(frozen=True, slots=True)
class ResidentEpisodeLimits:
    max_model_calls: int
    max_simulator_steps: int
    max_controller_ticks: int = 10_000
    max_visual_observation_frames: int = 8

    def __post_init__(self) -> None:
        for name in (
            "max_model_calls",
            "max_simulator_steps",
            "max_controller_ticks",
            "max_visual_observation_frames",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class ResidentEpisodeResult:
    termination: TerminationDecision
    action_count: int
    model_calls: int
    simulator_steps: int
    artifact_inventory: ArtifactInventory
    artifact_complete: bool
    checkpoints: tuple[VerificationCheckpoint, ...]


class ResidentEpisodeOrchestrator:
    """Run one episode to a typed terminal state without scene reloads."""

    def __init__(
        self,
        *,
        episode: EpisodeSpec,
        task: PublicTaskContext,
        registry: ActionRegistry,
        dispatcher: ActionPluginDispatcher,
        availability_provider: ActionAvailabilityProvider,
        model: ResidentModel,
        environment: ResidentEnvironment,
        verifier: PredicateVerifier,
        artifacts: ResidentArtifactSink,
        limits: ResidentEpisodeLimits,
        expected_leaf_count: int | None = None,
        require_all_initial_false: bool = False,
        diagnostic_target_new_leaf_count: int | None = None,
        stop_action_name: str = "stop",
    ) -> None:
        if task.episode != episode:
            raise ValueError("task context belongs to another episode")
        if set(task.enabled_actions).difference(registry.names):
            raise ValueError("task enables an action missing from registry")
        if not isinstance(stop_action_name, str) or not stop_action_name:
            raise ValueError("stop_action_name must not be empty")
        self.episode = episode
        self.task = task
        self.registry = registry
        self.dispatcher = dispatcher
        self.availability_provider = availability_provider
        self.model = model
        self.environment = environment
        self.verifier = verifier
        self.artifacts = artifacts
        self.limits = limits
        self.stop_action_name = stop_action_name
        if diagnostic_target_new_leaf_count is not None and (
            isinstance(diagnostic_target_new_leaf_count, bool)
            or not isinstance(diagnostic_target_new_leaf_count, int)
            or diagnostic_target_new_leaf_count <= 0
        ):
            raise ValueError(
                "diagnostic_target_new_leaf_count must be positive"
            )
        if (
            diagnostic_target_new_leaf_count is not None
            and expected_leaf_count is not None
            and diagnostic_target_new_leaf_count >= expected_leaf_count
        ):
            raise ValueError(
                "diagnostic target must be below full official leaf count"
            )
        self.diagnostic_target_new_leaf_count = (
            diagnostic_target_new_leaf_count
        )
        self.verification = EpisodeVerificationController(
            episode=episode,
            verifier=verifier,
            expected_leaf_count=expected_leaf_count,
            require_all_initial_false=require_all_initial_false,
        )
        self._public_history: list[Mapping[str, Any]] = []
        self._model_calls = 0
        self._simulator_steps = 0
        self._action_count = 0
        self._termination: TerminationDecision | None = None
        self._last_snapshot: PrivatePredicateSnapshot | None = None
        self._pending_model_visual_history: tuple[
            PublicRobotObservation, ...
        ] = ()
        self._started = False

    @property
    def public_history(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._public_history)

    def run(self) -> ResidentEpisodeResult:
        if self._started:
            raise RuntimeError("resident episode orchestrator is single-use")
        self._started = True
        self._initialize_artifact_streams()

        try:
            initial_observation = self._observe()
            self._record_frame("initial_frame", initial_observation)
            initial = self.verification.initialize(
                self.environment.evaluator_state()
            )
            self._last_snapshot = initial.snapshot
            self.artifacts.record(
                logical_name="initial_predicates",
                visibility=ArtifactVisibility.PRIVATE,
                payload=initial.to_private_record(),
            )
            self._event("initial_verifier", snapshot=initial.snapshot)
            self._termination = self.verification.terminal
        except Exception as exc:
            self._termination = _termination(
                "infrastructure_failure",
                f"episode initialization failed: {_error(exc)}",
            )

        while self._termination is None:
            if self._model_calls >= self.limits.max_model_calls:
                self._termination = _termination(
                    "timeout",
                    "model-call budget exhausted",
                )
                break
            if self._simulator_steps >= self.limits.max_simulator_steps:
                self._termination = _termination(
                    "timeout",
                    "simulator-step budget exhausted",
                )
                break
            self._run_one_decision()

        assert self._termination is not None
        return self._finalize()

    def _run_one_decision(self) -> None:
        action_index = self._action_count
        try:
            observation = self._observe()
            availability = self.availability_provider.evaluate(
                self.task,
                observation,
            )
            model_visual_sequence = _deduplicate_adjacent_observations(
                (
                    *self._pending_model_visual_history,
                    observation,
                ),
                frame_sha256=self.environment.frame_sha256,
            )[-self.limits.max_visual_observation_frames :]
            prompt = build_episode_prompt(
                task=self.task,
                registry=self.registry,
                availability=availability,
                frame_sha256=self.environment.frame_sha256(observation),
                public_history=self.public_history,
                ordered_observation_frame_sha256=tuple(
                    self.environment.frame_sha256(item)
                    for item in model_visual_sequence
                ),
            )
        except Exception as exc:
            self._termination = _termination(
                "infrastructure_failure",
                f"public observation or prompt failed: {_error(exc)}",
            )
            return

        try:
            complete_sequence = getattr(
                self.model,
                "complete_with_visual_history",
                None,
            )
            if callable(complete_sequence):
                raw = complete_sequence(prompt, model_visual_sequence)
            else:
                raw = self.model.complete(prompt, observation)
            self._model_calls += 1
        except Exception as exc:
            self._model_calls += 1
            self._termination = _termination(
                "infrastructure_failure",
                f"model call failed: {_error(exc)}",
            )
            self._event("model_call_failed", error=_error(exc))
            return
        self.artifacts.record(
            logical_name="model_raw_responses",
            visibility=ArtifactVisibility.PUBLIC,
            payload={
                "sequence": action_index,
                "prompt": prompt,
                "raw_response": raw,
            },
            append=True,
        )
        self._event("model_response", action_index=action_index)

        try:
            if not isinstance(raw, str):
                raise ActionProtocolError(
                    "model raw response must be a JSON string"
                )
            action = self.registry.parse_json(raw)
            if action.action not in self.task.enabled_actions:
                raise ActionProtocolError(
                    "model selected an action disabled for this task"
                )
            availability.require(action.action)
        except Exception as exc:
            self._termination = _termination(
                "model_protocol_failure",
                f"model action rejected: {_error(exc)}",
            )
            self._event("model_action_rejected", error=_error(exc))
            return

        self._action_count += 1
        self.artifacts.record(
            logical_name="parsed_actions",
            visibility=ArtifactVisibility.PUBLIC,
            payload={
                "sequence": action_index,
                "action": action.to_dict(),
            },
            append=True,
        )
        self._event(
            "controller_dispatch_selected",
            action_index=action_index,
            controller_key=self.registry.spec(action.action).controller_key,
        )
        self._record_frame(
            _step_name(action_index, "action_before"),
            observation,
        )

        before = self._safe_verify()
        self._last_snapshot = before
        verifier_bundle: dict[str, Any] = {
            "action_before": before.to_private_record(),
        }
        if not before.ok:
            self._termination = _termination(
                "verifier_failure",
                f"private verifier failed before action {action_index}: "
                f"{before.error}",
            )
            self._skip_step_tail(
                action_index,
                observation,
                verifier_bundle,
                "verifier_failed_before_action",
            )
            return
        if before.official_success:
            self._termination = _termination(
                "official_success",
                f"official task success observed before action {action_index}",
            )
            self._skip_step_tail(
                action_index,
                observation,
                verifier_bundle,
                "success_before_action",
            )
            return
        if action.action == self.stop_action_name:
            self._termination = self.verification.stop_requested()
            self._skip_step_tail(
                action_index,
                observation,
                verifier_bundle,
                "policy_early_stop",
            )
            return

        result = self._execute_action(
            action_index=action_index,
            action=action,
        )
        if result is None:
            self._skip_step_tail(
                action_index,
                observation,
                verifier_bundle,
                "executor_failure",
            )
            return
        action_result, action_steps, process_observations = result
        self._simulator_steps += action_steps
        self._public_history.append(
            {
                "sequence": action_index,
                "action": action.to_dict(),
                "result": {
                    "status": action_result.status,
                    **dict(action_result.public_feedback),
                },
                "simulator_steps": action_steps,
            }
        )
        try:
            self.environment.record_public_action_result(
                action,
                action_result,
            )
        except Exception as exc:
            self._termination = _termination(
                "infrastructure_failure",
                f"public action-state update failed: {_error(exc)}",
            )
            self._skip_step_tail(
                action_index,
                observation,
                verifier_bundle,
                "public_action_state_update_failed",
            )
            return

        try:
            action_after_observation = self._observe()
        except Exception as exc:
            self._termination = _termination(
                "infrastructure_failure",
                f"action-after observation failed: {_error(exc)}",
            )
            self._skip_step_tail(
                action_index,
                observation,
                verifier_bundle,
                "action_after_observation_failed",
            )
            return
        self._record_frame(
            _step_name(action_index, "action_after"),
            action_after_observation,
        )

        try:
            after = self.verification.after_action(
                action_index=action_index,
                evaluator_state=self.environment.evaluator_state(),
            )
        except Exception as exc:
            self._termination = _termination(
                "verifier_failure",
                f"private verifier contract failed after action "
                f"{action_index}: {_error(exc)}",
            )
            verifier_bundle["action_after"] = {
                "ok": False,
                "error": _error(exc),
            }
            self._record_frame(
                _step_name(action_index, "settle_final"),
                action_after_observation,
                skipped_reason="verifier_failed_after_action",
            )
            verifier_bundle["settle_final"] = {
                "skipped": "verifier_failed_after_action"
            }
            self._record_step_verifier(action_index, verifier_bundle)
            return
        self._last_snapshot = after.snapshot
        verifier_bundle["action_after"] = after.to_private_record()
        if self._simulator_steps > self.limits.max_simulator_steps:
            self._termination = _termination(
                "timeout",
                "simulator-step budget exceeded by action",
            )
        else:
            self._termination = self.verification.terminal
        if self._termination is not None:
            # Success or verifier failure after the action must not cause a
            # settle or another verifier call.
            self._record_frame(
                _step_name(action_index, "settle_final"),
                action_after_observation,
                skipped_reason="terminal_after_action",
            )
            verifier_bundle["settle_final"] = {
                "skipped": "terminal_after_action"
            }
            self._record_step_verifier(action_index, verifier_bundle)
            return

        remaining = self.limits.max_simulator_steps - self._simulator_steps
        try:
            settled_steps = self.environment.settle(
                action,
                action_result,
                max_steps=remaining,
            )
            if (
                isinstance(settled_steps, bool)
                or not isinstance(settled_steps, int)
                or settled_steps < 0
                or settled_steps > remaining
            ):
                raise ValueError(
                    "settle returned steps outside its bounded allowance"
                )
            self._simulator_steps += settled_steps
            settled_observation = self._observe()
        except Exception as exc:
            self._termination = _termination(
                "infrastructure_failure",
                f"settle failed: {_error(exc)}",
            )
            self._record_frame(
                _step_name(action_index, "settle_final"),
                action_after_observation,
                skipped_reason="settle_failed",
            )
            verifier_bundle["settle_final"] = {"skipped": "settle_failed"}
            self._record_step_verifier(action_index, verifier_bundle)
            return
        self._record_frame(
            _step_name(action_index, "settle_final"),
            settled_observation,
        )
        self._pending_model_visual_history = (
            observation,
            *process_observations,
            action_after_observation,
            settled_observation,
        )[-self.limits.max_visual_observation_frames :]

        try:
            final = self.verification.settle_final(
                action_index=action_index,
                evaluator_state=self.environment.evaluator_state(),
            )
        except Exception as exc:
            self._termination = _termination(
                "verifier_failure",
                f"private verifier contract failed after settle "
                f"{action_index}: {_error(exc)}",
            )
            verifier_bundle["settle_final"] = {
                "ok": False,
                "error": _error(exc),
            }
            self._record_step_verifier(action_index, verifier_bundle)
            return
        self._last_snapshot = final.snapshot
        verifier_bundle["settle_final"] = final.to_private_record()
        self._record_step_verifier(action_index, verifier_bundle)
        self._termination = self.verification.terminal
        if (
            self._termination is None
            and self.diagnostic_target_new_leaf_count is not None
        ):
            active_count = len(final.active_new_leaves)
            target_count = self.diagnostic_target_new_leaf_count
            if active_count > target_count:
                self._termination = classify_termination(
                    TerminationFacts(invalid_episode=True),
                    detail=(
                        "bounded physical canary exceeded its exact diagnostic "
                        f"target after settle: observed {active_count} newly "
                        f"active official goal leaves, expected {target_count}"
                    ),
                )
            elif active_count == target_count:
                self._termination = classify_termination(
                    TerminationFacts(canary_target_reached=True),
                    detail=(
                        "bounded physical canary reached exactly "
                        f"{active_count} newly active official goal "
                        "leaf/leaves after settle; full official task "
                        "success remains false"
                    ),
                )
        if (
            self._termination is None
            and self._simulator_steps >= self.limits.max_simulator_steps
        ):
            self._termination = _termination(
                "timeout",
                "simulator-step budget exhausted after settle",
            )

    def _execute_action(
        self,
        *,
        action_index: int,
        action: ActionEnvelope,
    ) -> tuple[
        ActionPluginResult,
        int,
        tuple[PublicRobotObservation, ...],
    ] | None:
        try:
            active = self.dispatcher.begin(
                self.registry,
                action,
                ActionExecutionContext(
                    episode_id=self.episode.episode_id,
                    action_index=action_index,
                    services=self.environment.controller_services(
                        action_index,
                        action,
                    ),
                ),
            )
            result: ActionPluginResult | None = None
            process_observations: list[PublicRobotObservation] = []
            for _ in range(self.limits.max_controller_ticks):
                result = self.dispatcher.tick(active)
                if active.capture_observation_requested:
                    process_observation = self._observe()
                    process_observations.append(process_observation)
                    self._record_frame(
                        _step_name(
                            action_index,
                            "action_process_"
                            f"{len(process_observations):03d}",
                        ),
                        process_observation,
                    )
                if result is not None:
                    break
            if result is None:
                result = self.dispatcher.abort(
                    active,
                    "controller_tick_budget_exceeded",
                )
            self._event(
                "action_executed",
                action_index=action_index,
                status=result.status,
                simulator_steps=active.simulator_steps,
            )
            return (
                result,
                active.simulator_steps,
                tuple(process_observations),
            )
        except Exception as exc:
            self._termination = _termination(
                "executor_failure",
                f"action executor failed: {_error(exc)}",
            )
            self._event(
                "action_executor_failed",
                action_index=action_index,
                error=_error(exc),
            )
            return None

    def _skip_step_tail(
        self,
        action_index: int,
        observation: PublicRobotObservation,
        verifier_bundle: dict[str, Any],
        reason: str,
    ) -> None:
        self._record_frame(
            _step_name(action_index, "action_after"),
            observation,
            skipped_reason=reason,
        )
        self._record_frame(
            _step_name(action_index, "settle_final"),
            observation,
            skipped_reason=reason,
        )
        verifier_bundle.setdefault("action_after", {"skipped": reason})
        verifier_bundle.setdefault("settle_final", {"skipped": reason})
        self._record_step_verifier(action_index, verifier_bundle)

    def _record_step_verifier(
        self,
        action_index: int,
        bundle: Mapping[str, Any],
    ) -> None:
        self.artifacts.record(
            logical_name=_step_name(action_index, "verifier"),
            visibility=ArtifactVisibility.PRIVATE,
            payload=dict(bundle),
        )

    def _record_frame(
        self,
        logical_name: str,
        observation: PublicRobotObservation,
        *,
        skipped_reason: str | None = None,
    ) -> None:
        artifact = self.environment.frame_artifact(observation)
        if not isinstance(artifact, Mapping):
            raise TypeError("environment frame_artifact must return a mapping")
        payload: dict[str, Any] = {
            "simulator_step": observation.simulator_step,
            "observation_schema_version": observation.schema_version,
            "frame_sha256": dict(
                self.environment.frame_sha256(observation)
            ),
            **dict(artifact),
        }
        if skipped_reason is not None:
            payload["skipped_reason"] = skipped_reason
        self.artifacts.record(
            logical_name=logical_name,
            visibility=ArtifactVisibility.MEDIA,
            payload=payload,
        )

    def _observe(self) -> PublicRobotObservation:
        observation = self.environment.observe()
        if not isinstance(observation, PublicRobotObservation):
            raise TypeError("environment returned an invalid public observation")
        return observation

    def _safe_verify(self) -> PrivatePredicateSnapshot:
        try:
            evaluator_state = self.environment.evaluator_state()
        except Exception as exc:
            return PrivatePredicateSnapshot(
                ok=False,
                official_success=False,
                official_q=None,
                error=f"evaluator state read failed: {_error(exc)}",
            )
        return safe_evaluate(
            self.verifier,
            self.episode,
            evaluator_state,
        )

    def _initialize_artifact_streams(self) -> None:
        self.artifacts.record(
            logical_name="events",
            visibility=ArtifactVisibility.SYSTEM,
            payload={
                "event": "episode_started",
                "episode_id": self.episode.episode_id,
            },
            append=True,
        )
        self.artifacts.record(
            logical_name="model_raw_responses",
            visibility=ArtifactVisibility.PUBLIC,
            payload=[],
        )
        self.artifacts.record(
            logical_name="parsed_actions",
            visibility=ArtifactVisibility.PUBLIC,
            payload=[],
        )

    def _event(self, event: str, **payload: Any) -> None:
        record = {"event": event, **payload}
        snapshot = record.pop("snapshot", None)
        if snapshot is not None:
            record["verifier_ok"] = snapshot.ok
            record["official_success"] = snapshot.official_success
        self.artifacts.record(
            logical_name="events",
            visibility=ArtifactVisibility.SYSTEM,
            payload=record,
            append=True,
        )

    def _finalize(self) -> ResidentEpisodeResult:
        assert self._termination is not None
        try:
            integrity_failure = execution_integrity_failure(self.environment.evaluator_state())
        except Exception as exc:
            integrity_failure = f"final evaluator state unavailable: {_error(exc)}"
        if integrity_failure is not None:
            self._last_snapshot = PrivatePredicateSnapshot(
                ok=False, official_success=False, official_q=None,
                error=integrity_failure,
            )
            self._termination = _termination("infrastructure_failure", integrity_failure)
        final_predicate_payload = (
            self._last_snapshot.to_private_record()
            if self._last_snapshot is not None
            else {
                "ok": False,
                "official_success": False,
                "official_q": None,
                "error": "no verifier snapshot was available",
            }
        )
        self.artifacts.record(
            logical_name="final_predicate_report",
            visibility=ArtifactVisibility.PRIVATE,
            payload={
                "latest": final_predicate_payload,
                "checkpoints": [
                    checkpoint.to_private_record()
                    for checkpoint in self.verification.ledger.checkpoints
                ],
            },
        )
        final_state = {
            "episode_id": self.episode.episode_id,
            "termination": self._termination.classification.value,
            "success": self._termination.success,
            "detail": self._termination.detail,
            "action_count": self._action_count,
            "model_calls": self._model_calls,
            "simulator_steps": self._simulator_steps,
        }
        self.artifacts.record(
            logical_name="final_state",
            visibility=ArtifactVisibility.SYSTEM,
            payload=final_state,
        )
        artifact_complete = False
        try:
            inventory = self.artifacts.finalize(final_state)
            EpisodeArtifactContract(
                action_count=self._action_count
            ).require_complete(inventory)
            artifact_complete = True
        except Exception as exc:
            inventory = (
                inventory
                if "inventory" in locals()
                and isinstance(inventory, ArtifactInventory)
                else ArtifactInventory()
            )
            self._termination = _termination(
                "infrastructure_failure",
                f"artifact contract failed: {_error(exc)}",
            )
        return ResidentEpisodeResult(
            termination=self._termination,
            action_count=self._action_count,
            model_calls=self._model_calls,
            simulator_steps=self._simulator_steps,
            artifact_inventory=inventory,
            artifact_complete=artifact_complete,
            checkpoints=self.verification.ledger.checkpoints,
        )


def _step_name(action_index: int, suffix: str) -> str:
    return f"step_{action_index:04d}_{suffix}"


def _termination(kind: str, detail: str) -> TerminationDecision:
    return classify_termination(
        TerminationFacts(**{kind: True}),
        detail=detail,
    )


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _deduplicate_adjacent_observations(
    observations: tuple[PublicRobotObservation, ...],
    *,
    frame_sha256: Callable[[PublicRobotObservation], Mapping[str, str]],
) -> tuple[PublicRobotObservation, ...]:
    """Keep chronological visual evidence without duplicate adjacent images.

    The settled observation is commonly sampled again at the beginning of the
    next decision.  Its simulator-step number may differ while every onboard
    image is byte-identical.  Sending both wastes one of the bounded image
    slots and can over-emphasize a non-event, so retain only the newer sample.
    """

    distinct: list[PublicRobotObservation] = []
    previous_identity: tuple[tuple[str, str], ...] | None = None
    for observation in observations:
        identity = tuple(sorted(frame_sha256(observation).items()))
        if identity == previous_identity:
            distinct[-1] = observation
        else:
            distinct.append(observation)
            previous_identity = identity
    return tuple(distinct)
