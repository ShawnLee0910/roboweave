"""Fail-closed assembly of the first resident BEHAVIOR-HL episode.

The factory is deliberately simulator-independent.  Live evaluator handles,
model calls, verifier truth, and artifact I/O remain injected boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from behavior_hl.embodied.runtime.availability import (
    EmbodimentActionAvailability,
)
from behavior_hl.embodied.runtime.b1_dispatcher_factory import (
    B1DispatcherBundle,
)
from behavior_hl.embodied.runtime.episode import (
    EpisodeSpec,
    PublicTaskContext,
)
from behavior_hl.embodied.runtime.readiness import (
    ControllerCoverageReport,
    audit_controller_coverage,
)
from behavior_hl.embodied.runtime.resident_episode import (
    ResidentArtifactSink,
    ResidentEnvironment,
    ResidentEpisodeLimits,
    ResidentEpisodeOrchestrator,
    ResidentModel,
)
from behavior_hl.embodied.runtime.verification import PredicateVerifier


B1_TASK_ID = "picking_up_toys"
B1_GOAL_LEAF_COUNT = 6


class B1EpisodeFactoryError(ValueError):
    """Raised before a malformed B1 episode can reach a live evaluator."""


@dataclass(frozen=True, slots=True)
class B1ResidentEpisodeBundle:
    orchestrator: ResidentEpisodeOrchestrator
    task: PublicTaskContext
    controller_coverage: ControllerCoverageReport


def build_b1_resident_episode(
    *,
    episode: EpisodeSpec,
    dispatcher_bundle: B1DispatcherBundle,
    model: ResidentModel,
    environment: ResidentEnvironment,
    verifier: PredicateVerifier,
    artifacts: ResidentArtifactSink,
    limits: ResidentEpisodeLimits,
    diagnostic_target_new_leaf_count: int | None = None,
) -> B1ResidentEpisodeBundle:
    """Wire one B1 episode after exact task/schema/controller checks."""

    if not isinstance(episode, EpisodeSpec):
        raise B1EpisodeFactoryError("episode must be an EpisodeSpec")
    if episode.task_id != B1_TASK_ID:
        raise B1EpisodeFactoryError(
            f"B1 accepts only task_id={B1_TASK_ID!r}"
        )
    if not isinstance(dispatcher_bundle, B1DispatcherBundle):
        raise B1EpisodeFactoryError(
            "dispatcher_bundle must be a B1DispatcherBundle"
        )
    registry = dispatcher_bundle.registry
    if episode.action_schema_version != registry.schema_version:
        raise B1EpisodeFactoryError(
            "episode action schema does not match the B1 registry"
        )
    _require_methods(model, ("complete",), "model")
    _require_methods(
        environment,
        (
            "observe",
            "frame_sha256",
            "frame_artifact",
            "evaluator_state",
            "controller_services",
            "record_public_action_result",
            "settle",
        ),
        "environment",
    )
    _require_methods(verifier, ("evaluate",), "verifier")
    _require_methods(artifacts, ("record", "finalize"), "artifacts")
    if not isinstance(limits, ResidentEpisodeLimits):
        raise B1EpisodeFactoryError(
            "limits must be ResidentEpisodeLimits"
        )

    task = PublicTaskContext(
        episode=episode,
        enabled_actions=registry.names,
    )
    coverage = audit_controller_coverage(
        registry=registry,
        dispatcher=dispatcher_bundle.dispatcher,
        enabled_actions=task.enabled_actions,
        intercepted_actions=("stop",),
    )
    coverage.require_ready()
    orchestrator = ResidentEpisodeOrchestrator(
        episode=episode,
        task=task,
        registry=registry,
        dispatcher=dispatcher_bundle.dispatcher,
        availability_provider=EmbodimentActionAvailability(),
        model=model,
        environment=environment,
        verifier=verifier,
        artifacts=artifacts,
        limits=limits,
        expected_leaf_count=B1_GOAL_LEAF_COUNT,
        require_all_initial_false=True,
        diagnostic_target_new_leaf_count=(
            diagnostic_target_new_leaf_count
        ),
    )
    return B1ResidentEpisodeBundle(
        orchestrator=orchestrator,
        task=task,
        controller_coverage=coverage,
    )


def _require_methods(value: Any, names: tuple[str, ...], label: str) -> None:
    missing = [
        name for name in names if not callable(getattr(value, name, None))
    ]
    if missing:
        raise B1EpisodeFactoryError(
            f"{label} is missing capabilities: {missing}"
        )
