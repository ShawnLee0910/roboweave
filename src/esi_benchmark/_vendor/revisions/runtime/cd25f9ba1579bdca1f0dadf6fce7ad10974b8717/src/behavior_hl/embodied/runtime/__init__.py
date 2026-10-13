"""Simulator-independent runtime contracts for high-level episodes."""

from behavior_hl.embodied.runtime.artifacts import (
    ArtifactInventory,
    ArtifactRecord,
    ArtifactVisibility,
)
from behavior_hl.embodied.runtime.availability import (
    ActionAvailability,
    ActionAvailabilityProvider,
    EmbodimentActionAvailability,
    GraspProxyAvailability,
)
from behavior_hl.embodied.runtime.artifact_contract import (
    EpisodeArtifactContract,
)
from behavior_hl.embodied.runtime.dispatch import (
    ActionExecutionContext,
    ActionPlugin,
    ActionPluginDispatcher,
    ActionPluginResult,
    ActionPluginTick,
    ActiveAction,
    ControllerLifecycleError,
    UnknownControllerError,
)
from behavior_hl.embodied.runtime.config_composition import EpisodeProfiles
from behavior_hl.embodied.runtime.episode import (
    EpisodeSpec,
    PublicBoundaryError,
    PublicTaskContext,
    TaskAdapter,
)
from behavior_hl.embodied.runtime.observation import (
    ObservationBoundaryError,
    PublicRobotObservation,
    RobotObservationAdapter,
)
from behavior_hl.embodied.runtime.policy_loop import (
    PolicyLoop,
    PolicyLoopResult,
    PublicPolicy,
)
from behavior_hl.embodied.runtime.prompt import (
    EPISODE_PROMPT_VERSION,
    build_episode_prompt,
    episode_prompt_sha256,
)
from behavior_hl.embodied.runtime.termination import (
    AmbiguousTerminationError,
    TerminationClass,
    TerminationDecision,
    TerminationFacts,
    classify_termination,
)
from behavior_hl.embodied.runtime.verifier import (
    PrivateVerifier,
    PrivateVerifierResult,
)
from behavior_hl.embodied.runtime.verification import (
    EpisodeVerificationController,
    EpisodeVerificationLedger,
    PredicateVerifier,
    PrivatePredicateSnapshot,
    VerificationCheckpoint,
    VerificationPhase,
    safe_evaluate,
)
from behavior_hl.embodied.runtime.writer import ArtifactWriter

__all__ = [
    "ActionExecutionContext",
    "ActionAvailability",
    "ActionAvailabilityProvider",
    "EmbodimentActionAvailability",
    "ActionPlugin",
    "ActionPluginDispatcher",
    "ActionPluginResult",
    "ActionPluginTick",
    "ActiveAction",
    "AmbiguousTerminationError",
    "ArtifactInventory",
    "ArtifactRecord",
    "ArtifactVisibility",
    "ArtifactWriter",
    "ControllerLifecycleError",
    "EpisodeSpec",
    "EpisodeArtifactContract",
    "EpisodeProfiles",
    "EPISODE_PROMPT_VERSION",
    "EpisodeVerificationController",
    "EpisodeVerificationLedger",
    "PrivateVerifier",
    "PrivateVerifierResult",
    "PrivatePredicateSnapshot",
    "ObservationBoundaryError",
    "GraspProxyAvailability",
    "PolicyLoop",
    "PolicyLoopResult",
    "PublicPolicy",
    "PublicBoundaryError",
    "PublicTaskContext",
    "PublicRobotObservation",
    "RobotObservationAdapter",
    "PredicateVerifier",
    "TaskAdapter",
    "TerminationClass",
    "TerminationDecision",
    "TerminationFacts",
    "UnknownControllerError",
    "VerificationCheckpoint",
    "VerificationPhase",
    "classify_termination",
    "safe_evaluate",
    "build_episode_prompt",
    "episode_prompt_sha256",
]
