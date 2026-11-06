"""Typed episode termination classification with no simulator dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AmbiguousTerminationError(ValueError):
    """Raised unless exactly one terminal fact is asserted."""


class TerminationClass(str, Enum):
    OFFICIAL_SUCCESS = "official_success"
    CANARY_TARGET_REACHED = "canary_target_reached"
    TASK_FAILURE = "task_failure"
    MODEL_PROTOCOL_FAILURE = "model_protocol_failure"
    EXECUTOR_FAILURE = "executor_failure"
    VERIFIER_FAILURE = "verifier_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    INVALID_EPISODE = "invalid_episode"
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True)
class TerminationFacts:
    """Mutually exclusive final facts captured by the episode owner."""

    official_success: bool = False
    canary_target_reached: bool = False
    task_failure: bool = False
    model_protocol_failure: bool = False
    executor_failure: bool = False
    verifier_failure: bool = False
    infrastructure_failure: bool = False
    invalid_episode: bool = False
    timeout: bool = False


@dataclass(frozen=True, slots=True)
class TerminationDecision:
    classification: TerminationClass
    success: bool
    valid_episode: bool
    retryable_infrastructure: bool
    detail: str


_FACT_TO_CLASS = {
    "official_success": TerminationClass.OFFICIAL_SUCCESS,
    "canary_target_reached": TerminationClass.CANARY_TARGET_REACHED,
    "task_failure": TerminationClass.TASK_FAILURE,
    "model_protocol_failure": TerminationClass.MODEL_PROTOCOL_FAILURE,
    "executor_failure": TerminationClass.EXECUTOR_FAILURE,
    "verifier_failure": TerminationClass.VERIFIER_FAILURE,
    "infrastructure_failure": TerminationClass.INFRASTRUCTURE_FAILURE,
    "invalid_episode": TerminationClass.INVALID_EPISODE,
    "timeout": TerminationClass.TIMEOUT,
}


def classify_termination(
    facts: TerminationFacts,
    *,
    detail: str,
) -> TerminationDecision:
    """Classify one exact terminal cause instead of applying hidden precedence."""

    if not isinstance(detail, str) or not detail.strip():
        raise ValueError("termination detail must be a non-empty string")
    asserted = [
        name for name in _FACT_TO_CLASS if getattr(facts, name)
    ]
    if len(asserted) != 1:
        raise AmbiguousTerminationError(
            "exactly one terminal fact is required; "
            f"received {asserted or 'none'}"
        )

    classification = _FACT_TO_CLASS[asserted[0]]
    return TerminationDecision(
        classification=classification,
        success=classification is TerminationClass.OFFICIAL_SUCCESS,
        valid_episode=classification
        not in {
            TerminationClass.INFRASTRUCTURE_FAILURE,
            TerminationClass.INVALID_EPISODE,
        },
        retryable_infrastructure=classification
        is TerminationClass.INFRASTRUCTURE_FAILURE,
        detail=detail.strip(),
    )
