"""Private predicate verification across an episode lifecycle."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from behavior_hl.embodied.runtime.episode import EpisodeSpec
from behavior_hl.embodied.runtime.termination import (
    TerminationDecision,
    TerminationFacts,
    classify_termination,
)


class VerificationPhase(str, Enum):
    INITIAL = "initial"
    AFTER_ACTION = "after_action"
    SETTLE_FINAL = "settle_final"


@dataclass(frozen=True, slots=True)
class PrivatePredicateSnapshot:
    """One fail-closed read of official task state at the evaluator boundary."""

    ok: bool
    official_success: bool
    official_q: float | None
    leaf_truth: Mapping[str, bool] = field(default_factory=dict)
    research_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        leaves = dict(self.leaf_truth)
        if any(not isinstance(key, str) or not key for key in leaves):
            raise ValueError("leaf identifiers must be non-empty strings")
        if any(not isinstance(value, bool) for value in leaves.values()):
            raise ValueError("leaf truth values must be boolean")
        object.__setattr__(self, "leaf_truth", MappingProxyType(leaves))
        object.__setattr__(
            self,
            "research_diagnostics",
            MappingProxyType(dict(self.research_diagnostics)),
        )
        if self.ok:
            if self.error is not None:
                raise ValueError("ok snapshot cannot contain an error")
            if (
                self.official_q is None
                or not math.isfinite(self.official_q)
                or not 0.0 <= self.official_q <= 1.0
            ):
                raise ValueError("ok snapshot requires official_q in [0, 1]")
            if not leaves:
                raise ValueError("ok snapshot requires official goal leaves")
        else:
            if self.official_success or self.official_q is not None or leaves:
                raise ValueError("failed snapshot must fail closed")
            if not isinstance(self.error, str) or not self.error:
                raise ValueError("failed snapshot requires an error")

    def to_private_record(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "official_success": self.official_success,
            "official_q": self.official_q,
            "leaf_truth": dict(self.leaf_truth),
            "research_diagnostics": dict(self.research_diagnostics),
            "error": self.error,
        }


class PredicateVerifier(Protocol):
    def evaluate(
        self,
        episode: EpisodeSpec,
        evaluator_state: Any,
    ) -> PrivatePredicateSnapshot:
        """Re-read live official state; never use cached policy claims."""


@dataclass(frozen=True, slots=True)
class VerificationCheckpoint:
    sequence: int
    phase: VerificationPhase
    action_index: int | None
    snapshot: PrivatePredicateSnapshot
    newly_completed_leaves: tuple[str, ...]
    active_new_leaves: tuple[str, ...]
    regressed_leaves: tuple[str, ...]
    research_partial_q: float | None

    def to_private_record(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "phase": self.phase.value,
            "action_index": self.action_index,
            "snapshot": self.snapshot.to_private_record(),
            "newly_completed_leaves": list(self.newly_completed_leaves),
            "active_new_leaves": list(self.active_new_leaves),
            "regressed_leaves": list(self.regressed_leaves),
            "research_partial_q": self.research_partial_q,
        }


class EpisodeVerificationLedger:
    """Keep initial truth separate and derive credit from current live truth."""

    def __init__(self) -> None:
        self._initial: Mapping[str, bool] | None = None
        self._previous: Mapping[str, bool] | None = None
        self._checkpoints: list[VerificationCheckpoint] = []

    @property
    def checkpoints(self) -> tuple[VerificationCheckpoint, ...]:
        return tuple(self._checkpoints)

    @property
    def initial_leaf_truth(self) -> Mapping[str, bool]:
        if self._initial is None:
            raise RuntimeError("initial predicate snapshot is not recorded")
        return self._initial

    def record(
        self,
        snapshot: PrivatePredicateSnapshot,
        *,
        phase: VerificationPhase,
        action_index: int | None = None,
    ) -> VerificationCheckpoint:
        if phase is VerificationPhase.INITIAL:
            if self._initial is not None:
                raise RuntimeError("initial predicate snapshot already recorded")
            if action_index is not None:
                raise ValueError("initial checkpoint cannot have action_index")
        elif self._initial is None:
            raise RuntimeError("initial predicate snapshot is required first")
        elif (
            isinstance(action_index, bool)
            or not isinstance(action_index, int)
            or action_index < 0
        ):
            raise ValueError("action checkpoint requires action_index")

        if not snapshot.ok:
            checkpoint = VerificationCheckpoint(
                sequence=len(self._checkpoints),
                phase=phase,
                action_index=action_index,
                snapshot=snapshot,
                newly_completed_leaves=(),
                active_new_leaves=(),
                regressed_leaves=(),
                research_partial_q=None,
            )
            self._checkpoints.append(checkpoint)
            return checkpoint

        current = dict(snapshot.leaf_truth)
        if phase is VerificationPhase.INITIAL:
            self._initial = MappingProxyType(current)
            previous = current
        else:
            if set(current) != set(self.initial_leaf_truth):
                raise ValueError("official goal leaf identities changed")
            previous = dict(self._previous or self.initial_leaf_truth)
        initial = dict(self.initial_leaf_truth)
        newly = tuple(
            sorted(
                leaf
                for leaf, value in current.items()
                if value and not previous.get(leaf, False)
            )
        )
        regressed = tuple(
            sorted(
                leaf
                for leaf, value in current.items()
                if not value and previous.get(leaf, False)
            )
        )
        active_new = tuple(
            sorted(
                leaf
                for leaf, value in current.items()
                if value and not initial[leaf]
            )
        )
        partial_q = len(active_new) / len(initial)
        checkpoint = VerificationCheckpoint(
            sequence=len(self._checkpoints),
            phase=phase,
            action_index=action_index,
            snapshot=snapshot,
            newly_completed_leaves=newly,
            active_new_leaves=active_new,
            regressed_leaves=regressed,
            research_partial_q=partial_q,
        )
        self._previous = MappingProxyType(current)
        self._checkpoints.append(checkpoint)
        return checkpoint


def safe_evaluate(
    verifier: PredicateVerifier,
    episode: EpisodeSpec,
    evaluator_state: Any,
) -> PrivatePredicateSnapshot:
    try:
        snapshot = verifier.evaluate(episode, evaluator_state)
    except Exception as exc:
        return PrivatePredicateSnapshot(
            ok=False,
            official_success=False,
            official_q=None,
            error=f"{type(exc).__name__}: {exc}",
        )
    if not isinstance(snapshot, PrivatePredicateSnapshot):
        return PrivatePredicateSnapshot(
            ok=False,
            official_success=False,
            official_q=None,
            error="verifier returned an invalid snapshot type",
        )
    return snapshot


class EpisodeVerificationController:
    """Own verifier timing and automatic official-success termination."""

    def __init__(
        self,
        *,
        episode: EpisodeSpec,
        verifier: PredicateVerifier,
        expected_leaf_count: int | None = None,
        require_all_initial_false: bool = False,
    ) -> None:
        if expected_leaf_count is not None and (
            isinstance(expected_leaf_count, bool)
            or not isinstance(expected_leaf_count, int)
            or expected_leaf_count <= 0
        ):
            raise ValueError("expected_leaf_count must be positive")
        if not isinstance(require_all_initial_false, bool):
            raise ValueError("require_all_initial_false must be boolean")
        self.episode = episode
        self.verifier = verifier
        self.expected_leaf_count = expected_leaf_count
        self.require_all_initial_false = require_all_initial_false
        self.ledger = EpisodeVerificationLedger()
        self.terminal: TerminationDecision | None = None

    def initialize(self, evaluator_state: Any) -> VerificationCheckpoint:
        snapshot = safe_evaluate(
            self.verifier,
            self.episode,
            evaluator_state,
        )
        checkpoint = self.ledger.record(
            snapshot,
            phase=VerificationPhase.INITIAL,
        )
        if snapshot.ok and (
            (
                self.expected_leaf_count is not None
                and len(snapshot.leaf_truth) != self.expected_leaf_count
            )
            or (
                self.require_all_initial_false
                and any(snapshot.leaf_truth.values())
            )
        ):
            self.terminal = classify_termination(
                TerminationFacts(invalid_episode=True),
                detail=(
                    "initial official goal leaves violate the frozen "
                    "episode contract"
                ),
            )
        else:
            self._fail_or_succeed(snapshot, phase="initial")
        return checkpoint

    def after_action(
        self,
        *,
        action_index: int,
        evaluator_state: Any,
    ) -> VerificationCheckpoint:
        self._require_active()
        snapshot = safe_evaluate(
            self.verifier,
            self.episode,
            evaluator_state,
        )
        checkpoint = self.ledger.record(
            snapshot,
            phase=VerificationPhase.AFTER_ACTION,
            action_index=action_index,
        )
        self._fail_or_succeed(snapshot, phase=f"action {action_index}")
        return checkpoint

    def settle_final(
        self,
        *,
        action_index: int,
        evaluator_state: Any,
    ) -> VerificationCheckpoint:
        self._require_active()
        snapshot = safe_evaluate(
            self.verifier,
            self.episode,
            evaluator_state,
        )
        checkpoint = self.ledger.record(
            snapshot,
            phase=VerificationPhase.SETTLE_FINAL,
            action_index=action_index,
        )
        self._fail_or_succeed(snapshot, phase=f"settle {action_index}")
        return checkpoint

    def stop_requested(self) -> TerminationDecision:
        if self.terminal is not None:
            return self.terminal
        self.terminal = classify_termination(
            TerminationFacts(task_failure=True),
            detail="policy requested stop before official success",
        )
        return self.terminal

    def _fail_or_succeed(
        self,
        snapshot: PrivatePredicateSnapshot,
        *,
        phase: str,
    ) -> None:
        if not snapshot.ok:
            self.terminal = classify_termination(
                TerminationFacts(verifier_failure=True),
                detail=f"private verifier failed at {phase}: {snapshot.error}",
            )
        elif snapshot.official_success:
            self.terminal = classify_termination(
                TerminationFacts(official_success=True),
                detail=f"official task success observed at {phase}",
            )

    def _require_active(self) -> None:
        if self.terminal is not None:
            raise RuntimeError("episode is already terminal")
