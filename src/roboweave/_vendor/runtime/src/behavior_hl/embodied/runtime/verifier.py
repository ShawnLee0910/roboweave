"""Private verifier result contract.

Nothing in this module is intended for policy serialization.  The explicit
``to_private_record`` name prevents accidental use as public action feedback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from behavior_hl.embodied.runtime.episode import EpisodeSpec


@dataclass(frozen=True, slots=True)
class PrivateVerifierResult:
    """Fail-closed official metric result plus separate research diagnostics."""

    ok: bool
    official_success: bool
    official_q: float | None
    newly_completed_leaves: tuple[str, ...] = ()
    research_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        leaves = tuple(self.newly_completed_leaves)
        if len(set(leaves)) != len(leaves):
            raise ValueError("newly_completed_leaves must not contain duplicates")
        if any(not isinstance(leaf, str) or not leaf for leaf in leaves):
            raise ValueError("goal leaf identifiers must be non-empty strings")
        object.__setattr__(self, "newly_completed_leaves", leaves)
        object.__setattr__(
            self,
            "research_diagnostics",
            MappingProxyType(dict(self.research_diagnostics)),
        )

        if self.ok:
            if self.error is not None:
                raise ValueError("successful verifier result cannot contain error")
            if self.official_q is None or not math.isfinite(self.official_q):
                raise ValueError("ok verifier result requires finite official_q")
            if not 0.0 <= self.official_q <= 1.0:
                raise ValueError("official_q must be in [0, 1]")
        else:
            if self.official_success:
                raise ValueError("failed verifier must fail closed")
            if self.official_q is not None:
                raise ValueError("failed verifier must not publish official_q")
            if leaves:
                raise ValueError("failed verifier must not publish completed leaves")
            if not isinstance(self.error, str) or not self.error:
                raise ValueError("failed verifier requires a non-empty error")

    def to_private_record(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "official_success": self.official_success,
            "official_q": self.official_q,
            "newly_completed_leaves": list(self.newly_completed_leaves),
            "research_diagnostics": dict(self.research_diagnostics),
            "error": self.error,
        }


class PrivateVerifier(Protocol):
    """Evaluator-side verifier; its result must never enter policy history."""

    def evaluate(
        self,
        episode: EpisodeSpec,
        evaluator_state: Any,
    ) -> PrivateVerifierResult:
        """Re-read live evaluator state and return a fail-closed result."""
