"""Explicit execution contracts for high-level goal-completion research.

The profiles in this module are opt-in.  They do not change the QA benchmark
or claim physical manipulation fidelity.  They make the research executor's
relaxations auditable while retaining four hard boundaries: official
predicate readback, no obvious structural penetration, no non-finite or
unsupported placement, and fatal handling of simulator / verifier corruption.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


DISABLED_SEMANTIC_EXECUTION_PROFILE = "disabled"
SEMANTIC_RESEARCH_V1 = "semantic_research_v1"


class SemanticActionOutcome(str, Enum):
    """Structured outcome shared by Oracle and public-action adapters."""

    COMMITTED = "committed"
    RECOVERABLE_NOT_EXECUTED = "recoverable_not_executed"
    FATAL_INVALID = "fatal_invalid"


@dataclass(frozen=True, slots=True)
class SemanticExecutionContract:
    name: str
    version: int
    execution_fidelity: str
    official_challenge_eligible: bool

    # Hard boundaries.
    official_predicate_readback_required: bool
    structural_penetration_prohibited: bool
    nonfinite_or_unsupported_placement_prohibited: bool
    simulator_or_verifier_corruption_is_fatal: bool

    # Deliberate high-level relaxations.
    normal_contact_is_rejection: bool
    complete_aabb_containment_required: bool
    container_motion_thresholds_required: bool
    failed_placement_retains_held_object: bool
    exhausted_candidates_are_recoverable: bool

    # Navigation policy.  Five pixels is the stable R1Pro base-only research
    # footprint used by the established route adapter on the 0.1 m map.  The
    # executor still performs <=5 cm swept structural checks; this is not a
    # physical full-body clearance claim.
    stable_base_footprint_required: bool
    navigation_erosion_kernel_pixels: int
    navigation_extra_margin_m: float
    maximum_swept_sample_m: float
    maximum_interaction_distance_m: float

    def __post_init__(self) -> None:
        if not self.name or self.version < 1:
            raise ValueError("semantic execution contract identity is invalid")
        if self.official_challenge_eligible:
            raise ValueError("research semantic execution cannot be challenge eligible")
        if self.navigation_erosion_kernel_pixels < 1:
            raise ValueError("navigation erosion kernel must be positive")
        if self.navigation_extra_margin_m < 0.0:
            raise ValueError("navigation margin must be non-negative")
        if not 0.0 < self.maximum_swept_sample_m <= 0.05:
            raise ValueError("semantic navigation sweep must be at most 5 cm")
        if self.maximum_interaction_distance_m <= 0.0:
            raise ValueError("interaction distance must be positive")
        if not all(
            (
                self.official_predicate_readback_required,
                self.structural_penetration_prohibited,
                self.nonfinite_or_unsupported_placement_prohibited,
                self.simulator_or_verifier_corruption_is_fatal,
                self.failed_placement_retains_held_object,
                self.exhausted_candidates_are_recoverable,
                self.stable_base_footprint_required,
            )
        ):
            raise ValueError("semantic research hard boundaries cannot be disabled")

    def to_private_report(self) -> Mapping[str, Any]:
        return MappingProxyType(asdict(self))


_SEMANTIC_RESEARCH_V1 = SemanticExecutionContract(
    name=SEMANTIC_RESEARCH_V1,
    version=1,
    execution_fidelity="kinematic_research",
    official_challenge_eligible=False,
    official_predicate_readback_required=True,
    structural_penetration_prohibited=True,
    nonfinite_or_unsupported_placement_prohibited=True,
    simulator_or_verifier_corruption_is_fatal=True,
    normal_contact_is_rejection=False,
    complete_aabb_containment_required=False,
    container_motion_thresholds_required=False,
    failed_placement_retains_held_object=True,
    exhausted_candidates_are_recoverable=True,
    stable_base_footprint_required=True,
    navigation_erosion_kernel_pixels=5,
    navigation_extra_margin_m=0.0,
    maximum_swept_sample_m=0.05,
    maximum_interaction_distance_m=1.30,
)


def semantic_execution_contract(name: str) -> SemanticExecutionContract | None:
    """Resolve an opt-in profile; ``disabled`` preserves legacy behaviour."""

    if name == DISABLED_SEMANTIC_EXECUTION_PROFILE:
        return None
    if name == SEMANTIC_RESEARCH_V1:
        return _SEMANTIC_RESEARCH_V1
    raise ValueError(f"unknown semantic execution profile: {name}")


__all__ = [
    "DISABLED_SEMANTIC_EXECUTION_PROFILE",
    "SEMANTIC_RESEARCH_V1",
    "SemanticActionOutcome",
    "SemanticExecutionContract",
    "semantic_execution_contract",
]
