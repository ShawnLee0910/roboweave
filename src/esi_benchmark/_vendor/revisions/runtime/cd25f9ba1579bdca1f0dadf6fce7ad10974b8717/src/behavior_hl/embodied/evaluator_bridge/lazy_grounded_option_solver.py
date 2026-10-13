"""Bounded lazy assignment for exponentially large grounded BDDL goals.

The official task remains the owner of predicate construction and truth.  This
module only chooses one deterministic, identity-attested option without
materializing or walking the complete official option table.

There are two deliberately separate entry points:

* :func:`solve_lazy_bddl_witness` is the CPU-only authoring path.  It expands
  the official BDDL quantifiers through the existing lazy backtracker and
  stops at the first candidate accepted by the canonical goal compiler and an
  optional geometry / capacity gate.
* :func:`attest_official_option_witness` is the resident live adapter.  It
  validates the frozen authoring witness against one indexed official option
  and compiles only that option.  The returned official predicate objects are
  private and are never part of a model observation.

Neither path changes predicate thresholds or synthesizes success.  Live truth
is still read through :class:`OfficialPredicateVerifier`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from itertools import islice
import json
from typing import Any, Callable, Mapping, Sequence

from behavior_hl.embodied.evaluator_bridge.canonical_bddl_grounder import (
    AtomicGroundLeafGate,
    iter_indexed_ground_options,
)
from behavior_hl.embodied.evaluator_bridge.inside_goal_graph import (
    grounded_goal_option_identity,
    select_canonical_grounded_option,
)


LAZY_GROUNDED_OPTION_SOLVER_VERSION = (
    "behavior2026.lazy-grounded-option-solver/v2"
)
MAXIMUM_LAZY_ASSIGNMENT_CANDIDATES = 4096


class LazyGroundedOptionError(RuntimeError):
    """Raised when a frozen live witness cannot be safely attested."""


CandidateCompiler = Callable[[Sequence[Any], int], Any]
CandidateGate = Callable[[Sequence[Any]], bool | tuple[bool, str]]
CompiledCandidateGate = Callable[
    [Sequence[Any], Any], bool | tuple[bool, str]
]


def _canonical_identity(option: Sequence[Any]) -> tuple[str, ...]:
    return grounded_goal_option_identity(option)


def _identity_sha256(identity: Sequence[str]) -> str:
    encoded = json.dumps(
        list(identity),
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_limit(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > MAXIMUM_LAZY_ASSIGNMENT_CANDIDATES
    ):
        raise ValueError(
            "maximum candidate assignments must be an integer in "
            f"[1, {MAXIMUM_LAZY_ASSIGNMENT_CANDIDATES}]"
        )
    return value


def _gate_decision(
    gate: Callable[..., bool | tuple[bool, str]] | None,
    *values: Any,
) -> tuple[bool, str]:
    if gate is None:
        return True, "not_configured"
    value = gate(*values)
    if isinstance(value, bool):
        return value, "accepted" if value else "candidate_gate_rejected"
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], bool)
        and isinstance(value[1], str)
        and value[1]
    ):
        return value
    raise TypeError("candidate gate must return bool or (bool, non-empty reason)")


@dataclass(frozen=True, slots=True)
class LazyGroundedOptionWitness:
    """One private deterministic witness and its bounded search audit."""

    option_index: int
    option: tuple[Any, ...]
    compiled_plan: Any
    identity: tuple[str, ...]
    identity_sha256: str
    candidates_examined: int
    rejected_candidate_counts: tuple[tuple[str, int], ...]
    source: str
    official_identity_attested: bool
    complete_option_table_enumerated: bool = False
    schema_version: str = LAZY_GROUNDED_OPTION_SOLVER_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.option_index, bool)
            or not isinstance(self.option_index, int)
            or self.option_index < 0
        ):
            raise ValueError("witness option index must be non-negative")
        if not self.option or not self.identity:
            raise ValueError("witness must contain one non-empty grounded option")
        if self.identity != _canonical_identity(self.option):
            raise ValueError("witness identity does not match its option")
        if self.identity_sha256 != _identity_sha256(self.identity):
            raise ValueError("witness identity hash drifted")
        if self.candidates_examined < 1:
            raise ValueError("witness must examine at least one candidate")
        if self.complete_option_table_enumerated:
            raise ValueError("lazy witness cannot enumerate the complete option table")

    def private_audit_record(self, *, raw_option_count: int | None = None) -> dict[str, Any]:
        """Return declarative private evidence without runtime object handles."""

        record: dict[str, Any] = {
            "schema_version": self.schema_version,
            "status": "witness",
            "option_index": self.option_index,
            "leaf_count": len(self.option),
            "identity_sha256": self.identity_sha256,
            "canonical_goal_leaves": list(self.identity),
            "candidates_examined": self.candidates_examined,
            "rejected_candidate_counts": dict(self.rejected_candidate_counts),
            "source": self.source,
            "official_identity_attested": self.official_identity_attested,
            "complete_option_table_enumerated": False,
            "maximum_candidate_assignments": MAXIMUM_LAZY_ASSIGNMENT_CANDIDATES,
            "private_only": True,
        }
        if raw_option_count is not None:
            record["official_raw_option_count"] = int(raw_option_count)
        return record


@dataclass(frozen=True, slots=True)
class LazyGroundingOutcome:
    """Structured CPU-only result for a bounded lazy search."""

    status: str
    witness: LazyGroundedOptionWitness | None
    candidates_examined: int
    rejected_candidate_counts: tuple[tuple[str, int], ...]
    error_code: str | None
    search_limit: int
    schema_version: str = LAZY_GROUNDED_OPTION_SOLVER_VERSION

    def __post_init__(self) -> None:
        if self.status not in {"witness", "no_solution"}:
            raise ValueError("lazy grounding status must be witness or no_solution")
        if (self.status == "witness") != (self.witness is not None):
            raise ValueError("lazy grounding status and witness disagree")
        if self.status == "no_solution" and not self.error_code:
            raise ValueError("no-solution result requires an error code")

    def private_audit_record(self) -> dict[str, Any]:
        if self.witness is not None:
            return self.witness.private_audit_record()
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "candidates_examined": self.candidates_examined,
            "rejected_candidate_counts": dict(self.rejected_candidate_counts),
            "error_code": self.error_code,
            "search_limit": self.search_limit,
            "complete_option_table_enumerated": False,
            "private_only": True,
        }


def solve_lazy_bddl_witness(
    *,
    goal_expression: object,
    object_map: Mapping[str, Sequence[str]],
    compile_candidate: CandidateCompiler,
    candidate_gate: CandidateGate | None = None,
    atomic_leaf_gate: AtomicGroundLeafGate | None = None,
    compiled_candidate_gate: CompiledCandidateGate | None = None,
    structural_error_types: tuple[type[BaseException], ...] = (ValueError,),
    maximum_candidate_assignments: int = MAXIMUM_LAZY_ASSIGNMENT_CANDIDATES,
) -> LazyGroundingOutcome:
    """Return the first deterministic structurally feasible BDDL witness.

    ``iter_ground_options`` provides the quantifier-aware lazy backtracking:
    ``forall`` propagates all required bindings, ``exists`` chooses one value,
    ``forn`` uses distinct combinations, and ``forpairs`` uses one-to-one
    permutations.  The canonical compiler then propagates relation conflicts
    such as one child committed to two Inside destinations.  A caller may add
    a live-independent capacity or geometry gate without changing goal
    semantics.

    The search is fail-closed and never inspects more than the declared bound.
    It never materializes the complete option table.
    """

    limit = _validate_limit(maximum_candidate_assignments)
    if not callable(compile_candidate):
        raise TypeError("compile_candidate must be callable")
    rejected: Counter[str] = Counter()
    examined = 0
    for option_index, candidate in (
        islice(
            iter_indexed_ground_options(
                goal_expression,
                object_map,
                atomic_leaf_gate=atomic_leaf_gate,
            ),
            limit,
        )
    ):
        examined += 1
        accepted, reason = _gate_decision(candidate_gate, candidate)
        if not accepted:
            rejected[reason] += 1
            continue
        try:
            plan = compile_candidate(candidate, option_index)
        except structural_error_types as exc:
            rejected[
                f"compiler_rejected:{type(exc).__name__}"
            ] += 1
            continue
        accepted, reason = _gate_decision(
            compiled_candidate_gate,
            candidate,
            plan,
        )
        if not accepted:
            rejected[reason] += 1
            continue
        identity = _canonical_identity(candidate)
        witness = LazyGroundedOptionWitness(
            option_index=option_index,
            option=tuple(candidate),
            compiled_plan=plan,
            identity=identity,
            identity_sha256=_identity_sha256(identity),
            candidates_examined=examined,
            rejected_candidate_counts=tuple(sorted(rejected.items())),
            source=(
                "official_bddl_constraint_pruned_lazy_backtracking"
                if atomic_leaf_gate is not None
                else "official_bddl_lazy_backtracking"
            ),
            official_identity_attested=False,
        )
        return LazyGroundingOutcome(
            status="witness",
            witness=witness,
            candidates_examined=examined,
            rejected_candidate_counts=tuple(sorted(rejected.items())),
            error_code=None,
            search_limit=limit,
        )
    return LazyGroundingOutcome(
        status="no_solution",
        witness=None,
        candidates_examined=examined,
        rejected_candidate_counts=tuple(sorted(rejected.items())),
        error_code=(
            "candidate_assignment_space_exhausted"
            if examined < limit
            else "candidate_assignment_search_limit_reached"
        ),
        search_limit=limit,
    )


def attest_official_option_witness(
    *,
    official_options: Sequence[Sequence[Any]],
    canonical_goal_leaves: Sequence[str],
    preferred_option_index: int,
    compile_candidate: CandidateCompiler,
    maximum_identity_probes: int = MAXIMUM_LAZY_ASSIGNMENT_CANDIDATES,
    allow_bounded_fallback: bool = False,
) -> LazyGroundedOptionWitness:
    """Attest and compile one indexed official option without table traversal."""

    probe_limit = _validate_limit(maximum_identity_probes)
    if not callable(compile_candidate):
        raise TypeError("compile_candidate must be callable")
    try:
        selection = select_canonical_grounded_option(
            official_options,
            canonical_goal_leaves,
            preferred_option_index=preferred_option_index,
            maximum_identity_probes=probe_limit,
            allow_bounded_fallback=allow_bounded_fallback,
        )
        plan = compile_candidate(selection.option, selection.option_index)
    except Exception as exc:
        raise LazyGroundedOptionError(
            "frozen canonical witness failed official identity attestation or "
            "canonical compilation"
        ) from exc
    identity = _canonical_identity(selection.option)
    return LazyGroundedOptionWitness(
        option_index=selection.option_index,
        option=tuple(selection.option),
        compiled_plan=plan,
        identity=identity,
        identity_sha256=_identity_sha256(identity),
        candidates_examined=selection.probed_option_count,
        rejected_candidate_counts=(),
        source="frozen_catalog_official_identity",
        official_identity_attested=True,
    )


def attest_indexed_official_option_witness(
    *,
    official_options: Sequence[Sequence[Any]],
    selected_option_index: int,
    compile_candidate: CandidateCompiler,
) -> LazyGroundedOptionWitness:
    """Freeze one *actual* indexed official option in O(1).

    Exponentially large official tables do not necessarily use the same raw
    ordering as the CPU authoring grounder.  In that case a catalog index is
    not an identity and must not be trusted on its own.  This adapter reads
    exactly one official option, compiles that exact predicate set, and
    freezes its declarative identity.  Subsequent users must compare the live
    option at ``selected_option_index`` with the returned identity instead of
    probing or enumerating the table again.
    """

    if not callable(compile_candidate):
        raise TypeError("compile_candidate must be callable")
    if (
        isinstance(selected_option_index, bool)
        or not isinstance(selected_option_index, int)
        or selected_option_index < 0
    ):
        raise ValueError("selected official option index must be non-negative")
    try:
        option_count = len(official_options)
    except Exception as exc:
        raise LazyGroundedOptionError(
            "official option table does not expose a bounded length"
        ) from exc
    if selected_option_index >= option_count:
        raise LazyGroundedOptionError(
            "selected official option index is out of bounds"
        )
    try:
        option = official_options[selected_option_index]
        plan = compile_candidate(option, selected_option_index)
        identity = _canonical_identity(option)
    except Exception as exc:
        raise LazyGroundedOptionError(
            "indexed official option failed identity capture or canonical "
            "compilation"
        ) from exc
    return LazyGroundedOptionWitness(
        option_index=selected_option_index,
        option=tuple(option),
        compiled_plan=plan,
        identity=identity,
        identity_sha256=_identity_sha256(identity),
        candidates_examined=1,
        rejected_candidate_counts=(),
        source="actual_indexed_official_option_identity",
        official_identity_attested=True,
    )
