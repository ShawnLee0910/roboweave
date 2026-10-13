"""Compile grounded BDDL Inside/Closed goals into a generic Oracle graph.

This module has no OmniGibson imports.  It consumes the declarative ``body``
of official grounded goal leaves and emits deterministic operations that can
be executed by the common navigation and symbolic-manipulation runtime.
"""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
import json
from typing import Any, Iterable, Sequence


class InsideGoalGraphError(ValueError):
    """Raised when a grounded goal cannot use the first-tier action set."""


@dataclass(frozen=True, slots=True)
class InsideRelation:
    child: str
    parent: str


@dataclass(frozen=True, slots=True)
class NotInsideRelation:
    child: str
    parent: str


@dataclass(frozen=True, slots=True)
class PreservedRelation:
    """A goal leaf that is already true and must remain true.

    The first-tier executor does not synthesize an action for this relation.
    It records the invariant so the live official verifier can fail closed if
    a navigation or manipulation action regresses it.  This is intentionally
    narrow: only positive ``OnTop`` is currently admitted.
    """

    predicate: str
    subject: str
    reference: str


@dataclass(frozen=True, slots=True)
class GoalGraphOperation:
    kind: str
    target: str
    destination: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"open", "place_inside", "close"}:
            raise ValueError("unsupported goal-graph operation")
        if not self.target:
            raise ValueError("goal-graph target must be non-empty")
        if self.kind == "place_inside" and not self.destination:
            raise ValueError("place_inside requires a destination")
        if self.kind in {"open", "close"} and self.destination is not None:
            raise ValueError(f"{self.kind} does not accept a destination")

    @property
    def node_id(self) -> str:
        return f"inside:{self.kind}:{self.target}:{self.destination or ''}"

    @property
    def action(self) -> str:
        return self.kind

    @property
    def dependencies(self) -> tuple[str, ...]:
        return ()

    @property
    def achieves(self) -> tuple[str, ...]:
        return (self.kind,) if self.kind != "open" else ()


@dataclass(frozen=True, slots=True)
class GroundedInsideGoalPlan:
    option_index: int
    positive_inside: tuple[InsideRelation, ...]
    negative_inside: tuple[NotInsideRelation, ...]
    closed_targets: tuple[str, ...]
    preserved_relations: tuple[PreservedRelation, ...]
    operations: tuple[GoalGraphOperation, ...]
    leaf_count: int

    @property
    def action_nodes(self) -> tuple[GoalGraphOperation, ...]:
        """Expose the existing ordered operations to the public executor."""

        return self.operations


@dataclass(frozen=True, slots=True)
class CanonicalGroundedOptionSelection:
    """One identity-attested official option and its bounded probe record."""

    option_index: int
    option: Sequence[Any]
    preferred_option_index: int | None
    probed_option_count: int
    bounded_fallback_used: bool


def _body(leaf: Any) -> list[Any]:
    value = getattr(leaf, "body", leaf)
    if not isinstance(value, (list, tuple)) or not value:
        raise InsideGoalGraphError("grounded goal leaf has no declarative body")
    return list(value)


def grounded_goal_option_identity(option: Sequence[Any]) -> tuple[str, ...]:
    """Return the deterministic declarative identity used by the catalog.

    The B100 profile compiler stores one canonical grounded option as sorted,
    compact JSON leaf bodies.  Keeping the live selector here prevents reset
    attestation and the Oracle graph executor from implementing subtly
    different option matching rules.
    """

    if not isinstance(option, (list, tuple)) or not option:
        raise InsideGoalGraphError("grounded goal option must be non-empty")
    return tuple(
        sorted(
            json.dumps(
                _body(leaf),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            for leaf in option
        )
    )


def _canonical_goal_option_identity(
    canonical_goal_leaves: Sequence[str],
) -> tuple[str, ...]:
    """Normalize one catalog identity through its declarative JSON bodies."""

    if not isinstance(canonical_goal_leaves, (list, tuple)):
        raise InsideGoalGraphError("canonical grounded goal identity is invalid")
    normalized: list[str] = []
    for encoded in canonical_goal_leaves:
        if not isinstance(encoded, str) or not encoded:
            raise InsideGoalGraphError("canonical grounded goal identity is invalid")
        try:
            body = json.loads(encoded)
            normalized.append(
                json.dumps(
                    _body(body),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        except (InsideGoalGraphError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise InsideGoalGraphError(
                "canonical grounded goal identity is invalid"
            ) from exc
    if not normalized:
        raise InsideGoalGraphError("canonical grounded goal identity is invalid")
    return tuple(sorted(normalized))


def select_canonical_grounded_option(
    options: Sequence[Sequence[Any]],
    canonical_goal_leaves: Sequence[str],
    *,
    preferred_option_index: int | None = None,
    maximum_identity_probes: int = 4096,
    allow_bounded_fallback: bool = True,
) -> CanonicalGroundedOptionSelection:
    """Resolve one frozen semantic option without unbounded live enumeration.

    The official BDDL exact-count grounder can emit the same grounded leaf set
    more than once.  In particular, OR choices made for children that are not
    selected by ``forn`` multiply raw option indices without changing the
    resulting predicates.  Those occurrences are one semantic option, not an
    ambiguity.  A caller may provide the catalog's preferred raw index for an
    O(1) identity-attested fast path.  Legacy catalogs whose index used a
    different lazy enumeration may use a bounded fallback scan.  The scan is
    never allowed to grow with an exponentially large live option table.
    """

    if (
        not isinstance(options, SequenceABC)
        or isinstance(options, (str, bytes))
        or len(options) < 1
    ):
        raise InsideGoalGraphError("official grounded goal options are unavailable")
    if (
        isinstance(maximum_identity_probes, bool)
        or not isinstance(maximum_identity_probes, int)
        or maximum_identity_probes < 1
    ):
        raise InsideGoalGraphError("maximum identity probes must be positive")
    if preferred_option_index is not None and (
        isinstance(preferred_option_index, bool)
        or not isinstance(preferred_option_index, int)
        or preferred_option_index < 0
    ):
        raise InsideGoalGraphError(
            "preferred canonical grounded goal option index is invalid"
        )
    canonical = _canonical_goal_option_identity(canonical_goal_leaves)
    probed = 0
    preferred_in_bounds = (
        preferred_option_index is not None
        and preferred_option_index < len(options)
    )
    if preferred_in_bounds:
        assert preferred_option_index is not None
        option = options[preferred_option_index]
        probed += 1
        if grounded_goal_option_identity(option) == canonical:
            return CanonicalGroundedOptionSelection(
                option_index=preferred_option_index,
                option=option,
                preferred_option_index=preferred_option_index,
                probed_option_count=probed,
                bounded_fallback_used=False,
            )
        if not allow_bounded_fallback:
            raise InsideGoalGraphError(
                "preferred canonical grounded goal option identity drifted"
            )
    elif preferred_option_index is not None and not allow_bounded_fallback:
        raise InsideGoalGraphError(
            "preferred canonical grounded goal option index is out of bounds"
        )

    probe_limit = min(len(options), maximum_identity_probes)
    for index in range(probe_limit):
        if preferred_in_bounds and index == preferred_option_index:
            continue
        option = options[index]
        probed += 1
        if grounded_goal_option_identity(option) == canonical:
            return CanonicalGroundedOptionSelection(
                option_index=index,
                option=option,
                preferred_option_index=preferred_option_index,
                probed_option_count=probed,
                bounded_fallback_used=preferred_option_index is not None,
            )
    if len(options) > probe_limit:
        raise InsideGoalGraphError(
            "canonical grounded goal option was not found within the bounded "
            f"identity probe ({probe_limit} of {len(options)} options)"
        )
    raise InsideGoalGraphError(
        "canonical grounded goal option is absent from live options; matched 0"
    )


def select_canonical_grounded_option_index(
    options: Sequence[Sequence[Any]],
    canonical_goal_leaves: Sequence[str],
    *,
    preferred_option_index: int | None = None,
    maximum_identity_probes: int = 4096,
    allow_bounded_fallback: bool = True,
) -> int:
    """Compatibility wrapper returning only the attested official index."""

    return select_canonical_grounded_option(
        options,
        canonical_goal_leaves,
        preferred_option_index=preferred_option_index,
        maximum_identity_probes=maximum_identity_probes,
        allow_bounded_fallback=allow_bounded_fallback,
    ).option_index


def compile_grounded_inside_option(
    option: Sequence[Any],
    *,
    option_index: int = 0,
    openable_targets: Iterable[str] = (),
) -> GroundedInsideGoalPlan:
    """Compile one official grounded option into dependency-safe operations."""

    if not isinstance(option, (list, tuple)) or not option:
        raise InsideGoalGraphError("grounded goal option must be non-empty")
    openable = frozenset(openable_targets)
    if any(not isinstance(target, str) or not target for target in openable):
        raise InsideGoalGraphError("openable targets must be non-empty object IDs")
    positive: list[InsideRelation] = []
    negative: list[NotInsideRelation] = []
    closed: list[str] = []
    preserved: list[PreservedRelation] = []
    for leaf in option:
        body = _body(leaf)
        polarity = True
        if body[0] == "not":
            if len(body) != 2 or not isinstance(body[1], (list, tuple)):
                raise InsideGoalGraphError("malformed negated grounded goal leaf")
            polarity = False
            body = list(body[1])
        if len(body) == 3 and body[0] == "inside":
            child, parent = body[1], body[2]
            if not all(isinstance(value, str) and value for value in (child, parent)):
                raise InsideGoalGraphError("inside arguments must be object IDs")
            relation = InsideRelation(child, parent)
            if polarity:
                positive.append(relation)
            else:
                negative.append(NotInsideRelation(child, parent))
        elif len(body) == 2 and body[0] == "open" and not polarity:
            target = body[1]
            if not isinstance(target, str) or not target:
                raise InsideGoalGraphError("closed target must be an object ID")
            closed.append(target)
        elif len(body) == 3 and body[0] == "ontop" and polarity:
            subject, reference = body[1], body[2]
            if not all(
                isinstance(value, str) and value
                for value in (subject, reference)
            ):
                raise InsideGoalGraphError("ontop arguments must be object IDs")
            preserved.append(
                PreservedRelation("ontop", subject, reference)
            )
        else:
            predicate = body[0] if body else "unknown"
            raise InsideGoalGraphError(
                f"predicate {predicate!r} is outside the first-tier action set"
            )

    if not positive:
        raise InsideGoalGraphError(
            "first-tier preserved relations require at least one Inside goal"
        )

    parent_by_child: dict[str, str] = {}
    for relation in positive:
        previous = parent_by_child.setdefault(relation.child, relation.parent)
        if previous != relation.parent:
            raise InsideGoalGraphError(
                "one object is required inside multiple distinct containers: "
                f"{relation.child!r} -> {previous!r}, {relation.parent!r}"
            )
    for relation in negative:
        if parent_by_child.get(relation.child) == relation.parent:
            raise InsideGoalGraphError(
                "the same option requires and forbids one Inside relation"
            )

    positive = list(dict.fromkeys(positive))
    negative = list(dict.fromkeys(negative))
    closed = list(dict.fromkeys(closed))
    preserved = list(dict.fromkeys(preserved))
    operations: list[GoalGraphOperation] = []
    emitted_edges: set[InsideRelation] = set()
    emitted_open: set[str] = set()
    emitted_closed: set[str] = set()
    # Emit complete dependency layers instead of depth-first chains.  Thus all
    # independent child -> inner-container relations finish before any inner
    # container is moved into an outer container.  This minimizes the time an
    # already-satisfied outer Inside relation is exposed to unrelated later
    # grasp / settle steps while retaining deterministic topological order.
    while len(emitted_edges) != len(positive):
        ready = sorted(
            (
                edge
                for edge in positive
                if edge not in emitted_edges
                and all(
                    prerequisite in emitted_edges
                    for prerequisite in positive
                    if prerequisite.parent == edge.child
                )
            ),
            key=lambda item: (item.child, item.parent),
        )
        if not ready:
            raise InsideGoalGraphError(
                "positive Inside relations are cyclic or disconnected"
            )
        for edge in ready:
            if edge.child in closed and edge.child not in emitted_closed:
                operations.append(GoalGraphOperation("close", edge.child))
                emitted_closed.add(edge.child)
            if edge.parent in openable and edge.parent not in emitted_open:
                # OPEN must occur before the corresponding place operation:
                # Symbolic OPEN is intentionally unavailable while carrying an
                # object, whereas PLACE_INSIDE first grasps its child.
                operations.append(GoalGraphOperation("open", edge.parent))
                emitted_open.add(edge.parent)
            operations.append(
                GoalGraphOperation("place_inside", edge.child, edge.parent)
            )
            emitted_edges.add(edge)
    for target in sorted(closed):
        if target not in emitted_closed:
            operations.append(GoalGraphOperation("close", target))
            emitted_closed.add(target)

    return GroundedInsideGoalPlan(
        option_index=int(option_index),
        positive_inside=tuple(positive),
        negative_inside=tuple(negative),
        closed_targets=tuple(closed),
        preserved_relations=tuple(preserved),
        operations=tuple(operations),
        leaf_count=len(option),
    )


def feasible_grounded_inside_plans(
    options: Iterable[Sequence[Any]],
    *,
    openable_targets: Iterable[str] = (),
) -> tuple[GroundedInsideGoalPlan, ...]:
    """Return all structurally feasible first-tier options."""

    plans: list[GroundedInsideGoalPlan] = []
    openable = tuple(openable_targets)
    for index, option in enumerate(options):
        try:
            plans.append(
                compile_grounded_inside_option(
                    option,
                    option_index=index,
                    openable_targets=openable,
                )
            )
        except InsideGoalGraphError:
            continue
    return tuple(plans)
