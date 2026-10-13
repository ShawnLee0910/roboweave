"""Compile grounded mixed BDDL goals into a deterministic private action DAG.

The compiler is deliberately task agnostic and has no OmniGibson imports.  It
consumes one already-grounded official goal option (plain leaf bodies or
objects exposing ``body``) and supports the first mixed household relation
set needed by the generic Oracle runtime:

* positive ``Inside``;
* positive ``Open`` and ``Closed`` (official ``not Open``);
* positive ``OnTop``;
* positive ``NextTo``, ``Under``, and ``Touching``;
* positive and negative ``ToggledOn``.

Existential container selection is resolved by the official grounder before
this boundary.  The resulting concrete child -> destination edges are frozen
as :class:`DestinationCommitment` values so execution cannot silently switch
to another satisfying container.  Initially-satisfied goal relations that do
not need a scheduled mutation become explicit preservation invariants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence


MIXED_GOAL_GRAPH_VERSION = "behavior2026.mixed-goal-graph/v4"


class MixedGoalGraphError(ValueError):
    """Raised when a grounded option cannot be planned by this contract."""


_RELATION_KINDS = frozenset(
    {
        "inside",
        "open",
        "closed",
        "on_top",
        "next_to",
        "under",
        "touching",
        "toggled_on",
        "not_toggled_on",
    }
)
_UNARY_RELATION_KINDS = frozenset(
    {"open", "closed", "toggled_on", "not_toggled_on"}
)
_SPATIAL_RELATION_KINDS = frozenset(
    {"inside", "on_top", "under", "next_to", "touching"}
)
_SYMMETRIC_SPATIAL_RELATION_KINDS = frozenset({"next_to", "touching"})
# Multiple Inside destinations are contradictory.  OnTop is deliberately not
# treated as an alternative primary location: an object can be both Inside a
# cabinet / bookcase and OnTop of another object that is co-located there.
# Under, NextTo, and Touching are likewise compatible geometric refinements.
_PRIMARY_LOCATION_KINDS = frozenset({"inside"})
_PLACEMENT_ACTIONS = frozenset(
    {
        "place_inside",
        "place_on_top",
        "place_under",
        "place_next_to",
        "place_touching",
    }
)
_ACTIONS = frozenset(
    {"open", "close", "toggle_on", "toggle_off", *_PLACEMENT_ACTIONS}
)

_ACTION_BY_RELATION_KIND = {
    "open": "open",
    "inside": "place_inside",
    "on_top": "place_on_top",
    "under": "place_under",
    "next_to": "place_next_to",
    "touching": "place_touching",
    "toggled_on": "toggle_on",
    "not_toggled_on": "toggle_off",
}

_SUBJECT_PLACEMENT_ORDER = {
    "inside": 0,
    "on_top": 0,
    "under": 1,
    "next_to": 2,
    "touching": 3,
}


@dataclass(frozen=True, slots=True)
class MixedGoalRelation:
    """One normalized final relation from an official grounded goal leaf."""

    kind: str
    subject: str
    reference: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _RELATION_KINDS:
            raise ValueError(f"unsupported mixed goal relation {self.kind!r}")
        if not isinstance(self.subject, str) or not self.subject:
            raise ValueError("mixed goal relation subject must be non-empty")
        if self.kind in _UNARY_RELATION_KINDS:
            if self.reference is not None:
                raise ValueError(
                    f"{self.kind} relation does not accept a reference"
                )
        elif not isinstance(self.reference, str) or not self.reference:
            raise ValueError(f"{self.kind} relation requires a reference")
        if (
            self.kind in _SYMMETRIC_SPATIAL_RELATION_KINDS
            and self.subject == self.reference
        ):
            raise ValueError(
                f"{self.kind} relation requires distinct concrete objects"
            )

    @property
    def identity(self) -> tuple[str, str, str]:
        return (self.kind, self.subject, self.reference or "")

    def to_official_body(self) -> list[Any]:
        """Return the equivalent declarative BDDL leaf body."""

        if self.kind == "closed":
            return ["not", ["open", self.subject]]
        if self.kind == "open":
            return ["open", self.subject]
        if self.kind == "toggled_on":
            return ["toggled_on", self.subject]
        if self.kind == "not_toggled_on":
            return ["not", ["toggled_on", self.subject]]
        predicate = "ontop" if self.kind == "on_top" else self.kind
        return [predicate, self.subject, self.reference]


@dataclass(frozen=True, slots=True)
class SymmetricRelationAchievement:
    """One physical action covering one or two official symmetric leaves."""

    kind: str
    subject: str
    reference: str
    official_relations: tuple[MixedGoalRelation, ...]

    def __post_init__(self) -> None:
        if self.kind not in _SYMMETRIC_SPATIAL_RELATION_KINDS:
            raise ValueError("physical achievement requires a symmetric kind")
        if not self.subject or not self.reference:
            raise ValueError("physical achievement endpoints must be non-empty")
        if self.subject == self.reference:
            raise ValueError("physical achievement endpoints must be distinct")
        if not self.official_relations:
            raise ValueError("physical achievement must retain official leaves")
        expected_pair = frozenset((self.subject, self.reference))
        identities = set()
        for relation in self.official_relations:
            if relation.kind != self.kind:
                raise ValueError("physical achievement mixes relation kinds")
            if frozenset((relation.subject, relation.reference)) != expected_pair:
                raise ValueError("physical achievement mixes endpoint pairs")
            identities.add(relation.identity)
        if len(identities) != len(self.official_relations):
            raise ValueError("physical achievement repeats an official leaf")
        if len(self.official_relations) > 2:
            raise ValueError("a symmetric pair has at most two directed leaves")


@dataclass(frozen=True, slots=True)
class DestinationCommitment:
    """A concrete destination selected by one grounded Inside option."""

    subject: str
    destination: str
    initially_satisfied: bool

    def __post_init__(self) -> None:
        if not isinstance(self.subject, str) or not self.subject:
            raise ValueError("commitment subject must be non-empty")
        if not isinstance(self.destination, str) or not self.destination:
            raise ValueError("commitment destination must be non-empty")
        if not isinstance(self.initially_satisfied, bool):
            raise TypeError("initially_satisfied must be boolean")


@dataclass(frozen=True, slots=True)
class PreservationInvariant:
    """An initially true goal relation that no planned action may regress."""

    relation: MixedGoalRelation
    phase: str = "trajectory"
    reason: str = "initially_satisfied_goal"

    def __post_init__(self) -> None:
        if self.phase != "trajectory":
            raise ValueError("mixed goal preservation is trajectory-wide")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("preservation reason must be non-empty")


@dataclass(frozen=True, slots=True)
class MixedGoalActionNode:
    """One deterministic high-level action and its prerequisite node IDs."""

    node_id: str
    action: str
    target: str
    destination: str | None
    dependencies: tuple[str, ...]
    achieves: tuple[MixedGoalRelation, ...] = ()
    replay_on_recovery: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("action node ID must be non-empty")
        if self.action not in _ACTIONS:
            raise ValueError(f"unsupported mixed goal action {self.action!r}")
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("action target must be non-empty")
        if self.action in _PLACEMENT_ACTIONS:
            if not isinstance(self.destination, str) or not self.destination:
                raise ValueError(f"{self.action} requires a destination")
        elif self.destination is not None:
            raise ValueError(f"{self.action} does not accept a destination")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("action-node dependencies must be unique")
        if self.node_id in self.dependencies:
            raise ValueError("action node cannot depend on itself")
        if not isinstance(self.replay_on_recovery, bool):
            raise TypeError("replay_on_recovery must be boolean")

    def to_executor_action(self) -> tuple[str, str, str | None]:
        """Return the thin runtime adapter payload without simulator state."""

        return (self.action, self.target, self.destination)


@dataclass(frozen=True, slots=True)
class GroundedMixedGoalPlan:
    """A fully covered mixed goal option in deterministic topological order."""

    option_index: int
    goal_relations: tuple[MixedGoalRelation, ...]
    destination_commitments: tuple[DestinationCommitment, ...]
    preservation_invariants: tuple[PreservationInvariant, ...]
    action_nodes: tuple[MixedGoalActionNode, ...]
    leaf_count: int
    schema_version: str = MIXED_GOAL_GRAPH_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.option_index, bool)
            or not isinstance(self.option_index, int)
            or self.option_index < 0
        ):
            raise ValueError("option_index must be a non-negative integer")
        if self.leaf_count != len(self.goal_relations) or self.leaf_count <= 0:
            raise ValueError("leaf_count must match unique goal relations")
        node_ids = tuple(node.node_id for node in self.action_nodes)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("mixed goal node IDs must be unique")
        position = {node_id: index for index, node_id in enumerate(node_ids)}
        for index, node in enumerate(self.action_nodes):
            for dependency in node.dependencies:
                if dependency not in position:
                    raise ValueError("action node depends on an unknown node")
                if position[dependency] >= index:
                    raise ValueError(
                        "action nodes must be emitted in topological order"
                    )

        achieved = {
            relation
            for node in self.action_nodes
            for relation in node.achieves
        }
        preserved = {
            invariant.relation for invariant in self.preservation_invariants
        }
        if achieved.intersection(preserved):
            raise ValueError("one goal relation is both achieved and preserved")
        if achieved.union(preserved) != set(self.goal_relations):
            raise ValueError("mixed goal plan does not cover every goal relation")

        commitments = {
            commitment.subject: commitment.destination
            for commitment in self.destination_commitments
        }
        if len(commitments) != len(self.destination_commitments):
            raise ValueError("one subject has multiple destination commitments")
        expected = {
            relation.subject: relation.reference
            for relation in self.goal_relations
            if relation.kind == "inside"
        }
        if commitments != expected:
            raise ValueError("destination commitments do not match Inside goals")

    @property
    def action_sequence(self) -> tuple[tuple[str, str, str | None], ...]:
        return tuple(node.to_executor_action() for node in self.action_nodes)

    @property
    def identity(self) -> tuple[tuple[str, str, str], ...]:
        """Declarative identity independent of source leaf ordering."""

        return tuple(relation.identity for relation in self.goal_relations)

    def destination_for(self, subject: str) -> str:
        matches = tuple(
            item.destination
            for item in self.destination_commitments
            if item.subject == subject
        )
        if len(matches) != 1:
            raise MixedGoalGraphError(
                f"subject {subject!r} has no unique destination commitment"
            )
        return matches[0]


@dataclass(slots=True)
class _NodeDraft:
    action: str
    target: str
    destination: str | None
    achieves: tuple[MixedGoalRelation, ...]
    replay_on_recovery: bool = True

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.action, self.target, self.destination or "")


def _leaf_body(leaf: Any) -> list[Any]:
    value = getattr(leaf, "body", leaf)
    if not isinstance(value, (list, tuple)) or not value:
        raise MixedGoalGraphError("grounded goal leaf has no declarative body")
    return list(value)


def relation_from_grounded_leaf(leaf: Any) -> MixedGoalRelation:
    """Normalize one supported official grounded goal leaf."""

    body = _leaf_body(leaf)
    predicate = body[0]
    if not isinstance(predicate, str):
        raise MixedGoalGraphError("grounded goal predicate must be text")
    predicate = predicate.casefold()
    if predicate == "not":
        if len(body) != 2 or not isinstance(body[1], (list, tuple)):
            raise MixedGoalGraphError("malformed negated grounded goal leaf")
        child = list(body[1])
        if len(child) != 2 or not isinstance(child[0], str):
            raise MixedGoalGraphError(
                "negated grounded goal must contain a unary predicate"
            )
        child_predicate = child[0].casefold()
        target = child[1]
        if not isinstance(target, str) or not target:
            raise MixedGoalGraphError(
                "negated grounded goal target must be a concrete object"
            )
        if child_predicate == "open":
            return MixedGoalRelation("closed", target)
        if child_predicate in {"toggled_on", "toggledon"}:
            return MixedGoalRelation("not_toggled_on", target)
        raise MixedGoalGraphError(
            "only official not Open (Closed) and not ToggledOn are supported"
        )
    if predicate in {"open", "closed"}:
        if len(body) != 2:
            raise MixedGoalGraphError(f"{predicate} requires one concrete object")
        target = body[1]
        if not isinstance(target, str) or not target:
            raise MixedGoalGraphError(
                f"{predicate} target must be a concrete object"
            )
        return MixedGoalRelation(predicate, target)
    if predicate in {"toggled_on", "toggledon"}:
        if len(body) != 2:
            raise MixedGoalGraphError(
                "ToggledOn requires one concrete object"
            )
        target = body[1]
        if not isinstance(target, str) or not target:
            raise MixedGoalGraphError(
                "ToggledOn target must be a concrete object"
            )
        return MixedGoalRelation("toggled_on", target)
    kind = {
        "inside": "inside",
        "ontop": "on_top",
        "on_top": "on_top",
        "nextto": "next_to",
        "next_to": "next_to",
        "under": "under",
        "touching": "touching",
    }.get(predicate)
    if kind is None:
        raise MixedGoalGraphError(
            f"predicate {predicate!r} is outside the mixed action contract"
        )
    if len(body) != 3:
        raise MixedGoalGraphError(f"{predicate} requires two concrete objects")
    subject, reference = body[1], body[2]
    if not all(
        isinstance(value, str) and value for value in (subject, reference)
    ):
        raise MixedGoalGraphError(
            f"{predicate} arguments must be concrete object IDs"
        )
    if kind in _SYMMETRIC_SPATIAL_RELATION_KINDS and subject == reference:
        raise MixedGoalGraphError(
            f"{predicate} requires two distinct concrete objects"
        )
    return MixedGoalRelation(kind, subject, reference)


def _relations(values: Iterable[Any]) -> tuple[MixedGoalRelation, ...]:
    result = tuple(
        value
        if isinstance(value, MixedGoalRelation)
        else relation_from_grounded_leaf(value)
        for value in values
    )
    return tuple(sorted(set(result), key=lambda item: item.identity))


def normalize_symmetric_relation_achievements(
    relations: Iterable[MixedGoalRelation],
    *,
    immovable_targets: Iterable[str] = (),
) -> tuple[SymmetricRelationAchievement, ...]:
    """Normalize symmetric official leaves into dependency-safe actions.

    The relation endpoints form an undirected physical-achievement graph.
    Each tree component is rooted at its highest-degree endpoint (then stable
    lexical identity), and every edge is oriented child -> parent.  Executing
    parent edges before child edges means no later symmetric action moves the
    reference of an already-achieved child relation.  Cyclic components fail
    closed because a single-object placement executor cannot prove a serial
    ordering that preserves every edge of a cycle.

    This normalization is intentionally separate from authoring admission.
    Admission can prune self-relations before grounding expansion; this API is
    the compiler's fail-closed boundary and the sole owner of reciprocal-leaf
    coalescing.
    """

    selected = tuple(sorted(set(relations), key=lambda item: item.identity))
    immovable = frozenset(immovable_targets)
    if any(not isinstance(target, str) or not target for target in immovable):
        raise MixedGoalGraphError(
            "immovable targets must be concrete non-empty object IDs"
        )
    if any(
        relation.kind not in _SYMMETRIC_SPATIAL_RELATION_KINDS
        for relation in selected
    ):
        raise MixedGoalGraphError(
            "symmetric achievement normalization received a directed relation"
        )
    if any(relation.subject == relation.reference for relation in selected):
        raise MixedGoalGraphError(
            "symmetric spatial relation requires distinct concrete objects"
        )

    relation_groups: dict[
        tuple[str, str, str], list[MixedGoalRelation]
    ] = {}
    endpoint_pairs: set[tuple[str, str]] = set()
    adjacency: dict[str, set[str]] = {}
    official_reference_count: dict[str, int] = {}
    for relation in selected:
        endpoint_a, endpoint_b = sorted(
            (relation.subject, str(relation.reference))
        )
        relation_groups.setdefault(
            (relation.kind, endpoint_a, endpoint_b), []
        ).append(relation)
        endpoint_pairs.add((endpoint_a, endpoint_b))
        adjacency.setdefault(endpoint_a, set()).add(endpoint_b)
        adjacency.setdefault(endpoint_b, set()).add(endpoint_a)
        official_reference_count[str(relation.reference)] = (
            official_reference_count.get(str(relation.reference), 0) + 1
        )

    parent_by_node: dict[str, str | None] = {}
    unseen = set(adjacency)
    while unseen:
        seed = min(unseen)
        component: set[str] = set()
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            if current in component:
                continue
            component.add(current)
            frontier.extend(sorted(adjacency[current] - component))
        unseen.difference_update(component)
        component_edges = {
            pair
            for pair in endpoint_pairs
            if pair[0] in component and pair[1] in component
        }
        if len(component_edges) != len(component) - 1:
            raise MixedGoalGraphError(
                "symmetric spatial achievement graph contains a cycle; "
                "serial single-object placement cannot preserve every edge"
            )
        fixed_in_component = component.intersection(immovable)
        if len(fixed_in_component) > 1:
            raise MixedGoalGraphError(
                "one active symmetric component contains multiple immovable "
                "endpoints; serial placement cannot move either fixed root"
            )
        if fixed_in_component:
            root = next(iter(fixed_in_component))
        else:
            # Prefer an already-authored reference when degree permits so a
            # lone official R(subject, reference) keeps its public action
            # orientation. Fully reciprocal ties use the lexically greatest
            # stable endpoint, making the action subject lexically smaller.
            root = max(
                component,
                key=lambda node: (
                    len(adjacency[node]),
                    official_reference_count.get(node, 0),
                    node,
                ),
            )
        parent_by_node[root] = None
        queue = [root]
        while queue:
            parent = queue.pop(0)
            for child in sorted(adjacency[parent]):
                if child in parent_by_node:
                    continue
                parent_by_node[child] = parent
                queue.append(child)

    result: list[SymmetricRelationAchievement] = []
    for (kind, endpoint_a, endpoint_b), official_relations in sorted(
        relation_groups.items()
    ):
        if parent_by_node.get(endpoint_a) == endpoint_b:
            subject, reference = endpoint_a, endpoint_b
        elif parent_by_node.get(endpoint_b) == endpoint_a:
            subject, reference = endpoint_b, endpoint_a
        else:  # pragma: no cover - protected by the tree construction above
            raise MixedGoalGraphError(
                "symmetric physical edge has no rooted-tree orientation"
            )
        result.append(
            SymmetricRelationAchievement(
                kind=kind,
                subject=subject,
                reference=reference,
                official_relations=tuple(
                    sorted(official_relations, key=lambda item: item.identity)
                ),
            )
        )
    return tuple(
        sorted(
            result,
            key=lambda item: (
                item.subject,
                _SUBJECT_PLACEMENT_ORDER[item.kind],
                item.reference,
                tuple(relation.identity for relation in item.official_relations),
            ),
        )
    )


def _node_priority(key: tuple[str, str, str]) -> tuple[Any, ...]:
    action, target, destination = key
    if action == "open":
        return (0, target, 0, target, destination)
    if action == "place_inside":
        return (0, destination, 1, target, destination)
    spatial_rank = {
        "place_on_top": 0,
        "place_under": 1,
        "place_next_to": 2,
        "place_touching": 3,
    }.get(action)
    if spatial_rank is not None:
        return (1, target, spatial_rank, destination)
    if action in {"toggle_on", "toggle_off"}:
        return (2, target, action, destination)
    return (3, target, action, destination)


def _topological_nodes(
    drafts: Mapping[tuple[str, str, str], _NodeDraft],
    dependencies: Mapping[
        tuple[str, str, str], set[tuple[str, str, str]]
    ],
) -> tuple[MixedGoalActionNode, ...]:
    remaining = set(drafts)
    ordered_keys: list[tuple[str, str, str]] = []
    while remaining:
        ready = sorted(
            (
                key
                for key in remaining
                if not dependencies[key].intersection(remaining)
            ),
            key=_node_priority,
        )
        if not ready:
            cycle = sorted(remaining, key=_node_priority)
            raise MixedGoalGraphError(
                f"mixed goal dependencies contain a cycle: {cycle}"
            )
        selected = ready[0]
        ordered_keys.append(selected)
        remaining.remove(selected)

    node_id = {
        key: f"action_{index:03d}"
        for index, key in enumerate(ordered_keys)
    }
    order = {key: index for index, key in enumerate(ordered_keys)}
    return tuple(
        MixedGoalActionNode(
            node_id=node_id[key],
            action=drafts[key].action,
            target=drafts[key].target,
            destination=drafts[key].destination,
            dependencies=tuple(
                node_id[dependency]
                for dependency in sorted(
                    dependencies[key], key=lambda item: order[item]
                )
            ),
            achieves=drafts[key].achieves,
            replay_on_recovery=drafts[key].replay_on_recovery,
        )
        for key in ordered_keys
    )


def compile_grounded_mixed_option(
    option: Sequence[Any],
    *,
    option_index: int = 0,
    openable_targets: Iterable[str] = (),
    immovable_targets: Iterable[str] = (),
    initially_true_relations: Iterable[Any] = (),
    source_openable_ancestors_by_subject: Mapping[
        str, Iterable[str]
    ] | None = None,
    precedence_relations: Sequence[tuple[Any, Any]] = (),
) -> GroundedMixedGoalPlan:
    """Compile one official grounded option into a dependency-safe action DAG.

    ``initially_true_relations`` must come from the private live verifier.  A
    true goal is preserved instead of redundantly re-executed, except that a
    closed destination receiving new contents is deliberately opened and then
    restored closed at the end of the plan.  The optional source-ancestor
    mapping is private live containment evidence, ordered outer-to-inner.  It
    turns access to objects inside openable containers into ordinary OPEN
    prerequisites and makes a matching terminal Closed leaf executable.
    """

    if not isinstance(option, (list, tuple)) or not option:
        raise MixedGoalGraphError("grounded mixed goal option must be non-empty")
    if (
        isinstance(option_index, bool)
        or not isinstance(option_index, int)
        or option_index < 0
    ):
        raise MixedGoalGraphError("option_index must be a non-negative integer")

    parsed = tuple(relation_from_grounded_leaf(leaf) for leaf in option)
    if len(set(parsed)) != len(parsed):
        raise MixedGoalGraphError("grounded mixed goal contains duplicate leaves")
    goals = tuple(sorted(parsed, key=lambda item: item.identity))
    initially_true = frozenset(_relations(initially_true_relations))
    precedence = tuple((relation_from_grounded_leaf(before), relation_from_grounded_leaf(after))
                       for before, after in precedence_relations)
    for before, after in precedence:
        if after not in goals or after.kind not in _SUBJECT_PLACEMENT_ORDER:
            raise MixedGoalGraphError("process placement must identify a selected goal")
    openable = frozenset(openable_targets)
    if any(not isinstance(target, str) or not target for target in openable):
        raise MixedGoalGraphError(
            "openable targets must be concrete non-empty object IDs"
        )
    immovable = frozenset(immovable_targets)
    if any(not isinstance(target, str) or not target for target in immovable):
        raise MixedGoalGraphError(
            "immovable targets must be concrete non-empty object IDs"
        )
    source_ancestors: dict[str, tuple[str, ...]] = {}
    if source_openable_ancestors_by_subject is not None:
        if not isinstance(source_openable_ancestors_by_subject, Mapping):
            raise MixedGoalGraphError(
                "source openable ancestors must be a subject mapping"
            )
        for subject, raw_ancestors in sorted(
            source_openable_ancestors_by_subject.items()
        ):
            if not isinstance(subject, str) or not subject:
                raise MixedGoalGraphError(
                    "source openable ancestor subject must be concrete"
                )
            if isinstance(raw_ancestors, (str, bytes)):
                raise MixedGoalGraphError(
                    "source openable ancestors must be an ordered iterable"
                )
            ancestors = tuple(raw_ancestors)
            if any(
                not isinstance(ancestor, str) or not ancestor
                for ancestor in ancestors
            ):
                raise MixedGoalGraphError(
                    "source openable ancestors must be concrete object IDs"
                )
            if len(set(ancestors)) != len(ancestors):
                raise MixedGoalGraphError(
                    "source openable ancestor chain contains duplicates"
                )
            unavailable = tuple(
                ancestor for ancestor in ancestors if ancestor not in openable
            )
            if unavailable:
                raise MixedGoalGraphError(
                    "source ancestor is not in the live openable target set: "
                    f"{unavailable}"
                )
            source_ancestors[subject] = ancestors

    primary_location_by_subject: dict[str, MixedGoalRelation] = {}
    for relation in goals:
        if relation.kind not in _PRIMARY_LOCATION_KINDS:
            continue
        previous = primary_location_by_subject.setdefault(
            relation.subject, relation
        )
        if previous != relation:
            raise MixedGoalGraphError(
                "one subject has multiple terminal primary locations: "
                f"{relation.subject!r} -> {previous.identity}, "
                f"{relation.identity}"
            )

    toggle_by_subject: dict[str, MixedGoalRelation] = {}
    for relation in goals:
        if relation.kind not in {"toggled_on", "not_toggled_on"}:
            continue
        previous = toggle_by_subject.setdefault(relation.subject, relation)
        if previous != relation:
            raise MixedGoalGraphError(
                "one subject has contradictory ToggledOn goals: "
                f"{relation.subject!r}"
            )

    inside = tuple(item for item in goals if item.kind == "inside")
    opened = tuple(item for item in goals if item.kind == "open")
    closed = tuple(item for item in goals if item.kind == "closed")
    if {item.subject for item in opened}.intersection(
        item.subject for item in closed
    ):
        raise MixedGoalGraphError("one target has contradictory Open goals")
    spatial = tuple(
        item
        for item in goals
        if item.kind in _SPATIAL_RELATION_KINDS and item.kind != "inside"
    )
    toggles = tuple(
        item
        for item in goals
        if item.kind in {"toggled_on", "not_toggled_on"}
    )
    closed_by_target = {item.subject: item for item in closed}

    placements = (*inside, *spatial)
    active_placements = {
        item for item in placements if item not in initially_true
    }
    # Any placement moves its subject.  Re-establish every initially true
    # relation involving that moved subject, and every relation whose
    # reference moves.  Iterate to a fixed point because re-establishing a
    # dependent relation moves its own subject in turn.
    while True:
        moving_subjects = {
            item.subject for item in active_placements
        }
        expanded = active_placements.union(
            item
            for item in placements
            if item.subject in moving_subjects
            or item.reference in moving_subjects
        )
        if expanded == active_placements:
            break
        active_placements = expanded

    active_inside = tuple(
        item for item in inside if item in active_placements
    )
    active_spatial = tuple(
        item for item in spatial if item in active_placements
    )
    active_directed_spatial = tuple(
        item
        for item in active_spatial
        if item.kind not in _SYMMETRIC_SPATIAL_RELATION_KINDS
    )
    symmetric_achievements = normalize_symmetric_relation_achievements(
        (
            item
            for item in active_spatial
            if item.kind in _SYMMETRIC_SPATIAL_RELATION_KINDS
        ),
        immovable_targets=immovable,
    )
    active_toggles = tuple(
        item for item in toggles if item not in initially_true
    )
    active_opened = tuple(
        item for item in opened if item not in initially_true
    )

    # A terminal ``Inside(subject, container)`` plus
    # ``OnTop(subject, support)`` is physically compatible only when the
    # support is already co-located with that container.  Do *not* manufacture
    # an auxiliary ``place_inside(support, container)`` action: the support
    # may be fixed furniture or may not expose a fillable-compatible motion
    # contract at all.  Live official readback remains the authority for the
    # conjunction.  This keeps the graph a compiler for authored goal leaves,
    # rather than an executor that invents unscored hidden world mutations.
    inside_destination_by_subject = {
        item.subject: str(item.reference) for item in inside
    }
    for relation in active_spatial:
        if relation.kind != "on_top":
            continue
        subject_destination = inside_destination_by_subject.get(
            relation.subject
        )
        if subject_destination is None:
            continue
        support = str(relation.reference)
        support_destination = inside_destination_by_subject.get(support)
        if (
            support_destination is not None
            and support_destination != subject_destination
        ):
            raise MixedGoalGraphError(
                "OnTop support has a conflicting terminal Inside destination: "
                f"{support!r} -> {support_destination!r}, expected "
                f"{subject_destination!r}"
            )
    auxiliary_inside: tuple[MixedGoalRelation, ...] = ()

    active_destinations = {
        item.reference for item in (*active_inside, *auxiliary_inside)
    }
    assert None not in active_destinations
    active_access_subjects = {
        item.subject
        for item in (
            *active_inside,
            *active_spatial,
            *active_toggles,
            *active_opened,
        )
    }
    active_source_ancestors = {
        subject: source_ancestors.get(subject, ())
        for subject in sorted(active_access_subjects)
        if source_ancestors.get(subject, ())
    }
    must_open = {
        destination
        for destination in active_destinations
        if destination in openable or destination in closed_by_target
    }
    must_open.update(
        ancestor
        for ancestors in active_source_ancestors.values()
        for ancestor in ancestors
    )
    must_open.update(item.subject for item in active_opened)
    active_closed = tuple(
        item
        for item in closed
        if item not in initially_true or item.subject in must_open
    )
    active = set(active_inside).union(
        active_spatial, active_opened, active_closed, active_toggles
    )
    preserved = tuple(
        PreservationInvariant(item)
        for item in goals
        if item in initially_true and item not in active
    )

    commitments = tuple(
        DestinationCommitment(
            relation.subject,
            str(relation.reference),
            relation in initially_true,
        )
        for relation in inside
    )

    drafts: dict[tuple[str, str, str], _NodeDraft] = {}
    dependencies: dict[
        tuple[str, str, str], set[tuple[str, str, str]]
    ] = {}

    def add(
        action: str,
        target: str,
        destination: str | None = None,
        *,
        achieves: tuple[MixedGoalRelation, ...] = (),
        replay_on_recovery: bool = True,
    ) -> tuple[str, str, str]:
        draft = _NodeDraft(
            action,
            target,
            destination,
            achieves,
            replay_on_recovery,
        )
        key = draft.key
        previous = drafts.get(key)
        if previous is not None and previous != draft:
            raise MixedGoalGraphError(f"conflicting duplicate action node {key}")
        drafts[key] = draft
        dependencies.setdefault(key, set())
        return key

    active_opened_by_target = {
        relation.subject: relation for relation in active_opened
    }
    open_key = {
        target: add(
            "open",
            target,
            achieves=(active_opened_by_target[target],)
            if target in active_opened_by_target
            else (),
            replay_on_recovery=(target in active_destinations),
        )
        for target in sorted(must_open)
    }
    inside_key = {
        relation: add(
            "place_inside",
            relation.subject,
            relation.reference,
            achieves=(relation,),
        )
        for relation in active_inside
    }
    auxiliary_inside_key = {
        relation: add(
            "place_inside",
            relation.subject,
            relation.reference,
            achieves=(),
        )
        for relation in auxiliary_inside
    }
    directed_spatial_key = {
        relation: add(
            _ACTION_BY_RELATION_KIND[relation.kind],
            relation.subject,
            relation.reference,
            achieves=(relation,),
        )
        for relation in active_directed_spatial
    }
    symmetric_achievement_key = {
        achievement: add(
            _ACTION_BY_RELATION_KIND[achievement.kind],
            achievement.subject,
            achievement.reference,
            achieves=achievement.official_relations,
        )
        for achievement in symmetric_achievements
    }
    toggle_key = {
        relation: add(
            _ACTION_BY_RELATION_KIND[relation.kind],
            relation.subject,
            achieves=(relation,),
        )
        for relation in active_toggles
    }
    close_key = {
        relation.subject: add(
            "close",
            relation.subject,
            achieves=(relation,),
        )
        for relation in active_closed
    }

    for relation, key in (*inside_key.items(), *auxiliary_inside_key.items()):
        if relation.reference in open_key:
            dependencies[key].add(open_key[str(relation.reference)])

    spatial_placement_items = (
        *directed_spatial_key.items(),
        *(
            (
                MixedGoalRelation(
                    achievement.kind,
                    achievement.subject,
                    achievement.reference,
                ),
                key,
            )
            for achievement, key in symmetric_achievement_key.items()
        ),
    )
    placement_items = (
        *inside_key.items(),
        *auxiliary_inside_key.items(),
        *spatial_placement_items,
    )
    inside_nodes_by_destination: dict[str, set[tuple[str, str, str]]] = {}
    for relation, key in (*inside_key.items(), *auxiliary_inside_key.items()):
        inside_nodes_by_destination.setdefault(str(relation.reference), set()).add(
            key
        )

    placements_by_subject: dict[
        str, list[tuple[MixedGoalRelation, tuple[str, str, str]]]
    ] = {}
    for relation, key in placement_items:
        placements_by_subject.setdefault(relation.subject, []).append(
            (relation, key)
        )
    for items in placements_by_subject.values():
        items.sort(
            key=lambda item: (
                _SUBJECT_PLACEMENT_ORDER[item[0].kind],
                item[0].identity,
            )
        )

    first_placement_key_by_subject = {
        subject: items[0][1]
        for subject, items in placements_by_subject.items()
    }
    final_placement_key_by_subject = {
        subject: items[-1][1]
        for subject, items in placements_by_subject.items()
    }
    source_access_nodes_by_ancestor: dict[
        str, set[tuple[str, str, str]]
    ] = {}
    for subject, ancestors in active_source_ancestors.items():
        access_key = first_placement_key_by_subject.get(subject)
        if access_key is None:
            matching_toggles = tuple(
                key
                for relation, key in toggle_key.items()
                if relation.subject == subject
            )
            if len(matching_toggles) == 1:
                access_key = matching_toggles[0]
        if access_key is None:
            access_key = open_key.get(subject)
        if access_key is None:
            raise MixedGoalGraphError(
                "source openable access subject has no active operation"
            )
        previous_open: tuple[str, str, str] | None = None
        for ancestor in ancestors:
            ancestor_open = open_key.get(ancestor)
            if ancestor_open is None:
                raise MixedGoalGraphError(
                    "source openable access node is unavailable"
                )
            if previous_open is not None:
                dependencies[ancestor_open].add(previous_open)
            dependencies[access_key].add(ancestor_open)
            source_access_nodes_by_ancestor.setdefault(
                ancestor, set()
            ).add(access_key)
            previous_open = ancestor_open
    for subject, items in placements_by_subject.items():
        # Fill a movable container before moving the container to its own final
        # placement.  Later compatible constraints refine the same pose in a
        # deterministic order: primary support, Under, NextTo, Touching.
        dependencies[items[0][1]].update(
            inside_nodes_by_destination.get(subject, set())
        )
        for previous, current in zip(items, items[1:]):
            dependencies[current[1]].add(previous[1])

    nested_subjects = {
        relation.subject for relation in (*active_inside, *auxiliary_inside)
    }
    soft_close_dependencies = []
    for target, key in close_key.items():
        if target in open_key:
            dependencies[key].add(open_key[target])
        dependencies[key].update(inside_nodes_by_destination.get(target, set()))
        dependencies[key].update(
            source_access_nodes_by_ancestor.get(target, set())
        )
        if target in nested_subjects:
            # Close an inner container before carrying it into its parent.
            dependencies[first_placement_key_by_subject[target]].add(key)
        else:
            # All non-nested Closed goals are finalization actions.  This also
            # puts laptop close after laptop OnTop and cabinet close after all
            # unrelated active spatial placement has settled.
            soft_close_dependencies.extend(
                (key, draft_key)
                for draft_key, draft in drafts.items()
                if draft.action in _PLACEMENT_ACTIONS
            )

    # Source chains are outer-to-inner.  When multiple matching Closed goals
    # exist, close the inner access scope before its outer ancestor.  This is
    # derived only from live containment and grounded terminal leaves.
    for ancestors in active_source_ancestors.values():
        for outer, inner in zip(ancestors, ancestors[1:]):
            outer_close = close_key.get(outer)
            inner_close = close_key.get(inner)
            if outer_close is not None and inner_close is not None:
                dependencies[outer_close].add(inner_close)

    for relation, key in directed_spatial_key.items():
        reference_key = final_placement_key_by_subject.get(
            str(relation.reference)
        )
        if reference_key is not None and reference_key != key:
            dependencies[key].add(reference_key)

    # A rooted symmetric tree moves each non-root object only in the action
    # that relates it to its parent.  Reference finalization first preserves
    # every already-achieved reciprocal official leaf.
    for achievement, key in symmetric_achievement_key.items():
        reference_key = final_placement_key_by_subject.get(
            achievement.reference
        )
        if reference_key is not None and reference_key != key:
            dependencies[key].add(reference_key)

    for relation, key in toggle_key.items():
        placement_key = final_placement_key_by_subject.get(relation.subject)
        if placement_key is not None:
            dependencies[key].add(placement_key)

    # A stateful object must be toggled before a goal-Closed containment
    # ancestor is finalized.  Follow only grounded Inside commitments; no
    # category or task knowledge participates in this dependency.
    inside_parent = {
        relation.subject: str(relation.reference) for relation in inside
    }
    for relation, key in toggle_key.items():
        ancestor = inside_parent.get(relation.subject)
        visited: set[str] = set()
        while ancestor is not None and ancestor not in visited:
            visited.add(ancestor)
            ancestor_close = close_key.get(ancestor)
            if ancestor_close is not None:
                dependencies[ancestor_close].add(key)
            ancestor = inside_parent.get(ancestor)

    producers = {relation: key for key, draft in drafts.items() for relation in draft.achieves}
    # Existing auxiliary Open/Inside nodes also establish process prerequisites
    # even though they intentionally achieve no official final leaf.
    relation_kind_by_action = {action: kind for kind, action in _ACTION_BY_RELATION_KIND.items()}
    for key, draft in drafts.items():
        if draft.action in relation_kind_by_action:
            relation = MixedGoalRelation(relation_kind_by_action[draft.action], draft.target, draft.destination)
            producers.setdefault(relation, key)
    for before, after in precedence:
        if before not in producers and before not in initially_true:
            raise MixedGoalGraphError("unsatisfied process prerequisite has no goal-plan producer")
        if after in producers and before in producers:
            dependencies[producers[after]].add(producers[before])
    # Check true dependency contradictions before optional finalization ordering.
    if precedence:
        _topological_nodes(drafts, dependencies)
    def depends_on(node, prerequisite):
        pending, seen = [node], set()
        while pending:
            current = pending.pop()
            if current == prerequisite:
                return True
            if current not in seen:
                seen.add(current)
                pending.extend(dependencies[current])
        return False
    for close, placement in soft_close_dependencies:
        if not precedence or not depends_on(placement, close):
            dependencies[close].add(placement)
    nodes = _topological_nodes(drafts, dependencies)
    return GroundedMixedGoalPlan(
        option_index=option_index,
        goal_relations=goals,
        destination_commitments=commitments,
        preservation_invariants=preserved,
        action_nodes=nodes,
        leaf_count=len(goals),
    )


def iter_feasible_grounded_mixed_plans(
    options: Iterable[Sequence[Any]],
    *,
    openable_targets: Iterable[str] = (),
    immovable_targets: Iterable[str] = (),
    initially_true_relations: Iterable[Any] = (),
    source_openable_ancestors_by_subject: Mapping[
        str, Iterable[str]
    ] | None = None,
) -> Iterator[GroundedMixedGoalPlan]:
    """Yield feasible plans in the official grounded-option order."""

    openable = tuple(openable_targets)
    immovable = tuple(immovable_targets)
    initially_true = tuple(initially_true_relations)
    for option_index, option in enumerate(options):
        try:
            yield compile_grounded_mixed_option(
                option,
                option_index=option_index,
                openable_targets=openable,
                immovable_targets=immovable,
                initially_true_relations=initially_true,
                source_openable_ancestors_by_subject=(
                    source_openable_ancestors_by_subject
                ),
            )
        except MixedGoalGraphError:
            continue


def feasible_grounded_mixed_plans(
    options: Iterable[Sequence[Any]],
    *,
    openable_targets: Iterable[str] = (),
    immovable_targets: Iterable[str] = (),
    initially_true_relations: Iterable[Any] = (),
    source_openable_ancestors_by_subject: Mapping[
        str, Iterable[str]
    ] | None = None,
) -> tuple[GroundedMixedGoalPlan, ...]:
    """Materialize all structurally feasible mixed plans for bounded tests."""

    return tuple(
        iter_feasible_grounded_mixed_plans(
            options,
            openable_targets=openable_targets,
            immovable_targets=immovable_targets,
            initially_true_relations=initially_true_relations,
            source_openable_ancestors_by_subject=(
                source_openable_ancestors_by_subject
            ),
        )
    )


def first_feasible_grounded_mixed_plan(
    options: Iterable[Sequence[Any]],
    *,
    openable_targets: Iterable[str] = (),
    immovable_targets: Iterable[str] = (),
    initially_true_relations: Iterable[Any] = (),
    source_openable_ancestors_by_subject: Mapping[
        str, Iterable[str]
    ] | None = None,
) -> GroundedMixedGoalPlan:
    """Commit to the first feasible option in official deterministic order."""

    for plan in iter_feasible_grounded_mixed_plans(
        options,
        openable_targets=openable_targets,
        immovable_targets=immovable_targets,
        initially_true_relations=initially_true_relations,
        source_openable_ancestors_by_subject=(
            source_openable_ancestors_by_subject
        ),
    ):
        return plan
    raise MixedGoalGraphError("no feasible grounded mixed goal option")
