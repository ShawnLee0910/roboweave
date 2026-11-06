"""Task-agnostic executable admission for authored grounded-goal witnesses.

The canonical goal compilers prove that a grounded option is structurally
representable.  They intentionally do not know which public actions are
registered or which executor preconditions are impossible by construction.
This module is the narrow CPU-only bridge between those contracts.

It does not inspect simulator state, task names, object categories, assets, or
private geometry.  Live predicate truth and success remain owned by the
official task and verifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from behavior_hl.embodied.policy.registry import ActionRegistry


EXECUTABLE_GROUNDED_OPTION_ADMISSION_VERSION = (
    "behavior2026.executable-grounded-option-admission/v1"
)

_SPATIAL_PREDICATES = frozenset(
    {
        "inside",
        "ontop",
        "on_top",
        "nextto",
        "next_to",
        "under",
        "touching",
    }
)
_PLACEMENT_ACTIONS = frozenset(
    {
        "place_inside",
        "place_on_top",
        "place_next_to",
        "place_under",
        "place_touching",
    }
)


def _leaf_body(leaf: Any) -> list[Any]:
    value = getattr(leaf, "body", leaf)
    if not isinstance(value, (list, tuple)) or not value:
        return []
    return list(value)


def _positive_spatial_edge(leaf: Any) -> tuple[str, str, str] | None:
    body = _leaf_body(leaf)
    if not body or not isinstance(body[0], str):
        return None
    if body[0].casefold() == "not":
        return None
    predicate = body[0].casefold()
    if predicate not in _SPATIAL_PREDICATES or len(body) != 3:
        return None
    subject, reference = body[1], body[2]
    if not all(
        isinstance(value, str) and value
        for value in (subject, reference)
    ):
        return None
    canonical = {
        "ontop": "on_top",
        "nextto": "next_to",
    }.get(predicate, predicate)
    return canonical, subject, reference


def _contains_directed_cycle(edges: Iterable[tuple[str, str]]) -> bool:
    adjacency: dict[str, set[str]] = {}
    for source, destination in edges:
        adjacency.setdefault(source, set()).add(destination)
        adjacency.setdefault(destination, set())
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for child in adjacency.get(node, ()):
            if visit(child):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in sorted(adjacency))


def _plan_operations(plan: Any) -> tuple[tuple[str, str, str | None], ...]:
    action_nodes = getattr(plan, "action_nodes", None)
    if isinstance(action_nodes, (list, tuple)):
        return tuple(
            (
                str(getattr(node, "action", "")),
                str(getattr(node, "target", "")),
                (
                    None
                    if getattr(node, "destination", None) is None
                    else str(getattr(node, "destination"))
                ),
            )
            for node in action_nodes
        )
    operations = getattr(plan, "operations", None)
    if isinstance(operations, (list, tuple)):
        return tuple(
            (
                str(getattr(operation, "kind", "")),
                str(getattr(operation, "target", "")),
                (
                    None
                    if getattr(operation, "destination", None) is None
                    else str(getattr(operation, "destination"))
                ),
            )
            for operation in operations
        )
    raise TypeError("compiled goal plan exposes no canonical action sequence")


@dataclass(frozen=True, slots=True)
class ExecutableGroundedOptionAdmission:
    """Frozen Action23 registry view plus generic executable constraints."""

    action_names: frozenset[str]
    controller_keys: tuple[tuple[str, str], ...]
    schema_version: str = EXECUTABLE_GROUNDED_OPTION_ADMISSION_VERSION

    @classmethod
    def from_registry(
        cls,
        registry: ActionRegistry,
    ) -> "ExecutableGroundedOptionAdmission":
        if not isinstance(registry, ActionRegistry):
            raise TypeError("executable admission requires an ActionRegistry")
        controller_keys = tuple(
            (name, registry.spec(name).controller_key)
            for name in registry.names
        )
        if any(not key for _, key in controller_keys):
            raise ValueError("registered action has no controller key")
        return cls(
            action_names=frozenset(registry.names),
            controller_keys=controller_keys,
        )

    def atomic_leaf_gate(self, leaf: Any) -> bool:
        """Prune a positive spatial self-relation before product expansion."""

        edge = _positive_spatial_edge(leaf)
        return edge is None or edge[1] != edge[2]

    def compiled_candidate_gate(
        self,
        option: Sequence[Any],
        plan: Any,
    ) -> tuple[bool, str]:
        """Admit only a registry-routable, non-contradictory action plan."""

        edges = tuple(
            edge
            for leaf in option
            if (edge := _positive_spatial_edge(leaf)) is not None
        )
        if any(subject == reference for _, subject, reference in edges):
            return False, "spatial_subject_equals_reference"

        inside_edges = tuple(
            (subject, reference)
            for kind, subject, reference in edges
            if kind == "inside"
        )
        if _contains_directed_cycle(inside_edges):
            return False, "cyclic_inside_commitment"

        # Normalize OnTop(A, B) and Under(B, A) to the same "above" edge.
        # A cycle in this strict ordering is physically contradictory.
        vertical_edges = tuple(
            (subject, reference)
            if kind == "on_top"
            else (reference, subject)
            for kind, subject, reference in edges
            if kind in {"on_top", "under"}
        )
        if _contains_directed_cycle(vertical_edges):
            return False, "cyclic_vertical_commitment"

        try:
            operations = _plan_operations(plan)
        except (TypeError, ValueError):
            return False, "compiled_plan_has_no_action_sequence"
        for action, target, destination in operations:
            if action not in self.action_names:
                return False, "compiled_action_not_registered"
            if not target:
                return False, "compiled_action_has_empty_target"
            if action in _PLACEMENT_ACTIONS:
                if not destination:
                    return False, "placement_has_no_destination"
                if target == destination:
                    return False, "placement_target_equals_destination"
            elif destination is not None:
                return False, "nonplacement_has_destination"
        return True, "executable_action_graph"


def build_action23_executable_admission(
    registry: ActionRegistry,
) -> ExecutableGroundedOptionAdmission:
    """Build the common admission contract from the active action registry."""

    return ExecutableGroundedOptionAdmission.from_registry(registry)
