"""CPU-only batch audit of frozen selected-option goal DAGs.

The auditor reads only profile catalogs and composes the existing pure
``inside_goal_graph`` and ``mixed_goal_graph`` compilers.  It never imports
OmniGibson, loads a scene, reads trajectory artifacts, or invents live
geometry.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from behavior_hl.embodied.evaluator_bridge.inside_goal_graph import (
    GroundedInsideGoalPlan,
    InsideGoalGraphError,
    compile_grounded_inside_option,
)
from behavior_hl.embodied.evaluator_bridge.mixed_goal_graph import (
    GroundedMixedGoalPlan,
    MixedGoalGraphError,
    compile_grounded_mixed_option,
    relation_from_grounded_leaf,
)


AUDIT_SCHEMA = "behavior2026.prerender-dag-audit/v1"
CATALOG_SCHEMAS = frozenset(
    {
        "behavior2026.inside27-profile-catalog/v1",
        "behavior2026.priority-mixed-profile-catalog/v1",
        "behavior2026.action23-oracle-profile-catalog/v1",
    }
)


class PrerenderDagAuditError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AuditNode:
    node_id: str
    action: str
    target: str
    destination: str | None
    dependencies: tuple[str, ...]
    achieves: tuple[str, ...]


def _leaf_body(value: Any) -> list[Any]:
    if not isinstance(value, (list, tuple)) or not value:
        raise PrerenderDagAuditError("canonical leaf must be a non-empty body")
    return list(value)


def _leaf_identity(value: Any) -> str:
    return json.dumps(
        _leaf_body(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _semantic_leaf_identity(value: Any) -> str:
    """Use the shared mixed compiler's canonical predicate spelling."""

    body = _leaf_body(value)
    if body[0] == "not":
        child = _leaf_body(body[1])
        if (
            len(child) == 3
            and isinstance(child[0], str)
            and child[0].casefold() == "inside"
        ):
            return json.dumps(
                ("not_inside", child[1], child[2]),
                separators=(",", ":"),
                allow_nan=False,
            )
    relation = relation_from_grounded_leaf(value)
    return json.dumps(
        relation.identity, separators=(",", ":"), allow_nan=False
    )


def _subjects_and_references(leaves: Sequence[Any]) -> frozenset[str]:
    values: set[str] = set()
    for leaf in leaves:
        body = _leaf_body(leaf)
        if body[0] == "not":
            body = _leaf_body(body[1])
        for value in body[1:]:
            if isinstance(value, str) and value:
                values.add(value)
    return frozenset(values)


def _inside_nodes(plan: GroundedInsideGoalPlan) -> tuple[AuditNode, ...]:
    nodes: list[AuditNode] = []
    placement_by_child: dict[str, str] = {}
    latest_open_by_target: dict[str, str] = {}
    latest_close_by_target: dict[str, str] = {}
    for index, operation in enumerate(plan.operations):
        node_id = f"inside:{index:04d}:{operation.kind}:{operation.target}"
        dependencies: set[str] = set()
        achieves: tuple[str, ...] = ()
        if operation.kind == "open":
            latest_open_by_target[operation.target] = node_id
        elif operation.kind == "place_inside":
            assert operation.destination is not None
            parent_open = latest_open_by_target.get(operation.destination)
            if parent_open is not None:
                dependencies.add(parent_open)
            child_close = latest_close_by_target.get(operation.target)
            if child_close is not None:
                dependencies.add(child_close)
            for relation in plan.positive_inside:
                if relation.parent == operation.target:
                    dependency = placement_by_child.get(relation.child)
                    if dependency is not None:
                        dependencies.add(dependency)
            placement_by_child[operation.target] = node_id
            achieves = (
                _semantic_leaf_identity(
                    ["inside", operation.target, operation.destination]
                ),
            )
        elif operation.kind == "close":
            dependencies.update(
                node.node_id
                for node in nodes
                if node.action == "place_inside"
                and node.destination == operation.target
            )
            latest_close_by_target[operation.target] = node_id
            achieves = (
                _semantic_leaf_identity(["not", ["open", operation.target]]),
            )
        nodes.append(
            AuditNode(
                node_id=node_id,
                action=operation.kind,
                target=operation.target,
                destination=operation.destination,
                dependencies=tuple(sorted(dependencies)),
                achieves=achieves,
            )
        )
    return tuple(nodes)


def _mixed_nodes(plan: GroundedMixedGoalPlan) -> tuple[AuditNode, ...]:
    return tuple(
        AuditNode(
            node_id=node.node_id,
            action=node.action,
            target=node.target,
            destination=node.destination,
            dependencies=node.dependencies,
            achieves=tuple(
                json.dumps(
                    relation.identity,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                for relation in node.achieves
            ),
        )
        for node in plan.action_nodes
    )


def _graph_checks(
    *, leaves: Sequence[Any], nodes: Sequence[AuditNode]
) -> Mapping[str, Any]:
    node_ids = tuple(node.node_id for node in nodes)
    position = {node_id: index for index, node_id in enumerate(node_ids)}
    unknown_dependencies = tuple(
        sorted(
            {dependency for node in nodes for dependency in node.dependencies}
            - set(node_ids)
        )
    )
    cycle_or_non_topological = bool(
        unknown_dependencies
        or any(
            position[dependency] >= position[node.node_id]
            for node in nodes
            for dependency in node.dependencies
            if dependency in position
        )
    )
    achievers: Counter[str] = Counter(
        identity for node in nodes for identity in node.achieves
    )
    leaf_identities = tuple(_semantic_leaf_identity(leaf) for leaf in leaves)
    preserved_kinds = {"ontop"}
    executable_or_preserved: set[str] = set(achievers)
    for leaf, identity in zip(leaves, leaf_identities, strict=True):
        body = _leaf_body(leaf)
        if body[0] in preserved_kinds:
            executable_or_preserved.add(identity)
        if body[0] == "not" and _leaf_body(body[1])[0] == "inside":
            executable_or_preserved.add(identity)
    unbound = tuple(
        identity for identity in leaf_identities if identity not in executable_or_preserved
    )
    multiple = tuple(sorted(identity for identity, count in achievers.items() if count > 1))
    open_before_place = True
    terminal_close = True
    for index, node in enumerate(nodes):
        if node.action == "open":
            placements = [
                pos
                for pos, other in enumerate(nodes)
                if other.action == "place_inside" and other.destination == node.target
            ]
            open_before_place &= not placements or index < min(placements)
        if node.action == "close":
            placements = [
                pos
                for pos, other in enumerate(nodes)
                if other.action == "place_inside" and other.destination == node.target
            ]
            terminal_close &= not placements or index > max(placements)
    return {
        "unbound_leaf_ids": list(unbound),
        "multiple_achiever_leaf_ids": list(multiple),
        "unknown_dependency_node_ids": list(unknown_dependencies),
        "cycle_or_non_topological": cycle_or_non_topological,
        "open_before_destination_place": open_before_place,
        "terminal_close_after_destination_places": terminal_close,
    }


def audit_profile(record: Mapping[str, Any], *, catalog: str) -> Mapping[str, Any]:
    leaves = tuple(record.get("canonical_goal_leaves", ()))
    expected = record.get("expected_goal_leaf_count")
    if not leaves or expected != len(leaves):
        raise PrerenderDagAuditError("canonical selected option leaf count drifted")
    signatures = tuple(record.get("goal_signatures", ()))
    mixed = any(signature.startswith("Mixed") for signature in signatures)
    identifiers = _subjects_and_references(leaves)
    closed_targets = {
        _leaf_body(_leaf_body(leaf)[1])[1]
        for leaf in leaves
        if _leaf_body(leaf)[0] == "not"
        and _leaf_body(_leaf_body(leaf)[1])[0] == "open"
    }
    openable_targets = frozenset(closed_targets)
    try:
        if mixed:
            plan = compile_grounded_mixed_option(
                leaves,
                option_index=int(record["canonical_option_index"]),
                openable_targets=openable_targets,
            )
            nodes = _mixed_nodes(plan)
        else:
            plan = compile_grounded_inside_option(
                leaves,
                option_index=int(record["canonical_option_index"]),
                openable_targets=openable_targets,
            )
            nodes = _inside_nodes(plan)
        compile_error = None
    except (InsideGoalGraphError, MixedGoalGraphError, ValueError) as exc:
        nodes = ()
        compile_error = f"{type(exc).__name__}: {exc}"
    action_types = tuple(sorted({node.action for node in nodes}))
    spatial_actions = {"place_on_top", "place_next_to", "place_under"}
    live_reasons: list[str] = []
    if any(action in spatial_actions for action in action_types):
        live_reasons.append("spatial placement requires live support/collision geometry")
    if closed_targets:
        live_reasons.append("openable binding and initial Open truth require live state")
    if any(_leaf_body(leaf)[0] == "ontop" for leaf in leaves):
        live_reasons.append("initial preservation truth requires live verifier")
    if any(_leaf_body(leaf)[0] == "not" and _leaf_body(_leaf_body(leaf)[1])[0] == "inside" for leaf in leaves):
        live_reasons.append("NotInside initial truth requires live verifier")
    checks = _graph_checks(leaves=leaves, nodes=nodes) if nodes else {
        "unbound_leaf_ids": [_leaf_identity(leaf) for leaf in leaves],
        "multiple_achiever_leaf_ids": [],
        "unknown_dependency_node_ids": [],
        "cycle_or_non_topological": True,
        "open_before_destination_place": False,
        "terminal_close_after_destination_places": False,
    }
    return {
        "catalog": catalog,
        "task_name": record["task_name"],
        "task_id": record["task_id"],
        "scene_model": record["scene_model"],
        "split": record["split"],
        "instance_id": record["instance_id"],
        "selected_option_index": record["canonical_option_index"],
        "selected_option_leaf_ids": [_leaf_identity(leaf) for leaf in leaves],
        "deterministic_nodes": [asdict(node) for node in nodes],
        "action_type_coverage": list(action_types),
        "bound_object_id_count": len(identifiers),
        "checks": dict(checks),
        "compile_error": compile_error,
        "requires_live_geometry": bool(live_reasons),
        "live_geometry_reasons": live_reasons,
    }


def audit_catalogs(paths: Iterable[Path]) -> Mapping[str, Any]:
    records: list[Mapping[str, Any]] = []
    sources: list[str] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") not in CATALOG_SCHEMAS:
            raise PrerenderDagAuditError(f"unsupported catalog schema: {path}")
        tasks = data.get("tasks")
        if not isinstance(tasks, list) or data.get("task_count") != len(tasks):
            raise PrerenderDagAuditError(f"catalog task count drifted: {path}")
        sources.append(str(path.resolve()))
        records.extend(
            audit_profile(record, catalog=path.name) for record in tasks
        )
    records.sort(key=lambda item: (item["task_id"], item["task_name"]))
    failed = [record for record in records if record["compile_error"] is not None]
    return {
        "schema_version": AUDIT_SCHEMA,
        "cpu_only": True,
        "simulator_imported": False,
        "catalogs": sources,
        "profile_count": len(records),
        "compile_failure_count": len(failed),
        "requires_live_geometry_count": sum(
            bool(record["requires_live_geometry"]) for record in records
        ),
        "profiles": records,
    }


def write_report(report: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
