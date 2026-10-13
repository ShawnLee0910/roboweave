"""Pure logical-inventory carry bookkeeping.

This module deliberately knows nothing about OmniGibson, PhysX, task names, or
public action envelopes.  The live symbolic executor supplies object handles,
poses, the official ``Inside`` predicate, and matrix operations.  Keeping the
closure algorithm here makes the nested-carry invariant testable without
starting Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable


_RELATION_KINDS = frozenset({"inside", "on_top", "attached_to"})


class LogicalInventoryError(RuntimeError):
    """A malformed or unsafe logical inventory closure."""


@dataclass(frozen=True, slots=True)
class LogicalInventoryRelation:
    """One directed relation retained while a logical bundle is carried."""

    kind: str
    child: Any
    parent: Any

    def __post_init__(self) -> None:
        if self.kind not in _RELATION_KINDS:
            raise ValueError(f"unsupported logical inventory relation {self.kind!r}")
        if self.child is None or self.parent is None or self.child is self.parent:
            raise ValueError("logical inventory relation endpoints must be distinct")


@dataclass(frozen=True)
class LogicalInventoryEntry:
    """One descendant and its transform relative to the captured root."""

    object_ref: Any
    origin_pose: Any
    relative_to_root: Any


@dataclass(frozen=True)
class LogicalInventoryRestoreReport:
    """Best-effort restore result; errors are retained instead of hidden."""

    attempted_count: int
    restored_count: int
    first_error: BaseException | None = None

    @property
    def ok(self) -> bool:
        return self.first_error is None

    def as_private_dict(self) -> dict[str, Any]:
        return {
            "logical_inventory_restored_count": int(self.restored_count),
            "logical_inventory_restore_error": (
                None
                if self.first_error is None
                else f"{type(self.first_error).__name__}: {self.first_error}"
            ),
        }


@dataclass(frozen=True)
class LogicalInventorySnapshot:
    """Captured transitive relation closure for one logical root.

    The transform for every descendant is always measured directly from the
    root origin.  This avoids both order dependence and numerical drift when a
    nested object happens to be present in more than one scope alias.
    """

    root: Any
    root_origin_pose: Any
    entries: tuple[LogicalInventoryEntry, ...]
    containment_edges: tuple[tuple[Any, Any], ...]
    relation_edges: tuple[LogicalInventoryRelation, ...] = ()
    root_private_pose: Any | None = None
    sync_count: int = 0

    VERSION = "v3"

    @property
    def descendant_count(self) -> int:
        return len(self.entries)

    @property
    def object_refs(self) -> tuple[Any, ...]:
        return (self.root, *(entry.object_ref for entry in self.entries))

    @classmethod
    def capture(
        cls,
        *,
        root: Any,
        candidates: Iterable[Any],
        get_pose: Callable[[Any], Any],
        is_inside: Callable[[Any, Any], bool],
        copy_pose: Callable[[Any], Any],
        inverse_pose: Callable[[Any], Any],
        compose_pose: Callable[[Any, Any], Any],
        valid_pose: Callable[[Any], bool],
        extra_relations: Iterable[LogicalInventoryRelation] = (),
        relation_holds: Callable[[LogicalInventoryRelation], bool] | None = None,
    ) -> "LogicalInventorySnapshot":
        """Capture a root closure before any object pose is mutated.

        ``candidates`` is injected by the private runtime. Only predicates
        reachable from ``root`` are evaluated and retained.
        """

        if root is None:
            raise LogicalInventoryError("logical inventory root is missing")
        requested_relations = tuple(extra_relations)
        if any(
            not isinstance(relation, LogicalInventoryRelation)
            for relation in requested_relations
        ):
            raise TypeError(
                "extra_relations must contain LogicalInventoryRelation values"
            )
        if requested_relations and not callable(relation_holds):
            raise LogicalInventoryError(
                "selected bundle relations require a live relation reader"
            )

        unique: list[Any] = []
        seen_ids: set[int] = set()
        relation_endpoints = (
            endpoint
            for relation in requested_relations
            for endpoint in (relation.parent, relation.child)
        )
        for candidate in (*tuple(candidates), *tuple(relation_endpoints)):
            if candidate is None:
                continue
            marker = id(candidate)
            if marker in seen_ids:
                continue
            seen_ids.add(marker)
            unique.append(candidate)
        if id(root) not in seen_ids:
            unique.insert(0, root)
            seen_ids.add(id(root))

        def checked_pose(obj: Any) -> Any:
            try:
                pose = copy_pose(get_pose(obj))
            except Exception as exc:  # pragma: no cover - exact provider errors vary
                raise LogicalInventoryError(
                    "logical inventory object pose is unavailable"
                ) from exc
            if not valid_pose(pose):
                raise LogicalInventoryError(
                    "logical inventory object pose is malformed or non-finite"
                )
            return pose

        root_origin = checked_pose(root)
        root_inverse = inverse_pose(root_origin)
        if not valid_pose(root_inverse):
            raise LogicalInventoryError("logical inventory root inverse is invalid")

        # Validate selected relations before reading or mutating descendant
        # poses. They seed the root-reachable graph below.
        edge_list: list[LogicalInventoryRelation] = []
        for relation in requested_relations:
            assert relation_holds is not None
            try:
                live = bool(relation_holds(relation))
            except Exception as exc:
                raise LogicalInventoryError(
                    f"official {relation.kind} predicate failed during bundle capture"
                ) from exc
            if not live:
                raise LogicalInventoryError(
                    f"selected {relation.kind} bundle relation is no longer true"
                )
            edge_list.append(relation)

        children_by_parent: dict[int, list[Any]] = {
            id(candidate): [] for candidate in unique
        }
        deduplicated_edges: list[LogicalInventoryRelation] = []
        seen_edges: set[tuple[str, int, int]] = set()

        def add_relation(relation: LogicalInventoryRelation) -> None:
            identity = (relation.kind, id(relation.child), id(relation.parent))
            if identity in seen_edges:
                return
            seen_edges.add(identity)
            deduplicated_edges.append(relation)
            children_by_parent[id(relation.parent)].append(relation.child)

        for relation in edge_list:
            add_relation(relation)

        scanned_inside_parents: set[int] = set()

        def scan_inside_children(parent: Any) -> None:
            parent_id = id(parent)
            if parent_id in scanned_inside_parents:
                return
            scanned_inside_parents.add(parent_id)
            for child in unique:
                if child is parent:
                    continue
                try:
                    contained = bool(is_inside(child, parent))
                except Exception as exc:
                    raise LogicalInventoryError(
                        "official Inside predicate failed during closure capture"
                    ) from exc
                if contained:
                    add_relation(LogicalInventoryRelation("inside", child, parent))

        # DFS from the root detects cycles instead of silently deduplicating
        # them.  A diamond is safe and is emitted only once in ``entries``.
        colours: dict[int, int] = {}
        ordered_descendants: list[Any] = []

        def visit(parent: Any) -> None:
            parent_id = id(parent)
            colour = colours.get(parent_id, 0)
            if colour == 1:
                raise LogicalInventoryError(
                    "logical inventory relation cycle detected"
                )
            if colour == 2:
                return
            colours[parent_id] = 1
            scan_inside_children(parent)
            for child in children_by_parent.get(parent_id, ()):
                child_id = id(child)
                if colours.get(child_id, 0) == 1:
                    raise LogicalInventoryError(
                        "logical inventory relation cycle detected"
                    )
                if colours.get(child_id, 0) == 0:
                    ordered_descendants.append(child)
                    visit(child)
            colours[parent_id] = 2

        visit(root)

        entries: list[LogicalInventoryEntry] = []
        for child in ordered_descendants:
            child_origin = checked_pose(child)
            try:
                relative = compose_pose(root_inverse, child_origin)
            except Exception as exc:
                raise LogicalInventoryError(
                    "logical inventory relative transform construction failed"
                ) from exc
            if not valid_pose(relative):
                raise LogicalInventoryError(
                    "logical inventory relative transform is invalid"
                )
            entries.append(
                LogicalInventoryEntry(
                    object_ref=child,
                    origin_pose=child_origin,
                    relative_to_root=copy_pose(relative),
                )
            )

        # Keep only edges in the root-reachable closure, in deterministic
        # candidate order, for private post-placement integrity telemetry.
        reachable = {id(root), *(id(entry.object_ref) for entry in entries)}
        relations = tuple(
            relation
            for relation in deduplicated_edges
            if id(relation.child) in reachable and id(relation.parent) in reachable
        )
        return cls(
            root=root,
            root_origin_pose=root_origin,
            entries=tuple(entries),
            containment_edges=tuple(
                (relation.child, relation.parent)
                for relation in relations
                if relation.kind == "inside"
            ),
            relation_edges=relations,
        )

    def with_private_pose(
        self,
        private_pose: Any,
        *,
        copy_pose: Callable[[Any], Any],
        valid_pose: Callable[[Any], bool],
    ) -> "LogicalInventorySnapshot":
        if not valid_pose(private_pose):
            raise LogicalInventoryError("logical inventory private pose is invalid")
        return LogicalInventorySnapshot(
            root=self.root,
            root_origin_pose=self.root_origin_pose,
            entries=self.entries,
            containment_edges=self.containment_edges,
            relation_edges=self.relation_edges,
            root_private_pose=copy_pose(private_pose),
            sync_count=self.sync_count,
        )

    def apply_root_pose(
        self,
        root_pose: Any,
        *,
        set_pose: Callable[[Any, Any], None],
        copy_pose: Callable[[Any], Any],
        compose_pose: Callable[[Any, Any], Any],
        valid_pose: Callable[[Any], bool],
    ) -> "LogicalInventorySnapshot":
        """Apply one root pose and every descendant root-relative pose."""

        root_pose_copy = copy_pose(root_pose)
        if not valid_pose(root_pose_copy):
            raise LogicalInventoryError("logical inventory sync root pose is invalid")
        poses: list[tuple[Any, Any]] = [(self.root, root_pose_copy)]
        for entry in self.entries:
            child_pose = compose_pose(root_pose_copy, entry.relative_to_root)
            if not valid_pose(child_pose):
                raise LogicalInventoryError(
                    "logical inventory sync produced an invalid child pose"
                )
            poses.append((entry.object_ref, copy_pose(child_pose)))
        # Validate every pose before mutating any object.  If a setter itself
        # fails, the caller's restore path still attempts all objects.
        for obj, pose in poses:
            set_pose(obj, pose)
        return LogicalInventorySnapshot(
            root=self.root,
            root_origin_pose=self.root_origin_pose,
            entries=self.entries,
            containment_edges=self.containment_edges,
            relation_edges=self.relation_edges,
            root_private_pose=root_pose_copy,
            sync_count=self.sync_count + 1,
        )

    def restore(
        self,
        *,
        set_pose: Callable[[Any, Any], None],
        copy_pose: Callable[[Any], Any],
    ) -> LogicalInventoryRestoreReport:
        """Best-effort restore of root and all descendants.

        Every setter is attempted even after one failure.  This is intentionally
        non-throwing so phase cleanup can clear internal state and report the
        first failure without leaving later descendants at the private pose.
        """

        targets = [(self.root, self.root_origin_pose)] + [
            (entry.object_ref, entry.origin_pose) for entry in self.entries
        ]
        first_error: BaseException | None = None
        restored = 0
        for obj, pose in targets:
            try:
                set_pose(obj, copy_pose(pose))
                restored += 1
            except BaseException as exc:  # cleanup must continue for all objects
                if first_error is None:
                    first_error = exc
        return LogicalInventoryRestoreReport(
            attempted_count=len(targets),
            restored_count=restored,
            first_error=first_error,
        )


def finite_matrix_pose(pose: Any) -> bool:
    """Numpy-friendly validation helper used by pure tests and adapters."""

    try:
        import numpy as np

        matrix = np.asarray(pose)
    except Exception:
        return False
    return bool(matrix.shape == (4, 4) and np.isfinite(matrix).all())
