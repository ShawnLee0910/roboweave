"""Pure contract for clearing a retained held object after a failed goal node.

The contract is predicate agnostic.  It does not move an object, inspect task
names, or import the simulator.  A live Oracle adapter remains responsible for
capturing / restoring a pre-operation checkpoint or for generating officially
legal staging candidates from live geometry.  This module only decides which
bounded clearance action may be attempted and which independent goal leaves
may continue after authoritative readback confirms that the hand is clear and
that no previously true selected-option leaf regressed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence


HELD_RECOVERY_STAGING_VERSION = "held_recovery_staging/v1"


class HeldRecoveryStagingError(ValueError):
    """Raised when a held-recovery request violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class HeldRecoveryStagingDirective:
    """One fail-closed held-object clearance / continuation decision."""

    version: str
    action: str
    failed_leaf_id: str
    held_object_scope_id: str | None
    staging_target_scope_id: str | None
    selected_option_leaf_ids: tuple[str, ...]
    deferred_leaf_ids: tuple[str, ...]
    independent_ready_leaf_ids: tuple[str, ...]
    before_true_selected_leaf_ids: tuple[str, ...]
    restore_attempt_number: int | None
    staging_attempt_number: int | None
    accepted: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalize_leaf_contract(
    *,
    required_leaf_ids: Sequence[str],
    truth_by_leaf_id: Mapping[str, bool],
    dependencies_by_leaf: Mapping[str, Iterable[str]],
) -> tuple[tuple[str, ...], dict[str, bool], dict[str, tuple[str, ...]]]:
    if not isinstance(required_leaf_ids, (list, tuple)) or not required_leaf_ids:
        raise HeldRecoveryStagingError("required leaf IDs must be non-empty")
    leaves = tuple(required_leaf_ids)
    if any(not isinstance(leaf_id, str) or not leaf_id for leaf_id in leaves):
        raise HeldRecoveryStagingError("leaf IDs must be non-empty strings")
    if len(set(leaves)) != len(leaves):
        raise HeldRecoveryStagingError("required leaf IDs must be unique")
    if set(truth_by_leaf_id) != set(leaves):
        raise HeldRecoveryStagingError(
            "truth mapping must exactly match the selected option leaves"
        )
    if any(not isinstance(value, bool) for value in truth_by_leaf_id.values()):
        raise HeldRecoveryStagingError("authoritative leaf truth must be boolean")
    unknown_dependency_owners = set(dependencies_by_leaf) - set(leaves)
    if unknown_dependency_owners:
        raise HeldRecoveryStagingError("dependency owner is outside selected option")
    normalized_dependencies: dict[str, tuple[str, ...]] = {}
    for leaf_id in leaves:
        prerequisites = tuple(dependencies_by_leaf.get(leaf_id, ()))
        if len(set(prerequisites)) != len(prerequisites):
            raise HeldRecoveryStagingError("leaf prerequisites must be unique")
        if any(
            not isinstance(prerequisite, str) or prerequisite not in set(leaves)
            for prerequisite in prerequisites
        ):
            raise HeldRecoveryStagingError(
                "leaf dependency references an unknown prerequisite"
            )
        if leaf_id in prerequisites:
            raise HeldRecoveryStagingError("leaf cannot depend on itself")
        normalized_dependencies[leaf_id] = prerequisites

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(leaf_id: str) -> None:
        if leaf_id in visiting:
            raise HeldRecoveryStagingError("selected-option dependencies contain a cycle")
        if leaf_id in visited:
            return
        visiting.add(leaf_id)
        for prerequisite in normalized_dependencies[leaf_id]:
            visit(prerequisite)
        visiting.remove(leaf_id)
        visited.add(leaf_id)

    for leaf_id in leaves:
        visit(leaf_id)
    return leaves, dict(truth_by_leaf_id), normalized_dependencies


def independent_ready_leaf_ids(
    *,
    required_leaf_ids: Sequence[str],
    truth_by_leaf_id: Mapping[str, bool],
    dependencies_by_leaf: Mapping[str, Iterable[str]],
    failed_leaf_id: str,
) -> tuple[str, ...]:
    """Return false, ready leaves whose dependency closure excludes failure.

    The frozen selected-option order is the deterministic tie-breaker.  A leaf
    is ready only when every direct prerequisite is authoritatively true.  A
    leaf depending transitively on the failed leaf is deliberately deferred.
    """

    leaves, truth, dependencies = _normalize_leaf_contract(
        required_leaf_ids=required_leaf_ids,
        truth_by_leaf_id=truth_by_leaf_id,
        dependencies_by_leaf=dependencies_by_leaf,
    )
    if failed_leaf_id not in set(leaves):
        raise HeldRecoveryStagingError("failed leaf is outside selected option")

    dependency_closure_cache: dict[str, frozenset[str]] = {}

    def dependency_closure(leaf_id: str) -> frozenset[str]:
        cached = dependency_closure_cache.get(leaf_id)
        if cached is not None:
            return cached
        closure: set[str] = set(dependencies[leaf_id])
        for prerequisite in dependencies[leaf_id]:
            closure.update(dependency_closure(prerequisite))
        result = frozenset(closure)
        dependency_closure_cache[leaf_id] = result
        return result

    return tuple(
        leaf_id
        for leaf_id in leaves
        if leaf_id != failed_leaf_id
        and not truth[leaf_id]
        and failed_leaf_id not in dependency_closure(leaf_id)
        and all(truth[prerequisite] for prerequisite in dependencies[leaf_id])
    )


def plan_held_recovery_staging(
    *,
    required_leaf_ids: Sequence[str],
    truth_by_leaf_id: Mapping[str, bool],
    dependencies_by_leaf: Mapping[str, Iterable[str]],
    failed_leaf_id: str,
    held_object_scope_id: str | None,
    preoperation_checkpoint_available: bool,
    legal_staging_target_scope_ids: Sequence[str],
    restore_attempts_used: int,
    staging_attempts_used: int,
    maximum_restore_attempts: int = 1,
    maximum_staging_attempts: int = 2,
) -> HeldRecoveryStagingDirective:
    """Plan one bounded, generic response to a retained held object.

    A checkpoint restore is preferred because it returns the object to the
    authoritative pre-operation state.  When that path is absent or exhausted,
    the first caller-supplied legal staging target is used; callers must order
    these IDs by deterministic live-geometry capacity.  No implicit drop is
    allowed.  Once the hand is already clear, the failed leaf is deferred and
    independent ready leaves can continue immediately.
    """

    leaves, truth, dependencies = _normalize_leaf_contract(
        required_leaf_ids=required_leaf_ids,
        truth_by_leaf_id=truth_by_leaf_id,
        dependencies_by_leaf=dependencies_by_leaf,
    )
    if failed_leaf_id not in set(leaves):
        raise HeldRecoveryStagingError("failed leaf is outside selected option")
    if held_object_scope_id is not None and (
        not isinstance(held_object_scope_id, str) or not held_object_scope_id
    ):
        raise HeldRecoveryStagingError("held object scope ID is invalid")
    if not isinstance(preoperation_checkpoint_available, bool):
        raise HeldRecoveryStagingError("checkpoint availability must be boolean")
    for label, value in (
        ("restore attempts used", restore_attempts_used),
        ("staging attempts used", staging_attempts_used),
        ("maximum restore attempts", maximum_restore_attempts),
        ("maximum staging attempts", maximum_staging_attempts),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise HeldRecoveryStagingError(f"{label} must be non-negative")
    if maximum_restore_attempts < 1 or maximum_staging_attempts < 1:
        raise HeldRecoveryStagingError("held-clearance budgets must be positive")
    staging_targets = tuple(legal_staging_target_scope_ids)
    if any(not isinstance(target, str) or not target for target in staging_targets):
        raise HeldRecoveryStagingError("staging target IDs must be non-empty strings")
    if len(set(staging_targets)) != len(staging_targets):
        raise HeldRecoveryStagingError("staging target IDs must be unique")

    before_true = tuple(leaf_id for leaf_id in leaves if truth[leaf_id])
    independent = independent_ready_leaf_ids(
        required_leaf_ids=leaves,
        truth_by_leaf_id=truth,
        dependencies_by_leaf=dependencies,
        failed_leaf_id=failed_leaf_id,
    )

    common = {
        "version": HELD_RECOVERY_STAGING_VERSION,
        "failed_leaf_id": failed_leaf_id,
        "held_object_scope_id": held_object_scope_id,
        "selected_option_leaf_ids": leaves,
        "deferred_leaf_ids": (failed_leaf_id,),
        "independent_ready_leaf_ids": independent,
        "before_true_selected_leaf_ids": before_true,
    }
    if held_object_scope_id is None:
        return HeldRecoveryStagingDirective(
            **common,
            action="defer_and_continue",
            staging_target_scope_id=None,
            restore_attempt_number=None,
            staging_attempt_number=None,
            accepted=True,
            reason="hand_clear_independent_nodes_may_continue",
        )
    if (
        preoperation_checkpoint_available
        and restore_attempts_used < maximum_restore_attempts
    ):
        return HeldRecoveryStagingDirective(
            **common,
            action="restore_preoperation_checkpoint",
            staging_target_scope_id=None,
            restore_attempt_number=restore_attempts_used + 1,
            staging_attempt_number=None,
            accepted=True,
            reason="bounded_checkpoint_restore_preferred",
        )
    if staging_targets and staging_attempts_used < maximum_staging_attempts:
        return HeldRecoveryStagingDirective(
            **common,
            action="stage_held_object",
            staging_target_scope_id=staging_targets[0],
            restore_attempt_number=None,
            staging_attempt_number=staging_attempts_used + 1,
            accepted=True,
            reason="bounded_geometry_ranked_staging_required",
        )
    return HeldRecoveryStagingDirective(
        **common,
        action="fail_closed",
        staging_target_scope_id=None,
        restore_attempt_number=None,
        staging_attempt_number=None,
        accepted=False,
        reason="held_object_clearance_budget_or_capability_exhausted",
    )


def confirm_held_recovery_clearance(
    *,
    directive: HeldRecoveryStagingDirective,
    held_object_scope_id_after: str | None,
    selected_truth_after: Mapping[str, bool],
) -> HeldRecoveryStagingDirective:
    """Admit continuation only after strict held-state and goal readback.

    The selected leaf identity set must remain exact.  Any previously true leaf
    regression or any retained held object rejects continuation, preserving the
    caller's right to try the next bounded clearance alternative.
    """

    if directive.action not in {
        "restore_preoperation_checkpoint",
        "stage_held_object",
    }:
        raise HeldRecoveryStagingError(
            "only a restore or staging directive can be confirmed"
        )
    expected_ids = set(directive.selected_option_leaf_ids)
    if set(selected_truth_after) != expected_ids:
        raise HeldRecoveryStagingError(
            "clearance truth mapping must exactly match selected leaf IDs"
        )
    if any(not isinstance(value, bool) for value in selected_truth_after.values()):
        raise HeldRecoveryStagingError("clearance leaf truth must be boolean")
    regressed = tuple(
        leaf_id
        for leaf_id in directive.before_true_selected_leaf_ids
        if not selected_truth_after[leaf_id]
    )
    cleared = held_object_scope_id_after is None
    accepted = cleared and not regressed
    reason = (
        "held_clear_goal_preserved_independent_nodes_may_continue"
        if accepted
        else (
            "held_object_still_retained"
            if not cleared
            else "previously_true_selected_leaf_regressed"
        )
    )
    return HeldRecoveryStagingDirective(
        version=directive.version,
        action="defer_and_continue" if accepted else "clearance_rejected",
        failed_leaf_id=directive.failed_leaf_id,
        held_object_scope_id=held_object_scope_id_after,
        staging_target_scope_id=directive.staging_target_scope_id,
        selected_option_leaf_ids=directive.selected_option_leaf_ids,
        deferred_leaf_ids=directive.deferred_leaf_ids,
        independent_ready_leaf_ids=(
            directive.independent_ready_leaf_ids if accepted else ()
        ),
        before_true_selected_leaf_ids=directive.before_true_selected_leaf_ids,
        restore_attempt_number=directive.restore_attempt_number,
        staging_attempt_number=directive.staging_attempt_number,
        accepted=accepted,
        reason=reason,
    )
