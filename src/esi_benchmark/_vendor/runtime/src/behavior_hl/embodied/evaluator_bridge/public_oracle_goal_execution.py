"""Bounded execution of a fixed witness against fresh canonical truth.

No simulator mutation, option selection, or Action20 retry policy lives here.
The caller owns public dispatch and supplies only attested local rejections.
"""

from collections import Counter


class PublicGoalRecoveryExhausted(RuntimeError):
    """The bounded Oracle attempt stopped; this is not an unsolvability proof."""


class PublicCandidateFailureMemory:
    """Episode-local failed selectors, scoped to the measured execution context.

    The public executor supplies a hashable context containing station, held
    identity/geometry and relevant scene state. It must not contain action
    indices, fresh C-labels or timestamps. Pose bins are for duplicate detection,
    not a new pose-validity gate. Missing context never creates a global ban.
    This memory does not authorize retries: the caller must first establish
    rollback/no-op safety or a completed action's readable postcondition.
    """

    def __init__(self):
        self._failed = {}

    def excluded_selectors(self, *, operation, target, context):
        if context is None:
            return frozenset()
        return frozenset(self._failed.get((operation, target, context), ()))

    def record_failure(self, *, operation, target, context, selector):
        if context is None or not isinstance(selector, str) or not selector:
            return False
        failed = self._failed.setdefault((operation, target, context), set())
        added = selector not in failed
        failed.add(selector)
        return added


def order_ready_goal_nodes(nodes, advisory_rank):
    """Stable topological order; a hint can never bypass a prerequisite."""
    remaining = list(nodes)
    completed = set()
    while remaining:
        ready = [node for node in remaining if set(node.dependencies) <= completed]
        if not ready:
            raise RuntimeError("public Goal-DAG dependencies are not a complete acyclic graph")
        node = min(ready, key=advisory_rank)
        remaining.remove(node)
        completed.add(node.node_id)
        yield node


def execute_live_goal_frontier(
    *, read_snapshot, compile_plan, leaf_ids_for_plan, execute_node,
    locally_rejected, hand_is_empty, record, maximum_node_attempts=2,
    maximum_recompilations=3, order_nodes=lambda nodes: nodes,
    operation_is_satisfied=lambda node: False,
    record_failed_postcondition=lambda node, receipt, before, after: None,
):
    """Recompile the same witness, never replay an obsolete plan after failure.

    A local rejection can defer an independent node only with an empty hand.
    It can be retried only after canonical truth changes. Completed actions
    with a false postcondition have a separate, small per-operation budget.
    """
    attempts = Counter()
    rejected = {}
    rejection_errors = {}
    for revision in range(maximum_recompilations):
        last_error = None
        snapshot = read_snapshot()
        if not snapshot.ok:
            raise RuntimeError("public Goal-DAG verifier is unavailable")
        if snapshot.official_success and float(snapshot.official_q) == 1.0:
            return snapshot
        plan = compile_plan(snapshot)
        leaf_ids = leaf_ids_for_plan(plan, snapshot)
        record({"event": "live_goal_plan_compiled", "revision": revision,
                "option_index": plan.option_index,
                "node_count": len(plan.action_nodes),
                "leaf_truth_private": dict(snapshot.leaf_truth),
                "action_nodes_private": [
                    {"node_id": node.node_id, "action": node.action,
                     "target": node.target, "destination": node.destination,
                     "dependencies": list(node.dependencies)}
                    for node in plan.action_nodes
                ]})
        deferred = set()
        executed = False
        for node in order_nodes(plan.action_nodes):
            snapshot = read_snapshot()
            if not snapshot.ok:
                raise RuntimeError("public Goal-DAG verifier is unavailable")
            if snapshot.official_success and float(snapshot.official_q) == 1.0:
                return snapshot
            ids = leaf_ids[node.node_id]
            identity = (node.action, node.target, node.destination)
            if ((ids and all(snapshot.leaf_truth[key] for key in ids))
                    or (not ids and operation_is_satisfied(node))):
                rejected.pop(identity, None)
                rejection_errors.pop(identity, None)
                record({"event": "goal_node_skipped", "node_id": node.node_id})
                continue
            signature = tuple(sorted(snapshot.leaf_truth.items()))
            if (any(key in deferred for key in node.dependencies)
                    or rejected.get(identity) == signature):
                deferred.add(node.node_id)
                last_error = rejection_errors.get(identity, last_error)
                continue
            if attempts[identity] >= maximum_node_attempts:
                raise PublicGoalRecoveryExhausted(
                    f"Goal-DAG node goal remains false: bounded recovery exhausted for {identity}"
                )
            attempts[identity] += 1
            try:
                execution_receipt = execute_node(node, snapshot, ids)
            except locally_rejected as exc:
                # Holding a subject needs operation-local recovery, not a
                # second pick or an implicit release by the coordinator.
                if not hand_is_empty():
                    raise
                last_error = exc
                rejected[identity] = signature
                rejection_errors[identity] = exc
                deferred.add(node.node_id)
                record({"event": "goal_node_deferred", "node_id": node.node_id,
                        "attempt": attempts[identity], "reason": str(exc)})
                continue
            executed = True
            rejected.pop(identity, None)
            rejection_errors.pop(identity, None)
            current = read_snapshot()
            if not current.ok:
                raise RuntimeError("public Goal-DAG verifier is unavailable")
            if current.official_success and float(current.official_q) == 1.0:
                return current
            if ids and not all(current.leaf_truth[key] for key in ids):
                # Persist the actual public selector/context before recompiling.
                # A false leaf is not automatically an Action protocol error;
                # the caller may exclude just this unchanged candidate context.
                record_failed_postcondition(node, execution_receipt, snapshot, current)
                record({"event": "goal_node_readback_recompile",
                        "node_id": node.node_id, "attempt": attempts[identity]})
                break
        if not executed:
            if last_error is not None:
                raise last_error
            raise PublicGoalRecoveryExhausted("public Goal-DAG has no executable unsatisfied node")
    raise PublicGoalRecoveryExhausted("Goal-DAG node goal remains false: recompile budget exhausted")
