"""Recognize one attested, robot-only articulated obstruction; no mutation."""
from collections.abc import Mapping


def completed_articulation_part_progress(*, operation, diagnostic, selected_joint, seen, allowed=None):
    """Recognize a new, verified part completion; never authorize blocked retry."""
    if operation not in {'open', 'close'} or not isinstance(diagnostic, Mapping):
        return None
    report = diagnostic.get('articulation_result_private')
    verdict = diagnostic.get('outcome_verifier_private', {})
    desired = operation == 'open'
    if (not isinstance(report, Mapping) or not isinstance(verdict, Mapping) or report.get('operation') != operation
            or report.get('selected_joint') != selected_joint or not selected_joint
            or selected_joint in seen or report.get('joint_readback_errors_private')
            or report.get('selected_open_before') is not (not desired)
            or report.get('selected_open_after') is not desired
            or verdict.get('semantic_state_pass') is not True):
        return None
    before, after = report.get('parts_before_private', ()), report.get('parts_after_private', ())
    if not isinstance(before, (list, tuple)) or not isinstance(after, (list, tuple)):
        return None
    if not all(isinstance(row, Mapping) and isinstance(row.get('selector'), str) for row in (*before, *after)):
        return None
    identities = frozenset(row['selector'] for row in before)
    if (not identities or len(identities) != len(before) or len(before) != len(after)
            or identities != frozenset(row['selector'] for row in after)
            or selected_joint not in identities or (allowed is not None and identities != allowed)):
        return None
    before_part = next(row for row in before if row['selector'] == selected_joint)
    after_part = next(row for row in after if row['selector'] == selected_joint)
    if before_part.get('open') is not (not desired) or after_part.get('open') is not desired:
        return None
    return identities


def articulated_candidate_retry_report(
    *, operation, status, diagnostic, target_link_paths, held_before, held_after,
    canonical_before, canonical_after, selected_joint=None,
):
    """Admit selector retry only from the current dispatch's verified no-op.

    The caller supplies fresh canonical reads (None when unavailable) and the
    direct action-token diagnostic, never an owner's cached exact report.
    Verified partial progress is deliberately not an action-start rollback.
    """
    if operation not in {"open", "close"} or status != "blocked":
        return None
    if held_before is not None or held_after is not None:
        return None
    if (not isinstance(canonical_before, Mapping)
            or not isinstance(canonical_after, Mapping)
            or canonical_before != canonical_after):
        return None
    if (not isinstance(diagnostic, Mapping) or diagnostic.get("phase") != "blocked"
            or diagnostic.get("terminal") is not True):
        return None
    report = diagnostic.get("symbolic_failure_private")
    if not isinstance(report, Mapping):
        return None
    if (report.get("schema_version") != "behavior2026.private-drawer-part/v8-zero-physics-atomic"
            or report.get("operation") != operation or report.get("blocked") is not True
            or report.get("selected_link") not in target_link_paths
            or (selected_joint is not None and report.get("selected_joint") != selected_joint)
            or report.get("rollback_applied") is not True
            or report.get("rollback_verified") is not True
            or type(report.get("physics_steps_executed")) is not int
            or report.get("physics_steps_executed") != 0
            or report.get("partial_progress_preserved") is True
            or report.get("task_state_unchanged") is False
            or report.get("recoverable_executor_noop") is False):
        return None
    scope = report.get("rollback_scope")
    if scope != "action_start" and not (
        scope == "last_safe_step"
        and report.get("task_state_unchanged") is True
        and report.get("recoverable_executor_noop") is True
        and report.get("last_safe_state_verified") is True
        and report.get("partial_progress_preserved") is False
    ):
        return None
    return dict(report)


def articulated_self_block_report(*, operation, status, diagnostic, robot_path,
                                  target_link_paths, held_before, held_after):
    if operation not in {"open", "close"} or status != "blocked":
        return None
    if held_before is not None or held_after is not None:
        return None
    if (not isinstance(diagnostic, Mapping) or diagnostic.get("phase") != "blocked"
            or diagnostic.get("terminal") is not True):
        return None
    report = diagnostic.get("symbolic_failure_private")
    if not isinstance(report, Mapping):
        return None
    # New Action can preserve a verified partial opening. A last-safe-step
    # rollback is not proof of an unchanged action-start world. Keep this
    # existing recovery lane no-op-only; do not silently broaden its contract.
    if (report.get("partial_progress_preserved") is True
            or report.get("task_state_unchanged") is False
            or report.get("recoverable_executor_noop") is False):
        return None
    if report.get("rollback_scope") == "last_safe_step" and not (
        report.get("task_state_unchanged") is True
        and report.get("recoverable_executor_noop") is True
        and report.get("last_safe_state_verified") is True
    ):
        return None
    if (report.get("operation") != operation or report.get("blocked") is not True
            or report.get("rollback_applied") is not True
            or report.get("rollback_verified") is not True
            or type(report.get("physics_steps_executed")) is not int
            or report.get("physics_steps_executed") != 0
            or report.get("obstructing_objects_private") not in ([], ())
            or report.get("selected_link") not in target_link_paths):
        return None
    if not isinstance(robot_path, str) or not robot_path.startswith("/") or robot_path == "/":
        return None
    pairs = report.get("blocking_contact_pairs_private")
    if not isinstance(pairs, (list, tuple)) or not pairs:
        return None
    prefix = robot_path.rstrip("/") + "/"
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None
        if not all(isinstance(path, str) and path.startswith(prefix) for path in pair):
            return None
    return dict(report)
