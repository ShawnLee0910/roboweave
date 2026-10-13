import pytest

from behavior_hl.embodied.evaluator_bridge.oracle_articulated_recovery import (
    articulated_candidate_retry_report, articulated_self_block_report,
)


def diagnostic():
    return {"phase": "blocked", "terminal": True, "symbolic_failure_private": {
        "operation": "open", "blocked": True, "rollback_applied": True,
        "rollback_verified": True, "physics_steps_executed": 0,
        "selected_link": "/World/box/lid", "obstructing_objects_private": [],
        "blocking_contact_pairs_private": [["/World/robot/arm", "/World/robot/arm/collision"]],
    }}


def check(report, **kwargs):
    options = dict(operation="open", status="blocked", diagnostic=report,
                   robot_path="/World/robot", target_link_paths=("/World/box/lid",),
                   held_before=None, held_after=None)
    return articulated_self_block_report(**{**options, **kwargs})


@pytest.mark.parametrize("operation", ["open", "close"])
def test_attested_robot_only_block_is_eligible(operation):
    d = diagnostic()
    d["symbolic_failure_private"]["operation"] = operation
    assert check(d, operation=operation)


@pytest.mark.parametrize("field,value", [
    ("rollback_verified", False), ("rollback_applied", None),
    ("physics_steps_executed", 1), ("physics_steps_executed", False),
    ("obstructing_objects_private", ["payload"]),
    ("selected_link", "/World/other/lid"), ("blocking_contact_pairs_private", []),
    ("blocking_contact_pairs_private", [["/World/robot2/arm", "/World/robot2/arm/collision"]]),
    ("blocking_contact_pairs_private", [["/World/wall", "/World/wall/collision"]]),
    ("blocking_contact_pairs_private", [["/World/robot/arm", "/World/wall/collision"]]),
])
def test_incomplete_external_or_ambiguous_proof_is_not_eligible(field, value):
    d = diagnostic()
    d["symbolic_failure_private"][field] = value
    assert check(d) is None


def test_no_uncertain_held_or_stale_diagnostic_retry():
    assert check(diagnostic(), status="uncertain") is None
    assert check(diagnostic(), held_after=object()) is None
    d = diagnostic()
    d["terminal"] = False
    assert check(d) is None
    assert check({}) is None


def test_latest_action_last_safe_step_is_not_automatically_a_noop():
    d = diagnostic()
    report = d["symbolic_failure_private"]
    report.update(rollback_scope="last_safe_step", last_safe_state_verified=True,
                  partial_progress_preserved=True, task_state_unchanged=False,
                  recoverable_executor_noop=False)
    assert check(d) is None
    report.update(partial_progress_preserved=False, task_state_unchanged=True,
                  recoverable_executor_noop=True)
    assert check(d)
    report.pop("task_state_unchanged")
    assert check(d) is None


def candidate_diagnostic():
    d = diagnostic()
    d["symbolic_failure_private"].update(
        schema_version="behavior2026.private-drawer-part/v8-zero-physics-atomic",
        selected_joint="lid_joint", rollback_scope="last_safe_step",
        partial_progress_preserved=False, task_state_unchanged=True,
        recoverable_executor_noop=True, last_safe_state_verified=True,
    )
    return d


def check_candidate(d, **kwargs):
    options = dict(operation="open", status="blocked", diagnostic=d,
                   target_link_paths=("/World/box/lid",), selected_joint="lid_joint",
                   held_before=None, held_after=None,
                   canonical_before={0: False, 1: True},
                   canonical_after={0: False, 1: True})
    return articulated_candidate_retry_report(**{**options, **kwargs})


@pytest.mark.parametrize("operation", ["open", "close"])
@pytest.mark.parametrize("scope", ["action_start", "last_safe_step"])
def test_generic_selector_retry_accepts_attested_noop_with_external_obstacle(operation, scope):
    d = candidate_diagnostic()
    d["symbolic_failure_private"].update(
        operation=operation, rollback_scope=scope,
        obstructing_objects_private=["content"],
        blocking_contact_pairs_private=[["/World/wall", "/World/wall/collision"]],
    )
    assert check_candidate(d, operation=operation)
    assert check(d, operation=operation) is None  # Not permission to change station.


@pytest.mark.parametrize("field,value", [
    ("schema_version", None), ("operation", "close"),
    ("selected_link", "/World/other/lid"), ("selected_joint", "other_joint"),
    ("rollback_scope", None), ("rollback_scope", "unknown"),
    ("rollback_applied", False), ("rollback_verified", False),
    ("physics_steps_executed", False), ("physics_steps_executed", 1),
    ("partial_progress_preserved", True), ("partial_progress_preserved", None),
    ("task_state_unchanged", False), ("recoverable_executor_noop", None),
    ("last_safe_state_verified", False),
])
def test_generic_selector_retry_rejects_incomplete_or_partial_report(field, value):
    d = candidate_diagnostic()
    d["symbolic_failure_private"][field] = value
    assert check_candidate(d) is None


def test_joint_only_or_stale_owner_report_is_not_retry_proof():
    assert check_candidate({}) is None
    d = candidate_diagnostic()
    d["terminal"] = False
    assert check_candidate(d) is None
    d = candidate_diagnostic()
    assert check_candidate(d, canonical_before=None) is None
    assert check_candidate(d, canonical_after={0: False, 1: False}) is None
    assert check_candidate(d, held_after=object()) is None
    assert check_candidate(d, held_before=object()) is None
    assert check_candidate(d, status="uncertain") is None


@pytest.mark.parametrize("block_count,expected_pass", [(1, True), (2, True), (3, False)])
def test_real_public_coordinator_repositions_with_fresh_markers(monkeypatch, tmp_path, block_count, expected_pass):
    from test_public_oracle_goal_and_marker_recovery import _run

    run = _run(monkeypatch, tmp_path, closed_destination_ancestor=True,
               articulated_self_blocks=block_count)
    assert (run.result["status"] == "PASS") is expected_pass, run.result.get("error")
    assert len(run.navigation) == min(block_count, 2)
    opens = [call for call in run.calls if call[0] == "open"]
    assert len(opens) == 2 * (min(block_count, 2) + 1)
    assert [call[2] for call in opens[::2]] == list(range(min(block_count, 2) + 1))
    if block_count >= 2:
        assert set(run.navigation[1]["excluded_station_cells"]) == {(1, 0), (1, 1)}
        assert run.navigation[1]["initial_blocked_transitions"] == (run.rejected_edge,)
    if not expected_pass:
        assert run.result["candidate_rejection_private"]["reason"] == "articulated_station_retry_budget_exhausted"


@pytest.mark.parametrize("kind", ["external", "unverified", "changed_truth"])
def test_real_public_coordinator_does_not_reposition_without_robot_only_proof(monkeypatch, tmp_path, kind):
    from test_public_oracle_goal_and_marker_recovery import _run

    run = _run(monkeypatch, tmp_path, closed_destination_ancestor=True,
               articulated_self_blocks=1, articulated_block_kind=kind)
    assert run.result["status"] == "FAIL_EXCEPTION"
    assert not run.navigation
    if kind == "changed_truth":
        assert run.result["failure_class"] == "ACTION20_PROTOCOL_ERROR"
