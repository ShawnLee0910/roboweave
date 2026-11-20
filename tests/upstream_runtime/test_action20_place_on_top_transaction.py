from __future__ import annotations

import ast
import hashlib
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from behavior_hl.embodied.evaluator_bridge.marker_symbolic_grasp_backend import (
    Action20CommitRejectedError,
    MarkerSymbolicBackendError,
    MarkerSymbolicGraspBackendAdapter,
    SymbolicPreparationResult,
)
from behavior_hl.embodied.evaluator_bridge import (
    run_b1_marker_symbolic_rollout as marker_rollout,
)
from behavior_hl.embodied.evaluator_bridge.private_global_se2_route import (
    GlobalSE2RoutePlan,
)
from behavior_hl.embodied.evaluator_bridge.private_instance_visibility import (
    PrivateInstanceSegmentationFrame,
)
from behavior_hl.embodied.evaluator_bridge.private_visual_markers import (
    MarkerPublicFailureReason,
    MarkerStatus,
    PrivateMarkerBinding,
    PrivateMarkerSnapshot,
    PrivateVisualMarkerStore,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_marker_symbolic_rollout import (
    _preflight_action20_on_top_route_collision,
    _preflight_action20_on_top_route_static,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_resident_environment import (
    ResidentEvaluatorTerminalError,
)
from behavior_hl.embodied.runtime.plugins.marker_symbolic_grasp import (
    MarkerSymbolicV34OperationPlugin,
    SymbolicGraspStatus,
)
from behavior_hl.embodied.runtime.public_action_feedback import (
    model_visible_action20_last_result,
)
from behavior_hl.embodied.policy.protocol import (
    ActionEnvelope,
    MarkerTargetOrRequestArgs,
)
from behavior_hl.embodied.runtime.dispatch import ActionPluginResult
from behavior_hl.navigation import PlanarPose


def _terminal_preparation(result):
    def prepare(_target, _operation):
        if False:
            yield 0
        return result

    return prepare


def _complete_commit_rejection_report():
    return {
        "status": "commit_rejected",
        "base_stayed_at_prepared_pose": False,
        "rollback_private": {
            "status": "rolled_back",
            "rollback_applied": True,
            "base_pose_restored": True,
            "protected_pose_restore_private": {"accepted": True},
            "logical_restore_private": {
                "held_identity_restored": True,
                "logical_inventory_identity_restored": True,
            },
            "inside_edges_restored": True,
        },
    }


def _backend(
    *,
    prepare,
    place_factory,
    command_step=lambda _command: 0,
    outcome_verifier=None,
    marker_store=None,
    frame_provider=lambda: None,
    rgb_provider=lambda: np.zeros((1, 1, 3), dtype=np.uint8),
    support_names=lambda: (),
    reach_admission=None,
    held=None,
    held_provider=None,
    physical_consequences=False,
):
    return MarkerSymbolicGraspBackendAdapter(
        marker_store=marker_store or PrivateVisualMarkerStore(),
        private_frame_provider=frame_provider,
        raw_head_rgb_provider=rgb_provider,
        eligible_objects_provider=lambda: {},
        graspable_object_names_provider=lambda: (),
        place_container_names_provider=lambda: (),
        official_bbox_provider=lambda _frame: (),
        symbolic_generator_factory=lambda _target: iter(()),
        symbolic_place_generator_factory=lambda _target: iter(()),
        command_step=command_step,
        hold_safe=lambda: None,
        local_reach_admission=reach_admission,
        local_prepare_factory=prepare,
        prepare_place_on_top=True,
        placement_physical_consequences_enabled=physical_consequences,
        support_object_names_provider=support_names,
        symbolic_operation_factories={"place_on_top": place_factory},
        private_outcome_verifier=outcome_verifier,
        held_object_provider=held_provider or (lambda: held),
    )


def _prepared_commit_backend(finalize, held):
    carry = [None]
    def finalize_after_release(success):
        # A commit rejection restores the original held identity.
        carry[0] = held
        return finalize(success)
    backend = _backend(
        prepare=_terminal_preparation(
            SymbolicPreparationResult({"planning_count": 0}, finalize_after_release)
        ),
        place_factory=lambda _target: iter(()),
        outcome_verifier=lambda *_args: {"semantic_state_pass": True},
        held_provider=lambda: carry[0],
    )
    token = backend.begin_symbolic_operation(
        operation="place_on_top", target_token=object()
    )
    token.acted_object = held
    backend.tick_symbolic_operation(token)
    backend.tick_symbolic_operation(token)
    return backend, token


def test_recovered_sampling_error_is_blocked_only_when_carry_is_restored():
    class ActionPrimitiveError(ValueError):
        reason = SimpleNamespace(name="SAMPLING_ERROR")

    held = object()

    def rejected_candidate(_target):
        if False:
            yield None
        raise ActionPrimitiveError("no stable candidate")

    backend = _backend(
        prepare=None,
        place_factory=rejected_candidate,
        held=held,
    )
    token = backend.begin_symbolic_operation(
        operation="place_on_top", target_token=object()
    )

    tick = backend.tick_symbolic_operation(token)

    assert tick.status is SymbolicGraspStatus.BLOCKED
    assert backend._held_object() is held

    unresolved = _backend(
        prepare=None,
        place_factory=rejected_candidate,
        held=None,
    )
    unresolved_token = unresolved.begin_symbolic_operation(
        operation="place_on_top", target_token=object()
    )

    assert (
        unresolved.tick_symbolic_operation(unresolved_token).status
        is SymbolicGraspStatus.FAILED
    )


@pytest.mark.parametrize("rollback, other_cause, expected", [
    ("boundary", False, "blocked"), ("prepared", False, "blocked"),
    ("boundary", True, "uncertain"), (None, False, "uncertain"),
])
def test_inside_group_preserves_collision_only_after_rollback(rollback, other_cause, expected):
    class ActionPrimitiveError(Exception):
        metadata = {"action20_blocked_category": "placement_collision_blocked"}

    class ActionPrimitiveErrorGroup(Exception):
        exceptions = [ActionPrimitiveError(), RuntimeError() if other_cause else ActionPrimitiveError()]

    def reject(_target):
        yield from ()
        raise ActionPrimitiveErrorGroup()

    backend = _backend(prepare=None, place_factory=reject, held=object())
    backend._place_generator = reject
    token = backend.begin_symbolic_operation(operation="place_inside", target_token=object())
    if rollback == "boundary":
        token.placement_rollback = lambda: {"accepted": True}
    elif rollback == "prepared":
        token.preparation_finalize = lambda _success: _complete_commit_rejection_report()["rollback_private"]
    state = SimpleNamespace(backend=backend, operation="place_inside", stage="symbolic", driver_token=token, status=None)
    plugin = MarkerSymbolicV34OperationPlugin("place_inside")
    assert plugin.tick(state, None).done
    result = plugin.finish(state, None)
    assert result.status == expected
    assert result.public_feedback.get("blocked_category") == (
        "placement_collision_blocked" if expected == "blocked" else None
    )


def test_unknown_placement_failure_remains_uncertain():
    held = object()

    def broken_candidate(_target):
        if False:
            yield None
        raise RuntimeError("unexpected implementation failure")

    backend = _backend(
        prepare=None,
        place_factory=broken_candidate,
        held=held,
    )
    token = backend.begin_symbolic_operation(
        operation="place_on_top", target_token=object()
    )

    tick = backend.tick_symbolic_operation(token)

    assert tick.status is SymbolicGraspStatus.FAILED


def test_action20_blocked_preparation_never_starts_placement_factory():
    factory_calls = []

    def prepare(_target, _operation):
        if False:
            yield 0
        error = RuntimeError("arrival candidate invalid")
        error.private_report = {
            "status": "blocked",
            "planning_count": 1,
            "adopted_station_count": 1,
            "committed_station_count": 0,
            "second_station_attempted": False,
        }
        raise error

    def place_factory(_target):
        factory_calls.append(True)
        return iter(())

    backend = _backend(prepare=prepare, place_factory=place_factory)
    token = backend.begin_symbolic_operation(
        operation="place_on_top", target_token=object()
    )

    tick = backend.tick_symbolic_operation(token)

    assert tick.status is SymbolicGraspStatus.FAILED
    assert factory_calls == []
    diagnostic = token.private_diagnostic()
    assert diagnostic["symbolic_generator_start_count"] == 0
    assert diagnostic["preparation_report_private"] == {
        "status": "blocked",
        "planning_count": 1,
        "adopted_station_count": 1,
        "committed_station_count": 0,
        "second_station_attempted": False,
    }


@pytest.mark.parametrize(
    "failure",
    ("no_geometry", "route_unavailable", "static_rejected", "arrival_invalid", "zero_route", "route_collision", "unexpected",
     "arrival_no_geometry", "zero_route_collision", "arrival_query_error", "arrival_malformed_hit"),
)
def test_special_on_top_station_failure_preserves_public_failure_cause(failure):
    source = ast.parse(inspect.getsource(marker_rollout._build_marker_runtime))
    callback = next(
        node for node in source.body[0].body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_action20_on_top_station_prepare"
    )
    held = SimpleNamespace(
        get_position_orientation=lambda: (np.zeros(3), (0.0, 0.0, 0.0, 1.0)),
        collision_points_world=np.array([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.1]]),
    )
    target = SimpleNamespace(name="support")
    pose = PlanarPose(0.0, 0.0, 0.0, 0.0)
    calls = {name: [] for name in ("route", "placement", "step", "rollback")}
    binding = SimpleNamespace(
        candidate=SimpleNamespace(name="candidate"),
        future_aabb_min_world=np.array([2.0, 0.0, 0.5]),
        future_aabb_max_world=np.array([2.2, 0.2, 0.6]),
    )
    collision_cases = {
        "zero_route_collision": "blocking_collision_shape_overlap",
        "arrival_query_error": "collision_query_error",
        "arrival_malformed_hit": "malformed_collision_overlap_hit",
    }
    plan_calls = []

    def live_plan(*_args, **kwargs):
        np.testing.assert_allclose(kwargs["collision_points_in_root"], held.collision_points_world)
        plan_calls.append(True)
        return SimpleNamespace(candidate_plan=SimpleNamespace(planning_metadata_private={}), bindings=(
            () if failure == "no_geometry" or (
                failure == "arrival_no_geometry" and len(plan_calls) > 1
            ) else (binding,)
        ))

    def collision_query(**_kwargs):
        return SimpleNamespace(
            accepted=False,
            to_private_dict=lambda: {
                "collision_query_reason": collision_cases[failure],
                "query_error": "query failed" if failure == "arrival_query_error" else None,
            },
        )

    def plan_route(**kwargs):
        calls["route"].append(kwargs)
        return SimpleNamespace(
            status="unavailable" if failure == "route_unavailable" else "planned",
            plan=SimpleNamespace(
                actions=() if failure in {"zero_route", "zero_route_collision"} else ("move_forward",),
                poses=(pose,),
            ),
            private_report={"planner_call_count": 1},
        )

    def execute_route(*_args, **_kwargs):
        if False:
            yield 0
        if failure == "unexpected":
            raise RuntimeError("private unexpected implementation failure")
        if failure == "route_collision":
            raise marker_rollout._Action20OnTopStationError(
                "private collision diagnostic", {"status": "blocked"},
                blocked_category="navigation_collision_blocked",
            )
        return {"arrived_pose": pose, "executed_route_action_count": 0}

    def finalize(success):
        calls["rollback"].append(success)
        return {"rollback_applied": True}

    namespace = {
        **vars(marker_rollout),
        "automatic_interaction_approach_enabled": False,
        "action20_enabled": True,
        "placement_physical_consequences_enabled": False,
        "symbolic": SimpleNamespace(
            _get_obj_in_hand=lambda: held,
            _logical_inventory_snapshot=None,
            _logical_inventory_carry=False,
            _logical_inventory_origin_pose=None,
        ),
        "evaluator": SimpleNamespace(env=SimpleNamespace(scene=SimpleNamespace(objects=(held, target)))),
        "_read_restricted_planar_pose": lambda: pose,
        "plan_on_top_support_layer_candidates": live_plan,
        "restrict_and_rank_on_top_plan": lambda plan, *_args: SimpleNamespace(
            bindings=plan.bindings if failure in collision_cases else (),
        ),
        "query_spatial_candidate_collision_overlap": collision_query,
        "og": None,
        "_private_global_se2_route_plan": plan_route,
        "_restricted_local_guard": lambda *_args: True,
        "last_restricted_guard_report": {},
        "_preflight_action20_on_top_route_static": lambda *_args, **_kwargs: (
            {"accepted": failure != "static_rejected"}, (),
        ),
        "_capture_action20_interaction_transaction": lambda **_kwargs: finalize,
        "_execute_action20_interaction_route": execute_route,
    }
    if failure in {"zero_route", "zero_route_collision"}:
        route_callback = next(
            node for node in source.body[0].body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_execute_action20_interaction_route"
        )
        exec(compile(ast.Module(body=[route_callback], type_ignores=[]), inspect.getfile(marker_rollout), "exec"), namespace)
    exec(compile(ast.Module(body=[callback], type_ignores=[]), inspect.getfile(marker_rollout), "exec"), namespace)

    def prepare(selected, operation):
        assert selected is target and operation == "place_on_top"
        return namespace[callback.name](
            selected,
            step_once=lambda: calls["step"].append(True),
            public_snapshot_provider=lambda: {},
        )

    backend = _backend(
        prepare=prepare,
        place_factory=lambda selected: calls["placement"].append(selected) or iter(()),
        held=held,
    )
    assert backend._automatic_interaction_approach_enabled is False
    token = backend.begin_symbolic_operation(operation="place_on_top", target_token=target)
    state = SimpleNamespace(
        backend=backend, operation="place_on_top", stage="symbolic",
        driver_token=token, status=None, public_feedback={},
    )
    plugin = MarkerSymbolicV34OperationPlugin("place_on_top")
    tick = plugin.tick(state, None)
    result = plugin.finish(state, None)

    assert tick.done and tick.simulator_steps == 0
    assert len(calls["route"]) == (0 if failure == "no_geometry" else 1)
    assert calls["placement"] == calls["step"] == []
    assert backend._held_object() is held
    assert token.symbolic_generator_start_count == 0
    unverified = failure in {"unexpected", "arrival_query_error", "arrival_malformed_hit"}
    assert result.status == ("uncertain" if unverified else "blocked")
    public = model_visible_action20_last_result(
        {"status": result.status, **result.public_feedback}, action="place_on_top",
    )
    if unverified:
        assert public == {"status": "blocked", "blocked_category": "execution_unverified"}
    elif failure in {"no_geometry", "arrival_no_geometry"}:
        assert public == {"status": "blocked", "blocked_category": "geometric_fit_blocked"}
    elif failure == "zero_route_collision":
        assert public == {"status": "blocked", "blocked_category": "placement_collision_blocked"}
        assert result.public_feedback["blocked_category"] == "placement_collision_blocked"
    else:
        assert public == model_visible_action20_last_result(
            {"status": "not_executed", "blocked_category": "navigation_blocked"},
            action="pick",
        ) == {"status": "blocked", "blocked_category": "out_of_reach"}
        assert result.public_feedback == {
            "object_acquired": True,
            "gripper_open_commanded": False,
            "blocked_category": (
                "navigation_collision_blocked" if failure == "route_collision"
                else "out_of_reach" if failure in {"arrival_invalid", "zero_route"}
                else "navigation_path_blocked"
            ),
        }
    assert calls["rollback"] == (
        [] if failure in {"no_geometry", "route_unavailable", "static_rejected"} else [False]
    )


@pytest.mark.parametrize(
    ("status", "zero_physics"),
    (("blocked", False), ("completed", True)),
)
def test_on_top_action_boundary_skips_unneeded_physics(
    monkeypatch,
    tmp_path,
    status,
    zero_physics,
):
    checkpoint_calls = []
    action_records = []
    diagnostic = {"translation_clipped": False} if zero_physics else None
    diagnostic_output = {"stale": True}
    active = SimpleNamespace(
        capture_observation_requested=False,
        simulator_steps=0,
        plugin=None,
        state=SimpleNamespace(private_diagnostic=lambda: diagnostic),
    )
    environment = SimpleNamespace(
        controller_services=lambda _index, _action: {},
        record_public_action_result=lambda _action, _result: None,
        checkpoint_once=lambda: checkpoint_calls.append(True) or 1,
    )
    runtime = SimpleNamespace(
        environment=environment,
        dispatcher=SimpleNamespace(
            begin=lambda _registry, _action, _context: active,
            tick=lambda _active: ActionPluginResult(status, {}),
        ),
        registry=SimpleNamespace(
            spec=lambda _action: SimpleNamespace(max_simulator_steps=1)
        ),
        store=SimpleNamespace(active_snapshot=None),
        symbolic_primitives=SimpleNamespace(
            _logical_inventory_carry=True,
            _get_obj_in_hand=lambda: object(),
            require_marker_visual_projection_inactive=lambda: None,
        ),
        carry_rebase_enabled=False,
        arm_presentation_ladder=None,
        zero_physics=zero_physics,
    )
    monkeypatch.setattr(marker_rollout, "_synchronize_runtime_control", lambda *_: 0)
    monkeypatch.setattr(
        marker_rollout,
        "_capture_action_boundary_observation",
        lambda _runtime: object(),
    )
    monkeypatch.setattr(marker_rollout, "_save_frame", lambda **_: {})
    monkeypatch.setattr(
        marker_rollout,
        "_symbolic_collision_filter_snapshot",
        lambda _runtime: {},
    )
    monkeypatch.setattr(
        marker_rollout,
        "_append_jsonl",
        lambda _path, record: action_records.append(record),
    )

    result, _frames, simulator_steps = marker_rollout._execute(
        runtime=runtime,
        action=ActionEnvelope(
            "place_on_top",
            MarkerTargetOrRequestArgs("C1"),
        ),
        action_index=1,
        phase_dir=tmp_path,
        enforce_public_carry_contract=False,
        execution_diagnostic_private=diagnostic_output,
    )

    assert result.status == status
    assert checkpoint_calls == []
    assert simulator_steps == 0
    assert action_records[0]["post_action_render_checkpoint_steps"] == 0
    assert diagnostic_output == (diagnostic or {})
    assert action_records[0]["driver_private_diagnostic"] == diagnostic


def test_completed_action_tail_accepts_verified_official_success(
    monkeypatch,
    tmp_path,
):
    action_records = []
    finalizations = []
    active = SimpleNamespace(
        capture_observation_requested=False,
        simulator_steps=0,
        plugin=None,
        state=SimpleNamespace(),
    )

    def terminate_at_checkpoint():
        raise ResidentEvaluatorTerminalError(
            terminated=True,
            truncated=False,
        )

    environment = SimpleNamespace(
        controller_services=lambda _index, _action: {},
        record_public_action_result=lambda _action, _result: None,
        checkpoint_once=terminate_at_checkpoint,
    )
    runtime = SimpleNamespace(
        environment=environment,
        dispatcher=SimpleNamespace(
            begin=lambda _registry, _action, _context: active,
            tick=lambda _active: ActionPluginResult("completed", {}),
        ),
        registry=SimpleNamespace(
            spec=lambda _action: SimpleNamespace(max_simulator_steps=1)
        ),
        store=SimpleNamespace(active_snapshot=None),
        symbolic_primitives=SimpleNamespace(
            _logical_inventory_carry=True,
            _get_obj_in_hand=lambda: None,
            require_marker_visual_projection_inactive=lambda: None,
            finalize_private_terminal_place_success=(
                lambda **record: finalizations.append(record)
            ),
        ),
        carry_rebase_enabled=False,
        arm_presentation_ladder=None,
    )
    monkeypatch.setattr(marker_rollout, "_synchronize_runtime_control", lambda *_: 0)
    monkeypatch.setattr(
        marker_rollout,
        "_capture_action_boundary_observation",
        lambda _runtime: object(),
    )
    monkeypatch.setattr(marker_rollout, "_save_frame", lambda **_: {})
    monkeypatch.setattr(
        marker_rollout,
        "_symbolic_collision_filter_snapshot",
        lambda _runtime: {},
    )
    monkeypatch.setattr(
        marker_rollout,
        "_append_jsonl",
        lambda _path, record: action_records.append(record),
    )
    snapshot = SimpleNamespace(
        ok=True,
        official_success=True,
        official_q=1.0,
        to_private_record=lambda: {"official_q": 1.0},
    )

    result, frames, simulator_steps = marker_rollout._execute(
        runtime=runtime,
        action=ActionEnvelope(
            "place_inside",
            MarkerTargetOrRequestArgs("C1"),
        ),
        action_index=202,
        phase_dir=tmp_path,
        enforce_public_carry_contract=False,
        terminal_success_check=lambda: snapshot,
    )

    assert result.status == "completed"
    assert len(frames) == 2
    assert simulator_steps == 1
    assert finalizations == [
        {"official_terminal_snapshot_private": {"official_q": 1.0}}
    ]
    assert action_records[0]["post_action_render_checkpoint_steps"] == 1
    assert action_records[0]["accepted_official_success_termination"] is True
    assert action_records[0]["official_terminal_snapshot_private"] == {
        "official_q": 1.0
    }


def test_action20_current_station_handoff_plans_and_moves_zero_times():
    factory_calls = []
    finalized = []

    def finalize(success):
        finalized.append(success)
        return {
            "status": "committed" if success else "rolled_back",
            "committed_station_count": 0,
        }

    def place_factory(_target):
        factory_calls.append(True)
        return iter(())

    backend = _backend(
        prepare=_terminal_preparation(
            SymbolicPreparationResult(
                {
                    "status": "current_station_reused",
                    "planning_count": 0,
                    "adopted_station_count": 0,
                    "route_action_count": 0,
                    "executed_route_action_count": 0,
                },
                finalize,
            )
        ),
        place_factory=place_factory,
        outcome_verifier=lambda *_args: {"semantic_state_pass": True},
    )
    token = backend.begin_symbolic_operation(
        operation="place_on_top", target_token=object()
    )

    assert backend.tick_symbolic_operation(token).status is SymbolicGraspStatus.RUNNING
    assert factory_calls == []
    assert backend.tick_symbolic_operation(token).status is SymbolicGraspStatus.RUNNING
    assert factory_calls == [True]
    assert backend.tick_symbolic_operation(token).status is SymbolicGraspStatus.SUCCEEDED
    assert finalized == [True]
    diagnostic = token.private_diagnostic()
    assert diagnostic["symbolic_generator_start_count"] == 1
    assert diagnostic["preparation_report_private"]["planning_count"] == 0
    assert diagnostic["preparation_report_private"]["route_action_count"] == 0


def test_action20_late_command_failure_closes_generator_and_rolls_back_once():
    closed = []
    finalized = []

    def finalize(success):
        finalized.append(success)
        return {"status": "committed" if success else "rolled_back"}

    def place_factory(_target):
        def generator():
            try:
                yield object()
            finally:
                closed.append(True)

        return generator()

    def fail_step(_command):
        raise RuntimeError("physics failed")

    backend = _backend(
        prepare=_terminal_preparation(
            SymbolicPreparationResult({"planning_count": 1}, finalize)
        ),
        place_factory=place_factory,
        command_step=fail_step,
    )
    token = backend.begin_symbolic_operation(
        operation="place_on_top", target_token=object()
    )

    backend.tick_symbolic_operation(token)
    backend.tick_symbolic_operation(token)
    tick = backend.tick_symbolic_operation(token)

    assert tick.status is SymbolicGraspStatus.FAILED
    assert closed == [True]
    assert finalized == [False]
    assert token.private_diagnostic()["preparation_finalization_private"] == {
        "status": "rolled_back"
    }


def test_action20_evaluator_terminal_escapes_backend_for_episode_owner():
    closed = []
    finalized = []

    def finalize(success):
        finalized.append(success)
        return {"status": "committed" if success else "rolled_back"}

    def place_factory(_target):
        def generator():
            try:
                yield object()
            finally:
                closed.append(True)

        return generator()

    def terminal_step(_command):
        raise ResidentEvaluatorTerminalError(
            terminated=True,
            truncated=False,
        )

    backend = _backend(
        prepare=_terminal_preparation(
            SymbolicPreparationResult({"planning_count": 1}, finalize)
        ),
        place_factory=place_factory,
        command_step=terminal_step,
    )
    token = backend.begin_symbolic_operation(
        operation="place_on_top", target_token=object()
    )

    backend.tick_symbolic_operation(token)
    backend.tick_symbolic_operation(token)
    with pytest.raises(ResidentEvaluatorTerminalError):
        backend.tick_symbolic_operation(token)

    assert closed == []
    assert finalized == []
    assert backend.abort_symbolic_operation(token).status is SymbolicGraspStatus.ABORTED
    assert closed == [True]
    assert finalized == [False]


def test_action20_failed_postcondition_rolls_back_once():
    finalized = []

    def finalize(success):
        finalized.append(success)
        return {"status": "committed" if success else "rolled_back"}

    backend = _backend(
        prepare=_terminal_preparation(
            SymbolicPreparationResult({"planning_count": 0}, finalize)
        ),
        place_factory=lambda _target: iter(()),
        outcome_verifier=lambda *_args: {"semantic_state_pass": False, "predicate_mode": "action20_symbolic"},
    )
    token = backend.begin_symbolic_operation(
        operation="place_on_top", target_token=object()
    )

    backend.tick_symbolic_operation(token)
    backend.tick_symbolic_operation(token)
    tick = backend.tick_symbolic_operation(token)

    assert tick.status is SymbolicGraspStatus.BLOCKED
    assert tick.blocked_category == "relation_not_realized"
    assert finalized == [False]


def test_action20_physical_consequence_commits_released_world_state():
    finalized = []
    hand = {"held": object()}

    def finalize(success):
        finalized.append(success)
        return {"status": "committed" if success else "rolled_back"}

    def place_factory(_target):
        hand["held"] = None
        return iter(())

    backend = _backend(
        prepare=_terminal_preparation(
            SymbolicPreparationResult({"planning_count": 0}, finalize)
        ),
        place_factory=place_factory,
        outcome_verifier=lambda *_args: {
            "semantic_state_pass": False,
            "physical_consequence_committed": True,
        },
        held_provider=lambda: hand["held"],
        physical_consequences=True,
    )
    token = backend.begin_symbolic_operation(
        operation="place_on_top", target_token=object()
    )

    backend.tick_symbolic_operation(token)
    backend.tick_symbolic_operation(token)
    tick = backend.tick_symbolic_operation(token)

    assert tick.status is SymbolicGraspStatus.SUCCEEDED
    assert finalized == [True]
    assert token.phase == "physical_consequence_committed"


def test_action20_abort_rolls_back_once_before_factory_start():
    finalized = []

    def finalize(success):
        finalized.append(success)
        return {"status": "committed" if success else "rolled_back"}

    backend = _backend(
        prepare=_terminal_preparation(
            SymbolicPreparationResult({"planning_count": 0}, finalize)
        ),
        place_factory=lambda _target: iter(()),
    )
    token = backend.begin_symbolic_operation(
        operation="place_on_top", target_token=object()
    )

    backend.tick_symbolic_operation(token)
    tick = backend.abort_symbolic_operation(token)

    assert tick.status is SymbolicGraspStatus.ABORTED
    assert finalized == [False]
    assert token.private_diagnostic()["symbolic_generator_start_count"] == 0


def test_action20_attested_commit_rejection_is_blocked_and_exactly_once():
    finalized = []
    held = object()
    report = _complete_commit_rejection_report()

    def finalize(success):
        finalized.append(success)
        raise Action20CommitRejectedError(
            "prepared station moved",
            private_report=report,
        )

    backend, token = _prepared_commit_backend(finalize, held)
    tick = backend.tick_symbolic_operation(token)

    assert tick.status is SymbolicGraspStatus.BLOCKED
    assert finalized == [True]
    assert token.private_diagnostic()["preparation_finalization_private"] == report


@pytest.mark.parametrize(
    "failure",
    ("runtime_error", "typed_subclass", "incomplete", "held_identity"),
)
def test_action20_other_commit_finalizer_failures_remain_hard(failure):
    finalized = []
    held = object()
    report = _complete_commit_rejection_report()
    if failure == "incomplete":
        report["rollback_private"]["inside_edges_restored"] = False

    class CommitRejectedSubclass(Action20CommitRejectedError):
        pass

    def finalize(success):
        finalized.append(success)
        if failure == "held_identity":
            backend._held_object = lambda: object()
        if failure == "runtime_error":
            error = RuntimeError("commit failed")
            error.private_report = report
            raise error
        raise (
            CommitRejectedSubclass
            if failure == "typed_subclass"
            else Action20CommitRejectedError
        )(
            "prepared station moved",
            private_report=report,
        )

    backend, token = _prepared_commit_backend(finalize, held)
    with pytest.raises(
        MarkerSymbolicBackendError,
        match="preparation commit finalizer failed",
    ):
        backend.tick_symbolic_operation(token)
    assert finalized == [True]
    assert token.private_diagnostic()["preparation_finalization_private"] == report


def test_action20_station_preparation_enforces_local_workspace_gate():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    segmentation = np.ones((8, 8), dtype=np.int64)
    source_rgb_sha256 = hashlib.sha256(rgb.tobytes()).hexdigest()
    source_segmentation_sha256 = hashlib.sha256(
        np.ascontiguousarray(segmentation, dtype=np.int64).tobytes()
    ).hexdigest()
    support = object()
    frame = PrivateInstanceSegmentationFrame(
        camera="head",
        simulator_step=4,
        source_rgb_sha256=source_rgb_sha256,
        segmentation=segmentation,
        instance_id_to_name={1: "counter"},
    )
    store = PrivateVisualMarkerStore()
    store.publish(
        PrivateMarkerSnapshot(
            camera="head",
            simulator_step=4,
            source_rgb_sha256=source_rgb_sha256,
            created_action_index=2,
            marked_rgb=rgb,
            bindings={
                "C1": PrivateMarkerBinding(
                    marker="C1",
                    object_name="counter",
                    instance_id=1,
                    visible_pixels=64,
                    official_bbox_xyxy=(0, 0, 7, 7),
                    target_token=support,
                )
            },
            operation="place_on_top",
            stage="target",
            source_segmentation_sha256=source_segmentation_sha256,
        )
    )
    reach_calls = []

    def reject_legacy_reach(target, operation):
        reach_calls.append((target, operation))
        return False, {"distance_m": 3.0}

    backend = _backend(
        prepare=_terminal_preparation(None),
        place_factory=lambda _target: iter(()),
        marker_store=store,
        frame_provider=lambda: frame,
        rgb_provider=lambda: rgb,
        support_names=lambda: ("counter",),
        reach_admission=reject_legacy_reach,
    )

    resolution = backend.consume_marker(
        marker="C1", action_index=3, operation="place_on_top"
    )

    assert resolution.status is MarkerStatus.UNKNOWN
    assert (
        resolution.public_failure_reason
        is MarkerPublicFailureReason.TARGET_OUTSIDE_LOCAL_WORKSPACE
    )
    assert reach_calls == [(support, "place_on_top")]


def test_action20_static_preflight_accepts_route_and_rejects_clipping():
    start = PlanarPose(0.0, 0.0, 0.0, 0.0)
    target = PlanarPose(0.15, 0.0, 0.0, 0.0)
    plan = GlobalSE2RoutePlan(
        status="planned",
        actions=("move_forward",),
        poses=(start, target),
        state_keys=((0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0)),
        selected_goal_index=0,
        expanded_states=1,
        generated_states=1,
        path_cost=1.0,
        reason="goal_reached",
    )
    accepted, collision_samples = _preflight_action20_on_top_route_static(
        plan,
        static_guard=lambda _start, _target: True,
        static_guard_diagnostic=lambda: {"accepted": True},
    )
    assert accepted["accepted"]
    assert accepted["checked_edge_count"] == 1
    assert collision_samples

    guard_calls = []

    def clipped_guard(observed_start, observed_target):
        guard_calls.append((observed_start, observed_target))
        return PlanarPose(0.10, 0.0, 0.0, 0.0)

    report, collision_samples = _preflight_action20_on_top_route_static(
        plan,
        static_guard=clipped_guard,
        static_guard_diagnostic=lambda: {"accepted": False},
    )

    assert len(guard_calls) == 1
    assert not report["accepted"]
    assert report["base_movement_count"] == 0
    assert report["edges_private"][0]["clipped_target_rejected"]
    assert collision_samples == ()


def test_source_snap_escape_exhausts_only_public_translations():
    pose = PlanarPose(0.0, 0.0, 0.0, 0.0)
    rejected = set()
    selected = []
    for _ in range(4):
        action = marker_rollout._oracle_source_snap_escape_action(
            pose=pose,
            waypoint=(1.0, 0.0),
            rejected_actions=rejected,
        )
        assert action in {
            "move_forward",
            "move_backward",
            "move_left",
            "move_right",
        }
        selected.append(action)
        rejected.add(action)
    assert len(set(selected)) == 4
    assert marker_rollout._oracle_source_snap_escape_action(
        pose=pose,
        waypoint=(1.0, 0.0),
        rejected_actions=rejected,
    ) is None


@pytest.mark.parametrize("case", ("same_pose", "yaw_roundoff", "changed_pose", "unexpected_samples"))
def test_zero_step_interaction_route_never_touches_motion_services(case):
    source = ast.parse(inspect.getsource(marker_rollout._build_marker_runtime))
    callback = next(
        node for node in source.body[0].body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_execute_action20_interaction_route"
    )
    pose = PlanarPose(0.0, 0.0, 0.0, 0.0)
    current_pose = (
        PlanarPose(0.0, 0.0, 0.0, 3.5e-9) if case == "yaw_roundoff"
        else PlanarPose(1.0, 0.0, 0.0, 0.0) if case == "changed_pose"
        else pose
    )
    plan = GlobalSE2RoutePlan(
        status="planned", actions=(), poses=(pose,),
        state_keys=((0, 0, 0, 0, 0, 0),), selected_goal_index=0,
        expanded_states=0, generated_states=0, path_cost=0.0, reason="goal_reached",
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("zero-step route must not touch motion or collision services")

    namespace = {
        **vars(marker_rollout),
        "_read_restricted_planar_pose": lambda: current_pose,
        "_action20_pose_matches": forbidden,
        "arm_presentation_ladder": SimpleNamespace(stow_for_base_motion=forbidden),
        "candidate_pose_collision_provider": None,
        "_preflight_action20_on_top_route_collision": forbidden,
        "_build_restricted_kinematic_executor": forbidden,
    }
    exec(compile(ast.Module(body=[callback], type_ignores=[]), inspect.getfile(marker_rollout), "exec"), namespace)
    route = namespace[callback.name](
        plan, (pose,) if case == "unexpected_samples" else (),
        logical_snapshot=None, report={}, step_once=forbidden,
        public_snapshot_provider=forbidden,
    )
    if case == "unexpected_samples":
        with pytest.raises(RuntimeError, match="unexpected collision samples"):
            next(route)
    else:
        with pytest.raises(StopIteration) as stopped:
            next(route)
        report = stopped.value.value
        assert report["arrived_pose"] == current_pose
        assert report["executed_route_action_count"] == 0
        assert report["route_execution_private"] == []
        assert report["full_body_preflight_private"] is None


def test_action20_full_body_preflight_uses_one_batch_and_reports_zero_movement():
    class Provider:
        def __init__(self):
            self.invalidations = 0
            self.batches = []

        def invalidate_world(self):
            self.invalidations += 1

        def check_many(self, samples):
            self.batches.append(tuple(samples))
            return SimpleNamespace(
                predicted_collisions=(False, True),
                audits=({"sample": 0}, {"sample": 1}),
                timing_private={"query_count": 2},
            )

    samples = (
        PlanarPose(0.05, 0.0, 0.0, 0.0),
        PlanarPose(0.10, 0.0, 0.0, 0.0),
    )
    provider = Provider()

    report = _preflight_action20_on_top_route_collision(
        samples,
        collision_provider=provider,
    )

    assert provider.invalidations == 1
    assert provider.batches == [samples]
    assert not report["accepted"]
    assert report["first_collision_sample_index"] == 1
    assert report["base_movement_count"] == 0
