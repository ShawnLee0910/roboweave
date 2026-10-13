"""Exercise the real public Oracle coordinator with CPU scene boundaries."""

import json
from types import SimpleNamespace

import pytest

from behavior_hl.embodied.evaluator_bridge import run_b1_marker_symbolic_rollout as rollout
from behavior_hl.embodied.evaluator_bridge.mixed_goal_graph import compile_grounded_mixed_option
from behavior_hl.embodied.evaluator_bridge.official_predicate_identity import build_official_goal_leaf_ids
from behavior_hl.embodied.runtime.verification import PrivatePredicateSnapshot
from behavior_hl.navigation import PlanarPose


def _run(monkeypatch, tmp_path, *, initial=(True, False), projected=False,
         empty_stage=None, empty_stations=1, mutation=None, no_route=False,
         wrong_leaf=False, regressed_before_node=False, publication_status=None,
         lose_protective_leaf=False, miss_first_placement=False,
         retain_held_first=False, lose_protective_once=False, closed_destination_ancestor=False,
         final_readback_failure=None, partial_q=False, persist_verifier=False,
         workspace_probe=None, articulated_self_blocks=0, articulated_block_kind="robot",
         publication_reason=None, full_goal_case=None, milestone_case=None, history_program=None,
         missing_pick_publications=0, pick_publication_mutation=None, pick_view_status='completed',
         pick_view_centered=False):
    bodies = [["ontop", f"item_{i}", "support"] for i in range(len(initial))]
    if milestone_case is not None:
        bodies = [['open', f'item_{i}'] for i in range(len(initial))]
    if full_goal_case == 'projection_swap':
        bodies = [['ontop','item_0','support_left'], ['ontop','item_1','support_right']]
    plan = compile_grounded_mixed_option(bodies)
    option = [SimpleNamespace(body=body) for body in bodies]
    leaf_ids = (tuple(json.dumps(r.identity, separators=(",", ":")) for r in plan.goal_relations)
                if projected else build_official_goal_leaf_ids([option])[0])
    scope = {name: SimpleNamespace(name=name) for body in bodies for name in body[1:]}
    if history_program is not None:
        scope.update({name: SimpleNamespace(name=name) for name in ('item_1', 'left', 'right')})
    task = SimpleNamespace(object_scope=scope, ground_goal_state_options=[option])
    evaluator = SimpleNamespace(robot=object(), env=SimpleNamespace(task=task))
    state = SimpleNamespace(truth=list(initial), held=None, station=0, reads=0, placements=0)
    state.cabinet_open = not closed_destination_ancestor
    state.open_attempts = 0
    state.pick_publications = 0
    state.history_locations = {'item_0':'support', 'item_1':'support'}
    if full_goal_case is not None:
        # An alternative physical goal may finish while the planning witness
        # remains false. The production verifier, not this fixture, scores OR.
        class Predicate:
            def __init__(self, body, truth):
                self.body, self.truth = body, truth
            def evaluate(self, evaluator):
                return self.truth()
        state.alternative = full_goal_case == 'initial_alternative'
        option = [Predicate(body, lambda i=i: state.truth[i]) for i, body in enumerate(bodies)]
        alternative = [Predicate(body, lambda i=i: state.truth[i]) for i, body in enumerate(bodies[:-1])]
        alternative.append(Predicate(['ontop', bodies[-1][1], 'other_support'], lambda: state.alternative))
        scope['other_support'] = SimpleNamespace(name='other_support')
        task.ground_goal_state_options = [option, alternative]
        if full_goal_case == 'projection_swap':
            # Two interchangeable items permit both destination allocations;
            # select the second full option without changing the complete goal.
            swapped = [Predicate(['ontop','item_0','support_right'], lambda: False),
                       Predicate(['ontop','item_1','support_left'], lambda: False)]
            task.ground_goal_state_options = [swapped, option]
        if not projected:
            leaf_ids = build_official_goal_leaf_ids([option])[0]
        task._evaluate_predicate = lambda *args, **kwargs: False
        if full_goal_case.startswith('projection_'):
            state.extra_on_support = full_goal_case == 'projection_initial_negative_false'
            scope['extra'] = SimpleNamespace(name='extra')
            scope.setdefault('support', SimpleNamespace(name='support'))
            negative = Predicate(['not', ['ontop', 'extra', 'support']], lambda: not state.extra_on_support)
            for branch in task.ground_goal_state_options:
                branch.append(negative)
        if full_goal_case == 'too_many_options':
            task.ground_goal_state_options *= 129
    if articulated_self_blocks:
        evaluator.robot = SimpleNamespace(prim_path="/World/robot")
    if closed_destination_ancestor:
        scope["cabinet"] = SimpleNamespace(name="cabinet", link_prim_paths=("/World/cabinet/lid",))
    calls, navigation, records = [], [], []

    def snapshot(*_args):
        if getattr(state, "final_readback", False):
            if final_readback_failure == "exception":
                raise RuntimeError("final read disconnected")
            if final_readback_failure == "unavailable":
                return PrivatePredicateSnapshot(ok=False, official_success=False,
                                                official_q=None, error="final state unavailable")
            if final_readback_failure == "invalid_q":
                return PrivatePredicateSnapshot(ok=True, official_success=False, official_q=None)
        state.reads += 1
        truth = tuple(state.truth)
        result = PrivatePredicateSnapshot(
            ok=True, official_success=all(truth),
            official_q=sum(truth) / len(truth) if partial_q else 1.0 if all(truth) else 0.0,
            leaf_truth=dict(zip(leaf_ids, truth, strict=True)),
            research_diagnostics={"predicate_mode": "action20_symbolic"},
        )
        if regressed_before_node and state.reads == 1:
            state.truth[0] = False
        return result

    store = SimpleNamespace(active_snapshot=None)
    store.clear = lambda: setattr(store, "active_snapshot", None)
    runtime = SimpleNamespace(
        registry=SimpleNamespace(names=rollout.MODEL_VISIBLE_ACTION20), store=store,
        predicate_mode="action20_symbolic",
        symbolic_primitives=SimpleNamespace(
            _get_obj_in_hand=lambda: state.held,
            private_oracle_exact_report=lambda: {},
            cleanup_symbolic_collision_filter=lambda: {},
        ),
    )

    def publish(operation, stage, created, targets):
        kind = stage if stage in {"surface_cell", "inside_cell", "position"} else "object"
        bindings = {
            f"C{i + 1 + state.station * 10}": SimpleNamespace(
                marker=f"C{i + 1 + state.station * 10}", binding_kind=kind, object_name=obj.name,
                target_token=obj, selector=f"surface_{i}" if kind != "object" else None,
            )
            for i, obj in enumerate(targets)
        }
        store.active_snapshot = SimpleNamespace(
            created_action_index=created, operation=operation, stage=stage,
            bindings=bindings, source_rgb_sha256=f"rgb-{created}",
            source_segmentation_sha256=f"seg-{created}",
        )

    def dispatch(**kwargs):
        from behavior_hl.embodied.evaluator_bridge.l1_process_audit import ACTIVE
        audit = ACTIVE.get() if milestone_case is not None or history_program is not None else None
        before = audit.snapshot(runtime) if audit is not None else None
        operation, args = kwargs["action_name"], kwargs["args"] or {}
        calls.append((operation, dict(args), state.station))
        stage, steps = None, 0
        if operation in {'look_up', 'look_down'}:
            assert store.active_snapshot is None
            status = pick_view_status
        elif not args:
            stage = "target" if operation in {"pick", "open", "close"} else "reference"
            publish(operation, stage, kwargs["action_index"], list(scope.values()))
            status = "selection_required"
        else:
            binding = store.active_snapshot.bindings[args["marker"]]
            if operation == "place_on_top" and binding.binding_kind == "object":
                stage = "surface_cell"
                publish(operation, stage, kwargs["action_index"],
                        [binding.target_token] * (2 if workspace_probe else 1))
                status = "selection_required"
            else:
                status = "completed"
                if operation == "pick":
                    state.held = binding.target_token
                elif milestone_case is not None and operation in {'open', 'close'}:
                    item = int(binding.target_token.name.removeprefix('item_'))
                    if milestone_case == 'false_completed' and operation == 'close':
                        pass
                    else:
                        state.truth[item] = operation == 'open'
                elif operation == "place_on_top":
                    item = int(state.held.name.removeprefix("item_"))
                    state.placements += 1
                    if history_program is not None:
                        state.history_locations[state.held.name] = binding.target_token.name
                        state.truth[0] = state.history_locations['item_0'] == 'support'
                    elif full_goal_case in {'alternative', 'legacy', 'protected_lost'}:
                        state.alternative = True
                        if full_goal_case == 'protected_lost':
                            state.truth[0] = False
                    elif not (miss_first_placement and state.placements == 1):
                        state.truth[(item + 1) % len(initial) if wrong_leaf else item] = True
                    if full_goal_case == 'projection_extra':
                        state.extra_on_support = True
                    if full_goal_case == 'projection_temporary':
                        state.extra_on_support = state.placements == 1
                    if lose_protective_leaf or (lose_protective_once and state.placements == 1):
                        state.truth[0] = False
                    if not (retain_held_first and state.placements == 1):
                        state.held = None
                elif operation == "open" and closed_destination_ancestor:
                    assert state.held is None
                    state.open_attempts += 1
                    if state.open_attempts <= articulated_self_blocks:
                        status = "blocked"
                        if articulated_block_kind == "changed_truth":
                            state.truth[0] = not state.truth[0]
                        path = "/World/wall" if articulated_block_kind == "external" else "/World/robot/right_arm"
                        kwargs["execution_diagnostic_private"].update({
                            "phase": "blocked", "terminal": True,
                            "symbolic_failure_private": {
                                "operation": "open", "blocked": True, "rollback_applied": True,
                                "rollback_verified": articulated_block_kind != "unverified",
                                "physics_steps_executed": 0, "selected_link": "/World/cabinet/lid",
                                "obstructing_objects_private": [],
                                "blocking_contact_pairs_private": [[path, path + "/collision"]],
                            },
                        })
                    else:
                        state.cabinet_open = True
                else:
                    raise AssertionError(f"unexpected operation: {operation}")
        if operation == 'pick' and not args:
            state.pick_publications += 1
            if state.pick_publications <= missing_pick_publications:
                store.active_snapshot.bindings = {k:b for k,b in store.active_snapshot.bindings.items()
                                                  if not b.object_name.startswith('item_')}
                if pick_publication_mutation == 'empty':
                    store.clear()
                    status = 'markers_unavailable'
                elif pick_publication_mutation == 'stale':
                    store.active_snapshot.created_action_index -= 1
                elif pick_publication_mutation == 'held':
                    state.held = scope['item_1']
                elif pick_publication_mutation == 'truth':
                    state.truth[-1] = not state.truth[-1]
                elif pick_publication_mutation == 'physics':
                    steps = 1
                elif pick_publication_mutation == 'pose':
                    state.station += 1
        if empty_stage is not None and stage == empty_stage and state.station < empty_stations:
            status = publication_status or "markers_unavailable"
            if publication_reason:
                kwargs["execution_diagnostic_private"].update(operation=operation, stage=publication_reason)
            store.clear()
            if mutation == "held":
                state.held = None
            elif mutation == "pose":
                state.station += 1
            elif mutation == "truth":
                state.truth[-1] = not state.truth[-1]
            elif mutation == "physics":
                steps = 1
        if status not in kwargs["expected_statuses"]:
            raise RuntimeError(f"public Action20 status drifted: {operation}={status}")
        if audit is not None:
            audit.observe(action=operation, status=status, before=before, after=audit.snapshot(runtime), action_index=kwargs['action_index'])
        return SimpleNamespace(status=status), steps, (kwargs['official_verifier_check']() if full_goal_case is not None or milestone_case is not None or history_program is not None else snapshot())

    rejected_edge = (1, 0, 0, 0, 0, 0, "turn_left")

    def navigate(**kwargs):
        navigation.append(kwargs)
        if milestone_case is None and history_program is None:
            assert store.active_snapshot is None
        else:
            store.clear()
        state.station += 1
        return kwargs["action_index"], 0, {
            "status": "all_stations_rejected" if no_route else "near",
            "selected_station_cell_private": [1, state.station],
            "blocked_transitions_private": [rejected_edge],
        }

    original_write_json = rollout._write_json

    def target_view_action(**kwargs):
        if pick_view_centered:
            raise RuntimeError('target is geometrically centered but absent from current segmentation')
        return 'look_down', {}

    def record_json(path, record):
        records.append((str(path), record))
        if persist_verifier and path.name == "private_verifier.json":
            original_write_json(path, record)

    patches = {
        "_build_marker_runtime": lambda **_kwargs: runtime,
        "_capture_rendered_oracle_observation": lambda _runtime: None,
        "_save_frame": lambda **_kwargs: {},
        "_record_stage": lambda *_args, **_kwargs: None,
        "_write_json": record_json,
        "_append_jsonl": lambda path, record: records.append((str(path), record)),
        "_episode": lambda *_args, **_kwargs: object(),
        "_oracle_marker_route_state": lambda **_kwargs: ("near", 0.5, None),
        "_public_action20_station_is_locally_usable": lambda **_kwargs: milestone_case is None and history_program is None,
        "_current_public_action20_target_visibility": lambda **_kwargs: {
            "visible_for_marker": True, "visible_pixels_private": 100,
        },
        "_public_action20_target_view_action": target_view_action,
        # Match the production readback interface (yaw_rad, not navigator yaw).
        "planar_base_pose_from_robot": lambda _robot: SimpleNamespace(x=0., y=state.station, z=0., yaw_rad=0.),
        "official_world_to_map_cell": lambda **_kwargs: (1, state.station),
        "_dispatch_public_action20_oracle_turn": dispatch,
        "_oracle_navigate_near_se2_navigation_leg": navigate,
    }
    evaluator.env.scene = SimpleNamespace(trav_map=object())
    for name, callback in patches.items():
        monkeypatch.setattr(rollout, name, callback)
    # This fixture isolates public coordination; real selected-region geometry
    # and workspace-kernel parity are covered in test_public_oracle_pregrasp.
    from behavior_hl.embodied.evaluator_bridge import public_oracle_pregrasp
    monkeypatch.setattr(public_oracle_pregrasp, "reachable_on_top_surface_markers",
                        workspace_probe or (lambda bindings, **kw: (bindings, (), [])))
    if closed_destination_ancestor:
        monkeypatch.setattr(rollout, "_closed_openable_ancestor_scope_names",
                            lambda **kw: ("cabinet",) if kw["child_scope_name"] == "support"
                            and not state.cabinet_open else ())
        monkeypatch.setattr(rollout, "_live_open_state_value", lambda *a, **kw: state.cabinet_open)
    if milestone_case is not None:
        monkeypatch.setattr(rollout, '_live_open_state_value', lambda obj, **kw: state.truth[int(obj.name.removeprefix('item_'))])
    profile = SimpleNamespace(task_name="cpu_generic", expected_goal_leaf_count=len(initial))
    if final_readback_failure:
        from behavior_hl.embodied.evaluator_bridge import public_oracle_goal_execution as execution
        original = execution.execute_live_goal_frontier

        def frontier(**kwargs):
            try:
                return original(**kwargs)
            except execution.PublicGoalRecoveryExhausted:
                state.final_readback = True
                raise

        monkeypatch.setattr(execution, "execute_live_goal_frontier", frontier)
    phase_kwargs = dict(
        evaluator=evaluator, control=None, run_dir=tmp_path, stages=tmp_path / "stages.jsonl",
        seed=0, profile=profile, motion_safety_starter=None,
        base_execution="restricted_kinematic", action_profile="marker_symbolic_action20",
        navigation_lane="se2_navigation_leg_v1", public_goal_plan=plan,
        predicate_verifier=SimpleNamespace(evaluate=snapshot),
    )
    if history_program is not None:
        from behavior_hl.embodied.evaluator_bridge import registered_l0_task_runtime as registered
        from behavior_hl.embodied.evaluator_bridge.l1_process_audit import run_process_checked_phase
        branch = {'process_requirements': {'no_manipulation_roles': [], 'before_placement': [], 'history_program': history_program},
                  'initial_reference_goals': [], 'initial_assertions': history_program['clauses']}
        contract = {'role_to_scope': {r:r for r in scope}, 'initial_state_recipe': branch['initial_assertions']}
        source = SimpleNamespace(record={'task_contract':branch}, template_contract=lambda:contract)
        monkeypatch.setattr(registered, '_role_object_for_contract', lambda evaluator, contract, role: scope[role])
        monkeypatch.setattr(registered, '_initial_recipe_truth', lambda a, resolve, **kw:
            state.history_locations.get(a['subject_role']) == a['object_role'])
        result = run_process_checked_phase(rollout._full_oracle_phase, task=source, phase_name='history_oracle', **phase_kwargs)
    elif milestone_case is not None:
        from behavior_hl.embodied.evaluator_bridge import registered_l0_task_runtime as registered
        from behavior_hl.embodied.evaluator_bridge.l1_am_process_programs import lower_process_program
        from behavior_hl.embodied.evaluator_bridge.l1_process_audit import run_process_checked_phase
        roles = list(scope)
        def atom(predicate, role):
            return dict(predicate=predicate, subject_role=role, desired=True)
        program = lower_process_program({'ordered_milestones': [dict(milestone_id='neutral', after=[],
            condition={'op': 'AND', 'args': [atom('closed', role) for role in roles]})]},
            final_goals=[atom('open', role) for role in roles], roles=roles,
            public_instructions={'challenge': 'Close both for inspection, then restore the original states.'})['process_program']
        branch = {'process_requirements': {'no_manipulation_roles': [], 'before_placement': [], 'process_program': program},
                  'initial_reference_goals': [], 'initial_assertions': [atom('open', role) for role in roles]}
        contract = {'role_to_scope': {r: r for r in roles}, 'initial_state_recipe': branch['initial_assertions']}
        source = SimpleNamespace(record={'task_contract': branch}, template_contract=lambda: contract)
        monkeypatch.setattr(registered, '_role_object_for_contract', lambda evaluator, contract, role: scope[role])
        monkeypatch.setattr(registered, '_initial_recipe_truth', lambda a, resolve, **kw:
            state.truth[int(a['subject_role'].removeprefix('item_'))] == (a['predicate'] == 'open'))
        result = run_process_checked_phase(rollout._full_oracle_phase, task=source, phase_name='milestone_oracle', **phase_kwargs)
    elif full_goal_case is None:
        result = rollout._full_oracle_phase(**phase_kwargs)
    else:
        monkeypatch.setattr(rollout, '_require_loaded_profile', lambda *args: None)
        monkeypatch.setattr(rollout, '_prepare_action20_private_segmentation_channel', lambda **kwargs: None)
        monkeypatch.setattr(rollout, '_build_full_motion_safety_starter', lambda evaluator: None)
        # Keep the actual registered/default verifier construction path. Only
        # live option ownership metadata is supplied at the simulator boundary.
        from behavior_hl.embodied.evaluator_bridge.inside_goal_graph import grounded_goal_option_identity
        monkeypatch.setattr(rollout, 'is_registered_l0_task', lambda name: True)
        owner = SimpleNamespace(witness=SimpleNamespace(option_index=0,
            identity=grounded_goal_option_identity(option), compiled_plan=plan),
            private_audit_record=lambda: {},
            compile_live_mixed_plan=lambda **kwargs: compile_grounded_mixed_option(option,
                initially_true_relations=kwargs['initially_true_relations']))
        monkeypatch.setattr(rollout, 'resolve_grounded_option_identity_owner', lambda **kwargs: owner)
        source = SimpleNamespace(record={'task_contract':{'full_goal_acceptance':full_goal_case != 'legacy'}})
        if full_goal_case.startswith('projection_'):
            source.record['task_contract']['positive_goal_projection'] = True
            source.template_contract = lambda: dict(role_to_scope={name:name for name in scope},
                full_goal_witness=['and', *bodies, ['not', ['ontop','extra','support']]])
        callback = rollout.make_public_action20_marker_oracle_callback(
            seed=0, profile=profile, navigation_lane='se2_navigation_leg_v1',
            task_contract_owner=source)
        result = callback(evaluator=evaluator, control=None, bindings=None, run_dir=tmp_path,
                          stages=tmp_path/'stages.jsonl')['strategies'][0]
    return SimpleNamespace(result=result, calls=calls, navigation=navigation,
                           records=records, state=state, rejected_edge=rejected_edge)


def test_full_goal_callback_accepts_nonwitness_option_without_leaf_identity_drift(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, full_goal_case='alternative')
    assert run.result['status'] == 'PASS', run.result.get('error')
    assert run.result['official_q'] == 1.
    assert run.state.truth == [True, False]
    assert run.state.alternative and run.state.placements == 1
    assert run.result['current_true_goal_leaves'] == 1
    assert run.result['official_task_success_scope'] == 'global_all_options'
    final = next(data['final'] for path, data in run.records if path.endswith('private_verifier.json'))
    assert final['research_diagnostics']['selected_witness_private']['official_q'] == 0.


@pytest.mark.parametrize('mode', ['initial_alternative', 'too_many_options'])
def test_full_goal_callback_rejects_invalid_admission_before_actions(monkeypatch, tmp_path, mode):
    run = _run(monkeypatch, tmp_path, full_goal_case=mode)
    assert run.result['status'] == 'FAIL_EXCEPTION'
    assert not run.calls
    assert ('already satisfied initially' if mode == 'initial_alternative' else 'option bound exceeded') in run.result['error']


@pytest.mark.parametrize('mode', ['legacy', 'protected_lost'])
def test_full_goal_preserves_selected_legacy_and_shared_protective_leaves(monkeypatch, tmp_path, mode):
    run = _run(monkeypatch, tmp_path, full_goal_case=mode)
    assert run.result['status'] != 'PASS'
    assert run.result.get('official_q') != 1.


def test_positive_projection_checks_initial_negative_before_any_action(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, full_goal_case='projection_initial_negative_false')
    assert not run.calls
    assert run.result['failure_class'] == 'UNSUPPORTED_NEGATIVE_GOAL_RECOVERY'


def test_positive_projection_does_not_promote_negative_to_process_invariant(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, initial=(False, False), full_goal_case='projection_temporary')
    assert run.result['status'] == 'PASS', run.result.get('error')
    assert run.state.placements == 2 and run.state.extra_on_support is False
    assert run.result['official_q'] == 1.


def test_positive_projection_accepts_other_legal_identity_allocation(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, initial=(False, False), full_goal_case='projection_swap')
    assert run.result['status'] == 'PASS', run.result.get('error')
    assert run.result['official_q'] == 1.
    assert run.state.placements == 2
    assert run.result['positive_goal_projection_private']['complete_option_index'] == 1


def test_positive_projection_refuses_extra_object_even_when_all_planned_leaves_true(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, full_goal_case='projection_extra')
    assert all(run.state.truth)
    assert run.result['status'] != 'PASS'
    assert run.result['failure_class'] == 'UNSUPPORTED_NEGATIVE_GOAL_RECOVERY'


@pytest.mark.parametrize("projected", [False, True])
def test_satisfied_leaf_is_skipped_but_still_required_at_final(monkeypatch, tmp_path, projected):
    run = _run(monkeypatch, tmp_path, projected=projected)
    assert run.result["status"] == "PASS", run.result.get("error")
    assert run.result["new_official_goal_leaves"] == 1
    assert run.result["current_true_goal_leaves"] == 2
    # Live compilation omits the already-true placement but retains its goal.
    assert len(run.result["route_reports"]) == 1
    assert run.result["route_reports"][0]["node_id"].find("item_0") == -1
    assert len(run.calls) == 5  # one Pick + one three-call OnTop; no protective re-placement
    assert not run.navigation


def test_current_truth_can_repair_an_initially_true_leaf_without_new_credit(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, regressed_before_node=True)
    assert run.result["status"] == "PASS", run.result.get("error")
    assert len(run.calls) == 10
    assert run.result["new_official_goal_leaves"] == 1


def test_unrelated_leaf_gain_does_not_complete_the_attempted_node(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, initial=(False, False), wrong_leaf=True)
    assert run.result["status"] == "FAIL_EXCEPTION"
    assert "node goal remains false" in run.result["error"]
    assert run.result["route_reports"][-1]["new_official_goal_leaves"] == 1


def test_skipped_protective_goal_cannot_be_lost_at_final(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, lose_protective_leaf=True)
    assert run.result["status"] == "FAIL_EXCEPTION"
    assert run.result["failure_code"] == "goal_recovery_budget_exhausted"
    assert run.result["official_q"] == 0
    assert run.result["new_official_goal_leaves"] == 1
    assert run.result["current_true_goal_leaves"] == 1
    assert run.result["failure_class"] == "ORACLE_RESOURCE_EXHAUSTED"
    assert run.result["failure_owner"] == "UNKNOWN"
    assert run.state.truth == [False, True]


@pytest.mark.parametrize("projected", [False, True])
def test_exhaustion_publishes_final_proof_before_episode_closure(monkeypatch, tmp_path, projected):
    from behavior_hl.embodied.evaluator_bridge.oracle_episode_receipt import (
        verification_scope, read_document, task_directory, commit_episode_result,
    )
    with verification_scope(tmp_path, "task"):
        run = _run(monkeypatch, tmp_path, lose_protective_leaf=True,
                   partial_q=True, projected=projected, persist_verifier=True)
    assert run.result["status"] == "FAIL_EXCEPTION"
    assert run.result["official_q"] == 0.5
    proof = read_document(task_directory(tmp_path, "task") / "final_verification.json")
    assert proof["official_q"] == 0.5
    from pathlib import Path
    private_path = [path for path, _ in run.records if path.endswith("private_verifier.json")][-1]
    private = read_document(Path(private_path))
    assert private["final"]["official_q"] == 0.5
    assert sum(private["final"]["leaf_truth"].values()) == 1
    # The outer failure receipt must retain formally committed verification.
    closed = commit_episode_result(tmp_path, "task", {
        "status": "FAIL", "official_q": None, "verification_status": "INCOMPLETE",
        "marker_action_gate_attempted": True, "error_type": run.result["error_type"],
    })
    assert closed["official_q"] == 0.5
    assert closed["verification_status"] == "COMPLETED"
    assert closed["goal_status"] == "FAIL"
    assert closed["oracle_execution_status"] == "INCOMPLETE"


@pytest.mark.parametrize("mode", ["exception", "unavailable", "invalid_q"])
def test_exhaustion_final_read_failure_does_not_reuse_old_q(monkeypatch, tmp_path, mode):
    from behavior_hl.embodied.evaluator_bridge.oracle_episode_receipt import (
        verification_scope, task_directory, commit_episode_result,
    )
    with verification_scope(tmp_path, "task"):
        run = _run(monkeypatch, tmp_path, lose_protective_leaf=True,
                   partial_q=True, final_readback_failure=mode)
    assert run.result["failure_code"] == "goal_recovery_budget_exhausted"
    assert run.result["official_q"] is None
    assert run.result["current_true_goal_leaves"] is None
    assert run.result["verification_status"] == "UNAVAILABLE"
    assert "final_verification_error" in run.result
    assert not (task_directory(tmp_path, "task") / "final_verification.json").exists()
    closed = commit_episode_result(tmp_path, "task", {
        **run.result, "status": "FAIL", "marker_action_gate_attempted": True,
    })
    assert closed["official_q"] is None
    assert closed["goal_status"] == "UNKNOWN"


def test_exhaustion_publication_failure_preserves_stop_and_measured_q(monkeypatch, tmp_path):
    from behavior_hl.embodied.evaluator_bridge import oracle_episode_receipt as receipt

    def broken_publication(**_kwargs):
        raise OSError("proof write unavailable")

    monkeypatch.setattr(receipt, "publish_final_verification", broken_publication)
    run = _run(monkeypatch, tmp_path, lose_protective_leaf=True, partial_q=True)
    assert run.result["failure_code"] == "goal_recovery_budget_exhausted"
    assert run.result["official_q"] == 0.5
    assert run.result["evidence_status"] == "INCOMPLETE"
    assert run.result["final_verification_evidence_error"]["error_type"] == "OSError"


def test_public_lane_recompiles_after_false_completed_placement(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, miss_first_placement=True)
    assert run.result["status"] == "PASS", run.result.get("error")
    assert run.state.placements == 2
    probe_rows = [row for path, row in run.records if path.endswith("public_pregrasp_probes.jsonl")]
    assert run.result["pregrasp_probe_attempt_count"] == len(probe_rows) == 2
    assert run.result["pregrasp_probe_budget_scope"] == "PLAN_REVISION"
    assert any(row.get("event") == "goal_node_readback_recompile" for _, row in run.records)


def test_recompile_does_not_replay_failed_surface_in_same_context(monkeypatch, tmp_path):
    monkeypatch.setattr(rollout, "_public_oracle_candidate_context", lambda **kw: ("station", "held", "scene"))
    def probe(bindings, **kw):
        return bindings, (), []
    run = _run(monkeypatch, tmp_path, miss_first_placement=True, workspace_probe=probe)
    assert run.result["status"] == "PASS", run.result.get("error")
    surface_actions = [args["marker"] for op, args, _ in run.calls
                       if op == "place_on_top" and args.get("marker") in {"C1", "C2"}]
    # Target reference uses a different label; applied surface C1 is not
    # retried after completed-but-false. The same fresh snapshot offers C2.
    assert surface_actions[-1] == "C2"
    assert run.state.placements == 2
    assert any(row.get("event") == "reuse_failed_selectors" for _, row in run.records)


def test_recompile_exhausted_single_surface_does_not_repeat_placement(monkeypatch, tmp_path):
    monkeypatch.setattr(rollout, "_public_oracle_candidate_context", lambda **kw: ("same",))
    run = _run(monkeypatch, tmp_path, miss_first_placement=True)
    assert run.result["status"] == "FAIL_EXCEPTION"
    assert run.state.placements == 1
    assert run.result["candidate_rejection_private"]["reason"] == "marker_selector_frontier_exhausted"


def test_candidate_context_ignores_projection_and_roundoff_but_tracks_scene(monkeypatch):
    import numpy as np
    held = SimpleNamespace(name="held")
    support = SimpleNamespace(name="support", n_joints=0)
    point = np.array([1.0, 2.0, 3.0])
    support.get_position_orientation = lambda: (point, np.array([0., 0., 0., 1.]))
    root_pose = np.eye(4)
    owner = SimpleNamespace(_get_obj_in_hand=lambda: held,
                            _logical_inventory_origin_pose=root_pose)
    runtime = SimpleNamespace(symbolic_primitives=owner)
    evaluator = SimpleNamespace(robot=object(), env=SimpleNamespace(
        task=SimpleNamespace(object_scope={"held": held, "support": support})))
    monkeypatch.setattr(rollout, "planar_base_pose_from_robot", lambda _: PlanarPose(0, 0, 0, 0))
    def key():
        return rollout._public_oracle_candidate_context(runtime=runtime, evaluator=evaluator,
                                                       target=support, opened_doors=set())
    baseline = key()
    assert baseline is not None
    point[0] += 1e-7
    root_pose[:3, 3] = [8, 9, 10]
    assert key() == baseline
    point[0] += .1
    assert key() != baseline


def test_stable_region_and_nested_part_identity_ignore_labels_and_float_noise():
    def surface(label, z):
        return SimpleNamespace(marker=label, binding_kind="surface_cell", selector=label,
            target_token=SimpleNamespace(region_center_xy=(1., 2.), region_extent_xy=(.5,.6), surface_z_m=z))
    assert rollout._public_oracle_selector_identity(surface("C1", .7)) == rollout._public_oracle_selector_identity(surface("C9", .70000001))
    def part(label, link):
        return SimpleNamespace(marker=label, binding_kind="inside_cell", selector=label,
            target_token=SimpleNamespace(target_token=SimpleNamespace(link_prim_path=link, selector="joint")))
    assert rollout._public_oracle_selector_identity(part("C1", "/drawer/a")) == rollout._public_oracle_selector_identity(part("C2", "/drawer/a"))
    assert rollout._public_oracle_selector_identity(part("C1", "/drawer/a")) != rollout._public_oracle_selector_identity(part("C1", "/drawer/b"))


def test_actual_retry_gate_consumes_current_articulation_report_not_joint_equality():
    from test_oracle_articulated_recovery import candidate_diagnostic
    args = dict(operation="open", status="blocked", private_report={}, held_before=None, held_after=None,
                open_joint_before=(0.,), open_joint_after=(0.,),
                target_link_paths=("/World/box/lid",), selected_joint="lid_joint",
                canonical_before={"leaf": False}, canonical_after={"leaf": False})
    assert not rollout._public_action20_candidate_retry_safe(**args)
    assert rollout._public_action20_candidate_retry_safe(**args, execution_diagnostic_private=candidate_diagnostic())
    d = candidate_diagnostic()
    d["symbolic_failure_private"]["partial_progress_preserved"] = True
    assert not rollout._public_action20_candidate_retry_safe(**args, execution_diagnostic_private=d)


def test_public_lane_continues_placement_without_second_pick_when_still_held(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, miss_first_placement=True, retain_held_first=True)
    assert run.result["status"] == "PASS", run.result.get("error")
    assert run.state.placements == 2
    assert len([call for call in run.calls if call[0] == "pick"]) == 2  # one selection + execution


def test_public_lane_repairs_lost_preservation_leaf(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, lose_protective_once=True)
    assert run.result["status"] == "PASS", run.result.get("error")
    assert run.state.placements == 2


def test_destination_closed_ancestor_is_opened_publicly_before_pick(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, closed_destination_ancestor=True)
    assert run.result["status"] == "PASS", run.result.get("error")
    assert run.state.cabinet_open
    operations = [operation for operation, _, _ in run.calls]
    assert operations.index("open") < operations.index("pick")
    assert run.result["semantic_completed_action_count"] == 3


def test_public_surface_selection_prefers_reachable_marker_not_smallest_label(monkeypatch, tmp_path):
    def probe(bindings, **kwargs):
        assert kwargs["maximum_distance_m"] == rollout.ACTION20_LOCAL_WORKSPACE_M
        return (bindings[-1],), (), [{"marker": bindings[0].marker, "workspace_candidate_count": 0}]

    run = _run(monkeypatch, tmp_path, workspace_probe=probe)
    assert run.result["status"] == "PASS", run.result.get("error")
    assert run.calls[-1][1] == {"marker": "C2"}
    assert not run.navigation


def test_all_far_surfaces_reject_station_before_placement_dispatch(monkeypatch, tmp_path):
    calls = []

    def probe(bindings, **kwargs):
        calls.append(kwargs["base_xy"])
        return ((), (object(),), []) if len(calls) == 1 else (bindings, (), [])

    run = _run(monkeypatch, tmp_path, workspace_probe=probe)
    assert run.result["status"] == "PASS", run.result.get("error")
    assert len(run.navigation) == 1
    assert callable(run.navigation[0]["station_workspace_validator"])
    assert len([call for call in run.calls if call[0] == "pick"]) == 2
    assert [row for _, row in run.records
            if row.get("station_rejection_reason") == "surface_workspace_unreachable"]
    assert run.result["semantic_completed_action_count"] == 2


def test_goal_leaf_mapping_fails_closed_on_unknown_snapshot_identity():
    body = ["ontop", "item", "support"]
    plan = compile_grounded_mixed_option([body])
    with pytest.raises(RuntimeError, match="leaf identity drifted"):
        rollout._public_action20_goal_node_leaf_ids(
            plan=plan, task=SimpleNamespace(ground_goal_state_options=[[SimpleNamespace(body=body)]]),
            snapshot=SimpleNamespace(ok=True, leaf_truth={"unrelated_id": True}),
        )


def test_initially_satisfied_task_is_not_promoted_as_an_oracle_success(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, initial=(True, True))
    assert run.result["status"] == "FAIL_EXCEPTION"
    assert "already satisfied initially" in run.result["error"]
    assert not run.calls


@pytest.mark.parametrize("empty_stage", ["reference", "surface_cell"])
def test_empty_publication_restarts_fresh_selection_after_shared_navigation(monkeypatch, tmp_path, empty_stage):
    run = _run(monkeypatch, tmp_path, empty_stage=empty_stage, empty_stations=2)
    assert run.result["status"] == "PASS", run.result.get("error")
    assert len(run.navigation) == 2
    assert set(run.navigation[1]["excluded_station_cells"]) == {(1, 0), (1, 1)}
    assert run.navigation[1]["initial_blocked_transitions"] == (run.rejected_edge,)
    assert run.state.held is None  # released only by the final placement
    for station in (1, 2):
        calls = [call for call in run.calls if call[2] == station]
        assert calls[0][:2] == ("place_on_top", {})  # no stale reference C-label
    failures = [record for path, record in run.records if "publication_failures" in path]
    assert len(failures) == 2
    assert all(row["stage"] == empty_stage for row in failures)
    assert all(row["publication_filter_reason"] == "UNKNOWN" for row in failures)


def test_empty_publication_exhaustion_is_bounded_and_retains_the_held_object(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, empty_stage="surface_cell", no_route=True)
    assert run.result["failure_class"] == "ORACLE_PLANNER_ERROR"
    assert run.result["candidate_rejection_private"]["reason"] == "marker_station_candidates_exhausted"
    assert len(run.navigation) == 1
    assert len(run.calls) == 4  # Pick, support selection, no physical placement
    assert run.state.held.name == "item_1"
    assert not run.result["alternative_plan_retry_safe"]


@pytest.mark.parametrize("mutation", ["held", "pose", "truth", "physics"])
def test_empty_publication_does_not_retry_a_mutated_state(monkeypatch, tmp_path, mutation):
    run = _run(monkeypatch, tmp_path, empty_stage="surface_cell", mutation=mutation)
    assert run.result["failure_class"] == "ACTION20_PROTOCOL_ERROR"
    assert "not a verified no-op" in run.result["error"]
    assert not run.navigation


def test_uncertain_publication_is_not_treated_as_safe_station_rejection(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, empty_stage="surface_cell", publication_status="uncertain")
    assert run.result["status"] == "FAIL_EXCEPTION"
    assert "status drifted" in run.result["error"]
    assert not run.navigation


@pytest.mark.parametrize("reason,should_move", [
    ("marker_frame_unavailable", False), ("reference_no_longer_visible", True),
])
def test_current_publication_reason_reaches_oracle_recovery(monkeypatch, tmp_path, reason, should_move):
    run = _run(monkeypatch, tmp_path, empty_stage="surface_cell", publication_reason=reason)
    assert bool(run.navigation) is should_move
    assert (run.result["status"] == "PASS") is should_move
    events = [row for path, row in run.records if path.endswith("public_action20_marker_publication_failures.jsonl")]
    assert events[0]["publication_filter_reason"] == reason
    assert events[0]["publication_diagnostic_private"] == {"operation": "place_on_top", "stage": reason}
