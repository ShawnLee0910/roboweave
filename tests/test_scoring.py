import copy
import json
from pathlib import Path

import pytest

from roboweave.scoring import score_record


def official():
    return dict(schema_version="roboweave/score-record-v1", route="official_full",
                initial_options=[[True, False, False]], snapshots=[{"options": [ [True, True, False]]}])


def test_official_new_credit_and_integrity():
    record = official()
    assert score_record(record)["final"]["q_score"] == pytest.approx(1/3)
    record["snapshots"][0]["options"] = [[True, True, True]]
    assert score_record(record)["final"]["success"]
    record["snapshots"][0]["integrity_ok"] = False
    assert score_record(record)["final"] == dict(operational=False, success=False, q_score=None,
                                                errors=["recorded execution integrity failure"])


@pytest.mark.parametrize("next_frame", [{"options": [[True, True, True]]},
    {"options": [[True, True, True]], "integrity_ok": True}])
def test_official_integrity_failure_cannot_be_cleared_by_a_later_snapshot(next_frame):
    record = official()
    record["snapshots"] += [dict(options=[[True, True, True]], integrity_ok=False), next_frame]
    result = score_record(record)
    assert result["snapshots"][0]["operational"] is True
    for snapshot in result["snapshots"][1:]:
        assert snapshot["operational"] is False
        assert snapshot["success"] is False
        assert snapshot["q_score"] is None


def test_projected_stability_initial_credit_and_missing_evidence():
    path = Path(__file__).parents[1] / "examples/score-record.json"
    record = json.loads(path.read_text())
    result = score_record(record)
    assert not result["snapshots"][0]["success"]
    assert result["final"]["success"]
    record["initial_truth"][0]["value"] = True
    result = score_record(record)
    assert not result["final"]["success"]
    assert result["final"]["q_score"] == 0
    assert result["final"]["early_done"]
    record["snapshots"][-1]["truth"] = []
    assert score_record(record)["final"]["operational"] is False


@pytest.mark.parametrize('route', ['official_projection', 'canonical_fallback'])
@pytest.mark.parametrize('phase', ['initial_truth', 'snapshot'])
@pytest.mark.parametrize('positive,negative', [('open', 'closed'), ('toggled_on', 'toggled_off')])
def test_projected_rejects_contradictory_native_state_readbacks(route, phase, positive, negative):
    record = dict(schema_version='roboweave/score-record-v1', route=route,
        goal_spec=dict(route=route, clauses=[dict(canonical_predicate=predicate, quantifier='ALL_LISTED',
            target_scopes=['device'], destination_scopes=[]) for predicate in (positive, negative)]),
        initial_truth=[dict(predicate=predicate, target='device', destination=None, value=value)
                       for predicate, value in [(positive, False), (negative, True)]],
        snapshots=[dict(truth=[dict(predicate=predicate, target='device', destination=None, value=value)
                              for predicate, value in [(positive, True), (negative, False)]])])
    rows = record['initial_truth'] if phase == 'initial_truth' else record['snapshots'][0]['truth']
    rows[1]['value'] = rows[0]['value']
    with pytest.raises(ValueError, match='contradictory.*state'):
        score_record(record)


def test_routes_and_shapes_cannot_be_substituted():
    record = official()
    record["snapshots"][0]["options"] = [[True]]
    with pytest.raises(ValueError, match="structure"):
        score_record(record)
    record = official()
    record["initial_options"] = [[1, False, False]]
    with pytest.raises(ValueError, match="boolean"):
        score_record(record)


def test_process_gate_preserves_canonical_q_and_rejects_missing_history():
    record = official()
    record["snapshots"] = [{"options": [[True, True, True]]}]
    record["requires_process"] = True
    result = score_record(record)
    assert result["canonical"]["success"] is True
    assert result["final"]["q_score"] == 1
    assert result["final"]["success"] is False
    assert result["final"]["failure_class"] == "VERIFIER_ERROR"


def test_original_process_audit_rejects_protected_object(tmp_path):
    record = official()
    record["snapshots"] = [{"options": [[True, True, True]]}]
    before = dict(held_roles=[], prerequisites=[], protected_relations=[True])
    after = dict(before, held_roles=["protected"])
    record["process"] = dict(roles=["protected"], requirements=dict(no_manipulation_roles=["protected"], before_placement=[]),
                             initial_truth=[], final_truth=[], events=[dict(action="pick", before=before, after=after,
                                                                           status="completed", action_index=0)])
    result = score_record(record, process_directory=tmp_path / "process")
    assert not result["final"]["success"]
    assert result["final"]["failure_class"] == "PUBLIC_PROCESS_REQUIREMENT_FAILED"
    assert result["process"]["violations"][0]["kind"] == "PROTECTED_OBJECT_MANIPULATED_OR_RELATION_LOST"
    assert (tmp_path / "process/private/task_process_events.jsonl").is_file()


def process_record():
    record = official()
    record["snapshots"] = [{"options": [[True, True, True]]}]
    snapshot = dict(held_roles=[], prerequisites=[], protected_relations=[True])
    record["process"] = dict(roles=["protected"],
                             requirements=dict(no_manipulation_roles=["protected"], before_placement=[]),
                             initial_truth=[], final_truth=[],
                             events=[dict(action="pick", before=copy.deepcopy(snapshot), after=snapshot,
                                          status="completed", action_index=0)])
    return record


@pytest.mark.parametrize("phase", ["before", "after"])
@pytest.mark.parametrize("snapshot", [None, {}, [], "unavailable", False,
    {"held_roles": [], "prerequisites": []},
    {"held_roles": "protected", "prerequisites": [], "protected_relations": [True]},
    {"held_roles": [], "prerequisites": [], "protected_relations": [1]},
    {"held_roles": [], "prerequisites": [], "protected_relations": {}},
    {"held_roles": [[]], "prerequisites": [], "protected_relations": [True]},
    {"held_roles": ["unknown"], "prerequisites": [], "protected_relations": [True]},
])
def test_process_replay_rejects_missing_or_malformed_snapshots(tmp_path, phase, snapshot):
    record = process_record()
    record["process"]["events"][0][phase] = snapshot
    directory = tmp_path / "process"
    with pytest.raises(ValueError):
        score_record(record, process_directory=directory)
    assert not directory.exists()


def test_process_replay_rejects_dropped_protection_evidence(tmp_path):
    record = process_record()
    record["process"]["events"][0]["after"]["protected_relations"] = []
    with pytest.raises(ValueError, match="readback counts changed"):
        score_record(record, process_directory=tmp_path / "process")


@pytest.mark.parametrize("status,selected,success,failure", [
    ("completed", [], True, None),
    ("completed", ["protected"], False, "PUBLIC_PROCESS_REQUIREMENT_FAILED"),
    ("blocked", ["protected"], False, "VERIFIER_ERROR"),
])
def test_replay_keeps_native_pass_violation_and_unavailable_classification(tmp_path, status, selected, success, failure):
    from behavior_hl.embodied.evaluator_bridge.l1_process_audit import ProcessAudit
    record = process_record()
    process = record["process"]
    event = process["events"][0]
    event["status"] = status
    event["before"]["selected_roles"] = selected
    result = score_record(record, process_directory=tmp_path / "replay")
    native = ProcessAudit(process["requirements"], role_object=lambda role: role, recipe_truth=lambda *args: True,
                          protected_relations=(), directory=tmp_path / "native")
    native.observe(**event)
    expected = native.report()
    for field in ("operational", "public_process_satisfied", "violations", "errors", "execution_events"):
        assert result["process"][field] == expected[field]
    assert result["canonical"]["success"] is True
    assert result["final"]["q_score"] == 1
    assert result["final"]["success"] is success
    assert result["final"].get("failure_class") == failure


@pytest.mark.parametrize("states", [None, [], {}, [1]])
def test_state_contract_requires_boolean_readbacks(tmp_path, states):
    record = process_record()
    record["process"]["requirements"]["state_protection_roles"] = ["protected"]
    for phase in ("before", "after"):
        record["process"]["events"][0][phase]["protected_states"] = states
    with pytest.raises(ValueError, match="protected_states"):
        score_record(record, process_directory=tmp_path / "process")


def test_state_change_remains_native_process_failure(tmp_path):
    record = process_record()
    record["process"]["requirements"]["state_protection_roles"] = ["protected"]
    record["process"]["events"][0]["before"]["protected_states"] = [True]
    record["process"]["events"][0]["after"]["protected_states"] = [False]
    result = score_record(record, process_directory=tmp_path / "process")
    assert result["final"]["failure_class"] == "PUBLIC_PROCESS_REQUIREMENT_FAILED"
    assert result["process"]["violations"][0]["kind"] == "PROTECTED_STATE_CHANGED"


@pytest.mark.parametrize("ready,expected", [(True, True), (False, False)])
def test_placement_contract_requires_prerequisites_and_goal_readbacks(tmp_path, ready, expected):
    record = process_record()
    record["process"]["requirements"] = dict(no_manipulation_roles=[], before_placement=[
        dict(subject_role="protected", required_relations=[dict(predicate="closed", subject_role="protected", desired=True)])])
    event = record["process"]["events"][0]
    event["action"] = "place_on_top"
    event["before"].update(held_roles=["protected"], prerequisites=[ready], placement_goals=[False])
    event["after"].update(prerequisites=[True], placement_goals=[True])
    result = score_record(record, process_directory=tmp_path / "valid")
    assert result["final"]["success"] is expected
    del event["after"]["placement_goals"]
    with pytest.raises(ValueError, match="placement_goals"):
        score_record(record, process_directory=tmp_path / "missing")


def test_current_reference_requires_complete_domain_readbacks(tmp_path):
    record = process_record()
    process = record["process"]
    process["roles"] += ["target", "support"]
    process["requirements"]["current_reference_contracts"] = [dict(binding_id="current",
        time="CURRENT_READBACK_BEFORE_TARGET_PLACEMENT", update="CURRENT", cardinality="EXACTLY_ONE",
        predicate="ontop", subject_role="protected", domain_roles=["support"],
        goal_predicate="ontop", goal_subject_role="target")]
    event = process["events"][0]
    event["action"] = "place_on_top"
    event["before"].update(held_roles=["target"], current_references=[dict(source_matches=["support"], target_matches=[])])
    event["after"]["current_references"] = [dict(source_matches=["support"], target_matches=["support"])]
    process["final_truth"] = [dict(atom=dict(predicate="ontop", subject_role="target", object_role="support", desired=True), value=True)]
    assert score_record(record, process_directory=tmp_path / "valid")["final"]["success"]
    for index, value in enumerate(([], [{}], [dict(source_matches=["support"], target_matches="support")])):
        event["after"]["current_references"] = value
        with pytest.raises(ValueError, match="current_references"):
            score_record(record, process_directory=tmp_path / str(index))


def milestone_record():
    record = process_record()
    process = record["process"]
    atom = dict(predicate="open", subject_role="protected", desired=True)
    process["requirements"] = dict(no_manipulation_roles=[], before_placement=[], process_program=dict(
        schema_version="l1-public-process-program/v1", milestones=[dict(milestone_id="opened", after=[], requirements=[atom])],
        final_requirements=[atom], final_placement_after=None,
        completion_rule="ALL_DECLARED_MILESTONES_AND_CURRENT_FINAL_REQUIREMENTS",
        condition_evaluation="CANONICAL_READBACK_AFTER_VERIFIED_EXECUTION_OR_INITIAL_READBACK",
        milestone_memory="LATCH_ONCE_AFTER_DECLARED_PREDECESSORS_SATISFIED",
        source_process={}, public_instruction_evidence={"instruction": "Open the protected device."}))
    process["initial_truth"] = [dict(atom=atom, value=False)]
    process["final_truth"] = [dict(atom=atom, value=True)]
    event = process["events"][0]
    event["action"] = "open"
    event["before"]["milestones"] = dict(opened=False)
    event["after"]["milestones"] = dict(opened=True)
    return record


@pytest.mark.parametrize("profile", ["l1-p-inside5", "l1-p-ontop5"])
def test_v1_older_profiles_replay_their_supported_single_milestone(profile, tmp_path):
    result = score_record(milestone_record(), profile=profile, process_directory=tmp_path / "process")
    assert result["final"]["success"] is True
    assert result["process"]["milestones_reached"] == {"opened": 1}


def test_milestone_contract_requires_declared_boolean_readbacks(tmp_path):
    record = milestone_record()
    process = record["process"]
    event = process["events"][0]
    assert score_record(record, process_directory=tmp_path / "valid")["final"]["success"]
    for index, value in enumerate((None, {}, dict(opened=1))):
        event["after"]["milestones"] = value
        with pytest.raises(ValueError, match="milestones"):
            score_record(record, process_directory=tmp_path / str(index))
    process["requirements"]["process_program"]["global_unary_phases"] = True
    event["before"]["milestones"] = dict(opened=False, _unary=dict(protected=False))
    event["after"]["milestones"] = dict(opened=True, _unary=dict(protected=True))
    assert score_record(record, process_directory=tmp_path / "unary-valid")["final"]["success"]
    for index, value in enumerate((None, {}, dict(protected=1))):
        event["after"]["milestones"]["_unary"] = value
        with pytest.raises(ValueError, match="_unary"):
            score_record(record, process_directory=tmp_path / f"unary-{index}")


def test_history_replay_rejects_absent_per_event_truth(tmp_path):
    record = process_record()
    atom = dict(predicate="open", subject_role="protected", desired=True)
    record["process"]["requirements"]["history_program"] = dict(schema_version="l1-history-bindings/v1",
        bindings=[], clauses=[atom], expression=atom,
        process=dict(ordered_milestones=[dict(milestone_id="opened", condition=atom)]))
    with pytest.raises(ValueError, match="per-event predicate truth"):
        score_record(record, process_directory=tmp_path / "process")
    assert not (tmp_path / "process").exists()


def history_record():
    return json.loads((Path(__file__).parents[1] / 'examples/history-score-record.json').read_text())


def native_v2(record, directory):
    """Direct frozen API sequence, independent of the wrapper reconstruction."""
    from types import SimpleNamespace
    from behavior_hl.embodied.evaluator_bridge.l1_process_audit import ProcessAudit
    process = record['process']
    objects = {r: object() for r in process['roles']}
    current = process['initial_truth']
    def truth(atom, _roles):
        return next(row['value'] for row in current if row['atom'] == atom)
    audit = ProcessAudit(process['requirements'], role_object=objects.__getitem__, recipe_truth=truth,
        protected_relations=process['protected_relations'],
        protected_state_relations=process['protected_state_relations'],
        placement_targets=process['placement_targets'], directory=directory)
    for event in process['events']:
        snapshots = []
        for phase in ('before', 'after'):
            current = event[phase]['truth']
            held = event[phase]['held_roles']
            runtime = SimpleNamespace(symbolic_primitives=SimpleNamespace(
                _get_obj_in_hand=lambda: objects[held[0]] if held else None))
            snapshot = audit.snapshot(runtime)
            assert snapshot is not None
            if 'selected_roles' in event[phase]:
                snapshot['selected_roles'] = event[phase]['selected_roles']
            snapshots.append(snapshot)
        audit.observe(action=event['action'], before=snapshots[0], after=snapshots[1],
                      status=event['status'], action_index=event['action_index'])
    current = process['final_truth']
    return audit.report()


def assert_native_v2_parity(record, tmp_path):
    result = score_record(record, process_directory=tmp_path / 'replay')
    expected = native_v2(record, tmp_path / 'native')
    for field in expected.keys() - {'process_events', 'observation_index'}:
        assert result['process'][field] == expected[field], field
    actual_events = (tmp_path / 'replay/private/task_process_events.jsonl').read_text()
    assert actual_events == (tmp_path / 'native/private/task_process_events.jsonl').read_text()
    return result


def set_atom(rows, subject, predicate, value, destination=None):
    row = next(r for r in rows if r['atom']['subject_role'] == subject
               and r['atom']['predicate'] == predicate and r['atom'].get('object_role') == destination)
    row['value'] = value


def test_v2_history_actual_subject_maintenance_and_order_match_native(tmp_path):
    result = assert_native_v2_parity(history_record(), tmp_path)
    assert result['final']['success'] is True
    assert result['process']['history_bindings'] == {'first': 'item_a'}
    assert result['process']['history_binding_events'] == {'first': 1}


@pytest.mark.parametrize('case', ['initial_truth', 'wrong_subject', 'early_close', 'early_return',
                                   'same_event_predecessor', 'blocked', 'uncertain', 'needs_marker'])
def test_v2_history_never_substitutes_final_static_truth_for_history(case, tmp_path):
    record = history_record()
    process = record['process']
    if case == 'initial_truth':
        # The initial map cannot replace later event readbacks.
        process['initial_truth'] = copy.deepcopy(process['final_truth'])
        assert assert_native_v2_parity(record, tmp_path)['final']['success'] is True
        return
    if case == 'wrong_subject':
        for subject, tray, table in [('item_a', True, False), ('item_b', False, True)]:
            set_atom(process['final_truth'], subject, 'ontop', tray, 'tray')
            set_atom(process['final_truth'], subject, 'ontop', table, 'table')
    elif case == 'early_close':
        set_atom(process['events'][0]['after']['truth'], 'cabinet', 'open', False)
        set_atom(process['events'][0]['after']['truth'], 'cabinet', 'closed', True)
    elif case == 'early_return':
        process['events'][2], process['events'][3] = process['events'][3], process['events'][2]
        for i, event in enumerate(process['events'], 1):
            event['event_id'] = i
    elif case == 'same_event_predecessor':
        # Closing at the same event as stored cannot establish its dependent stage.
        process['requirements']['history_program']['process'].pop('maintain_until')
        process['events'][2]['status'] = 'needs_marker'
        set_atom(process['events'][1]['after']['truth'], 'cabinet', 'closed', True)
        set_atom(process['events'][1]['after']['truth'], 'cabinet', 'open', False)
    else:
        process['events'][0]['status'] = case
    result = assert_native_v2_parity(record, tmp_path)
    assert result['canonical']['success'] is True
    assert result['final']['q_score'] == 1
    assert result['final']['success'] is False
    assert result['final']['failure_class'] in {'VERIFIER_ERROR', 'PUBLIC_PROCESS_REQUIREMENT_FAILED'}


@pytest.mark.parametrize('phase', ['before', 'after', 'final_truth'])
def test_v2_missing_phase_truth_fails_closed(tmp_path, phase):
    record = history_record()
    if phase == 'final_truth':
        record['process'][phase] = []
    else:
        record['process']['events'][0][phase]['truth'] = []
    with pytest.raises(ValueError, match='predicate truth'):
        score_record(record, process_directory=tmp_path / 'process')


@pytest.mark.parametrize('phase', ['initial_truth', 'before', 'after', 'final_truth'])
@pytest.mark.parametrize('positive,negative', [('open', 'closed'), ('toggled_on', 'toggled_off')])
def test_process_replay_rejects_contradictory_native_state_readbacks(phase, positive, negative, tmp_path):
    record = json.loads(json.dumps(history_record()).replace('"open"', '"' + positive + '"')
                        .replace('"closed"', '"' + negative + '"')
                        .replace('"close"', '"close"' if positive == 'open' else '"toggle_off"'))
    process = record['process']
    rows = process[phase] if phase.endswith('_truth') else process['events'][0][phase]['truth']
    observed = next(row['value'] for row in rows if row['atom']['predicate'] == positive)
    set_atom(rows, 'cabinet', negative, observed)
    directory = tmp_path / 'process'
    with pytest.raises(ValueError, match='contradictory.*state'):
        score_record(record, process_directory=directory)
    assert not directory.exists()


@pytest.mark.parametrize('case', ['order', 'duplicate_event', 'boolean_event', 'missing_private',
    'forged_history', 'forged_milestones', 'unknown_role', 'unhashable_role', 'multiple_held',
    'duplicate_truth', 'nonboolean_truth', 'after_selection'])
def test_v2_invalid_evidence_rejected_before_writing(tmp_path, case):
    record = history_record()
    process = record['process']
    event = process['events'][0]
    if case == 'order':
        process['events'].reverse()
    elif case == 'duplicate_event':
        process['events'][1]['event_id'] = 1
    elif case == 'boolean_event':
        event['event_id'] = True
    elif case == 'missing_private':
        del process['protected_relations']
    elif case.startswith('forged_'):
        event['before'][case.removeprefix('forged_')] = {'reached': {'stored': 0}}
    elif case in {'unknown_role', 'unhashable_role'}:
        event['before']['truth'][0]['atom']['subject_role'] = 'unknown' if case == 'unknown_role' else []
    elif case == 'multiple_held':
        event['before']['held_roles'] = ['item_a', 'item_b']
    elif case == 'duplicate_truth':
        event['before']['truth'].append(copy.deepcopy(event['before']['truth'][0]))
    elif case == 'nonboolean_truth':
        event['before']['truth'][0]['value'] = 1
    else:
        event['after']['selected_roles'] = []
    with pytest.raises(ValueError):
        score_record(record, process_directory=tmp_path / 'process')
    assert not (tmp_path / 'process').exists()


def test_v2_event_order_is_distinct_from_shared_or_absent_action_index(tmp_path):
    record = history_record()
    for index, event in enumerate(record['process']['events']):
        event['action_index'] = None if index < 2 else 0
    assert assert_native_v2_parity(record, tmp_path)['final']['success']


def test_score_profile_identity_conflict_rejected():
    record = history_record()
    record['profile'] = 'one'
    with pytest.raises(ValueError, match='profile differs'):
        score_record(record, profile='two')


@pytest.mark.parametrize('binding_kind', ['destination', 'layout'])
def test_v2_first_destination_and_first_complete_layout_match_native(binding_kind, tmp_path):
    record = history_record()
    process = record['process']
    program = process['requirements']['history_program']
    exists = dict(op='BINDING_EXISTS', binding_ref='first')
    def atom(role, destination):
        return dict(predicate='ontop', subject_role=role, object_role=destination, desired=True)
    if binding_kind == 'destination':
        program['bindings'][0].update(time='FIRST_VERIFIED_SUCCESS', event=dict(action='place_on_top',
            subject_role='item_a', destination_domain_roles=['tray', 'table'],
            required_public_result='completed', required_relation_readback=True))
        program['expression'] = dict(predicate='ontop', subject_role='item_b',
                                     object_ref={'binding_ref': 'first'}, desired=True)
        program['clauses'] = [atom('item_a', 'tray')]
        process['events'] = process['events'][:2]
        expected = {'first': 'tray'}
    else:
        mapping = dict(op='RELATION_MAP', roles=['item_a', 'item_b'], predicate='ontop',
                       destination_domain_roles=['tray', 'table'])
        program['bindings'] = [dict(binding_id='first', time='FIRST_VERIFIED_PHASE_COMPLETION',
            cardinality='TOTAL_BIJECTION', update='NEVER_AFTER_TRIGGER',
            on_failed_cancelled_or_selection='DO_NOT_BIND', select=mapping,
            trigger=dict(after_action='place_on_top', required_public_result='completed',
                         condition=dict(mapping, op='BIJECTION')))]
        program['expression'] = dict(op='AND', args=[dict(predicate='ontop', subject_role=role,
            object_ref=dict(op='MAP_LOOKUP', map_ref={'binding_ref': 'first'}, key_role=other), desired=True)
            for role, other in [('item_a', 'item_b'), ('item_b', 'item_a')]])
        program['clauses'] = [atom('item_a', 'table')]
        process['events'] = [process['events'][i] for i in [0, 1, 3]]
        expected = {'first': {'item_a': 'tray', 'item_b': 'table'}}
    program['process'] = dict(ordered_milestones=[dict(milestone_id='bound', condition=exists)])
    for i, event in enumerate(process['events'], 1):
        event['event_id'] = i
    process['final_truth'] = copy.deepcopy(process['events'][-1]['after']['truth'])
    result = assert_native_v2_parity(record, tmp_path)
    assert result['final']['success'] is True
    assert result['process']['history_bindings'] == expected


def cycle_record(ordinal='last'):
    def atom(role, predicate):
        return dict(predicate=predicate, subject_role=role, desired=True)
    stages = [dict(milestone_id=f'{role}_{predicate}', after=[] if predicate == 'open' else [f'{role}_open'],
                   requirements=[atom(role, predicate)]) for role in ['a', 'b'] for predicate in ['open', 'closed']]
    program = dict(schema_version='l1-public-process-program/v1', milestones=stages,
        final_requirements=[], final_placement_after=None,
        completion_rule='ALL_DECLARED_MILESTONES_AND_CURRENT_FINAL_REQUIREMENTS',
        condition_evaluation='CANONICAL_READBACK_AFTER_VERIFIED_EXECUTION_OR_INITIAL_READBACK',
        milestone_memory='LATCH_ONCE_AFTER_DECLARED_PREDECESSORS_SATISFIED',
        source_process={}, public_instruction_evidence={'instruction': 'Check both devices; leave the selected cycle device open.'},
        exact_cycles_per_role={'a': 1, 'b': 1}, cycle_order_final=dict(cycle_end_milestones=['a_closed', 'b_closed'],
            active_predicate='open', neutral_predicate='closed', ordinal=ordinal))
    opened = dict(a=False, b=False)
    def truth():
        return [dict(atom=atom(role, predicate), value=value if predicate == 'open' else not value)
                for role, value in opened.items() for predicate in ['open', 'closed']]
    process = dict(roles=['a', 'b'], requirements=dict(no_manipulation_roles=[], before_placement=[],
        process_program=program), initial_truth=truth(), final_truth=[], events=[],
        protected_relations=[], protected_state_relations=[], placement_targets=[])
    selected = 'b' if ordinal == 'last' else 'a'
    for index, (role, value) in enumerate([('a', True), ('a', False), ('b', True), ('b', False), (selected, True)], 1):
        before = dict(held_roles=[], truth=truth())
        opened[role] = value
        process['events'].append(dict(event_id=index, action='open' if value else 'close', status='completed',
            action_index=index, before=before, after=dict(held_roles=[], truth=truth())))
    process['final_truth'] = truth()
    return dict(official(), schema_version='roboweave/score-record-v2',
                snapshots=[{'options': [[True, True, True]]}], process=process)


def test_v1_milestone_readbacks_cannot_hide_an_extra_device_cycle(tmp_path):
    record = cycle_record()
    native_v2(record, tmp_path / 'native')
    events = [json.loads(line) for line in
              (tmp_path / 'native/private/task_process_events.jsonl').read_text().splitlines()]
    for event in events:
        event.pop('event_id')
    record['schema_version'] = 'roboweave/score-record-v1'
    process = record['process']
    process['events'] = events
    for field in ('protected_relations', 'protected_state_relations', 'placement_targets'):
        process.pop(field)
    closed = copy.deepcopy(events[1]['after'])
    opened = copy.deepcopy(closed)
    opened['milestones'].update(a_open=True, a_closed=False)
    extra = [dict(action='open', status='completed', action_index=2, before=closed, after=opened),
             dict(action='close', status='completed', action_index=3,
                  before=copy.deepcopy(opened), after=copy.deepcopy(closed))]
    events[2:2] = extra
    directory = tmp_path / 'conflicting'
    with pytest.raises(ValueError, match='contradictory milestone'):
        score_record(record, process_directory=directory)
    assert not directory.exists()
    extra[0]['after']['milestones']['_unary']['a'] = True
    extra[1]['before']['milestones']['_unary']['a'] = True
    result = score_record(record, process_directory=tmp_path / 'consistent')
    assert result['final']['success'] is False
    assert any(v['kind'] == 'EXACT_CYCLE_LIMIT_EXCEEDED' for v in result['process']['violations'])


@pytest.mark.parametrize('ordinal', ['last', 1])
@pytest.mark.parametrize('valid_final', [True, False])
def test_v2_dynamic_milestone_final_reads_end_truth(ordinal, valid_final, tmp_path):
    record = cycle_record(ordinal)
    if not valid_final:
        for row in record['process']['final_truth']:
            row['value'] = not row['value']
    result = assert_native_v2_parity(record, tmp_path)
    assert result['final']['success'] is valid_final
    assert result['process']['milestones_reached'] == {'a_open': 1, 'a_closed': 2, 'b_open': 3, 'b_closed': 4}


@pytest.mark.parametrize('phase', ['initial_truth', 'final_truth'])
def test_v2_dynamic_milestone_missing_begin_or_end_truth_rejected(phase, tmp_path):
    record = cycle_record()
    record['process'][phase] = []
    with pytest.raises(ValueError, match='predicate truth'):
        score_record(record, process_directory=tmp_path / 'process')


def test_v2_private_protection_recipes_use_original_snapshot(tmp_path):
    record = history_record()
    process = record['process']
    protected = dict(predicate='open', subject_role='cabinet', desired=True)
    process['requirements']['state_protection_roles'] = ['cabinet']
    process['protected_state_relations'] = [protected]
    result = assert_native_v2_parity(record, tmp_path)
    assert not result['final']['success']
    assert any(v['kind'] == 'PROTECTED_STATE_CHANGED' for v in result['process']['violations'])
    process['protected_state_relations'] = []
    with pytest.raises(ValueError, match='protected state identity'):
        score_record(record, process_directory=tmp_path / 'missing-protection')


def test_public_score_does_not_reorder_caller_import_paths():
    import sys
    original = list(sys.path)
    result = score_record(official())
    assert result['final']['q_score'] == pytest.approx(1 / 3)
    assert sys.path == original


@pytest.mark.parametrize('profile', ['l1-m30-public-v2', 'l2-collection37'])
def test_v2_history_uses_explicit_frozen_profile(profile, tmp_path):
    record = history_record()
    record['profile'] = profile
    assert score_record(record, process_directory=tmp_path / 'process')['final']['success']


@pytest.mark.parametrize('profile', ['l1-p-inside5', 'l1-p-ontop5'])
def test_v2_does_not_substitute_new_history_features_into_older_profiles(profile, tmp_path):
    record = history_record()
    with pytest.raises(ValueError, match='history'):
        score_record(record, profile=profile, process_directory=tmp_path / 'process')
    assert not (tmp_path / 'process').exists()
