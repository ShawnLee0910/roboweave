from types import SimpleNamespace as NS
from pathlib import Path
import json

import pytest
from behavior_hl.embodied.evaluator_bridge.l1_process_audit import ProcessAudit, validate_process_requirements, ACTIVE, run_process_checked_phase


def requirements(protection=False):
    return {'no_manipulation_roles':['can'] if protection else [],'before_placement':[] if protection else [
        {'subject_role':'tool','required_relations':[{'predicate':'closed','subject_role':'cab','desired':True}]}]}


def audit(tmp_path, protection=False):
    return ProcessAudit(requirements(protection),role_object=lambda r:r,recipe_truth=lambda *a:True,
                        protected_relations=[],directory=tmp_path)


def state(held=(),ready=True,goal=True,protected=True):
    return {'held_roles':list(held),'prerequisites':[ready],'placement_goals':[goal], 'protected_relations':[protected]}


def test_public_order_accepts_legal_placement_and_rejects_early_one(tmp_path):
    good=audit(tmp_path/'good');bad=audit(tmp_path/'bad')
    good.observe(action='place_on_top',before=state(['tool'],True),after=state(),status='completed',action_index=1)
    bad.observe(action='place_on_top',before=state(['tool'],False),after=state(),status='completed',action_index=1)
    assert good.report()['public_process_satisfied']
    assert not bad.report()['public_process_satisfied']
    bad.observe(action='place_on_top',before=state(['tool'],True),after=state(),status='completed',action_index=2)
    assert not bad.report()['public_process_satisfied']  # later repair cannot undo an explicit prior order violation


def test_marker_failure_and_temporary_other_placement_do_not_claim_final_placement(tmp_path):
    a=audit(tmp_path)
    a.observe(action='place_on_top',before=state(['tool'],False),after=state(['tool'],False),status='needs_marker',action_index=1)
    a.observe(action='place_on_top',before=state(['tool'],False),after=state(goal=False),status='completed',action_index=2)
    assert a.report()['violations']==[] and not a.report()['public_process_satisfied']
    a.observe(action='place_on_top',before=state(['tool'],True),after=state(),status='completed',action_index=3)
    assert a.report()['public_process_satisfied']


def test_protected_pick_or_failed_move_remains_violation(tmp_path):
    a=audit(tmp_path,True)
    a.observe(action='pick',before=state(),after=state(['can']),status='completed',action_index=1)
    a.observe(action='place_inside',before=state(['can']),after=state(),status='completed',action_index=2)
    assert not a.report()['public_process_satisfied']
    b=audit(tmp_path/'failed',True)
    b.observe(action='pick',before=state(),after=state(protected=False),status='blocked',action_index=1)
    assert not b.report()['public_process_satisfied']


def test_failed_release_is_unknown_not_completed(tmp_path):
    a=audit(tmp_path)
    a.observe(action='place_on_top',before=state(['tool']),after=state(),status='blocked',action_index=1)
    assert not a.report()['operational']


def test_episode_instances_do_not_share_violation_state(tmp_path):
    one=audit(tmp_path/'one',True);two=audit(tmp_path/'two',True)
    one.observe(action='pick',before=state(),after=state(['can']),status='completed',action_index=1)
    two.observe(action='open',before=state(),after=state(),status='completed',action_index=1)
    assert not one.report()['public_process_satisfied'] and two.report()['public_process_satisfied']


@pytest.mark.parametrize('mutate',[lambda c:c.update(unknown=True),lambda c:c['no_manipulation_roles'].append('missing')])
def test_unknown_process_semantics_rejected(mutate):
    c=requirements(True);mutate(c)
    with pytest.raises(ValueError):validate_process_requirements(c,{'can':'scope'})


def test_real_action_wrapper_and_final_score_keep_raw_canonical_result(tmp_path,monkeypatch):
    from behavior_hl.embodied.evaluator_bridge import registered_l0_task_runtime as registered
    from behavior_hl.embodied.evaluator_bridge import run_b1_marker_symbolic_rollout as rollout
    from behavior_hl.embodied.runtime.dispatch import ActionPluginResult
    from behavior_hl.embodied.evaluator_bridge.l1_initial_branch import validate_initial_branch
    can=object();held={'object':None}
    runtime=NS(symbolic_primitives=NS(_get_obj_in_hand=lambda:held['object']))
    monkeypatch.setattr(registered,'_role_object_for_contract',lambda *a:can)
    monkeypatch.setattr(registered,'_initial_recipe_truth',lambda *a:True)
    req=requirements(True)
    branch=dict(schema_version='l1-initial-branch/v2', axis='A', proposal_id='protected-static-goal',
                world_branch_id='one', pair_member='challenge',
                initial_assertions=[dict(predicate='ontop', subject_role='can', object_role='support', desired=True)],
                process_requirements=req, initial_reference_goals=[])
    validate_initial_branch(branch, {'can':'scope', 'support':'surface'})
    task=NS(record={'task_contract':branch},
            template_contract=lambda:{'role_to_scope':{'can':'scope','support':'surface'},'initial_state_recipe':branch['initial_assertions']})
    def core(**kw):
        held['object']=can
        return ActionPluginResult('completed'),(),0
    monkeypatch.setattr(rollout,'_execute_core',core)
    def phase(**kw):
        rollout._execute(runtime=runtime,action=NS(action='pick'),action_index=1,phase_dir=tmp_path)
        ACTIVE.get().frame({'path':'existing.png','png_sha256':'existing-hash'})
        return {'status':'PASS','official_q':1.,'official_success':True}
    result=run_process_checked_phase(phase,task=task,evaluator=None,run_dir=tmp_path,phase_name='oracle')
    assert result['official_q']==1 and result['official_success'] is True
    assert result['status']=='FAIL_PUBLIC_PROCESS' and result['task_success'] is False
    assert result['canonical_execution_status']=='PASS'
    assert ACTIVE.get() is None
    evidence=json.loads((tmp_path/'strategies/oracle/private/task_evidence_index.jsonl').read_text())
    assert evidence['path']=='existing.png'
    assert result['public_process_report']['extra_render_calls']==0


def test_ordinary_oracle_failure_is_not_relabelled_process_failure(tmp_path):
    task=NS(record={'task_contract':{'process_requirements':requirements(True),'initial_reference_goals':[]}},
            template_contract=lambda:{'role_to_scope':{'can':'scope'},'initial_state_recipe':[]})
    result=run_process_checked_phase(lambda **kw:{'status':'FAIL_MARKER','official_q':None,'failure_class':'MARKER_ERROR'},
                                    task=task,evaluator=None,run_dir=tmp_path,phase_name='oracle')
    assert result['failure_class']=='MARKER_ERROR'


def test_committed_bundle_pick_preserves_protection_and_failed_probe_is_unknown(tmp_path):
    a=audit(tmp_path/'a',True);before=state();before['selected_roles']=['can']
    a.observe(action='pick',before=before,after=state(),status='completed',action_index=1)
    assert not a.report()['public_process_satisfied']
    b=audit(tmp_path/'b',True)
    b.observe(action='pick',before=before,after=state(),status='blocked',action_index=1)
    assert not b.report()['operational']


def test_current_eight_package_requires_unchanged_process_contract():
    from copy import deepcopy
    from behavior_hl.embodied.evaluator_bridge.benchmark_package_correspondence import validate_package_correspondence
    root=Path(__file__).parent / 'restricted_fixtures/am_process_smoke'
    if not root.is_dir(): pytest.skip('restricted eight-task package is not distributed')
    specs=[json.loads(x) for x in (root/'task_specs.jsonl').read_text().splitlines()]
    assert len(specs)==8
    validate_package_correspondence(root/'package.json',specs,root/'runtime_registration/catalog.json')
    changed=deepcopy(specs);changed[0]['task_contract']['process_requirements']['no_manipulation_roles']=[]
    with pytest.raises(ValueError):
        validate_package_correspondence(root/'package.json',changed,root/'runtime_registration/catalog.json')


@pytest.mark.parametrize('predicate,state_name,action', [
    ('next_to', 'NextTo', 'place_next_to'), ('under', 'Under', 'place_under')])
def test_binary_reference_uses_live_reader_and_process_action_hook(tmp_path, monkeypatch, predicate, state_name, action):
    from behavior_hl.embodied.evaluator_bridge import registered_l0_task_runtime as registered
    from behavior_hl.embodied.evaluator_bridge.l1_initial_branch import validate_initial_branch
    from behavior_hl.embodied.evaluator_bridge.l1_process_audit import observe_execution
    from behavior_hl.embodied.runtime.dispatch import ActionPluginResult

    objects = {r: object() for r in ('source', 'tool', 'support', 'prerequisite')}
    relations = {'ready': False, 'placed': False}
    held = {'object': objects['tool']}
    reads = []
    def read(subject, kind, reference, *, predicate_mode):
        reads.append((subject, kind, reference, predicate_mode))
        assert kind == state_name and reference is objects['support']
        return relations['placed'] if subject is objects['tool'] else relations['ready']
    monkeypatch.setattr(registered, 'read_action_relation', read)
    monkeypatch.setattr(registered, '_role_object_for_contract', lambda evaluator, contract, role: objects[role])
    initial = dict(predicate='ontop', subject_role='source', object_role='support', desired=True)
    req = dict(no_manipulation_roles=[], before_placement=[dict(subject_role='tool', required_relations=[
        dict(predicate=predicate, subject_role='prerequisite', object_role='support', desired=True)])])
    branch = dict(schema_version='l1-initial-branch/v2', axis='M', proposal_id='generic',
        world_branch_id='first', pair_member='challenge', initial_assertions=[initial],
        process_requirements=req, initial_reference_goals=[dict(initial_assertion_index=0,
            goal_subject_role='tool', goal_predicate=predicate)])
    validate_initial_branch(branch, objects)
    template = dict(role_to_scope={r:r for r in objects}, initial_state_recipe=[initial])
    task = NS(record={'task_contract':branch}, template_contract=lambda:template)
    runtime = NS(symbolic_primitives=NS(_get_obj_in_hand=lambda:held['object']))
    def core(**kw):
        held['object'] = None
        relations['placed'] = True
        return ActionPluginResult('completed'), (), 0
    def phase(**kw):
        assert ACTIVE.get().placement_targets == [dict(predicate=predicate, subject_role='tool', object_role='support', desired=True)]
        observe_execution(core, runtime=runtime, action=NS(action=action), action_index=1)
        # Restoring the physical world cannot erase a public order violation.
        relations['ready'] = True
        held['object'] = objects['tool']
        relations['placed'] = False
        observe_execution(core, runtime=runtime, action=NS(action=action), action_index=2)
        return dict(status='PASS', official_q=1.)
    result = run_process_checked_phase(phase, task=task, evaluator=None, run_dir=tmp_path, phase_name='oracle')
    assert result['official_q'] == 1. and result['task_success'] is False
    assert result['status'] == 'FAIL_PUBLIC_PROCESS'
    assert result['public_process_report']['execution_events'] == 2
    assert result['public_process_report']['violations'][0]['kind'] == 'PLACEMENT_BEFORE_PUBLIC_PREREQUISITES'
    assert all(call[3] == 'action20_symbolic' for call in reads)
    assert ACTIVE.get() is None


@pytest.mark.parametrize('predicate', ['under', 'next_to'])
def test_binary_prerequisite_requires_bound_reference(predicate):
    req = requirements()
    req['before_placement'][0]['required_relations'] = [dict(predicate=predicate, subject_role='cab', desired=True)]
    with pytest.raises(ValueError, match='unbound process reference'):
        validate_process_requirements(req, {'tool':'t', 'cab':'c'})


@pytest.mark.parametrize('change_state', [False, True, 'unknown'])
def test_state_only_protection_allows_movement_but_latches_changes(tmp_path, monkeypatch, change_state):
    from behavior_hl.embodied.evaluator_bridge import registered_l0_task_runtime as registered
    from behavior_hl.embodied.evaluator_bridge.l1_initial_branch import validate_initial_branch
    from behavior_hl.embodied.evaluator_bridge.l1_process_audit import observe_execution
    from behavior_hl.embodied.runtime.dispatch import ActionPluginResult
    device = object()
    world = {'on': True, 'held': None}
    def read():
        if world['on'] == 'unknown':
            raise RuntimeError('state unavailable')
        return world['on']
    monkeypatch.setattr(registered, '_role_object_for_contract', lambda *args: device)
    monkeypatch.setattr(registered, 'state_by_name', lambda obj, kind: NS(get_value=read))
    initial = dict(predicate='toggled_on', subject_role='device', desired=True)
    req = dict(no_manipulation_roles=[], before_placement=[], state_protection_roles=['device'])
    branch = dict(schema_version='l1-initial-branch/v2', axis='A', proposal_id='state', world_branch_id='on',
                  pair_member='challenge', initial_assertions=[initial], initial_reference_goals=[], process_requirements=req)
    validate_initial_branch(branch, {'device':'d'})
    task = NS(record={'task_contract':branch}, template_contract=lambda:dict(
        role_to_scope={'device':'d'}, initial_state_recipe=[initial]))
    runtime = NS(symbolic_primitives=NS(_get_obj_in_hand=lambda:world['held']))
    def core(**kwargs):
        world['held'] = device
        world['on'] = 'unknown' if change_state == 'unknown' else not change_state
        return ActionPluginResult('completed'), (), 0
    def phase(**kwargs):
        observe_execution(core, runtime=runtime, action=NS(action='pick'), action_index=1)
        world['on'] = True
        return dict(status='PASS', official_q=1.)
    result = run_process_checked_phase(phase, task=task, evaluator=None, run_dir=tmp_path, phase_name='oracle')
    assert result['official_q'] == 1.
    assert result['task_success'] is (change_state is False)
    if change_state is True:
        assert result['public_process_report']['violations'][0]['kind'] == 'PROTECTED_STATE_CHANGED'
    if change_state == 'unknown':
        assert result['status'] == 'FAIL_PROCESS_VERIFIER'
    branch['initial_assertions'] = [dict(predicate='ontop', subject_role='device', object_role='support', desired=True)]
    with pytest.raises(ValueError, match='declared initial device states'):
        validate_initial_branch(branch, {'device':'d','support':'s'})


def test_factory_is_opt_in_and_episode_local(tmp_path):
    from behavior_hl.embodied.evaluator_bridge.l1_process_audit import create_process_audit
    for task in (None, NS(record={}), NS(record={'task_contract': {}})):
        assert create_process_audit(task=task, evaluator=None, directory=tmp_path) is None
    task = NS(record={'task_contract': {'process_requirements':
              {'no_manipulation_roles': ['can'], 'before_placement': []}}},
              template_contract=lambda: {'role_to_scope': {'can': 'can.n.01_1'}, 'initial_state_recipe': []})
    first = create_process_audit(task=task, evaluator=None, directory=tmp_path/'first')
    second = create_process_audit(task=task, evaluator=None, directory=tmp_path/'second')
    first.violations.append({'kind': 'test'})
    assert second.violations == []
    assert first.directory != second.directory
    assert ACTIVE.get() is None
