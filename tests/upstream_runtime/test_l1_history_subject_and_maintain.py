"""New memory contracts exercise the existing snapshot/action hook, not a second runner."""
from itertools import permutations
from types import SimpleNamespace as NS

import pytest

from behavior_hl.embodied.evaluator_bridge.l1_history_bindings import SCHEMA, validate_history_program
from behavior_hl.embodied.evaluator_bridge.l1_process_audit import ProcessAudit


ROLES = {'calculator', 'stapler', 'battery', 'tray', 'desktop', 'cabinet'}
ITEMS = ['calculator', 'stapler', 'battery']


def atom(role, dest=None, predicate='ontop'):
    return dict(predicate=predicate, subject_role=role, desired=True, **({'object_role':dest} if dest else {}))


def subject_program():
    first = dict(predicate='ontop', subject_ref={'binding_ref':'first'}, object_role='desktop', desired=True)
    return dict(schema_version=SCHEMA, bindings=[dict(binding_id='first', time='FIRST_VERIFIED_SUBJECT_SUCCESS',
        cardinality='EXACTLY_ONE', update='NEVER', on_failed_cancelled_or_selection='DO_NOT_BIND',
        event=dict(action='place_on_top', subject_domain_roles=ITEMS, object_role='tray',
                   required_public_result='completed', required_relation_readback=True))], clauses=[],
        expression=dict(op='AND', args=[first, dict(op='FOR_EACH_EXCEPT_BINDING', roles=ITEMS,
            binding_ref='first', predicate='ontop', object_role='tray')]),
        process=dict(ordered_milestones=[dict(milestone_id='all_stored', condition=dict(op='AND',
            args=[*[atom(r,'tray') for r in ITEMS],dict(op='BINDING_EXISTS',binding_ref='first')]))],
            before_placement=[dict(subject_ref={'binding_ref':'first'}, placement_scope=first, required_milestone='all_stored')]))


def maintained_program(predicate='open'):
    final = 'closed' if predicate == 'open' else 'toggled_off'
    return dict(schema_version=SCHEMA, bindings=[], clauses=[atom('cabinet',predicate=final)], expression=dict(op='AND',
        args=[*[atom(r,'tray') for r in ITEMS],atom('cabinet',predicate=final)]),
        process=dict(ordered_milestones=[dict(milestone_id='all_stored',condition=dict(op='AND',args=[atom(r,'tray') for r in ITEMS]))],
            maintain_until=[dict(condition=atom('cabinet',predicate=predicate), required_milestone='all_stored')]))


class World:
    def __init__(self, program, tmp_path):
        validate_history_program(program, ROLES)
        self.locations = dict.fromkeys(ITEMS, 'desktop')
        self.active = True
        self.held = None
        self.objects = {r:object() for r in ROLES}
        self.runtime = NS(symbolic_primitives=NS(_get_obj_in_hand=lambda:self.objects.get(self.held)))
        self.requirements = dict(no_manipulation_roles=[],before_placement=[],history_program=program)
        self.audit = ProcessAudit(self.requirements, role_object=self.objects.__getitem__,
            recipe_truth=lambda a,r:self.truth(a), protected_relations=[], directory=tmp_path)
        self.history = self.audit.history

    def truth(self,a):
        if a['predicate'] in {'open','toggled_on'}: return self.active
        if a['predicate'] in {'closed','toggled_off'}: return not self.active
        return self.locations.get(a['subject_role']) == a['object_role'] and self.held != a['subject_role']

    def event(self, action, effect, status='completed'):
        before = self.audit.snapshot(self.runtime)
        effect()
        self.audit.observe(action=action,before=before,after=self.audit.snapshot(self.runtime),status=status,action_index=self.audit.events+1)

    def place(self,role,dest,status='completed',release=True):
        self.held = role
        def effect():
            if release: self.locations[role] = dest; self.held = None
        self.event('place_on_top',effect,status)

    def close(self,status='completed'):
        self.event('close',lambda:setattr(self,'active',False),status)

    def execute(self,a):
        if a['predicate'] in {'closed','toggled_off'}: self.close()
        else: self.place(a['subject_role'],a['object_role'])


@pytest.mark.parametrize('order',list(permutations(ITEMS)))
def test_actual_first_subject_all_orders_and_oracle_continuation(order,tmp_path):
    w=World(subject_program(),tmp_path)
    w.place(order[0],'tray')
    assert w.history.bindings == {'first':order[0]}
    w.history.execute_pending(w.execute)
    assert w.audit.report()['public_process_satisfied']
    assert w.locations == {r:('desktop' if r==order[0] else 'tray') for r in ITEMS}
    w.locations[order[1]]='desktop'
    assert not w.history.complete


@pytest.mark.parametrize('status',['blocked','uncertain','needs_marker','awaiting_selection','cancelled'])
def test_failed_and_marker_subject_placement_never_binds(status,tmp_path):
    w=World(subject_program(),tmp_path);w.place('battery','tray',status)
    assert not w.history.bindings


def test_no_release_no_relation_and_premature_return(tmp_path):
    w=World(subject_program(),tmp_path);w.place('battery','tray',release=False)
    assert not w.history.bindings
    w.place('battery','desktop');assert not w.history.bindings
    w.place('battery','tray');w.place('battery','desktop')
    assert w.history.violations
    for r in ITEMS:w.place(r,'tray')
    w.place('battery','desktop')
    assert not w.history.complete


@pytest.mark.parametrize('predicate',['open','toggled_on'])
def test_maintained_state_until_successful_stage_then_final_close(predicate,tmp_path):
    w=World(maintained_program(predicate),tmp_path)
    w.history.execute_pending(w.execute)
    assert w.audit.report()['public_process_satisfied']
    assert not w.history.violations


@pytest.mark.parametrize('status',['completed','blocked','needs_marker'])
def test_early_close_reopen_never_erases_violation(status,tmp_path):
    w=World(maintained_program(),tmp_path)
    w.close(status)
    w.event('open',lambda:setattr(w,'active',True))
    for r in ITEMS:w.place(r,'tray')
    w.close()
    assert not w.audit.report()['public_process_satisfied']
    assert w.history.violations[0]['kind']=='HISTORY_STATE_NOT_MAINTAINED'


def test_failed_final_placement_does_not_release_maintain_guard(tmp_path):
    w=World(maintained_program(),tmp_path)
    for r in ITEMS[:-1]:w.place(r,'tray')
    w.place(ITEMS[-1],'tray','uncertain')
    assert not w.history.reached
    w.close()
    assert w.history.violations


def test_new_contract_capabilities_and_marker_without_state_change(tmp_path):
    from behavior_hl.embodied.evaluator_bridge.l1_history_bindings import required_history_capabilities
    assert required_history_capabilities(subject_program())=={'L1_HISTORY_BINDINGS_V1','L1_SUBJECT_HISTORY_BINDING_V1'}
    assert required_history_capabilities(maintained_program())=={'L1_HISTORY_BINDINGS_V1','L1_MAINTAIN_UNTIL_V1'}
    w=World(maintained_program(),tmp_path)
    w.event('close',lambda:None,'needs_marker')
    assert not w.history.reached and not w.history.violations
    p=maintained_program();p['process']['maintain_until'][0]['condition']=[]
    with pytest.raises(ValueError,match='maintained state'):validate_history_program(p,ROLES)


def test_real_rollout_hook_uses_subject_release_and_canonical_snapshot(monkeypatch,tmp_path):
    from behavior_hl.embodied.evaluator_bridge import l1_process_audit as audit_module
    from behavior_hl.embodied.evaluator_bridge import run_b1_marker_symbolic_rollout as rollout
    from behavior_hl.embodied.runtime.dispatch import ActionPluginResult
    w=World(subject_program(),tmp_path)
    def core(**kw):
        args=kw['action'].args
        w.locations[args['role']]=args['destination'];w.held=None
        return ActionPluginResult('completed'),(),0
    monkeypatch.setattr(rollout,'_execute_core',core)
    token=audit_module.ACTIVE.set(w.audit)
    try:
        for i,(role,dest) in enumerate([('battery','tray'),('calculator','tray'),('stapler','tray'),('battery','desktop')],1):
            w.held=role
            rollout._execute(runtime=w.runtime,action=NS(action='place_on_top',args=dict(role=role,destination=dest)),action_index=i,phase_dir=tmp_path)
        assert w.audit.report()['public_process_satisfied']
        assert w.history.bindings=={'first':'battery'}
    finally:audit_module.ACTIVE.reset(token)


def test_real_full_oracle_phase_plans_dynamic_subject_with_existing_budget(monkeypatch,tmp_path):
    from test_public_oracle_goal_and_marker_recovery import _run
    p=subject_program()
    p['bindings'][0]['event']['subject_domain_roles']=ITEMS[:2]
    p['expression']['args'][1]['roles']=ITEMS[:2]
    p['process']['ordered_milestones'][0]['condition']['args']=[atom(r,'tray') for r in ITEMS[:2]]+[
        dict(op='BINDING_EXISTS',binding_ref='first')]
    def rename(v):
        if isinstance(v,dict):return {k:rename(x) for k,x in v.items()}
        if isinstance(v,list):return [rename(x) for x in v]
        return {'calculator':'item_0','stapler':'item_1','tray':'left','desktop':'support'}.get(v,v)
    result=_run(monkeypatch,tmp_path,initial=(True,),history_program=rename(p))
    assert result.result['status']=='PASS',result.result.get('error')
    assert result.result['public_process_report']['history_bindings']=={'first':'item_0'}
    assert result.state.history_locations=={'item_0':'support','item_1':'left'}
    assert len({id(nav['navigation_work_used']) for nav in result.navigation})==1


def test_real_full_formula_accepts_actual_subject_not_fixed_witness(tmp_path):
    from behavior_hl.embodied.evaluator_bridge.bounded_full_goal_acceptance import BoundedFullGoalAcceptance, bounded_dnf
    from behavior_hl.embodied.evaluator_bridge.official_predicate_verifier import OfficialPredicateVerifier
    from behavior_hl.embodied.evaluator_bridge.benchmark_package_correspondence import _formula
    from behavior_hl.embodied.evaluator_bridge.l1_dynamic_process_goals import dynamic_goal_formula
    from behavior_hl.embodied.evaluator_bridge.l1_history_bindings import HistoryCompletionVerifier
    w=World(subject_program(),tmp_path)
    class Predicate:
        def __init__(self,body):self.body=list(body)
        def evaluate(self,evaluator):return w.truth(atom(self.body[1],self.body[2],self.body[0]))
    options=[[Predicate(body) for body in sorted(branch)] for branch in bounded_dnf(_formula(dynamic_goal_formula(history=subject_program())))]
    task=NS(ground_goal_state_options=options,_evaluate_predicate=lambda:None)
    evaluator=NS(env=NS(task=task))
    selected=next(i for i,option in enumerate(options) if any(p.body==['ontop','calculator','desktop'] for p in option))
    verifier=HistoryCompletionVerifier(BoundedFullGoalAcceptance(OfficialPredicateVerifier(selected_option_index=selected),predicate_mode='official'),w.history)
    assert not verifier.evaluate(None,evaluator).official_success
    for role in ['battery','stapler','calculator']:w.place(role,'tray')
    w.place('battery','desktop')
    final=verifier.evaluate(None,evaluator)
    assert final.official_success and final.official_q==1.
    assert not final.research_diagnostics['selected_witness_private']['official_success']
    # Another legal physical branch must NOT substitute for the actual first subject.
    w.locations['battery']='tray';w.locations['calculator']='desktop'
    wrong=verifier.evaluate(None,evaluator)
    assert wrong.official_q==1. and not wrong.official_success
