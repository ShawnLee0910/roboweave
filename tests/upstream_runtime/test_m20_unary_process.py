from itertools import permutations

import pytest

from behavior_hl.embodied.evaluator_bridge.l1_am_process_programs import (
    MilestoneProgress, lower_process_program, validate_runtime_program,
)


def atom(predicate, role):
    return dict(predicate=predicate, subject_role=role)


def build(process, roles, final=None):
    p = lower_process_program(process, final_goals=[] if final is None else final,
        roles=roles, public_instructions={'challenge':'Complete the declared checks and final action.'})['process_program']
    validate_runtime_program(p, roles)
    return p


def ledger(roles, count=1, ordinal=None):
    stages=[]
    for role in roles:
        old=[]
        for i in range(2*count):
            name=f'{role}_{i}'
            stages.append(dict(milestone_id=name, after=old, condition=atom('open' if i%2==0 else 'closed',role)))
            old=[name]
    process=dict(ordered_milestones=stages, exact_cycles_per_role={r:count for r in roles})
    if ordinal is not None:
        process['cycle_order_final']=dict(cycle_end_milestones=[f'{r}_{2*count-1}' for r in roles],
            ordinal=ordinal,active_predicate='open',neutral_predicate='closed')
    return build(process,roles,None if ordinal is not None else [atom('closed',r) for r in roles])


class World:
    def __init__(self,p,states):
        self.states=states
        self.progress=MilestoneProgress(p,lambda a: self.states[a['subject_role']] == (a['predicate'] in {'open','toggled_on'}))
        self.event=0
        self.progress.observe(self.progress.read(),0)

    def do(self, action, role, status='completed', mutate=True):
        before=self.progress.read()
        if status=='completed' and mutate:
            self.states[role]=action in {'open','toggle_on'}
        self.event+=1
        if status=='completed':
            self.progress.observe(self.progress.read(),self.event,before,action)


@pytest.mark.parametrize('order', [('a','b'),('b','a')])
def test_global_barriers_allow_both_orders_but_reject_early_open(order):
    p=build(dict(global_unary_phases=True,ordered_milestones=[
        dict(milestone_id='all_closed',after=[],condition={'op':'AND','args':[atom('closed',r) for r in order]}),
        dict(milestone_id='all_open',after=['all_closed'],condition={'op':'AND','args':[atom('open',r) for r in order]})]),
        order,[atom('open','a'),atom('closed','b')])
    w=World(p,{'a':True,'b':False})
    w.do('open','b')
    assert w.progress.violations[0]['kind']=='PREMATURE_UNARY_PHASE'
    w=World(p,{'a':True,'b':False})
    w.do('close','a',status='blocked')
    w.do('close','a',mutate=False)
    assert not w.progress.reached
    w.do('close','a')
    assert set(w.progress.reached)=={'all_closed'}
    for role in order:w.do('open',role)
    assert w.progress.complete
    w.do('close','b')
    assert w.progress.complete and not w.progress.violations
    w=World(p,{'a':False,'b':False})
    assert w.progress.reached=={'all_closed':0}
    assert not w.progress.complete
    w.progress.execute_pending(w.do)
    assert w.progress.complete


def test_exact_three_real_cycles_ignore_failed_marker_noop_and_replay():
    w=World(ledger(['a'],3),{'a':False})
    w.do('open','a',status='needs_marker')
    w.do('open','a',mutate=False)
    for i in range(3):
        w.do('open','a');w.do('open','a')
        assert not w.progress.complete
        w.do('close','a')
        assert w.progress.complete == (i==2)
    assert w.progress.cycles=={'a':3}
    before=w.progress.read()
    w.progress.observe(before,w.event,before,'close')
    assert w.progress.cycles=={'a':3}
    w.do('open','a');w.do('close','a')
    assert not w.progress.complete
    assert w.progress.violations[-1]['kind']=='EXACT_CYCLE_LIMIT_EXCEEDED'


def test_overrun_before_other_device_done_and_free_interleaving():
    w=World(ledger(['a','b']),{'a':False,'b':False})
    for action in ['open','close','open','close']:w.do(action,'a')
    w.do('open','b');w.do('close','b')
    assert not w.progress.complete
    for order in permutations([('open','a'),('close','a'),('open','b'),('close','b')]):
        if any(order.index(('close',r))<order.index(('open',r)) for r in ['a','b']):continue
        w=World(ledger(['a','b']),{'a':False,'b':False})
        for action,role in order:w.do(action,role)
        assert w.progress.complete


@pytest.mark.parametrize('ordinal', [2,'last'])
def test_order_selected_from_real_completions_and_final_reopen_is_not_cycle(ordinal):
    for order in permutations(['a','b','c']):
        w=World(ledger(['a','b','c'],ordinal=ordinal),dict.fromkeys(order,False))
        for role in order:w.do('open',role);w.do('close',role)
        assert not w.progress.complete
        selected=order[-1 if ordinal=='last' else 1]
        assert next(a['subject_role'] for a in w.progress.dynamic_final_atoms() if a['predicate']=='open')==selected
        w.progress.execute_pending(w.do)
        assert w.progress.complete and w.states[selected]
        assert w.progress.cycles==dict.fromkeys(order,1)


def test_bad_extension_contracts_fail_closed():
    p=ledger(['a','b'])
    p['exact_cycles_per_role']['a']=2
    with pytest.raises(ValueError,match='counts'):validate_runtime_program(p,{'a','b'})
    p=ledger(['a','b'],ordinal='last')
    p['cycle_order_final']['ordinal']=3
    with pytest.raises(ValueError,match='ordinal'):validate_runtime_program(p,{'a','b'})


def test_one_action_cannot_claim_two_device_transitions():
    w=World(ledger(['a','b']),{'a':False,'b':False})
    before=w.progress.read()
    w.states.update(a=True,b=True)
    w.progress.observe(w.progress.read(),1,before,'open')
    assert not w.progress.reached
    assert w.progress.violations[0]['kind']=='AMBIGUOUS_MULTI_DEVICE_TRANSITION'


def test_second_device_cannot_reopen_before_all_checks_complete():
    w=World(ledger(['a','b','c'],ordinal=2),dict.fromkeys(['a','b','c'],False))
    for role in ['a','b']:w.do('open',role);w.do('close',role)
    w.do('open','b')
    w.do('open','c');w.do('close','c')
    assert w.states=={'a':False,'b':True,'c':False}
    assert not w.progress.complete
    assert w.progress.violations[0]['kind']=='PREMATURE_CYCLE_FINAL'


def test_full_or_acceptance_does_not_reexecute_a_different_fixed_witness():
    from types import SimpleNamespace as NS
    from behavior_hl.embodied.runtime.verification import PrivatePredicateSnapshot
    from behavior_hl.embodied.evaluator_bridge.bounded_full_goal_acceptance import BoundedFullGoalAcceptance
    from behavior_hl.embodied.evaluator_bridge.l1_am_process_programs import ProcessCompletionVerifier
    from behavior_hl.embodied.evaluator_bridge.public_oracle_goal_execution import execute_live_goal_frontier
    w=World(ledger(['a','b','c'],ordinal=2),dict.fromkeys(['a','b','c'],False))
    for role in ['c','a','b']:w.do('open',role);w.do('close',role)
    w.progress.execute_pending(w.do)
    assert w.states=={'a':True,'b':False,'c':False}
    def snapshot(ok):
        return PrivatePredicateSnapshot(ok=True,official_success=ok,official_q=1. if ok else 0.,leaf_truth={'fixed_b_open':w.states['b']})
    task=NS(ground_goal_state_options=[[object()] for _ in range(3)])
    full=BoundedFullGoalAcceptance.__new__(BoundedFullGoalAcceptance)
    full.task=None
    full.selected=NS(evaluate=lambda *args:snapshot(w.states['b']))
    full.complete=NS(_resolve_task=lambda evaluator:task,evaluate=lambda *args:snapshot(sum(w.states.values())==1))
    verifier=ProcessCompletionVerifier(full,w.progress)
    def forbidden(*args):raise AssertionError('fixed witness must not run after a legal dynamic answer')
    final=execute_live_goal_frontier(read_snapshot=lambda:verifier.evaluate(None,None),
        compile_plan=forbidden,leaf_ids_for_plan=forbidden,execute_node=forbidden,
        locally_rejected=RuntimeError,hand_is_empty=lambda:True,record=forbidden)
    assert final.official_success and final.official_q==1.
    assert final.leaf_truth=={'fixed_b_open':False}
