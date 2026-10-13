from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json
import pytest

from behavior_hl.embodied.evaluator_bridge.process_goal_lifecycle import install_process_goal_lifecycle
from behavior_hl.embodied.evaluator_bridge.r1pro_b1_live_entrypoint import StrictEvaluatorStepOnce
from behavior_hl.embodied.evaluator_bridge.l1_am_process_programs import MilestoneProgress, ProcessCompletionVerifier


class Task:
    activity_name = 'process'
    def _create_termination_conditions(self):
        return {'predicate': SimpleNamespace(_check_goal_fn=lambda:(True,{'satisfied':[0]})), 'timeout':object()}


def test_native_static_goal_deferred_across_reset_and_task_switch_but_timeout_not_hidden():
    original=Task._create_termination_conditions
    cleanup=install_process_goal_lifecycle(Task,{'process'})
    try:
        task=Task();conditions=task._create_termination_conditions()
        predicate=conditions['predicate'];timeout=conditions['timeout']
        evaluator=SimpleNamespace(step=lambda:(predicate._check_goal_fn()[0],False))
        stepper=StrictEvaluatorStepOnce()
        for phase in ['load','reset','restore','prepare','action']:
            assert stepper(evaluator,phase)==1
        assert predicate._check_goal_fn()[1]=={'satisfied':[0]}
        assert conditions['timeout'] is timeout
        task.activity_name='ordinary'
        assert predicate._check_goal_fn()[0] is True
        task.activity_name='process'
        assert predicate._check_goal_fn()[0] is False
        evaluator.step=lambda:(False,True)
        with pytest.raises(RuntimeError,match='terminated'):stepper(evaluator,'real_timeout')
        with pytest.raises(RuntimeError,match='poisoned'):stepper(evaluator,'cannot_continue')
    finally:cleanup()
    assert Task._create_termination_conditions is original
    assert Task()._create_termination_conditions()['predicate']._check_goal_fn()[0]


def test_native_goal_readback_exceptions_are_not_swallowed():
    class Broken(Task):
        def _create_termination_conditions(self):
            def error():raise RuntimeError('predicate unavailable')
            return {'predicate':SimpleNamespace(_check_goal_fn=error)}
    cleanup=install_process_goal_lifecycle(Broken,{'process'})
    try:
        predicate=Broken()._create_termination_conditions()['predicate']
        with pytest.raises(RuntimeError,match='unavailable'):predicate._check_goal_fn()
    finally:cleanup()


@dataclass
class Snapshot:
    official_success: bool
    research_diagnostics: dict


def test_actual_six_contracts_need_process_and_final_state_not_initial_static_truth():
    path=Path(__file__).parent / 'restricted_fixtures/m-reliable6/task_specs.jsonl'
    if not path.exists():pytest.skip('local six-task delivery fixture unavailable')
    rows=[json.loads(line) for line in path.open()]
    assert len(rows)==6
    for row in rows:
        branch=row['task_contract'];program=branch['process_requirements']['process_program']
        states={}
        for atom in branch['initial_assertions']:
            if atom['predicate'] in {'open','closed','toggled_on','toggled_off'}:
                states[atom['subject_role']]=atom['predicate'] in {'open','toggled_on'}
        def truth(atom):
            if atom.get('object_role') is not None:return True
            return states[atom['subject_role']]==(atom['predicate'] in {'open','toggled_on'})
        progress=MilestoneProgress(program,truth)
        progress.observe(progress.read(),0)
        canonical=SimpleNamespace(evaluate=lambda *_:Snapshot(all(truth(a) for a in program['final_requirements']),{}))
        verifier=ProcessCompletionVerifier(canonical,progress)
        assert canonical.evaluate().official_success and not verifier.evaluate(None,None).official_success
        events=[]
        def execute(action,role):
            before=progress.read();states[role]=action in {'open','toggle_on'};events.append((action,role))
            progress.observe(progress.read(),len(events),before,action)
        progress.execute_pending(execute)
        # Restore tasks also need canonical final restoration after normalization.
        for atom in program['final_requirements']:
            if not truth(atom):execute({'open':'open','closed':'close','toggled_on':'toggle_on','toggled_off':'toggle_off'}[atom['predicate']],atom['subject_role'])
        assert events and verifier.evaluate(None,None).official_success
