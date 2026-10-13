"""Use Runtime's actual execution audit; never substitute final truth for history."""
from dataclasses import replace
from functools import wraps


def audited_step(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        from behavior_hl.embodied.evaluator_bridge.l1_process_audit import ACTIVE
        token = ACTIVE.set(getattr(self, '_process_audit', None))
        try:
            return method(self, *args, **kwargs)
        finally:
            ACTIVE.reset(token)
    return call


class ProcessVerifier:
    def __init__(self, canonical, audit, *, lifecycle=None):
        self.canonical, self.audit = canonical, audit
        self.lifecycle = canonical if lifecycle is None else lifecycle

    def evaluate(self, episode, evaluator):
        return self._checked(self.canonical.evaluate(episode, evaluator))

    def _checked(self, snapshot):
        report = self.audit.report()
        return replace(snapshot, ok=snapshot.ok and report['operational'],
                       official_success=snapshot.official_success and report['public_process_satisfied'],
                       research_diagnostics={**dict(snapshot.research_diagnostics),
                                             'canonical_final_success': snapshot.research_diagnostics.get('canonical_final_success', snapshot.official_success),
                                             'public_process_report': report})


    def __getattr__(self, name):
        # Preserve optional verifier lifecycle hooks without inventing absent ones.
        value = getattr(self.lifecycle if name in {'retire', 'settle_final'} else self.canonical, name)
        if name == 'settle_final' and callable(value):
            return lambda *args, **kwargs: self._checked(value(*args, **kwargs))
        return value


def bind(scene, canonical, directory):
    from behavior_hl.embodied.evaluator_bridge.l1_process_audit import create_process_audit
    audit = create_process_audit(task=scene._registered_task, evaluator=scene.evaluator,
                                 directory=directory)
    scene._process_audit = audit
    lifecycle = canonical
    record = scene._registered_task.record if scene._registered_task is not None else {}
    branch = record.get('task_contract') or {}
    if branch.get('full_goal_acceptance'):
        from behavior_hl.embodied.evaluator_bridge.bounded_full_goal_acceptance import BoundedFullGoalAcceptance
        canonical = BoundedFullGoalAcceptance(canonical, predicate_mode=scene._runtime.predicate_mode)
    if audit is not None:
        if audit.history is not None:
            from behavior_hl.embodied.evaluator_bridge.l1_history_bindings import HistoryCompletionVerifier
            from behavior_hl.embodied.evaluator_bridge.l1_dynamic_process_goals import dynamic_goal_formula
            if branch.get('full_goal_acceptance') and dynamic_goal_formula(history=audit.history.program) is None:
                raise ValueError('history full-goal composition lacks a dynamic goal')
            canonical = HistoryCompletionVerifier(canonical, audit.history)
        if audit.milestones is not None:
            from behavior_hl.embodied.evaluator_bridge.l1_am_process_programs import ProcessCompletionVerifier
            if branch.get('full_goal_acceptance') and 'cycle_order_final' not in audit.milestones.program:
                raise ValueError('milestone full-goal composition lacks cycle-order final')
            canonical = ProcessCompletionVerifier(canonical, audit.milestones)
        canonical = ProcessVerifier(canonical, audit, lifecycle=lifecycle)
    template = record.get('physical_template')
    if template:
        from behavior_hl.embodied.evaluator_bridge.physical_template_runtime import bind_physical_model_verifier
        canonical = bind_physical_model_verifier(scene, canonical, directory)
    return canonical
