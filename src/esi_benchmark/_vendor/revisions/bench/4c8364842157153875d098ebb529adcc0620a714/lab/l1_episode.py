"""Use Runtime's actual execution audit; never substitute final truth for history."""
from contextlib import nullcontext
from dataclasses import replace
from functools import wraps
from pathlib import Path


def audited_step(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        from behavior_hl.embodied.evaluator_bridge.l1_process_audit import ACTIVE
        from behavior_hl.embodied.evaluator_bridge.l2_mv_session import ACTIVE as MV_ACTIVE

        physical = getattr(self, '_physical_session', None)
        token = ACTIVE.set(getattr(self, '_process_audit', None))
        mv_token = MV_ACTIVE.set(getattr(self, '_mv_session', None))
        try:
            with (nullcontext() if physical is None else physical.observe()):
                return method(self, *args, **kwargs)
        finally:
            MV_ACTIVE.reset(mv_token)
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
        ok = snapshot.ok and report['operational']
        return replace(snapshot, ok=ok,
                       official_success=ok and snapshot.official_success and report['public_process_satisfied'],
                       official_q=snapshot.official_q if ok else None,
                       leaf_truth=snapshot.leaf_truth if ok else {},
                       error=None if ok else (snapshot.error or 'process audit is not operational'),
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
    from behavior_hl.embodied.evaluator_bridge.l2_mv_session import (
        MVSession,
        validate_sidecar,
    )

    task = scene._registered_task
    composition = {} if task is None else task.record.get('composition_contract', {})
    scene._physical_session = None
    has_physical_template = (
        task is not None and task.record.get('physical_template') is not None
    )
    if has_physical_template and not getattr(scene, '_physical_execution', False):
        raise ValueError('Physical task requires the Physical human execution lane')
    if getattr(scene, '_physical_execution', False) and not has_physical_template:
        raise ValueError('Physical human execution lane requires a physical template')
    if has_physical_template:
        from behavior_hl.embodied.evaluator_bridge.physical_human_session import (
            PhysicalHumanSession,
        )

        scene._physical_session = PhysicalHumanSession(
            task=task,
            evaluator=scene.evaluator,
            runtime=scene._runtime,
            directory=Path(directory),
        )
    audit = create_process_audit(task=task, evaluator=scene.evaluator, directory=directory)
    scene._process_audit = audit
    scene._mv_session = None
    if audit is None:
        if 'M' in composition.get('axes', ()) or composition.get('v_stage') in {'D0', 'D1'}:
            raise ValueError('declared M/V composition requires a process audit contract')
        return canonical
    if composition.get('v_stage') in {'D0', 'D1'}:
        spec = getattr(scene, '_registered_task_spec', None)
        if spec is None:
            spec = task.record
        if (spec.get('task_spec_id') != task.record.get('task_spec_id')
                or spec.get('composition_contract') != composition):
            raise ValueError('V TaskSpec differs from registered task')
        config = validate_sidecar(composition['v_sidecar'], spec)
        if (config.get('validation_mode') == 'PRIVILEGED_TEST_ONLY'
                and not getattr(scene, '_privileged_validation', False)):
            raise ValueError('test-only V requires an explicitly privileged episode')
        scene._mv_session = MVSession(config, audit, Path(directory) / 'private/l2_v')
    lifecycle = canonical
    branch = task.record['task_contract']
    if branch.get('full_goal_acceptance'):
        from behavior_hl.embodied.evaluator_bridge.bounded_full_goal_acceptance import BoundedFullGoalAcceptance
        canonical = BoundedFullGoalAcceptance(canonical, predicate_mode=scene._runtime.predicate_mode)
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
    return ProcessVerifier(canonical, audit, lifecycle=lifecycle)


def v_protocol_ready_for_terminal(scene):
    """Return whether a V sidecar has completed its non-goal obligations."""

    mv = getattr(scene, '_mv_session', None)
    if mv is None:
        return True
    return bool(
        mv.consumed
        and mv.delivered
        and not mv.failed
        and not mv.checkpoint_violations
        and (mv.config['branch'] == 'D0' or mv.recovered)
    )


def physical_goal_ready_for_terminal(scene):
    """Ask the shared Physical session; final settling still occurs at end."""

    physical = getattr(scene, '_physical_session', None)
    return physical is None or physical.complete_goal_now()


def finish_physical_protocol(scene):
    """Run the Runtime-owned Physical finalizer once, preserving its artifact."""

    physical = getattr(scene, '_physical_session', None)
    if physical is None:
        return None
    try:
        return physical.finish()
    except Exception as exc:
        return {
            'status': 'FAIL_PHYSICAL_TEMPLATE',
            'task_success': False,
            'failure_class': 'PHYSICAL_TEMPLATE_FINALIZATION_FAILED',
            'error_type': type(exc).__name__,
            'error': str(exc),
        }


def finish_v_protocol(scene, result):
    """Finalize Runtime's V session without allowing an artifact error to vanish."""

    mv = getattr(scene, '_mv_session', None)
    if mv is None:
        return None
    try:
        return mv.finish(dict(result))
    except Exception as exc:
        report = {
            'v_protocol_complete': False,
            'error_type': type(exc).__name__,
            'error': str(exc),
        }
        try:
            mv.save('final.json', report)
        except Exception:
            pass
        return report
