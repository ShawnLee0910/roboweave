"""Defer native static success to the existing canonical+process Oracle verifier."""

POLICY = 'STATIC_PREDICATE_TERMINATION_DEFERRED_TO_PROCESS_ORACLE_V1'


def install_process_goal_lifecycle(task_type, activity_names):
    """Scoped to registered activities; native failures/timeouts are untouched.

    Native PredicateGoal knows only BDDL leaves. In a process task these may be
    true before any required action. Keep its leaf readback, but let the existing
    ProcessCompletionVerifier decide success and stop the Oracle loop. Do not
    hide arbitrary terminated/truncated results in Evaluator.step().
    """
    names = frozenset(activity_names)
    original = task_type._create_termination_conditions

    def create(task):
        conditions = original(task)
        predicate = conditions.get('predicate')
        if predicate is None or not callable(getattr(predicate, '_check_goal_fn', None)):
            raise RuntimeError('native predicate termination interface changed')
        check = predicate._check_goal_fn
        task._behavior_hl_process_goal_lifecycle = {
            'policy': POLICY, 'activity_names': sorted(names), 'deferred_static_successes': 0,
            'canonical_process_verifier_required': True,
        }
        def guarded_goal():
            satisfied, leaves = check()  # Errors still propagate; no truth fabrication.
            if task.activity_name in names:
                if satisfied:
                    task._behavior_hl_process_goal_lifecycle['deferred_static_successes'] += 1
                return False, leaves
            return satisfied, leaves
        predicate._check_goal_fn = guarded_goal
        return conditions

    task_type._create_termination_conditions = create
    def cleanup():
        if task_type._create_termination_conditions is not create:
            raise RuntimeError('process-goal lifecycle hook ownership changed')
        task_type._create_termination_conditions = original
    return cleanup
