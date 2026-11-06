"""Bounded complete-goal scoring with a separate, fixed planning witness.

Private only. No new physical predicates or public observations are produced.
The bound rejects a whole formula; it never truncates its legal alternatives.
"""
from dataclasses import replace

CAPABILITY = 'L1_FULL_GOAL_ACCEPTANCE_V1'
PROJECTION_CAPABILITY = 'L1_POSITIVE_GOAL_PROJECTION_V1'
MAXIMUM_OPTIONS = 256
MAXIMUM_LEAVES = 4096


def is_binary_negative(literal):
    return literal[0] == 'not' and len(literal[1]) == 3


class UnsupportedNegativeGoalRecovery(RuntimeError):
    """Full negative goals are scored, but this mode does not plan their repair."""


class PositiveProjectionVerifier:
    def __init__(self, *, option, option_index, predicate_mode):
        from .official_predicate_verifier import OfficialPredicateVerifier
        from .official_predicate_identity import build_official_goal_leaf_ids
        from .inside_goal_graph import grounded_goal_option_identity
        from .benchmark_package_correspondence import _formula
        from .mixed_goal_graph import _leaf_body, relation_from_grounded_leaf
        import json
        ids = build_official_goal_leaf_ids([option])[0]
        self.negative_ids = frozenset(key for key, leaf in zip(ids, option, strict=True)
                                     if is_binary_negative(_formula(_leaf_body(leaf))))
        self.positive_ids = frozenset(ids) - self.negative_ids
        self.positive_keys = {key: json.dumps(relation_from_grounded_leaf(leaf).identity, separators=(',', ':'))
                              for key, leaf in zip(ids, option, strict=True) if key in self.positive_ids}
        if not self.positive_ids or not self.negative_ids:
            raise ValueError('positive projection requires both planned literals and binary negatives')
        self.full_witness = OfficialPredicateVerifier(selected_option_index=option_index,
            selected_option_identity=grounded_goal_option_identity(option), predicate_mode=predicate_mode)
        self.initial_positive = None

    def evaluate(self, episode, evaluator):
        snapshot = self.full_witness.evaluate(episode, evaluator)
        if not snapshot.ok:
            return snapshot
        positive = {self.positive_keys[key]: truth for key, truth in snapshot.leaf_truth.items() if key in self.positive_ids}
        if self.initial_positive is None:
            if not all(snapshot.leaf_truth[key] for key in self.negative_ids):
                raise UnsupportedNegativeGoalRecovery('UNSUPPORTED_NEGATIVE_GOAL_RECOVERY: initial forbidden relation holds')
            self.initial_positive = dict(positive)
        success = all(positive.values())
        q = 1. if success else sum(truth and not self.initial_positive[key] for key, truth in positive.items()) / len(positive)
        return replace(snapshot, official_success=success, official_q=q, leaf_truth=positive,
            research_diagnostics={**dict(snapshot.research_diagnostics),
                'leaf_truth_scope':'positive_planning_projection',
                'complete_witness_private': snapshot.to_private_record()})


def build_positive_goal_projection(evaluator, scope_witness, *, predicate_mode):
    """Attest one complete live option, then plan only its executable literals."""
    from .official_predicate_verifier import OfficialPredicateVerifier
    from .benchmark_package_correspondence import _formula
    from .mixed_goal_graph import _leaf_body, compile_grounded_mixed_option
    from .grounded_option_identity_owner import _openable_scope_names, _immovable_scope_names
    task = OfficialPredicateVerifier._resolve_task(evaluator)
    options = task.ground_goal_state_options
    if not 0 < len(options) <= MAXIMUM_OPTIONS:
        raise ValueError('complete goal option bound exceeded')
    _require_bounds(options)
    expected = _formula(scope_witness)
    matches = [(i, option) for i, option in enumerate(options)
               if _formula(['and', *(_leaf_body(leaf) for leaf in option)]) == expected]
    if len(matches) != 1:
        raise ValueError('complete witness must identify exactly one live grounded option')
    index, option = matches[0]
    projected = [leaf for leaf in option if not is_binary_negative(_formula(_leaf_body(leaf)))]
    plan = compile_grounded_mixed_option(projected, option_index=index,
        openable_targets=_openable_scope_names(task), immovable_targets=_immovable_scope_names(task))
    verifier = PositiveProjectionVerifier(option=option, option_index=index, predicate_mode=predicate_mode)
    audit = {'positive_goal_projection':True, 'complete_option_index':index,
             'complete_scope_witness':scope_witness, 'complete_goal_leaf_count':len(option),
             'planning_leaf_count':len(projected), 'binary_negative_recovery_supported':False}
    return plan, verifier, audit


def bounded_dnf(formula):
    """Expand normalized AND/OR/literals solely to attest a planning witness."""
    operator = formula[0]
    if operator == 'or':
        branches = []
        for child in formula[1:]:
            branches.extend(bounded_dnf(child))
            _require_bounds(branches)
        return branches
    if operator == 'and':
        branches = [frozenset()]
        for child in formula[1:]:
            options = bounded_dnf(child)
            if len(branches) * len(options) > MAXIMUM_OPTIONS:
                raise ValueError('complete goal option bound exceeded')
            branches = [a | b for a in branches for b in options]
            _require_bounds(branches)
        return branches
    if operator == 'not' and formula[1][0] in {'and', 'or', 'not'}:
        raise ValueError('complete goal requires literal negation')
    return [frozenset((formula,))]


def _require_bounds(options):
    if not 0 < len(options) <= MAXIMUM_OPTIONS or sum(map(len, options)) > MAXIMUM_LEAVES:
        raise ValueError('complete goal option/leaf bound exceeded')


class BoundedFullGoalAcceptance:
    """Keep selected leaf IDs for planning; score all options for acceptance."""
    def __init__(self, selected_verifier, *, predicate_mode):
        from .official_predicate_verifier import OfficialPredicateVerifier
        self.selected = selected_verifier
        self.complete = OfficialPredicateVerifier(require_single_option=False, predicate_mode=predicate_mode)
        self.task = None

    def evaluate(self, episode, evaluator):
        task = self.complete._resolve_task(evaluator)
        options = task.ground_goal_state_options
        # Check the count before iteration; large lazy grounded tables must not
        # be enumerated accidentally during a verifier call.
        if not 0 < len(options) <= MAXIMUM_OPTIONS:
            raise ValueError('complete goal option bound exceeded')
        _require_bounds(options)
        if self.task is not None and self.task is not task:
            raise ValueError('complete goal task changed without episode rebind')
        self.task = task
        selected = self.selected.evaluate(episode, evaluator)
        if not selected.ok:
            return selected
        complete = self.complete.evaluate(episode, evaluator)
        if not complete.ok:
            return complete
        return replace(selected, official_success=complete.official_success, official_q=complete.official_q,
            research_diagnostics={**dict(complete.research_diagnostics),
                'leaf_truth_scope': 'selected_planning_witness',
                'full_goal_acceptance': True,
                'selected_witness_private': selected.to_private_record(),
                'complete_goal_leaf_truth_private': dict(complete.leaf_truth)})
