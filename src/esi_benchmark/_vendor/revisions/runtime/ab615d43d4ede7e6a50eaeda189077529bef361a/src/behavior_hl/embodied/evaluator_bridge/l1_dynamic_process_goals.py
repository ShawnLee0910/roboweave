"""Finite physical goal alternatives; history still selects the actual answer."""
from copy import deepcopy

CAPABILITY = 'L1_DYNAMIC_PROCESS_GOAL_V1'


def validate_dynamic_goal(requirements, formula, roles):
    from .benchmark_package_correspondence import _formula, _role_formula
    expected = dynamic_goal_formula(history=requirements.get('history_program'), process=requirements.get('process_program'))
    if expected is None or formula is None or _formula(_role_formula(expected, roles)) != _formula(_role_formula(formula, roles)):
        raise ValueError('dynamic process physical alternatives differ from canonical goal')


def dynamic_goal_formula(*, history=None, process=None):
    if history is not None:
        if history['clauses']:
            return None
        bindings = [b for b in history['bindings'] if b['time'] == 'FIRST_VERIFIED_SUBJECT_SUCCESS']
        if len(bindings) != 1:
            raise ValueError('empty static history goal requires one subject binding')
        binding = bindings[0]
        name = binding['binding_id']
        domain = binding['event']['subject_domain_roles']

        def substitute(node, selected):
            if node.get('op') == 'AND':
                return ['and', *[substitute(a, selected) for a in node['args']]]
            if node.get('op') == 'FOR_EACH_EXCEPT_BINDING':
                if node['binding_ref'] != name or set(node['roles']) != set(domain):
                    raise ValueError('dynamic complement domain mismatch')
                return ['and', *[[node['predicate'], r, node['object_role']] for r in domain if r != selected]]
            atom = deepcopy(node)
            if 'subject_ref' in atom:
                if atom.pop('subject_ref') != {'binding_ref': name}:
                    raise ValueError('unsupported dynamic subject reference')
                atom['subject_role'] = selected
            if set(atom) - {'predicate', 'subject_role', 'object_role', 'desired'} or atom.get('desired', True) is not True:
                raise ValueError('unsupported dynamic physical goal atom')
            return [atom['predicate'], atom['subject_role'], *([atom['object_role']] if 'object_role' in atom else [])]
        return ['or', *[substitute(history['expression'], role) for role in domain]]
    if process is not None and 'cycle_order_final' in process:
        rule = process['cycle_order_final']
        stages = process.get('milestones', process.get('ordered_milestones'))
        by_id = {s['milestone_id']: s for s in stages}
        roles = []
        for name in rule['cycle_end_milestones']:
            stage = by_id[name]
            atom = stage['requirements'][0] if 'requirements' in stage else stage['condition']
            roles.append(atom['subject_role'])
        def leaf(predicate, role):
            if predicate in {'closed', 'toggled_off'}:
                return ['not', [{'closed': 'open', 'toggled_off': 'toggled_on'}[predicate], role]]
            return [predicate, role]
        return ['or', *[['and', *[leaf(rule['active_predicate'] if r == chosen else rule['neutral_predicate'], r) for r in roles]] for chosen in roles]]
    return None
