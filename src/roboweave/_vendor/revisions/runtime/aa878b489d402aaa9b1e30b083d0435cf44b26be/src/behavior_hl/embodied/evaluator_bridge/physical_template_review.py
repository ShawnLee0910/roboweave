"""Class-level post-review of measured evidence; never rewrites native Q.

Reuses complete-goal AND/OR expansion. Auxiliary facts require a named measurement
source; this module neither invents geometric predicates nor infers causality from Q.
"""
import json
import math

from .bounded_full_goal_acceptance import bounded_dnf


def capture_live_facts(task, *, role_object, holding_readback, upright_axes=None, certificates=None):
    """Read native predicates and declared pose axes; unsupported facts stay UNKNOWN.

    Caller binds actual live objects and action-owned release state. Causal/full-body
    certificates must be provided by the relevant measured-trace/geometry owner.
    This function does not sample physics or modify object poses.
    """
    import numpy as np
    from .spatial_symbolic_actions import canonical_relation, object_pose
    from .on_top_support_layers import quaternion_to_matrix_xyzw
    axes = upright_axes or {}
    certificates = certificates or {}
    atoms = set()
    def collect(node):
        if node[0] in ('and','or'):
            for child in node[1:]: collect(child)
        elif node[0]=='not': collect(node[1])
        else: atoms.add(tuple(node))
    collect(task['goal'])
    for rule in task['process_invariants']: collect(rule)
    facts = {}
    native = {'ontop':'OnTop','inside':'Inside','open':'Open','closed':'Open'}
    for atom in sorted(atoms):
        key = fact_key(atom)
        fact = {'status':'UNKNOWN','value':None,'source':None}
        try:
            if atom[0] in native:
                objects = [role_object(role) for role in atom[1:]]
                for obj in objects: obj.clear_states_cache()
                value = canonical_relation(objects[0],native[atom[0]],objects[1]) if len(objects)==2 else next(
                    state for kind,state in objects[0].states.items() if kind.__name__==native[atom[0]]).get_value()
                fact = dict(status='MEASURED', value=not bool(value) if atom[0]=='closed' else bool(value),
                            source='NATIVE_OBJECT_STATE_FRESH_READBACK')
            elif atom[0]=='upright' and atom[1] in axes:
                contract = axes[atom[1]]
                if contract.get('axis_status') != 'CALIBRATED' or not contract.get('source'):
                    raise ValueError('upright asset axis is not calibrated')
                axis = np.asarray(contract['up_local'],dtype=float)
                limit = contract['max_tilt_deg']
                if axis.shape!=(3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis),1) or not 0 < limit < 90:
                    raise ValueError('invalid upright contract')
                rotation = quaternion_to_matrix_xyzw(object_pose(role_object(atom[1]))[1])
                up_z = float((rotation@axis)[2])
                angle = math.degrees(math.acos(float(np.clip(up_z,-1,1))))
                fact = dict(status='MEASURED',value=angle<=limit,source=contract['source'],
                            tilt_deg=angle,max_tilt_deg=limit)
            elif atom[0]=='released':
                observed = holding_readback(role_object(atom[1]))
                fields = ('logical_held','logical_projection_controlled','official_held','attachment_present')
                if not observed.get('source') or any(type(observed.get(k)) is not bool for k in fields):
                    raise ValueError('release-owner evidence is incomplete')
                fact = dict(status='MEASURED',value=not any(observed[k] for k in fields),
                            source=observed['source'],owner_readback=observed)
            elif key in certificates:
                certificate = certificates[key]
                if certificate.get('status')=='MEASURED' and type(certificate.get('value')) is bool and certificate.get('source'):
                    fact = dict(certificate)
        except Exception as exc:
            fact['error'] = type(exc).__name__+': '+str(exc)
        facts[key] = fact
    return facts


def frozen(value):
    return tuple(frozen(x) for x in value) if isinstance(value, list) else value


def fact_key(atom):
    return json.dumps(atom, separators=(',', ':'), ensure_ascii=False)


def truth(atom, facts):
    if atom[0] == 'not':
        value = truth(atom[1], facts)
        return None if value is None else not value
    row = facts.get(fact_key(atom))
    if not isinstance(row, dict) or row.get('status') != 'MEASURED' or not row.get('source'):
        return None
    value = row.get('value')
    return value if type(value) is bool else None


def goal_truth(formula, facts):
    unknown = False
    for branch in bounded_dnf(frozen(formula)):
        values = [truth(atom, facts) for atom in branch]
        if all(value is True for value in values):
            return True
        if not any(value is False for value in values):
            unknown = True
    return None if unknown else False


def review_template(task, samples, *, native_q, required_window=10,
                    initial_goal_false=None, process_coverage_complete=False):
    """Accept only a stable complete branch; preserve every historical violation.

    Samples use globally increasing physics_step indices and contain final-scope
    measured facts. Caller must certify full process coverage for process tasks.
    No velocity or millimetre threshold is added here.
    """
    if native_q is not None and (type(native_q) not in (int,float)
                                or not math.isfinite(native_q) or not 0 <= native_q <= 1):
        raise ValueError('invalid native Q')
    if type(required_window) is not int or required_window < 1:
        raise ValueError('invalid window size')
    if any(type(s.get('physics_step')) is not int or s['physics_step'] < 0 or
           not isinstance(s.get('facts'), dict) for s in samples):
        raise ValueError('invalid measured sample')
    indices = [s['physics_step'] for s in samples]
    if any(b <= a for a,b in zip(indices, indices[1:])):
        raise ValueError('duplicate/reversed physics steps cannot fill a window')
    violations = []
    process_unknown = False
    for sample in samples:
        for rule in task['process_invariants']:
            value = truth(rule, sample['facts'])
            if value is False:
                violations.append({'physics_step':sample['physics_step'],'rule':rule})
            elif value is None:
                process_unknown = True
    if task['process_invariants'] and not process_coverage_complete:
        process_unknown = True
    tail = samples[-required_window:]
    goal_window = [goal_truth(task['goal'], s['facts']) for s in tail]
    # Require the same legal DNF branch throughout the window, not alternating goals.
    complete_branches = bounded_dnf(frozen(task['goal']))
    branch_status = []
    for branch in complete_branches:
        values = [truth(atom,s['facts']) for s in tail for atom in branch]
        branch_status.append(False if False in values else None if None in values else True)
    enough = len(tail) == required_window
    if violations or initial_goal_false is False:
        status = 'FAIL'
    elif enough and branch_status and all(v is False for v in branch_status):
        status = 'FAIL'
    elif not enough or process_unknown or initial_goal_false is not True:
        status = 'UNKNOWN'
    elif True in branch_status:
        status = 'PASS'
    else:
        status = 'UNKNOWN'
    return {'schema_version':'physical-template-review/v1','task_id':task['id'],
            'native_q_unchanged':native_q,'qualification_status':status,
            'goal_window':goal_window,'stable_legal_branch_status':branch_status,
            'process_violations':violations,'process_evidence_unknown':process_unknown,
            'window_samples':len(tail),'required_window':required_window,
            'initial_goal_false':initial_goal_false,
            'not_a_native_q_replacement':True}
