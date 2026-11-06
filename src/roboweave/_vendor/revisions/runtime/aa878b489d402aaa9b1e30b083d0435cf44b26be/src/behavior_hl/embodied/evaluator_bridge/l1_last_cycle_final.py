"""Dynamic final unary states derived from verified required-cycle completion.

The ledger owns transition verification. This module neither invents events nor
freezes a preferred subject. Repeated manipulation after a required check is
complete does not turn it into a second required check.
"""
OPPOSITE = {'open': 'closed', 'closed': 'open', 'toggled_on': 'toggled_off', 'toggled_off': 'toggled_on'}


def validate_last_cycle_final(rule, milestones):
    if (not isinstance(rule, dict) or set(rule) != {'cycle_end_milestones', 'active_predicate', 'neutral_predicate'}
            or rule['active_predicate'] not in OPPOSITE
            or rule['neutral_predicate'] != OPPOSITE[rule['active_predicate']]):
        raise ValueError('invalid last-cycle final rule')
    names = rule['cycle_end_milestones']
    if (not isinstance(names, list) or len(names) < 2 or any(not isinstance(n, str) for n in names)
            or len(names) != len(set(names))):
        raise ValueError('last-cycle final needs distinct cycle endpoints')
    by_name = {stage['milestone_id']: stage for stage in milestones}
    roles = set()
    for name in names:
        stage = by_name.get(name)
        if not stage or len(stage['requirements']) != 1 or len(stage['after']) != 1:
            raise ValueError('last-cycle endpoint must close a verified unary cycle')
        end = stage['requirements'][0]
        previous = by_name.get(stage['after'][0])
        if not previous or len(previous['requirements']) != 1:
            raise ValueError('last-cycle endpoint has no valid predecessor')
        start = previous['requirements'][0]
        if (end['predicate'] != rule['neutral_predicate'] or start['predicate'] != rule['active_predicate']
                or start['subject_role'] != end['subject_role'] or end['subject_role'] in roles):
            raise ValueError('last-cycle endpoints must close distinct same-kind device checks')
        roles.add(end['subject_role'])
        if any(name in other['after'] for other in milestones):
            raise ValueError('last-cycle endpoint is not the final required check for its device')
    if roles != {a['subject_role'] for stage in milestones for a in stage['requirements']}:
        raise ValueError('last-cycle final must cover every checked device')


def last_cycle_final_atoms(rule, milestones, reached):
    """Return None until every requested check completed with distinct event IDs."""
    names = rule['cycle_end_milestones']
    if not all(name in reached for name in names):
        return None
    events = [reached[name] for name in names]
    if any(type(event) is not int or event <= 0 for event in events) or len(events) != len(set(events)):
        return None
    latest = max(names, key=reached.__getitem__)
    by_name = {stage['milestone_id']: stage for stage in milestones}
    return [dict(predicate=rule['active_predicate'] if name == latest else rule['neutral_predicate'],
                 subject_role=by_name[name]['requirements'][0]['subject_role'], desired=True) for name in names]
