"""Private finite first-success and first-complete-layout task contracts.

The action hook supplies measured held identity and canonical readback. Oracle
choices are witnesses only: evaluation follows the actual first successful
choice. No binding is initialized from a preferred destination or reset branch.
"""
from copy import deepcopy
from dataclasses import replace

CAPABILITY = 'L1_HISTORY_BINDINGS_V1'
SUBJECT_CAPABILITY = 'L1_SUBJECT_HISTORY_BINDING_V1'
MAINTAIN_CAPABILITY = 'L1_MAINTAIN_UNTIL_V1'
SCHEMA = 'l1-history-bindings/v1'
PLACEMENTS = {'ontop': 'place_on_top', 'inside': 'place_inside',
              'under': 'place_under', 'nextto': 'place_next_to', 'next_to': 'place_next_to'}
UNARY = {'open': 'open', 'closed': 'close', 'toggled_on': 'toggle_on', 'toggled_off': 'toggle_off'}


def required_history_capabilities(program):
    required = {CAPABILITY}
    if any(b['time'] == 'FIRST_VERIFIED_SUBJECT_SUCCESS' for b in program['bindings']):
        required.add(SUBJECT_CAPABILITY)
    if program['process'].get('maintain_until'):
        required.add(MAINTAIN_CAPABILITY)
    if program['process'].get('before_unary_effect'):
        required.add('L1_CURRENT_UNARY_PREREQUISITES_V1')
    return required


def validate_history_program(program, roles):
    """Reject every expression/timing that the runtime cannot preserve."""
    roles = set(roles)
    if (not isinstance(program, dict) or set(program) != {'schema_version', 'bindings', 'clauses', 'expression', 'process'}
            or program['schema_version'] != SCHEMA):
        raise ValueError('unsupported history program schema')
    bindings = program['bindings']
    if not isinstance(bindings, list) or not 0 <= len(bindings) <= 16:
        raise ValueError('history requires bounded bindings')
    names = {}
    def domain(items):
        if not isinstance(items, list) or not items or len(items) != len(set(items)) or not set(items) <= roles:
            raise ValueError('invalid history role domain')
    for b in bindings:
        name = b.get('binding_id') if isinstance(b, dict) else None
        if not isinstance(name, str) or not name or name.casefold() in {n.casefold() for n in names}:
            raise ValueError('invalid history binding identity')
        names[name] = b
    def ref(value, *, subject=False):
        if not isinstance(value, dict): raise ValueError('invalid history reference')
        if set(value) == {'binding_ref'}:
            b = names.get(value['binding_ref'])
            timing = 'FIRST_VERIFIED_SUBJECT_SUCCESS' if subject else 'FIRST_VERIFIED_SUCCESS'
            if b is None or b['time'] != timing:
                raise ValueError('scalar history reference requires first-success binding')
        elif not subject and set(value) == {'op', 'map_ref', 'key_role'} and value['op'] == 'MAP_LOOKUP':
            m = value['map_ref']
            b = names.get(m.get('binding_ref')) if isinstance(m, dict) and set(m) == {'binding_ref'} else None
            if b is None or b['time'] != 'FIRST_VERIFIED_PHASE_COMPLETION' or value['key_role'] not in b['select']['roles']:
                raise ValueError('unbound history map lookup')
        else: raise ValueError('unsupported history reference')
    def expr(node, *, references=True):
        if not isinstance(node, dict): raise ValueError('invalid history expression')
        if node.get('op') == 'AND':
            if set(node) != {'op', 'args'} or not isinstance(node['args'], list) or not 0 < len(node['args']) <= 64:
                raise ValueError('invalid history conjunction')
            for child in node['args']: expr(child, references=references)
        elif node.get('op') in {'BIJECTION', 'RELATION_MAP'}:
            if set(node) != {'op', 'roles', 'predicate', 'destination_domain_roles'} or node['predicate'] not in PLACEMENTS:
                raise ValueError('unsupported history relation map')
            domain(node['roles']); domain(node['destination_domain_roles'])
            if len(node['roles']) != len(node['destination_domain_roles']) or set(node['roles']) & set(node['destination_domain_roles']):
                raise ValueError('history map requires disjoint equal-size domains')
        elif node.get('op') == 'BINDING_EXISTS' and references:
            if set(node) != {'op', 'binding_ref'} or node['binding_ref'] not in names:
                raise ValueError('unknown history binding')
        elif node.get('op') == 'FOR_EACH_EXCEPT_BINDING' and references:
            if set(node) != {'op','roles','binding_ref','predicate','object_role'} or node['predicate'] not in PLACEMENTS or node['object_role'] not in roles:
                raise ValueError('invalid history subject complement')
            domain(node['roles']); ref({'binding_ref':node['binding_ref']}, subject=True)
            if set(node['roles']) != set(names[node['binding_ref']]['event']['subject_domain_roles']) or node['object_role'] in node['roles']:
                raise ValueError('history subject complement must cover its binding domain')
        else:
            p = node.get('predicate')
            fields = {'predicate', 'subject_role', 'subject_ref', 'desired'}
            if p in PLACEMENTS: fields |= {'object_role', 'object_ref'}
            if p not in PLACEMENTS | UNARY or set(node) - fields or node.get('desired', True) is not True:
                raise ValueError('unsupported history atom')
            if ('subject_role' in node) == ('subject_ref' in node): raise ValueError('history atom needs one subject')
            if 'subject_role' in node:
                if node['subject_role'] not in roles: raise ValueError('unknown history subject')
            elif references: ref(node['subject_ref'], subject=True)
            else: raise ValueError('static history atom cannot depend on a subject binding')
            if p in PLACEMENTS:
                if ('object_role' in node) == ('object_ref' in node): raise ValueError('history relation needs one reference')
                if 'object_role' in node:
                    if node['object_role'] not in roles: raise ValueError('unknown history destination')
                elif references: ref(node['object_ref'])
                else: raise ValueError('binding trigger cannot depend on another history binding')
    for b in bindings:
        common = {'binding_id', 'time', 'cardinality', 'update', 'on_failed_cancelled_or_selection'}
        if b.get('on_failed_cancelled_or_selection') != 'DO_NOT_BIND': raise ValueError('invalid history failure semantics')
        if b.get('time') == 'FIRST_VERIFIED_SUBJECT_SUCCESS':
            e = b.get('event', {})
            if (set(b) != common | {'event'} or b.get('cardinality') != 'EXACTLY_ONE' or b.get('update') != 'NEVER'
                    or set(e) != {'action','subject_domain_roles','object_role','required_public_result','required_relation_readback'}
                    or e.get('action') not in PLACEMENTS.values() or e.get('object_role') not in roles
                    or e.get('required_public_result') != 'completed'
                    or not (e.get('required_relation_readback') is True or e.get('required_relation_readback') == 'TRUE')):
                raise ValueError('unsupported first successful subject binding')
            domain(e['subject_domain_roles'])
            if e['object_role'] in e['subject_domain_roles']: raise ValueError('self history placement')
        elif b.get('time') == 'FIRST_VERIFIED_SUCCESS':
            e = b.get('event', {})
            if (set(b) != common | {'event'} or b.get('cardinality') != 'EXACTLY_ONE' or b.get('update') != 'NEVER'
                    or set(e) != {'action','subject_role','destination_domain_roles','required_public_result','required_relation_readback'}
                    or e.get('action') not in PLACEMENTS.values() or e.get('subject_role') not in roles
                    or e.get('required_public_result') != 'completed'
                    or not (e.get('required_relation_readback') is True or e.get('required_relation_readback') == 'TRUE')):
                raise ValueError('unsupported first-success binding')
            domain(e['destination_domain_roles'])
            if e['subject_role'] in e['destination_domain_roles']: raise ValueError('self history placement')
        elif b.get('time') == 'FIRST_VERIFIED_PHASE_COMPLETION':
            t = b.get('trigger', {})
            if (set(b) != common | {'trigger','select'} or b.get('cardinality') != 'TOTAL_BIJECTION'
                    or b.get('update') != 'NEVER_AFTER_TRIGGER'
                    or not {'after_action','required_public_result','condition'} <= set(t)
                    or set(t) - {'after_action','required_public_result','required_relation_readback','condition'}
                    or t.get('after_action') not in set(PLACEMENTS.values()) | set(UNARY.values())
                    or t.get('required_public_result') != 'completed'
                    or not (t.get('required_relation_readback', True) is True or t.get('required_relation_readback') == 'TRUE')
                    or b['select'].get('op') != 'RELATION_MAP'):
                raise ValueError('unsupported phase binding')
            expr(t['condition'], references=False); expr(b['select'], references=False)
        else: raise ValueError('unsupported history timing')
    if not isinstance(program['clauses'], list): raise ValueError('invalid static history goals')
    if not program['clauses'] and not any(
            b['time'] in {'FIRST_VERIFIED_SUBJECT_SUCCESS', 'FIRST_VERIFIED_SUCCESS'}
            for b in bindings):
        raise ValueError('empty static history goals require a dynamic binding')
    for a in program['clauses']: expr(a, references=False)
    expr(program['expression'])
    process = program['process']
    if not isinstance(process, dict) or set(process) - {'ordered_milestones', 'before_placement', 'final_placement_after', 'maintain_until', 'before_unary_effect'}:
        raise ValueError('unsupported history process')
    known = set()
    stages = process.get('ordered_milestones', [])
    if not isinstance(stages, list) or not stages: raise ValueError('history stages required')
    for stage in stages:
        if (not isinstance(stage, dict) or set(stage) - {'milestone_id','condition','after'}
                or not isinstance(stage.get('milestone_id'), str) or not stage['milestone_id']
                or stage['milestone_id'].casefold() in {n.casefold() for n in known}
                or not isinstance(stage.get('after', []), list)
                or len(stage.get('after', [])) != len(set(stage.get('after', [])))
                or not set(stage.get('after', [])) <= known):
            raise ValueError('invalid ordered history stage')
        expr(stage['condition']); known.add(stage['milestone_id'])
    for rule in process.get('before_placement', []):
        subject_key = 'subject_ref' if 'subject_ref' in rule else 'subject_role'
        if (set(rule) != {subject_key,'placement_scope','required_milestone'}
                or rule['required_milestone'] not in known or rule['placement_scope'].get(subject_key) != rule[subject_key]
                or rule['placement_scope'].get('predicate') not in PLACEMENTS):
            raise ValueError('invalid history placement guard')
        expr(rule['placement_scope'])
    guards = process.get('before_unary_effect', [])
    if not isinstance(guards, list) or len(guards) > 16:
        raise ValueError('invalid current unary prerequisites')
    for rule in guards:
        if (not isinstance(rule, dict) or set(rule) != {'effect', 'requires_current'}
                or not isinstance(rule['requires_current'], list)
                or not 1 <= len(rule['requires_current']) <= 16):
            raise ValueError('invalid current unary prerequisite rule')
        for atom in [rule['effect'], *rule['requires_current']]:
            if (not isinstance(atom, dict) or set(atom) - {'predicate', 'subject_role', 'desired'}
                    or atom.get('predicate') not in UNARY or atom.get('desired', True) is not True):
                raise ValueError('current unary prerequisites require positive unary atoms')
            expr(atom, references=False)
    guards = process.get('maintain_until', [])
    if not isinstance(guards, list) or len(guards) > 16: raise ValueError('invalid history invariant list')
    for rule in guards:
        if (not isinstance(rule, dict) or set(rule) != {'condition','required_milestone'}
                or rule['required_milestone'] not in known or not isinstance(rule['condition'], dict)
                or rule['condition'].get('predicate') not in UNARY):
            raise ValueError('invalid history maintained state')
        expr(rule['condition'], references=False)
    gate = process.get('final_placement_after')
    if gate is not None:
        if set(gate) != {'roles','milestone_id','scope'} or gate['milestone_id'] not in known or gate['scope'] != 'DECLARED_FINAL_GOAL_RELATIONS':
            raise ValueError('invalid history final placement gate')
        domain(gate['roles'])


class HistoryProgress:
    def __init__(self, program, truth):
        self.program, self.truth = deepcopy(program), truth
        self.bindings, self.binding_events, self.reached = {}, {}, {}
        self.events, self.violations, self.errors = set(), [], []

    @property
    def tracked_roles(self):
        roles = set()
        for b in self.program['bindings']:
            roles.update(b['event'].get('subject_domain_roles', [b['event'].get('subject_role')]) if 'event' in b else b['select']['roles'])
        roles.update(a['subject_role'] for a in self.atoms(self.program['expression']) if 'subject_role' in a)
        for stage in self.program['process']['ordered_milestones']:
            roles.update(a['subject_role'] for a in self.atoms(stage['condition']) if 'subject_role' in a)
        return roles

    @staticmethod
    def atoms(node):
        if node.get('op') == 'AND': return [a for child in node['args'] for a in HistoryProgress.atoms(child)]
        return [node] if 'predicate' in node else []

    def resolve(self, atom):
        a = deepcopy(atom)
        subject = a.pop('subject_ref', None)
        if subject is not None:
            role = self.bindings.get(subject['binding_ref'])
            if role is None: return None
            a['subject_role'] = role
        r = a.pop('object_ref', None)
        if r is not None:
            if 'binding_ref' in r: destination = self.bindings.get(r['binding_ref'])
            else: destination = self.bindings.get(r['map_ref']['binding_ref'], {}).get(r['key_role'])
            if destination is None: return None
            a['object_role'] = destination
        a.setdefault('desired', True)
        return a

    def relation_map(self, node):
        found = {}
        for subject in node['roles']:
            matches = [d for d in node['destination_domain_roles'] if self.truth(dict(
                predicate=node['predicate'], subject_role=subject, object_role=d, desired=True))]
            if len(matches) != 1: return None
            found[subject] = matches[0]
        return found if len(set(found.values())) == len(found) else None

    def evaluate(self, node):
        op = node.get('op')
        if op == 'AND': return all(self.evaluate(a) for a in node['args'])
        if op == 'BINDING_EXISTS': return node['binding_ref'] in self.bindings
        if op in {'BIJECTION','RELATION_MAP'}: return self.relation_map(node) is not None
        if op == 'FOR_EACH_EXCEPT_BINDING':
            selected = self.bindings.get(node['binding_ref'])
            return selected is not None and all(self.truth(dict(predicate=node['predicate'], subject_role=r,
                object_role=node['object_role'], desired=True)) for r in node['roles'] if r != selected)
        atom = self.resolve(node)
        return atom is not None and bool(self.truth(atom))

    def snapshot(self):
        candidates = {}
        for b in self.program['bindings']:
            if 'event' in b:
                e = b['event']; predicate = next(p for p, a in PLACEMENTS.items() if a == e['action'])
                if b['time'] == 'FIRST_VERIFIED_SUBJECT_SUCCESS':
                    candidates[b['binding_id']] = [r for r in e['subject_domain_roles'] if self.truth(dict(
                        predicate=predicate, subject_role=r, object_role=e['object_role'], desired=True))]
                    continue
                matches = [d for d in e['destination_domain_roles'] if self.truth(dict(predicate=predicate,
                    subject_role=e['subject_role'], object_role=d, desired=True))]
                candidates[b['binding_id']] = matches[0] if len(matches) == 1 else None
            else:
                candidates[b['binding_id']] = self.relation_map(b['select']) if self.evaluate(b['trigger']['condition']) else None
        return {'candidates': candidates, 'reached': dict(self.reached),
                'unary_prerequisites': [dict(effect=self.evaluate(rule['effect']),
                    satisfied=all(self.evaluate(a) for a in rule['requires_current']))
                    for rule in self.program['process'].get('before_unary_effect', [])],
                'maintained_states': [self.evaluate(rule['condition']) for rule in self.program['process'].get('maintain_until', [])]}

    def _condition_actions(self, node):
        """Return public actions that can establish a history milestone."""
        op = node.get('op') if isinstance(node, dict) else None
        if op == 'AND':
            return {action for child in node['args'] for action in self._condition_actions(child)}
        if op == 'BINDING_EXISTS':
            binding = next((item for item in self.program['bindings']
                            if item['binding_id'] == node['binding_ref']), None)
            if binding is None:
                return set()
            event = binding.get('event', binding.get('trigger', {}))
            return {event.get('action', event.get('after_action'))} - {None}
        if op in {'BIJECTION', 'RELATION_MAP', 'FOR_EACH_EXCEPT_BINDING'}:
            predicate = node.get('predicate')
        else:
            predicate = node.get('predicate') if isinstance(node, dict) else None
        if predicate in PLACEMENTS:
            # Preserve legacy M fixtures that use one generic placement hook
            # while still requiring a real held-object release event.
            return set(PLACEMENTS.values())
        if predicate in UNARY:
            return {UNARY[predicate]}
        if isinstance(node, dict):
            actions = set()
            for key in ('subject_ref', 'object_ref'):
                ref = node.get(key)
                if isinstance(ref, dict) and 'binding_ref' in ref:
                    binding = next((item for item in self.program['bindings']
                                    if item['binding_id'] == ref['binding_ref']), None)
                    if binding is not None:
                        event = binding.get('event', binding.get('trigger', {}))
                        actions.add(event.get('action', event.get('after_action')))
            return actions - {None}
        return set()

    def _milestone_event_allowed(self, condition, action, before, after):
        actions = self._condition_actions(condition)
        if action not in actions:
            return False
        if action not in PLACEMENTS.values():
            return True
        released = set(before.get('held_roles', ())) - set(after.get('held_roles', ()))
        return bool(released)

    def observe(self, action, before, after, status, event_id):
        if event_id in self.events: return
        self.events.add(event_id)
        for i, rule in enumerate(self.program['process'].get('maintain_until', [])):
            if (rule['required_milestone'] not in before['history']['reached']
                    and (not before['history']['maintained_states'][i] or not after['history']['maintained_states'][i])):
                self.violations.append(dict(event_id=event_id, kind='HISTORY_STATE_NOT_MAINTAINED',
                    subject_role=rule['condition']['subject_role'], required_milestone=rule['required_milestone']))
        if status != 'completed':
            if (action in PLACEMENTS.values() and set(before['held_roles']) - set(after['held_roles'])
                    and status not in {'needs_marker','marker_required','awaiting_selection','selection_required'}):
                self.errors.append(dict(event_id=event_id, kind='HISTORY_RELEASE_WITHOUT_VERIFIED_SUCCESS'))
            return
        for i, rule in enumerate(self.program['process'].get('before_unary_effect', [])):
            pre = before['history']['unary_prerequisites'][i]
            post = after['history']['unary_prerequisites'][i]
            # Audit a verified agent-caused transition, not a latched milestone,
            # unrelated action, marker request, or external intervention event.
            if (action == UNARY[rule['effect']['predicate']] and not pre['effect']
                    and post['effect'] and not pre['satisfied']):
                self.violations.append(dict(event_id=event_id,
                    kind='HISTORY_UNARY_EFFECT_BEFORE_CURRENT_PREREQUISITES',
                    subject_role=rule['effect']['subject_role'], rule=i))
        prior = set(self.reached)
        for b in self.program['bindings']:
            name = b['binding_id']; candidate = after['history']['candidates'][name]
            if name in self.bindings or candidate is None: continue
            if 'event' in b:
                e = b['event']
                if b['time'] == 'FIRST_VERIFIED_SUBJECT_SUCCESS':
                    released = set(before['held_roles']) - set(after['held_roles'])
                    matches = released & set(candidate) - set(before['history']['candidates'][name])
                    eligible = action == e['action'] and len(matches) == 1
                    candidate = next(iter(matches)) if eligible else None
                else:
                    subject = e['subject_role']
                    eligible = action == e['action'] and subject in before['held_roles'] and subject not in after['held_roles']
            else: eligible = action == b['trigger']['after_action']
            if eligible:
                self.bindings[name] = deepcopy(candidate); self.binding_events[name] = event_id
        for stage in self.program['process']['ordered_milestones']:
            name = stage['milestone_id']
            if (name not in self.reached and set(stage.get('after', [])) <= prior
                    and self._milestone_event_allowed(stage['condition'], action, before, after)
                    and self.evaluate(stage['condition'])):
                self.reached[name] = event_id
        if action not in PLACEMENTS.values(): return
        guards = list(self.program['process'].get('before_placement', []))
        gate = self.program['process'].get('final_placement_after')
        if gate:
            guards += [dict(subject_role=a['subject_role'], placement_scope=a, required_milestone=gate['milestone_id'])
                       for a in self.atoms(self.program['expression']) + self.program['clauses']
                       if a.get('subject_role') in gate['roles'] and a['predicate'] in PLACEMENTS]
        for rule in guards:
            atom = self.resolve(rule['placement_scope'])
            role = None if atom is None else atom['subject_role']
            if (role in before['held_roles'] and role not in after['held_roles'] and self.evaluate(rule['placement_scope'])
                    and rule['required_milestone'] not in before['history']['reached']):
                self.violations.append(dict(event_id=event_id, kind='HISTORY_PLACEMENT_BEFORE_MILESTONE', subject_role=role))

    @property
    def complete(self):
        return (not self.errors and not self.violations and len(self.bindings) == len(self.program['bindings'])
                and len(self.reached) == len(self.program['process']['ordered_milestones'])
                and all(self.evaluate(a) for a in self.program['clauses']) and self.evaluate(self.program['expression']))

    def execute_pending(self, execute_atom):
        """Finite Oracle witness using the caller's existing action/budget owner."""
        def ensure(node):
            if self.evaluate(node): return
            if node.get('op') == 'AND':
                # Satisfy placements before closing destination doors.
                for a in sorted(node['args'], key=lambda a: a.get('predicate') == 'closed'): ensure(a)
            elif node.get('op') in {'BIJECTION','RELATION_MAP'}:
                # This choice guides the private Oracle only. No evaluation
                # binding is written here; the execution hook establishes it.
                for subject, destination in zip(node['roles'], node['destination_domain_roles']):
                    ensure(dict(predicate=node['predicate'], subject_role=subject, object_role=destination))
            elif node.get('op') == 'FOR_EACH_EXCEPT_BINDING':
                selected = self.bindings.get(node['binding_ref'])
                if selected is None: raise RuntimeError('history subject binding not established')
                for role in node['roles']:
                    if role != selected: ensure(dict(predicate=node['predicate'], subject_role=role, object_role=node['object_role']))
            elif node.get('op') == 'BINDING_EXISTS':
                b = next(b for b in self.program['bindings'] if b['binding_id'] == node['binding_ref'])
                if 'event' in b:
                    e = b['event']; p = next(p for p, a in PLACEMENTS.items() if a == e['action'])
                    if b['time'] == 'FIRST_VERIFIED_SUBJECT_SUCCESS':
                        execute_atom(dict(predicate=p, subject_role=e['subject_domain_roles'][0], object_role=e['object_role'], desired=True))
                    else:
                        execute_atom(dict(predicate=p, subject_role=e['subject_role'], object_role=e['destination_domain_roles'][0], desired=True))
                else: ensure(b['trigger']['condition'])
            else:
                atom = self.resolve(node)
                if atom is None: raise RuntimeError('history target binding not established')
                execute_atom(atom)
            if not self.evaluate(node): raise RuntimeError('history operation lacks canonical readback or verified binding')
        for stage in self.program['process']['ordered_milestones']:
            if stage['milestone_id'] in self.reached: continue
            if not set(stage.get('after', [])) <= set(self.reached): raise RuntimeError('history predecessor missing')
            ensure(stage['condition'])
            if stage['milestone_id'] not in self.reached: raise RuntimeError('history stage lacks a distinct verified event')
        ensure(self.program['expression'])
        for atom in self.program['clauses']: ensure(atom)
        if not self.complete: raise RuntimeError('history task requirements not satisfied')


class HistoryCompletionVerifier:
    def __init__(self, canonical, history): self.canonical, self.history = canonical, history

    def evaluate(self, episode, evaluator):
        snapshot = self.canonical.evaluate(episode, evaluator)
        complete = self.history.complete
        return replace(snapshot, official_success=snapshot.official_success and complete,
            research_diagnostics={**dict(snapshot.research_diagnostics),
                'canonical_final_success': snapshot.official_success, 'history_task_satisfied': complete,
                'history_binding_events': dict(self.history.binding_events)})
