"""Private, episode-scoped safety audit of public process requirements.

Only irreversible protection / before-placement conditions are supported here.
Canonical Q is never rewritten. No observations, actions, or repairs are added.
"""
from contextvars import ContextVar
from pathlib import Path

ACTIVE = ContextVar('l1_process_audit', default=None)
CAPABILITY = 'L1_PUBLIC_PROCESS_V1'
STATE_PROTECTION_CAPABILITY = 'L1_STATE_PROTECTION_V1'
STATE_PREDICATES = frozenset({'open', 'closed', 'toggled_on', 'toggled_off'})
CURRENT_REFERENCE_CAPABILITY = 'L1_CURRENT_REFERENCE_V1'
CURRENT_REFERENCE_FIELDS = {'binding_id','time','update','cardinality','predicate','subject_role',
                            'domain_roles','goal_predicate','goal_subject_role'}


def validate_current_references(rules, roles, protected):
    if not isinstance(rules, list) or not rules:
        raise ValueError('current reference contracts must be a nonempty list')
    names=set(); targets=set()
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != CURRENT_REFERENCE_FIELDS:
            raise ValueError('unsupported current reference fields')
        if (not isinstance(rule['binding_id'], str) or not rule['binding_id'] or rule['binding_id'] in names
                or rule['goal_subject_role'] in targets):
            raise ValueError('duplicate current reference identity or target')
        names.add(rule['binding_id']);targets.add(rule['goal_subject_role'])
        if (rule['time']!='CURRENT_READBACK_BEFORE_TARGET_PLACEMENT' or rule['update']!='CURRENT'
                or rule['cardinality']!='EXACTLY_ONE'):
            raise ValueError('unsupported current reference timing')
        binary={'ontop','inside','under','nextto','next_to'}
        domain=rule['domain_roles']
        if (rule['predicate'] not in binary or rule['goal_predicate'] not in binary
                or rule['subject_role'] not in roles or rule['subject_role'] not in protected
                or rule['goal_subject_role'] not in roles or rule['goal_subject_role']==rule['subject_role']
                or not isinstance(domain,list) or not domain or len(domain)!=len(set(domain))
                or any(r not in roles for r in domain) or rule['subject_role'] in domain or rule['goal_subject_role'] in domain):
            raise ValueError('invalid protected current reference domain')


def validate_current_reference_formula(branch, formula):
    rules=branch.get('process_requirements', {}).get('current_reference_contracts', [])
    if not rules:return
    if not branch.get('full_goal_acceptance') or formula is None:
        raise ValueError('current references require complete goal acceptance')
    from .bounded_full_goal_acceptance import bounded_dnf
    def frozen(node):
        if isinstance(node,(list,tuple)):
            return tuple(frozen(x) for x in node)
        return 'nextto' if node=='next_to' else node
    options=bounded_dnf(frozen(formula))
    for rule in rules:
        predicate='nextto' if rule['predicate']=='next_to' else rule['predicate']
        goal_predicate='nextto' if rule['goal_predicate']=='next_to' else rule['goal_predicate']
        initial=[a for a in branch['initial_assertions'] if a['subject_role']==rule['subject_role']
                 and a['predicate'].replace('next_to','nextto')==predicate and a.get('object_role') in rule['domain_roles']
                 and a.get('desired') is True]
        if len(initial)!=1:raise ValueError('current-reference cue needs one declared initial relation')
        for option in options:
            valid=False
            for destination in rule['domain_roles']:
                required={(predicate,rule['subject_role'],destination),(goal_predicate,rule['goal_subject_role'],destination)}
                required.update(('not',(predicate,rule['subject_role'],other)) for other in rule['domain_roles'] if other!=destination)
                valid |= required <= option
            if not valid:raise ValueError('complete formula lost current-reference target or uniqueness')


def validate_process_requirements(value, roles):
    required = {'no_manipulation_roles', 'before_placement'}
    if not isinstance(value, dict) or not required <= set(value) or set(value) - required - {'state_protection_roles','current_reference_contracts','process_program','history_program'}:
        raise ValueError('unsupported public process requirements')
    protected=value['no_manipulation_roles']; ordered=value['before_placement']
    if 'history_program' in value:
        from .l1_history_bindings import validate_history_program
        validate_history_program(value['history_program'], roles)
        if ordered or value.get('current_reference_contracts') or value.get('process_program'):
            raise ValueError('history composition requires guards inside the history program')
    if 'process_program' in value:
        from .l1_am_process_programs import validate_runtime_program
        validate_runtime_program(value['process_program'], roles)
        if ordered or value.get('current_reference_contracts'):
            raise ValueError('milestone composition with placement/current-reference guards is not supported')
    if not isinstance(protected,list) or len(set(protected))!=len(protected) or any(r not in roles for r in protected):
        raise ValueError('unbound protection role')
    states = value.get('state_protection_roles', [])
    if not isinstance(states, list) or len(set(states)) != len(states) or any(r not in roles for r in states):
        raise ValueError('unbound state protection role')
    if not isinstance(ordered,list):raise ValueError('invalid before-placement conditions')
    if 'current_reference_contracts' in value:
        validate_current_references(value['current_reference_contracts'],roles,protected)
        if ordered: raise ValueError('current reference and before-placement composition not supported')
    for rule in ordered:
        if not isinstance(rule,dict) or set(rule)!= {'subject_role','required_relations'} or rule['subject_role'] not in roles:
            raise ValueError('unbound before-placement subject')
        if not isinstance(rule['required_relations'],list) or not rule['required_relations']:
            raise ValueError('missing before-placement prerequisites')
        for atom in rule['required_relations']:
            if (set(atom)-{'predicate','subject_role','object_role','desired'} or atom.get('desired') is not True
                or atom.get('subject_role') not in roles or atom.get('predicate') not in {'inside','ontop','under','nextto','next_to','open','closed','toggled_on','toggled_off'}):
                raise ValueError('unsupported process relation')
            if atom['predicate'] in {'inside','ontop','under','nextto','next_to'} and atom.get('object_role') not in roles:
                raise ValueError('unbound process reference')
    if not protected and not ordered and not states and 'process_program' not in value and 'history_program' not in value:raise ValueError('empty process contract')


class ProcessAudit:
    def __init__(self, requirements, *, role_object, recipe_truth, protected_relations, directory, placement_targets=None,
                 protected_state_relations=()):
        self.requirements=requirements
        self.role_object=role_object
        self.recipe_truth=recipe_truth
        self.protected_relations=protected_relations
        self.protected_state_relations=protected_state_relations
        self.placement_targets=placement_targets
        self.directory=Path(directory)
        self.events=0;self.frames=0;self.violations=[];self.errors=[];self.placements=set();self.current_action_index=None
        self.precedence_relations = ()
        self.satisfied_prerequisites = lambda: ()
        self.current_reference_placements={}
        self.history = None
        if 'history_program' in requirements:
            from .l1_history_bindings import HistoryProgress
            self.history = HistoryProgress(requirements['history_program'],
                lambda atom: bool(self.recipe_truth(atom, self.role_object)))
        self.milestones = None
        if 'process_program' in requirements:
            from .l1_am_process_programs import MilestoneProgress
            self.milestones = MilestoneProgress(requirements['process_program'],
                lambda atom: bool(self.recipe_truth(atom, self.role_object)))
            self.milestones.observe(self.milestones.read(), 0)

    def append(self, name, data):
        from .oracle_progress_journal import json_safe
        import json
        path=self.directory/'private'/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as stream:stream.write(json.dumps(json_safe(data),ensure_ascii=False)+'\n')

    def snapshot(self, runtime):
        try:
            held=runtime.symbolic_primitives._get_obj_in_hand()
            current=self.requirements.get('current_reference_contracts', [])
            roles=set(self.requirements['no_manipulation_roles']) | {r['subject_role'] for r in self.requirements['before_placement']} | {r['goal_subject_role'] for r in current}
            if self.history is not None: roles |= self.history.tracked_roles
            return {'held_roles':[r for r in sorted(roles) if self.role_object(r) is held],
                    'history': self.history.snapshot() if self.history is not None else {},
                    'milestones': self.milestones.read() if self.milestones is not None else {},
                    'prerequisites':[all(self.recipe_truth(a,self.role_object) for a in rule['required_relations'])
                                     for rule in self.requirements['before_placement']],
                    'placement_goals':([bool(self.recipe_truth(a,self.role_object)) for a in self.placement_targets]
                                       if self.placement_targets is not None else [True]*len(self.requirements['before_placement'])),
                    'protected_relations':[bool(self.recipe_truth(a,self.role_object)) for a in self.protected_relations],
                    'protected_states':[bool(self.recipe_truth(a,self.role_object)) for a in self.protected_state_relations],
                    'current_references':[dict(
                        source_matches=[role for role in rule['domain_roles'] if self.recipe_truth(dict(predicate=rule['predicate'],subject_role=rule['subject_role'],object_role=role,desired=True),self.role_object)],
                        target_matches=[role for role in rule['domain_roles'] if self.recipe_truth(dict(predicate=rule['goal_predicate'],subject_role=rule['goal_subject_role'],object_role=role,desired=True),self.role_object)]) for rule in current]}
        except Exception as exc:
            self.errors.append(type(exc).__name__+': '+str(exc));return None

    def check_oracle_placement(self, runtime, operation, args):
        """Oracle-only pre-dispatch guard; public Action and audit stay unchanged."""
        predicate = {'place_inside': 'inside', 'place_on_top': 'ontop'}.get(operation)
        if predicate is None or not self.placement_targets or not isinstance(args, dict):
            return
        snapshot = getattr(runtime.store, 'active_snapshot', None)
        binding = None if snapshot is None else snapshot.bindings.get(args.get('marker'))
        if binding is None or binding.binding_kind not in {'inside_cell', 'surface_cell'}:
            return  # Object/reference publication is not placement execution.
        held = runtime.symbolic_primitives._get_obj_in_hand()
        for rule, target in zip(self.requirements['before_placement'], self.placement_targets):
            if (target['predicate'] == predicate and self.role_object(rule['subject_role']) is held
                    and binding.object_name == self.role_object(target['object_role']).name
                    and not all(self.recipe_truth(atom, self.role_object) for atom in rule['required_relations'])):
                raise RuntimeError('public process prerequisite not satisfied before final placement')

    def observe(self, *, action, before, after, status, action_index):
        self.events+=1
        if before is None or after is None:return
        if self.history is not None:
            self.history.observe(action, before, after, status, self.events)
            if self.history.errors:
                message='history release lacks verified successful execution'
                if message not in self.errors:self.errors.append(message)
        if self.milestones is not None and status == 'completed':
            self.milestones.observe(after['milestones'], self.events)
        protected=set(self.requirements['no_manipulation_roles'])
        if not all(before.get('protected_states', [])) or not all(after.get('protected_states', [])):
            self.violations.append({'event_id':self.events,'kind':'PROTECTED_STATE_CHANGED'})
        if protected.intersection(before['held_roles']+after['held_roles']) or not all(after['protected_relations']):
            self.violations.append({'event_id':self.events,'kind':'PROTECTED_OBJECT_MANIPULATED_OR_RELATION_LOST'})
        selected=protected.intersection(before.get('selected_roles',[]))
        if action=='pick' and selected:
            if status=='completed':
                self.violations.append({'event_id':self.events,'kind':'PROTECTED_OBJECT_PICK_COMMITTED'})
            elif status not in {'needs_marker','marker_required','awaiting_selection'} and not self.violations:
                self.errors.append('failed protected-object pick lacks a verified no-mutation receipt')
        if action in {'place_on_top','place_inside','place_under','place_next_to','place_on_floor'}:
            for i,rule in enumerate(self.requirements.get('current_reference_contracts', [])):
                subject=rule['goal_subject_role']
                if subject not in before['held_roles']: continue
                if status=='completed' and subject not in after['held_roles']:
                    source=before['current_references'][i]['source_matches']
                    destination=after['current_references'][i]['target_matches']
                    # A later correct placement may repair an earlier wrong
                    # placement; the source does not prohibit trial and repair.
                    self.current_reference_placements[i]=dict(event_id=self.events,action_index=action_index,
                        source_matches=list(source),target_matches=list(destination),
                        valid=len(source)==1 and source[0] in destination)
                elif status not in {'needs_marker','marker_required','awaiting_selection'} and subject not in after['held_roles']:
                    self.errors.append('current-reference target released without verified successful placement')
            for i,rule in enumerate(self.requirements['before_placement']):
                subject=rule['subject_role']
                if subject not in before['held_roles'] or not after.get('placement_goals',[True]*len(self.requirements['before_placement']))[i]:continue
                if status=='completed' and subject not in after['held_roles']:
                    self.placements.add(i)
                    if not before['prerequisites'][i]:
                        self.violations.append({'event_id':self.events,'kind':'PLACEMENT_BEFORE_PUBLIC_PREREQUISITES','rule':i})
                elif status not in {'needs_marker','marker_required','awaiting_selection'} and subject not in after['held_roles']:
                    self.errors.append('placement released subject without verified completed result')
        self.append('task_process_events.jsonl',dict(event_id=self.events,action_index=action_index,action=action,
                    status=status,before=before,after=after))

    def frame(self, record):
        self.frames+=1
        self.append('task_evidence_index.jsonl',dict(frame_index=self.frames,action_index=self.current_action_index,
            completed_audited_events=self.events,
            evidence_kind='EXISTING_ORACLE_OBSERVATION_NOT_MODEL_DELIVERY_CERTIFICATE',**record))

    def report(self):
        current_ok=True
        for i,rule in enumerate(self.requirements.get('current_reference_contracts', [])):
            receipt=self.current_reference_placements.get(i)
            if not receipt or not receipt['valid']:
                current_ok=False;continue
            try:
                current_ok &= bool(self.recipe_truth(dict(predicate=rule['goal_predicate'],
                    subject_role=rule['goal_subject_role'],object_role=receipt['source_matches'][0],desired=True),self.role_object))
            except Exception as exc:
                message='current-reference final readback: '+type(exc).__name__+': '+str(exc)
                if message not in self.errors:self.errors.append(message)
                current_ok=False
        history_ok=True
        if self.history is not None:
            try: history_ok=self.history.complete
            except Exception as exc:
                message='history final readback: '+type(exc).__name__+': '+str(exc)
                if message not in self.errors:self.errors.append(message)
                history_ok=False
        operational=not self.errors
        return dict(operational=operational,public_process_satisfied=operational and self.events>0 and not self.violations
                    and len(self.placements)==len(self.requirements['before_placement']) and current_ok
                    and (self.milestones is None or self.milestones.complete) and history_ok,
                    history_bindings={} if self.history is None else dict(self.history.bindings),
                    history_binding_events={} if self.history is None else dict(self.history.binding_events),
                    history_violations=[] if self.history is None else list(self.history.violations),
                    milestones_reached={} if self.milestones is None else dict(self.milestones.reached),
                    current_reference_placements=self.current_reference_placements,
                    violations=self.violations,errors=self.errors,execution_events=self.events,
                    existing_frame_references=self.frames,extra_render_calls=0,axis_review_status='PENDING_REVIEW',
                    process_events=str(self.directory/'private/task_process_events.jsonl'),
                    observation_index=str(self.directory/'private/task_evidence_index.jsonl'))


def observe_execution(execute, **kwargs):
    audit=ACTIVE.get()
    if audit is None:return execute(**kwargs)
    before=audit.snapshot(kwargs['runtime']);status='EXCEPTION'
    if before is not None and kwargs['action'].action=='pick':
        args=getattr(kwargs['action'],'args',None)
        marker=args.get('marker') if isinstance(args,dict) else getattr(args,'marker',None)
        snapshot=getattr(getattr(kwargs['runtime'],'store',None),'active_snapshot',None)
        binding=None if snapshot is None else snapshot.bindings.get(marker)
        if binding is not None:
            token=getattr(binding,'target_token',None)
            names={getattr(binding,'object_name',None),*getattr(token,'member_names',())}
            before['selected_roles']=[r for r in audit.requirements['no_manipulation_roles']
                                      if getattr(audit.role_object(r),'name',None) in names]
    previous=audit.current_action_index;audit.current_action_index=kwargs['action_index']
    try:
        result=execute(**kwargs);status=result[0].status;return result
    finally:
        audit.observe(action=kwargs['action'].action,before=before,after=audit.snapshot(kwargs['runtime']),
                      status=status,action_index=kwargs['action_index'])
        audit.current_action_index=previous


def create_process_audit(*, task, evaluator, directory):
    branch=None if task is None else task.record.get('task_contract')
    requirements=None if branch is None else branch.get('process_requirements')
    if requirements is None:return None
    from .registered_l0_task_runtime import _role_object_for_contract, _initial_recipe_truth
    contract=task.template_contract()
    validate_process_requirements(requirements,contract['role_to_scope'])
    protected=[r for r in contract['initial_state_recipe'] if r['subject_role'] in requirements['no_manipulation_roles']]
    protected_states = protected_state_assertions(branch)
    targets=[]
    for rule in requirements['before_placement']:
        matches=[r for r in branch['initial_reference_goals'] if r['goal_subject_role']==rule['subject_role']]
        if len(matches)!=1:raise ValueError('process placement must resolve one declared final reference')
        match=matches[0];initial=branch['initial_assertions'][match['initial_assertion_index']]
        targets.append(dict(predicate=match['goal_predicate'],subject_role=rule['subject_role'],object_role=initial['object_role'],desired=True))
    audit=ProcessAudit(requirements,role_object=lambda r:_role_object_for_contract(evaluator,contract,r),
        recipe_truth=_initial_recipe_truth,protected_relations=protected,placement_targets=targets,
        protected_state_relations=protected_states,
        directory=directory)
    def leaf(atom):
        predicate = atom['predicate']
        body = [{'ontop': 'ontop', 'closed': 'open', 'toggled_off': 'toggled_on'}.get(predicate, predicate),
                contract['role_to_scope'][atom['subject_role']]]
        if atom.get('object_role') is not None:
            body.append(contract['role_to_scope'][atom['object_role']])
        return ['not', body] if predicate in {'closed', 'toggled_off'} else body
    audit.precedence_relations = tuple(
        (leaf(atom), leaf(target))
        for rule, target in zip(requirements['before_placement'], targets)
        for atom in rule['required_relations'])
    audit.satisfied_prerequisites = lambda: tuple(
        leaf(atom) for rule in requirements['before_placement'] for atom in rule['required_relations']
        if _initial_recipe_truth(atom, audit.role_object))
    return audit


def run_process_checked_phase(execute, *, task=None, **kwargs):
    audit=create_process_audit(task=task, evaluator=kwargs['evaluator'],
        directory=kwargs['run_dir']/'strategies'/kwargs['phase_name'])
    if audit is None:return execute(**kwargs)
    from .resident_scene_batch import write_json_atomic
    token=ACTIVE.set(audit)
    try:
        result=execute(**kwargs)
    finally:
        ACTIVE.reset(token)
        write_json_atomic(audit.directory/'private/task_process_result.json',audit.report())
    report=audit.report()
    result=dict(result,canonical_execution_status=result.get('status'),public_process_report=report,
                task_success=result.get('status')=='PASS' and report['public_process_satisfied'])
    if not report['operational']:
        result.update(status='FAIL_PROCESS_VERIFIER',failure_class='VERIFIER_ERROR')
    elif report['violations'] or (result.get('status') == 'PASS' and not report['public_process_satisfied']):
        result.update(status='FAIL_PUBLIC_PROCESS',failure_class='PUBLIC_PROCESS_REQUIREMENT_FAILED')
    return result


def protected_state_assertions(branch):
    roles = branch['process_requirements'].get('state_protection_roles', [])
    assertions = [a for a in branch.get('initial_assertions', [])
                  if a.get('subject_role') in roles and a.get('predicate') in STATE_PREDICATES
                  and a.get('desired') is True and a.get('object_role') is None]
    if set(roles) != {a['subject_role'] for a in assertions}:
        raise ValueError('state protection requires declared initial device states')
    return assertions
