"""P12 binding, native-volume certificates and mandatory physical post-review.

Native Q remains the existing OfficialPredicateVerifier's output. The complete
public requirement is an additional acceptance gate, never an alternate native Q.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np

from .physical_template_review import capture_live_facts, fact_key, review_template, goal_truth
from .on_top_support_layers import quaternion_to_matrix_xyzw
from .spatial_symbolic_actions import object_pose, canonical_relation, object_aabb

CAPABILITY = 'P12_PHYSICAL_TEMPLATE_V1'
ACTIVE = ContextVar('p12_physical_template', default=None)
NATIVE = frozenset(('ontop','inside','open','closed'))
UNARY = frozenset(('upright','released','toppled','supported_or_held'))
BINARY = frozenset(('fully_inside','rolled_into','collision_caused','ontop_link','in_door_sweep'))


def validate_template(value, roles):
    if not isinstance(value,dict) or value.get('schema_version') != 'physical-template-runtime/v1':
        raise ValueError('unsupported physical template contract')
    task = value['task']
    if set(task['roles']) != set(roles) - {'_floor_anchor','staging_anchor'}:
        raise ValueError('physical template role membership mismatch')
    all_roles=set(task['roles'])|set(task.get('link_roles',{}))
    def validate(node):
        if not isinstance(node,list) or not node:raise ValueError('empty physical goal')
        op=node[0]
        if op in ('and','or'):
            if len(node)<2:raise ValueError('empty physical branch')
            for child in node[1:]:validate(child)
        elif op=='not':
            if len(node)!=2 or not isinstance(node[1],list) or node[1][0] in ('and','or','not'):
                raise ValueError('physical negation requires literal')
            validate(node[1])
        else:
            arity=1 if op in UNARY or op in ('open','closed') else 2
            if op not in NATIVE|UNARY|BINARY or len(node)!=arity+1 or any(r not in all_roles for r in node[1:]):
                raise ValueError('unsupported physical predicate or binding')
    validate(task['goal'])
    for atom in task['process_invariants']:validate(atom)
    if value.get('native_q_scope')!='NATIVE_DIAGNOSTIC_REQUIRES_TEMPLATE_ACCEPTANCE':
        raise ValueError('physical qualification cannot be omitted')
    limit=value.get('upright_limit_deg')
    if type(limit) not in (float,int) or not math.isfinite(limit) or not 0<limit<90:
        raise ValueError('invalid declared upright tolerance')
    for role,axis in value.get('up_axes',{}).items():
        vector=np.asarray(axis,dtype=float)
        if role not in roles or vector.shape!=(3,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector),1):
            raise ValueError('invalid declared upright axis')
    for step in value['oracle_prefix']:
        if step['action'] not in ('place_on_top','place_inside','open','close') or step['subject'] not in roles:
            raise ValueError('unsupported physical Oracle prefix')
        if step['action'].startswith('place_') and step.get('destination') not in roles:
            raise ValueError('unbound physical Oracle destination')


def complete_containment(subject, container):
    """Certify all rigid collision boundary vertices within one convex meta mesh.

    Each native meta mesh is convex per OG GeomPrim.check_points_in_volume.
    If the subject spans several convex cells, report UNKNOWN instead of falsely
    certifying the union from vertex tests alone. The hull encloses every handle.
    """
    from omnigibson.macros import macros
    import torch
    points=subject.collision_points_world
    if points.ndim!=2 or points.shape[0]<4 or points.shape[1]!=3 or not torch.isfinite(points).all():
        raise ValueError('complete rigid collision geometry unavailable')
    cells=[mesh for link in container.links.values()
           if link.is_meta_link and link.meta_link_type in macros.object_states.contains.CONTAINER_META_LINK_TYPES
           for mesh in link.visual_meshes.values()]
    if not cells:raise ValueError('no native fillable volume')
    masks=[mesh.check_points_in_volume(points) for mesh in cells]
    if any(bool(mask.all()) for mask in masks):
        value=True
    elif not bool(torch.stack(masks).any(dim=0).all()):
        value=False
    else:
        return {'status':'UNKNOWN','value':None,'source':'NATIVE_CONVEX_FILLABLE',
                'reason':'subject_spans_multiple_cells_whole_solid_union_not_certified'}
    return {'status':'MEASURED','value':value,'source':'ALL_COLLISION_BOUNDARY_POINTS_IN_NATIVE_CONVEX_FILLABLE',
            'point_count':int(points.shape[0]),'fillable_cell_count':len(cells),
            'qualification_scope':'NATIVE_FILLABLE_ANNOTATION_NOT_EXTERNAL_REAL_WORLD_GEOMETRY'}


TRANSPORT_LIMITATION='LOGICAL_PAYLOAD_TRANSPORT_IS_NOT_FREE_PHYSICAL_PAYLOAD_EVIDENCE'


def release_certifies_group_transport(report,closure_object_names):
    """True only when a complete Physical OnTop release matches the exact group.

    Kinematic logical transport is never itself dynamic evidence: a carried group
    is certified only by an actual place_on_top release of that same closure with
    complete evidence. Requires exact closure-name identity (no missing, duplicate
    or unknown bodies), full released coverage, a committed and observed release
    with no error, and a readback showing no held/snapshot/official attachment.
    """
    if not isinstance(report,dict):return False
    if report.get('operation')!='place_on_top':return False
    if report.get('error'):return False
    for key in ('release_applied','physical_consequence_committed','placement_committed','observation_complete'):
        if report.get(key) is not True:return False
    closure=report.get('closure_object_names');released=report.get('released_object_names')
    if not isinstance(closure,list) or not isinstance(released,list):return False
    for names in (closure_object_names,closure,released):
        if not names or any(not isinstance(name,str) or not name for name in names):return False
        if len(names)!=len(set(names)):return False
    if sorted(closure)!=sorted(closure_object_names):return False
    if set(closure_object_names)!=set(released):return False
    if type(report.get('physics_steps')) is not int or report['physics_steps']<=0:return False
    readback=report.get('release_readback')
    if not isinstance(readback,dict):return False
    for key in ('held','snapshot_present','official_constraint','official_held'):
        if readback.get(key) is not False:return False
    return True


def transport_obligations_unresolved(obligations):
    """Any pending shared-closure carry keeps qualification UNKNOWN."""
    return any(not isinstance(item,dict) or item.get('status')!='RESOLVED' for item in obligations)


def qualification_override(violations,errors,obligations):
    """Preserve FAIL-on-violation and UNKNOWN-on-error; add unresolved transport."""
    if violations:return 'FAIL',dict(process_violations=violations)
    if errors or transport_obligations_unresolved(obligations):return 'UNKNOWN',{}
    return None,{}


class PhysicalTemplateAudit:
    def __init__(self, contract, role_object, directory, evaluator):
        self.contract=contract; self.task=contract['task'];self.role_object=role_object
        self.directory=Path(directory);self.evaluator=evaluator;self.runtime=None
        self.samples=[];self.violations=[];self.errors=[];self.steps=0
        self.contacts=set();self.contact_steps={};self.toppled_steps={};self.motion={}
        self.manipulated=set();self.releases=[]
        self.initial_poses={r:object_pose(role_object(r)) for r in self.task['roles']}
        self.axes={r:dict(axis_status='CALIBRATED',source='DECLARED_INITIAL_UP_AXIS_AT_CONSTRUCTION',
                    up_local=contract['up_axes'][r],max_tilt_deg=contract['upright_limit_deg'])
                   for r in contract['up_axes']}
        self.coverage=False;self.active_release=None;self.contact_complete=True
        self.transport_obligations=[];self.release_reports=[]

    def bind_runtime(self, runtime):self.runtime=runtime

    def held_readback(self,obj):
        if self.runtime is None:
            raise RuntimeError('action release owner is not bound')
        p=self.runtime.symbolic_primitives
        held=p._get_obj_in_hand()
        snap=p._logical_inventory_snapshot
        robot=p.robot
        if not isinstance(robot._ag_obj_in_hand,dict) or not isinstance(robot._ag_obj_constraints,dict):
            raise RuntimeError('official holding owner unavailable')
        if any(constraint is not None and robot._ag_obj_in_hand.get(arm) is None
               for arm,constraint in robot._ag_obj_constraints.items()):
            raise RuntimeError('official attachment exists without held-object identity')
        if snap is None and getattr(p,'_logical_inventory_visual_projection_active',False):
            raise RuntimeError('visual projection active without a carry snapshot')
        official=any(v is obj for v in robot._ag_obj_in_hand.values())
        return dict(logical_held=held is obj,
            logical_projection_controlled=bool(snap is not None and any(v is obj for v in snap.object_refs)),
            official_held=official,attachment_present=any(value is obj and robot._ag_obj_constraints.get(arm) is not None
                for arm,value in robot._ag_obj_in_hand.items()),
            source='ACTION20_LOGICAL_AND_OFFICIAL_RELEASE_OWNERS')

    def certificate(self, atom):
        kind=atom[0];obj=self.role_object(atom[1])
        if kind=='fully_inside':return complete_containment(obj,self.role_object(atom[2]))
        if kind=='supported_or_held':
            held=self.held_readback(obj)['logical_held']
            others=self.evaluator.env.scene.objects
            value=held or any(other is not obj and canonical_relation(obj,'OnTop',other)
                              for other in others if getattr(other,'category',None) not in ('floors','floor'))
            return dict(status='MEASURED',value=bool(value),source='NATIVE_SUPPORT_OR_ACTION_HOLD')
        if kind=='toppled':
            current=object_pose(obj)[1]
            original=self.initial_poses[atom[1]][1]
            up=quaternion_to_matrix_xyzw(current)@quaternion_to_matrix_xyzw(original).T@np.array([0.,0.,1.])
            return dict(status='MEASURED',value=float(up[2]) < math.cos(math.radians(60)),
                        source='RELATIVE_TO_DECLARED_INITIAL_UP_AXIS',up_z=float(up[2]))
        if kind=='collision_caused':
            if not self.contact_complete:return dict(status='UNKNOWN',value=None,source=None,reason='incomplete contact trace')
            value=(atom[1],atom[2]) in self.contacts and any(r['subject']==atom[1] for r in self.releases)
            value=value and atom[2] not in self.manipulated
            value=value and atom[2] in self.toppled_steps and self.contact_steps.get((atom[1],atom[2]),float('inf')) <= self.toppled_steps[atom[2]]
            return dict(status='MEASURED',value=value,source='SUBSTEP_CONTACT_AFTER_RELEASE_WITHOUT_DIRECT_TARGET_MANIPULATION')
        if kind=='rolled_into':
            if not self.contact_complete:return dict(status='UNKNOWN',value=None,source=None,reason='incomplete contact trace')
            releases=[r for r in self.releases if r['subject']==atom[1] and r['target'].startswith('ramp_')]
            inside=canonical_relation(obj,'Inside',self.role_object(atom[2]))
            # Require measured travel and rotation since a release on a ramp.
            value=False
            for release in releases:
                pos,quat=object_pose(obj)
                travel=float(np.linalg.norm(pos[:2]-np.asarray(release['position'])[:2]))
                angle=release.get('rotation_accumulated_rad',0.)
                value |= bool(inside and travel>=0.05 and angle>=math.radians(30)
                              and (atom[1],release['target']) in self.contacts)
            return dict(status='MEASURED',value=value,source='RAMP_RELEASE_CONTACT_TRAVEL_ROTATION_AND_NATIVE_INSIDE',
                        minimum_travel_m=0.05,minimum_rotation_deg=30)
        return {'status':'UNKNOWN','value':None,'source':None}

    def facts(self):
        atoms=set()
        def collect(node):
            if node[0] in ('and','or'):
                for child in node[1:]:collect(child)
            elif node[0]=='not':collect(node[1])
            else:atoms.add(tuple(node))
        collect(self.task['goal'])
        for rule in self.task['process_invariants']:collect(rule)
        certs={}
        for atom in atoms:
            if atom[0] in BINARY|{'toppled','supported_or_held'}:
                try:certs[fact_key(atom)]=self.certificate(atom)
                except Exception as exc:certs[fact_key(atom)]=dict(status='UNKNOWN',value=None,source=None,error=str(exc))
        return capture_live_facts(self.task,role_object=self.role_object,holding_readback=self.held_readback,
                                  upright_axes=self.axes,certificates=certs)

    def on_release(self, subject, target):
        def role(obj):return next((r for r in self.task['roles'] if self.role_object(r) is obj),None)
        a,b=role(subject),role(target)
        if a is None:return
        self.manipulated.add(a)
        pos,quat=object_pose(subject)
        self.releases.append(dict(subject=a,target=b or '',position=pos.tolist(),orientation=quat.tolist(),step=self.steps,rotation_accumulated_rad=0.))
        self.motion[a]=quat.copy()
        self.active_release=a

    def end_release(self):self.active_release=None

    def record_transport_pick(self,snapshot,status):
        """Record a completed shared-closure carry as a PENDING release obligation.

        The logical-transport limitation is retained (never silently deleted and
        never treated as dynamic evidence); qualification stays UNKNOWN until a
        matching group Physical OnTop release with complete evidence resolves it.
        """
        if status!='completed':return
        objects=tuple(getattr(snapshot,'object_refs',()))
        if len(objects)<2:return
        names=[getattr(obj,'name',None) for obj in objects]
        # Descendants were also moved by logical transport; they cannot later be
        # presented as untouched causal targets.
        self.manipulated.update(role for role in self.task['roles']
                                if any(self.role_object(role) is obj for obj in objects))
        edges=[dict(kind=getattr(edge,'kind',None),
                    child_name=getattr(getattr(edge,'child',None),'name',None),
                    parent_name=getattr(getattr(edge,'parent',None),'name',None))
               for edge in getattr(snapshot,'relation_edges',())]
        self.transport_obligations.append(dict(
            status='PENDING',limitation=TRANSPORT_LIMITATION,
            closure_object_names=names,closure_relation_edges=edges,
            baseline_report_count=len(self.release_reports),resolved_by=None))

    def resolve_transports(self):
        """Resolve pending obligations only from reports appended after the pick.

        Chronology matters: a release report written before the pick cannot
        certify it, so only reports at or after the recorded baseline are eligible.
        """
        for obligation in self.transport_obligations:
            if obligation.get('status')=='RESOLVED':continue
            names=obligation.get('closure_object_names') or []
            for index in range(obligation.get('baseline_report_count',0),len(self.release_reports)):
                report=self.release_reports[index]
                if release_certifies_group_transport(report,names):
                    obligation['status']='RESOLVED'
                    obligation['resolved_by']=dict(
                        report_index=index,physics_steps=report.get('physics_steps'),
                        released_object_names=list(report.get('released_object_names') or []))
                    break

    def on_physics(self, api):
        """Read fresh poses and the just-collected tensor buffer, without flushing it."""
        self.steps+=1
        if self.steps>240000:
            self.coverage=False
            if 'trace_budget_exceeded' not in self.errors:self.errors.append('trace_budget_exceeded')
            return
        try:
            for role,previous in list(self.motion.items()):
                quat=object_pose(self.role_object(role))[1]
                angle=2*math.acos(float(np.clip(abs(np.dot(quat,previous)),0,1)))
                next(r for r in reversed(self.releases) if r['subject']==role)['rotation_accumulated_rad']+=angle
                self.motion[role]=quat.copy()
            for role in self.task['roles']:
                current=object_pose(self.role_object(role))[1]
                original=self.initial_poses[role][1]
                up=quaternion_to_matrix_xyzw(current)@quaternion_to_matrix_xyzw(original).T@np.array([0.,0.,1.])
                if float(up[2])<.5:self.toppled_steps.setdefault(role,self.steps)
            if self.runtime is not None:
                held=self.runtime.symbolic_primitives._get_obj_in_hand()
                self.manipulated.update(r for r in self.task['roles'] if self.role_object(r) is held)
            idx=self.evaluator.env.scene.idx
            pending=api._PENDING_IMPULSES.get(idx,[])
            if not pending and self.releases:
                self.contact_complete=False
            if pending:
                matrix=pending[-1]
                for release in self.releases:
                    a=release['subject'];obj=self.role_object(a)
                    rows=api.get_contact_row_indices(idx,[obj])
                    for b in self.task['roles']:
                        if a==b:continue
                        cols=api.get_contact_col_indices(idx,[self.role_object(b)])
                        if rows.numel() and cols.numel():
                            import torch
                            forces=matrix[rows,:][:,cols]
                            if not bool(torch.isfinite(forces).all()):
                                self.contact_complete=False
                            elif bool((forces!=0).any()):
                                self.contacts.add((a,b));self.contact_steps.setdefault((a,b),self.steps)
            facts={}
            # Pose protections are read every substep; contact-dependent final facts
            # are read only after the simulator's own contact-cache update.
            for rule in self.task['process_invariants']:
                if rule[0]=='upright':
                    local=dict(self.task,goal=rule,process_invariants=[])
                    facts.update(capture_live_facts(local,role_object=self.role_object,
                        holding_readback=self.held_readback,upright_axes=self.axes))
                elif rule[0]=='supported_or_held':
                    subject=self.role_object(rule[1])
                    held=self.held_readback(subject)['logical_held']
                    controlled=rule[1]==self.active_release
                    if held or controlled:
                        value=True
                    elif pending:
                        rows=api.get_contact_row_indices(idx,[subject])
                        supports=[self.role_object(r) for r in self.task['roles'] if r!=rule[1]
                                  and getattr(self.role_object(r),'category','') not in ('floor','floors')]
                        cols=api.get_contact_col_indices(idx,supports)
                        # For sleeping contacts use the simulator's own cached value;
                        # awake body pairs use the just-collected impulse buffer.
                        value=bool(rows.numel() and cols.numel() and (
                            (matrix[rows,:][:,cols]!=0).any() or
                            (subject.is_asleep and api._CURRENT_CONTACT_MATRIX[idx][rows,:][:,cols].any())))
                    else:
                        raise RuntimeError('missing substep contact buffer for support protection')
                    facts[fact_key(rule)]=dict(status='MEASURED',value=value,
                        source='SUBSTEP_CONTACT_HOLD_OR_ACTIVE_CONTROLLED_RELEASE',controlled_release=controlled)
                else:
                    facts[fact_key(rule)]={'status':'UNKNOWN','value':None,'source':None}
            sample=dict(physics_step=self.steps,facts=facts)
            if facts:self.append('physical_process_samples.jsonl',sample)
            for rule in self.task['process_invariants']:
                from .physical_template_review import truth
                if truth(rule,facts) is False:self.violations.append(dict(physics_step=self.steps,rule=rule))
                if truth(rule,facts) is None and rule not in self.errors:self.errors.append(rule)
        except Exception as exc:
            if len(self.errors)<20:self.errors.append(type(exc).__name__+': '+str(exc))

    def append(self,name,value):
        if name=='physical_release_lifecycle.jsonl' and isinstance(value,dict):
            self.release_reports.append(value)
        p=self.directory/'private'/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('a') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')

    @contextmanager
    def observe(self):
        # A resident human episode keeps this context around an Action20 turn;
        # finalization may then sample physics inside that turn.  Reinstalling
        # the global contact hook would double-count each substep.
        if ACTIVE.get() is self:
            yield
            return
        from omnigibson.utils.usd_utils import RigidContactAPI
        original=RigidContactAPI.add_contacts_from_physics_step
        def capture():
            original()
            self.on_physics(RigidContactAPI)
        RigidContactAPI.add_contacts_from_physics_step=capture
        token=ACTIVE.set(self);self.coverage=True
        try:yield
        finally:
            ACTIVE.reset(token)
            RigidContactAPI.add_contacts_from_physics_step=original


class PhysicalCompletionVerifier:
    """Use all legal physical goals for model decisions; retain native Q."""

    def __init__(self, canonical, audit):
        self.canonical, self.audit = canonical, audit

    def evaluate(self, episode, evaluator):
        snapshot = self.canonical.evaluate(episode, evaluator)
        complete = goal_truth(self.audit.task['goal'], self.audit.facts()) is True
        return replace(snapshot, official_success=bool(snapshot.ok and complete),
            research_diagnostics={**dict(snapshot.research_diagnostics),
                'native_proxy_success': snapshot.official_success,
                'physical_goal_current': complete})


def sample_physical_final_window(audit):
    """Capture the one required ten-sample Physical settling window."""
    import omnigibson as og

    for index in range(10):
        if index:
            og.sim.step_physics()
        sample = dict(physics_step=audit.steps, facts=audit.facts())
        audit.samples.append(sample)
        audit.append('physical_final_samples.jsonl', sample)


def finalize_physical_template(*, audit, verifier, evaluator, template,
                               initial_facts, initial_complete_goal, initial_native_q):
    """Review exactly the collected live Physical evidence and persist it once."""
    from .resident_scene_batch import write_json_atomic

    audit.resolve_transports()
    final = verifier.evaluate(None, evaluator)
    review = review_template(
        audit.task,
        audit.samples,
        native_q=final.official_q,
        initial_goal_false=(None if initial_complete_goal is None else not initial_complete_goal),
        process_coverage_complete=audit.coverage and not audit.errors,
    )
    override_status, override_extra = qualification_override(
        audit.violations, audit.errors, audit.transport_obligations
    )
    if override_status is not None:
        review.update(qualification_status=override_status, **override_extra)
    review.update(
        initial_facts=initial_facts,
        initial_native_q=initial_native_q,
        errors=audit.errors,
        physics_substeps_observed=audit.steps,
        releases=audit.releases,
        contact_pairs=sorted(audit.contacts),
        native_q_scope=template['native_q_scope'],
        transport_obligations=audit.transport_obligations,
    )
    write_json_atomic(audit.directory/'private/physical_template_result.json', review)
    return final, review


def write_physical_template_failure(audit, exc):
    from .resident_scene_batch import write_json_atomic

    write_json_atomic(audit.directory/'private/physical_template_result.json',
        dict(qualification_status='UNKNOWN',error=type(exc).__name__+': '+str(exc),
             process_violations=audit.violations,releases=audit.releases,
             physics_substeps_observed=audit.steps,
             transport_obligations=audit.transport_obligations))


def _mp_process_gate(result):
    """Require M's settled canonical process report without rejecting a legal P OR branch."""
    report = result.get('public_process_report') if isinstance(result, dict) else None
    if not (isinstance(report, dict)
            and report.get('operational') is True
            and report.get('public_process_satisfied') is True
            and type(report.get('execution_events')) is int
            and report.get('execution_events') > 0
            and not report.get('violations') and not report.get('errors')):
        return False
    status = result.get('status')
    return status == 'PASS' or (
        status == 'FAIL_POSTCONDITION'
        and result.get('error') is None
        and result.get('failure_class') is None
    )


def run_checked_phase(execute, *, task=None, **kwargs):
    value=None if task is None else task.record.get('physical_template')
    if value is None:return execute(**kwargs)
    if not kwargs.get('placement_physical_consequences_enabled'):
        raise ValueError('P12 requires explicit Physical execution')
    from .registered_l0_task_runtime import _role_object_for_contract
    from .official_predicate_verifier import OfficialPredicateVerifier
    contract=task.template_contract();validate_template(value,contract['role_to_scope'])
    audit=PhysicalTemplateAudit(value,lambda r:_role_object_for_contract(kwargs['evaluator'],contract,r),
        kwargs['run_dir']/'strategies'/kwargs['phase_name'],kwargs['evaluator'])
    verifier=OfficialPredicateVerifier(require_single_option=False,predicate_mode='official')
    initial=verifier.evaluate(None,kwargs['evaluator'])
    initial_facts=audit.facts()
    initial_complete_goal=goal_truth(audit.task['goal'],initial_facts)
    finalized = False
    def finalize_world():
        nonlocal finalized
        if finalized:
            return
        # Keep the process audit alive through the final physics advances.
        # Its final goal readback must follow these samples, without replaying
        # any Agent events or changing historical bindings.
        sample_physical_final_window(audit)
        finalized = True
    with audit.observe():
        try:
            result=execute(finalize_world=finalize_world, **kwargs)
            finalize_world()
        except BaseException as exc:
            write_physical_template_failure(audit, exc)
            raise
    final,review=finalize_physical_template(
        audit=audit,verifier=verifier,evaluator=kwargs['evaluator'],template=value,
        initial_facts=initial_facts,initial_complete_goal=initial_complete_goal,
        initial_native_q=initial.official_q)
    # The native proxy steers the scripted witness, not the legal answer set.
    # A different complete public solution must not fail merely for missing that proxy.
    physical_passed = final.ok and review['qualification_status']=='PASS'
    composition = task.record.get('composition_contract') or {}
    is_mp = composition.get('axes') in (['M', 'P'], ['M', 'V', 'P'])
    process_passed = _mp_process_gate(result) if is_mp else True
    passed = bool(physical_passed and process_passed)
    inner_status = result.get('status')
    if passed:
        status = 'PASS'
        failure_class = None
    elif is_mp and not process_passed and inner_status in {
        'FAIL_PROCESS_VERIFIER', 'FAIL_PUBLIC_PROCESS', 'FAIL_EXCEPTION',
        'FAIL_EARLY_STOP', 'FAIL_BUDGET_EXHAUSTED', 'FAIL_V_PROTOCOL', 'FAIL_PRIVILEGED_VALIDATION',
    }:
        status = inner_status
        failure_class = result.get('failure_class') or 'PUBLIC_PROCESS_REQUIREMENT_FAILED'
    else:
        status = 'FAIL_PHYSICAL_TEMPLATE'
        failure_class = 'PHYSICAL_TEMPLATE_REQUIREMENT_NOT_CERTIFIED'
    return dict(result,status=status,task_success=passed,
        official_q=final.official_q,official_success=final.official_success,
        native_proxy_success=final.official_success,
        physical_template_review=review,
        native_execution_status=result.get('canonical_execution_status', inner_status),
        physical_process_gate_passed=process_passed if is_mp else None,
        failure_class=failure_class)
