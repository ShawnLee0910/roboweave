"""Bounded construction-only source-guided joint placement; no public actions."""
from copy import deepcopy
import hashlib
import json
import time
import numpy as np
from .construction_progress import checkpoint

CAPABILITY='BOUNDED_INITIAL_LAYOUT_V1'


def same_support_groups(ordered_recipes):
    """Adjacent placements only: never move work across dependency/Inside steps."""
    from itertools import groupby
    def key(recipe):
        if (recipe.get('predicate') == 'ontop' and recipe.get('desired') is True
                and not recipe.get('placement_policy')):
            return recipe['object_role']
        return None
    return [list(group) for support, group in groupby(ordered_recipes, key=key)
            if support is not None]


class LayoutRestoreError(RuntimeError):
    session_invalidated=True


def validate_policy(recipe, roles):
    policy=recipe.get('placement_policy')
    if policy is None:return
    if recipe.get('predicate')!='ontop' or recipe.get('desired') is not True or 'placement_rank' in recipe:
        raise ValueError('layout policy requires positive unranked OnTop')
    if set(policy)!={'support_roles','sources','max_attempts','max_seconds','semantic_authorization'}:
        raise ValueError('unknown layout policy fields')
    supports=policy['support_roles']
    if (not isinstance(supports,list) or not supports or supports[0]!=recipe['object_role'] or
        len(supports)!=len(set(supports)) or not set(supports)<=set(roles) or recipe['subject_role'] in supports):
        raise ValueError('invalid declared support alternatives')
    if type(policy['max_attempts']) is not int or not 1<=policy['max_attempts']<=256:
        raise ValueError('invalid total layout candidate budget')
    if type(policy['max_seconds']) not in (int,float) or not 0<policy['max_seconds']<=600:
        raise ValueError('invalid total layout time budget')
    if not isinstance(policy['semantic_authorization'],dict) or policy['semantic_authorization'].get('temporary_support_confirmed') is not True:
        raise ValueError('temporary support authorization missing')
    if not isinstance(policy['sources'],list) or len(policy['sources'])>4:
        raise ValueError('invalid source placement evidence')
    for source in policy['sources']:
        if set(source)!={'path','sha256','subject_name','support_name','support_role'} or source['support_role'] not in supports:
            raise ValueError('invalid source pose reference')


def validate_layout_contract(contract, goals):
    recipes=[r for r in contract['initial_state_recipe'] if r.get('placement_policy')]
    if not recipes:return
    roles=contract['role_to_scope'];branch=contract.get('task_contract',{})
    if branch.get('axis')!='M' or branch.get('initial_reference_goals') or branch.get('inspection_region'):
        raise ValueError('layout freedom is not implemented for observation or initial-reference tasks')
    process=branch.get('process_requirements',{})
    if process.get('history_program') or process.get('current_reference_contracts'):
        raise ValueError('layout freedom needs explicit history compatibility')
    supports=recipes[0]['placement_policy']['support_roles']
    if any(r['placement_policy']['support_roles']!=supports for r in recipes):
        raise ValueError('one joint support group per construction contract is currently supported')
    def uses(value):
        if isinstance(value,list):return any(uses(v) for v in value)
        if not isinstance(value,dict):return False
        return (value.get('subject_role') in supports or value.get('object_role') in supports
                or any(uses(v) for k,v in value.items() if k not in {'source_process','public_instruction_evidence'}))
    if uses(goals) or uses(process):raise ValueError('alternate support participates in task goal or process')
    for recipe in recipes:
        validate_policy(recipe,roles)
        matches=[a for a in branch.get('initial_assertions',[]) if a.get('predicate')=='ontop' and a.get('subject_role')==recipe['subject_role']]
        if len(matches)!=1 or matches[0]!=recipe:
            raise ValueError('layout construction and branch initial assertion differ')
        for role in supports:
            binding=contract['scope_bindings'][roles[role]]
            if binding.get('binding_kind')!='SCENE_OBJECT':raise ValueError('alternate support is not native')
            if not set(binding.get('in_rooms',[])) & set(contract['rooms']):raise ValueError('alternate support outside task rooms')


def source_candidates(subject,support,role,sources):
    from pathlib import Path
    from .spatial_symbolic_actions import object_pose, SpatialPoseCandidate, candidate_aabb, object_aabb
    result=[]
    for i,source in enumerate(sources):
        if source['support_role']!=role:continue
        raw=Path(source['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=source['sha256']:
            raise ValueError('source placement template hash mismatch')
        document=json.loads(raw);registry=document['state']['registry']['object_registry']
        for name,obj in [(source['subject_name'],subject),(source['support_name'],support)]:
            args=document['objects_info']['init_info'][name]['args']
            if args.get('category')!=getattr(obj,'category',None) or args.get('model')!=getattr(obj,'model',None):
                raise ValueError('source placement asset identity mismatch')
        old=registry[source['subject_name']]['root_link'];anchor=registry[source['support_name']]['root_link']
        # Retain source XY offset and orientation; derive Z from current geometry.
        # No source scale is applied to the live asset.
        live_anchor=object_pose(support)[0]
        # The source support must not have rotated since the referenced layout.
        live_q=np.asarray(object_pose(support)[1]);source_q=np.asarray(anchor['ori'])
        if (source_q.shape!=(4,) or not np.isfinite(source_q).all() or np.linalg.norm(source_q)==0
            or np.asarray(old['ori']).shape!=(4,) or not np.isfinite(old['ori']).all()
            or np.asarray(old['pos']).shape!=(3,) or not np.isfinite(old['pos']).all()):
            raise ValueError('invalid source pose data')
        if abs(float(np.dot(live_q/np.linalg.norm(live_q),source_q/np.linalg.norm(source_q))))<1-1e-5:
            continue  # Not a semantic failure; source hint is inapplicable.
        position=np.asarray(old['pos'],dtype=float)-np.asarray(anchor['pos'],dtype=float)+live_anchor
        candidate=SpatialPoseCandidate(f'source_{i}',position,np.asarray(old['ori'],dtype=float))
        lo,_=candidate_aabb(subject,candidate);_,hi=object_aabb(support)
        from .spatial_symbolic_actions import DEFAULT_SUPPORT_CLEARANCE_M
        position=position.copy();position[2]+=hi[2]-lo[2]+DEFAULT_SUPPORT_CLEARANCE_M
        result.append(SpatialPoseCandidate(f'source_{i}',position,candidate.orientation_xyzw))
    return result


def free_region_candidates(subject,support,objects,limit=48):
    from .spatial_symbolic_actions import (object_pose,object_aabb,SpatialPoseCandidate,
        _aabb_overlap,DEFAULT_SUPPORT_CLEARANCE_M,DEFAULT_SUPPORT_INSET_M)
    pos,q=object_pose(subject);lo,hi=object_aabb(subject);slo,shi=object_aabb(support)
    half=(hi-lo)/2;offset=(lo+hi)/2-pos
    low=slo[:2]+half[:2]+DEFAULT_SUPPORT_INSET_M;high=shi[:2]-half[:2]-DEFAULT_SUPPORT_INSET_M
    if np.any(low>high):return ()
    # Obstacle-edge partitions find pockets that a uniform central grid misses.
    axes=[{float(low[k]),float(high[k]),float((low[k]+high[k])/2)} for k in range(2)]
    obstacles=[]
    for obj in objects:
        if obj is subject or obj is support:continue
        try:a,b=object_aabb(obj)
        except (ValueError,AttributeError):continue
        obstacles.append((str(getattr(obj,'name','')) or '<unnamed>',a,b))
        if b[2]<shi[2] or a[2]>shi[2]+2*half[2]:continue
        for k in range(2):
            for v in (a[k]-half[k]-DEFAULT_SUPPORT_INSET_M,b[k]+half[k]+DEFAULT_SUPPORT_INSET_M):
                if low[k]<=v<=high[k]:axes[k].add(float(v))
    axes=[sorted(values)[:32] for values in axes]
    candidates=[]
    for x in axes[0]:
        for y in axes[1]:
            center=np.array([x,y,shi[2]+half[2]+DEFAULT_SUPPORT_CLEARANCE_M])
            candidate=SpatialPoseCandidate(f'free_region_{len(candidates)}',center-offset,np.asarray(q))
            delta=candidate.position-pos
            blockers={name for name,a,b in obstacles if _aabb_overlap(lo+delta,hi+delta,a,b,margin_m=0.)}
            candidates.append((len(blockers),candidate))
    # AABB is a ranking hint only; exact collision-shape queries still decide.
    return tuple(c for _,c in sorted(candidates,key=lambda item:(item[0],item[1].name))[:limit])


def bounded_joint_search(items, supports, *, candidates, attempt, snapshot, restore, verify,
                         max_attempts,max_seconds,clock=time.monotonic):
    start=clock();count=0;events=[]
    checkpoint('LAYOUT_ROOT_SNAPSHOT')
    root=snapshot();support_limit=max_attempts;support_deadline=start+max_seconds
    original_restore=restore
    def restore(state):
        checkpoint('LAYOUT_RESTORE',attempt_count=count)
        try:original_restore(state)
        except BaseException as exc:raise LayoutRestoreError('joint layout restore failed') from exc
    def search(index,support):
        nonlocal count
        if index==len(items):return clock()<=start+max_seconds and verify()
        if count>=support_limit or clock()>=support_deadline:return False
        checkpoint('LAYOUT_SNAPSHOT',item=items[index],support=support)
        state=snapshot();item=items[index]
        checkpoint('LAYOUT_CANDIDATE_GENERATION',item=item,support=support)
        for candidate in candidates(item,support):
            if count>=support_limit or clock()>=support_deadline:return False
            restore(state);count+=1
            checkpoint('LAYOUT_ATTEMPT',item=item,support=support,candidate=candidate.name,attempt_count=count)
            outcome=attempt(item,support,candidate)
            events.append(dict(item=item,support=support,candidate=candidate.name,result=outcome))
            if outcome.get('status')=='PASS' and search(index+1,support):return True
        restore(state)
        return False
    try:
        for index,support in enumerate(supports):
            if count>=max_attempts or clock()>=start+max_seconds:break
            remaining=len(supports)-index
            support_limit=count+max(0,(max_attempts-count)//remaining)
            now=clock();support_deadline=now+max(0.,(start+max_seconds-now)/remaining)
            restore(root)
            if search(0,support):return dict(status='PASS',support_role=support,attempt_count=count,attempts=events)
        restore(root)
        return dict(status='FAIL',attempt_count=count,attempts=events,
                    failure='LAYOUT_BUDGET_OR_CANDIDATES_EXHAUSTED',physical_impossibility_proven=False)
    except BaseException as exc:
        try:
            restore(root)
        except LayoutRestoreError as restore_error:
            restore_error.layout_report=dict(status='FAIL',attempt_count=count,attempts=events,restore_failed=True)
            raise restore_error from exc
        exc.layout_report=dict(status='FAIL',attempt_count=count,attempts=events,restore_failed=False)
        raise
