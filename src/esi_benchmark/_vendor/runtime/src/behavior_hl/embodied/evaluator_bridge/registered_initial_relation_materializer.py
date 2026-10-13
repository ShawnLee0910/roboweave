"""Bounded Under/NextTo construction using existing live geometry kernels.

This is an initial-state materializer, never an Agent Action or Oracle repair.
Failed mutating trials restore a native scene checkpoint before another trial.
"""
from collections.abc import Mapping

import numpy as np

from .spatial_symbolic_actions import (
    object_aabb, object_pose, expanded_lateral_candidates,
    candidate_aabb, read_action_relation,
)
from .spatial_candidate_admission import (
    query_spatial_candidate_collision_overlap, query_symbolic_placement_final_collision,
)

CAPABILITY = 'RELATIONAL_INITIAL_PLACEMENT_V1'
MAXIMUM_SUPPORTS = 16
MAXIMUM_CANDIDATES = 64


class InitialRelationRestoreError(RuntimeError):
    session_invalidated = True
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _support_candidates(subject, target, scene_objects, predicate):
    target_min, target_max = object_aabb(target)
    result, errors = [], []
    for support in scene_objects:
        if support is subject or support is target:
            continue
        try:
            lo, hi = object_aabb(support)
            if np.any(hi[:2] <= target_min[:2]) or np.any(lo[:2] >= target_max[:2]) or hi[2] >= target_max[2]:
                continue
            if predicate == 'next_to' and not read_action_relation(target, 'OnTop', support, predicate_mode='action20_symbolic'):
                continue
            result.append((abs(float(hi[2]-target_min[2])), str(getattr(support, 'name', '')), support))
        except Exception as exc:
            errors.append({'support': str(getattr(support, 'name', '')), 'error': f'{type(exc).__name__}: {exc}'})
    result.sort(key=lambda row: row[:2])
    return tuple(row[2] for row in result[:MAXIMUM_SUPPORTS]), len(result), errors


def materialize_initial_relation(*, subject, target, predicate, scene, robot, omnigibson,
                                 settle, snapshot, restore, prior_truth,
                                 raytest_batch=None, tensor_module=None):
    """Read actual shapes, generate bounded poses, transact, then verify relations."""
    predicate = 'next_to' if predicate == 'nextto' else predicate
    if predicate not in {'under', 'next_to'}:
        raise ValueError('unsupported initial spatial relation')
    relation = {'under':'Under', 'next_to':'NextTo'}[predicate]
    raw_objects = scene.objects
    objects = tuple(raw_objects.values()) if isinstance(raw_objects, Mapping) else tuple(raw_objects)
    supports, available, support_errors = _support_candidates(subject, target, objects, predicate)
    report = dict(status='FAIL', predicate=predicate, attempts=[], support_count_considered=len(supports),
        support_count_available=available, support_query_errors=support_errors,
        maximum_supports=MAXIMUM_SUPPORTS, maximum_candidates=MAXIMUM_CANDIDATES,
        direct_state_modification=True, agent_actions_executed=0)
    if not all(prior_truth()):
        raise ValueError('previous initial relations are no longer satisfied before construction')
    plans = []
    if predicate == 'under':
        from .marker_symbolic_spatial_execution import plan_live_under_supports
        under_plans = plan_live_under_supports(held=subject, target=target, supports=supports, robot=robot,
            raytest_batch=raytest_batch, tensor_module=tensor_module, symbolic_support=True)
        report['under_support_plans'] = [plan.to_private_dict() for plan in under_plans]
        plans = [(plan.support, candidate) for plan in under_plans for candidate in plan.candidate_plan.candidates]
    else:
        plans = [(support, candidate) for support in supports
                 for candidate in expanded_lateral_candidates(subject, target, support, relation='NextTo')]
    report['candidate_count_generated'] = len(plans)
    if not plans:
        report['planning_error'] = 'no bounded candidate with verified support/underside evidence'
        return report
    # Cache geometry before any trial changes the source object's pose.
    candidates = [(support, candidate, candidate_aabb(subject, candidate))
                  for support, candidate in plans[:MAXIMUM_CANDIDATES]]
    checkpoint = snapshot()

    def rollback(attempt):
        try:
            restore(checkpoint)
            preserved = list(prior_truth())
            if not all(preserved):
                raise RuntimeError('restored initial relations failed canonical readback')
            attempt['rollback'] = dict(status='PASS', previous_relations=preserved,
                                       native_scene_checkpoint_restored=True)
        except Exception as exc:
            attempt['rollback'] = dict(status='FAIL', error=f'{type(exc).__name__}: {exc}')
            raise InitialRelationRestoreError('initial spatial trial restoration failed', report) from exc

    for support, candidate, bounds in candidates:
        attempt = dict(candidate=candidate.name, support=str(getattr(support, 'name', '')), accepted=False)
        report['attempts'].append(attempt)
        shape = query_spatial_candidate_collision_overlap(omnigibson=omnigibson, held=subject,
            closure_objects=(subject,), contact_objects=(support,), candidate=candidate, candidate_aabb_world=bounds)
        attempt['candidate_collision'] = shape.to_private_dict()
        if not shape.accepted:
            continue
        try:
            position, orientation = subject.get_position_orientation()
            def vector(value, reference):
                return reference.new_tensor(value) if hasattr(reference, 'new_tensor') else np.asarray(value).copy()
            subject.set_position_orientation(position=vector(candidate.position, position),
                orientation=vector(candidate.orientation_xyzw, orientation), frame='world')
            keep_still = getattr(subject, 'keep_still', None)
            if callable(keep_still):
                keep_still()
            settle()
            final_collision = query_symbolic_placement_final_collision(omnigibson=omnigibson,
                moving_objects=(subject,), contact_objects=(support,))
            attempt['final_collision'] = final_collision.to_private_dict()
            attempt['relation_satisfied'] = read_action_relation(subject, relation, target, predicate_mode='action20_symbolic')
            attempt['support_satisfied'] = read_action_relation(subject, 'OnTop', support, predicate_mode='action20_symbolic')
            attempt['previous_relations'] = list(prior_truth())
            attempt['accepted'] = bool(final_collision.accepted and attempt['relation_satisfied']
                and attempt['support_satisfied'] and all(attempt['previous_relations']))
            if attempt['accepted']:
                report.update(status='PASS', selected_candidate=candidate.name,
                    actual_position=object_pose(subject)[0].tolist(), actual_orientation_xyzw=object_pose(subject)[1].tolist())
                return report
        except Exception as exc:
            attempt['error'] = f'{type(exc).__name__}: {exc}'
        rollback(attempt)
    report['planning_error'] = 'bounded initial candidates exhausted; no infeasibility claim'
    return report
