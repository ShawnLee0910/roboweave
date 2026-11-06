"""Opt-in initial receiver checks; never changes Action or goal semantics."""
import numpy as np

from .spatial_symbolic_actions import (
    SpatialPoseCandidate, build_downward_support_ray_requests, candidate_aabb,
    object_pose, DEFAULT_SUPPORT_CLEARANCE_M,
)

CAPABILITY = 'PHYSICAL_RECEIVING_SUPPORT_V1'


def validate_receiving_support(recipe, roles):
    rule = recipe.get('receiving_support')
    if rule is None:
        return
    if (not isinstance(rule, dict) or set(rule) != {'placement', 'placement_subject_role'}
            or rule['placement'] != 'center_only'
            or rule['placement_subject_role'] not in roles
            or rule['placement_subject_role'] in {recipe.get('subject_role'), recipe.get('object_role')}
            or recipe.get('predicate') != 'ontop' or recipe.get('desired') is not True
            or recipe.get('subject_role') not in roles or recipe.get('object_role') not in roles
            or 'placement_rank' in recipe or 'placement_policy' in recipe):
        raise ValueError('invalid Physical receiving-support construction rule')


def centered_receiver_candidate(subject, support, candidates):
    """Certify the centre against actual collision surfaces before moving anything."""
    from omnigibson.utils.sampling_utils import raytest_batch
    from .run_b1_marker_symbolic_rollout import (
        _action20_upward_support_hit, _action20_flat_surface_cluster,
    )
    import torch
    centers = [c for c in candidates if c.name == 'on_top_center']
    if len(centers) != 1:
        return (), {'passed': False, 'reason': 'CENTER_CANDIDATE_UNAVAILABLE'}
    candidate = centers[0]
    rays = build_downward_support_ray_requests(subject, candidate)
    ignored = tuple(subject.link_prim_paths)
    paths = tuple(support.link_prim_paths)
    if not ignored or not paths:
        raise RuntimeError('receiving support requires live collision identities')
    results = raytest_batch(
        torch.as_tensor(np.asarray([r.start_xyz for r in rays]), dtype=torch.float32),
        torch.as_tensor(np.asarray([r.end_xyz for r in rays]), dtype=torch.float32),
        only_closest=True, ignore_bodies=ignored, ignore_collisions=ignored,
    )
    if len(results) != len(rays):
        raise RuntimeError('incomplete receiving support ray batch')
    hits = tuple((i, hit) for i, result in enumerate(results)
                 if (hit := _action20_upward_support_hit(result, support_paths=paths)) is not None)
    cluster = _action20_flat_surface_cluster(hits)
    # This canary requires the entire tray footprint on the table, not overhang.
    passed = len(cluster) == len(rays)
    report = dict(passed=passed, upward_hits=len(hits), coplanar_hits=len(cluster),
                  required_hits=len(rays), policy='CENTER_FULL_FOOTPRINT_LIVE_COLLISION_RAYS')
    if not passed:
        return (), report
    z = float(np.median([hit[2] for _, hit in cluster]))
    position = candidate.position.copy()
    position[2] += z + DEFAULT_SUPPORT_CLEARANCE_M - candidate_aabb(subject, candidate)[0][2]
    report['surface_z_m'] = z
    return (SpatialPoseCandidate(candidate.name, position, candidate.orientation_xyzw),), report


def receiver_upright(subject):
    from .run_b1_marker_symbolic_rollout import _ACTION20_SURFACE_NORMAL_MIN_Z
    from .on_top_support_layers import quaternion_to_matrix_xyzw
    _, orientation = object_pose(subject)
    up_z = float(quaternion_to_matrix_xyzw(orientation)[2, 2])
    return dict(passed=bool(up_z >= _ACTION20_SURFACE_NORMAL_MIN_Z), up_z=up_z,
                minimum_up_z=_ACTION20_SURFACE_NORMAL_MIN_Z,
                policy='EXISTING_ACTION20_UPWARD_SURFACE_NORMAL')


def receiving_surface_readback(evaluator, subject, support, placement_subject):
    """Use the unchanged Physical surface publisher's geometry/projection path."""
    from omnigibson.utils.sampling_utils import raytest_batch
    from .run_b1_marker_symbolic_rollout import (
        _action20_on_top_surface_regions, _live_head_camera_sensor,
    )
    import torch
    for obj in (subject, support, placement_subject):
        obj.clear_states_cache()
    from .spatial_symbolic_actions import canonical_relation
    on_top = canonical_relation(subject, 'OnTop', support)
    touching = canonical_relation(subject, 'Touching', support)
    upright = receiver_upright(subject)
    _, sensor = _live_head_camera_sensor(evaluator)
    diagnostic = {}
    regions = _action20_on_top_surface_regions(
        evaluator=evaluator, held=placement_subject,
        preferred_orientation_xyzw=object_pose(placement_subject)[1],
        eligible_supports_by_name={subject.name: subject},
        image_shape=(sensor.image_height, sensor.image_width, 3),
        raytest_batch=raytest_batch, tensor_module=torch, symbolic_support=False,
        diagnostic=diagnostic,
    )
    return dict(passed=bool(on_top and touching and upright['passed'] and regions),
                on_top=bool(on_top), touching=bool(touching), upright=upright,
                generated_region_count=len(regions), surface_diagnostic=diagnostic,
                public_marker_publication_checked=False,
                scope='INITIAL_RECEIVER_ONLY_NOT_TASK_GOAL_OR_ACTION_SUCCESS')
