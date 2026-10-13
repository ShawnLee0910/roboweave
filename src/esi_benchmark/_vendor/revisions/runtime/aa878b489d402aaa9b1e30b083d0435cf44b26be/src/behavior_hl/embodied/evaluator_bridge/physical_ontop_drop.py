"""Kinematic transfer to a declared column, then one irreversible physical release."""
import numpy as np

from .spatial_symbolic_actions import object_aabb, object_pose, SpatialPoseCandidate, candidate_aabb, canonical_relation
from .spatial_candidate_admission import query_spatial_candidate_collision_overlap, spatial_object_link_paths
from .on_top_support_layers import read_base_aligned_box, quaternion_to_matrix_xyzw
from .physical_placement_orientation import base_orientation, select_orientation

POLICY = 'physical-ontop-local-surface-overlap-drop/v3'
CLEARANCE_M = 0.005
MAX_RELEASE_QUERIES = 21


def physical_region_rank(binding):
    from .private_visual_markers import SpatialMarkerTargetToken
    token = binding.target_token
    if not isinstance(token, SpatialMarkerTargetToken) or token.operation != 'place_on_top':
        raise ValueError('invalid Physical OnTop region token')
    lo, hi = object_aabb(token.target_token)
    center = np.asarray(token.region_center_xy, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError('invalid Physical region center')
    return (token.selector != 'on_top_physical_center',
            float(np.linalg.norm(center - (lo[:2] + hi[:2]) / 2)), binding.marker)


def inward_release_xy(held, target, center_xy, extent_xy, orientation):
    lo, hi = object_aabb(target)
    center = np.asarray(center_xy, dtype=float)
    extent = np.asarray(extent_xy, dtype=float)
    if center.shape != (2,) or extent.shape != (2,) or not np.isfinite([center, extent]).all() or np.any(extent <= 0):
        raise ValueError('invalid selected Physical region')
    lower, upper = center - extent / 2, center + extent / 2
    box = read_base_aligned_box(held)
    half = (np.abs(quaternion_to_matrix_xyzw(orientation)) @ box.extent_local)[:2] / 2
    fit_lower, fit_upper = np.maximum(lower, lo[:2] + half), np.minimum(upper, hi[:2] - half)
    fits = bool(np.all(fit_lower <= fit_upper))
    point = np.clip((lo[:2] + hi[:2]) / 2, fit_lower if fits else lower, fit_upper if fits else upper)
    _, surface = selected_surface_height(target, point)
    # Never turn an inward adjustment into a fabricated surface in a hole/concavity.
    if surface['target_surface_hit_count'] == 0:
        point = center.copy()
    return point, dict(original_region_center_xy=center.tolist(), region_extent_xy=extent.tolist(),
        selected_release_xy=point.tolist(), footprint_fits_outer_bounds=fits,
        inward_surface_hit_count=surface['target_surface_hit_count'],
        rule='TOWARD_SUPPORT_CENTER_WITHIN_SELECTED_REGION',
        outer_bounds_not_contact_certificate=True)


def release_bounds(held, candidate):
    box = read_base_aligned_box(held)
    rotation = quaternion_to_matrix_xyzw(candidate.orientation_xyzw)
    center = candidate.position + rotation @ box.center_local
    extent = np.abs(rotation) @ box.extent_local
    return center - extent / 2, center + extent / 2


def _cpu_vector(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def closure_evidence(closure, snapshot):
    """Name and relate every carried object, persisted before any release."""
    return dict(
        closure_object_names=[getattr(obj, 'name', None) for obj in closure],
        closure_descendant_count=len(snapshot.entries) if snapshot is not None else 0,
        closure_relation_edges=[
            dict(kind=edge.kind, child_name=getattr(edge.child, 'name', None),
                 parent_name=getattr(edge.parent, 'name', None))
            for edge in (snapshot.relation_edges if snapshot is not None else ())
        ],
    )


def closure_release_geometry(held, candidate, snapshot, owner, transforms, tensor):
    """Union release AABB of the whole closure at one shared root candidate pose.

    Descendant poses reuse the captured root-relative transforms that the root
    pose application will use, so the queried volume is the volume released.
    """
    lower, upper = release_bounds(held, candidate)
    rows = [dict(object_name=getattr(held, 'name', None), release_lower=lower.tolist(),
                 release_upper=upper.tolist())]
    if snapshot is None:
        return (lower, upper), rows
    root_pose = transforms.pose2mat((tensor.as_tensor(candidate.position, dtype=tensor.float32),
                                     tensor.as_tensor(candidate.orientation_xyzw, dtype=tensor.float32)))
    for entry in snapshot.entries:
        position, orientation = transforms.mat2pose(
            owner._logical_inventory_compose_pose(root_pose, entry.relative_to_root))
        child = SpatialPoseCandidate('closure_release', _cpu_vector(position), _cpu_vector(orientation))
        child_lower, child_upper = release_bounds(entry.object_ref, child)
        if not np.isfinite([child_lower, child_upper]).all() or np.any(child_upper <= child_lower):
            raise ValueError('invalid Physical release geometry for a carried object')
        rows.append(dict(object_name=getattr(entry.object_ref, 'name', None),
                         release_lower=child_lower.tolist(), release_upper=child_upper.tolist()))
        lower, upper = np.minimum(lower, child_lower), np.maximum(upper, child_upper)
    return (lower, upper), rows


def release_candidate(held, target, center_xy, orientation, *, surface_z=None):
    lower, upper = object_aabb(target)
    center = (lower[:2] + upper[:2]) / 2 if center_xy is None else np.asarray(center_xy, dtype=float)
    rotation = np.asarray(orientation, dtype=float)
    if (center.shape != (2,) or rotation.shape != (4,) or not np.isfinite(center).all()
            or not np.isfinite(rotation).all() or not np.isfinite(upper).all()
            or not np.isclose(np.linalg.norm(rotation), 1)):
        raise ValueError('invalid Physical release coordinates')
    height = float(upper[2] if surface_z is None else surface_z)
    if not np.isfinite(height):
        raise ValueError('invalid Physical release surface height')
    candidate = SpatialPoseCandidate('selected_column_release', np.r_[center, height], rotation)
    lo, hi = release_bounds(held, candidate)
    # Align the object's actual AABB center, not its possibly offset origin.
    candidate.position[:2] += center - (lo[:2] + hi[:2]) / 2
    candidate.position[2] += height + CLEARANCE_M - lo[2]
    return candidate


def selected_surface_height(target, center_xy):
    """Use the actual selected surface, including an inner tray floor.

    A hole remains a legal drop experiment: a missed target ray uses the target's
    outer top as the initial reference, never a claim of support or stability.
    """
    from omnigibson.utils.sampling_utils import raytest
    from .run_b1_marker_symbolic_rollout import _action20_upward_support_hit
    lo, hi = object_aabb(target)
    xy = (lo[:2] + hi[:2]) / 2 if center_xy is None else np.asarray(center_xy, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError('invalid Physical release column')
    hits = raytest([*xy, float(hi[2]) + 0.01], [*xy, float(lo[2]) - 0.01], only_closest=False)
    paths = spatial_object_link_paths(target)
    points = [p for hit in hits if (p := _action20_upward_support_hit(hit, support_paths=paths)) is not None]
    z = max(float(p[2]) for p in points) if points else float(hi[2])
    return z, dict(source='SELECTED_TARGET_COLLISION_SURFACE_RAY' if points else 'TARGET_TOP_FALLBACK_FOR_GAP',
                   surface_z_m=z, selected_xy=xy.tolist(), target_surface_hit_count=len(points))


def raise_above_blockers(held, candidate, query, og, *, group_release_lower=None):
    """Keep XY and orientation; raise the bottom to real blocker tops + 5 mm.

    ``group_release_lower`` is the union release AABB lower corner of the whole
    carried closure. A descendant below the root owns the lowest point, so the
    shared root lift has to clear the blocker for the group, not for the root.
    """
    if query.query_error or not query.blocking_hits:
        raise RuntimeError('release overlap query has no usable blocker evidence')
    objects = [obj for scene in og.sim.scenes for obj in scene.objects]
    evidence = []
    for hit in query.blocking_hits:
        owners = [obj for obj in objects if obj is not held and any(
            value and (value == path or value.startswith(path + '/'))
            for value in (hit.rigid_body_path, hit.collision_path)
            for path in spatial_object_link_paths(obj))]
        if len(owners) != 1:
            raise RuntimeError('release blocker identity is missing or ambiguous')
        blocker = owners[0]
        top = float(object_aabb(blocker)[1][2])
        if not np.isfinite(top):
            raise RuntimeError('nonfinite release blocker height')
        evidence.append(dict(object_name=getattr(blocker, 'name', None),
                             rigid_body_path=hit.rigid_body_path, top_z_m=top))
    before = float(release_bounds(held, candidate)[0][2] if group_release_lower is None
                   else group_release_lower[2])
    after = max(row['top_z_m'] for row in evidence) + CLEARANCE_M
    if after <= before + 1e-7:
        raise RuntimeError('release blocker height does not resolve the overlap')
    candidate.position[2] += after - before
    return dict(blockers=evidence, previous_bottom_z_m=before, new_bottom_z_m=after,
                clearance_m=CLEARANCE_M, height_source='LIVE_BLOCKING_OBJECT_AABB_TOP',
                bottom_source=('CLOSURE_UNION_RELEASE_AABB' if group_release_lower is not None
                               else 'ROOT_RELEASE_AABB'))


PARTIAL_APPLY_RESTORE_REASON = 'physical_drop_partial_release_apply'


def closure_rollback(owner, closure, snapshot):
    """Restore the whole carried closure after an incomplete shared pose apply.

    Never reports a clean rollback it cannot prove: when the shared sync cleanup
    already restored and cleared the snapshot, that path is delegated, and a
    partial or failed restore keeps ``rollback_complete`` false.
    """
    if snapshot is None:
        return dict(rollback_performed=False, rollback_complete=False,
                    rollback_status='NO_LOGICAL_INVENTORY_SNAPSHOT')
    if owner._logical_inventory_snapshot is None:
        return dict(rollback_performed=False, rollback_complete=False,
                    rollback_status='DELEGATED_TO_SHARED_SYNC_CLEANUP')
    try:
        restore = owner._restore_logical_inventory(reason=PARTIAL_APPLY_RESTORE_REASON)
    except BaseException as exc:
        return dict(rollback_performed=True, rollback_complete=False, rollback_status='RESTORE_FAILED',
                    logical_inventory_restore_error=type(exc).__name__ + ': ' + str(exc))
    complete = (int(restore.get('logical_inventory_restored_count', 0)) == len(closure)
                and restore.get('logical_inventory_restore_error') is None)
    return dict(rollback_performed=True, rollback_complete=complete,
                rollback_status='COMPLETE' if complete else 'INCOMPLETE', **restore)


def execute_drop(owner, target, *, center_xy, og, tensor, transforms, extent_xy=None,
                 selector=None, published_surface_z_m=None, operation='place_on_top'):
    if operation not in ('place_on_top', 'place_inside'):
        raise ValueError('unsupported Physical drop operation')
    inside = operation == 'place_inside'
    if (center_xy is None) != (extent_xy is None) and selector is not None:
        raise ValueError('selected Physical region requires both center and extent')
    if published_surface_z_m is not None and not np.isfinite(published_surface_z_m):
        raise ValueError('nonfinite published Physical surface height')
    held = owner._get_obj_in_hand()
    if held is None or target is None or held is target:
        raise ValueError('Physical drop requires distinct held object and target')
    snapshot = owner._logical_inventory_snapshot
    closure = tuple(snapshot.object_refs) if snapshot is not None else (held,)
    if len(closure) > 1 and inside:
        raise ValueError('multi-object Physical Inside drop is unsupported')
    orientation = object_pose(held)[1]
    if owner._logical_inventory_origin_pose is not None:
        orientation = transforms.mat2pose(owner._logical_inventory_origin_pose)[1]
        if hasattr(orientation, 'detach'):
            orientation = orientation.detach().cpu().numpy()
    queries = []
    report = dict(operation=operation, policy=('physical-inside-above-container-drop/v1' if inside else POLICY), execution_fidelity='kinematic_research',
                  subject_name=getattr(held, 'name', None), target_name=getattr(target, 'name', None),
                  official_challenge_eligible=False, rollback_performed=False,
                  physical_consequence_committed=False, placement_committed=False,
                  release_started=False, physics_steps=0, observations=[], release_volume_queries=queries,
                  release_clearance_m=CLEARANCE_M, overlap_height_adjustments=[],
                  selected_selector=selector, published_surface_z_m=published_surface_z_m,
                  release_applied=False, observation_complete=False,
                  release_pose_apply_incomplete=False, rollback_complete=False,
                  released_object_names=[], **closure_evidence(closure, snapshot))
    owner._last_private_exact_report = report
    try:
        if not inside:
            from .physical_template_runtime import ACTIVE
            initial_audit = ACTIVE.get()
            fallback_source = ('pre_grasp_pose' if owner._logical_inventory_origin_pose is not None
                               else 'current_object_pose')
            for role, pose in getattr(initial_audit, 'initial_poses', {}).items():
                if initial_audit.role_object(role) is held:
                    orientation = pose[1]
                    fallback_source = 'task_initial_pose'
                    break
            orientation, report['base_orientation'] = base_orientation(
                held, orientation, fallback_source)
        if extent_xy is not None:
            center_xy, report['physical_landing_selection'] = inward_release_xy(
                held, target, center_xy, extent_xy, orientation)
        if not inside:
            # Choose yaw at this single landing point; never move/retry the world
            # to compare orientations and never redo XY after selecting a yaw.
            orientation, report['direction_selection'] = select_orientation(held, target, orientation, center_xy)
        if inside:
            # Conservative upper envelope, NOT an inferred cavity or rim.
            # In particular do not raycast to the inner floor and teleport there.
            surface_z = float(object_aabb(target)[1][2])
            surface_report = dict(source='CONTAINER_OUTER_TOP_CONSERVATIVE',
                                  surface_z_m=surface_z, actual_opening_rim_verified=False)
            report['full_body_containment'] = 'UNKNOWN'
        else:
            surface_z, surface_report = selected_surface_height(target, center_xy)
        report['initial_surface_reference'] = surface_report
        candidate = release_candidate(held, target, center_xy, orientation, surface_z=surface_z)
        # This tests empty starting volume only, never support fit or stability.
        for index in range(MAX_RELEASE_QUERIES):
            release_aabb, release_objects = closure_release_geometry(
                held, candidate, snapshot, owner, transforms, tensor)
            query = query_spatial_candidate_collision_overlap(
                omnigibson=og, held=held, closure_objects=closure, candidate=candidate,
                candidate_aabb_world=release_aabb,
                xy_margin_m=0.0, z_inset_m=0.0)
            queries.append(query.to_private_dict())
            if query.accepted:
                report['release_geometry'] = dict(
                    closure_release_lower=release_aabb[0].tolist(),
                    closure_release_upper=release_aabb[1].tolist(),
                    closure_release_objects=release_objects)
                break
            if index + 1 < MAX_RELEASE_QUERIES:
                report['overlap_height_adjustments'].append(raise_above_blockers(
                    held, candidate, query, og, group_release_lower=release_aabb[0]))
        else:
            raise RuntimeError('release blocker query budget exhausted; not a stability failure')
        owner._robot_object_collision_guard.restore(held, reason='physical_drop_release')
        official = owner.robot._ag_obj_in_hand.get(owner.arm)
        constraint = owner.robot._ag_obj_constraints.get(owner.arm)
        if official is not None or constraint is not None:
            raise RuntimeError('unexpected official attachment in logical-carry drop')
        pose = transforms.pose2mat((tensor.as_tensor(candidate.position, dtype=tensor.float32),
                                   tensor.as_tensor(candidate.orientation_xyzw, dtype=tensor.float32)))
        report['release_pose'] = {'position': candidate.position.tolist(), 'orientation': candidate.orientation_xyzw.tolist()}
        report['release_bottom_z_m'] = float(release_bounds(held, candidate)[0][2])
        report['release_started'] = True
        owner._placing_with_predicate = True
        try:
            try:
                if snapshot is not None:
                    owner._apply_logical_inventory_root_pose(pose)
                else:
                    held.set_position_orientation(*transforms.mat2pose(pose))
            except BaseException:
                # The shared apply may already have moved the root and earlier
                # descendants, so this is a world change, not an untouched one.
                report['release_pose_apply_incomplete'] = True
                report['physical_consequence_committed'] = True
                report.update(closure_rollback(owner, closure, snapshot))
                raise
            # Applying the pose already changes the world, even if release
            # cleanup or velocity initialization subsequently raises.
            report['release_applied'] = True
            report['physical_consequence_committed'] = True
            if inside:
                owner._physical_inside_applied = True
            released = report['released_object_names']
            for carried in closure:
                carried.keep_still()
                carried.enable_gravity()
                released.append(getattr(carried, 'name', None))
            report['release_applied'] = True
            report['physical_consequence_committed'] = True
        finally:
            owner._clear_logical_inventory_state()
        # Observation only: do not introduce a velocity success threshold.
        try:
            velocities = {}
            for name in ('linear', 'angular'):
                value = getattr(held, 'get_' + name + '_velocity')()
                if hasattr(value, 'detach'):
                    value = value.detach().cpu().numpy()
                value = np.asarray(value, dtype=float)
                if value.shape != (3,) or not np.isfinite(value).all():
                    raise ValueError('invalid release velocity readback')
                velocities[name] = value.tolist()
            report['release_velocity_readback'] = dict(status='MEASURED', **velocities)
        except Exception as exc:
            report['release_velocity_readback'] = dict(status='UNKNOWN', error=type(exc).__name__ + ': ' + str(exc))
        from .physical_template_runtime import ACTIVE
        audit = ACTIVE.get()
        if audit is not None:audit.on_release(held,target)
        for step in range(30):
            yield None
            report['physics_steps'] = step + 1
            held.clear_states_cache()
            target.clear_states_cache()
            position, rotation = object_pose(held)
            if not np.isfinite(position).all() or not np.isfinite(rotation).all():
                raise RuntimeError('nonfinite Physical drop state')
            if step >= 20:
                sample = dict(step=step+1, position=position.tolist(),
                              orientation_xyzw=rotation.tolist(),
                              touching=bool(canonical_relation(held, 'Touching', target)))
                sample['inside' if inside else 'on_top'] = bool(
                    canonical_relation(held, 'Inside' if inside else 'OnTop', target))
                report['observations'].append(sample)
        report['physical_consequence_committed'] = True
        report['placement_committed'] = True
        report['observation_complete'] = True
        report['semantic_state_pass'] = report['observations'][-1]['inside' if inside else 'on_top']
    except BaseException as exc:
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        from .physical_template_runtime import ACTIVE
        audit = ACTIVE.get()
        if audit is not None:audit.end_release()
        owner._placing_with_predicate = False
        report['release_readback'] = dict(held=owner._get_obj_in_hand() is not None,
            snapshot_present=owner._logical_inventory_snapshot is not None,
            official_constraint=owner.robot._ag_obj_constraints.get(owner.arm) is not None,
            official_held=owner.robot._ag_obj_in_hand.get(owner.arm) is not None,
            collision_guard=owner._robot_object_collision_guard.snapshot())
        owner._last_private_exact_report = report
        if audit is not None:
            # Independent of the Oracle dispatcher returning normally: retain
            # the irreversible release even on cancellation / GeneratorExit.
            try:
                audit.append('physical_release_lifecycle.jsonl', report)
            except Exception as exc:
                audit.errors.append('release_receipt_write_failed: ' + str(exc))
