#!/usr/bin/env python3
"""CPU-only, non-activating Perception reset-pose diagnostics.

Reuses the L0 map, station, Goal-DAG, route and camera implementation.
Support-center proxies are NOT constructed object positions or RGB evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

import compile_l0_fixed_reset_poses as l0
from behavior_hl.embodied.evaluator_bridge.construction_camera_pose import (
    CAMERA_CAPTURE_SCHEMA,
    camera_at_robot_pose,
    camera_pose_at_robot_pose,
    measured_head_camera,
    camera_pose_readback,
)
from behavior_hl.embodied.evaluator_bridge.on_top_support_layers import (
    quaternion_to_matrix_xyzw,
)


def visibility_roles(bindings):
    result = {}
    for scope, binding in bindings.items():
        roles = set()
        for role in binding.get('roles', []):
            if role.startswith(('distractor', 'foil')):
                roles.add('foil')
            elif 'landmark' in role:
                roles.add('landmark')
            elif role.startswith(('destination', 'target_destination')):
                roles.add('destination')
            elif role.startswith(('target', 'source')):
                roles.add('source')
        if roles and not l0._is_floor(scope, bindings):
            result[scope] = sorted(roles)
    return result


def proxy_positions(positions, bindings, role_to_scope, recipes):
    """Resolve declared placement dependencies; never consume parking poses."""
    result = {scope: list(xyz) for scope, xyz in positions.items()
              if bindings[scope].get('binding_kind') != 'AUTHORED_OBJECT'}
    pending = {scope for scope in positions if scope not in result}
    proxies = sorted(pending)
    while pending:
        before = len(pending)
        for recipe in recipes:
            if recipe.get('predicate') not in {'ontop', 'inside'} or recipe.get('desired', True) is not True:
                continue
            subject = role_to_scope.get(recipe.get('subject_role'))
            reference = role_to_scope.get(recipe.get('object_role'))
            if subject in pending and reference in result:
                result[subject] = list(result[reference])
                pending.remove(subject)
        if len(pending) == before:
            raise ValueError('UNRESOLVED_AUTHORED_SUPPORT:' + ','.join(sorted(pending)))
    if any(len(xyz) != 3 or not all(math.isfinite(v) for v in xyz) for xyz in result.values()):
        raise ValueError('NON_FINITE_POSITION')
    return result, proxies


def load_tasks(package):
    registry = {row['candidate_id']: row for row in l0._read_jsonl(package / 'registry.jsonl')}
    tasks = []
    for registered in l0._read_json(package / 'runtime_registration/catalog.json')['tasks']:
        row = registry[registered['candidate_id']]
        contract = l0._read_json(Path(registered['template_contract']['path']))
        bindings = row['scope_bindings']
        raw = l0._scope_positions(template=l0._read_json(Path(registered['template']['path'])),
                                 tro=l0._read_json(Path(registered['tro_state']['path'])), bindings=bindings)
        positions, proxies = proxy_positions(raw, bindings, row['role_to_scope'],
                                            registered['construction']['initial_relation_recipe'])
        # No speculative initially_true_relations: that API requires live truth.
        # Preservation leaves remain in this conservative, unverified route DAG.
        plan = l0.compile_grounded_mixed_option(tuple(registered['profile']['canonical_goal_leaves']))
        tasks.append(dict(task_spec_id=row['candidate_id'], task_name=registered['activity_name'],
                          camera_pose_rules={contract['role_to_scope'][role]: rule for role, rule in
                                             contract.get('camera_relative_orientations', {}).items()},
                          family_id=registered['source_family_id'], scene_model=row['scene_model'],
                          rooms=row['rooms'], scope_bindings=bindings, positions=positions,
                          operations=l0._plan_operations(plan), visibility_roles=visibility_roles(bindings),
                          proxy_scopes=proxies, initial_goal_false_cpu=row['initial_goal_false_cpu']))
    return tasks


def route_candidates(task, maps):
    visits = l0._route_visits(task)
    if not visits:
        return {}, ['NO_GOAL_DAG_OPERATIONS']
    cell = l0.world_to_cell(visits[0]['xyz'][:2], resolution=l0.MAP_RESOLUTION_M, map_size=maps.map_size)
    radius = math.ceil(l0.MAXIMUM_STATION_DISTANCE_M / l0.MAP_RESOLUTION_M) + 2
    labels = sorted(set(int(x) for x in maps.labels[
        max(0, cell[0]-radius):min(maps.map_size, cell[0]+radius+1),
        max(0, cell[1]-radius):min(maps.map_size, cell[1]+radius+1)].flat) - {0})
    candidates, failures = {}, []
    for component in labels:
        chain = l0._compile_station_chain(task, maps, component)
        if chain['status'] != 'PASS':
            failures.extend(chain['failures'])
            continue
        for candidate in l0._candidate_records(task, maps, chain, route_only=True):
            candidate['chain'] = chain
            candidates[tuple(candidate['cell'])] = candidate
    return candidates, sorted(set(failures))


def joint_key(members, cell, yaw):
    """Optimize a shared pose without making foil co-view a hard gate."""
    calibration_only = any(
        candidate.get('calibration_only', False) for candidate in members
    )
    if calibration_only:
        # A measured co-focus margin is available only when every requested
        # source/foil AABB is in frame.  Missing is -inf, so fallback remains
        # eligible while a real co-view candidate wins the tie.
        margins = []
        for candidate in members:
            value = candidate.get('calibration_cofocus_frame_margin')
            margins.append(
                float(value) if isinstance(value, (int, float)) and math.isfinite(value)
                else -math.inf
            )
        return (
            -min(c['visible_relevant_instance_count'] for c in members),
            -min(margins),
            not all(c['under_l0_route_target'] for c in members),
            -min(c['visible_role_kind_count'] for c in members),
            max(c['total_path_length_m'] for c in members),
            sum(c['total_path_length_m'] for c in members), tuple(cell), yaw,
        )
    return (not all(c['under_l0_route_target'] for c in members),
            -min(c['visible_relevant_instance_count'] for c in members),
            -min(c['visible_role_kind_count'] for c in members),
            max(c['total_path_length_m'] for c in members),
            sum(c['total_path_length_m'] for c in members), tuple(cell), yaw)


def _eligible_joint_candidates(options):
    return [o for o in options if all(
        c.get('camera_pose_constraints_pass', True) and c['all_required_routes_exist'] and c['visible_relevant_instance_count'] > 0
        and c['visible_source_count'] + c['visible_destination_count'] > 0
        and c.get('calibration_focus_visible', True) for c in o['members'])]


def choose_joint_candidate(options):
    eligible = _eligible_joint_candidates(options)
    return min(eligible, key=lambda o: joint_key(o['members'], o['cell'], o['yaw'])) if eligible else None


def export_preflight_candidates(ranked, member_index, limit=16):
    """CPU proposals only; same family rank in every counterbalanced member."""
    result = []
    for rank, option in enumerate(ranked[:limit], 1):
        c = option['members'][member_index]
        yaw = math.radians(option['yaw'])
        legs = []
        for leg in c['chain']['legs']:
            alternatives = leg['station_audit'].get('candidate_cells', [leg['station_cell']])
            legs.append({**leg, 'station_candidate_cells': alternatives,
                         'exact_manipulation_check': 'NOT_RUN'})
        result.append(dict(rank=rank, cell=list(option['cell']), yaw_deg=option['yaw'],
            robot_pose=dict(position=[*c['world_xy'], .005],
                            orientation=[0., 0., math.sin(yaw / 2), math.cos(yaw / 2)]),
            initial_camera_view_state=dict(pitch_state='center', torso_height_state='neutral',
                torso_qpos=list(l0.EMPTY_TORSO_GRID[('neutral', 'center')]), initial_view_actions=[]),
            egress_world_directions=c['egress_world_directions'], route_legs=legs,
            route_leg_lengths_m=c['route_leg_lengths_m'],
            total_path_length_m=c['total_path_length_m'],
            visible_relevant_instance_count=c['visible_relevant_instance_count'],
            stowed_whole_body_check='NOT_RUN', manipulation_path_check='NOT_RUN',
            route_lengths_apply_to='PRIMARY_STATIONS_ONLY'))
    return result


class CandidateAcceptanceInfrastructureError(RuntimeError):
    """Live candidate acceptance failed closed before a visibility decision."""

    def __init__(self, report):
        self.report = dict(report)
        super().__init__(self.report.get('error', 'candidate acceptance failed'))


def _choose_joint_candidate_with_acceptance(options, acceptance, budget):
    """Try live calibration acceptance in the existing joint-rank order."""
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError('candidate acceptance budget must be a positive integer')
    if not callable(acceptance):
        raise TypeError('candidate acceptance callback must be callable')
    eligible = sorted(
        _eligible_joint_candidates(options),
        key=lambda option: joint_key(option['members'], option['cell'], option['yaw']),
    )
    attempts = []
    selected = None

    def selection_report():
        return {
            'schema_version': 'calibration-head-segmentation-candidate-selection/v1',
            'enabled': True,
            'candidate_budget': budget,
            'static_eligible_candidate_count': len(eligible),
            'tested_candidate_count': len(attempts),
            'budget_exhausted': False,
            'selected_candidate_rank': None,
            'candidates': attempts,
        }

    def record_attempt(option, rank, result):
        result = dict(result)
        attempt = {
            'rank': rank,
            'cell': [int(value) for value in option['cell']],
            'yaw_deg': float(option['yaw']),
            **result,
            'accepted': result.get('accepted') is True,
        }
        attempts.append(attempt)
        return attempt

    def raise_infrastructure_error(option, rank, result, cause=None):
        result = dict(result)
        record_attempt(option, rank, result)
        report = dict(result)
        report.update({
            'failure_class': 'CALIBRATION_HEAD_SEGMENTATION_ERROR',
            'candidate_rank': rank,
            'candidate_cell': [int(value) for value in option['cell']],
            'candidate_yaw_deg': float(option['yaw']),
            'tested_candidate_count': len(attempts),
            'candidate_acceptance_attempts': attempts,
            'head_segmentation_selection': selection_report(),
        })
        error = CandidateAcceptanceInfrastructureError(report)
        if cause is None:
            raise error
        raise error from cause

    for rank, option in enumerate(eligible[:budget], start=1):
        try:
            result = dict(acceptance(option, rank))
        except Exception as exc:
            raise_infrastructure_error(option, rank, {
                'accepted': False,
                'failure_reason': 'CANDIDATE_ACCEPTANCE_CALLBACK_ERROR',
                'error_type': type(exc).__name__,
                'error': str(exc),
            }, cause=exc)
        if result.get('accepted') is not True and result.get('failure_reason') != (
                'CALIBRATION_SOURCE_BELOW_MARKER_PIXEL_GATE'):
            raise_infrastructure_error(option, rank, result)
        attempt = record_attempt(option, rank, result)
        if attempt['accepted']:
            selected = option
            break
    return selected, {
        'schema_version': 'calibration-head-segmentation-candidate-selection/v1',
        'enabled': True,
        'candidate_budget': budget,
        'static_eligible_candidate_count': len(eligible),
        'tested_candidate_count': len(attempts),
        'budget_exhausted': selected is None and len(eligible) > budget,
        'selected_candidate_rank': (
            attempts[-1]['rank'] if selected is not None else None
        ),
        'candidates': attempts,
    }


def _usable_camera_capture(capture):
    if not (isinstance(capture, dict)
            and capture.get('schema_version') == CAMERA_CAPTURE_SCHEMA
            and isinstance(capture.get('intrinsic_matrix'), list)
            and isinstance(capture.get('image_shape'), list)
            and len(capture['image_shape']) == 2
            and all(isinstance(v, int) and v > 0 for v in capture['image_shape'])):
        return False
    try:
        intrinsic = np.asarray(capture['intrinsic_matrix'], dtype=float)
        return (intrinsic.shape == (3, 3) and np.isfinite(intrinsic).all()
                and intrinsic[0, 0] > 0 and intrinsic[1, 1] > 0
                and all(field in capture for field in (
                    'robot_position', 'robot_orientation', 'head_position', 'head_orientation')))
    except (TypeError, ValueError):
        return False


def _measured_visibility_at_start(*, task, maps, cell, yaw, capture):
    """Use the existing renderer projection against measured camera geometry."""
    from behavior_hl.embodied.evaluator_bridge.run_b1_marker_symbolic_rollout import (
        _private_aabb_corners,
        _project_world_points_to_head_pixels,
    )

    source_scopes = {operation['target'] for operation in task['operations']}
    destination_scopes = {
        operation['destination'] for operation in task['operations']
        if operation['destination'] is not None
        and not l0._is_floor(operation['destination'], task['scope_bindings'])
    }
    visibility_roles_by_scope = task.get('visibility_roles') or {}
    relevant = sorted(visibility_roles_by_scope or source_scopes | destination_scopes)
    world_xy = l0.cell_to_world(cell, resolution=l0.MAP_RESOLUTION_M, map_size=maps.map_size)
    sensor = measured_head_camera(capture, world_xy, yaw)
    camera_position, camera_orientation = camera_pose_at_robot_pose(capture, world_xy, yaw)
    camera_cell = l0.world_to_cell(
        camera_position[:2], resolution=l0.MAP_RESOLUTION_M, map_size=maps.map_size
    )
    forward = quaternion_to_matrix_xyzw(camera_orientation) @ [0.0, 0.0, -1.0]
    camera_yaw = math.degrees(math.atan2(float(forward[1]), float(forward[0])))
    camera_pitch = math.degrees(math.atan2(float(forward[2]), float(math.hypot(forward[0], forward[1]))))
    intrinsic = capture['intrinsic_matrix']
    image_height, image_width = (int(capture['image_shape'][0]), int(capture['image_shape'][1]))
    horizontal_fov = math.degrees(2.0 * math.atan(image_width / (2.0 * float(intrinsic[0][0]))))
    vertical_fov = math.degrees(2.0 * math.atan(image_height / (2.0 * float(intrinsic[1][1]))))
    visible = []
    projected_pixels = {}
    projected_aabb_pixels = {}
    structural_los = {}
    focus_scopes = sorted(
        scope for scope, roles_for_scope in visibility_roles_by_scope.items()
        if 'source' in roles_for_scope
    ) or sorted(source_scopes)
    calibration_only = bool(task.get('calibration_only', False))
    foil_scopes = sorted(
        scope for scope, roles_for_scope in visibility_roles_by_scope.items()
        if 'foil' in roles_for_scope
    )
    cofocus_scopes = (
        sorted(set(focus_scopes) | set(foil_scopes))
        if calibration_only else list(focus_scopes)
    )
    focus_aabb_visible = {}
    cofocus_aabb_visible = {}
    for scope in relevant:
        xyz = task['positions'][scope]
        target_cell = l0.world_to_cell(xyz[:2], resolution=l0.MAP_RESOLUTION_M, map_size=maps.map_size)
        clear = l0.has_structural_line_of_sight(
            maps.structural_grid, camera_cell, target_cell, allow_blocked_target_endpoint=True
        )
        structural_los[scope] = bool(clear)
        pixels = _project_world_points_to_head_pixels(
            sensor, [xyz], capture['image_shape']
        )
        if pixels is not None:
            projected_pixels[scope] = list(pixels[0])
        in_frame = pixels is not None
        if calibration_only and scope in cofocus_scopes:
            geometry = (task.get('measured_geometry') or {}).get(scope)
            if geometry is None:
                geometry = (capture.get('scope_geometry') or {}).get(scope)
            try:
                lower = np.asarray(geometry['aabb_min'], dtype=float)
                upper = np.asarray(geometry['aabb_max'], dtype=float)
                if (lower.shape != (3,) or upper.shape != (3,)
                        or not np.isfinite(lower).all() or not np.isfinite(upper).all()
                        or np.any(lower > upper)):
                    raise ValueError('invalid measured AABB')
                corners = _private_aabb_corners(lower, upper)
                aabb_pixels = _project_world_points_to_head_pixels(
                    sensor, corners, capture['image_shape']
                )
            except (IndexError, KeyError, TypeError, ValueError):
                aabb_pixels = None
            cofocus_aabb_visible[scope] = aabb_pixels is not None
            if scope in focus_scopes:
                focus_aabb_visible[scope] = aabb_pixels is not None
            if aabb_pixels is not None:
                projected_aabb_pixels[scope] = [list(point) for point in aabb_pixels]
            if scope in focus_scopes:
                in_frame = in_frame and aabb_pixels is not None
        if clear and in_frame:
            visible.append(scope)
    roles = sorted({role for scope in visible for role in visibility_roles_by_scope.get(scope, ())})
    def frame_margin(scopes):
        margins = []
        for scope in scopes:
            points = projected_aabb_pixels.get(scope)
            if points is None:
                return None
            width = float(image_width)
            height = float(image_height)
            margins.append(min(
                min(float(x) for x, _ in points),
                min(width - 1.0 - float(x) for x, _ in points),
                min(float(y) for _, y in points),
                min(height - 1.0 - float(y) for _, y in points),
            ) / max(width, height))
        return min(margins, default=None)

    focus_visible = all(scope in visible for scope in focus_scopes)
    focus_aabb_pass = all(focus_aabb_visible.get(scope, False) for scope in focus_scopes)
    cofocus_required = bool(calibration_only and foil_scopes)
    cofocus_visible = (
        all(scope in visible for scope in cofocus_scopes)
        and all(cofocus_aabb_visible.get(scope, False) for scope in cofocus_scopes)
    )
    return {
        'calibration_only': calibration_only,
        'yaw_deg': float(yaw),
        'camera_yaw_deg': camera_yaw,
        'camera_pitch_deg': camera_pitch,
        'camera_height_m': float(camera_position[2]),
        'horizontal_fov_deg': horizontal_fov,
        'vertical_fov_deg': vertical_fov,
        'visible_relevant_instance_count': len(visible),
        'visible_source_count': sum('source' in visibility_roles_by_scope.get(scope, ()) for scope in visible),
        'visible_destination_count': sum('destination' in visibility_roles_by_scope.get(scope, ()) for scope in visible),
        'visible_foil_count': sum('foil' in visibility_roles_by_scope.get(scope, ()) for scope in visible),
        'visible_landmark_count': sum('landmark' in visibility_roles_by_scope.get(scope, ()) for scope in visible),
        'visible_role_kind_count': len(roles),
        'predicted_visible_instance_ids': [task['scope_bindings'][scope]['actual_name'] for scope in visible],
        'predicted_visible_scope_names': visible,
        'measured_projected_pixels': projected_pixels,
        'measured_projected_aabb_pixels': projected_aabb_pixels,
        'measured_structural_los': structural_los,
        'visibility_evidence': 'LIVE_GEOMETRY_RENDER_CAMERA_PROJECTION',
        'focus_projection_evidence': 'WHOLE_AABB' if calibration_only else 'CENTER_ONLY',
        'calibration_focus_scopes': focus_scopes,
        'calibration_focus_visible': (not calibration_only or focus_visible),
        'calibration_focus_aabb_visible': (
            not calibration_only or focus_aabb_pass
        ),
        'calibration_cofocus_scopes': cofocus_scopes,
        'calibration_cofocus_required': cofocus_required,
        'calibration_cofocus_visible': (
            not calibration_only or not cofocus_required or cofocus_visible
        ),
        'calibration_cofocus_aabb_visible': (
            not calibration_only
            or all(cofocus_aabb_visible.get(scope, False) for scope in cofocus_scopes)
        ),
        'calibration_focus_frame_margin': (
            frame_margin(focus_scopes) if calibration_only else None
        ),
        'calibration_cofocus_frame_margin': (
            frame_margin(cofocus_scopes)
            if calibration_only and (not cofocus_required or cofocus_visible)
            else None
        ),
        'calibration_cofocus_fallback': bool(
            calibration_only and cofocus_required and not cofocus_visible
        ),
    }


def compile_family(tasks, maps, max_cells, *, candidate_acceptance=None,
                   candidate_acceptance_budget=16):
    routes = [route_candidates(task, maps) for task in tasks]
    common = set(routes[0][0]).intersection(*(set(c) for c, _ in routes[1:]))
    # ponytail: bounded CPU shortlist, not a global visibility optimum. Increase
    # max_cells or rerun on frozen geometry when proxy screening is insufficient.
    cells = sorted(common, key=lambda cell: (
        not all(r[0][cell]['under_l0_route_target'] for r in routes),
        max(r[0][cell]['total_path_length_m'] for r in routes), cell))[:max_cells]
    options = []
    step = l0.ACTION20_RESTRICTED_KINEMATIC_PUBLIC_TURN_DEG
    for cell in cells:
        for index in range(math.ceil(360 / step)):
            yaw = l0._wrapped_degrees(index * step)
            members = []
            for task, route in zip(tasks, routes):
                candidate = {
                    **route[0][cell],
                    **l0._visibility_at_start(task=task, maps=maps, cell=cell, fixed_yaw_deg=yaw),
                }
                capture = task.get('camera_pose_capture')
                if _usable_camera_capture(capture):
                    candidate.update(_measured_visibility_at_start(
                        task=task, maps=maps, cell=cell, yaw=yaw, capture=capture
                    ))
                elif capture:
                    candidate['visibility_evidence'] = 'CPU_PROXY_INCOMPLETE_MEASURED_CAPTURE'
                members.append(candidate)
            for task, candidate in zip(tasks, members):
                rules = task.get('camera_pose_rules', {})
                if rules:
                    capture = task.get('camera_pose_capture')
                    if not capture:
                        candidate['camera_pose_constraints_pass'] = False
                        candidate['camera_pose_failure'] = 'MEASURED_CAMERA_CAPTURE_REQUIRED'
                        continue
                    geometry = task['measured_geometry']
                    camera = camera_at_robot_pose(capture, candidate['world_xy'], yaw)
                    checks = camera_pose_readback(rules, camera,
                        lambda scope: (geometry[scope]['position'], geometry[scope]['orientation']))
                    candidate['camera_pose_constraints_pass'] = all(r['passed'] for r in checks)
                    candidate['camera_pose_readback'] = checks
            options.append(dict(cell=cell, yaw=yaw, members=members))
    if candidate_acceptance is None:
        selected = choose_joint_candidate(options)
        acceptance_summary = None
    else:
        selected, acceptance_summary = _choose_joint_candidate_with_acceptance(
            options, candidate_acceptance, candidate_acceptance_budget
        )
    diagnostic = selected or (min(options, key=lambda o: joint_key(o['members'], o['cell'], o['yaw'])) if options else None)
    ranked = sorted(_eligible_joint_candidates(options),
                    key=lambda o: joint_key(o['members'], o['cell'], o['yaw']))
    rows = []
    for index, task in enumerate(tasks):
        diagnostic_member = diagnostic['members'][index] if diagnostic else None
        focus_no_candidate = bool(
            not selected
            and task.get('calibration_only', False)
            and diagnostic_member
            and diagnostic_member.get('visibility_evidence') == 'LIVE_GEOMETRY_RENDER_CAMERA_PROJECTION'
            and not diagnostic_member.get('calibration_focus_visible', True)
        )
        reason = ([] if selected else ['NO_PROXY_VISIBLE_SHARED_POSE'] if common
                  else ['NO_SHARED_ROUTE_CANDIDATE'])
        if focus_no_candidate:
            reason = ['NO_CALIBRATION_FOCUS_IN_VIEW']
        if acceptance_summary is not None and not selected:
            if acceptance_summary['static_eligible_candidate_count']:
                reason = [
                    'CALIBRATION_HEAD_SEGMENTATION_BUDGET_EXHAUSTED'
                    if acceptance_summary['budget_exhausted']
                    else 'NO_CALIBRATION_HEAD_SEGMENTATION_CANDIDATE'
                ]
        if not selected and common and task.get('camera_pose_rules'):
            reason.append('NO_SHARED_POSE_PRESERVING_CAMERA_CUES')
        segmentation_no_candidate = bool(
            acceptance_summary is not None
            and not selected
            and acceptance_summary['static_eligible_candidate_count']
        )
        row = dict(task_spec_id=task['task_spec_id'], source_family_id=task['family_id'],
                   scene_model=task['scene_model'], selection_status='CPU_PROXY_CANDIDATE' if selected else 'CPU_PROXY_NO_CANDIDATE',
                   runtime_enabled=False, render_status='NOT_RUN', live_status='NOT_RUN',
                   evidence_level=('LIVE_GEOMETRY_RENDER_CAMERA_PROJECTION'
                                   if diagnostic and diagnostic['members'][index].get('visibility_evidence') == 'LIVE_GEOMETRY_RENDER_CAMERA_PROJECTION'
                                   else 'SUPPORT_CENTER_PROXY_NOT_CONSTRUCTED_GEOMETRY'),
                   proxy_scope_names=task['proxy_scopes'], failure_reasons=reason,
                   no_candidate_diagnostic=(
                       {
                           'code': 'NO_CALIBRATION_FOCUS_IN_VIEW',
                           'focus_scopes': diagnostic_member.get('calibration_focus_scopes', []),
                           'focus_projection_evidence': diagnostic_member.get('focus_projection_evidence'),
                           'focus_aabb_visible': diagnostic_member.get('calibration_focus_aabb_visible'),
                       }
                       if focus_no_candidate else (
                           {
                               'code': (
                                   'CALIBRATION_HEAD_SEGMENTATION_BUDGET_EXHAUSTED'
                                   if acceptance_summary['budget_exhausted']
                                   else 'NO_CALIBRATION_HEAD_SEGMENTATION_CANDIDATE'
                               ),
                               'candidate_budget': acceptance_summary['candidate_budget'],
                               'tested_candidate_count': acceptance_summary['tested_candidate_count'],
                               'budget_exhausted': acceptance_summary['budget_exhausted'],
                           }
                           if segmentation_no_candidate else None
                       )
                   ),
                   station_failure_reasons=routes[index][1], shared_feasible_cell_count=len(common),
                   evaluated_cell_count=len(cells), complete_search=len(cells) == len(common),
                   goal_dag_operations=task['operations'], route_model='ALL_GOAL_LEAVES_NO_LIVE_INITIAL_TRUTH',
                   goal_initial_false_cpu=task['initial_goal_false_cpu'], goal_initial_false_live='NOT_RUN',
                   fixed_robot_pose=None, best_candidate_diagnostic=None,
                   selection_reason='Shared cell/yaw across family; soft 3m/6m preference; worst-member visibility, roles, route length, coordinates.')
        if acceptance_summary is not None:
            row['head_segmentation_selection'] = acceptance_summary
        row['oracle_preflight'] = dict(
            schema_version='action20.cpu-preflight-candidates/v1',
            status='PENDING_SCENE_LOADED_PREFLIGHT',
            candidate_limit=16, eligible_candidate_count=len(ranked),
            exported_candidates_complete=len(ranked) <= 16,
            map_resolution_m=l0.MAP_RESOLUTION_M, map_size=maps.map_size,
            candidates=export_preflight_candidates(ranked, index),
            exact_collision_checks='NOT_RUN',
            final_pose_frozen_by_this_compiler=False)
        if diagnostic:
            c = diagnostic['members'][index]
            yaw = math.radians(diagnostic['yaw'])
            pose = dict(position=[*c['world_xy'], .005], orientation=[0., 0., math.sin(yaw/2), math.cos(yaw/2)])
            row['fixed_robot_pose'] = pose if selected else None
            row['initial_camera_view_state'] = dict(pitch_state='center', torso_height_state='neutral',
                torso_qpos=list(l0.EMPTY_TORSO_GRID[('neutral', 'center')]), initial_view_actions=[],
                camera_pitch_deg=c['camera_pitch_deg'])
            row['best_candidate_diagnostic'] = {k: v for k, v in c.items() if k != 'chain'}
            row['best_candidate_diagnostic']['pose'] = pose
            row['route_legs'] = [{**leg, 'path_length_m': length} for leg, length in
                                 zip(c['chain']['legs'], c['route_leg_lengths_m'])]
            row['total_path_length_m'] = c['total_path_length_m']
            row['total_move015_lower_bound'] = math.ceil(c['total_path_length_m'] / l0.RESTRICTED_KINEMATIC_PUBLIC_MOVE_M)
            row['full_map_route_status'] = 'PASS_ON_PROXY'
            row['partial_map_route'] = l0._partial_route_report(task, maps,
                [tuple(c['cell']), *(tuple(leg['station_cell']) for leg in c['chain']['legs'])])
            row['under_3m_6m_targets'] = c['under_l0_route_target']
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--assets-root', type=Path, required=True)
    parser.add_argument('--robot-metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-cells', type=int, default=64)
    args = parser.parse_args()
    if args.max_cells < 1:
        parser.error('--max-cells must be positive')
    if args.output.exists():
        parser.error('output must be a new directory; existing artifacts are not overwritten')
    groups = defaultdict(list)
    for task in load_tasks(args.package):
        groups[task['family_id']].append(task)
    maps_by_scene, rows = {}, []
    for family, tasks in sorted(groups.items()):
        scene = tasks[0]['scene_model']
        if any(t['scene_model'] != scene for t in tasks):
            raise ValueError('family spans scenes: ' + family)
        if scene not in maps_by_scene:
            maps_by_scene[scene] = l0.SceneMaps(scene_model=scene, assets_root=args.assets_root,
                room_categories_path=args.assets_root / 'metadata/room_categories.txt',
                robot_metadata_path=args.robot_metadata)
        rows.extend(compile_family(tasks, maps_by_scene[scene], args.max_cells))
        print(f'{family}: {len(tasks)} members complete', flush=True)
    blocked = [dict(source_family_id=r['family_id'], selection_status='CONSTRUCTION_NOT_READY',
                    failure_reasons=sorted({b for m in r['members'] for b in m['blockers']}))
               for r in l0._read_jsonl(args.package / 'construction_results.jsonl') if r['family_id'] not in groups]
    selected = [r for r in rows if r['selection_status'] == 'CPU_PROXY_CANDIDATE']
    summary = dict(runtime_members=len(rows), runtime_families=len(groups), blocked_families=len(blocked),
        selected_proxy_members=len(selected), selected_proxy_families=len({r['source_family_id'] for r in selected}),
        formally_enabled_poses=0, live_status='NOT_RUN', max_cells=args.max_cells,
        failures=dict(Counter(x for r in rows for x in r['failure_reasons'])),
        proxy_visibility_distribution=dict(Counter(r['best_candidate_diagnostic']['visible_relevant_instance_count'] for r in selected)),
        under_3m_6m_targets=sum(r['under_3m_6m_targets'] for r in selected),
        full_pass_partial_fail=sum(r.get('partial_map_route', {}).get('status') == 'FAIL' for r in selected),
        limitations=['Support centers collapse authored object separation; not RGB or mesh visibility.',
            'All goal leaves retained without live initial truth; preservation routes can overestimate length.',
            'Bounded shortest-route cell shortlist; greedy shared stations; no global optimum claim.',
            'Official static map excludes authored geometry; collision, egress and goal require live attestation.',
            'Room count is not proof of room adjacency.',
            'Freeze construction geometry before applying a pose; camera-relative placement must not be rerun.',
            'Visual Variants inherit their Base pose; no Variant rendering or registration performed.'])
    args.output.mkdir(parents=True)
    for name, value in [('candidates.jsonl', rows), ('blocked_families.jsonl', blocked)]:
        (args.output / name).write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in value))
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    # Additive handoff: retain the original package's valid absolute references.
    # This field describes required integration, not an implemented Oracle gate.
    package = l0._read_json(args.package / 'package.json')
    candidate_path = (args.output / 'candidates.jsonl').resolve()
    package['oracle_preflight'] = dict(required=True, integration_status='NOT_CONNECTED',
        status='PENDING_SCENE_LOADED_PREFLIGHT',
        candidates=dict(path=str(candidate_path), sha256=l0._sha256(candidate_path)),
        source_package=dict(path=str((args.package / 'package.json').resolve()),
                            sha256=l0._sha256(args.package / 'package.json')))
    package['live_run_ready'] = False
    package['live_blockers'] = list(dict.fromkeys([
        *package.get('live_blockers', []), 'SCENE_LOADED_WHOLE_BODY_AND_ARM_PREFLIGHT_REQUIRED']))
    (args.output / 'package.json').write_text(json.dumps(package, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
