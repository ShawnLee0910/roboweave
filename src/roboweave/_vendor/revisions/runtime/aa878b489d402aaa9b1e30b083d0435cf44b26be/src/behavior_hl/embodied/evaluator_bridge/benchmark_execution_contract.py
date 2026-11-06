"""CPU-only execution policy resolution shared by campaign and worker."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from behavior_hl.embodied.evaluator_bridge.predeclared_camera_config import THREE_RGB_HEAD_EVAL_CAMERA_PROFILE
from behavior_hl.embodied.evaluator_bridge.gripper_visual_profiles import GRIPPER_VISUAL_PROFILE_INWARD_12CM_OFFICIAL_CAMERA


def execution_contract_sha256(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def runner_identity(source_root: Path) -> dict[str, Any]:
    from behavior_hl.embodied.evaluator_bridge.run_b1_single_load import capture_source_freeze
    freeze = capture_source_freeze(source_root)
    manifest = source_root / 'SOURCE_IDENTITY.json'
    if manifest.is_file():
        identity = json.loads(manifest.read_text())
        return {'commit': identity['upstream_revision'],
                'aggregate_sha256': freeze['aggregate_sha256'],
                'candidate_source_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                'candidate_adaptations': identity['candidate_adaptations']}
    top = subprocess.run(['git', '-C', str(source_root), 'rev-parse', '--show-toplevel'],
                         check=True, capture_output=True, text=True).stdout.strip()
    if Path(top).resolve() != source_root.resolve():
        raise ValueError('runner source root is not an independent checkout')
    commit = subprocess.run(['git', '-C', str(source_root), 'rev-parse', 'HEAD'],
                            check=True, capture_output=True, text=True).stdout.strip()
    return {'commit': commit, 'aggregate_sha256': freeze['aggregate_sha256']}


def resolve_execution_contract(args: Any, *, package_path: Path,
                               selected_task_ids: Sequence[str],
                               runner_source_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve the actual launcher namespace; never substitute campaign defaults."""
    package_path = package_path.resolve()
    raw = package_path.read_bytes()
    package = json.loads(raw)
    construction_only = bool(getattr(args, 'compile_initial_state', False))
    direct = bool(getattr(args, 'construct_and_oracle', False))
    model_only = bool(getattr(args, 'model_only_debug', False))
    physical_human_service = bool(getattr(args, 'physical_human_service', False))
    prepare_model = bool(getattr(args, 'prepare_model_state', False))
    if prepare_model and not (model_only and getattr(args, 'privileged_validation', False)):
        raise ValueError('model preparation requires privileged model-only validation')
    physical = bool(getattr(args, 'placement_physical_consequences', False))
    if physical and (construction_only or not (direct or model_only)):
        raise ValueError('Physical placement requires package-backed Oracle or model execution')
    if physical_human_service and not (physical and direct and not construction_only and not model_only):
        raise ValueError('Physical human service requires direct fresh Physical construction')
    if sum((construction_only, direct, model_only)) > 1:
        raise ValueError('execution modes are mutually exclusive')
    mode = ('CONSTRUCTION_ONLY' if construction_only else 'PHYSICAL_HUMAN_SERVICE' if physical_human_service
            else 'CONSTRUCT_AND_ORACLE' if direct
            else 'MODEL_ONLY_DEBUG' if model_only else 'ORACLE')
    if package.get('purpose') == 'CALIBRATION_ONLY' and not construction_only:
        raise ValueError('calibration package cannot execute Oracle')
    if package.get('purpose') == 'TASK_CONSTRUCTION' and not (direct or construction_only or model_only):
        raise ValueError('TASK_CONSTRUCTION requires construct-and-oracle or construction-only')
    allowed = package.get('allowed_execution_modes')
    if allowed is not None and mode not in allowed:
        raise ValueError(f'package execution mode conflict: {mode}')
    selected = list(selected_task_ids)
    members = package.get('task_spec_ids', [])
    if not selected or len(selected) != len(set(selected)) or any(i not in members for i in selected):
        raise ValueError('invalid execution contract task selection')
    timeout = getattr(args, 'oracle_timeout_seconds', None)
    if timeout is None and not (construction_only or model_only or physical_human_service):
        timeout = 1800.0
    if timeout is not None:
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Oracle timeout must be finite and positive')
    construction_timeout = getattr(args, 'initial_construction_timeout_seconds', None)
    if construction_timeout is not None:
        construction_timeout = float(construction_timeout)
        if not math.isfinite(construction_timeout) or construction_timeout <= 0:
            raise ValueError('construction timeout must be finite and positive')
    overlay = getattr(args, 'reset_pose_overlay', None)
    overlay_record = None
    if overlay is not None:
        path = Path(overlay).resolve()
        overlay_record = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    contract = {
        'benchmark_package': {'path': str(package_path), 'sha256': hashlib.sha256(raw).hexdigest()},
        'selected_task_spec_ids': selected,
        'execution_mode': mode,
        'action_mode': 'Physical' if physical else 'Symbolic',
        # Physical control / preparation / placement counts vary by action.
        'physics_steps_per_action': None if physical else 0,
        'scene_load': 'Partial' if getattr(args, 'partial_scene_load', False) else 'Full',
        'seed': int(args.seed),
        'navigation_lane': 'se2_navigation_leg_v1' if getattr(args, 'registered_public_action20_marker_se2_navigation', False) else 'legacy_xy_recovery_v1',
        'navigation_flags': {key: bool(getattr(args, key, False)) for key in (
            'registered_public_action20_marker_navigation', 'registered_public_action20_marker_gt_navigation',
            'registered_public_action20_marker_se2_navigation', 'registered_admission_double_gate')},
        'camera_profile': THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
        'gripper_visual_profile': GRIPPER_VISUAL_PROFILE_INWARD_12CM_OFFICIAL_CAMERA,
        'nearby_initial_pose_search': bool(getattr(args, 'nearby_initial_pose_search', False)),
        'reset_pose_overlay': overlay_record,
        'initial_pose_source': ('RESET_POSE_OVERLAY' if overlay_record else
                                'FROZEN_PREPARED_MODEL_STATE' if model_only and not prepare_model else 'TASK_CONSTRUCTION'),
        'oracle_timeout_seconds_per_task': timeout,
        'scene_loading_counts_toward_timeout': False,
        'initial_construction_simulator_steps_accounted_separately': True,
        'runner_source_identity': dict(runner_source_identity or {}),
    }
    # Preserve legacy execution identities unless the independent budget is requested.
    if construction_timeout is not None:
        contract['initial_construction_timeout_seconds'] = construction_timeout
    requested = package.get('requested_execution_policy', {})
    if requested.get('action_mode') == 'PhysicalPlacement':
        frozen = package.get('initial_state_compilation', {})
        if not physical or frozen.get('reset_mode') != 'RESTORE_AND_VERIFY_NO_RECONSTRUCTION':
            raise ValueError('legacy PhysicalPlacement requires Physical frozen restore')
        contract.update(
            initial_predicate_mode='action20_symbolic',
            placement_physical_consequences=True,
            stop_after_first_placement=False,
            execution_fidelity='kinematic_transport_with_physical_release',
            legacy_requested_execution_policy=dict(requested),
        )
        requested = dict(requested)
        requested['action_mode'] = 'Physical'
        if requested.get('physics_steps_per_action') == 'ACTION_DEPENDENT':
            requested['physics_steps_per_action'] = None
    for key, value in requested.items():
        if key == 'enforced_by_this_compiler':
            continue
        # Capture-only does not execute Oracle; its timeout request is inapplicable.
        if (construction_only or model_only) and key == 'oracle_timeout_seconds_per_task':
            continue
        if key not in contract or contract[key] != value:
            raise ValueError(f'package execution policy conflict: {key}: requested={value!r}, actual={contract.get(key)!r}')
    if physical:
        from .physical_ontop_drop import POLICY, CLEARANCE_M, MAX_RELEASE_QUERIES
        contract['physical_ontop_policy'] = POLICY
        contract['physical_robot_approach'] = False
        contract['release_clearance_m'] = CLEARANCE_M
        contract['release_height_reference'] = 'SELECTED_TARGET_SURFACE_OR_TARGET_TOP_FOR_GAP'
        contract['release_overlap_height_reference'] = 'LIVE_BLOCKING_OBJECT_AABB_TOP_PLUS_CLEARANCE'
        contract['maximum_release_queries'] = MAX_RELEASE_QUERIES
        contract['physical_center_mark'] = True
        contract['physical_landing_policy'] = 'TOWARD_SUPPORT_CENTER_WITHIN_SELECTED_REGION'
        contract['physical_oracle_region_policy'] = 'PHYSICAL_CENTER_THEN_INWARD'
    if package.get('required_qualification') == 'physical_template_review':
        if not physical:
            raise ValueError('physical template qualification requires Physical execution')
        contract['required_qualification'] = 'physical_template_review'
        stop_first = package.get('physical_batch_stop_after_first_failure', True)
        if type(stop_first) is not bool:
            raise ValueError('Physical batch stop policy must be boolean')
        contract['physical_batch_stop_after_first_failure'] = stop_first
    return contract
