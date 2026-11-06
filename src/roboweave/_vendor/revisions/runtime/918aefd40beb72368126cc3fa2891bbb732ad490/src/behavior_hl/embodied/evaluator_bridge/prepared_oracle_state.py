"""Hash-bound cross-task prepared robot/camera input for frozen-layout Oracles."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from .prepared_model_state import PreparedModelStateError, validate_bundle, restore_prepared_model_state

CAPABILITY = 'FROZEN_PREPARED_ORACLE_STATE_V1'
PROFILE = 'r1pro_three_rgb_head_eval'


def read_reference(ref):
    path = Path(ref['path'])
    if not path.is_absolute() or not path.is_file() or path.stat().st_size > 2*1024**2:
        raise ValueError('prepared Oracle reference path/size invalid')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('prepared Oracle evidence hash mismatch')
    return json.loads(raw)


def load_prepared_oracle_states(ref, records):
    index = read_reference(ref)
    if index.get('schema_version') != 'frozen-prepared-oracle-states/v1':
        raise ValueError('prepared Oracle index schema mismatch')
    tasks = {r['task_spec_id']: r for r in records}
    result = {}
    for item in index['items']:
        name = item['task_spec_id']
        if name not in tasks or name in result:
            raise ValueError('prepared Oracle task membership mismatch')
        task = tasks[name]
        if (item['scene'] != task['scene_model']
                or item['target_template_contract_sha256'] != task['template_contract']['sha256']
                or not task.get('construction', {}).get('frozen_initial_state')):
            raise ValueError('prepared Oracle target frozen contract mismatch')
        donor = read_reference(item['donor_success'])
        prepared = read_reference(item['prepared'])
        camera = read_reference(item['camera_structure'])
        tuning = read_reference(item['camera_tuning'])
        construction = read_reference(item['donor_initial_construction'])
        if (donor.get('status') != 'PASS' or type(donor.get('official_q')) not in (int,float)
                or donor['official_q'] != 1 or donor.get('task_success', donor.get('official_success')) is not True
                or donor.get('task_spec_id') != item['donor_task_spec_id']
                or prepared.get('task_name') != item['donor_task_spec_id']
                or donor.get('observation_profile') != PROFILE):
            raise ValueError('prepared Oracle donor success/identity mismatch')
        capture = construction['initial_state_capture']
        if capture['scene_document']['init_info']['args']['scene_model'] != item['scene']:
            raise ValueError('prepared Oracle donor scene mismatch')
        geometry = task['construction']['frozen_initial_state']['scope_geometry']
        mapping = item['new_scope_to_donor_scope']
        if set(mapping) != set(geometry):
            raise ValueError('prepared Oracle geometry membership mismatch')
        for scope, previous in mapping.items():
            # Serialized source identity check, not a Live numerical pose gate.
            expected = capture['scope_geometry'][previous]
            if any(geometry[scope][k] != expected[k] for k in ('position','orientation','aabb_min','aabb_max')):
                raise ValueError('prepared Oracle frozen geometry differs from donor')
        prepared = deepcopy(prepared)
        prepared['task_name'] = name  # Only the in-memory adapter is rebound; raw donor evidence is immutable.
        bundle = dict(schema_version='oracle-prepared-model-state/v1', instruction_id=name, seed=item['seed'],
            gripper_visual_profile='inward_12cm_head_dominant_official_camera_v1',
            execution_contract=dict(action_mode='Symbolic', physics_steps_per_action=0, scene_load='Full',
                observation_profile=PROFILE, overhead_view_mode='off'),
            prepared=prepared, camera_structure=camera, camera_tuning=tuning)
        validate_bundle(bundle, instruction_id=name, seed=item['seed'], observation_profile=PROFILE)
        if type(item['seed']) is not int:
            raise ValueError('prepared Oracle seed must be integer')
        result[name] = {'bundle': bundle, 'donor_task_spec_id': item['donor_task_spec_id']}
    if set(result) != set(tasks):
        raise ValueError('prepared Oracle index must cover the selected catalog')
    return result


def restore_prepared_oracle_state(*, evaluator, entry, task_id, seed, expected_room, semantic_check):
    """Restore only robot/camera, then re-admit. No alternate-pose fallback."""
    from .prepared_pose_admission import admit_prepared_pose
    deadline = time.monotonic() + 600.0
    bundle = entry['bundle']
    try:
        restored = restore_prepared_model_state(evaluator=evaluator, bundle=bundle,
            instruction_id=task_id, seed=seed, observation_profile=PROFILE, semantic_check=semantic_check)
        admitted = admit_prepared_pose(evaluator, {'prepared': bundle['prepared']},
            expected_room=expected_room, initial_state_valid=semantic_check, deadline=deadline)
        return dict(admitted, policy=CAPABILITY, robot_camera_restore=restored,
            donor_task_spec_id=entry['donor_task_spec_id'], donor_success_inherited=False,
            untucked_preamble_applied=False, inward_planning=False, pose_search=False)
    except Exception as exc:
        exc.session_invalid = True
        raise
