"""External V mutations, outside the Agent action/history stream.

Spatial effects replay a canonical staging pose measured in this episode.
They never invoke an Agent placement action or invent a landing coordinate.
"""
from copy import deepcopy

import numpy as np

from .spatial_symbolic_actions import object_pose, state_by_name, canonical_relation, read_action_relation


def _physical(runtime):
    return bool(getattr(runtime, 'placement_physical_consequences_enabled', False))


def _on_top(runtime, target, staging):
    if _physical(runtime):
        return canonical_relation(target, 'OnTop', staging)
    return read_action_relation(target, 'OnTop', staging, predicate_mode='action20_symbolic')


def capture(runtime, audit, effect):
    """Called only after a completed Agent action; None means no staging sample."""
    if effect['kind'] != 'RESTORE_STAGING_POSE':
        return None
    target = audit.role_object(effect['target_role'])
    staging = audit.role_object(effect['staging_role'])
    if runtime.symbolic_primitives._get_obj_in_hand() is not None:
        return None
    if not _on_top(runtime, target, staging):
        return None
    position, orientation = object_pose(target)
    support_position, support_orientation = object_pose(staging)
    return dict(position=position.tolist(), orientation=orientation.tolist(),
                staging_position=support_position.tolist(), staging_orientation=support_orientation.tolist(),
                target_name=target.name, staging_name=staging.name)


def apply(runtime, audit, effect, saved_pose=None):
    """Return mutation evidence. Any failure invalidates the V episode, not M history."""
    target = audit.role_object(effect['target_role'])
    if runtime.symbolic_primitives._get_obj_in_hand() is not None:
        raise RuntimeError('V intervention requires empty hands')
    record = dict(actor='INTERVENTION', effect=deepcopy(effect), physical=_physical(runtime))
    if effect['kind'] == 'UNARY_STATE':
        name = {'open': 'Open', 'toggled_on': 'ToggledOn'}[effect['predicate']]
        state = state_by_name(target, name)
        if state is None:
            raise RuntimeError('V state unavailable')
        if name == 'Open':
            if state.relevant_joints_info[0]:
                raise RuntimeError('V bidirectional door needs explicit direction')
            applied = state.set_value(effect['value'], fully=True)
        else:
            applied = state.set_value(effect['value'])
        if applied is not True:
            raise RuntimeError('V state setter failed')
        target.clear_states_cache()
        if bool(state.get_value()) != effect['value']:
            raise RuntimeError('V state readback mismatch')
    elif effect['kind'] == 'RESTORE_STAGING_POSE':
        staging = audit.role_object(effect['staging_role'])
        if (not saved_pose or saved_pose['target_name'] != target.name
                or saved_pose['staging_name'] != staging.name):
            raise RuntimeError('V needs same-episode verified staging pose')
        # Preserve the measured local placement if the support moved. This is a
        # rigid transform, not a guessed surface height or pose-equality gate.
        from .on_top_support_layers import quaternion_to_matrix_xyzw
        from scipy.spatial.transform import Rotation
        old_p = np.asarray(saved_pose['staging_position'])
        old_r = quaternion_to_matrix_xyzw(saved_pose['staging_orientation'])
        new_p, new_q = object_pose(staging)
        new_r = quaternion_to_matrix_xyzw(new_q)
        position = new_p + new_r @ old_r.T @ (np.asarray(saved_pose['position']) - old_p)
        orientation = Rotation.from_matrix(new_r @ old_r.T @ quaternion_to_matrix_xyzw(saved_pose['orientation'])).as_quat()
        if not np.isfinite(position).all() or not np.isfinite(orientation).all():
            raise RuntimeError('V invalid restored pose')
        import torch
        target.set_position_orientation(position=torch.as_tensor(position, dtype=torch.float32),
                                        orientation=torch.as_tensor(orientation, dtype=torch.float32))
        target.keep_still()
        record['source_staging_sample'] = deepcopy(saved_pose)
        record['applied_pose'] = dict(position=position.tolist(), orientation=orientation.tolist())
        if _physical(runtime):
            # Same 30-step release window and final 10 samples as Physical OnTop.
            target.enable_gravity()
            samples = []
            for step in range(30):
                runtime.environment.step_internal_once()
                target.clear_states_cache()
                staging.clear_states_cache()
                object_pose(target)  # validates finite pose
                if step >= 20:
                    samples.append(dict(step=step + 1,
                        ontop=bool(canonical_relation(target, 'OnTop', staging)),
                        touching=bool(canonical_relation(target, 'Touching', staging))))
            record.update(physics_steps=30, stability_samples=samples)
            if not all(sample['ontop'] and sample['touching'] for sample in samples):
                raise RuntimeError('V Physical staging did not settle on support')
        target.clear_states_cache()
        staging.clear_states_cache()
        if not _on_top(runtime, target, staging):
            raise RuntimeError('V staging relation readback failed')
    else:
        raise ValueError('unsupported V effect')
    runtime.store.clear()
    return record
