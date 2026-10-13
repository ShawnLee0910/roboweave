"""Current-scene pose admission, reused from cfb585f without untuck or IK."""
import time
from dataclasses import asdict

PRESERVE_POSE = "PRESERVE_POSE_THEN_LIVE_ADMISSION"

def admit_prepared_pose(evaluator, candidate, *, expected_room, initial_state_valid, deadline):
    """Read current geometry and full collision state; never untuck or plan."""
    import numpy as np
    from .prepared_model_state import array
    from .nearby_initial_pose import candidate_admission, measured_base_admission
    from .construction_camera_pose import capture_head_camera_pose
    from .run_b1_marker_symbolic_rollout import _build_full_motion_safety_starter, _private_room_at_xy
    from .r1pro_curobo_collision_provider import R1ProCuRoboCurrentQCollisionProvider
    from .r1pro_live_safe_carry_binding import read_r1pro_live_robot_carry_state
    from .moderate_inward_gripper_presentation import (
        _state_report, MINIMUM_CENTER_CLEARANCE_M, MINIMUM_EEF_SEPARATION_M,
    )
    robot=evaluator.robot
    report={'status':'FAIL', 'policy':PRESERVE_POSE, 'untucked_preamble_applied':False,
            'inward_planning':False, 'admission_checks':{}}
    def require(name, passed):
        report['admission_checks'][name]=bool(passed)
        if not passed:
            error=RuntimeError('prepared pose rejected: '+name)
            error.private_report=dict(report, failure_type='PREPARED_POSE_REJECTED')
            raise error
    position, orientation=robot.get_position_orientation()
    p=array(position,(3,),'current robot position');q=array(orientation,(4,),'current robot quaternion')
    joints=array(robot.get_joint_positions(),(28,),'current robot joints')
    lower=array(robot.joint_lower_limits,(28,),'joint lower limits')
    upper=array(robot.joint_upper_limits,(28,),'joint upper limits')
    require('finite_nonzero_orientation', np.linalg.norm(q)>0)
    require('joint_limits', np.all(lower<=joints) and np.all(joints<=upper))
    xy=tuple(p[:2])
    room=candidate_admission(xy,xy,expected_room=expected_room,
        room_lookup=lambda point:_private_room_at_xy(evaluator.env.scene,point),
        static_check=lambda _:measured_base_admission(evaluator.env.scene,robot))
    report['base_admission']=room
    require('room_and_static_base', room.get('accepted') is True)
    require('initial_semantics', initial_state_valid() is True)
    starter=_build_full_motion_safety_starter(evaluator)
    hands=[]
    for hand in ('left','right'):
        starter._selected_arm=hand
        require('empty_'+hand, starter._get_obj_in_hand() is None)
        hands.append(dict(hand=hand,after=_state_report(read_r1pro_live_robot_carry_state(starter,hand))))
    left,right=(np.asarray(h['after']['eef_body_position_m']) for h in hands)
    require('gripper_centre_clearance', left[1]>=MINIMUM_CENTER_CLEARANCE_M and right[1]<=-MINIMUM_CENTER_CLEARANCE_M)
    require('gripper_separation', np.linalg.norm(left-right)>=MINIMUM_EEF_SEPARATION_M)
    collision=R1ProCuRoboCurrentQCollisionProvider(evaluator=evaluator,starter=starter)
    try:
        colliding=collision()
    except Exception as error:
        error.session_invalid=True  # Unknown collision world is not another pose candidate.
        raise
    report['full_body_collision']=asdict(collision.last_audit)
    require('full_body_and_self_collision', not colliding)
    camera=capture_head_camera_pose(evaluator)
    intrinsic=array(camera['intrinsic_matrix'],(3,3),'camera intrinsic')
    require('camera_projection', intrinsic[0,0]>0 and intrinsic[1,1]>0 and bool(camera.get('image_shape')))
    report['head_camera_readback']=camera
    require('initial_semantics_after_admission', initial_state_valid() is True)
    if time.monotonic()>=deadline:
        error=RuntimeError('prepared pose admission deadline exhausted')
        error.private_report=dict(report,failure_type='PREPARATION_DEADLINE_EXHAUSTED')
        raise error
    joints=array(robot.get_joint_positions(),(28,),'post-admission joints')
    p=array(robot.get_position_orientation()[0],(3,),'post-admission position')
    report.update(status='PASS', hands=hands, numerical_deltas_are_diagnostic_only=True,
        maximum_joint_readback_delta=float(np.max(np.abs(joints-np.asarray(candidate['prepared']['robot_joint_positions'])))),
        position_readback_delta_m=float(np.linalg.norm(p-np.asarray(candidate['prepared']['robot_position']))))
    return report
