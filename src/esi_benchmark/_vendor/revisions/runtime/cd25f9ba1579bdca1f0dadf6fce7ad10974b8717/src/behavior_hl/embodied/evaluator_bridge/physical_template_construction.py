"""Initial-only P12 fixture setup using existing native assets and support queries."""
import numpy as np

from .physical_template_fixtures import incline_pose, apply_scene_material
from .on_top_support_layers import read_base_aligned_box
from .spatial_symbolic_actions import object_aabb, object_pose


def construct_fixtures(evaluator, control, bindings, task, role_object):
    import omnigibson as og
    import torch
    from omnigibson.utils.sampling_utils import raytest_batch
    from .registered_l0_task_runtime import _settle, state_by_name
    from .run_b1_marker_symbolic_rollout import _action20_upward_support_hit
    from .physical_ontop_drop import release_bounds
    from .spatial_symbolic_actions import SpatialPoseCandidate
    contract=task.record['physical_template']; template=contract['task'];kind=template['id']
    report={'status':'PASS','fixtures':[],'direct_state_modification':True,'source_assets_modified':False}
    def surface_z(obj,xy):
        lo,hi=object_aabb(obj)
        hit=raytest_batch(torch.tensor([[*xy,float(hi[2])+1]],dtype=torch.float32),
            torch.tensor([[*xy,float(lo[2])-1]],dtype=torch.float32),only_closest=False)
        hits=hit[0] if isinstance(hit[0],(tuple,list)) else [hit[0]]
        candidates=[p for h in hits if (p:=_action20_upward_support_hit(h,support_paths=tuple(obj.link_prim_paths))) is not None]
        if not candidates:raise RuntimeError('P12 fixture anchor ray missed actual support')
        return max(float(p[2]) for p in candidates)
    def set_at_surface(obj,xy,z,orientation=None):
        _,q=object_pose(obj)
        q=q if orientation is None else np.asarray(orientation,dtype=float)
        c=SpatialPoseCandidate('p12_initial',np.r_[xy,z],q)
        lo,hi=release_bounds(obj,c)
        c.position[:2]+=np.asarray(xy)-(lo[:2]+hi[:2])/2
        c.position[2]+=z+0.003-lo[2]
        obj.set_position_orientation(position=torch.tensor(c.position,dtype=torch.float32),
            orientation=torch.tensor(q,dtype=torch.float32))
        obj.keep_still()
    staging=role_object('staging');lo,hi=object_aabb(staging);center=(lo[:2]+hi[:2])/2
    dynamic=template.get('dynamic_propped_board')
    if dynamic:
        from .physical_template_fixtures import propped_board_pose
        from .spatial_symbolic_actions import canonical_relation
        board=role_object(dynamic['board_role']);prop=role_object(dynamic['prop_role'])
        base=role_object(dynamic['support_role']);blo,bhi=object_aabb(board);anchor=(blo[:2]+bhi[:2])/2
        if board.kinematic_only or prop.kinematic_only:raise RuntimeError('propped fixture must be dynamic')
        box=read_base_aligned_box(board)
        plo,phi=object_aabb(prop);height=float(phi[2]-plo[2])
        pos,quat,prop_xy=propped_board_pose(box.extent_local,box.center_local,anchor,surface_z(base,anchor),height)
        # Lift the board clear before placing the prop; no simulation occurs
        # between these initial construction edits.
        board.set_position_orientation(position=torch.tensor(pos+np.array([0,0,1]),dtype=torch.float32),
            orientation=torch.tensor(quat,dtype=torch.float32))
        set_at_surface(prop,prop_xy,surface_z(base,prop_xy),[1,0,0,0])
        board.set_position_orientation(position=torch.tensor(pos,dtype=torch.float32),orientation=torch.tensor(quat,dtype=torch.float32))
        for obj in (board,prop):obj.keep_still();obj.enable_gravity()
        for _ in range(4):_settle(evaluator,control,bindings,'dynamic_propped_board_initial_settle')
        for obj in (board,prop,base):obj.clear_states_cache()
        contacts=dict(board_table=bool(canonical_relation(board,'Touching',base)),
                      board_prop=bool(canonical_relation(board,'Touching',prop)),
                      prop_table=bool(canonical_relation(prop,'Touching',base)))
        if not all(contacts.values()):raise RuntimeError('dynamic propped board initial contacts failed: '+str(contacts))
        report['fixtures'].append(dict(type='DYNAMIC_BOARD_ON_INVERTED_BOWL',contacts=contacts,
            board_pose=[v.tolist() for v in object_pose(board)],prop_pose=[v.tolist() for v in object_pose(prop)],
            fixed_constraints_added=False))
    if kind in ('P3-1','P3-3'):
        fixture=template['fixture_spec'];parameters=fixture['parameters']
        for i,role in enumerate(fixture['roles']):
            obj=role_object(role)
            if not obj.fixed_base:raise RuntimeError('inclined object was not fixed before Environment creation')
            box=read_base_aligned_box(obj)
            xy=center+np.array([0,(i-.5)*(float(box.extent_local[1])+0.10)])
            z=surface_z(staging,xy)
            if kind=='P3-1':
                _,receiver_hi=object_aabb(role_object('receiver'))
                z=max(z,float(receiver_hi[2])+0.02)
            pose=incline_pose(extent_local=box.extent_local,center_local=box.center_local,
                anchor_xy=xy,lowest_surface_z=z,pitch_deg=parameters['pitch_deg'],yaw_deg=fixture['yaw_deg'][i],clearance_m=0)
            obj.set_position_orientation(position=torch.tensor(pose['position'],dtype=torch.float32),
                orientation=torch.tensor(pose['orientation_xyzw'],dtype=torch.float32))
            entry={'role':role,'pose':pose,'fixture_mount':'DECLARED_WORLD_FIXED_BASE',
                   'fixed_fixture_not_claimed_natural_support':True}
            if kind=='P3-3':
                key=('low_friction','high_friction')[i]
                entry['material']=apply_scene_material(obj,parameters['materials'][key],
                    experiment_key=task.candidate_id+'_'+role)
            report['fixtures'].append(entry)
        if kind=='P3-3':
            report['subject_material']=apply_scene_material(role_object('subject'),parameters['materials']['subject'],
                experiment_key=task.candidate_id+'_subject')
        else:
            ramp=role_object('ramp_a');rlo,rhi=object_aabb(ramp)
            receiver=role_object('receiver');blo,bhi=object_aabb(receiver)
            # Place receiver underneath the downhill edge of ramp A. B keeps its
            # opposite direction; these are authored fixture geometry, not labels.
            xy=np.array([rhi[0]+(bhi[0]-blo[0])*.20,(rlo[1]+rhi[1])/2])
            set_at_surface(receiver,xy,surface_z(staging,xy))
    elif kind=='P3-2':
        # Collision setup has an actual standing rectangular asset, not a renamed
        # symmetric die. The protected bottle is separated by a broad half-cell.
        target=role_object('target');protected=role_object('protected')
        set_at_surface(target,center,surface_z(staging,center),[0,np.sqrt(.5),0,np.sqrt(.5)])
        plo,phi=object_aabb(protected);tlo,thi=object_aabb(target)
        xy=center+np.array([0,(phi[1]-plo[1]+thi[1]-tlo[1])/2+0.15])
        set_at_surface(protected,xy,surface_z(staging,xy))
        report['fixtures'].append({'target':'target','standing_axis_local':[-1,0,0],'protected_region_xy':xy.tolist()})
    elif kind=='P4-2':
        box=role_object('box');lid=box.links[template['link_roles']['lid']['link']]
        llo,lhi=lid.aabb
        xy=((llo+lhi)/2)[:2].detach().cpu().numpy()
        set_at_surface(role_object('cup'),xy,float(lhi[2]))
        report['fixtures'].append({'cup_support_link':lid.prim_path,'owner':box.name})
    elif kind=='P4-3':
        cabinet=role_object('cabinet');bottle=role_object('bottle')
        state=state_by_name(cabinet,'Open');opened=cabinet.get_joint_positions().clone()
        if not state.set_value(False,fully=True):raise RuntimeError('cannot sample closed cabinet endpoint')
        closed=cabinet.get_joint_positions().clone()
        cabinet.set_joint_positions((opened+closed)/2,drive=False)
        og.sim.render()
        links=[link for name,link in cabinet.links.items() if name!=cabinet.root_link_name and not link.is_meta_link]
        if not links:raise RuntimeError('cabinet has no movable collision link')
        floor=role_object('_floor_anchor')
        chosen=None
        for link in links:
            blo,bhi=link.aabb
            xy=((blo+bhi)/2)[:2].detach().cpu().numpy()
            try:z=surface_z(floor,xy)
            except RuntimeError:continue
            chosen=(xy,z,link.prim_path);break
        cabinet.set_joint_positions(opened,drive=False)
        if chosen is None:raise RuntimeError('cabinet sweep has no floor-supported candidate')
        set_at_surface(bottle,chosen[0],chosen[1])
        report['fixtures'].append({'door_mid_sweep_link':chosen[2],'bottle_xy':chosen[0].tolist(),
                                  'sweep_response_verified':False})
    _settle(evaluator,control,bindings,'p12_fixture_settle')
    if kind=='P3-2':
        from .on_top_support_layers import quaternion_to_matrix_xyzw
        _,q=object_pose(role_object('target'))
        up=float((quaternion_to_matrix_xyzw(q)@np.array([-1.,0.,0.]))[2])
        if up < np.cos(np.deg2rad(contract['upright_limit_deg'])):
            raise RuntimeError('collision target is already toppled in the initial fixture')
        report['target_initial_up_z']=up
    if kind=='P4-2':
        from omnigibson.utils.usd_utils import RigidContactAPI
        lid=role_object('box').links[template['link_roles']['lid']['link']]
        pairs=RigidContactAPI.get_contact_pairs(evaluator.env.scene.idx,[role_object('cup')],[lid],True)
        if not pairs:raise RuntimeError('cup has no fresh contact with the declared lid link')
        report['lid_contact_pairs']=[list(pair) for pair in sorted(pairs)]
    # Fail closed on invalid geometry or visibly fallen starting protected objects.
    for role,axis in contract['up_axes'].items():
        from .on_top_support_layers import quaternion_to_matrix_xyzw
        _,q=object_pose(role_object(role))
        if float((quaternion_to_matrix_xyzw(q)@np.asarray(axis))[2]) < np.cos(np.deg2rad(contract['upright_limit_deg'])):
            raise RuntimeError(f'P12 initial upright requirement failed: {role}')
    return report
