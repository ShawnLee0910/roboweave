"""CPU fixture geometry and scene-local native physics material binding for P12.

Only called during authorized initial construction. No live asset file is edited.
"""
import math
import numpy as np

from .on_top_support_layers import quaternion_to_matrix_xyzw


def propped_board_pose(extent,offset,anchor_xy,table_z,prop_height):
    """Initial proposal only: same geometry as the successful robot-free fixture."""
    from scipy.spatial.transform import Rotation
    extent=np.asarray(extent);axis=int(np.argmax(extent[:2]))
    pitch=math.atan2(prop_height,.8*float(extent[axis]))
    angles=np.zeros(3);angles[1 if axis==0 else 0]=-pitch if axis==0 else pitch
    rotation=Rotation.from_rotvec(angles)
    half_z=float((np.abs(rotation.as_matrix())@extent)[2])/2
    center=np.array([*anchor_xy,table_z+.003+half_z])
    prop_xy=np.asarray(anchor_xy,dtype=float).copy();prop_xy[axis]+=.3*float(extent[axis])*math.cos(pitch)
    return center-rotation.apply(offset),rotation.as_quat(),prop_xy


def incline_pose(*, extent_local, center_local, anchor_xy, lowest_surface_z,
                 pitch_deg, yaw_deg=0.0, clearance_m=0.003):
    extent = np.asarray(extent_local,dtype=float)
    offset = np.asarray(center_local,dtype=float)
    xy = np.asarray(anchor_xy,dtype=float)
    if (extent.shape != (3,) or offset.shape != (3,) or xy.shape != (2,) or
        not np.isfinite(extent).all() or not np.isfinite(offset).all() or not np.isfinite(xy).all()
        or (extent <= 0).any() or not all(math.isfinite(v) for v in
            (lowest_surface_z,pitch_deg,yaw_deg,clearance_m)) or clearance_m < 0 or not 0 < abs(pitch_deg) < 45):
        raise ValueError('invalid slope geometry')
    pitch,yaw = map(math.radians,(pitch_deg,yaw_deg))
    q = np.array([-math.sin(yaw/2)*math.sin(pitch/2), math.cos(yaw/2)*math.sin(pitch/2),
                  math.sin(yaw/2)*math.cos(pitch/2), math.cos(yaw/2)*math.cos(pitch/2)])
    rotation = quaternion_to_matrix_xyzw(q)
    world_extent = np.abs(rotation) @ extent
    center = np.r_[xy,lowest_surface_z + clearance_m + world_extent[2]/2]
    return {'position':(center-rotation@offset).tolist(),'orientation_xyzw':q.tolist(),
            'world_aabb_min':(center-world_extent/2).tolist(),
            'world_aabb_max':(center+world_extent/2).tolist(),
            'construction_fixed_base_required':True, 'pitch_deg':pitch_deg,'yaw_deg':yaw_deg,
            'source':'LIVE_BASE_ALIGNED_COLLISION_GEOMETRY_AND_VERIFIED_ANCHOR_SURFACE'}


def validate_material(rule):
    if set(rule) != {'label_zh','static_friction','dynamic_friction','restitution','combine_mode'}:
        raise ValueError('material requires explicit physics and public label')
    values = [rule[k] for k in ('static_friction','dynamic_friction','restitution')]
    if (not isinstance(rule['label_zh'],str) or not rule['label_zh'] or
        any(type(x) not in (int,float) or not math.isfinite(x) or x < 0 for x in values) or
        rule['dynamic_friction'] > rule['static_friction'] or rule['restitution'] > 1 or
        rule['combine_mode'] != 'min'):
        raise ValueError('unsupported physical material')
    return rule


def apply_scene_material(obj, rule, *, experiment_key):
    """Bind fresh materials to actual collision meshes, then read USD bindings back.

    The caller owns a disposable scene / reset contract. This does not promise
    native load_state restores material properties between unrelated cohorts.
    """
    validate_material(rule)
    if not experiment_key or not all(c.isascii() and (c.isalnum() or c=='_') for c in experiment_key):
        raise ValueError('invalid experiment key')
    import omnigibson as og
    import omnigibson.lazy as lazy
    meshes = [mesh for link in obj.links.values() for mesh in link.collision_meshes.values()]
    if not meshes:
        raise ValueError('object has no collision meshes')
    reports = []
    with og.sim.editing_usd():
        with lazy.pxr.Usd.EditContext(og.sim.stage,og.sim.stage.GetSessionLayer()):
            path = f'/World/P12PhysicsMaterials/{experiment_key}'
            if og.sim.stage.GetPrimAtPath(path).IsValid():
                raise ValueError('material identity already exists; do not overwrite')
            material = lazy.isaacsim.core.api.materials.PhysicsMaterial(prim_path=path,
                static_friction=rule['static_friction'],dynamic_friction=rule['dynamic_friction'],
                restitution=rule['restitution'])
            physx = lazy.pxr.PhysxSchema.PhysxMaterialAPI.Apply(material.prim)
            physx.CreateFrictionCombineModeAttr().Set(rule['combine_mode'])
            physx.CreateRestitutionCombineModeAttr().Set('min')
    # GeomPrim owns its editing_usd context; never nest it.
    with lazy.pxr.Usd.EditContext(og.sim.stage,og.sim.stage.GetSessionLayer()):
        for mesh in meshes:
            mesh.apply_physics_material(material,weaker_than_descendants=False)
            bound,_ = mesh._binding_api.ComputeBoundMaterial(materialPurpose='physics')
            actual_path = str(bound.GetPrim().GetPath())
            api = lazy.pxr.UsdPhysics.MaterialAPI(bound.GetPrim())
            actual = {'static_friction':api.GetStaticFrictionAttr().Get(),
                      'dynamic_friction':api.GetDynamicFrictionAttr().Get(),
                      'restitution':api.GetRestitutionAttr().Get(),
                      'combine_mode':lazy.pxr.PhysxSchema.PhysxMaterialAPI(bound.GetPrim()).GetFrictionCombineModeAttr().Get()}
            if actual_path != path or actual['combine_mode'] != rule['combine_mode'] or any(
                not math.isclose(actual[k],rule[k],rel_tol=1e-6,abs_tol=1e-8)
                for k in ('static_friction','dynamic_friction','restitution')):
                raise RuntimeError('native material binding/readback mismatch; invalidate construction')
            reports.append({'collision_mesh':mesh.prim_path,'material_path':actual_path,**actual})
    return {'status':'BOUND_AND_USD_READ_BACK','public_label':rule['label_zh'],
            'mesh_count':len(meshes),'meshes':reports,'physical_response_live_verified':False,
            'source_asset_files_modified':False,'scope':'SCENE_SESSION_LAYER'}
