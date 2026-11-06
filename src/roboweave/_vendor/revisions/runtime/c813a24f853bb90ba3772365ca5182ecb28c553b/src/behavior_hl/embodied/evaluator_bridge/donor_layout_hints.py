"""Construction-only placement hints inside a prior measured occupied region."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .spatial_symbolic_actions import SpatialPoseCandidate, object_pose
from .on_top_support_layers import read_base_aligned_box, quaternion_to_matrix_xyzw


def near_edge_xy(lower,upper,observer,extent,along_offset,edge_margin):
    """Center inside the observer-facing edge, not at the island center."""
    lower,upper,observer,extent=map(lambda x:np.asarray(x,dtype=float),(lower,upper,observer,extent))
    center=(lower+upper)/2;delta=(observer-center)/(upper-lower)
    normal=int(np.argmax(np.abs(delta)));tangent=1-normal
    inward=extent[normal]/2+edge_margin
    xy=center.copy();xy[normal]=lower[normal]+inward if delta[normal]<0 else upper[normal]-inward
    xy[tangent]+=along_offset
    if np.any(xy-extent/2<lower) or np.any(xy+extent/2>upper):
        raise ValueError('near-edge initial footprint exceeds support bounds')
    return xy


def near_edge_candidate(subject,support,role,contract):
    spec=contract['near_edge_layout'];row=spec['placements'][role]
    if support.name!=spec['support_name']:raise ValueError('near-edge support identity mismatch')
    from .spatial_symbolic_actions import object_aabb
    from .physical_ontop_drop import selected_surface_height,release_bounds
    lo,hi=object_aabb(support);q=np.asarray(row['orientation_xyzw'],dtype=float)
    box=read_base_aligned_box(subject);R=quaternion_to_matrix_xyzw(q);extent=np.abs(R)@box.extent_local
    xy=near_edge_xy(lo[:2],hi[:2],spec['observer_xy'],extent[:2],row['along_offset_m'],row['edge_margin_m'])
    z,hit=selected_surface_height(support,xy)
    if not hit['target_surface_hit_count']:raise ValueError('near-edge point has no actual target surface')
    center=np.r_[xy,z+.003+extent[2]/2]
    return SpatialPoseCandidate('near_robot_edge_'+role,center-R@box.center_local,q)


def donor_candidate(subject,support,role,contract):
    if contract.get('near_edge_layout'):
        return near_edge_candidate(subject,support,role,contract)
    hint=contract.get('donor_layout_hints')
    if hint is None:return None
    path=Path(hint['source']['path']);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=hint['source']['sha256']:
        raise ValueError('donor layout source hash mismatch')
    capture=json.loads(raw)['initial_state_capture']
    geometry=capture['scope_geometry'];reference=geometry[hint['support_scope']]
    if reference['actual_name']!=support.name:raise ValueError('donor support identity mismatch')
    row=hint['placements'][role]
    if row['asset_id']!=f'{subject.category}/{subject.model}':raise ValueError('donor hint subject mismatch')
    region=geometry[hint['region_scope']]
    # The old region is placement evidence, not a mesh-free-space certificate.
    xy=np.asarray(row['center_xy_world'],dtype=float)
    old_p=np.asarray(reference['position']);old_R=quaternion_to_matrix_xyzw(reference['orientation'])
    p,q=object_pose(support);R=quaternion_to_matrix_xyzw(q)
    point=np.r_[xy,region['aabb_min'][2]]
    translated=np.asarray(p)+R@old_R.T@(point-old_p)
    orientation=np.asarray(row['orientation_xyzw'],dtype=float)
    box=read_base_aligned_box(subject);rotation=quaternion_to_matrix_xyzw(orientation)
    half=(np.abs(rotation)@box.extent_local)/2
    center=np.array([*translated[:2],translated[2]+.003+half[2]])
    if not np.isfinite(center).all():raise ValueError('nonfinite donor candidate')
    return SpatialPoseCandidate('donor_region_'+role,center-rotation@box.center_local,orientation)
