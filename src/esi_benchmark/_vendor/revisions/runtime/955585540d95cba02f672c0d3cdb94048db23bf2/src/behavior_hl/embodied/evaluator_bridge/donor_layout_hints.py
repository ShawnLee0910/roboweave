"""Construction-only placement hints inside a prior measured occupied region."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .spatial_symbolic_actions import SpatialPoseCandidate, object_pose
from .on_top_support_layers import read_base_aligned_box, quaternion_to_matrix_xyzw


def donor_candidate(subject,support,role,contract):
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
