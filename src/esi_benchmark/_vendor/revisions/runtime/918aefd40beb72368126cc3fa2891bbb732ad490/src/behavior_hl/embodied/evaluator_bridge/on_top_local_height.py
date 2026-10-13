"""Candidate-local ray height proposals; collision and final readback still decide."""
from dataclasses import replace
import numpy as np
from .on_top_support_layers import DEFAULT_SUPPORT_CLEARANCE_M
from .spatial_symbolic_actions import object_aabb, DEFAULT_BLOCKER_MARGIN_M
from .spatial_candidate_admission import spatial_object_link_paths


def rebase_on_top_plan(plan, *, target, held, robot, raytest_batch, tensor_module,
                      surface_z_m, height_tolerance_m=0.02):
    if surface_z_m is None or not plan.bindings:
        return plan
    lower, upper = object_aabb(target)
    paths = spatial_object_link_paths(target)
    ignored = tuple(dict.fromkeys((*spatial_object_link_paths(held), *spatial_object_link_paths(robot))))
    starts, ends = [], []
    for b in plan.bindings:
        lo, hi = b.future_aabb_min_world, b.future_aabb_max_world
        # Include the existing collision query's XY margin, not only the centre.
        xy0, xy1 = lo[:2]-DEFAULT_BLOCKER_MARGIN_M, hi[:2]+DEFAULT_BLOCKER_MARGIN_M
        for xy in ((xy0+xy1)/2, xy0, (xy0[0],xy1[1]), xy1, (xy1[0],xy0[1])):
            starts.append((*xy,float(upper[2])+0.05))
            ends.append((*xy,float(lower[2])-0.05))
    hits = raytest_batch(tensor_module.as_tensor(starts,dtype=tensor_module.float32),
        tensor_module.as_tensor(ends,dtype=tensor_module.float32),only_closest=True,
        ignore_bodies=ignored,ignore_collisions=ignored)
    if len(hits)!=len(starts):raise ValueError('incomplete OnTop local surface query')
    bindings, evidence = [], []
    for i,b in enumerate(plan.bindings):
        heights=[]
        for hit in hits[i*5:i*5+5]:
            hit = hit or {}
            body=str(hit.get('rigidBody',''));pos=np.asarray(hit.get('position',[]),dtype=float)
            normal=np.asarray(hit.get('normal',[]),dtype=float)
            if (not hit.get('hit') or not any(body==p or body.startswith(p+'/') for p in paths)
                    or pos.shape!=(3,) or normal.shape!=(3,) or not np.isfinite(pos).all()
                    or not np.isfinite(normal).all() or np.linalg.norm(normal)==0
                    or normal[2]/np.linalg.norm(normal)<np.cos(np.deg2rad(20))
                    or abs(pos[2]-surface_z_m)>height_tolerance_m):
                break
            heights.append(float(pos[2]))
        delta=0.0
        if len(heights)==5:
            delta=max(heights)+DEFAULT_SUPPORT_CLEARANCE_M-float(b.future_aabb_min_world[2])
            shift=np.array([0.,0.,delta])
            b=replace(b,candidate=replace(b.candidate,position=b.candidate.position+shift),
                future_aabb_min_world=b.future_aabb_min_world+shift,
                future_aabb_max_world=b.future_aabb_max_world+shift)
        bindings.append(b)
        evidence.append(dict(candidate=b.candidate.name,complete_target_footprint=len(heights)==5,
            hit_heights_m=heights,delta_z_m=delta))
    metadata=dict(plan.candidate_plan.planning_metadata_private or {})
    metadata['local_footprint_height_private']=evidence
    return replace(plan,bindings=tuple(bindings),candidate_plan=replace(plan.candidate_plan,
        candidates=tuple(b.candidate for b in bindings),planning_metadata_private=metadata))
