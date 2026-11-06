"""Six coarse orientations ranked by container bounds; never a fit rejection."""
import itertools
import numpy as np
from scipy.spatial.transform import Rotation
from .on_top_support_layers import read_base_aligned_box
from .spatial_symbolic_actions import object_aabb

POLICY='physical-inside-six-coarse-bounds/v1'


def coarse_rotations():
    """One nearest right-angle rotation per XYZ dimension permutation."""
    result=[]
    for permutation in itertools.permutations(range(3)):
        basis=np.eye(3)[list(permutation)]
        choices=[np.diag(signs)@basis for signs in itertools.product((1.,-1.),repeat=3)
                 if np.linalg.det(np.diag(signs)@basis)>0]
        result.append(max(choices,key=lambda r:float(np.trace(r))))
    return result


def choose_orientation(base,extent,container_extent):
    base=np.asarray(base,dtype=float);extent=np.asarray(extent,dtype=float)
    bounds=np.asarray(container_extent,dtype=float)
    if (base.shape!=(4,) or extent.shape!=(3,) or bounds.shape!=(3,)
        or not all(np.isfinite(x).all() for x in (base,extent,bounds))
        or np.any(extent<=0) or np.any(bounds<=0) or np.linalg.norm(base)==0):
        raise ValueError('invalid coarse orientation geometry')
    origin=Rotation.from_quat(base);candidates=[]
    for index,matrix in enumerate(coarse_rotations()):
        rotation=Rotation.from_matrix(matrix)*origin
        projected=np.abs(rotation.as_matrix())@extent
        ratios=projected/bounds
        entry=float(max(ratios[:2]));depth=float(ratios[2])
        # Prefer coarse fit if available, then smaller entry section. Ratios
        # never reject a candidate: even every-too-large chooses one release.
        rank=(max(0.,entry-1),max(0.,depth-1),entry,
              float(Rotation.from_matrix(matrix).magnitude()),index)
        candidates.append(dict(index=index,orientation_xyzw=rotation.as_quat().tolist(),
            projected_extent_m=projected.tolist(),rank=list(rank)))
    selected=min(candidates,key=lambda r:tuple(r['rank']))
    return np.asarray(selected['orientation_xyzw']),dict(policy=POLICY,candidates=candidates,
        selected_index=selected['index'],container_extent_m=bounds.tolist(),
        fit_is_rejection_gate=False,orientation_search_mutates_world=False)


def select_inside_orientation(held,target,base):
    """Use a native fillable-link envelope if available; otherwise outer box."""
    envelope=[]
    for link in getattr(target,'links',{}).values():
        if getattr(link,'is_meta_link',False) and getattr(link,'meta_link_type',None) in ('fillable','openfillable'):
            try:
                lo,hi=link.visual_aabb
                def array(x):return np.asarray(x.detach().cpu() if hasattr(x,'detach') else x,dtype=float)
                lo,hi=array(lo),array(hi)
                if lo.shape==(3,) and hi.shape==(3,) and np.isfinite([lo,hi]).all() and np.all(hi>lo):
                    envelope.append((lo,hi))
            except (AttributeError,TypeError,ValueError):
                continue
    if envelope:
        lower=np.min([p[0] for p in envelope],axis=0);upper=np.max([p[1] for p in envelope],axis=0)
        source='NATIVE_FILLABLE_AABB_ENVELOPE_APPROXIMATION'
    else:
        lower,upper=object_aabb(target);source='CONTAINER_OUTER_AABB_APPROXIMATION'
    orientation,report=choose_orientation(base,read_base_aligned_box(held).extent_local,upper-lower)
    report.update(geometry_source=source,actual_opening_verified=False,full_containment_required=False)
    return orientation,report
