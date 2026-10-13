"""Deterministic yaw-only placement selection; no simulator queries or retries."""
import numpy as np
from scipy.spatial.transform import Rotation

from .on_top_support_layers import read_base_aligned_box
from .spatial_symbolic_actions import object_aabb

POLICY = 'configured-base-two-horizontal-axes-aabb-coverage/v1'
# Asset-specific declarations, not category-wide assumptions about mesh axes.
# These specify intended poses, not certification of liquid retention.
BASE_POSES = {asset: (0., 0., 0., 1.) for asset in (
    'bowl/eipwho', 'bowl/belcml', 'mixing_bowl/bsgybx',
    'orange/cwcefi', 'can_of_corn/alphlq', 'water_bottle/ackxiy',
    'notebook/jzwprb', 'electric_kettle/hkdsla', 'plate/ivuveo',
    'hammer/bbzlel', 'mug/ppzttc', 'wineglass/kxovsj',
    'water_glass/gypzlg', 'cell_phone/dbhfuh', 'apple/hwrflj',
    'soap_dispenser/ijxawq', 'beaker/dtjmai', 'goggles/wszdxi',
    'graduated_cylinder/egpkea', 'calculator/kwmmty',
    'canned_food/acgdtc', 'stapler/wheosp',
)}


def base_orientation(held):
    asset = f"{getattr(held, 'category', '')}/{getattr(held, 'model', '')}"
    value = getattr(held, '_physical_placement_base_orientation_xyzw', None)
    source = 'object_asset_configuration'
    if value is None:
        value = BASE_POSES.get(asset)
        source = 'explicit_asset_registry'
    if value is None:
        raise ValueError(f'Physical OnTop needs an explicit base orientation for {asset}')
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1.):
        raise ValueError('invalid configured base orientation')
    return q.copy(), dict(asset_id=asset, source=source, orientation_xyzw=q.tolist())


def choose_yaw(base, extent, lower, upper, center):
    """Compare two support-AABB axes at exactly the same XY, using an AABB proxy."""
    base = np.asarray(base, dtype=float)
    extent, lower, upper, center = [np.asarray(x, dtype=float) for x in (extent, lower, upper, center)]
    if (base.shape != (4,) or extent.shape != (3,) or lower.shape != (2,)
            or upper.shape != (2,) or center.shape != (2,)
            or not all(np.isfinite(x).all() for x in (base, extent, lower, upper, center))
            or not np.isclose(np.linalg.norm(base), 1.) or np.any(extent <= 0) or np.any(upper <= lower)):
        raise ValueError('invalid orientation geometry')
    rotation = Rotation.from_quat(base)
    # Project each configured box axis; choose the longest horizontal one.
    vectors = rotation.as_matrix()[:2, :] * extent
    axis = int(np.argmax(np.linalg.norm(vectors, axis=0)))
    heading = np.arctan2(vectors[1, axis], vectors[0, axis])
    support_long = int(np.argmax(upper - lower))  # square ties resolve to X
    candidates = []
    for index, support_axis in enumerate((support_long, 1-support_long)):
        # Axis alignment has 180-degree symmetry: take the nearest yaw.
        delta = ((support_axis * np.pi/2 - heading + np.pi/2) % np.pi) - np.pi/2
        result = Rotation.from_rotvec([0., 0., delta]) * rotation
        half = (np.abs(result.as_matrix()) @ extent)[:2] / 2
        overlap = np.maximum(0., np.minimum(center+half, upper) - np.maximum(center-half, lower))
        coverage = float(np.prod(overlap) / np.prod(2*half))
        candidates.append(dict(index=index, yaw_delta_rad=float(delta), coverage_ratio=coverage,
                               orientation_xyzw=result.as_quat().tolist()))
    best = max(x['coverage_ratio'] for x in candidates)
    tied = [x for x in candidates if abs(x['coverage_ratio']-best) <= 1e-12]
    smallest = min(abs(x['yaw_delta_rad']) for x in tied)
    selected = next(x for x in tied if abs(abs(x['yaw_delta_rad'])-smallest) <= 1e-12)
    return np.asarray(selected['orientation_xyzw']), dict(policy=POLICY,
        geometry_source='PROJECTED_BASE_BOX_AND_SUPPORT_WORLD_AABB_APPROXIMATION',
        actual_contact_area_verified=False, global_stability_optimum_claimed=False,
        common_center_xy=center.tolist(), candidates=candidates, selected_index=selected['index'])


def select_orientation(held, target, base, center_xy):
    lo, hi = object_aabb(target)
    center = (lo[:2]+hi[:2])/2 if center_xy is None else center_xy
    return choose_yaw(base, read_base_aligned_box(held).extent_local, lo[:2], hi[:2], center)
