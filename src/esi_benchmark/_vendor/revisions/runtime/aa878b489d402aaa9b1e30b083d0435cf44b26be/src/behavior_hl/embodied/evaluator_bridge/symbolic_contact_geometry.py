"""Narrow a coarse target hit against authored surfaces, not its convex hull.

This certifies the same inset query box used by placement admission. It does
not add stability/weight rules, or treat a whole support as permitted contact.
"""

from __future__ import annotations

import numpy as np


def mesh_intersects_box(mesh, center, half_extent):
    """Triangle/box SAT plus closed-solid containment; no physics or sampling."""
    triangles = np.asarray(mesh.triangles, dtype=np.float64) - center
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles):
        raise ValueError("target collision mesh has no triangles")
    if not np.isfinite(triangles).all():
        raise ValueError("target collision mesh contains non-finite geometry")
    # Broad phase before the 13 separating axes (box, face, edge cross axes).
    nearby = triangles[
        np.all(triangles.min(axis=1) <= half_extent, axis=1)
        & np.all(triangles.max(axis=1) >= -half_extent, axis=1)
    ]
    if len(nearby):
        edges = np.roll(nearby, -1, axis=1) - nearby
        axes = np.concatenate((
            np.broadcast_to(np.eye(3), (len(nearby), 3, 3)),
            np.cross(edges[:, 0], edges[:, 1])[:, None, :],
            np.cross(edges[:, :, None, :], np.eye(3)[None, None, :, :]).reshape(-1, 9, 3),
        ), axis=1)
        projections = np.einsum("ntd,nad->nat", nearby, axes)
        radius = np.abs(axes) @ half_extent
        separated = (projections.min(axis=2) > radius) | (projections.max(axis=2) < -radius)
        if np.any(~np.any(separated, axis=1)):
            return True
    if mesh.is_watertight:
        if not mesh.is_winding_consistent:
            raise ValueError("closed target mesh has inconsistent winding")
        # A box entirely within solid material crosses no surface. Generalized
        # winding detects that case without optional rtree / collision libraries.
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
        numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
        denominator = (la * lb * lc + np.einsum("ij,ij->i", a, b) * lc
                       + np.einsum("ij,ij->i", b, c) * la
                       + np.einsum("ij,ij->i", c, a) * lb)
        return bool(abs(np.sum(2 * np.arctan2(numerator, denominator))) > 2 * np.pi)
    return False


def target_solid_intersects_query_box(target, query_box):
    """Read live USD transforms; never cache moving articulated world geometry."""
    from omnigibson.utils.usd_utils import mesh_prim_to_trimesh_mesh

    if not np.allclose(query_box.rotation_xyzw, (0, 0, 0, 1), rtol=0, atol=1e-12):
        raise ValueError("target surface admission requires an axis-aligned query box")
    mesh_count = 0
    for link in target.links.values():
        for collision_mesh in link.collision_meshes.values():
            mesh_count += 1
            mesh = mesh_prim_to_trimesh_mesh(
                collision_mesh.prim, include_normals=False,
                include_texcoord=False, world_frame=True,
            )
            if mesh_intersects_box(mesh, query_box.center_xyz, query_box.half_extent_xyz):
                return True
    if not mesh_count:
        raise ValueError("target collision surface geometry is unavailable")
    return False
