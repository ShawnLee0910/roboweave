from __future__ import annotations

import hashlib

import numpy as np

from behavior_hl.embodied.evaluator_bridge.on_top_support_layers import (
    plan_support_plane_lateral_bindings,
)
from behavior_hl.embodied.evaluator_bridge.private_instance_visibility import (
    PrivateInstanceSegmentationFrame,
)
from behavior_hl.embodied.evaluator_bridge.private_visual_markers import (
    NextToSupportPosition,
    PrivateMarkerBinding,
    SpatialMarkerTargetToken,
    build_private_next_to_position_snapshot,
    render_marker_overlay,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_b1_component_builders import (
    _spatial_marker_region,
)
class _BoxObject:
    def __init__(
        self,
        name: str,
        lower,
        upper,
        *,
        orientation=(0.0, 0.0, 0.0, 1.0),
    ) -> None:
        self.name = name
        self.aabb = (
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
        )
        self._position = (self.aabb[0] + self.aabb[1]) / 2.0
        self._orientation = np.asarray(orientation, dtype=np.float64)
        self.link_prim_paths = (f"/World/{name}",)

    def get_position_orientation(self):
        return self._position, self._orientation

    def get_base_aligned_bbox(self):
        return (
            self._position,
            self._orientation,
            self.aabb[1] - self.aabb[0],
            np.zeros(3, dtype=np.float64),
        )


def test_next_to_uses_pre_pick_orientation_at_eight_fixed_world_points() -> None:
    vertical = (
        0.0,
        np.sin(np.pi / 4.0),
        0.0,
        np.cos(np.pi / 4.0),
    )
    held = _BoxObject(
        "held_tile",
        (-0.30, -0.15, -0.01),
        (0.30, 0.15, 0.01),
        orientation=vertical,
    )
    target = _BoxObject(
        "target_tile",
        (-0.30, -0.15, 0.0),
        (0.30, 0.15, 0.02),
    )

    bindings = plan_support_plane_lateral_bindings(
        held,
        target,
        observer_xy=(0.0, -2.0),
        preferred_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    primary = tuple(item for item in bindings if item.orientation_rank == 0)

    assert [item.selector for item in primary] == [
        "screen_left",
        "screen_far_left",
        "screen_far",
        "screen_far_right",
        "screen_right",
        "screen_near_right",
        "screen_near",
        "screen_near_left",
    ]
    assert all(
        np.allclose(
            item.candidate.orientation_xyzw,
            (0.0, 0.0, 0.0, 1.0),
        )
        for item in primary
    )
    for selector in (item.selector for item in primary):
        centers = [
            item.center_xy
            for item in bindings
            if item.selector == selector
        ]
        assert all(np.allclose(center, centers[0]) for center in centers[1:])


def test_next_to_markers_keep_exact_world_bindings() -> None:
    held = _BoxObject(
        "held_tile",
        (-0.20, -0.10, 0.80),
        (0.20, 0.10, 0.82),
        orientation=(
            0.0,
            np.sin(np.pi / 4.0),
            0.0,
            np.cos(np.pi / 4.0),
        ),
    )
    target = _BoxObject(
        "target_tile",
        (-0.30, -0.15, 0.0),
        (0.30, 0.15, 0.02),
    )
    bindings = plan_support_plane_lateral_bindings(
        held,
        target,
        observer_xy=(0.0, -2.0),
        preferred_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    primary = tuple(item for item in bindings if item.orientation_rank == 0)
    positions = tuple(
        NextToSupportPosition(
            selector=binding.selector,
            anchor_uv=(20 + index % 4 * 40, 40 + index // 4 * 50),
            region_center_xy=tuple(binding.center_xy),
            region_extent_xy=tuple(binding.extent_xy),
        )
        for index, binding in enumerate(primary)
    )

    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    segmentation = np.zeros((120, 160), dtype=np.int64)
    segmentation[45:75, 65:95] = 7
    frame = PrivateInstanceSegmentationFrame(
        camera="head",
        simulator_step=3,
        source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        segmentation=segmentation,
        instance_id_to_name={7: target.name},
    )
    reference = PrivateMarkerBinding(
        marker="C1",
        object_name=target.name,
        instance_id=7,
        visible_pixels=900,
        official_bbox_xyxy=(65, 45, 94, 74),
        target_token=target,
    )
    snapshot = build_private_next_to_position_snapshot(
        frame=frame,
        public_head_rgb=rgb,
        reference_binding=reference,
        support_positions=positions,
        created_action_index=4,
    )

    assert len(snapshot.bindings) == 8
    for binding, position in zip(
        snapshot.bindings.values(),
        positions,
        strict=True,
    ):
        token = binding.target_token
        assert isinstance(token, SpatialMarkerTargetToken)
        assert token.selector == position.selector
        assert token.region_center_xy == position.region_center_xy
        assert token.region_extent_xy == position.region_extent_xy
        assert _spatial_marker_region(token, "place_next_to") == (
            position.region_center_xy,
            position.region_extent_xy,
            None,
        )


def test_next_to_position_labels_expand_in_their_selected_direction() -> None:
    anchor = (70, 70)
    color = np.asarray((0, 205, 225), dtype=np.uint8)
    endpoints = {
        "screen_left": (44, 70),
        "screen_far_left": (44, 44),
        "screen_far": (70, 44),
        "screen_far_right": (96, 44),
        "screen_right": (96, 70),
        "screen_near_right": (96, 96),
        "screen_near": (70, 96),
        "screen_near_left": (44, 96),
    }
    for selector, (x, y) in endpoints.items():
        binding = PrivateMarkerBinding(
            marker="C1",
            object_name="reference",
            instance_id=1,
            visible_pixels=10,
            official_bbox_xyxy=(65, 65, 75, 75),
            target_token=object(),
            binding_kind="position",
            selector=selector,
            anchor_uv=anchor,
        )
        rendered = render_marker_overlay(
            np.full((140, 140, 3), 245, dtype=np.uint8),
            {"C1": binding},
        )
        patch = rendered[y - 2 : y + 3, x - 2 : x + 3]
        assert np.any(np.all(patch == color, axis=2)), selector


def test_oblique_next_to_centers_use_actual_box_gap():
    from behavior_hl.embodied.evaluator_bridge.spatial_symbolic_actions import next_to_gap_for_extents,symbolic_next_to_relation
    for extents in [((.245,.231,.061),(.288,.386,.05)),((.129,.089,.0114),(.151,.118,.025))]:
        h,t=(np.asarray(v) for v in extents)
        held=_BoxObject('held',-h/2,h/2);target=_BoxObject('target',-t/2,t/2)
        bindings=plan_support_plane_lateral_bindings(held,target,observer_xy=(-.99,.48),preferred_orientation_xyzw=(0,0,0,1))
        gap=next_to_gap_for_extents(h,t)
        for b in bindings:
            if b.orientation_rank:continue
            lower,upper=b.future_aabb_min_world.copy(),b.future_aabb_max_world.copy()
            # Common support height, same as the live support-rebase contract.
            dz=target.aabb[0][2]+.003-lower[2];lower[2]+=dz;upper[2]+=dz
            candidate=_BoxObject('candidate',lower,upper)
            distance=np.linalg.norm(np.maximum(np.maximum(lower,target.aabb[0])-np.minimum(upper,target.aabb[1]),0))
            assert np.isclose(distance,gap,atol=1e-10)
            assert symbolic_next_to_relation(candidate,target)
