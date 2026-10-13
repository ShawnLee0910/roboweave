from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import pytest

from behavior_hl.embodied.evaluator_bridge.action23_outcome_verifier import (
    Action23OutcomeVerifier,
)
from behavior_hl.embodied.evaluator_bridge.inside_task_profiles import (
    get_inside_task_profile,
)
from behavior_hl.embodied.evaluator_bridge.marker_symbolic_grasp_backend import (
    MarkerInputUnavailableError,
    MarkerSymbolicGraspBackendAdapter,
)
from behavior_hl.embodied.evaluator_bridge.private_instance_visibility import (
    PrivateInstanceSegmentationFrame,
)
from behavior_hl.embodied.evaluator_bridge.private_observation_ingress import (
    HEAD_RGB_KEY,
    HEAD_INSTANCE_SEGMENTATION_KEY,
)
from behavior_hl.embodied.evaluator_bridge.private_visual_markers import (
    MarkerPublicFailureReason,
    MarkerStatus,
    OnTopSurfaceRegion,
    PickBundleTargetToken,
    PrivateMarkerBinding,
    PrivateMarkerSnapshot,
    PrivateVisualMarkerStore,
    SpatialMarkerTargetToken,
    _allocate_badge,
    _binding_target_mask,
    _draw_high_contrast_leader,
    build_private_container_compartment_snapshot,
    build_private_marker_snapshot,
    build_private_on_top_support_snapshot,
    build_private_on_top_surface_snapshot,
    build_private_pick_snapshot,
    current_target_visibility,
    render_marker_overlay,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_b1_component_builders import (
    R1ProB1ComponentBuilderError,
    R1ProMarkerSymbolicComponentBuilder,
    _current_visible_scene_objects,
    _generic_on_top_support_objects,
    _live_accessible_pickable_objects,
    _live_movable_objects,
    _live_pick_bundle_targets,
    _live_pickable_objects,
    _nearest_target_xy,
    _under_marker_target_objects,
    _read_marker_symbolic_canonical_state,
    _resolve_marker_geometry_target,
)
from behavior_hl.embodied.evaluator_bridge.simple_spatial_markers import (
    generate_shelf_compartment_markers,
)
from behavior_hl.embodied.runtime.plugins.marker_symbolic_grasp import (
    MARKER_SYMBOLIC_BACKEND_SERVICE,
    SymbolicGraspStatus,
)
from behavior_hl.embodied.policy.observations import CAMERA_SENSOR_NAMES
from behavior_hl.embodied.policy.protocol import ActionEnvelope, MarkerTargetOrRequestArgs


class _Object:
    def __init__(self, name, category, lower, upper):
        self.name = name
        self.category = category
        self.aabb = (np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))
        self._position = (self.aabb[0] + self.aabb[1]) / 2.0

    def get_position_orientation(self):
        return self._position, np.asarray((0.0, 0.0, 0.0, 1.0))


def test_action_outcome_reader_uses_symbolic_on_top_geometry(monkeypatch):
    held = _Object(
        "held",
        "item",
        (-0.05, -0.05, 0.503),
        (0.05, 0.05, 0.603),
    )
    support = _Object(
        "support",
        "table",
        (-0.5, -0.5, 0.0),
        (0.5, 0.5, 0.5),
    )
    held.link_prim_paths = ("/World/held",)
    support.link_prim_paths = ("/World/support",)

    def raytest_batch(starts, ends, **_kwargs):
        hits = []
        for start, end in zip(starts, ends, strict=True):
            height = next((z for z in (0.5, 0.3) if end[2] <= z <= start[2]), None)
            hits.append(
                {"hit": True, "rigidBody": "/World/support",
                 "position": (start[0], start[1], height), "normal": (0, 0, 1)}
                if height is not None else {"hit": False}
            )
        return hits

    monkeypatch.setitem(sys.modules, "omnigibson.utils.sampling_utils", SimpleNamespace(raytest_batch=raytest_batch))

    assert _read_marker_symbolic_canonical_state(
        held,
        "OnTop",
        SpatialMarkerTargetToken(
            support, "place_on_top", "top",
            region_center_xy=(0, 0), region_extent_xy=(1, 1), surface_z_m=0.5,
        ),
        symbolic_on_top=True,
    )
    internal = _Object(
        "internal",
        "item",
        (-0.05, -0.05, 0.303),
        (0.05, 0.05, 0.403),
    )
    internal.link_prim_paths = ("/World/internal",)
    surface = SpatialMarkerTargetToken(
        support,
        "place_on_top",
        "seat",
        region_center_xy=(0.0, 0.0),
        region_extent_xy=(0.5, 0.5),
        surface_z_m=0.3,
    )
    assert _read_marker_symbolic_canonical_state(
        internal,
        "OnTop",
        surface,
        symbolic_on_top=True,
    )


def _fillable(target):
    target.links = {
        "fillable": SimpleNamespace(is_meta_link=True, meta_link_type="fillable")
    }
    return target


def _frame(rgb, segmentation, names):
    return PrivateInstanceSegmentationFrame(
        camera="head",
        simulator_step=3,
        source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        segmentation=segmentation,
        instance_id_to_name=names,
    )


def test_pick_candidates_keep_physical_role_admission():
    profile = get_inside_task_profile("cleaning_up_plates_and_food")
    assert "plate.n.04" in profile.container_synsets
    assert "plate.n.04" not in profile.graspable_synsets

    def pose():
        return np.zeros(3), np.asarray((0.0, 0.0, 0.0, 1.0))

    def move(**_kwargs):
        return None

    plate = SimpleNamespace(
        name="plate",
        mass=1.0,
        fixed_base=False,
        aabb=(np.zeros(3), np.asarray((0.3, 0.3, 0.05))),
        get_position_orientation=pose,
        set_position_orientation=move,
    )
    bowl = SimpleNamespace(
        name="bowl",
        mass=1.0,
        fixed_base=None,
        _fixed_base=False,
        aabb=(np.zeros(3), np.asarray((0.2, 0.2, 0.1))),
        get_position_orientation=pose,
        set_position_orientation=move,
    )
    sink = SimpleNamespace(
        name="sink",
        mass=20.0,
        fixed_base=True,
        aabb=(np.zeros(3), np.asarray((0.6, 0.6, 0.3))),
        get_position_orientation=pose,
        set_position_orientation=move,
    )
    visual_only = SimpleNamespace(
        name="decoration",
        mass=1.0,
        fixed_base=False,
        aabb=(np.zeros(3), np.asarray((0.1, 0.1, 0.1))),
        get_position_orientation=pose,
    )
    chair = SimpleNamespace(
        name="chair",
        mass=10.0,
        fixed_base=False,
        aabb=(np.zeros(3), np.asarray((0.6, 0.6, 0.7))),
        get_position_orientation=pose,
        set_position_orientation=move,
    )
    coffee_table = SimpleNamespace(
        name="coffee_table",
        mass=22.5,
        fixed_base=False,
        aabb=(np.zeros(3), np.asarray((1.2, 0.8, 0.5))),
        get_position_orientation=pose,
        set_position_orientation=move,
    )

    assert _live_pickable_objects(
        (plate, bowl, sink, visual_only, chair, coffee_table),
    ) == {
        "plate": plate,
        "bowl": bowl,
        "chair": chair,
    }


@pytest.mark.parametrize(
    "model, extent, center",
    (
        ("vbgcmf", (0.4830796096872743, 0.611241623388989, 0.17985314695151192),
         (-0.006482992232772572, -0.005622086745361937, -0.0055465111399191774)),
        ("nxosyv", (0.4491219157375457, 0.7033743339622829, 0.22501796046904365),
         (0.0007322943705739826, 0.001960904598388158, -5.381532144888013e-05)),
        ("mcaxez", (0.5678395372343319, 0.8852591732983619, 0.2873626638717397),
         (-0.0038046681712786967, 0.0004463111384094276, -0.0031585867864012407)),
    ),
)
@pytest.mark.parametrize(
    "case, expected_stage",
    (
        ("object_pixels", "marker"),
        ("mesh_pixels", "marker"),
        ("fixed_base", "movability"),
        ("missing_mass", "capacity"),
        ("no_instance_pixels", "visibility"),
    ),
)
def test_real_pillow_metadata_through_pick_marker_pipeline(
    model, extent, center, case, expected_stage,
):
    # Frozen real assets: behavior-1k-assets/objects/pillow/<model>/misc/metadata.json
    # collision.axis_aligned extent/translation; mass from avg_category_specs.json.
    # FORCE_CATEGORY_MASS=True in the inspected OmniGibson build. Segmentation is
    # synthetic: this tests filtering/binding, not a live pillow's rendered mask.
    # nxosyv / mcaxez also occur as sofa_frlxgz's extrapart annotations, but
    # these fixtures represent independent objects, not embedded sofa meshes.
    extent, center = np.asarray(extent), np.asarray(center)

    def no_pose_write(**_kwargs):
        pytest.fail("Pick Marker publication must not move the pillow")

    pillow = SimpleNamespace(
        name=f"pillow_{model}_0", category="pillow", model=model,
        prim_path=f"/World/pillow_{model}_0", mass=1.069,
        fixed_base=case == "fixed_base", aabb=(center - extent / 2, center + extent / 2),
        get_position_orientation=lambda: (np.zeros(3), np.asarray((0., 0., 0., 1.))),
        set_position_orientation=no_pose_write, states={},
    )
    if case == "missing_mass":
        del pillow.mass
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    segmentation = np.zeros((32, 32), dtype=np.int64)
    mask = np.zeros_like(segmentation)
    mask[8:24, 8:24] = 1
    if case not in {"mesh_pixels", "no_instance_pixels"}:
        segmentation[:] = mask
    frame = PrivateInstanceSegmentationFrame(
        camera="head", simulator_step=0,
        source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        segmentation=segmentation, instance_id_to_name={1: pillow.name},
        part_segmentation=mask if case == "mesh_pixels" else None,
        part_id_to_prim_path=(
            {1: f"{pillow.prim_path}/base_link/visuals/mesh"}
            if case == "mesh_pixels" else {}
        ),
    )
    visible = _current_visible_scene_objects((pillow,), frame)
    movable = _live_movable_objects(visible)
    pickable = _live_accessible_pickable_objects((pillow,), _live_pickable_objects(visible))
    bundles, blocked = _live_pick_bundle_targets(movable, pickable)
    assert not blocked
    snapshot = build_private_pick_snapshot(
        frame=frame, public_head_rgb=rgb, eligible_objects_by_name=pickable,
        bundle_targets=bundles,
        official_bboxes=((8, 8, 23, 23, 1),), created_action_index=0,
    )
    stage = (
        "visibility" if not visible else "movability" if not movable
        else "capacity" if not pickable else "marker" if snapshot.bindings else "drawing"
    )
    assert stage == expected_stage
    if expected_stage == "marker":
        assert len(snapshot.bindings) == 1
        assert snapshot.bindings["C1"].target_token is pillow
        assert snapshot.bindings["C1"].visible_pixels == 256
    else:
        assert not snapshot.bindings


def test_pick_visibility_accepts_production_visual_mesh_registry():
    visible = SimpleNamespace(name="plate", prim_path="/World/plate")
    hidden = SimpleNamespace(name="table", prim_path="/World/table")
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    part_segmentation = np.zeros((8, 8), dtype=np.int64)
    part_segmentation[2:6, 2:6] = 7
    frame = PrivateInstanceSegmentationFrame(
        camera="head",
        simulator_step=3,
        source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        segmentation=np.zeros((8, 8), dtype=np.int64),
        instance_id_to_name={},
        part_segmentation=part_segmentation,
        part_id_to_prim_path={7: "/World/plate/base_link/visuals"},
    )

    assert _current_visible_scene_objects((visible, hidden), frame) == (visible,)


def test_pick_role_admission_uses_publication_frame_visibility(monkeypatch):
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    target = SimpleNamespace(
        name="stapler", prim_path="/World/stapler", mass=0.2,
        fixed_base=False, states={},
        aabb=(np.asarray((0.0, 0.0, 0.0)), np.asarray((0.1, 0.1, 0.1))),
        get_position_orientation=lambda: (
            np.asarray((0.05, 0.05, 0.05)),
            np.asarray((0.0, 0.0, 0.0, 1.0)),
        ),
        set_position_orientation=lambda **_kwargs: None,
    )
    evaluator = SimpleNamespace(
        obs={
            HEAD_RGB_KEY: rgb,
            HEAD_INSTANCE_SEGMENTATION_KEY: np.zeros((16, 16), dtype=np.int64),
        },
        robot=SimpleNamespace(sensors={
            CAMERA_SENSOR_NAMES["head"]: SimpleNamespace(
                INSTANCE_REGISTRY={1: target.name},
            ),
        }),
        env=SimpleNamespace(scene=SimpleNamespace(objects=(target,))),
    )
    step = {"value": 0}
    monkeypatch.setattr(
        "behavior_hl.embodied.evaluator_bridge.r1pro_b1_component_builders."
        "official_instance_bboxes",
        lambda _frame: ((4, 4, 11, 11, 1),),
    )
    store = PrivateVisualMarkerStore()
    builder = R1ProMarkerSymbolicComponentBuilder(
        symbolic_primitives=SimpleNamespace(
            apply_ref=lambda *_args, **_kwargs: iter(()),
            _get_obj_in_hand=lambda: None,
            _logical_inventory_snapshot=None,
        ),
        grasp_primitive=object(), place_inside_primitive=object(),
        open_primitive=object(), close_primitive=object(),
        release_primitive=object(),
        marker_store=store, implicit_hand="left",
        marker_objects_provider=lambda _evaluator: {target.name: target},
        marker_roles_provider=lambda _evaluator: ((), ()),
    )
    services = builder.build(
        action=ActionEnvelope("pick", MarkerTargetOrRequestArgs()),
        action_index=0, evaluator=evaluator,
        control=SimpleNamespace(hold=lambda: None),
        public_snapshot_provider=lambda: {"simulator_step": step["value"]},
        step_once=lambda *_args: 1, physics_step_once=lambda: 1,
        mark_boundary_already_checkpointed=lambda: None,
        public_carry_snapshot=lambda: {},
        public_grasp_proxy_source=SimpleNamespace(grasp_proxy_likely=lambda: True),
        active_hand=None,
    )

    evaluator.obs[HEAD_INSTANCE_SEGMENTATION_KEY][4:12, 4:12] = 1
    step["value"] = 1
    backend = services[MARKER_SYMBOLIC_BACKEND_SERVICE]
    assert backend.publish_operation_markers(operation="pick", action_index=1) == 1
    assert store.active_snapshot.bindings["C1"].target_token is target


class _RelationState:
    def __init__(self, *targets):
        self._targets = {id(target) for target in targets}

    def get_value(self, target):
        return id(target) in self._targets


class _BooleanState:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value


class _RelationObject:
    def __init__(self, name, lower, upper, *, mass=1.0):
        self.name = name
        self.mass = mass
        self.fixed_base = False
        self.aabb = (np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))
        self.states = {}

    def clear_states_cache(self):
        return None

    def get_position_orientation(self):
        return (
            (self.aabb[0] + self.aabb[1]) / 2.0,
            np.asarray((0.0, 0.0, 0.0, 1.0)),
        )

    def set_position_orientation(self, *_args, **_kwargs):
        return None

    def get_linear_velocity(self):
        return np.zeros(3)

    def get_angular_velocity(self):
        return np.zeros(3)


def test_pick_candidates_exclude_closed_openable_ancestry():
    class Inside:
        pass

    class Open:
        pass

    item = _RelationObject("item", (0, 0, 0), (0.1, 0.1, 0.1))
    drawer = _RelationObject("drawer", (0, 0, 0), (0.5, 0.5, 0.5))
    cabinet = _RelationObject("cabinet", (0, 0, 0), (1.0, 1.0, 1.0))
    counter_item = _RelationObject(
        "counter_item", (2, 0, 0), (2.1, 0.1, 0.1)
    )
    item.states[Inside] = _RelationState(drawer)
    drawer.states[Inside] = _RelationState(cabinet)
    drawer.states[Open] = _BooleanState(True)
    cabinet.states[Open] = _BooleanState(False)
    scene = (item, drawer, cabinet, counter_item)
    candidates = {"item": item, "counter_item": counter_item}

    assert _live_accessible_pickable_objects(scene, candidates) == {
        "counter_item": counter_item,
    }

    cabinet.states[Open].value = True
    assert _live_accessible_pickable_objects(scene, candidates) == candidates

    drawer.states[Open].value = False
    assert _live_accessible_pickable_objects(scene, candidates) == {
        "counter_item": counter_item,
    }


def test_task3_pick_markers_add_one_maximal_bundle_per_plate():
    class OnTop:
        pass

    left_plate = _RelationObject("left_plate", (0, 0, 0), (0.3, 0.3, 0.03))
    left_pizza = _RelationObject(
        "left_pizza", (0.02, 0.02, 0.03), (0.28, 0.28, 0.08)
    )
    right_plate = _RelationObject("right_plate", (1, 0, 0), (1.3, 0.3, 0.03))
    right_pizza = _RelationObject(
        "right_pizza", (1.02, 0.02, 0.03), (1.28, 0.28, 0.08)
    )
    left_pizza.states[OnTop] = _RelationState(left_plate)
    right_pizza.states[OnTop] = _RelationState(right_plate)
    objects = (left_pizza, left_plate, right_pizza, right_plate)
    movable = _live_movable_objects(objects)
    pickable = _live_pickable_objects(objects)
    bundles, blocked = _live_pick_bundle_targets(movable, pickable)

    assert not blocked
    assert len(bundles) == 2
    assert all(isinstance(bundle, PickBundleTargetToken) for bundle in bundles)
    assert {frozenset(bundle.member_names) for bundle in bundles} == {
        frozenset(("left_plate", "left_pizza")),
        frozenset(("right_plate", "right_pizza")),
    }

    rgb = np.zeros((32, 64, 3), dtype=np.uint8)
    segmentation = np.zeros((32, 64), dtype=np.int64)
    names = {}
    boxes = []
    for instance_id, (name, x0) in enumerate(
        (
            ("left_pizza", 4),
            ("left_plate", 16),
            ("right_pizza", 36),
            ("right_plate", 48),
        ),
        start=1,
    ):
        segmentation[10:16, x0 : x0 + 6] = instance_id
        names[instance_id] = name
        boxes.append((x0, 10, x0 + 5, 15, instance_id))
    snapshot = build_private_pick_snapshot(
        frame=_frame(rgb, segmentation, names),
        public_head_rgb=rgb,
        eligible_objects_by_name=pickable,
        bundle_targets=bundles,
        official_bboxes=boxes,
        created_action_index=1,
    )

    assert [binding.binding_kind for binding in snapshot.bindings.values()] == [
        "object",
        "object",
        "object",
        "object",
        "bundle",
        "bundle",
    ]
    assert {
        frozenset(binding.target_token.member_names)
        for binding in snapshot.bindings.values()
        if binding.binding_kind == "bundle"
    } == {
        frozenset(("left_plate", "left_pizza")),
        frozenset(("right_plate", "right_pizza")),
    }


def test_attached_pick_component_has_only_one_bundle_marker():
    class AttachedTo:
        pass

    root = _RelationObject("clip_a", (0, 0, 0), (0.2, 0.2, 0.1))
    middle = _RelationObject("clip_b", (0.2, 0, 0), (0.35, 0.2, 0.1))
    child = _RelationObject("clip_c", (0.35, 0, 0), (0.5, 0.2, 0.1))
    middle.states[AttachedTo] = _RelationState(root)
    child.states[AttachedTo] = _RelationState(middle)
    objects = (root, middle, child)
    pickable = _live_pickable_objects(objects)
    bundles, blocked = _live_pick_bundle_targets(
        _live_movable_objects(objects),
        pickable,
    )
    assert not blocked
    assert len(bundles) == 1 and bundles[0].exclusive
    assert {
        frozenset((relation.child.name, relation.parent.name))
        for relation in bundles[0].relations
    } == {
        frozenset(("clip_a", "clip_b")),
        frozenset(("clip_b", "clip_c")),
    }

    rgb = np.zeros((24, 44, 3), dtype=np.uint8)
    segmentation = np.zeros((24, 44), dtype=np.int64)
    segmentation[8:14, 4:10] = 1
    segmentation[8:14, 18:24] = 2
    segmentation[8:14, 32:38] = 3
    snapshot = build_private_pick_snapshot(
        frame=_frame(
            rgb,
            segmentation,
            {1: "clip_a", 2: "clip_b", 3: "clip_c"},
        ),
        public_head_rgb=rgb,
        eligible_objects_by_name=pickable,
        bundle_targets=bundles,
        official_bboxes=(
            (4, 8, 9, 13, 1),
            (18, 8, 23, 13, 2),
            (32, 8, 37, 13, 3),
        ),
        created_action_index=1,
    )
    assert len(snapshot.bindings) == 1
    assert snapshot.bindings["C1"].binding_kind == "bundle"

    heavy = _RelationObject(
        "heavy_clip",
        (0.5, 0, 0),
        (0.8, 0.2, 0.1),
        mass=10.0,
    )
    heavy.states[AttachedTo] = _RelationState(root)
    blocked_bundles, blocked_names = _live_pick_bundle_targets(
        _live_movable_objects((root, heavy)),
        _live_pickable_objects((root, heavy)),
    )
    assert not blocked_bundles
    assert blocked_names == {"clip_a", "heavy_clip"}


def test_shared_visual_fallback_only_relaxes_pick_marker_pixels():
    laptop = _RelationObject("laptop", (0, 0, 0), (0.4, 0.3, 0.03))
    dice = _RelationObject("dice", (0.5, 0, 0), (0.54, 0.04, 0.04))
    table = _RelationObject(
        "coffee_table",
        (1, 0, 0),
        (2.2, 0.8, 0.5),
        mass=22.5,
    )
    objects = (laptop, dice, table)
    pickable = _live_pickable_objects(objects)
    assert set(pickable) == {"laptop", "dice"}

    rgb = np.zeros((24, 48, 3), dtype=np.uint8)
    segmentation = np.zeros((24, 48), dtype=np.int64)
    segmentation[4:8, 4:8] = 1
    segmentation[4:8, 18:20] = 2
    segmentation[4:12, 30:38] = 3
    snapshot = build_private_pick_snapshot(
        frame=_frame(
            rgb,
            segmentation,
            {1: "laptop", 2: "dice", 3: "coffee_table"},
        ),
        public_head_rgb=rgb,
        eligible_objects_by_name=pickable,
        bundle_targets=(),
        official_bboxes=(
            (4, 4, 7, 7, 1),
            (18, 4, 19, 7, 2),
            (30, 4, 37, 11, 3),
        ),
        created_action_index=1,
    )
    assert [binding.object_name for binding in snapshot.bindings.values()] == [
        "laptop",
        "dice",
    ]
    assert [binding.candidate_source for binding in snapshot.bindings.values()] == [
        "generic",
        "visual_fallback",
    ]


@pytest.mark.parametrize("operation", (
    "pick", "place_inside", "place_on_top", "place_next_to", "place_under",
    "open", "close", "toggle_on", "toggle_off",
))
@pytest.mark.parametrize("recovery_source", ("instance_without_bbox", "mesh"))
def test_marker_actions_complete_each_missing_target(operation, recovery_source):
    class ToggledOn:
        pass

    objects = {
        name: _fillable(SimpleNamespace(
            name=name, prim_path=f"/World/{name}", states={},
        ))
        for name in ("primary", "recovered", "hidden", "ineligible")
    }
    eligible = {name: obj for name, obj in objects.items() if name != "ineligible"}
    rgb = np.zeros((24, 64, 3), dtype=np.uint8)
    segmentation = np.zeros((24, 64), dtype=np.int64)
    segmentation[4:12, 4:12] = 1
    segmentation[4:12, 48:56] = 4
    parts = np.zeros_like(segmentation)
    parts[4:12, 4:12] = 11
    names = {1: "primary", 3: "hidden", 4: "ineligible"}
    paths = {11: "/World/primary/button/visual"}
    if recovery_source == "mesh":
        parts[4:12, 24:32] = 12
        paths[12] = "/World/recovered/body/visual"
    else:
        segmentation[4:12, 24:32] = 2
        names[2] = "recovered"
    for obj in objects.values():
        obj.states[ToggledOn] = SimpleNamespace(
            visual_marker=SimpleNamespace(prim_path=f"{obj.prim_path}/button"),
        )
    frame = PrivateInstanceSegmentationFrame(
        camera="head", simulator_step=3,
        source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        segmentation=segmentation, instance_id_to_name=names,
        part_segmentation=parts, part_id_to_prim_path=paths,
    )
    store = PrivateVisualMarkerStore()

    def no_execution(*_args):
        pytest.fail("Marker publication must not execute or step physics")

    backend = MarkerSymbolicGraspBackendAdapter(
        marker_store=store, private_frame_provider=lambda: frame,
        raw_head_rgb_provider=lambda: rgb,
        eligible_objects_provider=lambda: objects,
        operation_marker_objects_provider=lambda _operation: eligible,
        graspable_object_names_provider=lambda: tuple(eligible),
        place_container_names_provider=lambda: tuple(eligible),
        openable_object_names_provider=lambda: tuple(eligible),
        support_object_names_provider=lambda: tuple(eligible),
        toggleable_object_names_provider=lambda: tuple(eligible),
        # Only one valid object has a bbox. Recovery must run even though the
        # first target (or Toggle button) already produced a marker.
        official_bbox_provider=lambda _frame: ((4, 4, 11, 11, 1),),
        symbolic_generator_factory=no_execution,
        symbolic_place_generator_factory=no_execution,
        command_step=no_execution, hold_safe=no_execution,
    )
    assert backend.publish_operation_markers(operation=operation, action_index=1) == 2
    bindings = store.active_snapshot.bindings
    assert [b.object_name for b in bindings.values()] == ["primary", "recovered"]
    assert bindings["C2"].target_token is objects["recovered"]
    assert bindings["C2"].candidate_source == "visual_fallback"
    assert backend.private_diagnostic()["visual_fallback_count"] == 1
    if operation.startswith("toggle"):
        assert bindings["C1"].binding_kind == "part"
        assert bindings["C2"].binding_kind == "object"
    assert backend.consume_marker(
        marker="C2", action_index=3, operation=operation,
    ).status is MarkerStatus.STALE
    assert backend.publish_operation_markers(operation=operation, action_index=3) == 2
    assert backend.consume_marker(
        marker="C2", action_index=4, operation=operation,
    ).status is MarkerStatus.READY
    assert backend.consume_marker(
        marker="C2", action_index=4, operation=operation,
    ).status is MarkerStatus.CONSUMED


def test_marker_recovery_never_uses_bbox_only_or_borrows_nested_mesh_pixels():
    rgb = np.zeros((16, 32, 3), dtype=np.uint8)
    segmentation = np.zeros((16, 32), dtype=np.int64)
    segmentation[4:12, 4:12] = 1
    parts = np.zeros_like(segmentation)
    parts[4:12, 4:12] = 7
    child = SimpleNamespace(name="child", prim_path="/World/parent/child")
    parent = SimpleNamespace(name="parent", prim_path="/World/parent")
    frame = PrivateInstanceSegmentationFrame(
        camera="head", simulator_step=3,
        source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        segmentation=segmentation, instance_id_to_name={1: "child", 2: "parent"},
        part_segmentation=parts,
        part_id_to_prim_path={7: "/World/parent/child/visual"},
    )
    snapshot = build_private_marker_snapshot(
        frame=frame, public_head_rgb=rgb,
        eligible_objects_by_name={"child": child, "parent": parent},
        # Duplicate boxes must not duplicate the child, and an empty parent
        # box must not acquire the already covered child's mesh pixels.
        official_bboxes=((4, 4, 11, 11, 1), (4, 4, 11, 11, 1), (0, 0, 31, 15, 2)),
        created_action_index=1,
    )
    assert [b.target_token for b in snapshot.bindings.values()] == [child]


def test_recovered_instance_mask_is_preserved_for_surface_readback():
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    segmentation = np.zeros((16, 16), dtype=np.int64)
    segmentation[4:8, 4:6] = 2
    segmentation[4:8, 6:8] = 3
    frame = _frame(rgb, segmentation, {1: "support", 2: "support", 3: "support"})
    snapshot = build_private_marker_snapshot(
        frame=frame, public_head_rgb=rgb,
        eligible_objects_by_name={"support": object()},
        official_bboxes=(), created_action_index=1,
    )
    binding = snapshot.bindings["C1"]
    assert binding.visible_pixels == 16
    assert np.array_equal(_binding_target_mask(frame, binding), segmentation > 0)


@pytest.mark.parametrize("mass", (10.0, 10.01, 0.0, -1.0, np.nan, np.inf, None))
def test_pick_visual_recovery_cannot_bypass_mass_admission(mass):
    target = _RelationObject("item", (0, 0, 0), (0.1, 0.1, 0.1), mass=mass)
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    segmentation = np.zeros((16, 16), dtype=np.int64)
    segmentation[4:8, 4:6] = 1
    snapshot = build_private_pick_snapshot(
        frame=_frame(rgb, segmentation, {1: "item"}), public_head_rgb=rgb,
        eligible_objects_by_name=_live_pickable_objects((target,)),
        bundle_targets=(), official_bboxes=(), created_action_index=1,
    )
    assert bool(snapshot.bindings) is (mass == 10.0)


def test_on_top_candidates_are_geometry_driven_not_category_whitelisted():
    held = _Object("held", "chicken", (4.0, 4.0, 0.8), (4.1, 4.1, 0.9))
    table = _Object("table", "table", (0.0, 0.0, 0.0), (1.2, 1.0, 0.8))
    box = _fillable(
        _Object("box", "box", (0.0, 2.0, 0.0), (1.0, 3.0, 1.0))
    )
    microwave = _Object(
        "microwave", "microwave", (2.0, 0.0, 0.8), (2.8, 0.8, 1.3)
    )
    chicken = _Object("chicken", "chicken", (6.0, 6.0, 0.8), (6.2, 6.2, 1.0))
    wall = _Object("wall", "wall", (8.0, 0.0, 0.0), (8.1, 4.0, 3.0))
    tiny_table = _Object(
        "tiny_table", "table", (10.0, 0.0, 0.0), (10.04, 0.04, 0.8)
    )
    occupied_table = _Object(
        "occupied_table", "table", (12.0, 0.0, 0.0), (13.0, 1.0, 0.8)
    )
    obstacle = _Object(
        "obstacle", "book", (12.0, 0.0, 0.8), (13.0, 1.0, 1.2)
    )
    soap_dish = _Object(
        "soap_dish.n.01_1", "soap_dish", (14.0, 0.0, 0.8), (14.2, 0.2, 0.9)
    )
    floor = _Object("floor", "floors", (0.0, 0.0, 0.0), (20.0, 20.0, 0.1))
    scene = (
        held,
        table,
        microwave,
        chicken,
        wall,
        tiny_table,
        occupied_table,
        obstacle,
        soap_dish,
        floor,
    )
    assert set(_generic_on_top_support_objects(scene, held)) == {
        "chicken",
        "obstacle",
        "occupied_table",
        "microwave",
        "soap_dish.n.01_1",
        "table",
        "tiny_table",
    }

    backend = MarkerSymbolicGraspBackendAdapter(
        marker_store=PrivateVisualMarkerStore(),
        private_frame_provider=lambda: None,
        raw_head_rgb_provider=lambda: np.zeros((1, 1, 3), dtype=np.uint8),
        eligible_objects_provider=lambda: {
            "chicken": chicken,
            "box": box,
            "table": table,
        },
        graspable_object_names_provider=lambda: ("chicken",),
        place_container_names_provider=lambda: ("box", "table"),
        openable_object_names_provider=lambda: (),
        support_object_names_provider=lambda: ("table",),
        operation_marker_objects_provider=lambda _operation: {
            "chicken": chicken,
            "box": box,
            "table": table,
        },
        official_bbox_provider=lambda _frame: (),
        symbolic_generator_factory=lambda _target: iter(()),
        symbolic_place_generator_factory=lambda _target: iter(()),
        command_step=lambda _command: 0,
        hold_safe=lambda: None,
    )
    assert set(backend._operation_marker_objects_for("place_inside")) == {"box"}
    assert set(backend._operation_marker_objects_for("place_on_top")) == {"table"}

    rgb = np.zeros((32, 96, 3), dtype=np.uint8)
    segmentation = np.zeros((32, 96), dtype=np.int64)
    names = {}
    boxes = []
    supports = {}
    for index in range(1, 9):
        x0 = (index - 1) * 12
        segmentation[4:28, x0 : x0 + 10] = index
        name = f"support_{index}"
        names[index] = name
        boxes.append((x0, 4, x0 + 9, 27, index))
        supports[name] = object()
    snapshot = build_private_on_top_support_snapshot(
        frame=_frame(rgb, segmentation, names),
        public_head_rgb=rgb,
        eligible_supports_by_name=supports,
        official_bboxes=boxes,
        created_action_index=1,
    )
    assert len(snapshot.bindings) == len(supports)
    assert {
        binding.object_name for binding in snapshot.bindings.values()
    } == set(supports)
    assert snapshot.operation == "place_on_top"
    assert snapshot.stage == "target"
    assert all(
        binding.binding_kind == "object" and binding.selector is None
        for binding in snapshot.bindings.values()
    )
    assert all(
        binding.target_token is supports[binding.object_name]
        for binding in snapshot.bindings.values()
    )

    first_stage_rgb = np.zeros((32, 108, 3), dtype=np.uint8)
    first_stage_segmentation = np.zeros((32, 108), dtype=np.int64)
    first_stage_names = {}
    first_stage_boxes = []
    containers = {}
    for index in range(1, 9):
        x0 = (index - 1) * 12
        first_stage_segmentation[4:28, x0 : x0 + 10] = index
        name = f"container_{index}"
        first_stage_names[index] = name
        first_stage_boxes.append((x0, 4, x0 + 9, 27, index))
        containers[name] = _fillable(SimpleNamespace(name=name))
    first_stage_segmentation[4:28, 96:106] = 9
    first_stage_names[9] = "bottle"
    first_stage_boxes.append((96, 4, 105, 27, 9))
    operation_candidates = {**containers, "bottle": object()}
    first_stage_frame = _frame(
        first_stage_rgb,
        first_stage_segmentation,
        first_stage_names,
    )
    store = PrivateVisualMarkerStore()
    first_stage_backend = MarkerSymbolicGraspBackendAdapter(
        marker_store=store,
        private_frame_provider=lambda: first_stage_frame,
        raw_head_rgb_provider=lambda: first_stage_rgb,
        eligible_objects_provider=lambda: operation_candidates,
        graspable_object_names_provider=lambda: (),
        place_container_names_provider=lambda: tuple(containers),
        openable_object_names_provider=lambda: (),
        support_object_names_provider=lambda: (),
        operation_marker_objects_provider=lambda _operation: operation_candidates,
        official_bbox_provider=lambda _frame: first_stage_boxes,
        symbolic_generator_factory=lambda _target: iter(()),
        symbolic_place_generator_factory=lambda _target: iter(()),
        command_step=lambda _command: 0,
        hold_safe=lambda: None,
    )
    assert first_stage_backend.publish_operation_markers(
        operation="place_inside",
        action_index=4,
    ) == len(containers)
    assert store.active_snapshot is not None
    assert store.active_snapshot.stage == "reference"
    assert {
        binding.object_name for binding in store.active_snapshot.bindings.values()
    } == set(containers)


def test_pick_uses_one_coherent_current_frame_component():
    rgb = np.zeros((40, 64, 3), dtype=np.uint8)
    segmentation = np.zeros((40, 64), dtype=np.int64)
    segmentation[4:8, 4:8] = 1
    segmentation[30:32, 58:60] = 1
    segmentation[10:13, 20:23] = 2
    apple = object()
    crumb = object()
    frame = _frame(rgb, segmentation, {1: "apple", 2: "crumb"})
    store = PrivateVisualMarkerStore()
    backend = MarkerSymbolicGraspBackendAdapter(
        marker_store=store,
        private_frame_provider=lambda: frame,
        raw_head_rgb_provider=lambda: rgb,
        eligible_objects_provider=lambda: {"apple": apple, "crumb": crumb},
        graspable_object_names_provider=lambda: ("apple", "crumb"),
        place_container_names_provider=lambda: (),
        official_bbox_provider=lambda _frame: (
            (4, 4, 59, 31, 1),
            (20, 10, 22, 12, 2),
        ),
        symbolic_generator_factory=lambda _target: iter(()),
        symbolic_place_generator_factory=lambda _target: iter(()),
        command_step=lambda _command: 0,
        hold_safe=lambda: None,
    )

    assert backend.publish_operation_markers(
        operation="pick", action_index=0
    ) == 2
    binding = store.active_snapshot.bindings["C1"]
    assert binding.object_name == "apple"
    assert binding.visible_pixels == 16
    assert binding.official_bbox_xyxy == (4, 4, 7, 7)
    crumb_binding = store.active_snapshot.bindings["C2"]
    assert crumb_binding.object_name == "crumb"
    assert crumb_binding.candidate_source == "visual_fallback"


def test_second_stage_reference_disappearance_returns_no_markers():
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    visible_segmentation = np.zeros((24, 32), dtype=np.int64)
    visible_segmentation[4:20, 6:22] = 1
    container = _fillable(SimpleNamespace(name="container"))
    other = SimpleNamespace(name="other")
    frames = [
        _frame(rgb, visible_segmentation, {1: "container"}),
        _frame(rgb, np.full((24, 32), 2, dtype=np.int64), {2: "other"}),
    ]
    store = PrivateVisualMarkerStore()
    backend = MarkerSymbolicGraspBackendAdapter(
        marker_store=store,
        private_frame_provider=lambda: frames[0],
        raw_head_rgb_provider=lambda: rgb,
        eligible_objects_provider=lambda: {"container": container, "other": other},
        graspable_object_names_provider=lambda: (),
        place_container_names_provider=lambda: ("container",),
        operation_marker_objects_provider=lambda _operation: {
            "container": container,
            "other": other,
        },
        official_bbox_provider=lambda _frame: ((6, 4, 21, 19, 1),),
        symbolic_generator_factory=lambda _target: iter(()),
        symbolic_place_generator_factory=lambda _target: iter(()),
        command_step=lambda _command: 0,
        hold_safe=lambda: None,
    )
    assert backend.publish_operation_markers(
        operation="place_inside", action_index=0
    ) == 1
    reference = store.active_snapshot.bindings["C1"]
    backend._frame = lambda: frames[1]

    assert backend.publish_operation_markers(
        operation="place_inside",
        action_index=1,
        reference_binding=reference,
    ) == 0
    assert store.active_snapshot is None

    backend._frame = lambda: frames[0]

    def unavailable_snapshot(**_kwargs):
        raise MarkerInputUnavailableError("reference geometry disappeared")

    backend._next_to_snapshot = unavailable_snapshot
    assert backend.publish_operation_markers(
        operation="place_next_to",
        action_index=2,
        reference_binding=reference,
    ) == 0

    def broken_snapshot(**_kwargs):
        raise RuntimeError("snapshot bug")

    backend._next_to_snapshot = broken_snapshot
    with pytest.raises(RuntimeError, match="snapshot bug"):
        backend.publish_operation_markers(
            operation="place_next_to",
            action_index=2,
            reference_binding=reference,
        )


def test_part_prefix_marker_remains_visible_on_fresh_frame():
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    segmentation = np.zeros((24, 32), dtype=np.int64)
    parts = np.zeros_like(segmentation)
    parts[5:17, 8:24] = 7
    support = _Object("counter", "table", (0, 0, 0), (1, 1, 1))
    support.prim_path = "/World/counter"
    frame = PrivateInstanceSegmentationFrame(
        camera="head",
        simulator_step=3,
        source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        segmentation=segmentation,
        instance_id_to_name={},
        part_segmentation=parts,
        part_id_to_prim_path={7: "/World/counter/visuals/top"},
    )

    snapshot = build_private_on_top_support_snapshot(
        frame=frame,
        public_head_rgb=rgb,
        eligible_supports_by_name={support.name: support},
        official_bboxes=(),
        created_action_index=1,
    )

    assert snapshot.bindings["C1"].target_token is support
    assert snapshot.bindings["C1"].mask_source == "seg_instance_id_prefix"
    fresh = PrivateInstanceSegmentationFrame(
        camera="head",
        simulator_step=4,
        source_rgb_sha256=frame.source_rgb_sha256,
        segmentation=segmentation,
        instance_id_to_name={},
        part_segmentation=parts,
        part_id_to_prim_path=frame.part_id_to_prim_path,
    )
    assert current_target_visibility(
        frame=fresh,
        target=support,
    ) == (192, "seg_instance_id_prefix")

    hidden = PrivateInstanceSegmentationFrame(
        camera="head",
        simulator_step=5,
        source_rgb_sha256=frame.source_rgb_sha256,
        segmentation=segmentation,
        instance_id_to_name={},
        part_segmentation=np.zeros_like(parts),
        part_id_to_prim_path=frame.part_id_to_prim_path,
    )
    assert current_target_visibility(
        frame=hidden,
        target=support,
    )[0] == 0


def test_verified_on_top_surface_snapshot_keeps_the_full_2_by_3_grid():
    rgb = np.zeros((60, 90, 3), dtype=np.uint8)
    segmentation = np.zeros((60, 90), dtype=np.int64)
    segmentation[:2] = 1
    part_segmentation = np.full((60, 90), 7, dtype=np.int64)
    support = _Object("counter", "countertop", (0, 0, 0), (3, 2, 1))
    support.prim_path = "/World/counter"
    regions = []
    for row in range(2):
        for column in range(3):
            x0, x1 = column * 30, (column + 1) * 30 - 1
            y0, y1 = row * 30, (row + 1) * 30 - 1
            regions.append(
                OnTopSurfaceRegion(
                    object_name=support.name,
                    selector=f"on_top_r{row}_c{column}",
                    polygon_uv=((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
                    anchor_uv=((x0 + x1) // 2, (y0 + y1) // 2),
                    region_center_xy=(column + 0.5, row + 0.5),
                    region_extent_xy=(1.0, 1.0),
                    surface_z_m=0.9,
                )
            )

    frame = PrivateInstanceSegmentationFrame(
        camera="head",
        simulator_step=3,
        source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        segmentation=segmentation,
        instance_id_to_name={1: support.name},
        part_segmentation=part_segmentation,
        part_id_to_prim_path={7: "/World/counter/visuals/top"},
    )
    snapshot = build_private_on_top_surface_snapshot(
        frame=frame,
        public_head_rgb=rgb,
        eligible_supports_by_name={support.name: support},
        official_bboxes=((0, 0, 89, 59, 1),),
        surface_regions=regions,
        created_action_index=2,
    )

    assert len(snapshot.bindings) == 6
    assert snapshot.stage == "surface_cell"
    assert all(
        binding.binding_kind == "surface_cell"
        and binding.polygon_uv is not None
        and binding.target_token.target_token is support
        and binding.target_token.region_center_xy is not None
        and binding.target_token.region_extent_xy == (1.0, 1.0)
        and binding.target_token.surface_z_m == 0.9
        and binding.mask_source == "seg_instance_or_prefix"
        for binding in snapshot.bindings.values()
    )
    assert np.any(snapshot.marked_rgb != rgb)


def test_on_top_task_fallback_stops_before_the_surface_stage():
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    segmentation = np.zeros((40, 40), dtype=np.int64)
    segmentation[10:30, 12:28] = 1
    soap_dish = _Object(
        "soap_dish.n.01_1",
        "soap_dish",
        (0.0, 0.0, 0.8),
        (0.2, 0.2, 0.9),
    )
    frame = _frame(rgb, segmentation, {1: soap_dish.name})

    reference = build_private_on_top_support_snapshot(
        frame=frame,
        public_head_rgb=rgb,
        eligible_supports_by_name={soap_dish.name: soap_dish},
        official_bboxes=((12, 10, 27, 29, 1),),
        created_action_index=2,
        stage="reference",
    )
    binding = reference.bindings["C1"]
    assert binding.binding_kind == "object"
    assert binding.target_token is soap_dish
    assert reference.stage == "reference"

    surface = build_private_on_top_surface_snapshot(
        frame=frame,
        public_head_rgb=rgb,
        eligible_supports_by_name={soap_dish.name: soap_dish},
        official_bboxes=((12, 10, 27, 29, 1),),
        surface_regions=(),
        created_action_index=3,
    )
    assert not surface.bindings


def test_placement_excludes_held_object_and_reference_selection_skips_reach():
    rgb = np.zeros((32, 64, 3), dtype=np.uint8)
    segmentation = np.zeros((32, 64), dtype=np.int64)
    segmentation[4:28, 4:24] = 1
    segmentation[4:28, 36:56] = 2
    held = _Object("held_box", "box", (0, 0, 0), (1, 1, 1))
    cabinet = _fillable(
        _Object("cabinet", "cabinet", (2, 0, 0), (3, 1, 2))
    )
    frame = _frame(rgb, segmentation, {1: held.name, 2: cabinet.name})
    store = PrivateVisualMarkerStore()
    reach_calls = []

    def reject_reach(target, operation):
        reach_calls.append((target, operation))
        return False, {"distance": 2.0}

    backend = MarkerSymbolicGraspBackendAdapter(
        marker_store=store,
        private_frame_provider=lambda: frame,
        raw_head_rgb_provider=lambda: rgb,
        eligible_objects_provider=lambda: {
            held.name: held,
            cabinet.name: cabinet,
        },
        graspable_object_names_provider=lambda: (),
        place_container_names_provider=lambda: (held.name, cabinet.name),
        official_bbox_provider=lambda _frame: (
            (4, 4, 23, 27, 1),
            (36, 4, 55, 27, 2),
        ),
        symbolic_generator_factory=lambda _target: iter(()),
        symbolic_place_generator_factory=lambda _target: iter(()),
        command_step=lambda _command: 0,
        hold_safe=lambda: None,
        held_object_provider=lambda: held,
        local_reach_admission=reject_reach,
    )

    assert backend.publish_operation_markers(
        operation="place_inside", action_index=0
    ) == 1
    assert store.active_snapshot.bindings["C1"].object_name == cabinet.name
    reference = backend.consume_marker(
        marker="C1", action_index=1, operation="place_inside"
    )
    assert reference.status is MarkerStatus.READY
    assert reach_calls == []

    assert backend.publish_operation_markers(
        operation="place_inside",
        action_index=1,
        reference_binding=reference.binding,
    ) == 1
    assert store.active_snapshot.bindings["C1"].selector == "container_default"
    target = backend.consume_marker(
        marker="C1", action_index=2, operation="place_inside"
    )
    assert target.status is MarkerStatus.UNKNOWN
    assert (
        target.public_failure_reason
        is MarkerPublicFailureReason.TARGET_OUTSIDE_LOCAL_WORKSPACE
    )
    assert reach_calls == [(cabinet, "place_inside")]

    assert backend.publish_operation_markers(
        operation="place_inside",
        action_index=1,
        reference_binding=reference.binding,
    ) == 1

    def unavailable_reach(_target, _operation):
        raise MarkerInputUnavailableError("target geometry disappeared")

    backend._local_reach_admission = unavailable_reach
    unavailable = backend.consume_marker(
        marker="C1", action_index=2, operation="place_inside"
    )
    assert unavailable.status is MarkerStatus.UNKNOWN
    assert unavailable.public_failure_reason is MarkerPublicFailureReason.EXECUTION_UNVERIFIED
    assert store.active_snapshot is None

    assert backend.publish_operation_markers(
        operation="place_inside",
        action_index=1,
        reference_binding=reference.binding,
    ) == 1

    def broken_reach(_target, _operation):
        raise RuntimeError("unexpected reach failure")

    backend._local_reach_admission = broken_reach
    with pytest.raises(RuntimeError, match="unexpected reach failure"):
        backend.consume_marker(
            marker="C1", action_index=2, operation="place_inside"
        )

    store.publish(
        PrivateMarkerSnapshot(
            camera="head",
            simulator_step=frame.simulator_step,
            source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            created_action_index=3,
            marked_rgb=rgb,
            bindings={
                "C1": PrivateMarkerBinding(
                    marker="C1",
                    object_name=held.name,
                    instance_id=1,
                    visible_pixels=480,
                    official_bbox_xyxy=(4, 4, 23, 27),
                    target_token=held,
                )
            },
            operation="place_inside",
            stage="target",
            source_segmentation_sha256=hashlib.sha256(
                np.ascontiguousarray(segmentation, dtype=np.int64).tobytes()
            ).hexdigest(),
        )
    )
    forced_self_target = backend.consume_marker(
        marker="C1", action_index=4, operation="place_inside"
    )
    assert forced_self_target.status is MarkerStatus.UNKNOWN
    assert reach_calls == [(cabinet, "place_inside")]


def test_place_inside_does_not_complete_while_canonical_carry_remains():
    held = object()
    backend = MarkerSymbolicGraspBackendAdapter(
        marker_store=PrivateVisualMarkerStore(),
        private_frame_provider=lambda: None,
        raw_head_rgb_provider=lambda: np.zeros((1, 1, 3), dtype=np.uint8),
        eligible_objects_provider=lambda: {},
        graspable_object_names_provider=lambda: (),
        place_container_names_provider=lambda: (),
        official_bbox_provider=lambda _frame: (),
        symbolic_generator_factory=lambda _target: iter(()),
        symbolic_place_generator_factory=lambda _target: iter(()),
        command_step=lambda _command: 0,
        hold_safe=lambda: None,
        held_object_provider=lambda: held,
    )
    token = backend.begin_symbolic_place_inside(target_token=object())

    tick = backend.tick_symbolic_place_inside(token)

    # Latest Action verifies release before committing the transaction. This
    # fixture has no rollback attestation: retained carry is a failed
    # postcondition, not an ordinary safely retryable blocked placement.
    assert tick.status is SymbolicGraspStatus.FAILED
    assert token.phase == "failed"
    assert token.error_type == "Action23PostconditionError"
    assert "canonical carry remained after release" in token.error
    assert backend._held_object() is held


def test_outcome_verifier_rejects_self_relation():
    verifier = Action23OutcomeVerifier(
        read_state=lambda _obj, _state, _other: True,
        read_held=lambda: None,
    )
    held = object()
    assert not verifier.verify(
        operation="place_inside",
        target=held,
        acted_object=held,
    )["semantic_state_pass"]
    assert verifier.verify(
        operation="place_inside",
        target=object(),
        acted_object=held,
    )["semantic_state_pass"]

    support = object()
    assert verifier.verify(
        operation="place_on_floor",
        target=support,
        acted_object=held,
    )["semantic_state_pass"]


def test_badge_allocator_is_deterministic_and_non_overlapping():
    occupied = []
    positions = [_allocate_badge((8, 8), 12, 8, (64, 48), occupied) for _ in range(6)]
    second_occupied = []
    assert positions == [
        _allocate_badge((8, 8), 12, 8, (64, 48), second_occupied)
        for _ in range(6)
    ]
    for index, first in enumerate(occupied):
        for second in occupied[index + 1 :]:
            assert (
                first[2] + 2 <= second[0]
                or second[2] + 2 <= first[0]
                or first[3] + 2 <= second[1]
                or second[3] + 2 <= first[1]
            )


def test_place_under_keeps_visible_object_region():
    rgb = np.zeros((32, 64, 3), dtype=np.uint8)
    segmentation = np.zeros((32, 64), dtype=np.int64)
    segmentation[4:28, 4:60] = 1
    segmentation[12:20, 24:40] = 2
    table = object()
    frame = _frame(rgb, segmentation, {1: "table", 2: "vase"})
    store = PrivateVisualMarkerStore()
    backend = MarkerSymbolicGraspBackendAdapter(
        marker_store=store,
        private_frame_provider=lambda: frame,
        raw_head_rgb_provider=lambda: rgb,
        eligible_objects_provider=lambda: {"table": table},
        graspable_object_names_provider=lambda: (),
        place_container_names_provider=lambda: (),
        openable_object_names_provider=lambda: (),
        support_object_names_provider=lambda: (),
        operation_marker_objects_provider=lambda _operation: {"table": table},
        official_bbox_provider=lambda _frame: ((4, 4, 59, 27, 1),),
        symbolic_generator_factory=lambda _target: iter(()),
        symbolic_place_generator_factory=lambda _target: iter(()),
        command_step=lambda _command: 0,
        hold_safe=lambda: None,
    )

    assert backend.publish_operation_markers(
        operation="place_under", action_index=0
    ) == 1
    binding = store.active_snapshot.bindings["C1"]
    assert binding.selector is None
    x, y = binding.anchor_uv
    assert segmentation[y, x] == 1


def test_place_under_non_goal_markers_require_a_ray_verified_candidate(monkeypatch):
    held = object()
    feasible = object()
    blocked = object()
    calls = []

    def plans(**kwargs):
        calls.append(kwargs["target"])
        candidates = (object(),) if kwargs["target"] is feasible else ()
        return (
            SimpleNamespace(
                candidate_plan=SimpleNamespace(candidates=candidates)
            ),
        )

    monkeypatch.setattr(
        "behavior_hl.embodied.evaluator_bridge.r1pro_b1_component_builders."
        "plan_live_under_supports",
        plans,
    )

    assert _under_marker_target_objects(
        {"held": held, "table": feasible, "battery": blocked},
        held=held,
        scene_objects=(held, feasible, blocked),
        robot=object(),
        logical_snapshot=None,
    ) == {"table": feasible}
    assert calls == [feasible, blocked]


@pytest.mark.parametrize("visible", [True, False])
def test_under_goal_fallback_marks_visible_target_without_admitting_a_pose(monkeypatch, visible):
    held, child, desk, other = (
        _Object(name, "object", (0, 0, 0), (1, 1, 1))
        for name in ("held", "child", "desk", "other")
    )
    probed = []

    def no_plans(**kwargs):
        probed.append(kwargs["target"])
        return ()

    monkeypatch.setattr(
        "behavior_hl.embodied.evaluator_bridge.r1pro_b1_component_builders."
        "plan_live_under_supports", no_plans,
    )
    candidates = _under_marker_target_objects(
        {"held": held, "child": child, "other": other},
        held=held,
        scene_objects=(held, child, desk, other),
        robot=object(),
        logical_snapshot=SimpleNamespace(object_refs=(held, child)),
        task_fallback_names={"held", "child", "desk"},
    )
    assert candidates == {"desk": desk}
    assert probed == [other]

    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    segmentation = np.zeros((32, 32), dtype=np.int64)
    if visible:
        segmentation[4:28, 4:28] = 1
    store = PrivateVisualMarkerStore()
    physics_commands = []
    evaluator = SimpleNamespace(
        obs={HEAD_RGB_KEY: rgb, HEAD_INSTANCE_SEGMENTATION_KEY: segmentation},
        robot=SimpleNamespace(sensors={
            CAMERA_SENSOR_NAMES["head"]: SimpleNamespace(INSTANCE_REGISTRY={1: "desk"}),
        }),
        env=SimpleNamespace(scene=SimpleNamespace(objects=(held, child, desk, other))),
    )
    monkeypatch.setattr(
        "behavior_hl.embodied.evaluator_bridge.r1pro_b1_component_builders."
        "official_instance_bboxes", lambda _frame: ((4, 4, 27, 27, 1),),
    )
    builder = R1ProMarkerSymbolicComponentBuilder(
        symbolic_primitives=SimpleNamespace(
            apply_ref=lambda *_args, **_kwargs: iter(()),
            _get_obj_in_hand=lambda: held,
            _logical_inventory_snapshot=SimpleNamespace(object_refs=(held, child)),
        ),
        grasp_primitive=object(), place_inside_primitive=object(),
        open_primitive=object(), close_primitive=object(), release_primitive=object(),
        toggle_on_primitive=object(), toggle_off_primitive=object(),
        spatial_generator_factory=lambda **_kwargs: iter(()),
        marker_store=store,
        implicit_hand="left",
        local_approach_enabled=False,
        marker_objects_provider=lambda _evaluator: {"other": other},
        marker_roles_provider=lambda _evaluator: ((), ()),
        under_task_fallback_names_provider=lambda: {"desk"},
    )
    services = builder.build(
        action=ActionEnvelope("place_under", MarkerTargetOrRequestArgs()),
        action_index=0,
        evaluator=evaluator,
        control=SimpleNamespace(hold=lambda: None),
        public_snapshot_provider=lambda: {"simulator_step": 0},
        step_once=lambda *_args: physics_commands.append("step") or 1,
        physics_step_once=lambda: physics_commands.append("physics") or 1,
        mark_boundary_already_checkpointed=lambda: None,
        public_carry_snapshot=lambda: {},
        public_grasp_proxy_source=SimpleNamespace(grasp_proxy_likely=lambda: True),
        active_hand="left",
    )
    backend = services[MARKER_SYMBOLIC_BACKEND_SERVICE]
    assert backend.publish_operation_markers(
        operation="place_under", action_index=0,
    ) == int(visible)
    assert not physics_commands
    if visible:
        assert store.active_snapshot.stage == "target"
        assert store.active_snapshot.bindings["C1"].target_token is desk


def _publish_task_fallback_markers(
    monkeypatch,
    *,
    operation,
    scene_objects,
    segmentation,
    instance_names,
    fallback_names,
    marker_objects=(),
    held=None,
):
    rgb = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
    boxes = []
    for instance_id in sorted(set(np.unique(segmentation)) - {0}):
        ys, xs = np.nonzero(segmentation == instance_id)
        boxes.append((xs.min(), ys.min(), xs.max(), ys.max(), instance_id))
    monkeypatch.setattr(
        "behavior_hl.embodied.evaluator_bridge.r1pro_b1_component_builders."
        "official_instance_bboxes",
        lambda _frame: tuple(boxes),
    )
    store = PrivateVisualMarkerStore()
    evaluator = SimpleNamespace(
        obs={HEAD_RGB_KEY: rgb, HEAD_INSTANCE_SEGMENTATION_KEY: segmentation},
        robot=SimpleNamespace(sensors={
            CAMERA_SENSOR_NAMES["head"]: SimpleNamespace(
                INSTANCE_REGISTRY=instance_names,
            ),
        }),
        env=SimpleNamespace(scene=SimpleNamespace(objects=tuple(scene_objects))),
    )
    builder = R1ProMarkerSymbolicComponentBuilder(
        symbolic_primitives=SimpleNamespace(
            apply_ref=lambda *_args, **_kwargs: iter(()),
            _get_obj_in_hand=lambda: held,
            _logical_inventory_snapshot=None,
        ),
        grasp_primitive=object(), place_inside_primitive=object(),
        open_primitive=object(), close_primitive=object(), release_primitive=object(),
        toggle_on_primitive=object(), toggle_off_primitive=object(),
        spatial_generator_factory=lambda **_kwargs: iter(()),
        marker_store=store, implicit_hand="left", local_approach_enabled=False,
        marker_objects_provider=lambda _evaluator: dict(marker_objects),
        marker_roles_provider=lambda _evaluator: ((), ()),
        task_target_fallback_names_provider=(
            lambda requested: fallback_names if requested == operation else ()
        ),
    )
    services = builder.build(
        action=ActionEnvelope(operation, MarkerTargetOrRequestArgs()),
        action_index=0, evaluator=evaluator,
        control=SimpleNamespace(hold=lambda: None),
        public_snapshot_provider=lambda: {"simulator_step": 0},
        step_once=lambda *_args: 1, physics_step_once=lambda: 1,
        mark_boundary_already_checkpointed=lambda: None,
        public_carry_snapshot=lambda: {},
        public_grasp_proxy_source=SimpleNamespace(grasp_proxy_likely=lambda: True),
        active_hand="left" if operation.startswith("place_") else None,
    )
    backend = services[MARKER_SYMBOLIC_BACKEND_SERVICE]
    return backend.publish_operation_markers(
        operation=operation, action_index=0,
    ), store


def test_task_target_fallback_preserves_visibility_and_role_filters(monkeypatch):
    held, target, other = (
        _Object(name, "object", (0, 0, 0), (1, 1, 1))
        for name in ("held", "target", "other")
    )
    segmentation = np.zeros((32, 64), dtype=np.int64)
    segmentation[4:20, 4:20] = 1
    segmentation[4:20, 36:52] = 2
    count, store = _publish_task_fallback_markers(
        monkeypatch, operation="place_next_to",
        scene_objects=(held, target, other), held=held,
        segmentation=segmentation, instance_names={1: "other", 2: "target"},
        fallback_names={"target"}, marker_objects=(("other", other),),
    )
    assert count == 2
    assert {binding.object_name for binding in store.active_snapshot.bindings.values()} == {
        "other", "target",
    }

    count, store = _publish_task_fallback_markers(
        monkeypatch, operation="place_next_to",
        scene_objects=(held, target, other), held=held,
        segmentation=np.where(segmentation == 2, 0, segmentation),
        instance_names={1: "other", 2: "target"},
        fallback_names={"target"}, marker_objects=(("other", other),),
    )
    assert count == 1
    assert store.active_snapshot.bindings["C1"].object_name == "other"

    count, _ = _publish_task_fallback_markers(
        monkeypatch, operation="open", scene_objects=(target,),
        segmentation=np.where(segmentation == 2, 1, 0),
        instance_names={1: "target"}, fallback_names={"target"},
    )
    assert count == 0


def test_task_target_fallback_keeps_toggle_and_open_part_contracts(monkeypatch):
    class ToggledOn:
        pass

    toggle = _Object("toggle", "object", (0, 0, 0), (1, 1, 1))
    toggle.states = {ToggledOn: SimpleNamespace()}
    segmentation = np.zeros((32, 32), dtype=np.int64)
    segmentation[4:28, 4:28] = 1
    count, store = _publish_task_fallback_markers(
        monkeypatch, operation="toggle_on", scene_objects=(toggle,),
        segmentation=segmentation, instance_names={1: "toggle"},
        fallback_names={"toggle"},
    )
    assert count == 1
    assert store.active_snapshot.bindings["C1"].target_token is toggle

    class Open:
        pass

    links = {
        name: SimpleNamespace(prim_path=f"/World/cabinet/{name}")
        for name in ("left", "right")
    }
    joints = tuple(
        SimpleNamespace(
            name=name, joint_type="prismatic",
            body1=links[name].prim_path,
        )
        for name in links
    )
    cabinet = _Object("cabinet", "object", (0, 0, 0), (1, 1, 1))
    cabinet.links = links
    cabinet.states = {
        Open: SimpleNamespace(relevant_joints_info=(None, joints, (1, 1))),
    }
    count, _ = _publish_task_fallback_markers(
        monkeypatch, operation="open", scene_objects=(cabinet,),
        segmentation=segmentation, instance_names={1: "cabinet"},
        fallback_names={"cabinet"},
    )
    assert count == 0


def test_part_marker_draws_local_box_and_high_contrast_leader():
    color = (0, 205, 225)
    leader = Image.new("RGB", (100, 60), (255, 255, 255))
    _draw_high_contrast_leader(
        ImageDraw.Draw(leader),
        (5, 5, 25, 17),
        (80, 40),
        color,
    )
    leader_pixels = np.asarray(leader)
    assert np.count_nonzero(np.all(leader_pixels == color, axis=2)) > 20
    assert np.count_nonzero(np.all(leader_pixels == (0, 0, 0), axis=2)) > 20
    assert tuple(leader_pixels[40, 80]) == color

    binding = PrivateMarkerBinding(
        marker="C1",
        object_name="drawer_bank",
        instance_id=1,
        visible_pixels=120,
        official_bbox_xyxy=(45, 30, 75, 50),
        target_token=object(),
        binding_kind="part",
        anchor_uv=(60, 40),
    )
    rendered = render_marker_overlay(
        np.full((80, 120, 3), 245, dtype=np.uint8),
        {"C1": binding},
    )
    assert tuple(rendered[27, 42]) == color
    assert tuple(rendered[40, 60]) == color


def test_place_inside_ignores_visual_grid_and_uses_container_default():
    rgb = np.full((240, 240, 3), 45, dtype=np.uint8)
    mask = np.zeros((240, 240), dtype=np.uint8)
    for y in (20, 70, 120, 170, 219):
        mask[max(20, y - 3) : min(220, y + 4), 20:220] = 1
    for x in (20, 86, 153, 219):
        mask[20:220, max(20, x - 3) : min(220, x + 4)] = 1
    rgb[mask.astype(bool)] = 235

    batch = generate_shelf_compartment_markers(
        target_mask=mask,
        public_head_rgb=rgb,
        official_bbox_xyxy=(20, 20, 219, 219),
    )
    assert len(batch.candidates) == 12
    assert {
        (candidate.row_count, candidate.column_count)
        for candidate in batch.candidates
    } == {(4, 3)}
    assert len({candidate.anchor_uv for candidate in batch.candidates}) == 12

    reference = PrivateMarkerBinding(
        marker="C1",
        object_name="container",
        instance_id=1,
        visible_pixels=int(np.count_nonzero(mask)),
        official_bbox_xyxy=(20, 20, 219, 219),
        target_token=_Object(
            "container",
            "fridge",
            (0.0, 0.0, 0.0),
            (3.0, 1.0, 4.0),
        ),
    )
    snapshot = build_private_container_compartment_snapshot(
        frame=_frame(rgb, mask, {1: "container"}),
        public_head_rgb=rgb,
        reference_binding=reference,
        created_action_index=2,
    )
    assert len(snapshot.bindings) == 1
    binding = snapshot.bindings["C1"]
    assert binding.selector == "container_default"
    assert binding.target_token.selector == "container_default"
    assert binding.target_token.target_token is reference.target_token
    assert binding.official_bbox_xyxy == (20, 20, 219, 219)
    assert snapshot.operation == "place_inside"
    assert snapshot.stage == "inside_cell"


def test_place_inside_default_marker_keeps_second_stage_contract():
    rgb = np.full((120, 180, 3), 45, dtype=np.uint8)
    mask = np.zeros((120, 180), dtype=np.uint8)
    mask[15:105, 15:165] = 1
    container = _Object(
        "container", "fridge", (1.0, 0.0, 0.0), (3.0, 1.0, 2.0)
    )
    reference = PrivateMarkerBinding(
        marker="C1",
        object_name=container.name,
        instance_id=1,
        visible_pixels=int(np.count_nonzero(mask)),
        official_bbox_xyxy=(15, 15, 164, 104),
        target_token=container,
    )

    snapshot = build_private_container_compartment_snapshot(
        frame=_frame(rgb, mask, {1: container.name}),
        public_head_rgb=rgb,
        reference_binding=reference,
        created_action_index=2,
    )

    assert len(snapshot.bindings) == 1
    binding = snapshot.bindings["C1"]
    assert binding.selector == "container_default"
    assert binding.binding_kind == "inside_cell"
    assert snapshot.operation == "place_inside"
    assert snapshot.stage == "inside_cell"


def test_nearest_target_xy_uses_face_toward_robot():
    target = _Object("fridge", "fridge", (1.0, -1.0, 0.0), (3.0, 1.0, 2.0))

    nearest, source = _nearest_target_xy(target, (0.0, 0.4))

    assert source == "aabb_nearest_face"
    assert np.allclose(nearest, (1.0, 0.4))
    nearest_from_right, _ = _nearest_target_xy(target, (4.0, 0.2))
    assert np.allclose(nearest_from_right, (3.0, 0.2))


def test_only_missing_target_geometry_is_marker_unavailable():
    with pytest.raises(MarkerInputUnavailableError):
        _resolve_marker_geometry_target(SimpleNamespace(), "pick")

    invalid_position = SimpleNamespace(
        aabb=None,
        get_position_orientation=lambda: (
            np.asarray((np.nan, 0.0, 0.0)),
            np.asarray((0.0, 0.0, 0.0, 1.0)),
        ),
    )
    with pytest.raises(MarkerInputUnavailableError):
        _nearest_target_xy(invalid_position, (0.0, 0.0))

    def broken_pose():
        raise RuntimeError("pose getter bug")

    with pytest.raises(RuntimeError, match="pose getter bug"):
        _nearest_target_xy(
            SimpleNamespace(aabb=None, get_position_orientation=broken_pose),
            (0.0, 0.0),
        )
    with pytest.raises(R1ProB1ComponentBuilderError):
        _nearest_target_xy(invalid_position, (np.nan, 0.0))
