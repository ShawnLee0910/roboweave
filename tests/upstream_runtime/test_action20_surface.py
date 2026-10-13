import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from behavior_hl.embodied.evaluator_bridge import (
    run_b1_marker_symbolic_rollout as marker_rollout,
)
from behavior_hl.embodied.evaluator_bridge.inside_task_profiles import (
    get_inside_task_profile,
)
from behavior_hl.embodied.evaluator_bridge.marker_symbolic_grasp_backend import (
    MarkerInputUnavailableError,
)
from behavior_hl.embodied.evaluator_bridge.private_visual_markers import (
    MarkerStatus,
    PrivateMarkerBinding,
    PrivateMarkerResolution,
    SpatialMarkerTargetToken,
    build_private_on_top_surface_snapshot,
)
from behavior_hl.embodied.evaluator_bridge.private_instance_visibility import PrivateInstanceSegmentationFrame
from behavior_hl.embodied.evaluator_bridge.r1pro_b1_component_builders import (
    _spatial_marker_region,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_b1_service_factory import (
    R1ProB1ServiceFactory,
)
from behavior_hl.embodied.policy.action_specs.marker_symbolic_grasp import (
    marker_symbolic_v34_specs,
)
from behavior_hl.embodied.policy.observations import HEAD_RGB_MINIMAL_CAMERA_PROFILE
from behavior_hl.embodied.policy.protocol import (
    ActionEnvelope,
    MarkerTargetOrRequestArgs,
    NoArgs,
)
from behavior_hl.embodied.policy.profiles import (
    build_action23_registry,
    build_inside_task_registry,
    build_picking_up_toys_registry,
)
from behavior_hl.embodied.runtime.availability import ActionAvailability
from behavior_hl.embodied.runtime.dispatch import ActionExecutionContext
from behavior_hl.embodied.runtime.episode import EpisodeSpec, PublicTaskContext
from behavior_hl.embodied.runtime.prompt import (
    ACTION20_PUBLIC_PROFILE,
    ACTION20_RESIDENT_MODEL_PROFILE,
    build_episode_prompt,
)
from behavior_hl.embodied.runtime.plugins.marker_symbolic_grasp import (
    MARKER_SYMBOLIC_BACKEND_SERVICE,
    MarkerSymbolicV34OperationPlugin,
)
from behavior_hl.embodied.runtime.plugins.torso import (
    EMPTY_TORSO_GRID,
    SymbolicTorsoGridPlugin,
)
from behavior_hl.embodied.runtime.plugins.wait import SymbolicWaitPlugin


def test_place_touching_is_not_an_action20_action() -> None:
    action20 = build_picking_up_toys_registry(
        calibrated_hands=("left",),
        grasp_mode="marker_symbolic_action20",
        simple_spatial_selection=True,
    )
    assert len(action20.names) == 19
    assert "place_touching" not in action20.names
    assert "place_touching" in build_action23_registry(
        calibrated_hands=("left",)
    ).names


def test_action20_runtime_requires_logical_nested_carry() -> None:
    with pytest.raises(
        ValueError,
        match="Action20 requires private logical inventory carry",
    ):
        marker_rollout._build_marker_runtime(
            evaluator=object(),
            control=object(),
            profile=object(),
            full_actions=True,
            logical_inventory_carry=False,
            base_execution="restricted_kinematic",
            action_profile="marker_symbolic_action20",
        )


def test_action20_yaml_matches_the_registry() -> None:
    action20 = build_picking_up_toys_registry(
        calibrated_hands=("left",),
        grasp_mode="marker_symbolic_action20",
        simple_spatial_selection=True,
    )
    config = yaml.safe_load(
        (
            __import__("behavior_hl.project_identity", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT
            / "configs/actions/r1pro_esi_marker_symbolic_action20_v1.yaml"
        ).read_text(encoding="utf-8")
    )

    assert config["action_schema_version"] == action20.schema_version
    assert set(config["actions"]) == set(action20.names)
    assert set(config["bindings"]) == set(action20.names)
    assert config["physics_mode"] == "disabled"
    assert config["inside_solid_overlap_ratio"] == 0.0
    assert config["placement_solid_overlap_tolerance_m"] == 0.0005
    assert all("zero_physics" in action20.spec(name).invariants for name in action20.names)
    assert all(
        "final_state_no_solid_overlap" in action20.spec(name).invariants
        for name in {
            "place_inside",
            "place_on_floor",
            "place_on_top",
            "place_next_to",
            "place_under",
        }
    )


def test_action20_symbolic_look_and_wait_use_zero_steps() -> None:
    context = ActionExecutionContext("episode", 0)
    torso = {"qpos": EMPTY_TORSO_GRID[("neutral", "center")]}
    look = SymbolicTorsoGridPlugin(
        controller_key="torso.look_up",
        action_name="look_up",
        read_qpos=lambda: torso["qpos"],
        write_qpos=lambda value: torso.__setitem__("qpos", tuple(value)) or value,
    )

    look_state = look.begin(ActionEnvelope("look_up", NoArgs()), context)
    look_tick = look.tick(look_state, context)
    assert look_tick.done and look_tick.simulator_steps == 0
    assert torso["qpos"] == EMPTY_TORSO_GRID[("neutral", "up")]
    assert look.finish(look_state, context).status == "completed"

    wait = SymbolicWaitPlugin()
    wait_state = wait.begin(ActionEnvelope("wait", NoArgs()), context)
    wait_tick = wait.tick(wait_state, context)
    assert wait_tick.done and wait_tick.simulator_steps == 0
    assert wait_tick.capture_observation is True
    assert wait.finish(wait_state, context).status == "completed"


def test_action20_symbolic_wait_does_not_build_an_actuator() -> None:
    factory = object.__new__(R1ProB1ServiceFactory)
    factory._wait_requires_actuator = False

    assert factory._component_for(
        ActionEnvelope("wait", NoArgs()),
        carry_snapshot={"grasp_proxy_likely": False, "active_hand": None},
    ) is None


def test_action20_prompt_describes_zero_physics_wait_and_navigation() -> None:
    registry = build_picking_up_toys_registry(
        calibrated_hands=("left",),
        grasp_mode="marker_symbolic_action20",
        simple_spatial_selection=True,
    )
    task = PublicTaskContext(
        EpisodeSpec(
            episode_id="episode",
            task_id="Task1",
            split="test",
            instance_id="instance",
            seed=1,
            instruction="Complete the task.",
            action_profile=ACTION20_PUBLIC_PROFILE,
            observation_profile=HEAD_RGB_MINIMAL_CAMERA_PROFILE,
            model_profile=ACTION20_RESIDENT_MODEL_PROFILE,
            action_schema_version=registry.schema_version,
        ),
        registry.names,
    )
    payload = json.loads(
        build_episode_prompt(
            task=task,
            registry=registry,
            availability=ActionAvailability(registry.names, {}),
            frame_sha256={"head": "0" * 64},
            public_history=(),
        )
    )
    constraints = " ".join(payload["constraints"])

    assert "without advancing simulator time" in constraints
    assert "exactly 30 simulator steps" not in constraints
    assert "collision-checked symbolic endpoints" in constraints
    assert "place_on_top = request supports, select a support" in constraints
    assert "place_on_top returns one choice per visible support" not in constraints


def test_action20_surface_grid_requires_upward_first_hits() -> None:
    class Sensor:
        intrinsic_matrix = np.asarray(
            ((100.0, 0.0, 80.0), (0.0, 100.0, 60.0), (0.0, 0.0, 1.0))
        )

        @staticmethod
        def get_position_orientation():
            return np.zeros(3), np.asarray((0.0, 0.0, 0.0, 1.0))

    class Tensors:
        float32 = np.float32

        @staticmethod
        def as_tensor(value, dtype):
            return np.asarray(value, dtype=dtype)

    sensor = Sensor()
    robot = SimpleNamespace(
        sensors={"head": sensor},
        link_prim_paths=("/World/robot",),
    )
    evaluator = SimpleNamespace(
        robot=robot,
        robot_camera_names={"head": "robot::head"},
    )
    support = SimpleNamespace(
        aabb=(
            np.asarray((-0.6, -0.4, -2.2)),
            np.asarray((0.6, 0.4, -2.0)),
        ),
        link_prim_paths=("/World/table",),
    )
    held = SimpleNamespace(
        get_base_aligned_bbox=lambda: (
            np.zeros(3),
            np.asarray((0.0, 0.0, 0.0, 1.0)),
            np.asarray((0.1, 0.1, 0.1)),
            np.zeros(3),
        )
    )

    def rays_with_normal(normal):
        def raytest(starts, ends, **_kwargs):
            direction = ends - starts
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            expected = starts + (
                direction * marker_rollout._ACTION20_SURFACE_RAY_OVERSHOOT_M
            )
            return [
                {
                    "hit": True,
                    "rigidBody": "/World/table/top",
                    "position": point,
                    "normal": normal,
                }
                for point in expected
            ]

        return raytest

    regions = marker_rollout._action20_on_top_surface_regions(
        evaluator=evaluator,
        held=held,
        preferred_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        eligible_supports_by_name={"table": support},
        image_shape=(120, 160, 3),
        raytest_batch=rays_with_normal((0.0, 0.0, 1.0)),
        tensor_module=Tensors,
    )
    assert len(regions) == 6
    assert {region.selector for region in regions} == {
        f"on_top_r{row}_c{column}"
        for row in range(2)
        for column in range(3)
    }
    assert all(np.isclose(region.surface_z_m, -2.0) for region in regions)

    cropped_support = SimpleNamespace(
        aabb=(
            np.asarray((-2.4, -0.4, -2.2)),
            np.asarray((2.4, 0.4, -2.0)),
        ),
        link_prim_paths=("/World/table",),
    )
    cropped = marker_rollout._action20_on_top_surface_regions(
        evaluator=evaluator,
        held=held,
        preferred_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        eligible_supports_by_name={"table": cropped_support},
        image_shape=(120, 160, 3),
        raytest_batch=rays_with_normal((0.0, 0.0, 1.0)),
        tensor_module=Tensors,
    )
    assert len(cropped) == 6
    assert all(
        0 <= x < 160 and 0 <= y < 120
        for region in cropped
        for x, y in (*region.polygon_uv, region.anchor_uv)
    )

    assert marker_rollout._action20_on_top_surface_regions(
        evaluator=evaluator,
        held=held,
        preferred_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        eligible_supports_by_name={"table": support},
        image_shape=(120, 160, 3),
        raytest_batch=rays_with_normal((1.0, 0.0, 0.0)),
        tensor_module=Tensors,
    ) == ()

    def partially_obscured_surface(starts, ends, **_kwargs):
        direction = ends - starts
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        expected = starts + (
            direction * marker_rollout._ACTION20_SURFACE_RAY_OVERSHOOT_M
        )
        return [
            {
                "hit": True,
                "rigidBody": (
                    "/World/appliance" if index % 5 == 1 else "/World/table/top"
                ),
                "position": point,
                "normal": (0.0, 0.0, 1.0),
            }
            for index, point in enumerate(expected)
        ]

    partial = marker_rollout._action20_on_top_surface_regions(
        evaluator=evaluator,
        held=held,
        preferred_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        eligible_supports_by_name={"table": support},
        image_shape=(120, 160, 3),
        raytest_batch=partially_obscured_surface,
        tensor_module=Tensors,
    )
    assert len(partial) == 6

    def center_obscured_surface(starts, ends, **_kwargs):
        results = rays_with_normal((0.0, 0.0, 1.0))(starts, ends)
        for index, result in enumerate(results):
            if index % 5 == 0:
                result["rigidBody"] = "/World/appliance"
        return results

    assert marker_rollout._action20_on_top_surface_regions(
        evaluator=evaluator,
        held=held,
        preferred_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        eligible_supports_by_name={"table": support},
        image_shape=(120, 160, 3),
        raytest_batch=center_obscured_surface,
        tensor_module=Tensors,
    ) == ()

    def lower_surface(starts, ends, **_kwargs):
        results = rays_with_normal((0.0, 0.0, 1.0))(starts, ends)
        for result in results:
            result["position"] -= np.asarray((0.0, 0.0, 0.12))
        return results

    lower = marker_rollout._action20_on_top_surface_regions(
        evaluator=evaluator,
        held=held,
        preferred_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        eligible_supports_by_name={"table": support},
        image_shape=(120, 160, 3),
        raytest_batch=lower_surface,
        tensor_module=Tensors,
    )
    assert len(lower) == 6
    assert all(np.isclose(region.surface_z_m, -2.12) for region in lower)

    def split_level_surface(starts, ends, **_kwargs):
        results = rays_with_normal((0.0, 0.0, 1.0))(starts, ends)
        for result in results:
            if result["position"][1] < 0.0:
                result["position"] -= np.asarray((0.0, 0.0, 0.12))
        return results

    multilevel = marker_rollout._action20_on_top_surface_regions(
        evaluator=evaluator,
        held=held,
        preferred_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        eligible_supports_by_name={"table": support},
        image_shape=(120, 160, 3),
        raytest_batch=split_level_surface,
        tensor_module=Tensors,
    )
    assert len(multilevel) == 6
    assert {round(region.surface_z_m, 2) for region in multilevel} == {
        -2.12,
        -2.0,
    }

    def uneven_surface(starts, ends, **_kwargs):
        results = rays_with_normal((0.0, 0.0, 1.0))(starts, ends)
        for index, result in enumerate(results):
            result["position"] -= np.asarray((0.0, 0.0, 0.03 * (index % 5)))
        return results

    assert marker_rollout._action20_on_top_surface_regions(
        evaluator=evaluator,
        held=held,
        preferred_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        eligible_supports_by_name={"table": support},
        image_shape=(120, 160, 3),
        raytest_batch=uneven_surface,
        tensor_module=Tensors,
    ) == ()

    oversized = SimpleNamespace(
        get_base_aligned_bbox=lambda: (
            np.zeros(3),
            np.asarray((0.0, 0.0, 0.0, 1.0)),
            np.asarray((2.0, 2.0, 2.0)),
            np.zeros(3),
        )
    )
    assert marker_rollout._action20_on_top_surface_regions(
        evaluator=evaluator,
        held=oversized,
        preferred_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        eligible_supports_by_name={"table": support},
        image_shape=(120, 160, 3),
        raytest_batch=rays_with_normal((0.0, 0.0, 1.0)),
        tensor_module=Tensors,
    ) == ()


@pytest.mark.parametrize("held_extent", ((0.28, 0.20, 0.02), (0.40, 0.30, 0.02)))
def test_symbolic_surface_grid_adapts_to_notebook_and_allows_overhang(monkeypatch, held_extent):
    held, support, evaluator, tensors = _local_surface_world(monkeypatch, held_extent)
    support.aabb = (np.array((-0.16, -0.12, -2.1)), np.array((0.16, 0.12, -2.0)))
    batches = []

    def rays(starts, ends, **_kwargs):
        batches.append(len(starts))
        return [_surface_hit(start, -2.0) for start in starts]

    regions = marker_rollout._action20_on_top_surface_regions(
        evaluator=evaluator, held=held, preferred_orientation_xyzw=(0, 0, 0, 1),
        eligible_supports_by_name={"support": support}, image_shape=(120, 160, 3),
        raytest_batch=rays, tensor_module=tensors,
        symbolic_support=True, local_fallback=True,
    )
    assert batches == [5]  # No fallback query when the ordinary surface exists.
    assert len(regions) == 1
    assert regions[0].region_center_xy == (0.0, 0.0)
    assert np.allclose(regions[0].region_extent_xy, (0.32, 0.24))


def _local_surface_world(monkeypatch, extent=(0.1, 0.1, 0.1)):
    held = SimpleNamespace(
        link_prim_paths=("/World/held",),
        get_base_aligned_bbox=lambda: (
            np.zeros(3), np.array((0, 0, 0, 1)), np.array(extent), np.zeros(3),
        ),
    )
    support = SimpleNamespace(
        aabb=(np.array((-0.6, -0.4, -2.3)), np.array((0.6, 0.4, -2.0))),
        link_prim_paths=("/World/support",),
    )
    evaluator = SimpleNamespace(robot=SimpleNamespace(link_prim_paths=("/World/robot",)))
    tensors = SimpleNamespace(float32=np.float32, as_tensor=np.asarray)
    monkeypatch.setattr(marker_rollout, "_live_head_camera_sensor", lambda _: (None, object()))
    monkeypatch.setattr(marker_rollout, "_project_world_points_to_head_pixels",
                        lambda _sensor, points, _shape, **_kwargs:
                        tuple((int(80 + p[0] * 50), int(60 + p[1] * 50)) for p in points))
    return held, support, evaluator, tensors


def _surface_hit(start, height, path="/World/support"):
    return {"hit": True, "rigidBody": path,
            "position": np.array((start[0], start[1], height)), "normal": (0, 0, 1)}


@pytest.mark.parametrize("physical", (False, True))
@pytest.mark.parametrize("case", (
    "coarse_cluster_miss", "hidden_center", "hidden_coarse_hits", "hidden_every_hit",
    "no_hit", "other_object", "vertical",
))
def test_on_top_provider_preserves_surface_evidence_through_publication(monkeypatch, physical, case):
    held, support, evaluator, tensors = _local_surface_world(monkeypatch)
    held.name, support.name = "held", "support"
    evaluator.env = SimpleNamespace(scene=SimpleNamespace(objects=(held, support)))
    batches, first_hit_pixels = [], []

    def rays(starts, ends, **_kwargs):
        batches.append(len(starts))
        if len(batches) == 1:
            first_hit_pixels.extend(marker_rollout._project_world_points_to_head_pixels(None, starts, (120, 160)))
        hits = [_surface_hit(start, -2.12 if case == "coarse_cluster_miss"
                             and min(abs(start[0] - x) for x in (-0.4, 0, 0.4)) < 0.09 else -2.0)
                for start in starts]
        for hit in hits:
            if case == "no_hit":
                hit["hit"] = False
            elif case == "other_object":
                hit["rigidBody"] = "/World/other"
            elif case == "vertical":
                hit["normal"] = (1, 0, 0)
        return hits

    source = ast.parse(Path(marker_rollout.__file__).read_text())
    provider = next(node for node in ast.walk(source)
                    if isinstance(node, ast.FunctionDef) and node.name == "on_top_snapshot_provider")
    namespace = {
        **vars(marker_rollout), "evaluator": evaluator,
        "symbolic": SimpleNamespace(_get_obj_in_hand=lambda: held, _logical_inventory_origin_pose=None),
        "profile": SimpleNamespace(canonical_goal_leaves=()),
        "placement_physical_consequences_enabled": physical, "raytest_batch": rays, "th": tensors,
    }
    exec(compile(ast.Module(body=[provider], type_ignores=[]), "on_top_provider", "exec"), namespace)
    publish = namespace["on_top_snapshot_provider"]
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    segmentation = np.full((120, 160), 7, dtype=np.int64)

    def snapshot_args(step):
        return dict(
            frame=PrivateInstanceSegmentationFrame(
                camera="head", simulator_step=step,
                source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                segmentation=segmentation, instance_id_to_name={7: "support"},
            ),
            public_head_rgb=rgb, official_bboxes=(), created_action_index=step,
            eligible_supports_by_name={"held": held, "support": support},
        )

    # No task-goal fallback: the real shared ray evidence must admit the reference.
    reference = publish(**snapshot_args(0))
    assert batches == [30]
    if case in {"no_hit", "other_object", "vertical"} or (physical and case == "coarse_cluster_miss"):
        assert not reference.bindings
        return
    assert len(reference.bindings) == 1
    selected = reference.bindings["C1"]
    assert selected.target_token is support

    if case in {"hidden_center", "hidden_coarse_hits"}:
        for x, y in first_hit_pixels[::5] if case == "hidden_center" else first_hit_pixels:
            segmentation[y, x] = 0
    elif case == "hidden_every_hit":
        segmentation[:] = 0
        segmentation[:4, :4] = 7  # A visible object alone never invents a surface anchor.
    surface = publish(**snapshot_args(1), reference_binding=selected)
    recovers = not physical and case != "hidden_every_hit"
    assert bool(surface.bindings) is recovers
    needs_local = not physical and case in {"coarse_cluster_miss", "hidden_coarse_hits", "hidden_every_hit"}
    assert batches == [30, 30] + ([30] if needs_local else [])
    for binding in surface.bindings.values():
        assert binding.target_token.target_token is support
        x, y = binding.anchor_uv
        assert segmentation[y, x] == 7
        if case == "hidden_center":
            assert (x, y) in first_hit_pixels
            assert (x, y) not in first_hit_pixels[::5]
        if needs_local:
            assert binding.selector.startswith("on_top_local_")


@pytest.mark.parametrize("symbolic,fallback,expected_batches", (
    (False, True, [30]), (True, False, [30]), (True, True, [30, 30]),
))
def test_selected_surface_recovers_local_seat_without_changing_physical(
    monkeypatch, symbolic, fallback, expected_batches,
):
    held, support, evaluator, tensors = _local_surface_world(monkeypatch)
    batches = []

    def rays(starts, ends, **kwargs):
        batches.append(len(starts))
        if symbolic:
            assert "/World/held" in kwargs["ignore_bodies"]
        # Seat strips cross each cell: its centre is on the seat, but the four
        # distant corners hit a higher back. Small patches really are flat.
        return [_surface_hit(start, -2.12 if min(abs(start[0] - x) for x in (-0.4, 0, 0.4)) < 0.09 else -2.0)
                for start in starts]

    regions = marker_rollout._action20_on_top_surface_regions(
        evaluator=evaluator, held=held, preferred_orientation_xyzw=(0, 0, 0, 1),
        eligible_supports_by_name={"support": support}, image_shape=(120, 160, 3),
        raytest_batch=rays, tensor_module=tensors,
        symbolic_support=symbolic, local_fallback=fallback,
    )
    assert batches == expected_batches
    assert bool(regions) is (symbolic and fallback)
    assert all(region.object_name == "support" and region.selector.startswith("on_top_local_")
               and np.isclose(region.surface_z_m, -2.12) for region in regions)
    if regions:
        rgb = np.zeros((120, 160, 3), dtype=np.uint8)
        frame = PrivateInstanceSegmentationFrame(
            camera="head", simulator_step=0,
            source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            segmentation=np.full((120, 160), 7, dtype=np.int64),
            instance_id_to_name={7: "support"},
        )
        snapshot = build_private_on_top_surface_snapshot(
            frame=frame, public_head_rgb=rgb, eligible_supports_by_name={"support": support},
            official_bboxes=(), surface_regions=regions, created_action_index=1,
        )
        assert len(snapshot.bindings) == len(regions)
        assert snapshot.stage == "surface_cell"
        for binding in snapshot.bindings.values():
            assert binding.target_token.target_token is support
            center, extent, height = _spatial_marker_region(binding.target_token, "place_on_top")
            assert np.allclose(extent, (0.1, 0.1)) and np.isclose(height, -2.12)


@pytest.mark.parametrize("reason", ("no_hit", "other_object", "vertical", "invisible"))
def test_local_surface_fallback_never_invents_evidence(monkeypatch, reason):
    held, support, evaluator, tensors = _local_surface_world(monkeypatch)
    if reason == "invisible":
        monkeypatch.setattr(marker_rollout, "_project_world_points_to_head_pixels", lambda *_a, **_k: None)

    def rays(starts, ends, **_kwargs):
        hits = [_surface_hit(start, -2.0, "/World/other" if reason == "other_object" else "/World/support")
                for start in starts]
        for hit in hits:
            if reason == "no_hit":
                hit["hit"] = False
            if reason == "vertical":
                hit["normal"] = (1, 0, 0)
        return hits

    assert not marker_rollout._action20_on_top_surface_regions(
        evaluator=evaluator, held=held, preferred_orientation_xyzw=(0, 0, 0, 1),
        eligible_supports_by_name={"support": support}, image_shape=(120, 160, 3),
        raytest_batch=rays, tensor_module=tensors, symbolic_support=True, local_fallback=True,
    )


@pytest.mark.parametrize("physical", (False, True))
def test_goal_support_recovery_and_selected_surface_wiring(physical):
    source = ast.parse(Path(marker_rollout.__file__).read_text())
    provider = next(node for node in ast.walk(source)
                    if isinstance(node, ast.FunctionDef) and node.name == "on_top_snapshot_provider")
    held, sofa = SimpleNamespace(name="held"), SimpleNamespace(name="sofa")
    calls = []
    class Snapshot(dict):
        bindings = {}

    def regions(**kwargs):
        calls.append(kwargs)
        return ()  # Even a zero-region first stage must preserve goal fallback.

    namespace = {
        "np": np, "Any": object, "Mapping": dict, "PrivateMarkerSnapshot": object,
        "symbolic": SimpleNamespace(_get_obj_in_hand=lambda: held, _logical_inventory_origin_pose=None),
        "evaluator": SimpleNamespace(env=SimpleNamespace(scene=SimpleNamespace(objects=(held, sofa)))),
        "profile": object(), "placement_physical_consequences_enabled": physical,
        "_action20_task_relation_target_names": lambda *_a, **_k: {"sofa", "held"},
        "_action20_on_top_surface_regions": regions, "raytest_batch": object(), "th": object(),
        "build_private_on_top_support_snapshot": lambda **kwargs: kwargs,
        "build_private_on_top_surface_snapshot": lambda **kwargs: Snapshot(kwargs),
    }
    exec(compile(ast.Module(body=[provider], type_ignores=[]), "on_top_provider", "exec"), namespace)
    publish = namespace["on_top_snapshot_provider"]
    first = publish(public_head_rgb=np.zeros((8, 8, 3)), eligible_supports_by_name={"held": held})
    assert first["eligible_supports_by_name"] == {"sofa": sofa}
    assert calls[-1]["local_fallback"] is False
    second = publish(public_head_rgb=np.zeros((8, 8, 3)), eligible_supports_by_name={},
                     reference_binding=SimpleNamespace(target_token=sofa))
    assert second["eligible_supports_by_name"] == {"sofa": sofa}
    assert calls[-1]["local_fallback"] is True
    assert calls[-1]["symbolic_support"] is (not physical)


def test_symbolic_surface_query_fault_is_not_an_empty_valid_surface(monkeypatch):
    held, support, evaluator, tensors = _local_surface_world(monkeypatch)
    kwargs = dict(evaluator=evaluator, held=held, preferred_orientation_xyzw=(0, 0, 0, 1),
                  eligible_supports_by_name={"support": support}, image_shape=(120, 160, 3),
                  raytest_batch=lambda *_a, **_k: [], tensor_module=tensors)
    assert marker_rollout._action20_on_top_surface_regions(**kwargs) == ()
    with pytest.raises(RuntimeError, match="incomplete batch"):
        marker_rollout._action20_on_top_surface_regions(**kwargs, symbolic_support=True, local_fallback=True)


@pytest.mark.parametrize("symbolic", (False, True))
def test_next_to_publisher_uses_symbolic_overhang_support(monkeypatch, symbolic):
    held, target, evaluator, tensors = _local_surface_world(monkeypatch, (0.4, 0.2, 0.02))
    target.aabb = (np.array((-0.3, -0.15, -2.0)), np.array((0.3, 0.15, -1.98)))
    evaluator.robot.get_position_orientation = lambda: (np.array((0, -2, 0)), np.array((0, 0, 0, 1)))
    sensor = SimpleNamespace(get_position_orientation=lambda: (np.zeros(3), np.array((0, 0, 0, 1))))
    monkeypatch.setattr(marker_rollout, "_live_head_camera_sensor", lambda _: (None, sensor))
    batches = []

    def rays(starts, ends, **_kwargs):
        batches.append(len(starts))
        if len(batches) == 1:
            return [_surface_hit(start, -2.0, "/World/small_support") if index % 5 == 0 else {"hit": False}
                    for index, start in enumerate(starts)]
        return [_surface_hit(start + (-2.0 - start[2]) / (end[2] - start[2]) * (end - start),
                             -2.0, "/World/small_support")
                for start, end in zip(starts, ends, strict=True)]

    positions = marker_rollout._action20_next_to_support_positions(
        evaluator=evaluator, held=held, target=target, preferred_orientation_xyzw=(0, 0, 0, 1),
        image_shape=(120, 160, 3), raytest_batch=rays, tensor_module=tensors, symbolic_support=symbolic,
    )
    assert len(positions) == (8 if symbolic else 0)
    assert batches == ([40, 8] if symbolic else [40])


def test_action20_task_fallback_resolves_scope_to_live_support_name() -> None:
    profile = get_inside_task_profile("tidying_bathroom")
    bar_soap = SimpleNamespace(name="bar_soap_live")
    soap_dish = SimpleNamespace(name="soap_dish_live")
    tissue_dispenser = SimpleNamespace(name="tissue_dispenser_live")
    sink = SimpleNamespace(name="sink_live")
    scope = {
        "bar_soap.n.01_1": bar_soap,
        "soap_dish.n.01_1": soap_dish,
        "tissue_dispenser.n.01_1": tissue_dispenser,
        "sink.n.01_1": sink,
    }

    assert marker_rollout._action20_task_relation_target_names(
        profile,
        (bar_soap,),
        scope,
        relation_kind="on_top",
    ) == frozenset({"soap_dish_live"})
    assert marker_rollout._action20_task_relation_target_names(
        profile,
        (tissue_dispenser,),
        scope,
        relation_kind="on_top",
    ) == frozenset({"sink_live"})
    assert not marker_rollout._action20_task_relation_target_names(
        profile,
        (SimpleNamespace(name="unrelated_live"),),
        scope,
        relation_kind="on_top",
    )

    bundle_profile = SimpleNamespace(
        canonical_goal_leaves=(
            '["ontop","bar_soap.n.01_1","soap_dish.n.01_1"]',
            '["ontop","soap_dish.n.01_1","sink.n.01_1"]',
        )
    )
    assert marker_rollout._action20_task_relation_target_names(
        bundle_profile,
        (soap_dish, bar_soap),
        scope,
        relation_kind="on_top",
    ) == frozenset({"sink_live"})


def test_under_task_fallback_only_uses_current_bundle_positive_under_goals() -> None:
    held, child, desk, other = (
        SimpleNamespace(name=name) for name in ("held", "child", "desk_live", "other")
    )
    profile = SimpleNamespace(canonical_goal_leaves=(
        '["under","held","desk"]',
        '["under","child","desk"]',
        '["under","held","child"]',
        '["under","other","unrelated_target"]',
        '["ontop","held","top_target"]',
        '["not",["under","held","negated_target"]]',
        '["not",["inside","held","container"]]',
    ))
    scope = {
        "held": held, "child": child, "desk": desk, "other": other,
        **{key: SimpleNamespace(name=key) for key in (
            "unrelated_target", "top_target", "negated_target", "container",
        )},
    }
    assert marker_rollout._action20_task_relation_target_names(
        profile, (held, child), scope, relation_kind="under",
    ) == frozenset({"desk_live"})
    assert not marker_rollout._action20_task_relation_target_names(
        profile, (), scope, relation_kind="under",
    )


def test_action20_task_fallback_covers_symmetric_unary_and_pick_targets() -> None:
    objects = {
        name: SimpleNamespace(name=f"{name}_live")
        for name in (
            "rose", "vase", "cup", "book", "lamp", "drawer", "cabinet",
            "plate", "table", "box", "shelf", "other",
        )
    }
    profile = SimpleNamespace(canonical_goal_leaves=(
        '["inside","rose","vase"]',
        '["nextto","cup","book"]',
        '["touching","cup","lamp"]',
        '["ontop","plate","table"]',
        '["under","box","shelf"]',
        '["open","drawer"]',
        '["not",["open","cabinet"]]',
        '["toggled_on","lamp"]',
        '["not",["toggled_on","other"]]',
        '["not",["inside","rose","other"]]',
    ))

    assert marker_rollout._action20_task_relation_target_names(
        profile, (objects["rose"],), objects, relation_kind="inside",
    ) == frozenset({"vase_live"})
    assert marker_rollout._action20_task_relation_target_names(
        profile, (objects["cup"],), objects, relation_kind="next_to",
    ) == frozenset({"book_live"})
    assert marker_rollout._action20_task_relation_target_names(
        profile, (objects["book"],), objects, relation_kind="next_to",
    ) == frozenset({"cup_live"})
    assert marker_rollout._action20_task_relation_target_names(
        profile, (objects["lamp"],), objects, relation_kind="touching",
    ) == frozenset({"cup_live"})
    assert marker_rollout._action20_task_relation_target_names(
        profile, (), objects, relation_kind="open",
    ) == frozenset({"drawer_live"})
    assert marker_rollout._action20_task_relation_target_names(
        profile, (), objects, relation_kind="closed",
    ) == frozenset({"cabinet_live"})
    assert marker_rollout._action20_task_relation_target_names(
        profile, (), objects, relation_kind="toggled_on",
    ) == frozenset({"lamp_live"})
    assert marker_rollout._action20_task_relation_target_names(
        profile, (), objects, relation_kind="not_toggled_on",
    ) == frozenset({"other_live"})
    assert marker_rollout._action20_task_pick_target_names(
        profile, objects,
    ) == frozenset({
        "rose_live", "cup_live", "book_live", "lamp_live",
        "plate_live", "box_live",
    })


def test_action20_next_to_filters_points_hidden_behind_geometry() -> None:
    class Sensor:
        intrinsic_matrix = np.asarray(
            ((100.0, 0.0, 80.0), (0.0, 100.0, 60.0), (0.0, 0.0, 1.0))
        )

        @staticmethod
        def get_position_orientation():
            return np.zeros(3), np.asarray((0.0, 0.0, 0.0, 1.0))

    class Tensors:
        float32 = np.float32

        @staticmethod
        def as_tensor(value, dtype):
            return np.asarray(value, dtype=dtype)

    held = SimpleNamespace(
        aabb=(
            np.asarray((-0.2, -0.1, -10.01)),
            np.asarray((0.2, 0.1, -9.99)),
        ),
        get_position_orientation=lambda: (
            np.asarray((0.0, 0.0, -10.0)),
            np.asarray((0.0, 0.0, 0.0, 1.0)),
        ),
        get_base_aligned_bbox=lambda: (
            np.asarray((0.0, 0.0, -10.0)),
            np.asarray((0.0, 0.0, 0.0, 1.0)),
            np.asarray((0.4, 0.2, 0.02)),
            np.zeros(3),
        ),
    )
    target = SimpleNamespace(
        aabb=(
            np.asarray((-0.3, -0.15, -2.0)),
            np.asarray((0.3, 0.15, -1.98)),
        ),
        link_prim_paths=("/World/target",),
    )
    sensor = Sensor()
    robot = SimpleNamespace(
        sensors={"head": sensor},
        link_prim_paths=("/World/robot",),
        get_position_orientation=lambda: (
            np.asarray((0.0, -2.0, 0.0)),
            np.asarray((0.0, 0.0, 0.0, 1.0)),
        ),
    )
    evaluator = SimpleNamespace(
        robot=robot,
        robot_camera_names={"head": "robot::head"},
    )
    ray_counts = []

    def raytest(starts, ends, **_kwargs):
        ray_counts.append(len(starts))
        if len(ray_counts) == 2:
            direction = ends - starts
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            expected = ends - (
                direction * marker_rollout._ACTION20_SURFACE_RAY_OVERSHOOT_M
            )
            expected[1] = (starts[1] + expected[1]) / 2.0
            return [
                {
                    "hit": True,
                    "rigidBody": (
                        "/World/wall" if index == 1 else "/World/floor"
                    ),
                    "position": point,
                    "normal": (
                        np.asarray((1.0, 0.0, 0.0))
                        if index == 1
                        else np.asarray((0.0, 0.0, 1.0))
                    ),
                }
                for index, point in enumerate(expected)
            ]
        return [
            {
                "hit": True,
                "rigidBody": "/World/floor",
                "position": np.asarray((start[0], start[1], -2.0)),
                "normal": np.asarray((0.0, 0.0, 1.0)),
            }
            for start in starts
        ]

    positions = marker_rollout._action20_next_to_support_positions(
        evaluator=evaluator,
        held=held,
        target=target,
        preferred_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        image_shape=(120, 160, 3),
        raytest_batch=raytest,
        tensor_module=Tensors,
    )

    assert ray_counts == [40, 8]
    assert tuple(position.selector for position in positions) == (
        "screen_left",
        "screen_far",
        "screen_far_right",
        "screen_right",
        "screen_near_right",
        "screen_near",
        "screen_near_left",
    )
    assert len({position.anchor_uv for position in positions}) == 7


def test_next_to_missing_reference_geometry_is_unavailable() -> None:
    with pytest.raises(MarkerInputUnavailableError):
        marker_rollout._action20_next_to_support_positions(
            evaluator=object(),
            held=object(),
            target=SimpleNamespace(),
            preferred_orientation_xyzw=None,
            image_shape=(120, 160, 3),
            raytest_batch=lambda *_args, **_kwargs: (),
            tensor_module=object(),
        )


def test_on_top_surface_token_freezes_the_selected_world_region() -> None:
    token = SpatialMarkerTargetToken(
        object(),
        "place_on_top",
        "on_top_r0_c1",
        region_center_xy=np.asarray((1.25, -0.5)),
        region_extent_xy=np.asarray((0.6, 0.4)),
        surface_z_m=0.7,
    )

    assert _spatial_marker_region(token, "place_on_top") == (
        (1.25, -0.5),
        (0.6, 0.4),
        0.7,
    )
    assert repr(token) == "<SpatialMarkerTargetToken redacted>"


def test_action20_place_on_top_selects_a_verified_surface_cell() -> None:
    action20 = build_picking_up_toys_registry(
        calibrated_hands=("left",),
        grasp_mode="marker_symbolic_action20",
        simple_spatial_selection=True,
    )
    description = action20.spec("place_on_top").model_description

    assert "request support markers, select one" in description
    assert "select a marked flat surface" in description
    assert "If no surface marker appears" not in description
    assert "approach that support once" in description
    assert "ray-verified" not in description
    assert "goal fallback" not in description
    spec = action20.spec("place_on_top")
    assert spec.max_simulator_steps == 4096
    assert {
        "holonomic_base",
        "official_traversability_map",
    }.issubset(spec.required_capabilities)
    assert "no_long_range_navigation" not in spec.invariants
    assert "single_interaction_station" in spec.invariants
    assert "selected_support_then_horizontal_surface_cell" in spec.invariants
    shared = {
        spec.name: spec
        for spec in marker_symbolic_v34_specs(
            simple_spatial_selection=True
        )
    }["place_on_top"]
    assert "select one support C-marker" in shared.model_description
    assert "coarse regions" not in shared.model_description
    assert shared.max_simulator_steps == 1200


def test_support_only_simple_spatial_profiles_are_versioned() -> None:
    action23_simple = build_picking_up_toys_registry(
        calibrated_hands=("left",),
        grasp_mode="marker_symbolic_action23",
        simple_spatial_selection=True,
    )
    v34_simple = build_picking_up_toys_registry(
        calibrated_hands=("left",),
        grasp_mode="marker_symbolic_v34",
        simple_spatial_selection=True,
    )

    assert "simple-spatial-support-v2" in action23_simple.schema_version
    assert "simple-spatial-support-v1" in v34_simple.schema_version

    task_registry = build_inside_task_registry(
        task_name="Task25",
        calibrated_hands=("left",),
        grasp_mode="marker_symbolic_action20",
        base_execution="restricted_kinematic",
        simple_spatial_selection=True,
    )
    assert "on-top-two-stage-multilevel-v13-zero-physics-final-collision" in (
        task_registry.schema_version
    )
    assert "task-Task25-inside-restricted-kinematic-v2" in (
        task_registry.schema_version
    )
    with pytest.raises(ValueError, match="requires restricted_kinematic"):
        build_inside_task_registry(
            task_name="Task25",
            calibrated_hands=("left",),
            grasp_mode="marker_symbolic_action20",
            base_execution="physical_velocity",
            simple_spatial_selection=True,
        )


def test_action_first_on_top_executes_the_selected_surface_cell() -> None:
    support = object()
    surface_token = SpatialMarkerTargetToken(
        support,
        "place_on_top",
        "on_top_r0_c1",
        region_center_xy=(1.0, 2.0),
        region_extent_xy=(0.5, 0.5),
        surface_z_m=0.7,
    )
    calls = []

    class Backend:
        def publish_visible_markers(self, *, action_index):
            raise AssertionError(action_index)

        def consume_marker(self, *, marker, action_index, operation):
            assert (marker, action_index, operation) == (
                "C1",
                3,
                "place_on_top",
            )
            return PrivateMarkerResolution(
                MarkerStatus.READY,
                PrivateMarkerBinding(
                    marker="C1",
                    object_name="counter",
                    instance_id=1,
                    visible_pixels=64,
                    official_bbox_xyxy=(0, 0, 7, 7),
                    target_token=surface_token,
                    binding_kind="surface_cell",
                    selector="on_top_r0_c1",
                    anchor_uv=(3, 3),
                    polygon_uv=((0, 0), (7, 0), (7, 7), (0, 7)),
                ),
            )

        def begin_symbolic_operation(self, *, operation, target_token):
            calls.append((operation, target_token))
            return object()

        def tick_symbolic_operation(self, _token):
            raise AssertionError("the first plugin tick must only start execution")

        def abort_symbolic_operation(self, _token):
            raise AssertionError("the action must not abort")

    plugin = MarkerSymbolicV34OperationPlugin(
        "place_on_top",
        two_stage_on_top=True,
    )
    state = plugin.begin(
        ActionEnvelope(
            "place_on_top",
            MarkerTargetOrRequestArgs("C1"),
        ),
        ActionExecutionContext(
            "episode",
            3,
            {MARKER_SYMBOLIC_BACKEND_SERVICE: Backend()},
        ),
    )
    tick = plugin.tick(
        state,
        ActionExecutionContext(
            "episode",
            3,
            {MARKER_SYMBOLIC_BACKEND_SERVICE: Backend()},
        ),
    )

    assert not tick.done
    assert calls == [("place_on_top", surface_token)]


def test_action20_on_top_object_selection_publishes_surface_stage() -> None:
    support = object()
    publications = []

    class Backend:
        def publish_visible_markers(self, *, action_index):
            raise AssertionError(action_index)

        def consume_marker(self, *, marker, action_index, operation):
            assert (marker, action_index, operation) == (
                "C1",
                3,
                "place_on_top",
            )
            return PrivateMarkerResolution(
                MarkerStatus.READY,
                PrivateMarkerBinding(
                    marker="C1",
                    object_name="sofa",
                    instance_id=1,
                    visible_pixels=64,
                    official_bbox_xyxy=(0, 0, 7, 7),
                    target_token=support,
                ),
            )

        def publish_operation_markers(
            self,
            *,
            operation,
            action_index,
            reference_binding,
        ):
            publications.append(
                (operation, action_index, reference_binding.target_token)
            )
            return 3

        def begin_symbolic_operation(self, **_kwargs):
            raise AssertionError("a support choice must not execute placement")

        def tick_symbolic_operation(self, _token):
            raise AssertionError("a support choice must not execute placement")

        def abort_symbolic_operation(self, _token):
            raise AssertionError("a support choice must not abort")

    backend = Backend()
    context = ActionExecutionContext(
        "episode",
        3,
        {MARKER_SYMBOLIC_BACKEND_SERVICE: backend},
    )
    plugin = MarkerSymbolicV34OperationPlugin(
        "place_on_top",
        two_stage_on_top=True,
    )
    state = plugin.begin(
        ActionEnvelope(
            "place_on_top",
            MarkerTargetOrRequestArgs("C1"),
        ),
        context,
    )

    tick = plugin.tick(state, context)

    assert tick.done
    assert tick.simulator_steps == 0
    assert state.status == "selection_required"
    assert publications == [("place_on_top", 3, support)]
