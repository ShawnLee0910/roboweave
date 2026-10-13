import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from behavior_hl.embodied.evaluator_bridge.official_predicate_verifier import (
    OfficialPredicateVerifier,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_b1_component_builders import (
    R1ProMarkerSymbolicComponentBuilder,
    _read_marker_symbolic_canonical_state,
)
from behavior_hl.embodied.evaluator_bridge.private_visual_markers import (
    PrivateVisualMarkerStore,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_marker_symbolic_rollout import (
    _MarkerRuntime,
)
from behavior_hl.embodied.evaluator_bridge.spatial_symbolic_actions import (
    sample_spatial_stability,
    symbolic_on_top_relation,
)
from behavior_hl.embodied.policy.protocol import ActionEnvelope, NoArgs
from behavior_hl.embodied.runtime.plugins.marker_symbolic_grasp import (
    MARKER_SYMBOLIC_BACKEND_SERVICE,
)
from behavior_hl.embodied.runtime.verification import safe_evaluate


class _Predicate:
    def __init__(self, name: str) -> None:
        self.body = ("inside", name, "cabinet")
        self.truth = False

    def evaluate(self, _predicate_evaluator: object) -> bool:
        return self.truth


class OnTop:
    pass


class _OnTopPredicate:
    STATE_NAME = "ontop"
    inputs = ("item", "support")

    def evaluate(self, predicate_evaluator):
        return predicate_evaluator(OnTop, *self.inputs)


class _Box:
    initialized = True

    def __init__(self, center, extent) -> None:
        self.center = center
        self.extent = extent

    def get_position_orientation(self):
        return self.center, (0.0, 0.0, 0.0, 1.0)

    def get_linear_velocity(self):
        return (0.0, 0.0, 0.0)

    def get_angular_velocity(self):
        return (0.0, 0.0, 0.0)

    @property
    def aabb(self):
        half = tuple(value / 2.0 for value in self.extent)
        return tuple(a - b for a, b in zip(self.center, half)), tuple(
            a + b for a, b in zip(self.center, half)
        )


def _mock_box_surface(monkeypatch, held, support):
    """These fixtures are solid boxes, with an actual flat top to raycast."""
    held.link_prim_paths = ("/World/item/base_link",)
    support.link_prim_paths = ("/World/support/base_link",)
    _set_adjacency(held, support)
    calls = []

    def raytest_batch(starts, ends, **_kwargs):
        calls.append(len(starts))
        surface_z = support.aabb[1][2]
        return [
            {"hit": True, "rigidBody": support.link_prim_paths[0],
             "position": (start[0], start[1], surface_z), "normal": (0, 0, 1)}
            if end[2] <= surface_z <= start[2]
            and all(lo <= xy <= hi for xy, lo, hi in zip(start[:2], support.aabb[0][:2], support.aabb[1][:2]))
            else {"hit": False}
            for start, end in zip(starts, ends, strict=True)
        ]

    monkeypatch.setitem(
        sys.modules, "omnigibson.utils.sampling_utils",
        SimpleNamespace(raytest_batch=raytest_batch),
    )
    return calls


def _set_adjacency(held, support):
    state_type = type("VerticalAdjacency", (), {})
    held.states = {**getattr(held, "states", {}), state_type: SimpleNamespace(
        get_value=lambda: SimpleNamespace(negative_neighbors={support}, positive_neighbors=set()),
    )}


def test_global_model_scoring_does_not_change_selected_oracle_scoring() -> None:
    options = tuple(
        tuple(_Predicate(f"plate_{option}_{leaf}") for leaf in range(2))
        for option in range(2)
    )
    task = SimpleNamespace(
        ground_goal_state_options=options,
        _evaluate_predicate=lambda: None,
        success=False,
    )
    evaluator = SimpleNamespace(env=SimpleNamespace(task=task))
    model = OfficialPredicateVerifier(require_single_option=False)
    oracle = OfficialPredicateVerifier(
        require_single_option=True,
        selected_option_index=0,
    )
    model.evaluate(object(), evaluator)
    oracle.evaluate(object(), evaluator)

    options[1][0].truth = True
    assert model.evaluate(object(), evaluator).official_q == 0.5
    oracle_partial = oracle.evaluate(object(), evaluator)
    assert oracle_partial.official_q == 0.0
    assert (
        oracle_partial.research_diagnostics["official_task_success_scope"]
        == "selected_option"
    )

    options[1][1].truth = True
    task.success = True
    model_complete = model.evaluate(object(), evaluator)
    oracle_complete = oracle.evaluate(object(), evaluator)
    assert model_complete.official_success is True
    assert model_complete.official_q == 1.0
    assert oracle_complete.official_success is False
    assert oracle_complete.official_q == 0.0


@pytest.mark.parametrize("selected_option", (None, 0))
@pytest.mark.parametrize("predicate_mode", ("official", "action20_symbolic"))
def test_scores_use_current_leaves_without_reading_cached_task_success(
    selected_option, predicate_mode,
):
    predicate = _Predicate("item")

    class Task:
        ground_goal_state_options = ((predicate,),)
        object_scope = {}
        _evaluate_predicate = staticmethod(lambda: None)

        @property
        def success(self):
            raise AssertionError("task.success is a stale step-time cache")

    verifier = OfficialPredicateVerifier(
        selected_option_index=selected_option, predicate_mode=predicate_mode,
    )
    evaluator = SimpleNamespace(env=SimpleNamespace(task=Task()))
    for truth in (False, True, False):
        predicate.truth = truth
        result = verifier.evaluate(object(), evaluator)
        assert result.official_success is truth
        assert result.official_q == float(truth)
        assert result.research_diagnostics["raw_official_success_private"] is truth
        assert result.research_diagnostics["raw_official_q_private"] == float(truth)


def test_action20_uses_geometric_on_top_and_preserves_raw_official_score(monkeypatch) -> None:
    held = _Box((0.0, 0.0, 0.8), (0.1, 0.1, 0.1))
    support = _Box((0.0, 0.0, 0.25), (1.0, 1.0, 0.5))
    _mock_box_surface(monkeypatch, held, support)
    task = SimpleNamespace(
        ground_goal_state_options=((_OnTopPredicate(),),),
        object_scope={"item": held, "support": support},
        _evaluate_predicate=lambda _predicate, *_entities: False,
        success=False,
    )
    evaluator = SimpleNamespace(env=SimpleNamespace(task=task))
    episode = SimpleNamespace(action_profile="marker_symbolic_action20")
    verifier = OfficialPredicateVerifier(predicate_mode="action20_symbolic")

    verifier.evaluate(episode, evaluator)
    held.center = (0.0, 0.0, 0.553)
    result = verifier.evaluate(episode, evaluator)

    assert result.official_success is True
    assert result.official_q == 1.0
    assert tuple(result.leaf_truth.values()) == (True,)
    assert result.research_diagnostics["authoritative_score_source"] == (
        "action20_symbolic_geometry"
    )
    assert result.research_diagnostics["raw_official_success_private"] is False
    assert result.research_diagnostics["raw_official_q_private"] == 0.0


@pytest.mark.parametrize("predicate_mode, expected", (
    ("official", False), ("action20_symbolic", True),
))
@pytest.mark.parametrize("profile", ("marker_symbolic_action20", "unrelated_profile"))
def test_final_verifier_uses_explicit_mode_not_profile_name(
    monkeypatch, predicate_mode, expected, profile,
):
    held = _Box((0.0, 0.0, 0.553), (0.1, 0.1, 0.1))
    support = _Box((0.0, 0.0, 0.25), (1.0, 1.0, 0.5))
    calls = _mock_box_surface(monkeypatch, held, support)
    task = SimpleNamespace(
        ground_goal_state_options=((_OnTopPredicate(),),),
        object_scope={"item": held, "support": support},
        _evaluate_predicate=lambda *_args: False,
        success=True,  # Deliberately stale, also in Physical / official mode.
    )
    verifier = OfficialPredicateVerifier(predicate_mode=predicate_mode)
    result = verifier.evaluate(
        SimpleNamespace(action_profile=profile), SimpleNamespace(task=task),
    )
    assert result.official_success is expected
    assert result.official_q == float(expected)
    assert result.research_diagnostics["predicate_mode"] == predicate_mode
    assert result.research_diagnostics["raw_official_q_private"] == 0.0
    assert bool(calls) is expected


@pytest.mark.parametrize("mode", (None, True, "physical", "action20", ""))
def test_invalid_predicate_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="predicate_mode"):
        OfficialPredicateVerifier(predicate_mode=mode)


@pytest.mark.parametrize("zero_physics, physical, expected", (
    (False, False, False), (True, False, True), (True, True, False),
))
@pytest.mark.parametrize("held_extent", ((0.1, 0.1, 0.1), (1.2, 1.2, 0.1)))
def test_runtime_action_and_task_share_placement_mode(monkeypatch, zero_physics, physical, expected, held_extent):
    class Touching:
        pass

    held = _Box((0.0, 0.0, 0.553), held_extent)
    support = _Box((0.0, 0.0, 0.25), (1.0, 1.0, 0.5))
    calls = _mock_box_surface(monkeypatch, held, support)
    held.states = {
        state: SimpleNamespace(get_value=lambda _other: False)
        for state in (OnTop, Touching)
    }
    _set_adjacency(held, support)
    task = SimpleNamespace(
        ground_goal_state_options=((_OnTopPredicate(),),),
        object_scope={"item": held, "support": support},
        _evaluate_predicate=lambda *_args: False,
    )
    env = SimpleNamespace(
        task=task,
        scene=SimpleNamespace(objects=(), object_registry=lambda *_args: support),
    )
    primitives = SimpleNamespace(
        env=env, apply_ref=lambda *_args, **_kwargs: iter(()),
        _get_obj_in_hand=lambda: None,
        _last_private_exact_report={"selected_support_private": "support"},
    )
    runtime = _MarkerRuntime(
        registry=None, dispatcher=None, environment=None, store=PrivateVisualMarkerStore(),
        control=None, carry_tracker=None, symbolic_primitives=primitives,
        carry_rebase_enabled=False, carry_recovery_rebase_enabled=False,
        zero_physics=zero_physics, placement_physical_consequences_enabled=physical,
    )
    assert runtime.predicate_mode == ("action20_symbolic" if expected else "official")
    builder = R1ProMarkerSymbolicComponentBuilder(
        symbolic_primitives=primitives,
        grasp_primitive=object(), place_inside_primitive=object(),
        open_primitive=object(), close_primitive=object(), release_primitive=object(),
        toggle_on_primitive=object(), toggle_off_primitive=object(),
        spatial_generator_factory=lambda **_kwargs: iter(()),
        marker_store=runtime.store, implicit_hand="left", local_approach_enabled=False,
        marker_objects_provider=lambda _evaluator: {},
        marker_roles_provider=lambda _evaluator: ((), ()),
        symbolic_zero_physics=zero_physics,
        placement_physical_consequences_enabled=physical,
        render_without_physics=lambda: None,
    )
    services = builder.build(
        action=ActionEnvelope("place_on_floor", NoArgs()), action_index=0,
        evaluator=SimpleNamespace(env=env), control=SimpleNamespace(hold=lambda: None),
        public_snapshot_provider=lambda: {"simulator_step": 0},
        step_once=lambda *_args: 1, physics_step_once=lambda: 1,
        mark_boundary_already_checkpointed=lambda: None,
        public_carry_snapshot=lambda: {},
        public_grasp_proxy_source=SimpleNamespace(grasp_proxy_likely=lambda: True),
        active_hand="left",
    )
    backend = services[MARKER_SYMBOLIC_BACKEND_SERVICE]
    for operation in ("place_on_top", "place_on_floor"):
        report = backend._private_outcome_verifier(operation, support, held, 0)
        assert report["semantic_state_pass"] is expected
        assert report["predicate_mode"] == runtime.predicate_mode
        assert ("raw_official_observed_private" in report) is expected

    sample = sample_spatial_stability(
        held, support, relation="OnTop", support=support,
        symbolic_on_top=runtime.predicate_mode == "action20_symbolic",
    )
    assert sample.relation is expected
    final = OfficialPredicateVerifier(predicate_mode=runtime.predicate_mode).evaluate(
        SimpleNamespace(action_profile="marker_symbolic_action20"), SimpleNamespace(env=env),
    )
    assert final.official_success is expected
    assert final.research_diagnostics["raw_official_q_private"] == 0.0
    # Two Action attestations, one sample, one Task read; Physical never rays.
    assert calls == ([5] * 4 if expected else [])


def test_all_marker_rollout_verifiers_receive_the_runtime_mode():
    source = (__import__("behavior_hl.project_identity", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT / "src/behavior_hl/embodied/evaluator_bridge"
              / "run_b1_marker_symbolic_rollout.py")
    calls = [node for node in ast.walk(ast.parse(source.read_text()))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "OfficialPredicateVerifier"]
    assert len(calls) == 7
    for call in calls:
        mode = next(keyword for keyword in call.keywords if keyword.arg == "predicate_mode")
        assert ast.unparse(mode.value) == "runtime.predicate_mode"


@pytest.mark.parametrize("surface_z", (0.3, 0.6, 1.0))
@pytest.mark.parametrize("held_extent", ((0.1, 0.1, 0.1), (1.2, 1.2, 0.1)))
@pytest.mark.parametrize(
    "gap, expected",
    ((-0.0004, True), (0.003, True), (0.0149, True), (0.0151, False), (0.2, False)),
)
def test_final_on_top_matches_placement_surface_gap(
    monkeypatch, surface_z, held_extent, gap, expected,
):
    class VerticalAdjacency:
        pass

    class Touching:
        pass

    held = _Box((0.0, 0.0, surface_z + gap + 0.05), held_extent)
    support = _Box((0.0, 0.0, 0.5), (1.0, 1.0, 1.0))
    held.link_prim_paths = ("/World/item/base_link",)
    support.link_prim_paths = ("/World/support/base_link",)
    # Adjacency alone deliberately says true, even for the hovering case.
    held.states = {
        VerticalAdjacency: SimpleNamespace(
            get_value=lambda: SimpleNamespace(
                negative_neighbors={support}, positive_neighbors=set(),
            )
        ),
        Touching: SimpleNamespace(get_value=lambda _other: False),
    }
    calls = []

    def raytest_batch(starts, ends, **kwargs):
        assert kwargs["only_closest"] is True
        assert tuple(kwargs["ignore_bodies"]) == held.link_prim_paths
        assert tuple(kwargs["ignore_collisions"]) == held.link_prim_paths
        assert 1 <= len(starts) <= 5
        assert all(start[2] > held.aabb[1][2] for start in starts)
        calls.append(len(starts))
        return [
            {
                "hit": True,
                "rigidBody": support.link_prim_paths[0],
                "position": (start[0], start[1], surface_z),
                "normal": (0.0, 0.0, 1.0),
            }
            if end[2] <= surface_z <= start[2]
            else {"hit": False}
            for start, end in zip(starts, ends, strict=True)
        ]

    monkeypatch.setitem(
        sys.modules, "omnigibson.utils.sampling_utils",
        SimpleNamespace(raytest_batch=raytest_batch),
    )
    task = SimpleNamespace(
        ground_goal_state_options=((_OnTopPredicate(),),),
        object_scope={"item": held, "support": support},
        _evaluate_predicate=lambda _predicate, *_entities: False,
        success=False,
    )
    evaluator = SimpleNamespace(env=SimpleNamespace(task=task))
    episode = SimpleNamespace(action_profile="marker_symbolic_action20")
    placement = sample_spatial_stability(
        held, support, relation="OnTop", support=support,
        symbolic_on_top=True, support_surface_z_m=surface_z,
    )
    assert placement.relation is expected
    # The selected height only prechecks the gap; successful readback still
    # needs a live surface under the final pose, exactly like Action / Task.
    assert calls == ([] if surface_z + gap > 1.015 else [5])
    assert _read_marker_symbolic_canonical_state(
        held, "OnTop", support, symbolic_on_top=True,
    ) is expected
    verifier = OfficialPredicateVerifier(predicate_mode="action20_symbolic")
    result = verifier.evaluate(episode, evaluator)
    assert result.official_success is expected
    assert result.official_q == float(expected)
    # An AABB may reject an object beyond the furniture's overall height;
    # it must never prove that a real surface exists underneath.
    assert len(calls) == (0 if surface_z + gap > 1.015 else 3)

    # No cached successful pose or old surface height may survive world changes.
    held.center = (0.0, 0.0, surface_z + 0.25)
    assert verifier.evaluate(episode, evaluator).official_success is False


@pytest.mark.parametrize("foot_height", (1.0, 1.003))
def test_near_furniture_aabb_top_is_not_evidence_of_a_surface(monkeypatch, foot_height):
    held = _Box((0.0, 0.0, foot_height + 0.05), (0.1, 0.1, 0.1))
    support = _Box((0.0, 0.0, 0.5), (1.0, 1.0, 1.0))
    held.link_prim_paths = ("/World/item/base_link",)
    support.link_prim_paths = ("/World/support/base_link",)
    _set_adjacency(held, support)
    calls = []

    def raytest_batch(starts, ends, **_kwargs):
        calls.append(len(starts))
        # The sofa back reaches z=1, but its seat below the item is at z=0.8.
        assert all(end[2] > 0.8 for end in ends)
        return [{"hit": False} for _ in starts]

    monkeypatch.setitem(
        sys.modules, "omnigibson.utils.sampling_utils",
        SimpleNamespace(raytest_batch=raytest_batch),
    )
    assert not symbolic_on_top_relation(held, support, support_surface_z_m=0.8)
    assert not symbolic_on_top_relation(held, support)
    assert calls == [5, 5]
    # A selected cell's stale / overly broad plane must not bypass that miss.
    assert not symbolic_on_top_relation(held, support, support_surface_z_m=foot_height)
    assert calls == [5, 5, 5]


@pytest.mark.parametrize("negated", (False, True))
@pytest.mark.parametrize("fault", (
    "query_error", "short_batch", "missing_hit", "missing_identity",
    "non_finite", "missing_links", "missing_object", "uninitialized_object",
    "no_hit",
))
def test_unavailable_on_top_evidence_is_not_a_satisfied_negated_goal(
    monkeypatch, fault, negated,
):
    held = _Box((0.0, 0.0, 0.353), (0.1, 0.1, 0.1))
    support = _Box((0.0, 0.0, 0.5), (1.0, 1.0, 1.0))
    held.link_prim_paths = ("/World/item/base_link",)
    support.link_prim_paths = () if fault == "missing_links" else ("/World/support/base_link",)
    _set_adjacency(held, support)
    if fault == "uninitialized_object":
        support.initialized = False

    class Goal:
        body = ("not", ("ontop", "item", "support")) if negated else (
            "ontop", "item", "support",
        )

        def evaluate(self, evaluate_predicate):
            truth = evaluate_predicate(OnTop, "item", "support")
            return not truth if negated else truth

    def raytest_batch(starts, ends, **_kwargs):
        if fault == "query_error":
            raise RuntimeError("injected ray query failure")
        if fault == "short_batch":
            return []
        if fault == "missing_hit":
            return [{} for _ in starts]
        if fault == "no_hit":
            return [{"hit": False} for _ in starts]
        hit = {
            "hit": True, "rigidBody": "/World/support/base_link",
            "position": (0.0, 0.0, np.nan if fault == "non_finite" else 0.3),
            "normal": (0.0, 0.0, 1.0),
        }
        if fault == "missing_identity":
            del hit["rigidBody"]
        return [hit for _ in starts]

    monkeypatch.setitem(
        sys.modules, "omnigibson.utils.sampling_utils",
        SimpleNamespace(raytest_batch=raytest_batch),
    )
    task = SimpleNamespace(
        ground_goal_state_options=((Goal(),),),
        object_scope={"item": held, "support": None if fault == "missing_object" else support},
        _evaluate_predicate=lambda *_args: False,
    )
    verifier = OfficialPredicateVerifier(predicate_mode="action20_symbolic")
    result = safe_evaluate(verifier, object(), SimpleNamespace(task=task))
    if fault == "no_hit":
        assert result.ok is True
        assert result.official_success is negated
        assert result.official_q == float(negated)
    else:
        assert result.ok is False
        assert result.official_success is False
        assert result.official_q is None
        assert result.error
        assert verifier._initial_states is None


@pytest.mark.parametrize(
    "evidence, expected",
    (
        ("one_hit", True), ("other_support", False), ("no_hit", False),
        ("non_finite", ValueError), ("downward_face", False), ("query_error", RuntimeError),
    ),
)
def test_internal_on_top_requires_nearby_surface_evidence(
    monkeypatch, evidence, expected,
):
    held = _Box((0.0, 0.0, 0.353), (0.1, 0.1, 0.1))
    support = _Box((0.0, 0.0, 0.5), (1.0, 1.0, 1.0))
    held.link_prim_paths = ("/World/item/base_link",)
    support.link_prim_paths = ("/World/support/base_link",)
    _set_adjacency(held, support)

    def raytest_batch(starts, ends, **kwargs):
        if evidence == "query_error":
            raise RuntimeError("ray query unavailable")
        hit = {
            "hit": evidence != "no_hit",
            "rigidBody": (
                "/World/other/base_link" if evidence == "other_support"
                else support.link_prim_paths[0]
            ),
            "position": (
                0.0, 0.0, float("nan") if evidence == "non_finite" else 0.3,
            ),
            "normal": (0.0, 0.0, -1.0 if evidence == "downward_face" else 1.0),
        }
        # A single supporting sample is enough: no new all-corners or
        # coplanarity requirement is imposed on the Symbolic verifier.
        return [hit] + [{"hit": False}] * (len(starts) - 1)

    monkeypatch.setitem(
        sys.modules, "omnigibson.utils.sampling_utils",
        SimpleNamespace(raytest_batch=raytest_batch),
    )
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            symbolic_on_top_relation(held, support)
    else:
        assert symbolic_on_top_relation(held, support) is expected
