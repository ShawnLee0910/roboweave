from types import SimpleNamespace as NS
import hashlib

import numpy as np
import pytest

from test_action20_place_on_top_transaction import _backend
from test_action20_surface import _local_surface_world, _surface_hit
from behavior_hl.embodied.evaluator_bridge import run_b1_marker_symbolic_rollout as rollout
from behavior_hl.embodied.evaluator_bridge.private_instance_visibility import PrivateInstanceSegmentationFrame


def test_each_empty_publication_replaces_previous_diagnostic():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    frame = PrivateInstanceSegmentationFrame(camera="head", simulator_step=0,
        source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        segmentation=np.zeros((8, 8), dtype=np.int64), instance_id_to_name={})
    backend = _backend(prepare=None, place_factory=lambda _: iter(()), frame_provider=lambda: frame, rgb_provider=lambda: rgb)
    backend._last_private_diagnostic = {"surface_count": 23, "error": "OLD"}
    for index, operation in enumerate(("open", "place_inside", "place_on_top")):
        assert backend.publish_operation_markers(operation=operation, action_index=index) == 0
        report = backend.private_diagnostic()
        assert "surface_count" not in report and "error" not in report
        publication = report["marker_publication_private"]
        assert publication["operation"] == operation
        assert publication["action_index"] == index
        assert publication["candidate_count"] == 0
        assert publication["empty_reason"] == "FRAME_UNAVAILABLE"


def test_provider_exception_keeps_current_not_previous_failure():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    frame = PrivateInstanceSegmentationFrame(camera="head", simulator_step=0,
        source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        segmentation=np.zeros((8, 8), dtype=np.int64), instance_id_to_name={1: "box"})
    backend = _backend(prepare=None, place_factory=lambda _: iter(()), frame_provider=lambda: frame, rgb_provider=lambda: rgb)
    def fail(_operation):
        raise RuntimeError("current failure")
    backend._operation_marker_objects = fail
    backend._last_private_diagnostic = {"surface_count": 99, "error": "previous failure"}
    with pytest.raises(RuntimeError, match="current failure"):
        backend.publish_operation_markers(operation="open", action_index=3)
    report = backend.private_diagnostic()
    assert "surface_count" not in report and "error" not in report
    assert report["marker_publication_private"]["error"] == "current failure"
    assert report["marker_publication_private"]["candidate_count"] is None


def test_inside_role_filters_report_exact_reason_without_extra_queries(monkeypatch):
    import behavior_hl.embodied.evaluator_bridge.marker_symbolic_grasp_backend as module
    backend = _backend(prepare=None, place_factory=lambda _: iter(()))
    objects = {name: object() for name in ("wrong_role", "held", "no_volume", "valid")}
    backend._operation_marker_objects = lambda _: objects
    backend._container_names = lambda: ("held", "no_volume", "valid")
    backend._held_object = lambda: objects["held"]
    calls = []
    monkeypatch.setattr(module, "_has_official_fillable_meta_link", lambda obj: calls.append(obj) or obj is objects["valid"])
    backend._publication_report = {"target_filters_private": []}
    assert backend._operation_marker_objects_for("place_inside") == {"valid": objects["valid"]}
    assert calls == [objects["no_volume"], objects["valid"]]
    assert [row["reason"] for row in backend._publication_report["target_filters_private"]] == [
        "ROLE_NOT_ADMITTED", "HELD_OBJECT", "NO_OFFICIAL_FILLABLE_REGION",
    ]


@pytest.mark.parametrize("hit", [False, True])
def test_surface_diagnostic_does_not_change_candidates(monkeypatch, hit):
    held, support, evaluator, tensors = _local_surface_world(monkeypatch)
    def rays(starts, ends, **kwargs):
        return [_surface_hit(start, -2.0) if hit else {"hit": False} for start in starts]
    kwargs = dict(evaluator=evaluator, held=held, preferred_orientation_xyzw=(0, 0, 0, 1),
        eligible_supports_by_name={"support": support}, image_shape=(120, 160, 3),
        raytest_batch=rays, tensor_module=tensors, symbolic_support=True)
    plain = rollout._action20_on_top_surface_regions(**kwargs)
    diagnostic = {}
    observed = rollout._action20_on_top_surface_regions(**kwargs, diagnostic=diagnostic)
    assert [r.selector for r in observed] == [r.selector for r in plain]
    assert diagnostic["region_event_count"] > 0
    assert ("NO_UPWARD_TARGET_HIT" in diagnostic["reason_counts"]) is (not hit)
