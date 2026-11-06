"""Single declassification boundary for evaluator-only head segmentation."""

from __future__ import annotations

from contextvars import ContextVar
import hashlib
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from behavior_hl.embodied.evaluator_bridge.private_instance_visibility import (
    PrivateInstanceSegmentationFrame,
)

from behavior_hl.embodied.policy.observations import (
    CAMERA_SENSOR_NAMES,
    ROBOT_NAME,
    official_field_contracts,
)


HEAD_PREFIX = f"{ROBOT_NAME}::{CAMERA_SENSOR_NAMES['head']}"
HEAD_RGB_KEY = f"{HEAD_PREFIX}::rgb"
HEAD_SEMANTIC_SEGMENTATION_KEY = f"{HEAD_PREFIX}::seg_semantic"
HEAD_INSTANCE_SEGMENTATION_KEY = f"{HEAD_PREFIX}::seg_instance"
HEAD_SEGMENTATION_KEYS = frozenset(
    {HEAD_SEMANTIC_SEGMENTATION_KEY, HEAD_INSTANCE_SEGMENTATION_KEY}
)
HEAD_INSTANCE_ONLY_SEGMENTATION_KEYS = frozenset(
    {HEAD_INSTANCE_SEGMENTATION_KEY}
)
ACTION_BOUNDARY_TIMING_PRIVATE: ContextVar[dict[str, Any] | None] = ContextVar(
    "action_boundary_timing_private",
    default=None,
)


def declassify_official_observation(
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove only private instance segmentation; reject other extra fields."""

    if not isinstance(raw, Mapping):
        raise TypeError("official observation must be a mapping")
    public_keys = set(official_field_contracts())
    extras = set(raw) - public_keys
    if extras and frozenset(extras) not in {
        HEAD_INSTANCE_ONLY_SEGMENTATION_KEYS,
        HEAD_SEGMENTATION_KEYS,
    }:
        raise ValueError(
            f"unexpected evaluator-only fields: {sorted(extras)}"
        )
    return {key: raw[key] for key in public_keys if key in raw}


def capture_private_head_frame(
    *,
    raw: Mapping[str, Any],
    public_head_rgb: Any,
    simulator_step: int,
    instance_registry: Mapping[int, str],
    part_segmentation: Any | None = None,
    part_instance_registry: Mapping[int, str] | None = None,
) -> PrivateInstanceSegmentationFrame:
    """Freeze one same-observation private mask with an exact RGB witness."""

    declassify_official_observation(raw)
    if HEAD_INSTANCE_SEGMENTATION_KEY not in raw:
        raise ValueError("head instance segmentation is missing")
    public_rgb = np.ascontiguousarray(np.asarray(public_head_rgb))
    raw_rgb = np.ascontiguousarray(np.asarray(raw[HEAD_RGB_KEY])[..., :3])
    if public_rgb.shape != raw_rgb.shape or not np.array_equal(
        public_rgb,
        raw_rgb,
    ):
        raise ValueError("public and private head RGB frames differ")
    return PrivateInstanceSegmentationFrame(
        camera="head",
        simulator_step=simulator_step,
        source_rgb_sha256=hashlib.sha256(
            public_rgb.tobytes(order="C")
        ).hexdigest(),
        segmentation=np.asarray(raw[HEAD_INSTANCE_SEGMENTATION_KEY]),
        instance_id_to_name=instance_registry,
        part_segmentation=(
            None if part_segmentation is None else np.asarray(part_segmentation)
        ),
        part_id_to_prim_path=(
            {} if part_instance_registry is None else part_instance_registry
        ),
    )


def capture_action_boundary_observation(
    runtime: Any,
    *,
    force_render: bool = False,
    timing_private: dict[str, Any] | None = None,
) -> Any:
    """Capture one lane-neutral action boundary without advancing physics.

    Logical carry keeps the object closure at a private off-scene pose during
    every physics step.  When a public RGB boundary needs a carried object (or
    needs to flush a just-released projection), temporarily project it into
    the frozen two-finger hold, render only, and restore it in ``finally``.
    Exact Oracle callers may force the same refresh because their internal
    symbolic execution deliberately suppresses rendering.
    """

    if not isinstance(force_render, bool):
        raise TypeError("force_render must be boolean")
    if timing_private is None:
        timing_private = ACTION_BOUNDARY_TIMING_PRIVATE.get()
    if timing_private is not None and not isinstance(timing_private, dict):
        raise TypeError("timing_private must be a dict or None")
    capture_started = perf_counter()
    if timing_private is not None:
        timing_private.update(
            {
                "schema_version": (
                    "behavior2026.action-boundary-capture-timing/v1"
                ),
                "force_render": force_render,
                "status": "started",
            }
        )

    symbolic_primitives = getattr(runtime, "symbolic_primitives", None)
    capture_required = getattr(
        symbolic_primitives,
        "marker_visual_carry_capture_required",
        None,
    )
    if not callable(capture_required):
        if bool(
            getattr(
                symbolic_primitives,
                "_logical_inventory_carry",
                False,
            )
        ):
            raise RuntimeError(
                "symbolic visual-carry capture contract is unavailable"
            )
        observation_started = perf_counter()
        observation = runtime.environment.observe()
        if timing_private is not None:
            timing_private.update(
                {
                    "mode": "resident_observe_without_render_drain",
                    "resident_observe_seconds": (
                        perf_counter() - observation_started
                    ),
                    "pull_count": 0,
                    "status": "PASS",
                    "total_seconds": perf_counter() - capture_started,
                }
            )
        return observation
    # Every public Action20 boundary uses the same fixed renderer drain.
    # Restricting it to logical-carry frames leaves ordinary base/head/wait
    # After images one camera payload behind the completed physical state.
    # Query the contract for fail-closed compatibility, but freshness is no
    # longer conditional on whether a carry projection is currently needed.
    capture_required()

    import omni.usd
    import omnigibson as og

    evaluator = runtime.environment.evaluator_state()
    if not callable(getattr(getattr(evaluator, "env", None), "get_obs", None)):
        raise RuntimeError("official evaluator environment cannot refresh observations")
    preprocess = getattr(evaluator, "_preprocess_obs", None)
    synchronize = getattr(evaluator, "_sync_lights_and_get_obs", None)
    if not callable(preprocess) or not callable(synchronize):
        raise RuntimeError("official evaluator observation pipeline changed")
    if getattr(runtime, "predicate_mode", None) == "action20_symbolic":
        from .symbolic_light_sync import sync_symbolic_lights
        light_report = sync_symbolic_lights(evaluator.env.scene, symbolic_primitives)
        if timing_private is not None:
            timing_private["light_sync_private"] = light_report
        # The native special-task synchronizer flips on edges and resets all
        # lamps to on. Do not let it undo our absolute projection during pulls.
        synchronize = lambda observation: observation
    begin_projection = getattr(
        runtime.symbolic_primitives,
        "begin_marker_visual_carry_projection",
        None,
    )
    end_projection = getattr(
        runtime.symbolic_primitives,
        "end_marker_visual_carry_projection",
        None,
    )
    consume_projection_warmup = getattr(
        runtime.symbolic_primitives,
        "consume_marker_visual_carry_projection_warmup",
        None,
    )
    consume_release_flush = getattr(
        runtime.symbolic_primitives,
        "consume_marker_visual_release_flush",
        None,
    )
    if not callable(begin_projection) or not callable(end_projection):
        raise RuntimeError("symbolic visual-carry projection API is unavailable")
    if not callable(consume_projection_warmup):
        raise RuntimeError("symbolic visual-carry warmup API is unavailable")
    if not callable(consume_release_flush):
        raise RuntimeError("symbolic visual-release flush API is unavailable")
    projected = False
    pull_records: list[dict[str, Any]] = []
    try:
        # A first logical-carry projection is a large visual teleport.  Render
        # once while the object is still at its private off-scene pose so the
        # RGB history observes the source location empty before projection.
        # The existing five pulls then drain the projected frame as before.
        # All pulls are render-only: physics and task state never advance.
        release_flush = bool(consume_release_flush())
        projection_warmup = bool(consume_projection_warmup())
        warmup_render = release_flush or projection_warmup
        renderer_accumulation_reset = False
        if warmup_render:
            usd_context = omni.usd.get_context()
            reset_renderer_accumulation = getattr(
                usd_context,
                "reset_renderer_accumulation",
                None,
            )
            if not callable(reset_renderer_accumulation):
                raise RuntimeError(
                    "USD context cannot reset renderer accumulation"
                )
            reset_renderer_accumulation()
            renderer_accumulation_reset = True
            og.sim.render()
            payload = evaluator.env.get_obs()
            if not isinstance(payload, tuple) or len(payload) != 2:
                raise RuntimeError(
                    "official environment get_obs must return "
                    "(observation, info)"
                )
            raw_observation, _ = payload
            evaluator.obs = preprocess(synchronize(raw_observation))
        projected = bool(begin_projection())
        observation = None
        for _ in range(5):
            pull_index = len(pull_records)
            render_started = perf_counter()
            og.sim.render()
            render_seconds = perf_counter() - render_started
            get_obs_started = perf_counter()
            payload = evaluator.env.get_obs()
            get_obs_seconds = perf_counter() - get_obs_started
            if not isinstance(payload, tuple) or len(payload) != 2:
                raise RuntimeError(
                    "official environment get_obs must return "
                    "(observation, info)"
                )
            raw_observation, _ = payload
            preprocess_started = perf_counter()
            evaluator.obs = preprocess(synchronize(raw_observation))
            preprocess_seconds = perf_counter() - preprocess_started
            materialize_observation = pull_index == 4
            observe_seconds = 0.0
            hash_seconds = 0.0
            camera_sha256: Mapping[str, Any] | None = None
            if materialize_observation:
                observe_started = perf_counter()
                observation = runtime.environment.observe()
                observe_seconds = perf_counter() - observe_started
                hash_started = perf_counter()
                try:
                    camera_sha256 = dict(
                        runtime.environment.frame_sha256(observation)
                    )
                except Exception as exc:  # Timing cannot alter semantics.
                    camera_sha256 = {
                        "diagnostic_error_type": type(exc).__name__,
                    }
                hash_seconds = perf_counter() - hash_started
            pull_records.append(
                {
                    "pull_index": pull_index,
                    "observation_materialized": materialize_observation,
                    "render_seconds": render_seconds,
                    "get_obs_seconds": get_obs_seconds,
                    "preprocess_sync_seconds": preprocess_seconds,
                    "resident_observe_seconds": observe_seconds,
                    "camera_hash_seconds": hash_seconds,
                    "camera_sha256": camera_sha256,
                }
            )
        if observation is None:
            raise RuntimeError("action-boundary RGB capture produced no observation")
        if timing_private is not None:
            timing_private.update(
                {
                    "mode": "fixed_render_drain",
                    "fixed_pull_budget": 5,
                    "warmup_pull_count": int(warmup_render),
                    "projection_source_clear_pull_count": int(
                        projection_warmup
                    ),
                    "renderer_accumulation_reset": (
                        renderer_accumulation_reset
                    ),
                    "pull_count": len(pull_records),
                    "projected": projected,
                    "pulls_private": tuple(pull_records),
                    "status": "PASS",
                }
            )
        return observation
    finally:
        projection_end_started = perf_counter()
        if projected:
            end_projection()
        if timing_private is not None:
            timing_private["projection_end_seconds"] = (
                perf_counter() - projection_end_started
            )
            timing_private["total_seconds"] = (
                perf_counter() - capture_started
            )
            if timing_private.get("status") == "started":
                timing_private["status"] = "exception"


def capture_rendered_oracle_observation(runtime: Any) -> Any:
    """Compatibility wrapper for exact Oracle render-only boundaries."""

    return capture_action_boundary_observation(runtime, force_render=True)
