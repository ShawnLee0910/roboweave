"""Private, persistent head segmentation side channel for visual Mark.

The public evaluator camera remains RGB-only.  This module attaches one raw
visual-mesh instance-ID annotator to the existing head render product exactly
once, outside the Mark action path.  Object masks are derived privately from
the visible mesh prim paths, so no semantic or remapped-object annotator graph
is needed.  Captures only render and read the already attached annotator; they
never mutate VisionSensor modalities, evaluator observation fields,
camera-parameter annotators, or the physics topology.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from typing import Any

import numpy as np

from behavior_hl.embodied.evaluator_bridge.private_instance_visibility import (
    PrivateInstanceSegmentationFrame,
)
from behavior_hl.embodied.evaluator_bridge.private_observation_ingress import (
    HEAD_INSTANCE_SEGMENTATION_KEY,
    HEAD_RGB_KEY,
    HEAD_SEMANTIC_SEGMENTATION_KEY,
    capture_private_head_frame,
)


PRIVATE_HEAD_SEGMENTATION_SIDE_CHANNEL_VERSION = (
    "behavior2026.private-head-segmentation-side-channel/v3"
)
_PRIVATE_MODALITIES = ("seg_instance_id",)
_DEFAULT_RENDER_PASSES = 4
_OWNER_CHANNEL_ATTRIBUTE = (
    "_behavior_hl_private_head_segmentation_side_channel"
)
_ENVIRONMENT_WRAPPER_INSTALL_TOKEN = object()
_POST_LOAD_INSTALL_TOKEN = object()

AnnotatorFactory = Callable[[Any, str], Any]
EditingUsdContextFactory = Callable[[], Any]
FrameReader = Callable[
    [Any, Mapping[str, Any]],
    tuple[
        Any,
        Any,
        Any,
        Mapping[int, str],
        Any,
        Mapping[int, str],
    ],
]


class PrivateHeadSegmentationSideChannelError(RuntimeError):
    """Raised when the private side channel cannot fail closed."""


class PrivateHeadSegmentationUnavailableError(
    PrivateHeadSegmentationSideChannelError
):
    """Raised when one private frame read is temporarily unavailable."""


def _as_numpy(value: Any) -> np.ndarray:
    for method in ("detach", "cpu"):
        callback = getattr(value, method, None)
        if callable(callback):
            value = callback()
    callback = getattr(value, "numpy", None)
    if callable(callback):
        value = callback()
    return np.asarray(value)


def _rgb_array(value: Any) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim != 3 or array.shape[2] < 3:
        raise PrivateHeadSegmentationSideChannelError(
            "head RGB must be HxWx3+"
        )
    return np.ascontiguousarray(array[..., :3], dtype=np.uint8)


def _absolute_prim_path_registry(
    registry: Mapping[Any, Any],
) -> dict[int, str]:
    """Keep only visual-mesh entries that can bind to a simulator link.

    OmniGibson's global ``INSTANCE_ID_REGISTRY`` intentionally also contains
    sentinel labels such as ``background`` and ``unlabelled``.  They may occur
    in the mask but are not part targets, so the private part registry excludes
    them instead of treating them as malformed prim paths.
    """

    paths: dict[int, str] = {}
    for raw_id, raw_value in registry.items():
        if isinstance(raw_id, bool):
            continue
        try:
            instance_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        prim_path = str(raw_value)
        if prim_path.startswith("/"):
            paths[instance_id] = prim_path
    return paths


def _default_render() -> None:
    import omnigibson as og

    og.sim.render()


def _default_editing_usd_context() -> Any:
    """Return the official USD-edit guard used by ``VisionSensor``."""

    import omnigibson as og

    return og.sim.editing_usd()


def _default_annotator_factory(sensor: Any, modality: str) -> Any:
    """Attach one raw annotator inside the caller's USD-edit guard."""

    if modality not in _PRIVATE_MODALITIES:
        raise PrivateHeadSegmentationSideChannelError(
            f"unsupported private modality: {modality!r}"
        )
    from omnigibson import lazy

    raw_types = getattr(sensor, "_RAW_SENSOR_TYPES", None)
    render_product = getattr(sensor, "_render_product", None)
    if not isinstance(raw_types, Mapping) or modality not in raw_types:
        raise PrivateHeadSegmentationSideChannelError(
            "head sensor raw modality registry is unavailable"
        )
    if render_product is None:
        raise PrivateHeadSegmentationSideChannelError(
            "head sensor render product is unavailable"
        )
    annotator = lazy.omni.replicator.core.AnnotatorRegistry.get_annotator(
        raw_types[modality]
    )
    # This is deliberately not registered in sensor._annotators or
    # sensor.modalities. The caller owns one og.sim.editing_usd() context that
    # covers both raw annotator attachments during Environment-wrapper
    # construction, before Evaluator creates its articulation handles.
    annotator.attach([render_product])
    return annotator


def _official_processed_modality(
    *,
    sensor: Any,
    modality: str,
    raw_observation: Any,
) -> Any:
    import omnigibson as og

    if isinstance(raw_observation, Mapping):
        if "data" not in raw_observation:
            raise PrivateHeadSegmentationUnavailableError(
                f"{modality} annotator omitted data"
            )
        value = raw_observation["data"]
    else:
        value = raw_observation
    device = str(og.sim.device)
    if device == "cpu":
        value = sensor._preprocess_cpu_obs(value, modality)
    elif "cuda" in device:
        value = sensor._preprocess_gpu_obs(value, modality)
    else:
        raise PrivateHeadSegmentationSideChannelError(
            f"unsupported simulator device: {device!r}"
        )
    observation = {modality: value}
    info: dict[str, Any] = {}
    sensor._remap_modality(
        modality,
        observation,
        info,
        raw_observation,
    )
    return observation[modality]


def _default_frame_reader(
    sensor: Any,
    annotators: Mapping[str, Any],
) -> tuple[
    Any,
    Any,
    Any,
    Mapping[int, str],
    Any,
    Mapping[int, str],
]:
    """Read one visual-mesh mask and expose an empty object-mask fallback."""

    import omnigibson as og

    payload = sensor.get_obs()
    sensor_observation = payload[0] if isinstance(payload, tuple) else payload
    if not isinstance(sensor_observation, Mapping) or "rgb" not in sensor_observation:
        raise PrivateHeadSegmentationUnavailableError(
            "RGB-only head sensor returned no RGB"
        )
    modality = "seg_instance_id"
    annotator = annotators.get(modality)
    if annotator is None:
        raise PrivateHeadSegmentationSideChannelError(
            f"private annotator is missing: {modality}"
        )
    raw = annotator.get_data(device=og.sim.device)
    part_instance = _official_processed_modality(
        sensor=sensor,
        modality=modality,
        raw_observation=raw,
    )
    part_registry = getattr(sensor, "INSTANCE_ID_REGISTRY", None)
    if not isinstance(part_registry, Mapping):
        raise PrivateHeadSegmentationSideChannelError(
            "head sensor has no visual-mesh instance registry"
        )
    part_array = _as_numpy(part_instance)
    empty_object_mask = np.zeros(part_array.shape[:2], dtype=np.int64)
    return (
        sensor_observation["rgb"],
        empty_object_mask,
        empty_object_mask,
        {},
        part_instance,
        _absolute_prim_path_registry(part_registry),
    )


class PrivateHeadSegmentationSideChannel:
    """Persistent private segmentation reader isolated from evaluator cameras."""

    def __init__(
        self,
        *,
        evaluator: Any | None = None,
        head_sensor_name: str,
        sensor: Any | None = None,
        annotator_factory: AnnotatorFactory | None = None,
        editing_usd_context_factory: EditingUsdContextFactory | None = None,
        frame_reader: FrameReader | None = None,
        render_callback: Callable[[], None] | None = None,
        render_passes: int = _DEFAULT_RENDER_PASSES,
    ) -> None:
        if evaluator is None and sensor is None:
            raise TypeError("evaluator or sensor must be supplied")
        if not isinstance(head_sensor_name, str) or not head_sensor_name:
            raise TypeError("head_sensor_name must be a non-empty string")
        for callback, name in (
            (annotator_factory, "annotator_factory"),
            (editing_usd_context_factory, "editing_usd_context_factory"),
            (frame_reader, "frame_reader"),
            (render_callback, "render_callback"),
        ):
            if callback is not None and not callable(callback):
                raise TypeError(f"{name} must be callable")
        if (
            isinstance(render_passes, bool)
            or not isinstance(render_passes, int)
            or render_passes <= 0
        ):
            raise ValueError("render_passes must be a positive integer")
        self._evaluator = evaluator
        self._sensor = sensor
        self._environment_owner: Any | None = None
        self._head_sensor_name = head_sensor_name
        self._uses_production_annotator_factory = annotator_factory is None
        self._annotator_factory = (
            annotator_factory or _default_annotator_factory
        )
        self._editing_usd_context_factory = (
            editing_usd_context_factory
            or (
                _default_editing_usd_context
                if self._uses_production_annotator_factory
                else nullcontext
            )
        )
        self._frame_reader = frame_reader or _default_frame_reader
        self._render = render_callback or _default_render
        self._render_passes = render_passes
        self._annotators: dict[str, Any] = {}
        self._started = False
        self._installation_attempted = False
        self._installation_phase: str | None = None
        self._resident_modalities: frozenset[str] | None = None
        self._cached_frame: PrivateInstanceSegmentationFrame | None = None
        self._last_report: dict[str, Any] = {
            "schema_version": PRIVATE_HEAD_SEGMENTATION_SIDE_CHANNEL_VERSION,
            "status": "idle",
        }

    @property
    def evaluator(self) -> Any:
        return self._evaluator

    @property
    def head_sensor_name(self) -> str:
        return self._head_sensor_name

    @property
    def started(self) -> bool:
        return self._started

    @property
    def last_report(self) -> Mapping[str, Any]:
        return dict(self._last_report)

    def invalidate_cache(self) -> None:
        """Drop one cached frame after a render-only camera/scene reposition."""

        self._cached_frame = None
        self._last_report = {
            **self._last_report,
            "status": "cache_invalidated",
            "cache_invalidated": True,
        }

    def _head_sensor(self) -> Any:
        if self._sensor is not None:
            return self._sensor
        if self._evaluator is None:
            raise PrivateHeadSegmentationSideChannelError(
                "official head VisionSensor is unavailable"
            )
        try:
            sensor = self._evaluator.robot.sensors[self._head_sensor_name]
        except (AttributeError, KeyError, TypeError) as exc:
            raise PrivateHeadSegmentationSideChannelError(
                "official head VisionSensor is unavailable"
            ) from exc
        self._sensor = sensor
        return sensor

    def _bind_evaluator(self, evaluator: Any) -> None:
        """Bind the later-created Evaluator without touching render topology."""

        if evaluator is None:
            raise TypeError("evaluator must not be None")
        if self._evaluator is not None and self._evaluator is not evaluator:
            raise PrivateHeadSegmentationSideChannelError(
                "private segmentation side channel belongs to another evaluator"
            )
        try:
            evaluator_sensor = evaluator.robot.sensors[self._head_sensor_name]
        except (AttributeError, KeyError, TypeError) as exc:
            raise PrivateHeadSegmentationSideChannelError(
                "official head VisionSensor is unavailable"
            ) from exc
        if evaluator_sensor is not self._head_sensor():
            raise PrivateHeadSegmentationSideChannelError(
                "preinstalled private side channel head sensor identity changed"
            )
        self._evaluator = evaluator

    def _install(self, *, phase: str) -> None:
        if self._started:
            return
        if self._installation_attempted:
            raise PrivateHeadSegmentationSideChannelError(
                "private segmentation side-channel installation cannot be retried"
            )
        self._installation_attempted = True
        self._installation_phase = phase
        sensor = self._head_sensor()
        resident_before = set(getattr(sensor, "modalities", ()))
        try:
            if resident_before not in {
                frozenset({"rgb"}),
                frozenset({"rgb", "depth_linear"}),
                frozenset({"rgb", "depth_linear", "seg_semantic", "seg_instance"}),
            }:
                raise PrivateHeadSegmentationSideChannelError(
                    "private side channel requires a declared RGB or RGB-D head sensor"
                )
            annotators: dict[str, Any] = {}
            # Both Replicator graph edits occur under one official guard while
            # the Environment wrapper is still being constructed. Captures do
            # not enter this context and never attach or detach anything.
            with self._editing_usd_context_factory():
                for modality in _PRIVATE_MODALITIES:
                    annotator = self._annotator_factory(sensor, modality)
                    if annotator is None:
                        raise PrivateHeadSegmentationSideChannelError(
                            f"private annotator is missing: {modality}"
                        )
                    annotators[modality] = annotator
            resident_after = set(getattr(sensor, "modalities", ()))
            if resident_after != resident_before:
                raise PrivateHeadSegmentationSideChannelError(
                    "private side channel mutated resident camera modalities"
                )
        except BaseException as exc:
            self._last_report = {
                "schema_version": PRIVATE_HEAD_SEGMENTATION_SIDE_CHANNEL_VERSION,
                "status": "installation_failed",
                "installation_phase": phase,
                "retry_allowed": False,
            }
            if isinstance(exc, PrivateHeadSegmentationSideChannelError):
                raise
            raise PrivateHeadSegmentationSideChannelError(
                "private segmentation side-channel initialization failed"
            ) from exc
        self._annotators = annotators
        self._resident_modalities = frozenset(resident_after)
        self._started = True
        self._last_report = {
            "schema_version": PRIVATE_HEAD_SEGMENTATION_SIDE_CHANNEL_VERSION,
            "status": "started",
            "installation_phase": phase,
            "installed_before_evaluator_handles": (
                phase == "environment_wrapper_construction"
            ),
            "installed_after_post_load_reset": (
                phase == "post_load_reset_before_strategy"
            ),
            "resident_modalities": sorted(resident_after),
            "private_annotators": list(_PRIVATE_MODALITIES),
            "private_annotator_attach_count": len(annotators),
            "editing_usd_guarded": (
                phase
                in {
                    "environment_wrapper_construction",
                    "post_load_reset_before_strategy",
                }
            ),
            "persistent_until_kit_exit": True,
            "vision_sensor_modality_mutation": False,
            "camera_parameters_attached": False,
            "evaluator_rgb_written": False,
            "evaluator_segmentation_fields_written": False,
            "late_start_allowed": False,
            "post_attach_evaluator_reset_budget": (
                0 if phase == "post_load_reset_before_strategy" else None
            ),
        }

    def _install_from_environment_wrapper(self, *, token: object) -> None:
        if token is not _ENVIRONMENT_WRAPPER_INSTALL_TOKEN:
            raise PrivateHeadSegmentationSideChannelError(
                "private side channel installation requires wrapper authority"
            )
        self._install(phase="environment_wrapper_construction")

    def _install_after_post_load_reset(self, *, token: object) -> None:
        if token is not _POST_LOAD_INSTALL_TOKEN:
            raise PrivateHeadSegmentationSideChannelError(
                "post-load private side-channel installation lacks authority"
            )
        self._install(phase="post_load_reset_before_strategy")

    def start(self) -> None:
        """Reject production startup after Environment construction.

        Injected annotator factories remain startable for pure unit fixtures;
        the production factory can only be invoked by the Environment-wrapper
        installer above.
        """

        if self._started:
            return
        if self._uses_production_annotator_factory:
            raise PrivateHeadSegmentationSideChannelError(
                "late private side-channel start is forbidden; install it "
                "during Environment-wrapper construction"
            )
        self._install(phase="injected_test_fixture")

    def capture(
        self,
        *,
        simulator_step: int,
        public_head_rgb: Any,
    ) -> PrivateInstanceSegmentationFrame:
        """Read a current private mask without changing render topology."""

        if not self._started:
            raise PrivateHeadSegmentationSideChannelError(
                "private side channel must be installed before Mark"
            )
        if self._evaluator is None:
            raise PrivateHeadSegmentationSideChannelError(
                "private side channel must be bound to Evaluator before Mark"
            )
        if (
            isinstance(simulator_step, bool)
            or not isinstance(simulator_step, int)
            or simulator_step < 0
        ):
            raise ValueError("simulator_step must be a non-negative integer")
        public_rgb = _rgb_array(public_head_rgb)
        public_digest = hashlib.sha256(
            public_rgb.tobytes(order="C")
        ).hexdigest()
        cached = self._cached_frame
        if (
            cached is not None
            and cached.simulator_step == simulator_step
            and cached.source_rgb_sha256 == public_digest
        ):
            self._last_report = {
                **self._last_report,
                "status": "cache_hit",
                "simulator_step": simulator_step,
                "render_passes": 0,
            }
            return cached

        sensor = self._head_sensor()
        resident_before = set(getattr(sensor, "modalities", ()))
        if self._resident_modalities is None or resident_before != set(
            self._resident_modalities
        ):
            raise PrivateHeadSegmentationSideChannelError(
                "resident head sensor changed after side-channel start"
            )
        try:
            for _ in range(self._render_passes):
                self._render()
            try:
                frame_payload = self._frame_reader(sensor, self._annotators)
            except PrivateHeadSegmentationSideChannelError:
                raise
            except Exception as exc:
                raise PrivateHeadSegmentationSideChannelError(
                    "private segmentation side-channel read failed"
                ) from exc
            try:
                if len(frame_payload) == 6:
                    (
                        captured_rgb_value,
                        semantic,
                        instance,
                        registry,
                        part_instance,
                        part_registry,
                    ) = frame_payload
                elif len(frame_payload) == 4:
                    # Backwards-compatible injected test fixtures. Production
                    # always supplies the visual-mesh-level pair as well.
                    (
                        captured_rgb_value,
                        semantic,
                        instance,
                        registry,
                    ) = frame_payload
                    part_instance = None
                    part_registry = {}
                else:
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise PrivateHeadSegmentationSideChannelError(
                    "private frame reader returned an invalid payload"
                ) from exc
            try:
                captured_rgb = _rgb_array(captured_rgb_value)
                if captured_rgb.shape != public_rgb.shape:
                    raise PrivateHeadSegmentationUnavailableError(
                        "private side-channel RGB shape differs from the public camera"
                    )
                # The public RGB supplied to Mark is the immutable
                # declassification boundary. Rendering the attached annotators
                # advances no physics.
                private_raw = {
                    HEAD_RGB_KEY: public_rgb,
                    HEAD_INSTANCE_SEGMENTATION_KEY: _as_numpy(instance),
                }
                frame = capture_private_head_frame(
                    raw=private_raw,
                    public_head_rgb=public_rgb,
                    simulator_step=simulator_step,
                    instance_registry=registry,
                    part_segmentation=(
                        None
                        if part_instance is None
                        else _as_numpy(part_instance)
                    ),
                    part_instance_registry=(
                        None if part_instance is None else part_registry
                    ),
                )
                if _as_numpy(semantic).shape[:2] != captured_rgb.shape[:2]:
                    raise PrivateHeadSegmentationUnavailableError(
                        "private semantic mask shape differs from the public camera"
                    )
            except PrivateHeadSegmentationUnavailableError:
                raise
            except (TypeError, ValueError) as exc:
                raise PrivateHeadSegmentationSideChannelError(
                    "private segmentation frame is invalid"
                ) from exc
        finally:
            evaluator_observation = getattr(self._evaluator, "obs", None)
            if (
                isinstance(evaluator_observation, Mapping)
                and (
                    HEAD_INSTANCE_SEGMENTATION_KEY in evaluator_observation
                    or HEAD_SEMANTIC_SEGMENTATION_KEY in evaluator_observation
                )
            ):
                raise PrivateHeadSegmentationSideChannelError(
                    "private segmentation leaked into evaluator observation"
                )
            resident_after = set(getattr(sensor, "modalities", ()))
            if resident_after != resident_before:
                raise PrivateHeadSegmentationSideChannelError(
                    "private capture mutated resident camera modalities"
                )
        self._cached_frame = frame
        self._last_report = {
            "schema_version": PRIVATE_HEAD_SEGMENTATION_SIDE_CHANNEL_VERSION,
            "status": "captured",
            "installation_phase": self._installation_phase,
            "installed_before_evaluator_handles": (
                self._installation_phase == "environment_wrapper_construction"
            ),
            "installed_after_post_load_reset": (
                self._installation_phase == "post_load_reset_before_strategy"
            ),
            "simulator_step": simulator_step,
            "render_passes": self._render_passes,
            "resident_modalities": sorted(resident_after),
            "private_annotators": list(_PRIVATE_MODALITIES),
            "persistent_until_kit_exit": True,
            "vision_sensor_modality_mutation": False,
            "topology_mutation_during_capture": False,
            "camera_parameters_attached": False,
            "evaluator_rgb_written": False,
            "evaluator_segmentation_fields_written": False,
            "post_attach_evaluator_reset_budget": (
                0
                if self._installation_phase
                == "post_load_reset_before_strategy"
                else None
            ),
        }
        return frame


def install_private_head_segmentation_side_channel(
    *,
    environment: Any,
    head_sensor_name: str,
    annotator_factory: AnnotatorFactory | None = None,
    editing_usd_context_factory: EditingUsdContextFactory | None = None,
    frame_reader: FrameReader | None = None,
    render_callback: Callable[[], None] | None = None,
    render_passes: int = _DEFAULT_RENDER_PASSES,
) -> PrivateHeadSegmentationSideChannel:
    """Install one persistent channel during RGB-only wrapper construction."""

    if environment is None:
        raise TypeError("environment must not be None")
    try:
        sensor = environment.robots[0].sensors[head_sensor_name]
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise PrivateHeadSegmentationSideChannelError(
            "official head VisionSensor is unavailable"
        ) from exc
    current = getattr(environment, _OWNER_CHANNEL_ATTRIBUTE, None)
    if current is not None:
        if not isinstance(current, PrivateHeadSegmentationSideChannel):
            raise PrivateHeadSegmentationSideChannelError(
                "environment private segmentation slot has an incompatible value"
            )
        if current.head_sensor_name != head_sensor_name:
            raise PrivateHeadSegmentationSideChannelError(
                "environment private segmentation side-channel identity changed"
            )
        if current._head_sensor() is not sensor:
            raise PrivateHeadSegmentationSideChannelError(
                "environment private segmentation head sensor identity changed"
            )
        if not current.started:
            raise PrivateHeadSegmentationSideChannelError(
                "environment private segmentation installation is incomplete"
            )
        return current

    current = PrivateHeadSegmentationSideChannel(
        sensor=sensor,
        head_sensor_name=head_sensor_name,
        annotator_factory=annotator_factory,
        editing_usd_context_factory=editing_usd_context_factory,
        frame_reader=frame_reader,
        render_callback=render_callback,
        render_passes=render_passes,
    )
    current._environment_owner = environment
    # Publish the attempt before touching Replicator. A partially failed graph
    # edit poisons this resident session and must never be retried in place.
    setattr(environment, _OWNER_CHANNEL_ATTRIBUTE, current)
    current._install_from_environment_wrapper(
        token=_ENVIRONMENT_WRAPPER_INSTALL_TOKEN
    )
    return current


def get_preinstalled_private_head_segmentation_side_channel(
    *,
    evaluator: Any,
    head_sensor_name: str,
) -> PrivateHeadSegmentationSideChannel:
    """Bind and return the wrapper-installed channel without graph edits."""

    if evaluator is None:
        raise TypeError("evaluator must not be None")
    evaluator_channel = getattr(evaluator, _OWNER_CHANNEL_ATTRIBUTE, None)
    environment = getattr(evaluator, "env", None)
    environment_channel = (
        getattr(environment, _OWNER_CHANNEL_ATTRIBUTE, None)
        if environment is not None
        else None
    )
    if (
        evaluator_channel is not None
        and environment_channel is not None
        and evaluator_channel is not environment_channel
    ):
        raise PrivateHeadSegmentationSideChannelError(
            "evaluator and environment private segmentation slots disagree"
        )
    current = evaluator_channel or environment_channel
    if current is None:
        raise PrivateHeadSegmentationSideChannelError(
            "private segmentation side channel was not installed by the "
            "Environment wrapper"
        )
    if not isinstance(current, PrivateHeadSegmentationSideChannel):
        raise PrivateHeadSegmentationSideChannelError(
            "private segmentation slot has an incompatible value"
        )
    if current.head_sensor_name != head_sensor_name:
        raise PrivateHeadSegmentationSideChannelError(
            "private segmentation side-channel identity changed"
        )
    if current._environment_owner is not environment:
        raise PrivateHeadSegmentationSideChannelError(
            "private segmentation side channel belongs to another environment"
        )
    if not current.started or current._installation_phase not in {
        "environment_wrapper_construction",
        "post_load_reset_before_strategy",
    }:
        raise PrivateHeadSegmentationSideChannelError(
            "private segmentation side channel has no authorized installation"
        )
    current._bind_evaluator(evaluator)
    setattr(evaluator, _OWNER_CHANNEL_ATTRIBUTE, current)
    return current


def install_post_load_private_head_segmentation_side_channel(
    *,
    evaluator: Any,
    head_sensor_name: str,
    annotator_factory: AnnotatorFactory | None = None,
    editing_usd_context_factory: EditingUsdContextFactory | None = None,
    frame_reader: FrameReader | None = None,
    render_callback: Callable[[], None] | None = None,
    render_passes: int = _DEFAULT_RENDER_PASSES,
) -> PrivateHeadSegmentationSideChannel:
    """Install once after the official post-load reset and before strategy.

    This is intentionally a narrow Action23 canary hook.  The caller must
    guarantee that no ``evaluator.reset()`` occurs after installation.
    """

    if evaluator is None:
        raise TypeError("evaluator must not be None")
    environment = getattr(evaluator, "env", None)
    if environment is None:
        raise PrivateHeadSegmentationSideChannelError(
            "evaluator environment is unavailable"
        )
    try:
        sensor = evaluator.robot.sensors[head_sensor_name]
    except (AttributeError, KeyError, TypeError) as exc:
        raise PrivateHeadSegmentationSideChannelError(
            "official head VisionSensor is unavailable"
        ) from exc
    if getattr(environment, _OWNER_CHANNEL_ATTRIBUTE, None) is not None or getattr(
        evaluator, _OWNER_CHANNEL_ATTRIBUTE, None
    ) is not None:
        raise PrivateHeadSegmentationSideChannelError(
            "post-load private side channel must be installed exactly once"
        )

    channel = PrivateHeadSegmentationSideChannel(
        evaluator=evaluator,
        sensor=sensor,
        head_sensor_name=head_sensor_name,
        annotator_factory=annotator_factory,
        editing_usd_context_factory=editing_usd_context_factory,
        frame_reader=frame_reader,
        render_callback=render_callback,
        render_passes=render_passes,
    )
    channel._environment_owner = environment
    # Publish before graph mutation. Any partial installation failure poisons
    # this resident process and must not be retried or reset in place.
    setattr(environment, _OWNER_CHANNEL_ATTRIBUTE, channel)
    setattr(evaluator, _OWNER_CHANNEL_ATTRIBUTE, channel)
    channel._install_after_post_load_reset(token=_POST_LOAD_INSTALL_TOKEN)
    return channel


def get_or_create_private_head_segmentation_side_channel(
    *,
    evaluator: Any,
    head_sensor_name: str,
    start: bool = True,
) -> PrivateHeadSegmentationSideChannel:
    """Compatibility accessor; creation and late startup are disabled."""

    if not isinstance(start, bool):
        raise TypeError("start must be boolean")
    return get_preinstalled_private_head_segmentation_side_channel(
        evaluator=evaluator,
        head_sensor_name=head_sensor_name,
    )


__all__ = [
    "PRIVATE_HEAD_SEGMENTATION_SIDE_CHANNEL_VERSION",
    "PrivateHeadSegmentationSideChannel",
    "PrivateHeadSegmentationSideChannelError",
    "PrivateHeadSegmentationUnavailableError",
    "get_preinstalled_private_head_segmentation_side_channel",
    "get_or_create_private_head_segmentation_side_channel",
    "install_post_load_private_head_segmentation_side_channel",
    "install_private_head_segmentation_side_channel",
]
