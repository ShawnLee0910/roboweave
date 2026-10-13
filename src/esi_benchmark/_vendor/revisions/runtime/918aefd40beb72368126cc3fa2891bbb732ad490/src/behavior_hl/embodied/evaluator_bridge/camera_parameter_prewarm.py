"""Freeze lazy evaluator camera-parameter annotators before any reset.

The official evaluator reads ``VisionSensor.camera_parameters`` while
preprocessing every reset observation.  The first property access lazily
attaches a ``camera_params`` annotator and performs four renders.  Doing that
between the initial reset and task-instance load leaves the SyntheticData
graph mutable across the post-load reset.  This adapter performs the same
official property access earlier and proves that no camera structure changed.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


EXPECTED_EVAL_CAMERA_ROLES = frozenset(
    {"head", "left_wrist", "right_wrist"}
)
EXPECTED_HEAD_RGB_CAMERA_ROLES = frozenset({"head"})


class CameraParameterPrewarmError(RuntimeError):
    """Raised when camera-parameter graph freezing cannot be proven."""


def _render_product_path(render_product: Any) -> str:
    path = getattr(render_product, "path", None)
    if path is None:
        path = getattr(render_product, "prim_path", None)
    if path is None:
        raise CameraParameterPrewarmError(
            "camera render product has no stable path"
        )
    return str(path)


def _finite_view_transform(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.size != 16:
        raise CameraParameterPrewarmError(
            "cameraViewTransform must contain exactly 16 values"
        )
    try:
        finite = np.isfinite(array.astype(np.float64, copy=False))
    except (TypeError, ValueError) as exc:
        raise CameraParameterPrewarmError(
            "cameraViewTransform must be numeric"
        ) from exc
    if not bool(np.all(finite)):
        raise CameraParameterPrewarmError(
            "cameraViewTransform contains non-finite values"
        )
    return array.reshape(4, 4)


def prewarm_evaluator_camera_parameters(
    evaluator: Any,
    *,
    camera_roles: frozenset[str] | set[str] | tuple[str, ...] = (
        "head",
        "left_wrist",
        "right_wrist",
    ),
) -> dict[str, Any]:
    """Materialize all official eval ``camera_params`` annotators once.

    This function deliberately uses the public ``camera_parameters`` property
    rather than calling ``add_modality`` or private Replicator APIs.  It does
    not change camera modalities, resolution, render products, observation
    space, physics state, or task state.
    """

    requested_roles = frozenset(camera_roles)
    if requested_roles not in {
        EXPECTED_EVAL_CAMERA_ROLES,
        EXPECTED_HEAD_RGB_CAMERA_ROLES,
    }:
        raise CameraParameterPrewarmError(
            "camera_roles must be the full R1Pro set or the head-only set"
        )
    role_mapping = getattr(evaluator, "robot_camera_names", None)
    if not isinstance(role_mapping, Mapping):
        raise CameraParameterPrewarmError(
            "evaluator robot_camera_names is unavailable"
        )
    if set(role_mapping) != requested_roles:
        raise CameraParameterPrewarmError(
            "evaluator camera roles do not match the R1Pro contract"
        )
    robot = getattr(evaluator, "robot", None)
    sensors = getattr(robot, "sensors", None)
    if not isinstance(sensors, Mapping):
        raise CameraParameterPrewarmError("robot sensors are unavailable")

    report: dict[str, Any] = {
        "schema_version": "behavior2026.camera-parameter-prewarm/v1",
        "camera_roles": sorted(requested_roles),
        "cameras": {},
    }
    for role, flattened_name in role_mapping.items():
        if not isinstance(flattened_name, str) or "::" not in flattened_name:
            raise CameraParameterPrewarmError(
                f"invalid flattened camera name for role {role}"
            )
        sensor_name = flattened_name.split("::", 1)[1]
        sensor = sensors.get(sensor_name)
        if sensor is None:
            raise CameraParameterPrewarmError(
                f"missing camera sensor for role {role}"
            )
        annotators_before = getattr(sensor, "_annotators", None)
        if not isinstance(annotators_before, Mapping):
            raise CameraParameterPrewarmError(
                f"camera annotators are unavailable for role {role}"
            )
        existing_annotator = annotators_before.get("camera_params")
        modalities_before = tuple(sorted(sensor.modalities))
        resolution_before = (
            int(sensor.image_height),
            int(sensor.image_width),
        )
        render_product_before = sensor.render_product
        if render_product_before is None:
            raise CameraParameterPrewarmError(
                f"render product is unavailable for role {role}"
            )
        render_product_path = _render_product_path(render_product_before)

        # Exactly one property read per camera.  The official property owns
        # lazy annotator creation and its required four render updates.
        parameters = sensor.camera_parameters
        if not isinstance(parameters, Mapping):
            raise CameraParameterPrewarmError(
                f"camera parameters are not a mapping for role {role}"
            )
        if "cameraViewTransform" not in parameters:
            raise CameraParameterPrewarmError(
                f"cameraViewTransform is missing for role {role}"
            )
        view_transform = _finite_view_transform(
            parameters["cameraViewTransform"]
        )

        annotators_after = getattr(sensor, "_annotators", None)
        if not isinstance(annotators_after, Mapping):
            raise CameraParameterPrewarmError(
                f"camera annotators disappeared for role {role}"
            )
        parameter_annotator = annotators_after.get("camera_params")
        if parameter_annotator is None:
            raise CameraParameterPrewarmError(
                f"camera_params annotator was not initialized for role {role}"
            )
        if (
            existing_annotator is not None
            and parameter_annotator is not existing_annotator
        ):
            raise CameraParameterPrewarmError(
                f"camera_params annotator identity changed for role {role}"
            )
        if sensor.render_product is not render_product_before:
            raise CameraParameterPrewarmError(
                f"render product identity changed for role {role}"
            )
        if _render_product_path(sensor.render_product) != render_product_path:
            raise CameraParameterPrewarmError(
                f"render product path changed for role {role}"
            )
        if tuple(sorted(sensor.modalities)) != modalities_before:
            raise CameraParameterPrewarmError(
                f"camera modalities changed for role {role}"
            )
        if (
            int(sensor.image_height),
            int(sensor.image_width),
        ) != resolution_before:
            raise CameraParameterPrewarmError(
                f"camera resolution changed for role {role}"
            )

        report["cameras"][role] = {
            "sensor_name": sensor_name,
            "already_initialized": existing_annotator is not None,
            "camera_params_annotator_initialized": True,
            "camera_params_annotator_identity": id(parameter_annotator),
            "render_product_identity": id(render_product_before),
            "render_product_path": render_product_path,
            "modalities": list(modalities_before),
            "resolution": list(resolution_before),
            "camera_view_transform_shape": list(view_transform.shape),
            "camera_view_transform_finite": True,
        }
    return report


__all__ = [
    "CameraParameterPrewarmError",
    "EXPECTED_EVAL_CAMERA_ROLES",
    "EXPECTED_HEAD_RGB_CAMERA_ROLES",
    "prewarm_evaluator_camera_parameters",
]
