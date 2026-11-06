"""Public observations, optional Marker depth, and RGB composition."""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
THREE_VIEW_CAMERAS = frozenset({"head", "left_wrist", "right_wrist"})
FOUR_VIEW_CAMERAS = THREE_VIEW_CAMERAS | {"overhead"}
THREE_VIEW_COMPOSITE_FILENAME = "three_view_composite.png"
RGBD_PROTOCOL_VERSION = "rgbd_v1"
MARKER_DEPTH_FIELDS = frozenset(
    {"marker", "camera", "approx_depth_m", "depth_valid_ratio"}
)


def approximate_depth_at_anchor(
    depth: Any,
    anchor_uv: tuple[int, int],
    *,
    radius: int = 2,
    invalid_values: Sequence[float] = (),
) -> tuple[float | None, float]:
    """Return local median head depth and the finite-positive sample ratio."""

    array = np.asarray(depth, dtype=np.float64).squeeze()
    if array.ndim != 2 or radius < 0:
        raise ValueError("depth must be HxW and radius must be non-negative")
    x, y = anchor_uv
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (x, y)):
        raise ValueError("anchor_uv must contain integer pixel coordinates")
    if not (0 <= x < array.shape[1] and 0 <= y < array.shape[0]):
        return None, 0.0
    window = array[
        max(0, y - radius) : min(array.shape[0], y + radius + 1),
        max(0, x - radius) : min(array.shape[1], x + radius + 1),
    ]
    valid = np.isfinite(window) & (window > 0)
    for raw in invalid_values:
        value = float(raw)
        if math.isfinite(value):
            valid &= window != value
    ratio = float(valid.mean()) if window.size else 0.0
    ratio = round(ratio, 2)
    if not valid.any():
        return None, 0.0
    return round(float(np.median(window[valid])) * 10.0) / 10.0, ratio


def marker_depth_records(
    *,
    head_camera: Mapping[str, Any],
    anchors: Mapping[str, tuple[int, int]],
    radius: int = 2,
) -> tuple[dict[str, Any], ...]:
    """Build allowlisted records without object identity or world geometry."""

    if "depth_linear" not in head_camera:
        raise ValueError("head camera has no aligned depth_linear")
    records: list[dict[str, Any]] = []
    for marker, anchor in sorted(
        anchors.items(), key=lambda item: int(item[0][1:])
    ):
        if not marker.startswith("C") or not marker[1:].isdigit():
            raise ValueError("marker identity is invalid")
        approximate, valid_ratio = approximate_depth_at_anchor(
            head_camera["depth_linear"], anchor, radius=radius
        )
        records.append(
            {
                "marker": marker,
                "camera": "head",
                "approx_depth_m": approximate,
                "depth_valid_ratio": valid_ratio,
            }
        )
    return tuple(records)


def validate_public_marker_depth(value: Any) -> tuple[dict[str, Any], ...]:
    """Fail closed on extra fields, paths, reachability, or private geometry."""

    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("marker_depth must be a nonempty list or tuple")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != MARKER_DEPTH_FIELDS:
            raise ValueError("marker depth fields changed")
        marker, camera = item["marker"], item["camera"]
        depth = item["approx_depth_m"]
        valid_ratio = item["depth_valid_ratio"]
        if (
            not isinstance(marker, str)
            or not marker.startswith("C")
            or not marker[1:].isdigit()
            or not 1 <= int(marker[1:]) <= 999
            or marker in seen
            or camera != "head"
            or isinstance(valid_ratio, bool)
            or not isinstance(valid_ratio, (int, float))
            or not math.isfinite(float(valid_ratio))
            or not 0.0 <= float(valid_ratio) <= 1.0
            or (depth is None) != (float(valid_ratio) == 0.0)
            or (
                depth is not None
                and (
                    isinstance(depth, bool)
                    or not isinstance(depth, (int, float))
                    or not math.isfinite(float(depth))
                    or float(depth) <= 0
                )
            )
        ):
            raise ValueError("marker depth record is invalid")
        seen.add(marker)
        records.append(dict(item))
    return tuple(records)


def model_visible_marker_depth_text(records: Sequence[Mapping[str, Any]]) -> str:
    """Render only the four public fields for a Marker observation."""

    checked = validate_public_marker_depth(tuple(records))
    return (
        "Marker head-camera depth (approximate optical-axis depth at each "
        "C-label anchor; not arm reachability). `depth_valid_ratio` is the "
        "fraction of nearby finite positive depth samples:\n"
        + json.dumps(
            list(checked),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


@dataclass(frozen=True)
class PublicImage:
    """One public camera PNG."""

    camera: str
    path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.camera, str) or not self.camera.strip():
            raise ValueError("camera must be a nonempty string")
        object.__setattr__(self, "path", Path(self.path).resolve())

    def read_png(self) -> bytes:
        try:
            value = self.path.read_bytes()
        except OSError as exc:
            raise ValueError("public image is unreadable") from exc
        if not value.startswith(PNG_SIGNATURE):
            raise ValueError("public image is not a PNG")
        return value

    def artifact_record(self) -> dict[str, str]:
        return {
            "camera": self.camera,
            "path": str(self.path),
        }


@dataclass(frozen=True)
class PublicObservation:
    """One chronological observation bundle, typically the three RGB views."""

    observation_id: str
    images: tuple[PublicImage, ...]
    marker_depth: tuple[Mapping[str, Any], ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id, str) or not self.observation_id:
            raise ValueError("observation_id must be a nonempty string")
        if (
            not isinstance(self.images, tuple)
            or not 1 <= len(self.images) <= 4
            or any(not isinstance(image, PublicImage) for image in self.images)
        ):
            raise ValueError("observation must contain one to four public images")
        cameras = [image.camera for image in self.images]
        if len(cameras) != len(set(cameras)):
            raise ValueError("observation cameras must be unique")
        if self.marker_depth is not None:
            object.__setattr__(
                self,
                "marker_depth",
                validate_public_marker_depth(self.marker_depth),
            )

    def artifact_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "observation_id": self.observation_id,
            "images": [image.artifact_record() for image in self.images],
        }
        if self.marker_depth is not None:
            record["marker_depth"] = [dict(item) for item in self.marker_depth]
        return record

    def read_model_composite_png(
        self,
        *,
        maximum_size: tuple[int, int] | None = None,
    ) -> bytes:
        """Return the one model-visible PNG for this observation bundle.

        The production three-view benchmark layout is fixed: left wrist in
        the upper-left quadrant, right wrist in the lower-left quadrant, and
        the head view filling the double-width right panel.  Raw per-camera
        PNGs remain separate private/audit artifacts; the provider receives
        only this one composite image URL per chronological observation.
        """

        if maximum_size is not None and (
            not isinstance(maximum_size, tuple)
            or len(maximum_size) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in maximum_size
            )
        ):
            raise ValueError("model composite maximum_size is invalid")
        if len(self.images) == 1:
            raw = self.images[0].read_png()
        else:
            by_camera = {image.camera: image for image in self.images}
            if set(by_camera) not in {
                THREE_VIEW_CAMERAS,
                FOUR_VIEW_CAMERAS,
            }:
                raise ValueError(
                    "model observation must be one image or an exact supported view set"
                )
            decoded: dict[str, Image.Image] = {}
            composite: Image.Image | None = None
            try:
                for camera, public_image in by_camera.items():
                    with Image.open(io.BytesIO(public_image.read_png())) as image:
                        image.load()
                        decoded[camera] = image.convert("RGB")
                left = decoded["left_wrist"]
                right = decoded["right_wrist"]
                head = decoded["head"]
                if (
                    left.width != left.height
                    or right.width != right.height
                    or head.width != head.height
                    or left.size != right.size
                ):
                    raise ValueError("three-view cameras must be square with equal wrists")
                wrist_edge = left.width
                if wrist_edge <= 0:
                    raise ValueError("three-view camera size is invalid")
                head_edge = wrist_edge * 2
                if head.size != (head_edge, head_edge):
                    head = head.resize(
                        (head_edge, head_edge),
                        resample=Image.Resampling.LANCZOS,
                    )
                if set(by_camera) == FOUR_VIEW_CAMERAS:
                    overhead = decoded["overhead"]
                    if overhead.width != overhead.height or head_edge % 3:
                        raise ValueError(
                            "four-view cameras require square overhead and divisible head size"
                        )
                    tile_edge = head_edge // 3
                    composite = Image.new(
                        "RGB", (head_edge + tile_edge, head_edge)
                    )
                    for index, camera in enumerate(
                        ("left_wrist", "right_wrist", "overhead")
                    ):
                        tile = decoded[camera].resize(
                            (tile_edge, tile_edge),
                            resample=Image.Resampling.LANCZOS,
                        )
                        try:
                            composite.paste(tile, (0, index * tile_edge))
                        finally:
                            tile.close()
                    composite.paste(head, (tile_edge, 0))
                else:
                    composite = Image.new(
                        "RGB", (wrist_edge * 3, wrist_edge * 2)
                    )
                    composite.paste(left, (0, 0))
                    composite.paste(right, (0, wrist_edge))
                    composite.paste(head, (wrist_edge, 0))
                encoded = io.BytesIO()
                composite.save(encoded, format="PNG")
                raw = encoded.getvalue()
            finally:
                if composite is not None:
                    composite.close()
                for image in decoded.values():
                    image.close()
        if maximum_size is None:
            return raw
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            if image.width <= maximum_size[0] and image.height <= maximum_size[1]:
                return raw
            compact = image.convert("RGB")
        try:
            compact.thumbnail(maximum_size, resample=Image.Resampling.LANCZOS)
            encoded = io.BytesIO()
            compact.save(encoded, format="PNG")
            return encoded.getvalue()
        finally:
            compact.close()


def materialize_observation(
    value: Mapping[str, Any],
    *,
    output_dir: Path,
    scene_run_dir: Path,
    link_or_copy: Callable[[Path, Path], None],
    hash_file: Callable[[Path], str],
    error_type: type[Exception],
) -> PublicObservation:
    """Validate and copy one resident observation into the run artifacts."""

    if (
        not isinstance(value, Mapping)
        or set(value)
        not in (
            {"observation_id", "kind", "images", "marker_targets"},
            {
                "observation_id",
                "kind",
                "images",
                "marker_targets",
                "marker_depth",
            },
        )
        or value.get("kind") not in {"physical", "marker", "transient"}
        or not isinstance(value.get("images"), list)
        or not 1 <= len(value["images"]) <= 4
    ):
        raise error_type("scene observation contract changed")
    observation_id = value["observation_id"]
    if "marker_depth" in value and value.get("kind") != "marker":
        raise error_type("only Marker observations may expose depth")
    if (
        not isinstance(observation_id, str)
        or not observation_id
        or not re.fullmatch(r"[A-Za-z0-9_-]+", observation_id)
    ):
        raise error_type("scene observation id is invalid")
    target_dir = Path(output_dir) / "observations" / observation_id
    target_dir.mkdir(parents=True, exist_ok=False)
    public_images: list[PublicImage] = []
    seen: set[str] = set()
    for item in value["images"]:
        if not isinstance(item, Mapping) or set(item) != {
            "camera",
            "path",
            "sha256",
        }:
            raise error_type("scene image contract changed")
        camera = item["camera"]
        source = Path(item["path"]).resolve()
        digest = item["sha256"]
        if (
            camera not in FOUR_VIEW_CAMERAS
            or camera in seen
            or not isinstance(digest, str)
            or len(digest) != 64
            or not source.is_file()
            or not source.is_relative_to(Path(scene_run_dir).resolve())
        ):
            raise error_type("scene image identity is invalid")
        if hash_file(source) != digest:
            raise error_type("scene image hash changed")
        destination = target_dir / f"{camera}.png"
        link_or_copy(source, destination)
        public_images.append(PublicImage(camera=camera, path=destination))
        seen.add(camera)
    observation = PublicObservation(
        observation_id=observation_id,
        images=tuple(public_images),
        marker_depth=(
            None
            if "marker_depth" not in value
            else tuple(value["marker_depth"])
        ),
    )
    composite = observation.read_model_composite_png()
    (target_dir / THREE_VIEW_COMPOSITE_FILENAME).write_bytes(composite)
    return observation


def observation_from_artifact(
    value: Mapping[str, Any],
    *,
    output_dir: Path,
    hash_file: Callable[[Path], str],
    error_type: type[Exception],
) -> PublicObservation:
    """Restore one materialized observation from a checkpoint artifact."""

    if (
        not isinstance(value, Mapping)
        or set(value)
        not in (
            {"observation_id", "images"},
            {"observation_id", "images", "marker_depth"},
        )
        or not isinstance(value.get("images"), list)
    ):
        raise error_type("repeat observation artifact changed")
    root = Path(output_dir).resolve()
    images: list[PublicImage] = []
    for item in value["images"]:
        if (
            not isinstance(item, Mapping)
            or not {"camera", "path"}.issubset(item)
            or not set(item).issubset({"camera", "path", "sha256"})
        ):
            raise error_type("repeat image artifact changed")
        path = Path(item["path"]).resolve()
        if not path.is_relative_to(root):
            raise error_type("repeat image escaped its artifact root")
        if "sha256" in item and hash_file(path) != item["sha256"]:
            raise error_type("legacy repeat image hash changed")
        image = PublicImage(camera=item["camera"], path=path)
        image.read_png()
        images.append(image)
    return PublicObservation(
        observation_id=value["observation_id"],
        images=tuple(images),
        marker_depth=(
            None
            if "marker_depth" not in value
            else tuple(value["marker_depth"])
        ),
    )
