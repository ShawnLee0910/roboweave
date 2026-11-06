"""Small, current-frame-only spatial marker proposals.

This module deliberately does *not* infer 3-D planes, normals, world poses, or
hidden geometry.  It turns one already-selected object's visible 2-D instance
mask and official image-space bounding box into three simple marker families:

* ``next_to`` exposes up to eight screen-relative positions around the visible
  reference object;
* ``touching`` exposes four screen-relative directions and anchors each one on
  an extreme visible mask pixel.
* ``shelf_compartment`` finds visible openings bounded by horizontal and
  vertical shelf dividers and places one point in each cubby.

The proposals are visual choices, not claims that an OmniGibson predicate is
already true.  The private executor remains responsible for binding and
checking the selected proposal after the model chooses it.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Sequence, TypeAlias, Union

import numpy as np


SIMPLE_SPATIAL_MARKER_VERSION = "behavior2026.simple-spatial-markers/v11"
TOUCHING_SCREEN_DIRECTIONS = (
    "screen_left",
    "screen_right",
    "screen_near",
    "screen_far",
)
NEXT_TO_SCREEN_POSITIONS = (
    "screen_left",
    "screen_far_left",
    "screen_far",
    "screen_far_right",
    "screen_right",
    "screen_near_right",
    "screen_near",
    "screen_near_left",
)
VERTICAL_SHELF_MAX_LEVELS = 8
VERTICAL_SHELF_CENTRAL_WIDTH_FRACTION = 0.60
VERTICAL_SHELF_MIN_OPENING_HEIGHT_FRACTION = 0.018
VERTICAL_SHELF_MAX_ABS_SLOPE = 0.35
VERTICAL_SHELF_SLOPE_CANDIDATES = 61
VERTICAL_SHELF_GT_VALIDATION_COLUMNS = (
    0.14,
    0.26,
    0.38,
    0.50,
    0.62,
    0.74,
    0.86,
)
VERTICAL_SHELF_GT_MIN_VALID_COLUMNS = 4
SHELF_MIN_OPENING_WIDTH_FRACTION = 0.04
SHELF_MAX_COLUMNS = 8

JsonScalar: TypeAlias = Union[None, bool, int, float, str]
JsonValue: TypeAlias = Union[JsonScalar, list["JsonValue"], dict[str, "JsonValue"]]


@dataclass(frozen=True, slots=True)
class TouchingDirectionMarker:
    """One coarse, screen-relative touching direction.

    Image ``v`` grows downwards, so ``screen_near`` means the bottom-most
    visible edge and ``screen_far`` means the top-most visible edge.  These are
    intentionally coarse UI choices, not estimated surface normals.
    """

    marker: str
    direction: str
    anchor_uv: tuple[int, int]
    screen_delta_uv: tuple[int, int]

    def as_private_dict(self) -> dict[str, JsonValue]:
        return {
            "marker": self.marker,
            "kind": "touching_direction",
            "direction": self.direction,
            "anchor_uv": list(self.anchor_uv),
            "screen_delta_uv": list(self.screen_delta_uv),
        }


@dataclass(frozen=True, slots=True)
class NextToPositionMarker:
    """One coarse image-space position around a visible reference object."""

    marker: str
    position: str
    anchor_uv: tuple[int, int]

    def as_private_dict(self) -> dict[str, JsonValue]:
        return {
            "marker": self.marker,
            "kind": "next_to_position",
            "position": self.position,
            "anchor_uv": list(self.anchor_uv),
        }


@dataclass(frozen=True, slots=True)
class ShelfCompartmentMarker:
    """One visible cubby in a currently visible shelf box.

    This is intentionally a 2-D choice.  It does not claim that a hidden
    physical shelf plane or an official Inside region has been reconstructed.
    """

    marker: str
    row_index: int
    row_count: int
    column_index: int
    column_count: int
    anchor_uv: tuple[int, int]
    cell_bounds_xyxy: tuple[int, int, int, int]
    visible_pixels: int
    private_mask_validated_columns: int = 0
    private_mask_required_columns: int = VERTICAL_SHELF_GT_MIN_VALID_COLUMNS

    def as_private_dict(self) -> dict[str, JsonValue]:
        return {
            "marker": self.marker,
            "kind": "shelf_compartment",
            "row_index": self.row_index,
            "row_count": self.row_count,
            "column_index": self.column_index,
            "column_count": self.column_count,
            "anchor_uv": list(self.anchor_uv),
            "cell_bounds_xyxy": list(self.cell_bounds_xyxy),
            "visible_pixels": self.visible_pixels,
            "private_mask_validated_columns": (
                self.private_mask_validated_columns
            ),
            "private_mask_required_columns": (
                self.private_mask_required_columns
            ),
        }


SpatialMarkerCandidate: TypeAlias = Union[
    NextToPositionMarker,
    TouchingDirectionMarker,
    ShelfCompartmentMarker,
]


@dataclass(frozen=True, slots=True, eq=False)
class SpatialMarkerBatch:
    """Immutable result of one current-frame marker proposal call."""

    kind: str
    image_shape_hw: tuple[int, int]
    clamped_bbox_xyxy: tuple[int, int, int, int] | None
    target_mask: np.ndarray = field(repr=False)
    visible_pixels: int
    minimum_visible_pixels: int
    candidates: tuple[SpatialMarkerCandidate, ...]

    def __post_init__(self) -> None:
        mask = np.ascontiguousarray(np.asarray(self.target_mask, dtype=np.bool_))
        if mask.ndim != 2 or tuple(mask.shape) != self.image_shape_hw:
            raise ValueError("target_mask must match image_shape_hw")
        frozen = np.array(mask, dtype=np.bool_, copy=True, order="C")
        frozen.setflags(write=False)
        object.__setattr__(self, "target_mask", frozen)
        object.__setattr__(self, "candidates", tuple(self.candidates))

    def as_private_dict(self) -> dict[str, JsonValue]:
        """Return a plain JSON-safe report without numpy values or private GT."""

        bbox: JsonValue
        if self.clamped_bbox_xyxy is None:
            bbox = None
        else:
            bbox = list(self.clamped_bbox_xyxy)
        return {
            "version": SIMPLE_SPATIAL_MARKER_VERSION,
            "kind": self.kind,
            "image_shape_hw": list(self.image_shape_hw),
            "clamped_bbox_xyxy": bbox,
            "visible_pixels": self.visible_pixels,
            "minimum_visible_pixels": self.minimum_visible_pixels,
            "candidates": [candidate.as_private_dict() for candidate in self.candidates],
        }


def generate_next_to_position_markers(
    *,
    target_mask: np.ndarray,
    official_bbox_xyxy: Sequence[int],
    minimum_visible_pixels: int = 8,
) -> SpatialMarkerBatch:
    """Generate up to eight coarse positions around one visible reference.

    These are image-space choices only; they reveal no world pose.  Positions
    that collapse onto the same pixel after image-edge clamping are removed.
    """

    mask = _validated_mask(target_mask)
    minimum_visible_pixels = _positive_integer(
        minimum_visible_pixels, name="minimum_visible_pixels"
    )
    bbox = _clamp_bbox(
        official_bbox_xyxy,
        width=mask.shape[1],
        height=mask.shape[0],
    )
    visible = _mask_inside_bbox(mask, bbox)
    visible_pixels = int(np.count_nonzero(visible))
    candidates: list[NextToPositionMarker] = []

    if bbox is not None and visible_pixels >= minimum_visible_pixels:
        x0, y0, x1, y1 = bbox
        centre_x = (x0 + x1) / 2.0
        centre_y = (y0 + y1) / 2.0
        # Eight choices around a tiny reference need enough screen-space for
        # their C-labels.  The selector remains only a coarse direction; the
        # executor computes the actual semantic placement independently.
        # A fixed 64-pixel ring is intentionally simple and remains readable
        # at the public 720p head-camera resolution.
        offset = 64
        anchors = {
            "screen_left": (x0 - offset, centre_y),
            "screen_far_left": (x0 - offset, y0 - offset),
            "screen_far": (centre_x, y0 - offset),
            "screen_far_right": (x1 + offset, y0 - offset),
            "screen_right": (x1 + offset, centre_y),
            "screen_near_right": (x1 + offset, y1 + offset),
            "screen_near": (centre_x, y1 + offset),
            "screen_near_left": (x0 - offset, y1 + offset),
        }
        seen: set[tuple[int, int]] = set()
        for index, position in enumerate(NEXT_TO_SCREEN_POSITIONS, start=1):
            raw_u, raw_v = anchors[position]
            anchor = (
                max(0, min(mask.shape[1] - 1, int(round(raw_u)))),
                max(0, min(mask.shape[0] - 1, int(round(raw_v)))),
            )
            if anchor in seen:
                continue
            seen.add(anchor)
            candidates.append(
                NextToPositionMarker(
                    marker=f"N{index}",
                    position=position,
                    anchor_uv=anchor,
                )
            )

    return SpatialMarkerBatch(
        kind="next_to_positions",
        image_shape_hw=(int(mask.shape[0]), int(mask.shape[1])),
        clamped_bbox_xyxy=bbox,
        target_mask=visible,
        visible_pixels=visible_pixels,
        minimum_visible_pixels=minimum_visible_pixels,
        candidates=tuple(candidates),
    )


def generate_touching_direction_markers(
    *,
    target_mask: np.ndarray,
    official_bbox_xyxy: Sequence[int],
    minimum_visible_pixels: int = 8,
) -> SpatialMarkerBatch:
    """Generate four coarse direction markers on visible mask boundaries."""

    mask = _validated_mask(target_mask)
    minimum_visible_pixels = _positive_integer(
        minimum_visible_pixels, name="minimum_visible_pixels"
    )
    bbox = _clamp_bbox(official_bbox_xyxy, width=mask.shape[1], height=mask.shape[0])
    visible = _mask_inside_bbox(mask, bbox)
    visible_pixels = int(np.count_nonzero(visible))
    candidates: list[TouchingDirectionMarker] = []

    if bbox is not None and visible_pixels >= minimum_visible_pixels:
        coordinates_yx = np.argwhere(visible)
        centre_x = (bbox[0] + bbox[2]) / 2.0
        centre_y = (bbox[1] + bbox[3]) / 2.0
        direction_specs = (
            ("screen_left", (-1, 0)),
            ("screen_right", (1, 0)),
            ("screen_near", (0, 1)),
            ("screen_far", (0, -1)),
        )
        for index, (direction, delta_uv) in enumerate(direction_specs, start=1):
            anchor = _extreme_boundary_anchor(
                coordinates_yx,
                direction=direction,
                centre_x=centre_x,
                centre_y=centre_y,
            )
            candidates.append(
                TouchingDirectionMarker(
                    marker=f"T{index}",
                    direction=direction,
                    anchor_uv=anchor,
                    screen_delta_uv=delta_uv,
                )
            )

    return SpatialMarkerBatch(
        kind="touching_directions",
        image_shape_hw=(int(mask.shape[0]), int(mask.shape[1])),
        clamped_bbox_xyxy=bbox,
        target_mask=visible,
        visible_pixels=visible_pixels,
        minimum_visible_pixels=minimum_visible_pixels,
        candidates=tuple(candidates),
    )


def generate_shelf_compartment_markers(
    *,
    target_mask: np.ndarray,
    public_head_rgb: np.ndarray,
    official_bbox_xyxy: Sequence[int],
    minimum_visible_pixels: int = 8,
    minimum_level_pixels: int = 1,
) -> SpatialMarkerBatch:
    """Put one point in every visible row-by-column shelf opening.

    The detector is deliberately current-frame-only.  It estimates the common
    image-space slope of the visible shelf boards, deskews the mask and RGB
    into shelf coordinates, and then finds openings bounded by object-mask
    board evidence above and below.  This handles camera roll and moderate
    perspective without using an asset name, a fixed level count, or hidden
    geometry.  A dark gap below the bottom board is rejected because it has no
    second shelf board underneath it.

    When the current RGB view does not expose a privately validated opening,
    the function returns no compartment marker.  A poor view must therefore be
    changed by the caller instead of inventing a shelf compartment.
    """

    mask = _validated_mask(target_mask)
    minimum_visible_pixels = _positive_integer(
        minimum_visible_pixels,
        name="minimum_visible_pixels",
    )
    minimum_level_pixels = _positive_integer(
        minimum_level_pixels,
        name="minimum_level_pixels",
    )
    rgb = _validated_rgb(public_head_rgb, image_shape_hw=mask.shape)
    bbox = _clamp_bbox(
        official_bbox_xyxy,
        width=mask.shape[1],
        height=mask.shape[0],
    )
    visible = _mask_inside_bbox(mask, bbox)
    visible_pixels = int(np.count_nonzero(visible))
    candidates: list[ShelfCompartmentMarker] = []

    if bbox is not None and visible_pixels >= minimum_visible_pixels:
        x0, y0, x1, y1 = bbox
        bbox_height = y1 - y0 + 1
        bbox_width = x1 - x0 + 1
        anchor_x = int(round((x0 + x1) / 2.0))
        strip_width = max(
            3,
            int(round(bbox_width * VERTICAL_SHELF_CENTRAL_WIDTH_FRACTION)),
        )
        strip_x0 = max(x0, anchor_x - strip_width // 2)
        strip_x1 = min(x1, strip_x0 + strip_width - 1)
        luminance = (
            0.2126 * rgb[..., 0].astype(np.float64)
            + 0.7152 * rgb[..., 1].astype(np.float64)
            + 0.0722 * rgb[..., 2].astype(np.float64)
        )
        estimated_slope, mask_profile, row_profile = (
            _estimate_shelf_slope_and_profiles(
                target_mask=visible,
                luminance=luminance,
                bbox_xyxy=(x0, y0, x1, y1),
                strip_x0=strip_x0,
                strip_x1=strip_x1,
                anchor_x=anchor_x,
            )
        )
        smoothing_width = max(3, int(round(bbox_height * 0.01)))
        if smoothing_width % 2 == 0:
            smoothing_width += 1
        smoothed = _moving_average(row_profile, smoothing_width)
        dark_quantile = float(np.quantile(smoothed, 0.25))
        bright_quantile = float(np.quantile(smoothed, 0.75))
        dark_threshold = dark_quantile + 0.45 * (
            bright_quantile - dark_quantile
        )
        minimum_opening_height = max(
            5,
            int(round(
                bbox_height * VERTICAL_SHELF_MIN_OPENING_HEIGHT_FRACTION
            )),
        )
        luminance_openings = _contiguous_true_runs(
            smoothed <= dark_threshold,
            minimum_length=minimum_opening_height,
        )
        luminance_openings = tuple(
            (start, end)
            for start, end in luminance_openings
            if start > 0 and end < bbox_height - 1
        )
        mask_smoothed = _moving_average(mask_profile, smoothing_width)
        mask_low = float(np.quantile(mask_smoothed, 0.20))
        mask_high = float(np.quantile(mask_smoothed, 0.90))
        board_openings: tuple[tuple[int, int], ...] = ()
        if mask_high - mask_low >= 0.08:
            board_threshold = mask_low + 0.55 * (mask_high - mask_low)
            minimum_board_height = max(2, int(round(bbox_height * 0.006)))
            board_runs = _contiguous_true_runs(
                mask_smoothed >= board_threshold,
                minimum_length=minimum_board_height,
            )
            board_runs = _merge_nearby_runs(
                board_runs,
                maximum_gap=max(1, int(round(bbox_height * 0.008))),
            )
            bounded_gaps: list[tuple[int, int]] = []
            for upper, lower in zip(board_runs, board_runs[1:]):
                gap_start = upper[1] + 1
                gap_end = lower[0] - 1
                if gap_end - gap_start + 1 >= minimum_opening_height:
                    bounded_gaps.append((gap_start, gap_end))
            board_openings = tuple(bounded_gaps)

        # Object-mask board pairs are stronger than luminance alone because
        # they cannot mistake the floor below a shelf for another compartment.
        proposed_openings = (
            board_openings if board_openings else luminance_openings
        )
        validated_openings = tuple(
            (run, validated_columns)
            for run in proposed_openings
            if (
                validated_columns := _validate_shelf_opening_with_private_mask(
                    target_mask=visible,
                    luminance=luminance,
                    bbox_xyxy=(x0, y0, x1, y1),
                    anchor_x=anchor_x,
                    estimated_slope=estimated_slope,
                    canonical_band=run,
                    dark_threshold=dark_threshold,
                )
            ) >= VERTICAL_SHELF_GT_MIN_VALID_COLUMNS
        )
        if len(validated_openings) > VERTICAL_SHELF_MAX_LEVELS:
            strongest = sorted(
                validated_openings,
                key=lambda item: (
                    float(np.mean(smoothed[item[0][0] : item[0][1] + 1])),
                    item[0][0],
                ),
            )[:VERTICAL_SHELF_MAX_LEVELS]
            validated_openings = tuple(sorted(strongest))

        bands = validated_openings
        columns = _shelf_column_openings(
            target_mask=visible,
            luminance=luminance,
            bbox_xyxy=(x0, y0, x1, y1),
            anchor_x=anchor_x,
            estimated_slope=estimated_slope,
            canonical_bands=tuple(run for run, _ in bands),
        )
        if not columns:
            columns = ((x0, x1),)

        row_count = len(bands)
        column_count = len(columns)
        for row_index, (
            (canonical_y0, canonical_y1),
            validated_columns,
        ) in enumerate(bands):
            band_bbox = _shelf_band_source_bbox(
                bbox_xyxy=(x0, y0, x1, y1),
                estimated_slope=estimated_slope,
                anchor_x=anchor_x,
                canonical_band=(canonical_y0, canonical_y1),
            )
            for column_index, (cell_x0, cell_x1) in enumerate(columns):
                cell_pixels = int(
                    (canonical_y1 - canonical_y0 + 1)
                    * (cell_x1 - cell_x0 + 1)
                )
                if cell_pixels < minimum_level_pixels:
                    continue
                anchored = _shelf_compartment_anchor(
                    target_mask=visible,
                    luminance=luminance,
                    bbox_xyxy=(x0, y0, x1, y1),
                    anchor_x=anchor_x,
                    estimated_slope=estimated_slope,
                    canonical_band=(canonical_y0, canonical_y1),
                    column_bounds=(cell_x0, cell_x1),
                    dark_threshold=dark_threshold,
                )
                if anchored is None:
                    continue
                anchor_uv, evidence_pixels = anchored
                candidates.append(
                    ShelfCompartmentMarker(
                        marker=f"S{len(candidates) + 1}",
                        row_index=row_index,
                        row_count=row_count,
                        column_index=column_index,
                        column_count=column_count,
                        anchor_uv=anchor_uv,
                        cell_bounds_xyxy=(
                            cell_x0,
                            band_bbox[1],
                            cell_x1,
                            band_bbox[3],
                        ),
                        visible_pixels=max(evidence_pixels, cell_pixels),
                        private_mask_validated_columns=validated_columns,
                    )
                )

    return SpatialMarkerBatch(
        kind="shelf_compartments",
        image_shape_hw=(int(mask.shape[0]), int(mask.shape[1])),
        clamped_bbox_xyxy=bbox,
        target_mask=visible,
        visible_pixels=visible_pixels,
        minimum_visible_pixels=minimum_visible_pixels,
        candidates=tuple(candidates),
    )


def _shelf_column_openings(
    *,
    target_mask: np.ndarray,
    luminance: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    anchor_x: int,
    estimated_slope: float,
    canonical_bands: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Find common left-to-right openings across the visible shelf rows."""

    if not canonical_bands:
        return ()
    x0, y0, x1, y1 = bbox_xyxy
    width = x1 - x0 + 1
    profile = np.full(width, 255.0, dtype=np.float64)
    for offset, source_x in enumerate(range(x0, x1 + 1)):
        values: list[float] = []
        for band_y0, band_y1 in canonical_bands:
            margin = max(1.0, 0.18 * (band_y1 - band_y0 + 1))
            canonical_ys = np.linspace(
                band_y0 + margin,
                band_y1 - margin,
                5,
            )
            source_ys = np.rint(
                y0 + canonical_ys
                + estimated_slope * (source_x - anchor_x)
            ).astype(np.int64)
            valid = source_ys[(source_ys >= y0) & (source_ys <= y1)]
            values.extend(float(luminance[y, source_x]) for y in valid)
        if values:
            # One occupied cubby must not hide a divider shared by the other rows.
            profile[offset] = float(np.quantile(values, 0.30))

    smoothing_width = max(3, int(round(width * 0.01)))
    if smoothing_width % 2 == 0:
        smoothing_width += 1
    smoothed = _moving_average(profile, smoothing_width)
    dark = float(np.quantile(smoothed, 0.25))
    bright = float(np.quantile(smoothed, 0.90))
    if bright - dark < 8.0:
        return ()
    threshold = dark + 0.45 * (bright - dark)
    runs = _contiguous_true_runs(
        smoothed <= threshold,
        minimum_length=max(
            5,
            int(round(width * SHELF_MIN_OPENING_WIDTH_FRACTION)),
        ),
    )
    boundary_radius = max(3, int(round(width * 0.025)))
    validated: list[tuple[int, int]] = []
    for relative_start, relative_end in runs:
        if relative_start == 0 or relative_end == width - 1:
            continue
        start, end = x0 + relative_start, x0 + relative_end
        boundary_ok = True
        for boundary_x in (start, end):
            samples: list[bool] = []
            for band_y0, band_y1 in canonical_bands:
                for canonical_y in np.linspace(band_y0, band_y1, 5):
                    source_y = int(round(
                        y0 + canonical_y
                        + estimated_slope * (boundary_x - anchor_x)
                    ))
                    if y0 <= source_y <= y1:
                        left = max(x0, boundary_x - boundary_radius)
                        right = min(x1, boundary_x + boundary_radius)
                        samples.extend(
                            bool(value)
                            for value in target_mask[source_y, left : right + 1]
                        )
            if not samples or float(np.mean(samples)) < 0.12:
                boundary_ok = False
                break
        if boundary_ok:
            validated.append((start, end))
    if len(validated) > SHELF_MAX_COLUMNS:
        validated = sorted(
            sorted(validated, key=lambda run: run[1] - run[0], reverse=True)[
                :SHELF_MAX_COLUMNS
            ]
        )
    return tuple(validated)


def _validate_shelf_opening_with_private_mask(
    *,
    target_mask: np.ndarray,
    luminance: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    anchor_x: int,
    estimated_slope: float,
    canonical_band: tuple[int, int],
    dark_threshold: float,
) -> int:
    """Validate one RGB proposal with private current-frame object pixels.

    RGB proposes an opening.  The private instance mask is used only to
    confirm that the proposed gap is bracketed by visible pixels belonging to
    the selected shelf object at multiple horizontal positions.  It never
    invents a level that was absent from the current RGB frame.
    """

    x0, y0, x1, y1 = bbox_xyxy
    bbox_width = x1 - x0 + 1
    bbox_height = y1 - y0 + 1
    band_y0, band_y1 = canonical_band
    boundary_radius = max(2, int(round(bbox_height * 0.018)))
    column_radius = max(1, int(round(bbox_width * 0.015)))
    validated = 0
    for fraction in VERTICAL_SHELF_GT_VALIDATION_COLUMNS:
        source_x = int(round(x0 + fraction * max(0, bbox_width - 1)))
        source_x0 = max(x0, source_x - column_radius)
        source_x1 = min(x1, source_x + column_radius)
        source_top = int(round(
            y0 + band_y0 + estimated_slope * (source_x - anchor_x)
        ))
        source_bottom = int(round(
            y0 + band_y1 + estimated_slope * (source_x - anchor_x)
        ))
        upper_y0 = max(y0, source_top - boundary_radius)
        upper_y1 = min(y1, source_top + boundary_radius)
        lower_y0 = max(y0, source_bottom - boundary_radius)
        lower_y1 = min(y1, source_bottom + boundary_radius)
        upper_patch = target_mask[
            upper_y0 : upper_y1 + 1,
            source_x0 : source_x1 + 1,
        ]
        lower_patch = target_mask[
            lower_y0 : lower_y1 + 1,
            source_x0 : source_x1 + 1,
        ]
        upper_object = float(np.mean(upper_patch)) >= 0.12
        lower_object = float(np.mean(lower_patch)) >= 0.12
        centre_y = max(
            y0,
            min(y1, int(round((source_top + source_bottom) / 2.0))),
        )
        centre_mask = target_mask[
            centre_y,
            source_x0 : source_x1 + 1,
        ]
        centre_luminance = luminance[
            centre_y,
            source_x0 : source_x1 + 1,
        ]
        visible_opening = bool(
            np.any(~centre_mask)
            or np.mean(centre_luminance) <= dark_threshold
        )
        if (
            upper_object
            and lower_object
            and visible_opening
        ):
            validated += 1
    return validated


def _estimate_shelf_slope_and_profiles(
    *,
    target_mask: np.ndarray,
    luminance: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    strip_x0: int,
    strip_x1: int,
    anchor_x: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate a common board slope using only current-frame evidence."""

    slopes = np.linspace(
        -VERTICAL_SHELF_MAX_ABS_SLOPE,
        VERTICAL_SHELF_MAX_ABS_SLOPE,
        VERTICAL_SHELF_SLOPE_CANDIDATES,
        dtype=np.float64,
    )
    scored: list[tuple[float, float, np.ndarray, np.ndarray]] = []
    for slope in slopes:
        mask_profile, luminance_profile = _deskewed_row_profiles(
            target_mask=target_mask,
            luminance=luminance,
            bbox_xyxy=bbox_xyxy,
            strip_x0=strip_x0,
            strip_x1=strip_x1,
            anchor_x=anchor_x,
            slope=float(slope),
        )
        mask_contrast = float(
            np.quantile(mask_profile, 0.90)
            - np.quantile(mask_profile, 0.20)
        )
        luminance_smoothed = _moving_average(luminance_profile, 3)
        edge_strength = float(
            np.quantile(np.abs(np.diff(luminance_smoothed)), 0.90)
        ) / 255.0
        score = 2.0 * mask_contrast + edge_strength
        # Prefer the smaller correction only when evidence scores tie.
        scored.append((score, -abs(float(slope)), mask_profile, luminance_profile))
    selected_index = max(
        range(len(scored)),
        key=lambda index: (scored[index][0], scored[index][1]),
    )
    _, _, mask_profile, luminance_profile = scored[selected_index]
    return float(slopes[selected_index]), mask_profile, luminance_profile


def _deskewed_row_profiles(
    *,
    target_mask: np.ndarray,
    luminance: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    strip_x0: int,
    strip_x1: int,
    anchor_x: int,
    slope: float,
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = bbox_xyxy
    del x0, x1
    height = y1 - y0 + 1
    source_y = np.arange(y0, y1 + 1, dtype=np.int64)[:, None]
    source_x = np.arange(strip_x0, strip_x1 + 1, dtype=np.int64)[None, :]
    canonical_y = np.rint(
        (source_y - y0) - slope * (source_x - anchor_x)
    ).astype(np.int64)
    valid = (canonical_y >= 0) & (canonical_y < height)
    repeated_y = np.broadcast_to(source_y, canonical_y.shape)
    repeated_x = np.broadcast_to(source_x, canonical_y.shape)
    bins = canonical_y[valid]
    counts = np.bincount(bins, minlength=height).astype(np.float64)
    mask_sums = np.bincount(
        bins,
        weights=target_mask[repeated_y[valid], repeated_x[valid]].astype(np.float64),
        minlength=height,
    )
    luminance_sums = np.bincount(
        bins,
        weights=luminance[repeated_y[valid], repeated_x[valid]],
        minlength=height,
    )
    safe_counts = np.maximum(counts, 1.0)
    return mask_sums / safe_counts, luminance_sums / safe_counts


def _merge_nearby_runs(
    runs: Sequence[tuple[int, int]],
    *,
    maximum_gap: int,
) -> tuple[tuple[int, int], ...]:
    if not runs:
        return ()
    merged: list[tuple[int, int]] = [tuple(runs[0])]
    for start, end in runs[1:]:
        previous_start, previous_end = merged[-1]
        if start - previous_end - 1 <= maximum_gap:
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    return tuple(merged)


def _shelf_compartment_anchor(
    *,
    target_mask: np.ndarray,
    luminance: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    anchor_x: int,
    estimated_slope: float,
    canonical_band: tuple[int, int],
    column_bounds: tuple[int, int],
    dark_threshold: float,
) -> tuple[tuple[int, int], int] | None:
    """Choose a current-frame opening pixel inside one detected cubby."""

    _, y0, _, y1 = bbox_xyxy
    cell_x0, cell_x1 = column_bounds
    source_y = np.arange(y0, y1 + 1, dtype=np.int64)[:, None]
    source_x = np.arange(cell_x0, cell_x1 + 1, dtype=np.int64)[None, :]
    canonical_y = (
        (source_y - y0) - estimated_slope * (source_x - anchor_x)
    )
    band_y0, band_y1 = canonical_band
    in_band = (canonical_y >= band_y0) & (canonical_y <= band_y1)
    repeated_y = np.broadcast_to(source_y, canonical_y.shape)
    repeated_x = np.broadcast_to(source_x, canonical_y.shape)
    open_background = in_band & ~target_mask[repeated_y, repeated_x]
    evidence = in_band & (
        luminance[repeated_y, repeated_x] <= dark_threshold
    )
    if np.count_nonzero(evidence) < 3:
        evidence = open_background
    if np.count_nonzero(evidence) < 3:
        return None
    ys = repeated_y[evidence]
    xs = repeated_x[evidence]
    desired_x = int(round((cell_x0 + cell_x1) / 2.0))
    desired_canonical_y = (band_y0 + band_y1) / 2.0
    desired_y = int(round(
        y0 + desired_canonical_y
        + estimated_slope * (desired_x - anchor_x)
    ))
    distance = np.square(xs - desired_x) + np.square(ys - desired_y)
    order = np.lexsort((xs, ys, distance))
    chosen = int(order[0])
    return (int(xs[chosen]), int(ys[chosen])), int(len(xs))


def _shelf_band_source_bbox(
    *,
    bbox_xyxy: tuple[int, int, int, int],
    estimated_slope: float,
    anchor_x: int,
    canonical_band: tuple[int, int],
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox_xyxy
    band_y0, band_y1 = canonical_band
    transformed_y = [
        y0 + canonical_y + estimated_slope * (source_x - anchor_x)
        for canonical_y in (band_y0, band_y1)
        for source_x in (x0, x1)
    ]
    return (
        x0,
        max(y0, int(np.floor(min(transformed_y)))),
        x1,
        min(y1, int(np.ceil(max(transformed_y)))),
    )


def _validated_rgb(
    raw_rgb: np.ndarray,
    *,
    image_shape_hw: Sequence[int],
) -> np.ndarray:
    rgb = np.asarray(raw_rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("public_head_rgb must have shape HxWxC with C >= 3")
    if tuple(rgb.shape[:2]) != tuple(image_shape_hw):
        raise ValueError("public_head_rgb must match target_mask image shape")
    if rgb.dtype.kind not in {"b", "i", "u", "f"}:
        raise TypeError("public_head_rgb must have numeric dtype")
    if rgb.dtype.kind == "f" and not np.isfinite(rgb[..., :3]).all():
        raise ValueError("public_head_rgb must contain only finite values")
    return np.ascontiguousarray(rgb[..., :3])


def _moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return np.asarray(values, dtype=np.float64)
    padding = width // 2
    padded = np.pad(
        np.asarray(values, dtype=np.float64),
        (padding, padding),
        mode="edge",
    )
    kernel = np.full(width, 1.0 / width, dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid")


def _contiguous_true_runs(
    values: np.ndarray,
    *,
    minimum_length: int,
) -> tuple[tuple[int, int], ...]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(np.asarray(values, dtype=np.bool_)):
        if bool(value) and start is None:
            start = index
        if start is not None and (not bool(value) or index == len(values) - 1):
            end = index if bool(value) and index == len(values) - 1 else index - 1
            if end - start + 1 >= minimum_length:
                runs.append((start, end))
            start = None
    return tuple(runs)


def _validated_mask(raw_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(raw_mask)
    if mask.ndim != 2:
        raise ValueError("target_mask must be a non-empty 2-D array")
    if mask.shape[0] <= 0 or mask.shape[1] <= 0:
        raise ValueError("target_mask must be a non-empty 2-D array")
    if mask.dtype.kind not in {"b", "i", "u"}:
        raise TypeError("target_mask must have boolean or integer dtype")
    return np.ascontiguousarray(mask != 0, dtype=np.bool_)


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return int(result)


def _clamp_bbox(
    raw_bbox: Sequence[int],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if isinstance(raw_bbox, (str, bytes)) or len(raw_bbox) != 4:
        raise ValueError("official_bbox_xyxy must contain exactly four integers")
    coordinates: list[int] = []
    for value in raw_bbox:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("official_bbox_xyxy coordinates must be integers")
        try:
            coordinate = operator.index(value)
        except TypeError as exc:
            raise TypeError("official_bbox_xyxy coordinates must be integers") from exc
        coordinates.append(int(coordinate))
    x0, y0, x1, y1 = coordinates
    if x0 > x1 or y0 > y1:
        raise ValueError("official_bbox_xyxy must satisfy x0 <= x1 and y0 <= y1")
    if x1 < 0 or y1 < 0 or x0 >= width or y0 >= height:
        return None
    return (
        max(0, x0),
        max(0, y0),
        min(width - 1, x1),
        min(height - 1, y1),
    )


def _mask_inside_bbox(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
) -> np.ndarray:
    visible = np.zeros(mask.shape, dtype=np.bool_)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        visible[y0 : y1 + 1, x0 : x1 + 1] = mask[y0 : y1 + 1, x0 : x1 + 1]
    return visible


def _extreme_boundary_anchor(
    coordinates_yx: np.ndarray,
    *,
    direction: str,
    centre_x: float,
    centre_y: float,
) -> tuple[int, int]:
    ys = coordinates_yx[:, 0]
    xs = coordinates_yx[:, 1]
    if direction == "screen_left":
        extreme = int(xs.min())
        indices = np.flatnonzero(xs == extreme)
        selected = min(indices, key=lambda index: (abs(float(ys[index]) - centre_y), int(ys[index])))
    elif direction == "screen_right":
        extreme = int(xs.max())
        indices = np.flatnonzero(xs == extreme)
        selected = min(indices, key=lambda index: (abs(float(ys[index]) - centre_y), int(ys[index])))
    elif direction == "screen_near":
        extreme = int(ys.max())
        indices = np.flatnonzero(ys == extreme)
        selected = min(indices, key=lambda index: (abs(float(xs[index]) - centre_x), int(xs[index])))
    elif direction == "screen_far":
        extreme = int(ys.min())
        indices = np.flatnonzero(ys == extreme)
        selected = min(indices, key=lambda index: (abs(float(xs[index]) - centre_x), int(xs[index])))
    else:  # Defensive: callers use the fixed constant above.
        raise ValueError(f"unknown touching direction: {direction!r}")
    return int(xs[selected]), int(ys[selected])


__all__ = [
    "NEXT_TO_SCREEN_POSITIONS",
    "SIMPLE_SPATIAL_MARKER_VERSION",
    "TOUCHING_SCREEN_DIRECTIONS",
    "VERTICAL_SHELF_MAX_LEVELS",
    "NextToPositionMarker",
    "SpatialMarkerBatch",
    "TouchingDirectionMarker",
    "ShelfCompartmentMarker",
    "generate_next_to_position_markers",
    "generate_touching_direction_markers",
    "generate_shelf_compartment_markers",
]
