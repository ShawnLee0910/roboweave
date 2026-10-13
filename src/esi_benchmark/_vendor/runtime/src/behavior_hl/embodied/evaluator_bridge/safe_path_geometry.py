"""Task-agnostic, simulator-free helpers for safe grid path simplification.

The official floor map can return an eight-connected cell path.  Converting
every diagonal edge into alternating cardinal actions creates the visibly
curved ``forward, left, forward, left`` staircase seen in older rollouts.
This module instead joins the farthest later waypoint whose complete
supercover remains open.  Consequently it reduces action boundaries without
allowing a straight-line shortcut through a wall or a blocked corner.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from behavior_hl.embodied.evaluator_bridge.interaction_station_geometry import (
    Cell,
    supercover_line_cells,
)


def grid_segment_is_open(
    open_grid: np.ndarray,
    start: Cell,
    end: Cell,
) -> bool:
    """Return whether every cell touched by ``start -> end`` is open."""

    if open_grid.ndim != 2:
        raise ValueError("open_grid must be 2-D")
    height, width = open_grid.shape
    for row, col in supercover_line_cells(start, end):
        if not (0 <= row < height and 0 <= col < width):
            return False
        if not bool(open_grid[row, col]):
            return False
    return True


def greedy_farthest_visible_path_simplification(
    path: Sequence[Cell],
    open_grid: np.ndarray,
) -> tuple[Cell, ...]:
    """Simplify an open grid path using safe farthest-visible shortcuts.

    Starting at each retained waypoint, the algorithm selects the farthest
    later input waypoint connected by an entirely open supercover segment.
    This is deterministic, preserves both endpoints, and never crosses a
    blocked cell.  The input path itself must also be safe; a blocked cell or
    an unsafe diagonal edge is rejected rather than hidden by simplification.
    """

    if open_grid.ndim != 2:
        raise ValueError("open_grid must be 2-D")
    if not path:
        return ()

    normalized: list[Cell] = []
    for raw_cell in path:
        cell = (int(raw_cell[0]), int(raw_cell[1]))
        if not normalized or cell != normalized[-1]:
            normalized.append(cell)

    height, width = open_grid.shape
    for cell in normalized:
        if not (0 <= cell[0] < height and 0 <= cell[1] < width):
            raise ValueError(f"path cell lies outside open_grid: {cell}")
        if not bool(open_grid[cell]):
            raise ValueError(f"path contains blocked cell: {cell}")
    for start, end in zip(normalized, normalized[1:]):
        if not grid_segment_is_open(open_grid, start, end):
            raise ValueError(
                f"input path edge crosses a blocked cell: {start} -> {end}"
            )

    if len(normalized) <= 2:
        return tuple(normalized)

    simplified = [normalized[0]]
    anchor_index = 0
    final_index = len(normalized) - 1
    while anchor_index < final_index:
        next_index = anchor_index + 1
        for candidate_index in range(final_index, anchor_index, -1):
            if grid_segment_is_open(
                open_grid,
                normalized[anchor_index],
                normalized[candidate_index],
            ):
                next_index = candidate_index
                break
        simplified.append(normalized[next_index])
        anchor_index = next_index

    return tuple(simplified)

