"""Private Oracle ordering only; public Action admission stays authoritative."""
import math

from behavior_hl.embodied.runtime.plugins.torso import (
    EMPTY_TORSO_GRID, TORSO_PITCH_LEVELS, nearest_torso_grid_cell,
    r1pro_head_optical_pitch_deg,
)


def improving_discrete_look(qpos, pitch_error_deg):
    """Compare adjacent Action endpoints against measured optical pitch.

    The existing optical-pitch proxy predicts angular change, not visibility or
    collision freedom. Actual movement and fresh segmentation remain required.
    """
    qpos = tuple(float(q) for q in qpos)
    if not math.isfinite(pitch_error_deg):
        raise ValueError("non-finite view error")
    height, pitch = nearest_torso_grid_cell(qpos)
    current = r1pro_head_optical_pitch_deg(qpos)
    choices = []
    for action, delta in (("look_up", 1), ("look_down", -1)):
        index = TORSO_PITCH_LEVELS.index(pitch) + delta
        if 0 <= index < len(TORSO_PITCH_LEVELS):
            cell = (height, TORSO_PITCH_LEVELS[index])
            predicted = pitch_error_deg - (r1pro_head_optical_pitch_deg(EMPTY_TORSO_GRID[cell]) - current)
            choices.append({"action": action, "profile": ".".join(cell), "predicted_error_deg": predicted})
    improving = [row for row in choices if abs(row["predicted_error_deg"]) < abs(pitch_error_deg)]
    best = min(improving, key=lambda row: abs(row["predicted_error_deg"]), default=None)
    return (None if best is None else best["action"]), {
        "torso_profile_private": ".".join((height, pitch)),
        "torso_qpos_private": list(qpos),
        "optical_pitch_proxy_deg_private": current,
        "discrete_look_candidates_private": choices,
    }


def rank_observation_stations(rows, *, target_xy, failed_view_xy):
    """Stable maximin bearing diversity; retain every existing legal candidate."""
    if not failed_view_xy:
        return tuple(rows)
    def bearing(xy):
        return math.atan2(xy[1] - target_xy[1], xy[0] - target_xy[0])
    failed = [bearing(xy) for xy in failed_view_xy]
    def diversity(row):
        angle = bearing(row[1])
        return min(abs(math.atan2(math.sin(angle - old), math.cos(angle - old))) for old in failed)
    return tuple(sorted(rows, key=lambda row: -diversity(row)))
