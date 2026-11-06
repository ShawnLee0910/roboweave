"""Shared phase contract for marker-bound symbolic carry assertions.

The runner and component builders must use one typed vocabulary.  Accepting
plain strings here would silently restore the split, hand-maintained
allowlists that caused public carry steps to fail only after a grasp.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class MarkerSymbolicCarryContractError(ValueError):
    """Raised when a marker-symbolic carry assertion contract is malformed."""


class MarkerSymbolicCarryPhase(str, Enum):
    """Every phase at which the marker-symbolic hold may be asserted."""

    BEFORE_SETTLE_STEP = "before_settle_step"
    AFTER_SETTLE_STEP = "after_settle_step"
    BEFORE_PRE_ACTION_SYNCHRONIZATION = "before_pre_action_synchronization"
    AFTER_PRE_ACTION_SYNCHRONIZATION = "after_pre_action_synchronization"
    BEFORE_POST_GRASP_CARRY_REBASE = "before_post_grasp_carry_rebase"
    AFTER_POST_GRASP_CARRY_REBASE = "after_post_grasp_carry_rebase"
    BEFORE_PRIVILEGED_STARTER_NAVIGATION = (
        "before_privileged_starter_navigation"
    )
    DURING_PRIVILEGED_STARTER_NAVIGATION = (
        "during_privileged_starter_navigation"
    )
    AFTER_PRIVILEGED_STARTER_NAVIGATION = (
        "after_privileged_starter_navigation"
    )
    BEFORE_PRIVILEGED_SYMBOLIC_NAVIGATION = (
        "before_privileged_symbolic_navigation"
    )
    DURING_PRIVILEGED_SYMBOLIC_NAVIGATION = (
        "during_privileged_symbolic_navigation"
    )
    AFTER_PRIVILEGED_SYMBOLIC_NAVIGATION = (
        "after_privileged_symbolic_navigation"
    )
    BEFORE_RESTRICTED_KINEMATIC_BASE_STEP = (
        "before_restricted_kinematic_base_step"
    )
    DURING_RESTRICTED_KINEMATIC_BASE_STEP = (
        "during_restricted_kinematic_base_step"
    )
    AFTER_RESTRICTED_KINEMATIC_BASE_STEP = (
        "after_restricted_kinematic_base_step"
    )
    BEFORE_MINIMAL_KINEMATIC_ROUTE_REPLAY = (
        "before_minimal_kinematic_route_replay"
    )
    AFTER_MINIMAL_KINEMATIC_ROUTE_REPLAY = (
        "after_minimal_kinematic_route_replay"
    )
    DURING_ACTIONIZED_MINIMAL_KINEMATIC_ROUTE = (
        "during_actionized_minimal_kinematic_route"
    )
    DURING_MINIMAL_LOCAL_SE2_RECOVERY = "during_minimal_local_se2_recovery"


MARKER_SYMBOLIC_CARRY_PHASES = frozenset(MarkerSymbolicCarryPhase)


def require_marker_symbolic_carry_phase(
    phase: Any,
) -> MarkerSymbolicCarryPhase:
    """Return one typed phase or fail closed with diagnosable type detail."""

    if not isinstance(phase, MarkerSymbolicCarryPhase):
        phase_type = f"{type(phase).__module__}.{type(phase).__qualname__}"
        raise MarkerSymbolicCarryContractError(
            "marker symbolic grasp assertion phase is invalid: "
            f"phase={phase!r}, type={phase_type}"
        )
    return phase
