"""Research-only checkpoint frequency configuration before Simulator birth."""

from __future__ import annotations

import math
import os
from typing import Any, Mapping


CHECKPOINT_FREQUENCY_ENV = "BEHAVIOR_HL_SIM_CHECKPOINT_FREQUENCY_HZ"


def configure_preconstruction_checkpoint_frequency(
    *,
    global_macros: Any,
    macros_module: Any,
    environ: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
    values = os.environ if environ is None else environ
    raw = values.get(CHECKPOINT_FREQUENCY_ENV, "30")
    try:
        frequency_hz = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{CHECKPOINT_FREQUENCY_ENV} must be an integer") from exc
    if frequency_hz <= 0:
        raise ValueError(f"{CHECKPOINT_FREQUENCY_ENV} must be positive")
    physics_frequency_hz = int(global_macros.DEFAULT_PHYSICS_FREQ)
    if physics_frequency_hz % frequency_hz != 0:
        raise ValueError(
            f"{CHECKPOINT_FREQUENCY_ENV} must divide physics frequency "
            f"{physics_frequency_hz}"
        )
    original_sim_frequency_hz = int(global_macros.DEFAULT_SIM_STEP_FREQ)
    original_rendering_frequency_hz = int(
        global_macros.DEFAULT_RENDERING_FREQ
    )
    with macros_module.unlocked():
        global_macros.DEFAULT_SIM_STEP_FREQ = frequency_hz
        global_macros.DEFAULT_RENDERING_FREQ = frequency_hz
    return {
        "checkpoint_frequency_hz": frequency_hz,
        "physics_frequency_hz": physics_frequency_hz,
        "physics_substeps_per_checkpoint": (
            physics_frequency_hz // frequency_hz
        ),
        "original_sim_frequency_hz": original_sim_frequency_hz,
        "original_rendering_frequency_hz": original_rendering_frequency_hz,
        "configured_before_simulator_construction": True,
        "runtime_dt_mutation": False,
        "official_challenge_eligible": frequency_hz == 30,
    }
