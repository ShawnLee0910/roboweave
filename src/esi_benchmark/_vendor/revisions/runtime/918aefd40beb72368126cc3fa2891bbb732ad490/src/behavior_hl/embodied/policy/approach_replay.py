"""Offline replay of saved approach actions through the current protocol.

The replay consumes historical public actions and public base velocity samples
stored inside the evaluator trace.  It never imports or starts OmniGibson.
Robot world poses in the trace are intentionally ignored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from behavior_hl.embodied.policy.approach import (
    APPROACH_BASE_ACTIONS,
    ApproachLedger,
    ApproachLimits,
)
from behavior_hl.embodied.policy.approach_execution import (
    BaseMacroConfig,
    classify_macro_displacement,
)
from behavior_hl.embodied.policy.feedback import (
    PublicActionResult,
    PublicStatus,
    movement_result,
)
from behavior_hl.embodied.policy.profiles import (
    build_picking_up_toys_registry,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def replay_saved_approach(
    *,
    action_records: Iterable[dict[str, Any]],
    trace_records: Iterable[dict[str, Any]],
    limits: ApproachLimits | None = None,
    base_config: BaseMacroConfig | None = None,
    command_frequency_hz: float = 30.0,
) -> dict[str, Any]:
    """Replay one saved trajectory without depth, model, or simulator calls."""

    if (
        isinstance(command_frequency_hz, bool)
        or not isinstance(command_frequency_hz, (int, float))
        or not np.isfinite(command_frequency_hz)
        or command_frequency_hz <= 0
    ):
        raise ValueError("command_frequency_hz must be finite and positive")
    actions = list(action_records)
    trace = list(trace_records)
    registry = build_picking_up_toys_registry()
    # This utility replays historical v2 trajectories whose contract exposed
    # executor-directed reposition cooldowns.  New live visual-self-assessment
    # episodes use the v3 default and do not invoke this legacy interception.
    replay_limits = limits or ApproachLimits(
        enforce_executor_directed_reposition=True
    )
    ledger = ApproachLedger(replay_limits)
    movement_config = base_config or BaseMacroConfig()
    replay_records: list[dict[str, Any]] = []
    original_grasp_queries = 0
    skipped_depth_reads = 0
    typed_terminal: str | None = None

    for expected_sequence, original in enumerate(actions):
        if original.get("sequence") != expected_sequence:
            raise ValueError("saved action sequence is not contiguous")
        if ledger.grasp_query_budget_exhausted:
            typed_terminal = "G4A_REACH_QUERY_BUDGET_EXHAUSTED"
            break

        ledger.note_simulator_steps(1)
        ledger.note_model_call()
        envelope = registry.parse_payload(original["action"])
        if envelope.action == "grasp_at":
            original_grasp_queries += 1
        queries_before = ledger.grasp_queries
        immediate = ledger.begin_action(envelope)
        if immediate is not None:
            skipped_depth_reads += 1
            item = ledger.history[-1]
            replay_records.append(
                {
                    "original_sequence": expected_sequence,
                    "action": envelope.to_dict(),
                    "result": item.result,
                    "actual_reach_query": False,
                    "depth_read": False,
                    "cooldown_intercepted": True,
                }
            )
            continue

        result = _replay_result(
            sequence=expected_sequence,
            action=envelope.action,
            original_result=original["result"],
            trace=trace,
            base_config=movement_config,
            command_frequency_hz=float(command_frequency_hz),
        )
        item = ledger.finish_action(result)
        actual_query = ledger.grasp_queries > queries_before
        replay_records.append(
            {
                "original_sequence": expected_sequence,
                "action": envelope.to_dict(),
                "result": item.result,
                "actual_reach_query": actual_query,
                "depth_read": actual_query,
                "cooldown_intercepted": False,
            }
        )
        if ledger.grasp_query_budget_exhausted:
            typed_terminal = "G4A_REACH_QUERY_BUDGET_EXHAUSTED"
            break

    return {
        "schema_version": "behavior-hl-approach-offline-replay-v1",
        "simulator_started": False,
        "model_called": False,
        "world_pose_used": False,
        "movement_odometry_source": "public_base_qvel_integral",
        "original_action_count": len(actions),
        "replayed_action_count": len(replay_records),
        "original_grasp_queries": original_grasp_queries,
        "replayed_grasp_queries": ledger.grasp_queries,
        "cooldown_grasps_intercepted": skipped_depth_reads,
        "depth_reads_skipped": skipped_depth_reads,
        "typed_terminal": typed_terminal,
        "ledger": ledger.public_summary(),
        "records": replay_records,
    }


def _replay_result(
    *,
    sequence: int,
    action: str,
    original_result: dict[str, Any],
    trace: list[dict[str, Any]],
    base_config: BaseMacroConfig,
    command_frequency_hz: float,
) -> PublicActionResult:
    if action not in APPROACH_BASE_ACTIONS:
        return PublicActionResult(
            status=PublicStatus(original_result["status"]),
            simulator_steps=int(original_result["simulator_steps"]),
            hand=original_result.get("hand"),
        )

    interval = _trace_interval(trace, sequence)
    active_segment = f"g4a.approach.{action}"
    active = [
        record
        for record in interval
        if record.get("segment") == active_segment
    ]
    integrated = np.zeros(3, dtype=np.float64)
    for record in active:
        integrated += (
            np.asarray(
                record["public_state"]["base_qvel"],
                dtype=np.float64,
            )
            / command_frequency_hz
        )
    contact = any(bool(record.get("base_contact")) for record in active)
    initial_clearance = float(
        interval[0]["public_state"]["head_forward_clearance_m"]
    )
    hard_stop = (
        action == "move_forward"
        and initial_clearance
        <= base_config.forward_hard_stop_clearance_m
    )
    status = classify_macro_displacement(
        action,
        forward_m=float(integrated[0]),
        left_m=float(integrated[1]),
        yaw_deg=float(np.degrees(integrated[2])),
        config=base_config,
        contact_detected=contact,
        clearance_hard_stop=hard_stop,
    )
    return movement_result(
        status=status,
        simulator_steps=int(original_result["simulator_steps"]),
        forward_m=float(integrated[0]),
        left_m=float(integrated[1]),
        yaw_deg=float(np.degrees(integrated[2])),
    )


def _trace_interval(
    trace: list[dict[str, Any]],
    sequence: int,
) -> list[dict[str, Any]]:
    start_segment = f"g4a.approach.snapshot_{sequence + 1:03d}"
    end_segment = f"g4a.approach.snapshot_{sequence + 2:03d}"
    start = next(
        (
            index
            for index, record in enumerate(trace)
            if record.get("segment") == start_segment
        ),
        None,
    )
    if start is None:
        raise ValueError(f"missing trace boundary {start_segment}")
    end = next(
        (
            index
            for index, record in enumerate(trace[start + 1 :], start + 1)
            if record.get("segment") == end_segment
        ),
        len(trace),
    )
    return trace[start:end]
