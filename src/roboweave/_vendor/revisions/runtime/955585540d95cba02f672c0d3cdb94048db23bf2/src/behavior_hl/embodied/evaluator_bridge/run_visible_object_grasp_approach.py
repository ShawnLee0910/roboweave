"""Observation-only Qwen loop for category-level official physical grasp."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from behavior_hl.embodied.evaluator_bridge.r1pro_observation_only_grasp_lift import (
    ObservationOnlyGraspLiftError,
    PublicApproachSnapshot,
    SharedControlApproachMacroExecutor,
)
from behavior_hl.embodied.policy.approach import (
    APPROACH_BASE_ACTIONS,
    APPROACH_LOOK_ACTIONS,
    ApproachBudgetExceeded,
    ApproachLedger,
    ApproachLimits,
    ApproachStopped,
)
from behavior_hl.embodied.policy.feedback import (
    PublicActionResult,
    PublicStatus,
)
from behavior_hl.embodied.policy.profiles import build_picking_up_toys_registry
from behavior_hl.embodied.policy.protocol import (
    ActionEnvelope,
    VisibleObjectGraspArgs,
)


@dataclass(frozen=True, slots=True)
class VisibleObjectGraspExecution:
    """Public executor outcome; private target identity is deliberately absent."""

    status: PublicStatus
    simulator_steps: int

    def __post_init__(self) -> None:
        if self.status not in {
            PublicStatus.GRASP_LIKELY,
            PublicStatus.GRASP_UNCERTAIN,
        }:
            raise ValueError("visible-object execution has invalid status")
        if (
            isinstance(self.simulator_steps, bool)
            or not isinstance(self.simulator_steps, int)
            or self.simulator_steps < 0
        ):
            raise ValueError("simulator_steps must be non-negative")


@dataclass(frozen=True, slots=True)
class VisibleObjectGraspApproachResult:
    """Terminal public result after one visually assessed successful grasp."""

    final_action: ActionEnvelope
    simulator_steps: int
    model_calls: int
    grasp_attempts: int
    public_actions: tuple[Mapping[str, Any], ...]
    ledger: ApproachLedger = field(repr=False)


def run_visible_object_grasp_approach(
    *,
    run_dir: Path,
    capture_public_snapshot: Callable[[], PublicApproachSnapshot],
    macro_executor: SharedControlApproachMacroExecutor,
    execute_grasp: Callable[
        [PublicApproachSnapshot, ActionEnvelope], VisibleObjectGraspExecution
    ],
    limits: ApproachLimits,
    tokenplan_wall_timeout_seconds: int,
    persist_public_result: Callable[[Any, str], None] | None = None,
    tokenplan_call: Callable[..., tuple[Any, Mapping[str, Any], str]],
    hand: str = "right",
    ledger: ApproachLedger | None = None,
) -> VisibleObjectGraspApproachResult:
    """Let Qwen navigate/look/retry grasp using images, never text failures."""

    for callback, name in (
        (capture_public_snapshot, "capture_public_snapshot"),
        (execute_grasp, "execute_grasp"),
        (tokenplan_call, "tokenplan_call"),
    ):
        if not callable(callback):
            raise TypeError(f"{name} must be callable")
    if not isinstance(macro_executor, SharedControlApproachMacroExecutor):
        raise TypeError("macro_executor has the wrong type")
    if not isinstance(limits, ApproachLimits):
        raise TypeError("limits has the wrong type")

    registry = build_picking_up_toys_registry(
        calibrated_hands=(hand,),
        grasp_mode="visible_object_official_macro",
    )
    active_ledger = ledger or ApproachLedger(limits)
    if not isinstance(active_ledger, ApproachLedger):
        raise TypeError("ledger must be an ApproachLedger")
    if active_ledger.limits != limits:
        raise ValueError("provided ledger limits differ from active limits")
    history_snapshot_paths: list[Path] = []
    public_actions: list[Mapping[str, Any]] = []

    def remember(path: Path) -> None:
        history_snapshot_paths.append(Path(path))
        del history_snapshot_paths[:-5]

    def persist(item: Any, digest: str) -> None:
        public_actions.append(
            {
                **item.to_dict(),
                "source_head_rgb_sha256": digest,
            }
        )
        if persist_public_result is not None:
            persist_public_result(item, digest)

    while True:
        if active_ledger.grasp_query_budget_exhausted:
            raise ApproachBudgetExceeded(
                "visible-object grasp-attempt budget exhausted"
            )
        frame = capture_public_snapshot()
        if not isinstance(frame, PublicApproachSnapshot):
            raise ObservationOnlyGraspLiftError(
                "capture callback returned an invalid public snapshot"
            )
        active_ledger.note_simulator_steps(frame.simulator_steps)
        envelope = None
        frame_hash = ""
        while envelope is None:
            envelope, _, frame_hash = tokenplan_call(
                run_dir=Path(run_dir),
                snapshot_path=frame.snapshot_path,
                history_snapshot_paths=tuple(history_snapshot_paths),
                registry=registry,
                ledger=active_ledger,
                wall_timeout_seconds=tokenplan_wall_timeout_seconds,
            )
        if frame_hash != frame.source_head_rgb_sha256:
            raise ObservationOnlyGraspLiftError(
                "TokenPlan decision frame differs from frozen public target"
            )
        immediate = active_ledger.begin_action(envelope)
        if immediate is not None:
            persist(active_ledger.history[-1], frame_hash)
            remember(frame.snapshot_path)
            continue
        if envelope.action in APPROACH_BASE_ACTIONS:
            item = active_ledger.finish_action(
                macro_executor.execute_base(envelope.action)
            )
            persist(item, frame_hash)
            remember(frame.snapshot_path)
            continue
        if envelope.action in APPROACH_LOOK_ACTIONS:
            item = active_ledger.finish_action(
                macro_executor.execute_look(envelope.action)
            )
            persist(item, frame_hash)
            remember(frame.snapshot_path)
            continue
        if envelope.action == "stop":
            item = active_ledger.finish_action(
                PublicActionResult(PublicStatus.COMPLETED, 0)
            )
            persist(item, frame_hash)
            raise ApproachStopped("Qwen stopped before a confirmed grasp")
        if envelope.action != "grasp" or not isinstance(
            envelope.args,
            VisibleObjectGraspArgs,
        ):
            raise ObservationOnlyGraspLiftError(
                "visible-object approach dispatched an unknown action"
            )
        execution = execute_grasp(frame, envelope)
        if not isinstance(execution, VisibleObjectGraspExecution):
            raise ObservationOnlyGraspLiftError(
                "visible-object executor returned an invalid result"
            )
        active_ledger.note_grasp_execution()
        item = active_ledger.finish_action(
            PublicActionResult(
                status=execution.status,
                simulator_steps=execution.simulator_steps,
                hand=hand,
            )
        )
        persist(item, frame_hash)
        if execution.status is PublicStatus.GRASP_LIKELY:
            return VisibleObjectGraspApproachResult(
                final_action=envelope,
                simulator_steps=active_ledger.simulator_steps,
                model_calls=active_ledger.model_calls,
                grasp_attempts=active_ledger.grasp_executions,
                public_actions=tuple(public_actions),
                ledger=active_ledger,
            )
        remember(frame.snapshot_path)
