"""Strict public protocol and budgets for observation-only visual approach."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from behavior_hl.embodied.policy.feedback import (
    PublicActionResult,
    PublicStatus,
)
from behavior_hl.embodied.policy.protocol import (
    ActionEnvelope,
    PointActionArgs,
    VisibleObjectGraspArgs,
)
from behavior_hl.embodied.policy.registry import ActionRegistry


APPROACH_BASE_ACTIONS = frozenset(
    {
        "move_forward",
        "move_backward",
        "move_left",
        "move_right",
        "turn_left",
        "turn_right",
    }
)
APPROACH_TRANSLATION_ACTIONS = frozenset(
    {"move_forward", "move_backward", "move_left", "move_right"}
)
APPROACH_LOOK_ACTIONS = frozenset(
    {"look_up", "look_down", "look_center"}
)
APPROACH_ACTIONS = frozenset(
    {
        *APPROACH_BASE_ACTIONS,
        *APPROACH_LOOK_ACTIONS,
        "grasp_at",
        "stop",
    }
)
VISIBLE_OBJECT_APPROACH_ACTIONS = frozenset(
    {
        *APPROACH_BASE_ACTIONS,
        *APPROACH_LOOK_ACTIONS,
        "grasp",
        "stop",
    }
)
APPROACH_PROMPT_VERSION = (
    "picking-up-toys-g4a-visual-self-assessment-v4"
)
APPROACH_SYSTEM_INSTRUCTION = (
    "You are the visual high-level controller of a physical R1Pro robot in "
    "BEHAVIOR-1K 2026. You will receive onboard RGB observations consisting "
    "of up to five recent past views followed by the CURRENT view, which is "
    "always last. Use only these "
    "chronologically ordered views and your own prior actions. Infer action outcomes from "
    "the before/after images; no executor outcome label is provided. Choose exactly one "
    "registered local action. Do not assume hidden coordinates, maps, object "
    "IDs, or guaranteed motion. Return exactly one JSON object and no other "
    "text."
)


class ApproachBudgetExceeded(RuntimeError):
    """Raised before an approach can exceed a frozen public budget."""


class ApproachDecisionRejected(ValueError):
    """Raised when a registry-valid action is illegal in approach phase."""


class ApproachStopped(RuntimeError):
    """Raised when the model deliberately stops before reaching grasp."""


@dataclass(frozen=True, slots=True)
class ApproachLimits:
    max_model_calls: int = 32
    max_invalid_responses: int = 2
    max_consecutive_same_action: int = 4
    max_consecutive_blocked: int = 3
    max_grasp_queries: int = 8
    max_grasp_executions: int = 1
    max_simulator_steps: int = 2400
    prompt_history_items: int = 8
    translations_after_out_of_reach: int = 2
    minimum_effective_translation_m: float = 0.05
    enforce_executor_directed_reposition: bool = False

    def __post_init__(self) -> None:
        for field in (
            "max_model_calls",
            "max_invalid_responses",
            "max_consecutive_same_action",
            "max_consecutive_blocked",
            "max_grasp_queries",
            "max_grasp_executions",
            "max_simulator_steps",
            "prompt_history_items",
            "translations_after_out_of_reach",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        minimum_translation = self.minimum_effective_translation_m
        if (
            isinstance(minimum_translation, bool)
            or not isinstance(minimum_translation, (int, float))
            or not math.isfinite(float(minimum_translation))
            or minimum_translation <= 0.0
        ):
            raise ValueError(
                "minimum_effective_translation_m must be positive"
            )
        if not isinstance(self.enforce_executor_directed_reposition, bool):
            raise ValueError(
                "enforce_executor_directed_reposition must be boolean"
            )


@dataclass(frozen=True, slots=True)
class ApproachHistoryItem:
    sequence: int
    action: dict[str, Any]
    result: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "action": self.action,
            "result": self.result,
        }


def validate_approach_decision(envelope: ActionEnvelope) -> None:
    """Apply phase restrictions after the shared ActionRegistry parser."""

    if envelope.action not in APPROACH_ACTIONS.union(
        VISIBLE_OBJECT_APPROACH_ACTIONS
    ):
        raise ApproachDecisionRejected(
            f"action {envelope.action!r} is not allowed during approach"
        )
    if envelope.action == "grasp_at":
        if not isinstance(envelope.args, PointActionArgs):
            raise ApproachDecisionRejected("grasp_at arguments are invalid")
        if envelope.args.camera != "head":
            raise ApproachDecisionRejected(
                "approach calibration accepts only the current head image"
            )
        if envelope.args.hand != "right":
            raise ApproachDecisionRejected(
                "approach calibration freezes grasp_at to the right hand"
            )
    if envelope.action == "grasp" and not isinstance(
        envelope.args,
        VisibleObjectGraspArgs,
    ):
        raise ApproachDecisionRejected("grasp arguments are invalid")


class ApproachLedger:
    """Track only public decisions, results, and frozen episode budgets."""

    def __init__(self, limits: ApproachLimits | None = None) -> None:
        self.limits = limits or ApproachLimits()
        self.model_calls = 0
        self.invalid_responses = 0
        self.simulator_steps = 0
        self.grasp_queries = 0
        self.grasp_executions = 0
        self._translations_required_before_grasp = 0
        self._last_action: str | None = None
        self._consecutive_same_action = 0
        self._consecutive_blocked = 0
        self._pending: ActionEnvelope | None = None
        self._history: list[ApproachHistoryItem] = []

    @property
    def history(self) -> tuple[ApproachHistoryItem, ...]:
        return tuple(self._history)

    @property
    def remaining_model_calls(self) -> int:
        return self.limits.max_model_calls - self.model_calls

    @property
    def remaining_simulator_steps(self) -> int:
        return self.limits.max_simulator_steps - self.simulator_steps

    @property
    def remaining_grasp_queries(self) -> int:
        return max(0, self.limits.max_grasp_queries - self.grasp_queries)

    @property
    def translations_required_before_grasp(self) -> int:
        return self._translations_required_before_grasp

    @property
    def grasp_query_budget_exhausted(self) -> bool:
        """Let a resident runner terminate before making another model call."""

        return self.remaining_grasp_queries == 0

    @property
    def grasp_at_available(self) -> bool:
        return (
            not self.grasp_query_budget_exhausted
            and self.translations_required_before_grasp == 0
        )

    @property
    def grasp_at_unavailable_reason(self) -> str | None:
        if self.grasp_query_budget_exhausted:
            return "reach_query_budget_exhausted"
        if self.translations_required_before_grasp:
            return "reposition_required"
        return None

    def note_model_call(self) -> int:
        if self.model_calls >= self.limits.max_model_calls:
            raise ApproachBudgetExceeded("Qwen call budget exhausted")
        self.model_calls += 1
        return self.model_calls

    def note_invalid_response(self) -> None:
        self.invalid_responses += 1
        if self.invalid_responses > self.limits.max_invalid_responses:
            raise ApproachBudgetExceeded(
                "invalid Qwen response budget exhausted"
            )

    def note_simulator_steps(self, steps: int) -> None:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("simulator steps must be a non-negative integer")
        next_steps = self.simulator_steps + steps
        if next_steps > self.limits.max_simulator_steps:
            raise ApproachBudgetExceeded(
                "approach simulator-step budget exhausted"
            )
        self.simulator_steps = next_steps

    def begin_action(
        self, envelope: ActionEnvelope
    ) -> PublicActionResult | None:
        """Begin an executable action or return a public immediate result.

        A grasp request made during the reposition cooldown never becomes
        pending, never reads depth, and never consumes a reach query.  The
        resident runner must branch on the returned result and skip executor
        dispatch when it is not ``None``.
        """

        if self._pending is not None:
            raise RuntimeError("an approach action is already active")
        validate_approach_decision(envelope)
        if (
            envelope.action in {"grasp_at", "grasp"}
            and self.grasp_query_budget_exhausted
        ):
            raise ApproachBudgetExceeded(
                "grasp_at reach-query budget exhausted"
            )
        if (
            envelope.action == "grasp_at"
            and self.limits.enforce_executor_directed_reposition
            and self._translations_required_before_grasp > 0
        ):
            result = PublicActionResult(
                status=PublicStatus.REPOSITION_REQUIRED,
                simulator_steps=0,
                hand="right",
            )
            self._append_history(envelope, result)
            return result
        if envelope.action == self._last_action:
            consecutive = self._consecutive_same_action + 1
        else:
            consecutive = 1
        if consecutive > self.limits.max_consecutive_same_action:
            raise ApproachBudgetExceeded(
                "consecutive identical action budget exhausted"
            )
        if envelope.action in {"grasp_at", "grasp"}:
            self.grasp_queries += 1
        self._last_action = envelope.action
        self._consecutive_same_action = consecutive
        self._pending = envelope
        return None

    def note_grasp_execution(self) -> None:
        if (
            self._pending is None
            or self._pending.action not in {"grasp_at", "grasp"}
        ):
            raise RuntimeError(
                "physical grasp execution requires a pending grasp action"
            )
        if self.grasp_executions >= self.limits.max_grasp_executions:
            raise ApproachBudgetExceeded(
                "physical grasp execution budget exhausted"
            )
        self.grasp_executions += 1

    def finish_action(self, result: PublicActionResult) -> ApproachHistoryItem:
        if self._pending is None:
            raise RuntimeError("no approach action is active")
        pending = self._pending
        self._pending = None
        self.note_simulator_steps(result.simulator_steps)
        if (
            pending.action == "grasp_at"
            and result.status is PublicStatus.OUT_OF_REACH
            and self.limits.enforce_executor_directed_reposition
        ):
            self._translations_required_before_grasp = (
                self.limits.translations_after_out_of_reach
            )
        elif (
            pending.action in APPROACH_TRANSLATION_ACTIONS
            and self._is_effective_translation(result)
            and self._translations_required_before_grasp > 0
        ):
            self._translations_required_before_grasp -= 1
        if result.status is PublicStatus.BLOCKED:
            self._consecutive_blocked += 1
        else:
            self._consecutive_blocked = 0
        if self._consecutive_blocked > self.limits.max_consecutive_blocked:
            raise ApproachBudgetExceeded(
                "consecutive blocked-action budget exhausted"
            )
        return self._append_history(pending, result)

    def _append_history(
        self,
        envelope: ActionEnvelope,
        result: PublicActionResult,
    ) -> ApproachHistoryItem:
        item = ApproachHistoryItem(
            sequence=len(self._history),
            action=envelope.to_dict(),
            result=result.to_dict(),
        )
        self._history.append(item)
        return item

    def _is_effective_translation(
        self, result: PublicActionResult
    ) -> bool:
        if result.status not in {
            PublicStatus.COMPLETED,
            PublicStatus.MOTION_PARTIAL,
        }:
            return False
        odometry = result.body_odometry
        if odometry is None:
            return False
        progress = (
            odometry.forward_m**2 + odometry.left_m**2
        ) ** 0.5
        return progress >= self.limits.minimum_effective_translation_m

    def public_summary(self) -> dict[str, Any]:
        """Return the full sensor-derived diagnostic record for artifacts."""

        return {
            "model_calls": self.model_calls,
            "remaining_model_calls": self.remaining_model_calls,
            "invalid_responses": self.invalid_responses,
            "simulator_steps": self.simulator_steps,
            "remaining_simulator_steps": self.remaining_simulator_steps,
            "grasp_queries": self.grasp_queries,
            "remaining_grasp_queries": self.remaining_grasp_queries,
            "grasp_executions": self.grasp_executions,
            "action_availability": {
                "grasp_at": {
                    "available": self.grasp_at_available,
                    "unavailable_reason": self.grasp_at_unavailable_reason,
                    "translations_required": (
                        self.translations_required_before_grasp
                    ),
                    "remaining_reach_queries": (
                        self.remaining_grasp_queries
                    ),
                }
            },
            "history": [
                item.to_dict()
                for item in self._history[
                    -self.limits.prompt_history_items :
                ]
            ],
        }

    def model_visible_summary(self) -> dict[str, Any]:
        """Return only the model's own actions, never executor-derived state."""

        return {
            "prior_actions": [
                {
                    "sequence": item.sequence,
                    "action": dict(item.action),
                }
                for item in self._history[
                    -self.limits.prompt_history_items :
                ]
            ],
        }


def approach_action_manifest(registry: ActionRegistry) -> dict[str, Any]:
    """Expose only phase-allowed registry entries to the model."""

    return {
        "schema_version": registry.schema_version,
        "phase": "g4a.approach",
        "actions": [
            registry.spec(name).model_manifest()
            for name in sorted(
                set(APPROACH_ACTIONS)
                .union(VISIBLE_OBJECT_APPROACH_ACTIONS)
                .intersection(registry.names)
            )
        ],
    }


def build_approach_prompt(
    *,
    task_language: str,
    registry: ActionRegistry,
    ledger: ApproachLedger,
    frame_sha256: str,
) -> str:
    """Build a deterministic prompt from public, serializable inputs."""

    if not isinstance(task_language, str) or not task_language.strip():
        raise ValueError("task_language must not be empty")
    if (
        not isinstance(frame_sha256, str)
        or len(frame_sha256) != 64
        or any(character not in "0123456789abcdef" for character in frame_sha256)
    ):
        raise ValueError("frame_sha256 must be a lowercase SHA256 digest")
    category_grasp = "grasp" in registry.names
    public_context = {
        "prompt_version": APPROACH_PROMPT_VERSION,
        "task": task_language.strip(),
        "phase_goal": (
            "Use local physical motion to approach the visible tennis ball. "
            + (
                "When you judge a grasp attempt appropriate, call grasp with "
                "object=tennis_ball. After any attempt, infer what physically "
                "happened only from the next onboard images."
                if category_grasp
                else
                "When you judge a grasp attempt appropriate, select its center "
                "in the current head image with grasp_at. After any attempt, "
                "infer what physically happened only from the next onboard images."
            )
        ),
        "image": {
            "camera": "head",
            "coordinates": (
                "normalized [x,y], [0,0] top-left, [1,1] bottom-right"
            ),
            "sha256": frame_sha256,
        },
        "constraints": [
            "Choose exactly one action.",
            "Each move/turn is a short local attempt and may be blocked.",
            "Use only visible evidence and your own prior actions.",
            "Never assume navigate_to, a map, object coordinates, or teleport.",
            *(
                [
                    "grasp accepts only a currently visible task-object category.",
                    "Do not emit pixels, points, poses, camera names, instance IDs, or a hand.",
                ]
                if category_grasp
                else
                [
                    "grasp_at must use camera=head and hand=right.",
                    "grasp_at point values are normalized decimals, for example "
                    "[0.50,0.50]; never emit image pixels such as [562,390].",
                ]
            ),
            "stop before a confirmed grasp ends this calibration as failure.",
        ],
        "action_manifest": approach_action_manifest(registry),
        "model_state": ledger.model_visible_summary(),
        "output": (
            "Return one strict JSON action envelope accepted by the manifest. "
            "A short reasoning string is allowed inside the JSON."
        ),
    }
    return json.dumps(
        public_context,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def prompt_sha256(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
