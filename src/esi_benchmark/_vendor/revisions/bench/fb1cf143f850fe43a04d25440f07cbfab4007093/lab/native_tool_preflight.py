#!/usr/bin/env python3
"""Simulator-free multi-turn native-tool compatibility probe.

The probe exercises the exact Chat Completions message contract used by the
resident Prompt lab: an Assistant native function call, a ``role=tool`` result,
and a fresh User observation.  It checks completed and blocked results as well
as a temporary visual target selection before any simulator is touched.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping


from Agent import (  # type: ignore[import-not-found]
    ACTION20_TOOL_SCHEMA,
    ACTION20_TOOL_SCHEMA_PATH,
    NativeToolDecision,
    SystemPromptRevision,
    parse_native_chat_completion,
)


PREFLIGHT_SCHEMA_VERSION = "behavior-hl-native-tool-preflight/v3"
MAXIMUM_ATTEMPTS_PER_STEP = 3
ATTEMPT_SEEDS = (19506, 19507, 19508)
PostCallback = Callable[[Mapping[str, Any]], Mapping[str, Any]]

PROTOCOL_ONLY_PREFIX = (
    "Protocol-only capability check; this is not a household task and there "
    "is no scene to interpret. Follow the requested call exactly and do not "
    "choose or add another action or argument. "
)


class NativeToolPreflightError(RuntimeError):
    """The provider failed the simulator-free native-tool contract."""


@dataclass(frozen=True)
class ProbeStep:
    name: str
    instruction: str
    expected_action: str
    expected_target: int | None
    expected_target_description: str | None = None
    preceding_tool_result: Mapping[str, Any] | None = None


PROBE_STEPS = (
    ProbeStep(
        name="initial_wait",
        instruction=(
            PROTOCOL_ONLY_PREFIX
            + "Capability preflight step 1/5: make exactly one native "
            "robot_action tool call with action wait and no target."
        ),
        expected_action="wait",
        expected_target=None,
    ),
    ProbeStep(
        name="completed_followup",
        instruction=(
            PROTOCOL_ONLY_PREFIX
            + "Capability preflight step 2/5. A new observation is available. "
            "Make exactly one native robot_action tool call with action "
            "turn_left and no target."
        ),
        expected_action="turn_left",
        expected_target=None,
        preceding_tool_result={"status": "completed"},
    ),
    ProbeStep(
        name="blocked_followup",
        instruction=(
            PROTOCOL_ONLY_PREFIX
            + "Capability preflight step 3/5. The preceding action was blocked "
            "and a new observation is available. Make exactly one native "
            "robot_action tool call with action turn_right and no target."
        ),
        expected_action="turn_right",
        expected_target=None,
        preceding_tool_result={
            "status": "blocked",
            "body_odometry": {
                "forward_m": 0.0,
                "left_m": 0.0,
                "yaw_deg": 0.0,
            },
        },
    ),
    ProbeStep(
        name="candidate_request",
        instruction=(
            PROTOCOL_ONLY_PREFIX
            + "Capability preflight step 4/5. A new observation is available. "
            "Make exactly one native robot_action tool call with action pick; "
            "omit target to request visible candidates."
        ),
        expected_action="pick",
        expected_target=None,
        preceding_tool_result={
            "status": "completed",
            "body_odometry": {
                "forward_m": 0.0,
                "left_m": 0.0,
                "yaw_deg": -10.0,
            },
        },
    ),
    ProbeStep(
        name="candidate_selection",
        instruction=(
            PROTOCOL_ONLY_PREFIX
            + "Capability preflight step 5/5. The current onboard observation "
            "shows temporary candidate C3 from the immediately preceding pick "
            "request. Make exactly one native robot_action tool call with "
            "action pick, target 3, and target_description red cup on counter."
        ),
        expected_action="pick",
        expected_target=3,
        expected_target_description="red cup on counter",
        preceding_tool_result={"status": "completed"},
    ),
)


def run_native_tool_preflight(
    *,
    post: PostCallback,
    model: str,
    prompt_revision: SystemPromptRevision,
    max_tokens: int = 512,
    thinking_tool_choice_auto: bool = False,
    tool_schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the five-step native-tool round trip and return an audit report."""

    if not callable(post):
        raise TypeError("preflight post callback must be callable")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("preflight model must be nonempty")
    if not isinstance(prompt_revision, SystemPromptRevision):
        raise TypeError("prompt_revision must be SystemPromptRevision")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise ValueError("preflight max_tokens must be a positive integer")
    if not isinstance(thinking_tool_choice_auto, bool):
        raise TypeError("thinking tool-choice mode must be boolean")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt_revision.text},
        {
            "role": "user",
            "content": PROBE_STEPS[0].instruction,
        },
    ]
    step_records: list[dict[str, Any]] = []
    previous_decision: NativeToolDecision | None = None

    for index, step in enumerate(PROBE_STEPS):
        if index:
            assert previous_decision is not None
            assert step.preceding_tool_result is not None
            messages.extend(
                [
                    previous_decision.assistant_message(),
                    {
                        "role": "tool",
                        "tool_call_id": previous_decision.tool_call_id,
                        "content": json.dumps(
                            step.preceding_tool_result,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": step.instruction}
                        ],
                    },
                ]
            )

        request = {
            "model": model.strip(),
            "messages": _json_clone(messages),
            "tools": [_json_clone(ACTION20_TOOL_SCHEMA if tool_schema is None else tool_schema)],
            "parallel_tool_calls": False,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
        if thinking_tool_choice_auto:
            request.update(
                {
                    "tool_choice": "auto",
                    "enable_thinking": True,
                }
            )
        else:
            request.update(
                {
                    "tool_choice": {
                        "type": "function",
                        "function": {"name": "robot_action"},
                    },
                    "enable_thinking": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            )
        failure: str | None = None
        failure_cause: Exception | None = None
        response: Mapping[str, Any] | None = None
        decision: NativeToolDecision | None = None
        attempt_count = 0
        for attempt_count, seed in enumerate(ATTEMPT_SEEDS, start=1):
            request["seed"] = seed
            response = post(request)
            try:
                candidate = parse_native_chat_completion(response)
            except Exception as exc:
                failure = (
                    f"returned an invalid response: {type(exc).__name__}: {exc}"
                )
                failure_cause = exc
                continue
            if (
                step.expected_target is None
                and candidate.action == step.expected_action
                and candidate.target is not None
            ):
                candidate = candidate.without_target()
            if candidate.tool_call_id is None:
                failure = "omitted tool_call_id"
                failure_cause = None
                continue
            if (
                candidate.action != step.expected_action
                or candidate.target != step.expected_target
                or candidate.target_description
                != step.expected_target_description
            ):
                failure = (
                    "expected "
                    f"{_format_expected(step.expected_action, step.expected_target, step.expected_target_description)} "
                    "but received "
                    f"{_format_expected(candidate.action, candidate.target, candidate.target_description)}"
                )
                failure_cause = None
                continue
            decision = candidate
            break
        if decision is None or response is None:
            error = NativeToolPreflightError(
                f"native-tool preflight {step.name} {failure} after "
                f"{MAXIMUM_ATTEMPTS_PER_STEP} attempts"
            )
            if failure_cause is not None:
                raise error from failure_cause
            raise error
        step_records.append(
            {
                "index": index,
                "name": step.name,
                "expected_action": step.expected_action,
                "expected_target": step.expected_target,
                "expected_target_description": (
                    step.expected_target_description
                ),
                "native_reasoning_present": (
                    decision.reasoning_content is not None
                ),
                "attempt_count": attempt_count,
                "recovered_after_retry": attempt_count > 1,
                "response_sha256": _canonical_sha256(response),
            }
        )
        previous_decision = decision

    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "PASS",
        "provider_checked": True,
        "cached": False,
        "model": model.strip(),
        "prompt_sha256": hashlib.sha256(
            prompt_revision.text.encode("utf-8")
        ).hexdigest(),
        "fixed_tool_schema_sha256": (
            hashlib.sha256(ACTION20_TOOL_SCHEMA_PATH.read_bytes()).hexdigest()
            if tool_schema is None else _canonical_sha256(tool_schema)
        ),
        "simulator_touched": False,
        "step_count": len(step_records),
        "maximum_attempts_per_step": MAXIMUM_ATTEMPTS_PER_STEP,
        "model_call_count": sum(
            record["attempt_count"] for record in step_records
        ),
        "steps": step_records,
        "native_reasoning_present": any(
            record["native_reasoning_present"] for record in step_records
        ),
        "thinking_tool_choice_auto": thinking_tool_choice_auto,
        "secrets_serialized": False,
    }


def _format_expected(
    action: str,
    target: int | None,
    target_description: str | None,
) -> str:
    value: dict[str, Any] = {"action": action}
    if target is not None:
        value["target"] = target
        value["target_description"] = target_description
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_clone(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
