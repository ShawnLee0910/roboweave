"""Native Action20 decisions, public results, and provider parsing."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping

from action20_protocol import (
    ACTION_DECISION_LIMIT,
    BASE_ODOMETRY_ACTIONS,
    BASE_TRANSLATION_ACTIONS,
    BASE_TURN_ACTIONS,
    MARKER_ACTIONS,
    NAVIGATION_MOVE_DISTANCES_M,
    NAVIGATION_REPEAT_MAX,
    NAVIGATION_TURN_ANGLES_DEG,
    PUBLIC_TOOL_ACTIONS,
    navigation_repeat_max,
    navigation_step_multiplier,
)


ODOMETRY_FIELDS = frozenset({"forward_m", "left_m", "yaw_deg"})
ACTION20_BLOCKED_CATEGORIES = frozenset(
    {
        "articulated_part_blocked",
        "execution_unverified",
        "geometric_fit_blocked",
        "navigation_blocked",
        "no_selectable_candidates",
        "out_of_reach",
        "placement_collision_blocked",
        "placement_support_unavailable",
        "relation_not_realized",
        "requires_empty_left_hand",
        "requires_object_in_left_hand",
        "selection_unavailable",
        "view_limit_reached",
        "view_motion_blocked_while_holding",
    }
)
_THINK_OPEN = "<think>"
ARTICULATED_BLOCKED_DETAILS = frozenset({
    "robot_obstruction", "external_obstruction", "contents_obstruction",
    "multiple_obstructions",
})
_THINK_CLOSE = "</think>"
_MAX_REASONING_CHARS = 131_072
_TOOL_CONTENT_ECHO = re.compile(
    r"(?:\s*(?:<tool_call>\s*)?<function=robot_action>\s*"
    r"(?:<parameter=[a-z_]+>[^<>]*</parameter>\s*)+"
    r"</function>\s*</tool_call>\s*)+"
)
# Include binary-float noise at the exact 1 mm boundary.
_NAVIGATION_DISTANCE_ABS_TOL_M = 1e-3 + 1e-12


def _clone_extra_content(value: Any) -> dict[str, Any]:
    """Copy provider extra_content without rewriting thought signatures."""

    if not isinstance(value, Mapping) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("native extra_content must be a mapping")
    try:
        cloned = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("native extra_content is not JSON") from exc
    if not isinstance(cloned, dict):
        raise ValueError("native extra_content must be a mapping")
    return cloned


def validate_native_reasoning_text(value: Any) -> str:
    """Validate reasoning before it enters history or tagged artifacts."""

    if not isinstance(value, str) or len(value) > _MAX_REASONING_CHARS:
        raise ValueError("native reasoning is invalid")
    if _THINK_OPEN in value or _THINK_CLOSE in value:
        raise ValueError("native reasoning contains a reserved think delimiter")
    return value


def replay_native_reasoning(
    message: MutableMapping[str, Any],
    reasoning: Any,
    source: Any,
) -> None:
    """Restore validated reasoning to its original provider message field."""

    if reasoning is None:
        if source is not None:
            raise ValueError("native reasoning source has no reasoning text")
        return
    reasoning = validate_native_reasoning_text(reasoning)
    if source in ("reasoning_content", "reasoning"):
        message[source] = reasoning
    elif source == "content_think":
        if message.get("content") not in (None, ""):
            raise ValueError("content reasoning would overwrite assistant text")
        message["content"] = f"{_THINK_OPEN}{reasoning}{_THINK_CLOSE}"
    else:
        raise ValueError("native reasoning source is invalid")


def isolate_native_reasoning(
    message: Mapping[str, Any],
) -> tuple[str | None, str | None, str | None]:
    """Return isolated reasoning, its source, and non-reasoning content."""

    if not isinstance(message, Mapping):
        raise ValueError("native assistant message must be a mapping")
    reasoning_content = message.get("reasoning_content")
    reasoning_vllm = message.get("reasoning")
    for value in (reasoning_content, reasoning_vllm):
        if value is not None:
            validate_native_reasoning_text(value)
    if (
        reasoning_content is not None
        and reasoning_vllm is not None
        and reasoning_content != reasoning_vllm
    ):
        raise ValueError("native reasoning appeared differently in multiple fields")
    reasoning = (
        reasoning_content
        if reasoning_content is not None
        else reasoning_vllm
    )
    source = (
        "reasoning_content"
        if reasoning_content is not None
        else "reasoning"
        if reasoning_vllm is not None
        else None
    )

    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise ValueError("native assistant content is invalid")
    if not isinstance(content, str):
        return reasoning, source, content

    # Some gateways return one native call plus a textual tool-template echo.
    # Native tool_calls alone is executable; never recover actions from this text.
    # The caller still validates the native call and all argument constraints.
    calls = message.get("tool_calls")
    echo_content = content.replace(_THINK_CLOSE, "")
    if (len(content) <= _MAX_REASONING_CHARS
            and isinstance(calls, list) and len(calls) == 1
            and isinstance(calls[0], Mapping)
            and calls[0].get("type") == "function"
            and isinstance(calls[0].get("function"), Mapping)
            and calls[0]["function"].get("name") == "robot_action"
            and content.strip()
            and (not echo_content.strip()
                 or _TOOL_CONTENT_ECHO.fullmatch(echo_content))):
        return reasoning, source, None

    stripped = content.lstrip()
    if not stripped.startswith(_THINK_OPEN):
        if _THINK_OPEN in content or _THINK_CLOSE in content:
            raise ValueError("content reasoning block is malformed")
        return reasoning, source, content

    close_index = stripped.find(_THINK_CLOSE, len(_THINK_OPEN))
    if close_index < 0:
        raise ValueError("content reasoning block is malformed")
    content_reasoning = stripped[len(_THINK_OPEN):close_index]
    remaining = stripped[close_index + len(_THINK_CLOSE):]
    if any(
        marker in value
        for value in (content_reasoning, remaining)
        for marker in (_THINK_OPEN, _THINK_CLOSE)
    ):
        raise ValueError("content reasoning block is malformed")
    content_reasoning = content_reasoning.strip()
    validate_native_reasoning_text(content_reasoning)
    if reasoning is not None and reasoning.strip() != content_reasoning:
        raise ValueError("native reasoning appeared differently in multiple fields")
    if reasoning is None:
        reasoning = content_reasoning
        source = "content_think"
    return reasoning, source, None if not remaining.strip() else remaining


def _normalize_body_odometry(value: Any) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or set(value) != ODOMETRY_FIELDS:
        raise ValueError("public body_odometry fields changed")
    raw: dict[str, float] = {}
    normalized: dict[str, float] = {}
    for field in sorted(ODOMETRY_FIELDS):
        item = value[field]
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ValueError("public body_odometry must be finite numeric")
        raw[field] = float(item)
        rounded = round(raw[field], 2)
        normalized[field] = 0.0 if rounded == 0.0 else rounded
    if (
        math.hypot(raw["forward_m"], raw["left_m"]) > 0.60 + 1e-9
        or abs(raw["yaw_deg"]) > 30.0 + 1e-9
    ):
        raise ValueError("public body_odometry exceeds one Action20 step")
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class PublicToolResult:
    """The complete allowlisted public result of the preceding action."""

    status: str
    body_odometry: Mapping[str, float] | None = None
    stage: int | None = None
    repeat_requested: int | None = None
    repeat_executed: int | None = None
    steps: tuple[Mapping[str, Any], ...] | None = None
    blocked_category: str | None = None
    action_decisions_remaining: int | None = None
    blocked_detail: str | None = None

    def __post_init__(self) -> None:
        if self.action_decisions_remaining is not None and (
            isinstance(self.action_decisions_remaining, bool)
            or not isinstance(self.action_decisions_remaining, int)
            or not 0
            <= self.action_decisions_remaining
            <= ACTION_DECISION_LIMIT
        ):
            raise ValueError("public remaining action decisions are invalid")
        if self.status not in {"completed", "blocked", "selection_required"}:
            raise ValueError(
                "public status must be completed, blocked, or selection_required"
            )
        if self.status == "selection_required":
            if self.stage not in {1, 2}:
                raise ValueError(
                    "selection_required must identify Marker stage 1 or 2"
                )
            if self.body_odometry is not None:
                raise ValueError(
                    "selection_required cannot expose body odometry"
                )
        elif self.stage is not None:
            raise ValueError("final public status cannot expose a selection stage")
        if self.blocked_category is not None and (
            self.status != "blocked"
            or self.blocked_category not in ACTION20_BLOCKED_CATEGORIES
        ):
            raise ValueError("public blocked category is invalid")
        if self.blocked_detail is not None and (
            self.status != "blocked"
            or self.blocked_category != "articulated_part_blocked"
            or not isinstance(self.blocked_detail, str)
            or self.blocked_detail not in ARTICULATED_BLOCKED_DETAILS
        ):
            raise ValueError("public blocked detail is invalid")
        if self.steps is not None:
            if (
                self.status not in {"completed", "blocked"}
                or self.body_odometry is not None
                or isinstance(self.repeat_requested, bool)
                or not isinstance(self.repeat_requested, int)
                or not 2
                <= self.repeat_requested
                <= NAVIGATION_REPEAT_MAX
                or isinstance(self.repeat_executed, bool)
                or not isinstance(self.repeat_executed, int)
                or not 1 <= self.repeat_executed <= self.repeat_requested
                or not isinstance(self.steps, tuple)
                or len(self.steps) != self.repeat_executed
            ):
                raise ValueError("public repeated-step result is invalid")
            normalized_steps: list[Mapping[str, Any]] = []
            for index, step in enumerate(self.steps, start=1):
                step_keys = {"step_index", "status", "body_odometry"}
                if (
                    not isinstance(step, Mapping)
                    or set(step)
                    not in (step_keys, step_keys | {"blocked_category"})
                    or step.get("step_index") != index
                    or step.get("status") not in {"completed", "blocked"}
                    or (index < len(self.steps) and step.get("status") != "completed")
                    or (
                        step.get("blocked_category") is not None
                        and (
                            step.get("status") != "blocked"
                            or step.get("blocked_category")
                            not in ACTION20_BLOCKED_CATEGORIES
                        )
                    )
                ):
                    raise ValueError("public repeated-step entry is invalid")
                normalized_step = {
                    "step_index": index,
                    "status": step["status"],
                    "body_odometry": _normalize_body_odometry(
                        step["body_odometry"]
                    ),
                }
                if step.get("blocked_category") is not None:
                    normalized_step["blocked_category"] = step[
                        "blocked_category"
                    ]
                normalized_steps.append(MappingProxyType(normalized_step))
            if normalized_steps[-1]["status"] != self.status:
                raise ValueError("public repeated-step final status changed")
            if (
                normalized_steps[-1].get("blocked_category")
                != self.blocked_category
            ):
                raise ValueError("public repeated-step blocked category changed")
            object.__setattr__(self, "steps", tuple(normalized_steps))
            return
        if self.repeat_requested is not None or self.repeat_executed is not None:
            raise ValueError("repeat counts require repeated-step results")
        if self.body_odometry is None:
            return
        object.__setattr__(
            self,
            "body_odometry",
            _normalize_body_odometry(self.body_odometry),
        )

    def to_public_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"status": self.status}
        if self.action_decisions_remaining is not None:
            value["action_decisions_remaining"] = (
                self.action_decisions_remaining
            )
        if self.stage is not None:
            value["stage"] = self.stage
        if self.body_odometry is not None:
            value["body_odometry"] = dict(self.body_odometry)
        if self.blocked_category is not None:
            value["blocked_category"] = self.blocked_category
        if self.blocked_detail is not None:
            value["blocked_detail"] = self.blocked_detail
        if self.steps is not None:
            value["repeat_requested"] = self.repeat_requested
            value["repeat_executed"] = self.repeat_executed
            value["steps"] = [
                dict(step)
                | {"body_odometry": dict(step["body_odometry"])}
                for step in self.steps
            ]
        return value


@dataclass(frozen=True)
class NativeToolDecision:
    """One normalized provider-native function call and native reasoning."""

    action: str
    target: int | None
    target_description: str | None
    reasoning_content: str | None
    reasoning_source: str | None
    tool_call_id: str | None
    repeat: int | None = None
    move_distance_m: float | None = None
    turn_angle_deg: float | None = None
    ignored_target_fields: int = 0
    ignored_navigation_fields: int = 0
    eager_marker_target_removed: bool = False
    tool_call_extra_content: Mapping[str, Any] | None = None
    message_extra_content: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.ignored_target_fields, bool)
            or not isinstance(self.ignored_target_fields, int)
            or self.ignored_target_fields < 0
            or isinstance(self.ignored_navigation_fields, bool)
            or not isinstance(self.ignored_navigation_fields, int)
            or self.ignored_navigation_fields < 0
        ):
            raise ValueError("normalization field count is invalid")
        if self.action not in PUBLIC_TOOL_ACTIONS:
            raise ValueError("model selected an unavailable action")
        if self.repeat is not None:
            if (
                self.action not in BASE_ODOMETRY_ACTIONS
                or isinstance(self.repeat, bool)
                or not isinstance(self.repeat, int)
                or not 1 <= self.repeat <= navigation_repeat_max(self.action)
            ):
                raise ValueError("model selected an invalid navigation repeat")
        navigation_step_multiplier(
            self.action,
            move_distance_m=self.move_distance_m,
            turn_angle_deg=self.turn_angle_deg,
        )
        if self.target is None:
            if self.target_description is not None:
                raise ValueError("target_description requires target")
            return
        if (
            self.action not in MARKER_ACTIONS
            or isinstance(self.target, bool)
            or not isinstance(self.target, int)
            or not 1 <= self.target <= 999
        ):
            raise ValueError("model selected an invalid target")
        if self.target_description is not None:
            _validate_target_description(self.target_description)

    @property
    def public_repeat(self) -> int:
        return self.repeat or 1

    @property
    def navigation_substeps(self) -> int:
        return navigation_step_multiplier(
            self.action,
            move_distance_m=self.move_distance_m,
            turn_angle_deg=self.turn_angle_deg,
        )

    @property
    def physical_repeat(self) -> int:
        return self.public_repeat * self.navigation_substeps

    def function_arguments(self) -> dict[str, Any]:
        value: dict[str, Any] = {"action": self.action}
        if self.repeat is not None:
            value["repeat"] = self.repeat
        if self.move_distance_m is not None:
            value["move_distance_m"] = self.move_distance_m
        if self.turn_angle_deg is not None:
            value["turn_angle_deg"] = self.turn_angle_deg
        if self.target is not None:
            value["target"] = self.target
            if self.target_description is not None:
                value["target_description"] = self.target_description
        return value

    def assistant_message(self) -> dict[str, Any]:
        """Return the canonical Assistant message for this tool decision."""

        if self.tool_call_id is None:
            raise ValueError("native tool call id is required for tool response")
        tool_call: dict[str, Any] = {
            "id": self.tool_call_id,
            "type": "function",
            "function": {
                "name": "robot_action",
                "arguments": json.dumps(
                    self.function_arguments(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        }
        if self.tool_call_extra_content is not None:
            tool_call["extra_content"] = _clone_extra_content(
                self.tool_call_extra_content
            )
        message: dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call],
        }
        if self.message_extra_content is not None:
            message["extra_content"] = _clone_extra_content(
                self.message_extra_content
            )
        return message

    def without_target(self) -> "NativeToolDecision":
        """Return the same decision as a targetless candidate request."""

        removed_fields = int(self.target is not None) + int(
            self.target_description is not None
        )
        return NativeToolDecision(
            action=self.action,
            target=None,
            target_description=None,
            reasoning_content=self.reasoning_content,
            reasoning_source=self.reasoning_source,
            tool_call_id=self.tool_call_id,
            repeat=self.repeat,
            move_distance_m=self.move_distance_m,
            turn_angle_deg=self.turn_angle_deg,
            ignored_target_fields=(
                self.ignored_target_fields + removed_fields
            ),
            ignored_navigation_fields=self.ignored_navigation_fields,
            eager_marker_target_removed=(
                self.eager_marker_target_removed or self.target is not None
            ),
            tool_call_extra_content=self.tool_call_extra_content,
            message_extra_content=self.message_extra_content,
        )

    def runtime_action(
        self,
        *,
        pending_action: str | None = None,
        pending_targets: frozenset[int] = frozenset(),
    ) -> dict[str, Any]:
        """Resolve a target only through the current one-shot C-label namespace."""

        value: dict[str, Any] = {"action": self.action}
        if self.target is not None:
            if (
                pending_action != self.action
                or self.target not in pending_targets
            ):
                raise ValueError(
                    "target is not in the current same-action Marker set"
                )
            value["args"] = {"marker": f"C{self.target}"}
        return value


@dataclass(frozen=True)
class NativeActionDispatch:
    """Validated public decision plus its backend Action20 representation."""

    decision: NativeToolDecision
    runtime_action: Mapping[str, Any]


def parse_native_tool_message(
    message: Mapping[str, Any],
    *,
    allow_navigation_repeat: bool = True,
    allow_navigation_step_sizes: bool = True,
) -> NativeToolDecision:
    """Validate exactly one native tool call without synthesizing reasoning."""

    allowed_message_fields = {
        "role",
        "content",
        "reasoning_content",
        "reasoning",
        "tool_calls",
        "refusal",
        "annotations",
        "audio",
        "function_call",
        "extra_content",
    }
    if (
        not isinstance(message, Mapping)
        or not set(message).issubset(allowed_message_fields)
        or message.get("role") != "assistant"
        or message.get("refusal") is not None
        or message.get("annotations") is not None
        or message.get("audio") is not None
        or message.get("function_call") is not None
    ):
        raise ValueError("native assistant message fields changed")
    reasoning, reasoning_source, content = isolate_native_reasoning(message)
    if content not in {None, ""}:
        raise ValueError("native assistant content must contain only reasoning")
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        raise ValueError("exactly one native tool call is required")
    tool_call = tool_calls[0]
    if (
        not isinstance(tool_call, Mapping)
        or not set(tool_call).issubset(
            {"id", "index", "type", "function", "extra_content"}
        )
        or tool_call.get("type") != "function"
        or not isinstance(tool_call.get("function"), Mapping)
    ):
        raise ValueError("native tool call fields changed")
    function = tool_call["function"]
    if (
        set(function) != {"name", "arguments"}
        or function.get("name") != "robot_action"
    ):
        raise ValueError("native tool function identity changed")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            # Cosmos3's vLLM template can wrap the same native call once more
            # after a tool-result turn. Accept only that exact, lossless shape.
            wrapped = re.fullmatch(
                r"\s*<tool_call>\s*(\{.*\})\s*</tool_call>\s*",
                arguments,
                flags=re.DOTALL,
            )
            if wrapped is None:
                raise ValueError("native function arguments are not JSON") from exc
            try:
                envelope = json.loads(wrapped.group(1))
            except json.JSONDecodeError:
                raise ValueError("native function arguments are not JSON") from exc
            if (
                not isinstance(envelope, Mapping)
                or set(envelope) != {"name", "arguments"}
                or envelope.get("name") != "robot_action"
                or not isinstance(envelope.get("arguments"), Mapping)
            ):
                raise ValueError("native function arguments are not JSON") from exc
            arguments = envelope["arguments"]
    if (
        not isinstance(arguments, Mapping)
        or "action" not in arguments
        or not set(arguments).issubset(
            {
                "action",
                "move_distance_m",
                "repeat",
                "target",
                "target_description",
                "turn_angle_deg",
            }
        )
    ):
        raise ValueError("robot_action arguments changed")
    action = arguments.get("action")
    if not isinstance(action, str) or action not in PUBLIC_TOOL_ACTIONS:
        raise ValueError("model selected an unavailable action")
    repeat_present = "repeat" in arguments
    repeat = arguments.get("repeat") if repeat_present else None
    ignored_navigation_fields = 0
    if repeat_present and action in BASE_ODOMETRY_ACTIONS:
        if (
            not allow_navigation_repeat
            or isinstance(repeat, bool)
            or not isinstance(repeat, int)
            or not 1 <= repeat <= navigation_repeat_max(action)
        ):
            raise ValueError("model selected an invalid navigation repeat")
    elif repeat_present:
        # Some structured-output providers populate every optional schema
        # field.  Repeat has no meaning outside navigation, so discard it.
        repeat = None
        ignored_navigation_fields += 1
    if (
        "move_distance_m" in arguments
        and action not in BASE_TRANSLATION_ACTIONS
    ):
        ignored_navigation_fields += 1
    if "turn_angle_deg" in arguments and action not in BASE_TURN_ACTIONS:
        ignored_navigation_fields += 1
    move_distance_m = (
        arguments.get("move_distance_m")
        if action in BASE_TRANSLATION_ACTIONS
        and "move_distance_m" in arguments
        else None
    )
    turn_angle_deg = (
        arguments.get("turn_angle_deg")
        if action in BASE_TURN_ACTIONS and "turn_angle_deg" in arguments
        else None
    )
    if (move_distance_m is not None or turn_angle_deg is not None) and not (
        allow_navigation_step_sizes
    ):
        raise ValueError("model selected an unavailable navigation step size")
    if move_distance_m is not None:
        if (
            isinstance(move_distance_m, bool)
            or not isinstance(move_distance_m, (int, float))
            or not (
                min(NAVIGATION_MOVE_DISTANCES_M) - _NAVIGATION_DISTANCE_ABS_TOL_M
                <= move_distance_m
                <= max(NAVIGATION_MOVE_DISTANCES_M) + _NAVIGATION_DISTANCE_ABS_TOL_M
            )
        ):
            raise ValueError("model selected an invalid move distance")
        # Normalize millimetre-scale noise only; never round arbitrary distances.
        canonical = next(
            (distance for distance in NAVIGATION_MOVE_DISTANCES_M
             if math.isclose(move_distance_m, distance, rel_tol=0.0,
                             abs_tol=_NAVIGATION_DISTANCE_ABS_TOL_M)),
            None,
        )
        if canonical is None:
            raise ValueError("model selected an invalid move distance")
        move_distance_m = canonical
    if turn_angle_deg is not None:
        # Some compatible providers serialize numeric tool arguments as strings.
        # Convert only JSON numbers; keep the original finite-angle whitelist.
        if isinstance(turn_angle_deg, str):
            try:
                turn_angle_deg = json.loads(turn_angle_deg)
            except (ValueError, RecursionError):
                raise ValueError("model selected an invalid turn angle") from None
        if (
            isinstance(turn_angle_deg, bool)
            or not isinstance(turn_angle_deg, (int, float))
            or float(turn_angle_deg) not in NAVIGATION_TURN_ANGLES_DEG
        ):
            raise ValueError("model selected an invalid turn angle")
        turn_angle_deg = float(turn_angle_deg)
    target_present = "target" in arguments
    description_present = "target_description" in arguments
    ignored_target_fields = 0
    if action not in MARKER_ACTIONS:
        # Some OpenAI-compatible structured-output backends populate optional
        # target fields even for actions that cannot consume a visual target.
        # They have no runtime meaning, so normalize them away. Marker actions
        # remain strict below because their C-label identity is semantic.
        target = None
        target_description = None
        ignored_target_fields = int(target_present) + int(description_present)
    else:
        target = arguments.get("target") if target_present else None
        # A description has no identity semantics without a C-label.  Some
        # OpenAI-compatible providers nevertheless emit this optional field
        # while requesting candidates, so discard only that orphan field and
        # preserve the targetless action as the candidate request.
        target_description = (
            arguments.get("target_description")
            if target_present and description_present
            else None
        )
        if not target_present and description_present:
            ignored_target_fields = 1
        if target_present and (
            isinstance(target, bool)
            or not isinstance(target, int)
            or not 1 <= target <= 999
        ):
            raise ValueError("model selected an invalid target")
        if target_present and description_present:
            _validate_target_description(target_description)
    tool_call_id = tool_call.get("id")
    if tool_call_id is not None and (
        not isinstance(tool_call_id, str) or not tool_call_id
    ):
        raise ValueError("native tool call id is invalid")
    tool_call_extra_content = None
    if "extra_content" in tool_call:
        tool_call_extra_content = _clone_extra_content(tool_call.get("extra_content"))
    message_extra_content = None
    if "extra_content" in message:
        message_extra_content = _clone_extra_content(message.get("extra_content"))
    return NativeToolDecision(
        action=action,
        target=target,
        target_description=target_description,
        reasoning_content=reasoning,
        reasoning_source=reasoning_source,
        tool_call_id=tool_call_id,
        repeat=repeat,
        move_distance_m=move_distance_m,
        turn_angle_deg=turn_angle_deg,
        ignored_target_fields=ignored_target_fields,
        ignored_navigation_fields=ignored_navigation_fields,
        tool_call_extra_content=tool_call_extra_content,
        message_extra_content=message_extra_content,
    )


def parse_native_chat_completion(
    response: Mapping[str, Any],
    *,
    allow_navigation_repeat: bool = True,
    allow_navigation_step_sizes: bool = True,
) -> NativeToolDecision:
    """Validate one OpenAI-compatible native-tool completion."""

    if not isinstance(response, Mapping):
        raise ValueError("native chat completion must be a mapping")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("exactly one native completion choice is required")
    choice = choices[0]
    if (
        not isinstance(choice, Mapping)
        or choice.get("finish_reason") not in {"tool_calls", "stop"}
        or not isinstance(choice.get("message"), Mapping)
    ):
        raise ValueError("native completion did not finish with a tool call")
    return parse_native_tool_message(
        choice["message"],
        allow_navigation_repeat=allow_navigation_repeat,
        allow_navigation_step_sizes=allow_navigation_step_sizes,
    )


def _validate_target_description(value: Any) -> None:
    """Validate without normalizing or rewriting the agent-authored belief."""

    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 64
        or value != value.strip()
        or not value.isprintable()
    ):
        raise ValueError(
            "target_description must be 1-64 printable characters without "
            "leading or trailing whitespace"
        )
