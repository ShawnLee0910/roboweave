import json
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from action20_protocol import (
    BASE_ODOMETRY_ACTIONS,
    MARKER_ACTIONS,
    PUBLIC_TOOL_ACTIONS,
    STRICT_ACTION20,
    TERMINATION_ACTION,
    THREE_CALL_MARKER_ACTIONS,
    TWO_CALL_MARKER_ACTIONS,
)
from agent_common import (
    BLOCKED_OBSERVATION_REMINDER,
    MODEL_OBSERVATION_MAX,
    PHYSICAL_HISTORY_MAX,
    PHYSICAL_HISTORY_OPTIONS,
    json_clone as _json_clone,
)
from agent_observation import (
    PNG_SIGNATURE,
    THREE_VIEW_CAMERAS,
    THREE_VIEW_COMPOSITE_FILENAME,
    PublicImage,
    PublicObservation,
)
from agent_prompt import (
    ACTION20_CORE_INSTRUCTIONS,
    ACTION20_CORE_INSTRUCTIONS_PATH,
    ACTION20_TOOL_SCHEMA,
    ACTION20_TOOL_SCHEMA_PATH,
    COMMON_HARNESS_ADAPTER_INSTRUCTIONS,
    INTERFACE_ROOT,
    SystemPromptRevision,
    load_action20_core_instructions,
    load_action20_tool_schema,
    load_common_system_prompt_revision,
    load_prompt_revision,
    with_automatic_interaction_approach_tool_schema,
    with_overhead_view_tool_schema,
)
from agent_protocol import (
    NativeActionDispatch,
    NativeToolDecision,
    PublicToolResult,
    parse_native_chat_completion,
    parse_native_tool_message,
)
from episode_conversation import (
    PRIVATE_EPISODE_CONTRACT_VERSION,
    PRIVATE_TRANSCRIPT_VERSION,
    EpisodeConversationSupport as _EpisodeConversationSupport,
    ObservationKind,
    PrivateEpisodeContract,
)


# Multi-turn protocol: assistant -> tool -> observation -> assistant.
class EpisodeConversation(_EpisodeConversationSupport):
    """The readable multi-turn control path for one Action20 episode."""

    def append_assistant_completion(
        self,
        response: Mapping[str, Any],
        *,
        remaining_action_decisions: int | None = None,
        provider_response: Mapping[str, Any] | None = None,
    ) -> NativeActionDispatch:
        """Append one native Assistant tool call and consume old Markers."""

        self._require_state("awaiting_assistant")
        if provider_response is not None and not isinstance(
            provider_response, Mapping
        ):
            raise ValueError("provider response must be a mapping")
        decision = parse_native_chat_completion(
            response,
            allow_navigation_repeat=self._navigation_repeat_enabled,
            allow_navigation_step_sizes=self._navigation_step_sizes_enabled,
        )
        if (
            decision.action in MARKER_ACTIONS
            and decision.target is not None
            and self._pending_marker_action is None
        ):
            # No C-label namespace exists yet, so an eager provider-supplied
            # target cannot identify anything. Safely downgrade it to the
            # targetless request that creates the current action's namespace.
            decision = decision.without_target()
        cancelled_marker_action = (
            self._pending_marker_action
            if self._pending_marker_action is not None
            and decision.target is None
            else None
        )
        if (
            cancelled_marker_action is not None
            and remaining_action_decisions is not None
            and remaining_action_decisions < 2
        ):
            raise ValueError(
                "marker replacement requires two remaining action decisions"
            )
        runtime_action = decision.runtime_action(
            pending_action=self._pending_marker_action,
            pending_targets=self._pending_marker_targets,
        )
        selected_target_evidence: dict[str, Any] | None = None
        if decision.action in MARKER_ACTIONS:
            if decision.target is None:
                inflight_marker_stage = 0
            else:
                if self._pending_marker_stage not in {1, 2}:
                    raise RuntimeError("selected Marker has no pending stage")
                inflight_marker_stage = self._pending_marker_stage
                selected_target_evidence = (
                    self._selected_target_visual_evidence_record(
                        decision,
                        stage=inflight_marker_stage,
                    )
                )
        else:
            inflight_marker_stage = None
        assistant_event = {
            "event_index": len(self._events),
            "kind": "assistant_tool_call",
            "role": "assistant",
            "message": decision.assistant_message(),
            "native_reasoning_content": decision.reasoning_content,
            "native_reasoning_source": decision.reasoning_source,
            "raw_response": _json_clone(
                response if provider_response is None else provider_response
            ),
        }
        if selected_target_evidence is not None:
            assistant_event["selected_target_visual_evidence_private"] = (
                selected_target_evidence
            )
        if cancelled_marker_action is not None:
            cancelled: dict[str, Any] = {
                "action": cancelled_marker_action,
                "stage": self._pending_marker_stage,
            }
            if self._held_object_description is not None:
                cancelled["held_object_description"] = (
                    self._held_object_description
                )
            assistant_event["cancelled_marker_selection"] = cancelled
        self._events.append(assistant_event)
        self._pending_marker_action = None
        self._pending_marker_targets = frozenset()
        self._pending_marker_stage = None
        self._inflight_marker_stage = inflight_marker_stage
        self._pending_decision = decision
        self._pending_result = None
        self._pending_observations_remaining = 0
        self._state = "awaiting_tool_result"
        return NativeActionDispatch(
            decision=decision,
            runtime_action=MappingProxyType(runtime_action),
        )

    def append_tool_result(self, result: PublicToolResult) -> None:
        """Append the text-only Tool response for the pending function call."""

        self._require_state("awaiting_tool_result")
        assert self._pending_decision is not None
        action = self._pending_decision.action
        requested_repeat = self._pending_decision.public_repeat
        if requested_repeat > 1:
            if (
                result.steps is None
                or result.repeat_requested != requested_repeat
            ):
                raise ValueError("navigation repeat result changed")
        elif result.steps is not None:
            raise ValueError("single action cannot expose repeated-step results")
        if (
            result.status == "selection_required"
            and result.stage
            != self._expected_selection_stage_for_pending_decision()
        ):
            raise ValueError(
                "selection_required does not match the in-flight Marker stage"
            )
        if (
            action in BASE_ODOMETRY_ACTIONS
            and result.steps is None
            and result.body_odometry is None
        ):
            raise ValueError("base actions require public body_odometry")
        if action not in BASE_ODOMETRY_ACTIONS and (
            result.body_odometry is not None or result.steps is not None
        ):
            raise ValueError("non-base actions cannot expose body_odometry")
        if result.blocked_detail is not None and action not in {"open", "close"}:
            raise ValueError("articulated blocked detail requires open or close")
        assistant_event = self._events[-1]
        evidence = assistant_event.get(
            "selected_target_visual_evidence_private"
        )
        if evidence is not None:
            evidence["status"] = result.status
            if result.blocked_category is not None:
                evidence["blocked_category"] = result.blocked_category
            if result.blocked_detail is not None:
                evidence["blocked_detail"] = result.blocked_detail
        self._update_agent_authored_object_memory(result)
        if evidence is not None:
            evidence["held_object_description_after"] = (
                self._held_object_description
            )
        content = json.dumps(
            result.to_public_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self._events.append(
            {
                "event_index": len(self._events),
                "kind": "tool_result",
                "role": "tool",
                "action": action,
                "message": {
                    "role": "tool",
                    "tool_call_id": self._pending_decision.tool_call_id,
                    "content": content,
                },
            }
        )
        self._pending_result = result
        self._pending_observations_remaining = result.repeat_executed or 1
        self._state = "awaiting_observation"

    def append_observation(
        self,
        observation: PublicObservation,
        *,
        kind: ObservationKind,
        marker_targets: Iterable[int] = (),
    ) -> None:
        """Append a physical, Marker, or other transient current observation."""

        self._require_state("awaiting_observation")
        if observation.observation_id in self._seen_observation_ids:
            raise ValueError("observation_id must be unique")
        if kind not in {"physical", "marker", "transient"}:
            raise ValueError("observation kind changed")
        targets = self._normalize_marker_targets(marker_targets)
        assert self._pending_decision is not None
        assert self._pending_result is not None
        repeat_step_index: int | None = None
        if self._pending_result.steps is not None:
            if kind != "physical" or targets:
                raise ValueError("navigation repeat requires physical observations")
            repeat_step_index = (
                self._pending_result.repeat_executed
                - self._pending_observations_remaining
                + 1
            )
        next_marker_stage = self._next_marker_stage_for_observation(kind)
        if kind == "marker":
            if not targets:
                raise ValueError("Marker observation requires candidate targets")
            self._display_current = observation
            self._display_current_kind = kind
            self._pending_marker_action = self._pending_decision.action
            self._pending_marker_targets = targets
            self._pending_marker_stage = next_marker_stage
        else:
            if targets:
                raise ValueError(
                    f"{kind} observation cannot carry Marker targets"
                )
            if kind == "physical":
                self._physical_history.append(self._physical_current)
                self._physical_history = self._physical_history[
                    -self._physical_history_max:
                ]
                self._physical_current = observation
            self._display_current = observation
            self._display_current_kind = kind
            self._pending_marker_action = None
            self._pending_marker_targets = frozenset()
            self._pending_marker_stage = None
        self._seen_observation_ids.add(observation.observation_id)
        self._events.append(
            self._observation_event(
                observation,
                kind,
                targets,
                marker_stage=next_marker_stage,
                repeat_step_index=repeat_step_index,
                repeat_step_count=self._pending_result.repeat_executed,
            )
        )
        self._pending_observations_remaining -= 1
        if self._pending_observations_remaining:
            return
        self._inflight_marker_stage = None
        self._pending_decision = None
        self._pending_result = None
        self._state = "awaiting_assistant"

    def finish_after_done(self) -> None:
        """Finish after the model sends a completed ``done`` action."""

        self._require_state("awaiting_observation")
        if (
            self._pending_decision is None
            or self._pending_decision.action != TERMINATION_ACTION
            or self._pending_result is None
            or self._pending_result.status != "completed"
        ):
            raise RuntimeError("only a completed done call can finish the episode")
        self._finish_episode(termination_action=TERMINATION_ACTION)

    def finish_after_official_success(self) -> None:
        """Finish when the simulator's official verifier succeeds."""

        self._require_state("awaiting_assistant")
        self._finish_episode(termination_source="official_task_success")

    def finish_after_consecutive_blocked_limit(self) -> None:
        """Finish when the client-side blocked-streak limit is reached."""

        self._require_state("awaiting_assistant")
        if self.consecutive_blocked_macro_actions <= 0:
            raise RuntimeError("blocked-streak termination has no blocked action")
        self._finish_episode(
            termination_source="client_policy",
            termination_reason="consecutive_blocked_limit",
        )

    def build_native_request(
        self,
        *,
        preserve_reasoning: bool = False,
    ) -> dict[str, Any]:
        """Build one complete OpenAI-compatible native-tool request."""

        tool_schema = _json_clone(self._tool_schema)
        messages = self.build_messages(preserve_reasoning=preserve_reasoning)
        image_count = self._count_image_urls(messages)
        if (
            image_count != len(self.active_observation_ids)
            or not 1 <= image_count <= self._physical_history_max + 1
        ):
            raise RuntimeError(
                "model visual window is not one composite per observation"
            )
        return {
            "model": self._contract.model,
            "messages": messages,
            "tools": [tool_schema],
            "tool_choice": {
                "type": "function",
                "function": {"name": "robot_action"},
            },
            "parallel_tool_calls": False,
        }


# Backward-compatible public imports for existing ``from Agent import ...`` callers.
__all__ = [
    "ACTION20_CORE_INSTRUCTIONS",
    "ACTION20_CORE_INSTRUCTIONS_PATH",
    "ACTION20_TOOL_SCHEMA",
    "ACTION20_TOOL_SCHEMA_PATH",
    "BASE_ODOMETRY_ACTIONS",
    "BLOCKED_OBSERVATION_REMINDER",
    "COMMON_HARNESS_ADAPTER_INSTRUCTIONS",
    "EpisodeConversation",
    "INTERFACE_ROOT",
    "MARKER_ACTIONS",
    "MODEL_OBSERVATION_MAX",
    "NativeActionDispatch",
    "NativeToolDecision",
    "PHYSICAL_HISTORY_MAX",
    "PHYSICAL_HISTORY_OPTIONS",
    "PNG_SIGNATURE",
    "PRIVATE_EPISODE_CONTRACT_VERSION",
    "PRIVATE_TRANSCRIPT_VERSION",
    "PUBLIC_TOOL_ACTIONS",
    "PrivateEpisodeContract",
    "PublicImage",
    "PublicObservation",
    "PublicToolResult",
    "STRICT_ACTION20",
    "SystemPromptRevision",
    "TERMINATION_ACTION",
    "THREE_CALL_MARKER_ACTIONS",
    "THREE_VIEW_CAMERAS",
    "THREE_VIEW_COMPOSITE_FILENAME",
    "TWO_CALL_MARKER_ACTIONS",
    "load_action20_core_instructions",
    "load_action20_tool_schema",
    "load_common_system_prompt_revision",
    "load_prompt_revision",
    "with_automatic_interaction_approach_tool_schema",
    "with_overhead_view_tool_schema",
    "parse_native_chat_completion",
    "parse_native_tool_message",
]
