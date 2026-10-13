"""Action20 episode state, history window, and model context."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping

from action20_protocol import (
    BASE_ODOMETRY_ACTIONS,
    MARKER_ACTIONS,
    TERMINATION_ACTION,
    THREE_CALL_MARKER_ACTIONS,
    TWO_CALL_MARKER_ACTIONS,
)
from agent_artifacts import build_chatml_artifact
from agent_common import (
    BLOCKED_OBSERVATION_REMINDER,
    PHYSICAL_HISTORY_MAX,
)
from agent_common import json_clone as _json_clone
from agent_common import validate_physical_history_max as _validate_physical_history_max
from agent_observation import (
    PublicObservation,
    model_visible_marker_depth_text,
)
from agent_prompt import (
    load_action20_tool_schema,
    load_common_system_prompt_revision,
    SystemPromptRevision,
)
from agent_protocol import (
    NativeActionDispatch,
    NativeToolDecision,
    PublicToolResult,
    replay_native_reasoning,
)
from agent_resume import (
    PRIVATE_TRANSCRIPT_VERSION,
    restore_episode_conversation,
)


def _transcript_with_image_paths(
    transcript: Mapping[str, Any],
    *,
    artifact_root: Path,
    relative: bool,
) -> dict[str, Any]:
    value = _json_clone(transcript)
    root = Path(artifact_root).resolve()
    for event in value.get("events", ()):
        if not isinstance(event, dict):
            continue
        observations = [event.get("observation")]
        evidence = event.get("selected_target_visual_evidence_private")
        if isinstance(evidence, dict):
            marker_frame = evidence.get("marker_frame")
            if isinstance(marker_frame, dict):
                observations.append(marker_frame.get("observation"))
        for observation in observations:
            images = (
                observation.get("images")
                if isinstance(observation, dict)
                else None
            )
            observation_id = (
                observation.get("observation_id")
                if isinstance(observation, dict)
                else None
            )
            if not isinstance(images, list):
                continue
            for image in images:
                if not isinstance(image, dict) or not isinstance(
                    image.get("path"), str
                ):
                    continue
                path = Path(image["path"])
                resolved = (
                    path.resolve()
                    if path.is_absolute()
                    else (root / path).resolve()
                )
                if (
                    not relative
                    and path.is_absolute()
                    and not resolved.is_relative_to(root)
                    and isinstance(observation_id, str)
                    and path.parts[-3:]
                    == ("observations", observation_id, path.name)
                ):
                    relocated = (
                        root / "observations" / observation_id / path.name
                    ).resolve()
                    if relocated.is_file():
                        resolved = relocated
                if relative:
                    try:
                        image["path"] = resolved.relative_to(root).as_posix()
                    except ValueError as exc:
                        raise ValueError(
                            "transcript image escaped its artifact root"
                        ) from exc
                else:
                    image["path"] = str(resolved)
    return value


PRIVATE_EPISODE_CONTRACT_VERSION = (
    "behavior-hl-action20-private-episode-contract-v3"
)
ObservationKind = Literal["physical", "marker", "transient"]
ConversationState = Literal[
    "awaiting_assistant",
    "awaiting_tool_result",
    "awaiting_observation",
    "terminal",
]


@dataclass(frozen=True)
class PrivateEpisodeContract:
    """Frozen backend identity; only ``instruction`` is model-visible."""

    episode_id: str
    task_id: str
    split: str
    instance_id: str
    seed: int
    model: str
    project_root: Path
    instruction: str
    physical_history_max: int = PHYSICAL_HISTORY_MAX

    def __post_init__(self) -> None:
        for field in (
            "episode_id",
            "task_id",
            "split",
            "instance_id",
            "model",
            "instruction",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a nonempty string")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= 0xFFFF_FFFF
        ):
            raise ValueError("seed must be a canonical uint32")
        _validate_physical_history_max(self.physical_history_max)
        root = Path(self.project_root).resolve()
        object.__setattr__(self, "project_root", root)

    def artifact_record(self) -> dict[str, Any]:
        return {
            "contract_version": PRIVATE_EPISODE_CONTRACT_VERSION,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "split": self.split,
            "instance_id": self.instance_id,
            "seed": self.seed,
            "model": self.model,
            "project_root": str(self.project_root),
            "instruction": self.instruction,
            "physical_history_max": self.physical_history_max,
        }


class EpisodeConversationSupport:
    """Stateful Action20 conversation with a bounded active visual window."""

    def __init__(
        self,
        contract: PrivateEpisodeContract,
        initial_observation: PublicObservation,
        *,
        system_prompt_revision: SystemPromptRevision | None = None,
        tool_schema: Mapping[str, Any] | None = None,
    ) -> None:
        resolved_prompt = (
            load_common_system_prompt_revision(physical_history_max=contract.physical_history_max)
            if system_prompt_revision is None
            else system_prompt_revision
        )
        resolved_schema = _json_clone(
            load_action20_tool_schema() if tool_schema is None else tool_schema
        )
        properties = resolved_schema["function"]["parameters"]["properties"]
        if frozenset(properties) not in {
            frozenset({"action", "target", "target_description"}),
            frozenset({"action", "repeat", "target", "target_description"}),
            frozenset(
                {
                    "action",
                    "move_distance_m",
                    "repeat",
                    "target",
                    "target_description",
                    "turn_angle_deg",
                }
            ),
        }:
            raise ValueError("Action20 tool schema parameter contract changed")
        self._contract = contract
        self._physical_history_max = contract.physical_history_max
        self._system_prompt_revision = resolved_prompt
        self._tool_schema = resolved_schema
        self._navigation_repeat_enabled = "repeat" in properties
        self._navigation_step_sizes_enabled = {
            "move_distance_m",
            "turn_angle_deg",
        }.issubset(properties)
        self._transcript_version = PRIVATE_TRANSCRIPT_VERSION
        self._state: ConversationState = "awaiting_assistant"
        self._events: list[dict[str, Any]] = [
            {
                "event_index": 0,
                "kind": "task",
                "role": "user",
                "instruction": contract.instruction,
            }
        ]
        self._events.append(
            self._observation_event(initial_observation, "physical", ())
        )
        self._seen_observation_ids = {initial_observation.observation_id}
        self._physical_history: list[PublicObservation] = []
        self._physical_current = initial_observation
        self._display_current = initial_observation
        self._display_current_kind: ObservationKind = "physical"
        self._pending_marker_action: str | None = None
        self._pending_marker_targets: frozenset[int] = frozenset()
        self._pending_marker_stage: int | None = None
        self._inflight_marker_stage: int | None = None
        self._held_object_description: str | None = None
        self._pending_decision: NativeToolDecision | None = None
        self._pending_result: PublicToolResult | None = None
        self._pending_observations_remaining = 0

    @classmethod
    def from_artifact_transcript(
        cls,
        transcript: Mapping[str, Any],
        *,
        expected_contract: PrivateEpisodeContract,
        system_prompt_revision: SystemPromptRevision,
        artifact_root: Path,
        tool_schema: Mapping[str, Any] | None = None,
    ) -> "EpisodeConversation":
        """Restore one committed conversation without touching the simulator."""

        normalized = transcript
        if isinstance(transcript, Mapping):
            normalized = _transcript_with_image_paths(
                transcript,
                artifact_root=artifact_root,
                relative=False,
            )
        return restore_episode_conversation(
            cls,
            normalized,
            expected_contract=expected_contract,
            system_prompt_revision=system_prompt_revision,
            artifact_root=artifact_root,
            tool_schema=tool_schema,
        )

    @property
    def state(self) -> ConversationState:
        return self._state

    @property
    def pending_navigation_repeat_dispatch(self) -> NativeActionDispatch:
        self._require_state("awaiting_tool_result")
        assert self._pending_decision is not None
        if (
            self._pending_decision.action not in BASE_ODOMETRY_ACTIONS
            or self._pending_decision.public_repeat <= 1
        ):
            raise RuntimeError("pending action is not a navigation repeat")
        return NativeActionDispatch(
            decision=self._pending_decision,
            runtime_action=MappingProxyType(
                self._pending_decision.runtime_action()
            ),
        )

    @property
    def active_observation_ids(self) -> tuple[str, ...]:
        if self._display_current_kind in {"marker", "transient"}:
            observations = [
                *self._physical_history[-(self._physical_history_max - 1) :],
                self._physical_current,
                self._display_current,
            ]
        else:
            observations = [*self._physical_history, self._display_current]
        return tuple(observation.observation_id for observation in observations)

    @property
    def pending_marker_action(self) -> str | None:
        return self._pending_marker_action

    @property
    def pending_marker_targets(self) -> frozenset[int]:
        return self._pending_marker_targets

    @property
    def pending_marker_stage(self) -> int | None:
        return self._pending_marker_stage

    @property
    def expected_selection_stage(self) -> int | None:
        """Return the next Marker stage for the in-flight public call."""

        if self._state != "awaiting_tool_result":
            return None
        return self._expected_selection_stage_for_pending_decision()

    @property
    def macro_action_count(self) -> int:
        """Count resolved semantic actions, excluding Marker micro-steps."""

        return len(self._resolved_macro_statuses())

    @property
    def consecutive_blocked_macro_actions(self) -> int:
        """Count the current suffix of fully resolved blocked macro actions."""

        count = 0
        for status in reversed(self._resolved_macro_statuses()):
            if status != "blocked":
                break
            count += 1
        return count

    @property
    def maximum_consecutive_blocked_macro_actions(self) -> int:
        """Return the largest fully resolved blocked streak in this episode."""

        maximum = 0
        current = 0
        for status in self._resolved_macro_statuses():
            if status == "blocked":
                current += 1
                maximum = max(maximum, current)
            else:
                current = 0
        return maximum

    def _resolved_macro_statuses(self) -> tuple[str, ...]:
        """Return resolved semantic statuses without Marker micro-steps."""

        statuses: list[str] = []
        for event in self._events:
            if event.get("kind") == "assistant_tool_call":
                if "cancelled_marker_selection" in event:
                    statuses.append("selection_cancelled")
                continue
            if event.get("kind") != "tool_result":
                continue
            action = event.get("action")
            if action == TERMINATION_ACTION:
                continue
            result = self._public_result_from_tool_event(event)
            status = result.get("status")
            if status in {"completed", "blocked"}:
                statuses.append(status)
        return tuple(statuses)

    def _selected_target_visual_evidence_record(
        self,
        decision: NativeToolDecision,
        *,
        stage: int,
    ) -> dict[str, Any]:
        """Freeze the public Marker frame behind one agent-authored selection."""

        if (
            decision.target is None
            or self._display_current_kind != "marker"
            or self._pending_marker_stage != stage
            or stage not in {1, 2}
        ):
            raise RuntimeError("selected-target evidence has no current Marker frame")
        record: dict[str, Any] = {
            "schema_version": (
                "behavior-hl-agent-selected-target-visual-evidence-private/v2"
            ),
            "action": decision.action,
            "stage": stage,
            "selected_marker": f"C{decision.target}",
            "target_description_provided": (
                decision.target_description is not None
            ),
            "status": "pending",
            "held_object_description_before": self._held_object_description,
            "marker_frame": {
                "observation": self._display_current.artifact_record(),
            },
        }
        if decision.target_description is not None:
            record["target_description"] = decision.target_description
        return record

    def _update_agent_authored_object_memory(
        self,
        result: PublicToolResult,
    ) -> None:
        """Persist only the model's own selected-object wording."""

        assert self._pending_decision is not None
        if result.status != "completed":
            return
        decision = self._pending_decision
        if decision.action == "pick" and decision.target is not None:
            self._held_object_description = decision.target_description
            return
        if decision.action == "place_on_floor":
            self._held_object_description = None
            return
        final_placement = (
            decision.target is not None
            and (
                decision.action in {"place_on_top", "place_under"}
                or (
                    decision.action in THREE_CALL_MARKER_ACTIONS
                    and self._inflight_marker_stage == 2
                )
            )
        )
        if final_placement:
            self._held_object_description = None

    @staticmethod
    def _semantic_selection_history_message(
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Render an expired C-label selection without retaining its pointer."""

        status = evidence.get("status")
        if status not in {"completed", "blocked", "selection_required"}:
            raise RuntimeError("selected-target semantic history is unfinished")
        semantic_event: dict[str, Any] = {
            "action": evidence.get("action"),
            "stage": evidence.get("stage"),
            "status": status,
        }
        description = evidence.get("target_description")
        if isinstance(description, str):
            semantic_event["target_description"] = description
        held = evidence.get("held_object_description_before")
        if isinstance(held, str):
            semantic_event["held_object_description"] = held
        blocked_category = evidence.get("blocked_category")
        if isinstance(blocked_category, str):
            semantic_event["blocked_category"] = blocked_category
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Previous selected-target action: "
                        + json.dumps(
                            semantic_event,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    ),
                }
            ],
        }

    @staticmethod
    def _public_result_from_tool_event(
        event: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return json.loads(event["message"]["content"])

    @staticmethod
    def _assistant_arguments_from_event(
        event: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        call = event["message"]["tool_calls"][0]
        return json.loads(call["function"]["arguments"])

    def _closed_marker_transaction(
        self,
        events: list[dict[str, Any]],
        start_index: int,
    ) -> tuple[int, dict[str, Any]] | None:
        """Collapse one resolved Marker micro-protocol for model history."""

        arguments = self._assistant_arguments_from_event(events[start_index])
        action = arguments.get("action")
        if action not in MARKER_ACTIONS or "target" in arguments:
            return None
        if start_index + 3 >= len(events):
            return None
        first_stage_result = self._public_result_from_tool_event(
            events[start_index + 1]
        )
        if (
            first_stage_result.get("stage") != 1
            or first_stage_result.get("status") != "selection_required"
        ):
            return None

        first_followup = events[start_index + 3]
        cancellation = first_followup.get("cancelled_marker_selection")
        final_status: str | None = None
        final_result: Mapping[str, Any] | None = None
        if (
            isinstance(cancellation, Mapping)
            and cancellation.get("action") == action
            and cancellation.get("stage") == 1
        ):
            final_status = "selection_cancelled"
            final_observation_index = start_index + 3
            first_evidence = None
            evidences: tuple[Mapping[str, Any], ...] = ()
        else:
            first_evidence = first_followup.get(
                "selected_target_visual_evidence_private"
            )
            if not isinstance(first_evidence, Mapping):
                return None

            if action in TWO_CALL_MARKER_ACTIONS:
                final_observation_index = start_index + 5
                if final_observation_index >= len(events):
                    return None
                final_result_index = start_index + 4
                evidences = (first_evidence,)
            else:
                if start_index + 6 >= len(events):
                    return None
                second_stage_result = self._public_result_from_tool_event(
                    events[start_index + 4]
                )
                if (
                    second_stage_result.get("stage") != 2
                    or second_stage_result.get("status")
                    != "selection_required"
                ):
                    return None
                second_followup = events[start_index + 6]
                cancellation = second_followup.get(
                    "cancelled_marker_selection"
                )
                if (
                    isinstance(cancellation, Mapping)
                    and cancellation.get("action") == action
                    and cancellation.get("stage") == 2
                ):
                    final_status = "selection_cancelled"
                    final_observation_index = start_index + 6
                    evidences = (first_evidence,)
                else:
                    second_evidence = second_followup.get(
                        "selected_target_visual_evidence_private"
                    )
                    if not isinstance(second_evidence, Mapping):
                        return None
                    final_observation_index = start_index + 8
                    if final_observation_index >= len(events):
                        return None
                    final_result_index = start_index + 7
                    evidences = (first_evidence, second_evidence)

            if final_status is None:
                final_result = self._public_result_from_tool_event(
                    events[final_result_index]
                )
                final_status = final_result.get("status")
                if final_status not in {"completed", "blocked"}:
                    return None

        semantic_event: dict[str, Any] = {
            "action": action,
            "status": final_status,
        }
        if final_result is not None:
            remaining = final_result.get("action_decisions_remaining")
            if isinstance(remaining, int) and not isinstance(remaining, bool):
                semantic_event["action_decisions_remaining"] = remaining
        if isinstance(cancellation, Mapping):
            held = cancellation.get("held_object_description")
        elif first_evidence is not None:
            held = first_evidence.get("held_object_description_before")
        else:
            held = None
        if isinstance(held, str):
            semantic_event["held_object_description"] = held
        if first_evidence is not None:
            first_description = first_evidence.get("target_description")
            if isinstance(first_description, str):
                semantic_event["target_description"] = first_description
        if len(evidences) == 2:
            second_description = evidences[1].get("target_description")
            if isinstance(second_description, str):
                semantic_event["placement_description"] = second_description
        if evidences:
            blocked_category = evidences[-1].get("blocked_category")
            if isinstance(blocked_category, str):
                semantic_event["blocked_category"] = blocked_category
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Previous manipulation outcome: "
                        + json.dumps(
                            semantic_event,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    ),
                }
            ],
        }
        return final_observation_index, message

    def build_messages(
        self,
        *,
        preserve_reasoning: bool = False,
    ) -> list[dict[str, Any]]:
        """Build bounded images plus semantic long-term selected-target history."""

        self._require_state("awaiting_assistant")
        active_ids = frozenset(self.active_observation_ids)
        current_id = self._display_current.observation_id
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self._system_prompt_revision.text,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Task: {self._contract.instruction}",
                    }
                ],
            },
        ]
        events = self._events[1:]
        index = 0
        while index < len(events):
            event = events[index]
            kind = event["kind"]
            if kind == "observation":
                observation_id = event["observation"]["observation_id"]
                if observation_id not in active_ids:
                    index += 1
                    continue
                observation = self._observation_by_id(observation_id)
                blocked_current = False
                if observation_id == current_id and index > 0:
                    result_offset = event.get("repeat_step_index", 1)
                    previous_index = index - result_offset
                    if (
                        previous_index >= 0
                        and events[previous_index]["kind"] == "tool_result"
                    ):
                        blocked_current = (
                            self._public_result_from_tool_event(
                                events[previous_index]
                            ).get("status")
                            == "blocked"
                        )
                messages.append(
                    self._render_observation_message(
                        observation,
                        current=observation_id == current_id,
                        marker=event["observation_kind"] == "marker",
                        blocked_current=blocked_current,
                        repeat_step_index=event.get("repeat_step_index"),
                        repeat_step_count=event.get("repeat_step_count"),
                    )
                )
            elif kind == "assistant_tool_call":
                closed_transaction = self._closed_marker_transaction(
                    events,
                    index,
                )
                if closed_transaction is not None:
                    final_observation_index, semantic_message = (
                        closed_transaction
                    )
                    messages.append(semantic_message)
                    index = final_observation_index
                    continue
                evidence = event.get(
                    "selected_target_visual_evidence_private"
                )
                if evidence is not None:
                    marker_id = evidence["marker_frame"]["observation"][
                        "observation_id"
                    ]
                    if marker_id not in active_ids:
                        messages.append(
                            self._semantic_selection_history_message(evidence)
                        )
                        index += 2
                        continue
                assistant_message = _json_clone(event["message"])
                reasoning = event.get("native_reasoning_content")
                if preserve_reasoning and reasoning:
                    replay_native_reasoning(
                        assistant_message,
                        reasoning,
                        event.get("native_reasoning_source"),
                    )
                messages.append(assistant_message)
            elif kind == "tool_result":
                messages.append(_json_clone(event["message"]))
            index += 1
        return messages

    def artifact_transcript(
        self,
        *,
        artifact_root: Path | None = None,
    ) -> dict[str, Any]:
        """Return the complete, unpruned private audit record."""

        transcript = {
            "transcript_version": self._transcript_version,
            "private_episode_contract": self._contract.artifact_record(),
            "system_prompt_revision": (
                self._system_prompt_revision.artifact_record()
            ),
            "conversation_state": self._state,
            "visual_history": {
                "completed_physical_history_max": self._physical_history_max,
                "model_observation_max": self._physical_history_max + 1,
            },
            "active_observation_ids": list(self.active_observation_ids),
            "pending_marker": {
                "action": self._pending_marker_action,
                "targets": sorted(self._pending_marker_targets),
                "stage": self._pending_marker_stage,
            },
            "inflight_marker_stage": self._inflight_marker_stage,
            "agent_authored_state": {
                "held_object_description": self._held_object_description,
            },
            "events": _json_clone(self._events),
        }
        if artifact_root is None:
            return transcript
        return _transcript_with_image_paths(
            transcript,
            artifact_root=artifact_root,
            relative=True,
        )

    def artifact_selected_target_visual_evidence(self) -> dict[str, Any]:
        """Return additive private evidence without exposing benchmark truth."""

        records = [
            _json_clone(event["selected_target_visual_evidence_private"])
            for event in self._events
            if "selected_target_visual_evidence_private" in event
        ]
        return {
            "schema_version": (
                "behavior-hl-agent-selected-target-visual-evidence-private/v2"
            ),
            "visibility": "PRIVATE_AUDIT_ONLY",
            "identity_source": "selected_marker_frame",
            "target_description_optional": True,
            "ground_truth_semantics_included": False,
            "selected_target_crop_provided_to_model": False,
            "record_count": len(records),
            "records": records,
        }

    def artifact_chatml(self, *, artifact_root: Path) -> dict[str, Any]:
        """Return a readable ChatML view derived from the private ledger.

        This is an additive raw chronological artifact for inspection or
        downstream conversion.  It does not replace the native
        OpenAI-compatible request manifests or their semantic long-history
        rendering.
        """

        return build_chatml_artifact(
            self.artifact_transcript(artifact_root=artifact_root),
            system_prompt=self._system_prompt_revision.text,
            artifact_root=artifact_root,
        )

    def _observation_event(
        self,
        observation: PublicObservation,
        kind: ObservationKind,
        targets: Iterable[int],
        *,
        marker_stage: int | None = None,
        repeat_step_index: int | None = None,
        repeat_step_count: int | None = None,
    ) -> dict[str, Any]:
        event = {
            "event_index": len(self._events),
            "kind": "observation",
            "role": "user",
            "observation_kind": kind,
            "marker_targets": sorted(targets),
            "marker_stage": marker_stage,
            "observation": observation.artifact_record(),
        }
        if repeat_step_index is not None:
            event["repeat_step_index"] = repeat_step_index
            event["repeat_step_count"] = repeat_step_count
        return event

    def _next_marker_stage_for_observation(
        self,
        kind: ObservationKind,
    ) -> int | None:
        """Validate the exact Action20 Marker transition for one observation."""

        assert self._pending_decision is not None
        assert self._pending_result is not None
        decision = self._pending_decision
        result = self._pending_result
        if result.status == "blocked":
            if kind == "marker":
                raise ValueError("blocked action cannot return Marker candidates")
            return None
        if result.status == "selection_required":
            if kind != "marker":
                raise ValueError(
                    "selection_required must return a Marker observation"
                )
            return result.stage
        if kind == "marker":
            raise ValueError(
                "only selection_required may return Marker candidates"
            )
        if decision.action not in MARKER_ACTIONS:
            return None
        if decision.target is None:
            if self._inflight_marker_stage != 0:
                raise RuntimeError("Marker request stage identity changed")
            raise ValueError(
                "targetless Marker request cannot complete without candidates"
            )
        if decision.action in TWO_CALL_MARKER_ACTIONS:
            if self._inflight_marker_stage != 1:
                raise ValueError("two-call Marker selection has invalid stage")
            return None
        if decision.action in THREE_CALL_MARKER_ACTIONS:
            if self._inflight_marker_stage == 1:
                raise ValueError(
                    "three-call intermediate selection cannot complete without "
                    "fresh candidates"
                )
            if self._inflight_marker_stage == 2:
                return None
            raise ValueError("three-call Marker selection has invalid stage")
        raise RuntimeError("Marker action has no protocol depth")

    def _expected_selection_stage_for_pending_decision(self) -> int | None:
        """Resolve the only valid Marker stage for the pending decision."""

        assert self._pending_decision is not None
        decision = self._pending_decision
        if decision.action not in MARKER_ACTIONS:
            return None
        if decision.target is None:
            if self._inflight_marker_stage != 0:
                raise RuntimeError("Marker request stage identity changed")
            return 1
        if (
            decision.action in THREE_CALL_MARKER_ACTIONS
            and self._inflight_marker_stage == 1
        ):
            return 2
        return None

    def _observation_by_id(self, observation_id: str) -> PublicObservation:
        observations = (
            *self._physical_history,
            self._physical_current,
            self._display_current,
        )
        for observation in observations:
            if observation.observation_id == observation_id:
                return observation
        raise RuntimeError("active observation is missing from artifact ledger")

    @staticmethod
    def _render_observation_message(
        observation: PublicObservation,
        *,
        current: bool,
        marker: bool,
        blocked_current: bool,
        repeat_step_index: int | None = None,
        repeat_step_count: int | None = None,
    ) -> dict[str, Any]:
        if current and marker:
            label = "Current onboard observation with temporary C-labels."
        elif current:
            label = "Current onboard observation."
        else:
            label = "Historical completed physical observation."
        if repeat_step_index is not None:
            label += (
                f" Repeat step {repeat_step_index} of {repeat_step_count}."
            )
        if blocked_current:
            label += "\n\n" + BLOCKED_OBSERVATION_REMINDER
        if marker and observation.marker_depth is not None:
            label += "\n\n" + model_visible_marker_depth_text(
                observation.marker_depth
            )
        png = observation.read_model_composite_png()
        content: list[dict[str, Any]] = [
            {"type": "text", "text": label},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,"
                    + base64.b64encode(png).decode("ascii")
                },
            },
        ]
        return {"role": "user", "content": content}

    def _finish_episode(self, **fields: Any) -> None:
        self._events.append(
            {
                "event_index": len(self._events),
                "kind": "episode_end",
                **fields,
            }
        )
        self._inflight_marker_stage = None
        self._pending_marker_action = None
        self._pending_marker_targets = frozenset()
        self._pending_marker_stage = None
        self._pending_decision = None
        self._pending_result = None
        self._pending_observations_remaining = 0
        self._state = "terminal"

    def _require_state(self, expected: ConversationState) -> None:
        if self._state != expected:
            raise RuntimeError(
                f"conversation is {self._state}; expected {expected}"
            )

    @staticmethod
    def _normalize_marker_targets(values: Iterable[int]) -> frozenset[int]:
        try:
            normalized = frozenset(values)
        except TypeError as exc:
            raise ValueError(
                "Marker targets must be an iterable of integers"
            ) from exc
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= 999
            for value in normalized
        ):
            raise ValueError("Marker targets must be integers from 1 to 999")
        return normalized

    @staticmethod
    def _count_image_urls(messages: Iterable[Mapping[str, Any]]) -> int:
        count = 0
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, Mapping) and item.get("type") == "image_url":
                    count += 1
        return count
