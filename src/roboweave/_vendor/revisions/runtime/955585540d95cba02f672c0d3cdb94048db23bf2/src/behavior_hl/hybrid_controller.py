"""Fail-closed checkpoint controller for hybrid BEHAVIOR task-and-QA episodes.

The module deliberately depends on the hard-trajectory manifest structurally:
an object with a ``phases`` attribute can be passed as ``trajectory``.  This
keeps simulator grounding and the declarative trajectory catalogue separate.
Ground truth is frozen by the evaluator-side grounder and is never returned by
any public event or summary.
"""

from __future__ import annotations

import hashlib
import json
import random
import secrets
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Protocol

from behavior_hl.actions import HighLevelAction
from behavior_hl.controller import EpisodeController, EpisodeStep


class HybridControllerError(RuntimeError):
    """Base class for hybrid episode protocol and state errors."""


class HybridNotStarted(HybridControllerError):
    """Raised when input is submitted before ``start``."""


class HybridAlreadyStarted(HybridControllerError):
    """Raised when ``start`` is called more than once."""


class ActionWhileAwaitingProbe(HybridControllerError):
    """Raised when an action is submitted while a checkpoint question is pending."""


class ProbeProtocolError(ValueError):
    """Raised when an answer does not match the strict public envelope."""


class CheckpointMismatch(ProbeProtocolError):
    """Raised when an answer does not identify the currently exposed probe."""


class ProbeAlreadyAnswered(ProbeProtocolError):
    """Raised when an already consumed probe is submitted again."""


class TrajectoryDesignError(HybridControllerError):
    """Raised for a manifest/runtime ordering error that invalidates evaluation."""


class GroundingTransactionError(HybridControllerError):
    """A grounding stage failed; public diagnostics never include oracle data."""

    def __init__(
        self,
        checkpoint_id: str,
        stage: str,
        *,
        rollback_attempted: bool,
        rollback_succeeded: bool | None,
    ) -> None:
        self.checkpoint_id = checkpoint_id
        self.stage = stage
        self.rollback_attempted = rollback_attempted
        self.rollback_succeeded = rollback_succeeded
        if not rollback_attempted:
            rollback_status = "not_attempted"
        elif rollback_succeeded:
            rollback_status = "succeeded"
        else:
            rollback_status = "failed"
        super().__init__(
            f"checkpoint {checkpoint_id!r} grounding failed during {stage}; "
            f"rollback={rollback_status}"
        )


class HybridState(str, Enum):
    NOT_STARTED = "not_started"
    READY = "ready"
    AWAITING_PROBE = "awaiting_probe"
    TERMINATED = "terminated"
    DESIGN_ERROR = "design_error"


@dataclass(frozen=True, slots=True)
class HybridEvent:
    """One JSON-safe public event.

    ``payload`` is deliberately a public-data whitelist assembled by this
    controller.  Evaluator ground truth is held only by private queue entries.
    """

    event: str
    state: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"event": self.event, "state": self.state, **self.payload}


@dataclass(frozen=True, slots=True)
class AnswerEnvelope:
    """Strict sibling envelope used alongside the action envelope.

    Public JSON shape::

        {"checkpoint_id": "...", "probe_id": "...", "answer": <JSON value>}
    """

    checkpoint_id: str
    probe_id: str
    answer: Any

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnswerEnvelope":
        if not isinstance(payload, Mapping):
            raise ProbeProtocolError("answer payload must be a JSON object")
        expected_fields = {"checkpoint_id", "probe_id", "answer"}
        unknown = set(payload) - expected_fields
        missing = expected_fields - set(payload)
        if unknown:
            raise ProbeProtocolError(f"unknown answer fields: {sorted(unknown)}")
        if missing:
            raise ProbeProtocolError(f"missing answer fields: {sorted(missing)}")
        checkpoint_id = _nonempty_string(payload["checkpoint_id"], "checkpoint_id")
        probe_id = _nonempty_string(payload["probe_id"], "probe_id")
        _canonical_json(payload["answer"], field_name="answer")
        return cls(checkpoint_id=checkpoint_id, probe_id=probe_id, answer=payload["answer"])

    @classmethod
    def from_json(cls, text: str) -> "AnswerEnvelope":
        if not isinstance(text, str):
            raise ProbeProtocolError("answer input must be a JSON string")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProbeProtocolError(f"invalid JSON: {exc.msg}") from exc
        return cls.from_dict(payload)


class ProbeGrounder(Protocol):
    """Evaluator-side atomic scene-grounding interface."""

    def snapshot(self) -> object:
        """Return an opaque restorable simulator snapshot."""

    def apply_intervention(self, intervention: Any) -> int | None:
        """Apply the checkpoint's private scene intervention."""

    def settle(self, simulator_steps: int) -> int | None:
        """Advance physics for the manifest's non-negative settling budget."""

    def freeze_public_evidence(self, phase: Any) -> Mapping[str, Any]:
        """Freeze policy-visible JSON evidence separately from private truth."""

    def freeze_ground_truth(self, phase: Any) -> Mapping[str, Any]:
        """Freeze one JSON answer per probe id after settling."""


    def commit(self, snapshot: object) -> None:
        """Release a successfully committed evaluator snapshot."""

    def rollback(self, snapshot: object) -> None:
        """Restore a snapshot after any failed grounding stage."""


@dataclass(frozen=True, slots=True)
class _PhaseView:
    raw: Any
    checkpoint_id: str
    trigger_kind: str
    target_instances: tuple[str, ...]
    target_prefixes: tuple[str, ...]
    support_instances: tuple[str, ...]
    support_prefixes: tuple[str, ...]
    trigger_action: str | None
    trigger_target: str | None
    trigger_support: str | None
    trigger_occurrence: int
    settle_steps: int
    intervention: Any
    probes: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _GroundedProbe:
    checkpoint_id: str
    probe_id: str
    ordinal: int
    count: int
    public: dict[str, Any]
    expected_canonical: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class _AnswerRecord:
    checkpoint_id: str
    probe_id: str
    correct: bool


class HybridEpisodeController:
    """Wrap an :class:`EpisodeController` with transactional QA checkpoints.

    ``trajectory`` may be a hard-trajectory manifest exposing ``phases`` or a
    sequence of phase-like objects.  Phase/probe fields are accessed through a
    small structural adapter so this evaluator remains decoupled from the
    manifest catalogue and simulator.
    """

    def __init__(
        self,
        episode: EpisodeController,
        trajectory: Any,
        grounder: ProbeGrounder,
        *,
        episode_seed: int | None = None,
    ) -> None:
        if episode_seed is not None and (
            not isinstance(episode_seed, int)
            or isinstance(episode_seed, bool)
            or episode_seed < 0
        ):
            raise ValueError("episode_seed must be a non-negative integer")
        self.episode = episode
        self.grounder = grounder
        self._episode_seed = secrets.randbits(128) if episode_seed is None else episode_seed
        self._require_public_evidence = getattr(trajectory, "variant", None) == "hard_v1"
        raw_phases = getattr(trajectory, "phases", trajectory)
        if isinstance(raw_phases, (str, bytes)) or not isinstance(raw_phases, Sequence):
            raise TypeError("trajectory must expose a sequence of phases")

        self._phases = tuple(_adapt_phase(phase) for phase in raw_phases)
        self._validate_manifest()
        task_name = getattr(trajectory, "task_name", None)
        if task_name is not None and task_name != self.episode.task.name:
            raise ValueError(
                f"trajectory task {task_name!r} does not match episode task {self.episode.task.name!r}"
            )
        self._phase_by_id = {phase.checkpoint_id: phase for phase in self._phases}
        self._triggered_checkpoints: set[str] = set()
        self._event_history: list[tuple[str, dict[str, str] | None]] = []
        self._deferred_action: str | None = None
        self._queue: deque[_GroundedProbe] = deque()
        self._answers: list[_AnswerRecord] = []
        self._answered_probe_ids: set[tuple[str, str]] = set()
        self._state = HybridState.NOT_STARTED
        self._design_error: str | None = None
        self._termination_reason: str | None = None

    @property
    def state(self) -> HybridState:
        return self._state

    @property
    def awaiting_probe(self) -> bool:
        return self._state is HybridState.AWAITING_PROBE

    @property
    def current_probe(self) -> dict[str, Any] | None:
        if not self._queue:
            return None
        return _public_probe(self._queue[0])

    def start(self) -> HybridEvent:
        if self._state is not HybridState.NOT_STARTED:
            raise HybridAlreadyStarted("hybrid episode has already started")
        if self.episode.terminated:
            raise TrajectoryDesignError("underlying episode is already terminated")

        self._state = HybridState.READY
        self._record_event("episode_start", None)
        self._try_trigger_active("episode_start", action=None)
        self._sync_state()
        payload: dict[str, Any] = {"summary": self.public_summary()}
        if self.current_probe is not None:
            payload["probe"] = self.current_probe
        return HybridEvent("hybrid_start", self._state.value, payload)

    def step(self, action: HighLevelAction) -> HybridEvent:
        self._require_started()
        if self._state is HybridState.AWAITING_PROBE:
            current = self._queue[0]
            raise ActionWhileAwaitingProbe(
                f"checkpoint {current.checkpoint_id!r} probe {current.probe_id!r} must be answered first"
            )
        if self._state is HybridState.DESIGN_ERROR:
            raise TrajectoryDesignError(self._design_error or "trajectory is invalid")

        action_payload = action.to_dict()
        canonical_action = _canonical_json(action_payload, field_name="action")
        if canonical_action != self._deferred_action:
            self._record_event("before_action", action_payload)
        triggered_before_action = self._try_trigger_active(
            "before_action", action=action_payload
        )
        if triggered_before_action:
            self._sync_state()
            if self.episode.terminated and not self._queue:
                self._deferred_action = None
                return HybridEvent(
                    "hybrid_timeout",
                    self._state.value,
                    {"summary": self.public_summary()},
                )
            self._deferred_action = canonical_action
            return HybridEvent(
                "hybrid_checkpoint",
                self._state.value,
                {
                    "deferred_action": action_payload,
                    "probe": self.current_probe,
                    "summary": self.public_summary(),
                },
            )

        self._deferred_action = None
        result = self.episode.step(action)
        self._record_event("after_action", action_payload)

        if result.success:
            if result.reason is not None:
                self._termination_reason = result.reason
            if not self.episode.terminate_on_success:
                self._try_trigger_active("after_action", action=action_payload)
                self._sync_state()
                if self._queue:
                    payload = {"step": _public_episode_step(result), "summary": self.public_summary()}
                    payload["probe"] = self.current_probe
                    return HybridEvent("hybrid_step", self._state.value, payload)
                active = self._active_phase()
                if active is not None and active.trigger_kind == "before_final_success":
                    if not any(event == "before_final_success" for event, _ in self._event_history):
                        self._record_event("before_final_success", None)
                    self._try_trigger_active("before_final_success", action=None)
                    self._sync_state()
                    payload = {"step": _public_episode_step(result), "summary": self.public_summary()}
                    if self.current_probe is not None:
                        payload["probe"] = self.current_probe
                    return HybridEvent("hybrid_step", self._state.value, payload)
            pending = self._pending_probe_count()
            if pending:
                self._mark_design_error(
                    "task reached success before all trajectory probes were triggered and answered"
                )
                payload = {
                    "step": _public_episode_step(result),
                    "design_error": self._design_error,
                    "summary": self.public_summary(),
                }
                return HybridEvent("trajectory_design_error", self._state.value, payload)
            if not self.episode.terminate_on_success and not self.episode.terminated:
                self.episode.finalize_success()
                result = replace(result, terminated=True, reason="task_success_finalized")
                self._termination_reason = "task_success_finalized"
            self._state = HybridState.TERMINATED
        elif result.terminated:
            self._termination_reason = result.reason
            self._state = HybridState.TERMINATED
        else:
            self._try_trigger_active("after_action", action=action_payload)
            self._sync_state()
            if self.episode.terminated:
                result = replace(
                    result,
                    total_simulator_steps=self.episode.simulator_steps,
                    terminated=True,
                    reason=self._termination_reason,
                )

        payload = {"step": _public_episode_step(result), "summary": self.public_summary()}
        if self.current_probe is not None:
            payload["probe"] = self.current_probe
        return HybridEvent("hybrid_step", self._state.value, payload)

    def answer(self, envelope: AnswerEnvelope | Mapping[str, Any]) -> HybridEvent:
        self._require_started()
        if self._state is HybridState.DESIGN_ERROR:
            raise TrajectoryDesignError(self._design_error or "trajectory is invalid")
        if not isinstance(envelope, AnswerEnvelope):
            envelope = AnswerEnvelope.from_dict(envelope)

        key = (envelope.checkpoint_id, envelope.probe_id)
        if key in self._answered_probe_ids:
            raise ProbeAlreadyAnswered(
                f"probe {envelope.probe_id!r} at checkpoint {envelope.checkpoint_id!r} was already answered"
            )
        if not self._queue:
            raise ProbeProtocolError("no probe is currently awaiting an answer")

        current = self._queue[0]
        if envelope.checkpoint_id != current.checkpoint_id:
            raise CheckpointMismatch(
                f"expected checkpoint {current.checkpoint_id!r}, got {envelope.checkpoint_id!r}"
            )
        if envelope.probe_id != current.probe_id:
            raise CheckpointMismatch(f"expected probe {current.probe_id!r}, got {envelope.probe_id!r}")

        submitted = _canonical_json(envelope.answer, field_name="answer")
        correct = submitted == current.expected_canonical
        self._answers.append(
            _AnswerRecord(
                checkpoint_id=current.checkpoint_id,
                probe_id=current.probe_id,
                correct=correct,
            )
        )
        self._answered_probe_ids.add(key)
        self._queue.popleft()
        self._sync_state()

        self._trigger_automatic_phase()
        self._sync_state()
        if (
            self.episode.backend.success()
            and self._pending_probe_count() == 0
            and not self.episode.terminate_on_success
            and not self.episode.terminated
        ):
            self.episode.finalize_success()
            self._termination_reason = "task_success_finalized"
            self._sync_state()
        payload: dict[str, Any] = {
            "checkpoint_id": current.checkpoint_id,
            "probe_id": current.probe_id,
            "accepted": True,
            "summary": self.public_summary(),
        }
        if self.current_probe is not None:
            payload["next_probe"] = self.current_probe
        return HybridEvent("probe_answer", self._state.value, payload)

    def signal_event(self, kind: str) -> HybridEvent:
        """Submit an evaluator event that cannot be inferred from an action.

        Hard-v1 uses this for ``after_revisit``. Action and episode-start
        triggers remain controlled by :meth:`step` and :meth:`start`.
        """

        self._require_started()
        if self._state is HybridState.AWAITING_PROBE:
            current = self._queue[0]
            raise ActionWhileAwaitingProbe(
                f"checkpoint {current.checkpoint_id!r} probe {current.probe_id!r} must be answered first"
            )
        if self._state is HybridState.DESIGN_ERROR:
            raise TrajectoryDesignError(self._design_error or "trajectory is invalid")
        if self._state is HybridState.TERMINATED:
            raise HybridControllerError("hybrid episode has terminated")
        if kind != "after_revisit":
            raise ProbeProtocolError(f"unsupported external checkpoint signal {kind!r}")
        active = self._active_phase()
        if active is None or active.trigger_kind != kind:
            raise ProbeProtocolError(
                f"checkpoint signal {kind!r} does not match the active trajectory phase"
            )
        self._record_event(kind, None)
        triggered = self._try_trigger_active(kind, action=None)
        self._sync_state()
        payload: dict[str, Any] = {"triggered": triggered, "summary": self.public_summary()}
        if self.current_probe is not None:
            payload["probe"] = self.current_probe
        return HybridEvent("hybrid_signal", self._state.value, payload)


    def summary(self) -> dict[str, Any]:
        answered = len(self._answers)
        correct = sum(record.correct for record in self._answers)
        pending = self._pending_probe_count()
        expected = self._total_probe_count()
        accuracy = correct / answered if answered else None
        coverage = answered / expected if expected else 1.0
        score = correct / expected if expected else None
        task_success = bool(self.episode.backend.success())
        task_q_score = float(self.episode.backend.q_score())
        return {
            "qa": {
                "answered": answered,
                "correct": correct,
                "accuracy": accuracy,
                "pending": pending,
                "expected": expected,
                "coverage": coverage,
                "score": score,
            },
            "task": {
                "success": task_success,
                "q_score": task_q_score,
            },
            "complete": (
                task_success
                and pending == 0
                and self._state is HybridState.TERMINATED
                and self._state is not HybridState.DESIGN_ERROR
            ),
            "design_error": self._design_error,
            "termination_reason": self._termination_reason,
        }
    def evaluator_summary(self) -> dict[str, Any]:
        """Return complete evaluator-only task and QA metrics."""

        return self.summary()

    def public_summary(self) -> dict[str, Any]:
        """Return online-safe progress; reveal scores only at a terminal state."""

        if self._state in {HybridState.TERMINATED, HybridState.DESIGN_ERROR}:
            return self.summary()
        complete = self.summary()
        return {
            "qa": {
                "answered": complete["qa"]["answered"],
                "pending": complete["qa"]["pending"],
            },
            "task": {"terminated": self.episode.terminated},
            "complete": False,
            "design_error": None,
        }


    def _validate_manifest(self) -> None:
        checkpoint_ids: set[str] = set()
        probe_keys: set[tuple[str, str]] = set()
        for phase in self._phases:
            if phase.checkpoint_id in checkpoint_ids:
                raise ValueError(f"duplicate checkpoint id {phase.checkpoint_id!r}")
            checkpoint_ids.add(phase.checkpoint_id)
            for probe in phase.probes:
                probe_id = _probe_id(probe)
                key = (phase.checkpoint_id, probe_id)
                if key in probe_keys:
                    raise ValueError(
                        f"duplicate probe id {probe_id!r} within checkpoint {phase.checkpoint_id!r}"
                    )
                probe_keys.add(key)

    def _require_started(self) -> None:
        if self._state is HybridState.NOT_STARTED:
            raise HybridNotStarted("call start() before submitting actions or answers")

    def _active_phase(self) -> _PhaseView | None:
        for phase in self._phases:
            if phase.probes and phase.checkpoint_id not in self._triggered_checkpoints:
                return phase
        return None

    def _try_trigger_active(self, event: str, action: dict[str, str] | None) -> bool:
        triggered = False
        while True:
            phase = self._active_phase()
            if phase is None or not _trigger_matches(phase, event, action):
                break
            occurrence = sum(
                _trigger_matches(phase, past_event, past_action)
                for past_event, past_action in self._event_history
            )
            if occurrence < phase.trigger_occurrence:
                break
            self._ground_phase(phase)
            triggered = True
            if self.episode.terminated:
                break
        return triggered

    def _record_event(self, event: str, action: dict[str, str] | None) -> None:
        copied_action = None if action is None else dict(action)
        self._event_history.append((event, copied_action))

    def _trigger_automatic_phase(self) -> None:
        if self._queue or self._state is HybridState.DESIGN_ERROR:
            return
        phase = self._active_phase()
        if phase is None:
            return
        automatic_kinds: set[str] = set()
        if self.episode.backend.success():
            automatic_kinds.add("before_final_success")
        if self.episode.macro_steps == 0:
            automatic_kinds.add("episode_start")
        if phase.trigger_kind in automatic_kinds:
            if not any(event == phase.trigger_kind for event, _ in self._event_history):
                self._record_event(phase.trigger_kind, None)
            self._try_trigger_active(phase.trigger_kind, action=None)

    def _commit_grounding_snapshot(self, snapshot: object) -> None:
        commit = getattr(self.grounder, "commit", None)
        if commit is None:
            if self._require_public_evidence:
                raise TypeError("hard_v1 grounder must implement commit")
            return
        commit(snapshot)

    def _ground_phase(self, phase: _PhaseView) -> None:
        stage = "snapshot"
        try:
            snapshot = self.grounder.snapshot()
        except Exception:
            self._mark_design_error(f"grounding failed at checkpoint {phase.checkpoint_id!r}")
            raise GroundingTransactionError(
                phase.checkpoint_id,
                stage,
                rollback_attempted=False,
                rollback_succeeded=None,
            ) from None

        try:
            stage = "intervention"
            intervention_steps = _external_step_count(
                self.grounder.apply_intervention(phase.intervention)
            )
            stage = "settle"
            settle_steps = _external_step_count(self.grounder.settle(phase.settle_steps))
            stage = "simulator_step_accounting"
            timeout_reason = self.episode.charge_simulator_steps(intervention_steps + settle_steps)
            if timeout_reason is not None:
                stage = "commit"
                self._commit_grounding_snapshot(snapshot)
                self._termination_reason = timeout_reason
                self._triggered_checkpoints.add(phase.checkpoint_id)
                return
            stage = "freeze_public_evidence"
            evidence_method = getattr(self.grounder, "freeze_public_evidence", None)
            if evidence_method is None:
                if self._require_public_evidence:
                    raise TypeError("hard_v1 grounder must implement freeze_public_evidence")
                public_evidence = {_probe_id(probe): {} for probe in phase.probes}
            else:
                public_evidence = evidence_method(phase.raw)
            public_evidence = _validate_public_evidence(phase, public_evidence)
            stage = "freeze_ground_truth"
            frozen = self.grounder.freeze_ground_truth(phase.raw)
            grounded = self._validate_and_freeze_answers(phase, frozen, public_evidence)
            stage = "commit"
            self._commit_grounding_snapshot(snapshot)
        except Exception:
            rollback_succeeded = False
            try:
                self.grounder.rollback(snapshot)
                rollback_succeeded = True
            except Exception:
                pass
            self._mark_design_error(f"grounding failed at checkpoint {phase.checkpoint_id!r}")
            raise GroundingTransactionError(
                phase.checkpoint_id,
                stage,
                rollback_attempted=True,
                rollback_succeeded=rollback_succeeded,
            ) from None

        self._triggered_checkpoints.add(phase.checkpoint_id)
        self._queue.extend(grounded)

    def _validate_and_freeze_answers(
        self,
        phase: _PhaseView,
        frozen: Mapping[str, Any],
        public_evidence: Mapping[str, Any],
    ) -> tuple[_GroundedProbe, ...]:
        if not isinstance(frozen, Mapping):
            raise TypeError("frozen ground truth must be a mapping")
        probe_ids = tuple(_probe_id(probe) for probe in phase.probes)
        if set(frozen) != set(probe_ids):
            raise ValueError("frozen ground truth keys must exactly match phase probe ids")

        count = len(phase.probes)
        grounded: list[_GroundedProbe] = []
        for ordinal, probe in enumerate(phase.probes, start=1):
            probe_id = _probe_id(probe)
            public = _probe_public_dict(probe)
            public, option_id_map = _shuffle_public_options(
                public,
                episode_seed=self._episode_seed,
                checkpoint_id=phase.checkpoint_id,
                probe_id=probe_id,
            )
            public["evidence"] = public_evidence[probe_id]
            frozen_answer = frozen[probe_id]
            if hasattr(frozen_answer, "option_id"):
                frozen_answer = frozen_answer.option_id
            if option_id_map:
                if not isinstance(frozen_answer, str) or frozen_answer not in option_id_map:
                    raise ValueError("frozen multiple-choice answer is not a private option id")
                frozen_answer = option_id_map[frozen_answer]
            expected = _canonical_json(frozen_answer, field_name="ground truth")
            grounded.append(
                _GroundedProbe(
                    checkpoint_id=phase.checkpoint_id,
                    probe_id=probe_id,
                    ordinal=ordinal,
                    count=count,
                    public=public,
                    expected_canonical=expected,
                )
            )
        return tuple(grounded)

    def _sync_state(self) -> None:
        if self._state is HybridState.DESIGN_ERROR:
            return
        if self._queue:
            self._state = HybridState.AWAITING_PROBE
        elif self.episode.terminated:
            self._state = HybridState.TERMINATED
        else:
            self._state = HybridState.READY

    def _mark_design_error(self, message: str) -> None:
        self._design_error = message
        self._state = HybridState.DESIGN_ERROR

    def _total_probe_count(self) -> int:
        return sum(len(phase.probes) for phase in self._phases)

    def _pending_probe_count(self) -> int:
        return self._total_probe_count() - len(self._answers)


def _adapt_phase(phase: Any) -> _PhaseView:
    checkpoint_id = _nonempty_string(
        _first_attr(phase, "checkpoint_id", "phase_id", "id"),
        "checkpoint_id",
    )
    trigger = _first_attr(phase, "trigger")
    trigger_kind = _enum_value(_first_attr(trigger, "kind", "event"))
    allowed_triggers = {
        "episode_start",
        "before_action",
        "after_action",
        "after_revisit",
        "before_final_success",
        "final_success_check",
    }
    if trigger_kind not in allowed_triggers:
        raise ValueError(f"unsupported trigger kind {trigger_kind!r}")
    trigger_action = _optional_enum_value(
        _first_attr(trigger, "action", "action_name", default=None)
    )
    trigger_target = _optional_string(_first_attr(trigger, "target", default=None), "target")
    trigger_support = _optional_string(_first_attr(trigger, "support", default=None), "support")
    target_instances = _string_tuple(_first_attr(trigger, "target_instances", default=()), "target_instances")
    target_prefixes = _string_tuple(_first_attr(trigger, "target_prefixes", default=()), "target_prefixes")
    support_instances = _string_tuple(_first_attr(trigger, "support_instances", default=()), "support_instances")
    support_prefixes = _string_tuple(_first_attr(trigger, "support_prefixes", default=()), "support_prefixes")
    trigger_occurrence = _first_attr(trigger, "occurrence", default=1)
    if (
        not isinstance(trigger_occurrence, int)
        or isinstance(trigger_occurrence, bool)
        or trigger_occurrence < 1
    ):
        raise ValueError(f"checkpoint {checkpoint_id!r} occurrence must be a positive integer")
    settle_steps = _first_attr(trigger, "settle_steps", default=0)
    if not isinstance(settle_steps, int) or isinstance(settle_steps, bool) or settle_steps < 0:
        raise ValueError(f"checkpoint {checkpoint_id!r} settle_steps must be non-negative")
    missing = object()
    probes = _first_attr(phase, "probes", default=missing)
    if probes is missing:
        single_probe = _first_attr(phase, "probe", default=None)
        probes = () if single_probe is None else (single_probe,)
    if isinstance(probes, (str, bytes)) or not isinstance(probes, Sequence):
        raise TypeError(f"checkpoint {checkpoint_id!r} probes must be a sequence")
    return _PhaseView(
        raw=phase,
        checkpoint_id=checkpoint_id,
        trigger_kind=trigger_kind,
        trigger_action=trigger_action,
        trigger_target=trigger_target,
        trigger_support=trigger_support,
        target_instances=target_instances,
        target_prefixes=target_prefixes,
        support_instances=support_instances,
        support_prefixes=support_prefixes,
        intervention=_first_attr(phase, "intervention", default=None),
        probes=tuple(probes),
        trigger_occurrence=trigger_occurrence,
        settle_steps=settle_steps,
    )


def _trigger_matches(
    phase: _PhaseView,
    event: str,
    action: dict[str, str] | None,
) -> bool:
    if phase.trigger_kind != event:
        return False
    if event in {"episode_start", "after_revisit", "before_final_success"}:
        return True
    if action is None:
        return False
    if phase.trigger_action is not None and action.get("action") != phase.trigger_action:
        return False
    if phase.trigger_target is not None and action.get("target") != phase.trigger_target:
        return False
    if phase.trigger_support is not None and action.get("support") != phase.trigger_support:
        return False
    target = action.get("target")
    if phase.target_instances or phase.target_prefixes:
        if target is None or not (
            target in phase.target_instances
            or any(target.startswith(prefix) for prefix in phase.target_prefixes)
        ):
            return False
    support = action.get("support")
    if phase.support_instances or phase.support_prefixes:
        if support is None or not (
            support in phase.support_instances
            or any(support.startswith(prefix) for prefix in phase.support_prefixes)
        ):
            return False
    return True



_PRIVILEGED_EVIDENCE_KEY_PARTS = (
    "oracle",
    "expected",
    "correct",
    "answer",
    "ground_truth",
    "predicate",
    "contact",
    "goal_status",
    "q_score",
    "success",
)


def _validate_public_evidence(
    phase: _PhaseView,
    evidence: Any,
) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        raise TypeError("public evidence must be a mapping")
    probe_ids = {_probe_id(probe) for probe in phase.probes}
    if set(evidence) != probe_ids:
        raise ValueError("public evidence keys must exactly match phase probe ids")
    validated: dict[str, Any] = {}
    for probe_id in probe_ids:
        bundle = evidence[probe_id]
        _reject_privileged_evidence_keys(bundle)
        validated[probe_id] = _json_copy(bundle, field_name="public evidence")
    return validated


def _reject_privileged_evidence_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("public evidence object keys must be strings")
            normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
            if any(part in normalized for part in _PRIVILEGED_EVIDENCE_KEY_PARTS):
                raise ValueError(f"privileged public evidence key is forbidden: {key!r}")
            _reject_privileged_evidence_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_privileged_evidence_keys(item)


def _probe_id(probe: Any) -> str:
    return _nonempty_string(_first_attr(probe, "probe_id", "id"), "probe_id")


def _probe_public_dict(probe: Any) -> dict[str, Any]:
    """Build a strict whitelist; never serialize the full manifest object."""

    safe_serializer = getattr(probe, "to_public_dict", None)
    if callable(safe_serializer):
        serialized = safe_serializer()
        if not isinstance(serialized, Mapping):
            raise TypeError("probe.to_public_dict() must return a mapping")
        public = {"probe_id": _probe_id(probe)}
        safe_keys = (
            "family",
            "big_task",
            "small_task",
            "runner_task",
            "adaptation",
            "prompt",
            "options",
            "evidence_tiers",
            "answer_format",
        )
        for key in safe_keys:
            if key in serialized:
                public[key] = _json_copy(serialized[key], field_name=key)
        return public

    public = {
        "probe_id": _probe_id(probe),
        "family": _enum_value(_first_attr(probe, "family")),
        "prompt": _nonempty_string(
            _first_attr(probe, "prompt", "question"),
            "prompt",
        ),
    }
    choices = _first_attr(probe, "choices", "options", default=())
    if choices:
        public["choices"] = _json_copy(choices, field_name="choices")
    answer_format = _first_attr(probe, "answer_format", "answer_type", default=None)
    if answer_format is not None:
        public["answer_format"] = _enum_value(answer_format)
    evidence_tiers = _first_attr(probe, "evidence_tiers", default=None)
    if evidence_tiers is not None:
        public["evidence_tiers"] = sorted(_enum_value(tier) for tier in evidence_tiers)
    else:
        evidence_tier = _first_attr(probe, "evidence_tier", default=None)
        if evidence_tier is not None:
            public["evidence_tier"] = _enum_value(evidence_tier)
    return public


def _public_probe(probe: _GroundedProbe) -> dict[str, Any]:
    return {
        "checkpoint_id": probe.checkpoint_id,
        "ordinal": probe.ordinal,
        "count": probe.count,
        **_json_copy(probe.public, field_name="public probe"),
    }

def _shuffle_public_options(
    public: Mapping[str, Any],
    *,
    episode_seed: int,
    checkpoint_id: str,
    probe_id: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    options = public.get("options")
    if not isinstance(options, list) or len(options) < 2:
        return dict(public), {}
    copied_options = _json_copy(options, field_name="options")
    private_ids: list[str] = []
    for option in copied_options:
        if not isinstance(option, Mapping):
            raise ValueError("public multiple-choice options must be JSON objects")
        private_id = _nonempty_string(option.get("id"), "option id")
        private_ids.append(private_id)
    material = f"{episode_seed}:{checkpoint_id}:{probe_id}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    order = list(range(len(copied_options)))
    rng.shuffle(order)
    shuffled: list[dict[str, Any]] = []
    option_id_map: dict[str, str] = {}
    for public_index, private_index in enumerate(order, start=1):
        option = dict(copied_options[private_index])
        private_id = private_ids[private_index]
        public_id = f"option_{public_index}"
        option["id"] = public_id
        shuffled.append(option)
        option_id_map[private_id] = public_id
    result = dict(public)
    result["options"] = shuffled
    return result, option_id_map



def _public_episode_step(step: EpisodeStep) -> dict[str, Any]:
    if not isinstance(step, EpisodeStep):
        raise TypeError("EpisodeController.step() must return EpisodeStep")
    return {
        "action": dict(step.action),
        "macro_step": step.macro_step,
        "simulator_steps": step.simulator_steps,
        "total_simulator_steps": step.total_simulator_steps,
        "terminated": step.terminated,
    }


def _first_attr(value: Any, *names: str, default: Any = ...) -> Any:
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
        if isinstance(value, Mapping) and name in value:
            return value[name]
    if default is not ...:
        return default
    joined = ", ".join(repr(name) for name in names)
    raise TypeError(f"object is missing required field ({joined})")

def _external_step_count(value: Any) -> int:
    if value is None:
        return 0
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("grounder simulator-step result must be a non-negative integer or None")
    return value



def _enum_value(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value
    return _nonempty_string(value, "enum value")


def _optional_enum_value(value: Any) -> str | None:
    if value is None:
        return None
    return _enum_value(value)



def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a sequence of non-empty strings")
    return tuple(_nonempty_string(item, field_name) for item in value)


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, field_name)


def _nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name!r} must be a non-empty string")
    return value.strip()


def _json_copy(value: Any, *, field_name: str) -> Any:
    canonical = _canonical_json(value, field_name=field_name)
    return json.loads(canonical)


def _canonical_json(value: Any, *, field_name: str) -> str:
    _validate_json_value(value, field_name=field_name)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProbeProtocolError(f"{field_name} must be finite JSON data") from exc


def _validate_json_value(value: Any, *, field_name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ProbeProtocolError(f"{field_name} must not contain NaN or infinity")
        return
    if isinstance(value, list) or isinstance(value, tuple):
        for item in value:
            _validate_json_value(item, field_name=field_name)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProbeProtocolError(f"{field_name} object keys must be strings")
            _validate_json_value(item, field_name=field_name)
        return
    raise ProbeProtocolError(f"{field_name} contains non-JSON value {type(value).__name__}")
