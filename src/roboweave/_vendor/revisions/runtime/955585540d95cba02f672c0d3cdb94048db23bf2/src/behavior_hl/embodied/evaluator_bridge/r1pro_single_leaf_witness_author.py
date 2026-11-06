"""Pre-reset TokenPlan author for one physical ``Inside`` leaf.

The author starts from the already-held G3b state.  It reuses the normal B1
ActionPluginDispatcher and resident controller services for carry, look, and
``place_inside_at``.  Only public onboard RGB-D/proprio reaches TokenPlan.
Official predicate truth is sampled in a disjoint private admission path and
can only approve freezing after exactly one new leaf.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from behavior_hl.embodied.evaluator_bridge.official_predicate_verifier import (
    OfficialPredicateVerifier,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_public_wrist_evidence import (
    PublicWristCalibrationRecord,
)
from behavior_hl.embodied.evaluator_bridge.safe_carry_calibration import (
    SafeCarryCalibrationRecord,
)
from behavior_hl.embodied.policy.profiles import (
    PICKING_UP_TOYS_PROFILE,
    build_picking_up_toys_registry,
)
from behavior_hl.embodied.policy.protocol import ActionEnvelope
from behavior_hl.embodied.policy.registry import ActionRegistry
from behavior_hl.embodied.runtime.dispatch import (
    ActionExecutionContext,
    ActionPluginDispatcher,
    ActionPluginResult,
)
from behavior_hl.embodied.runtime.episode import EpisodeSpec
from behavior_hl.embodied.runtime.observation import PublicRobotObservation
from behavior_hl.embodied.runtime.public_single_leaf_witness import (
    PrivateSingleLeafAdmissionEvidence,
    PublicSingleLeafWitness,
    PublicWitnessDecision,
)
from behavior_hl.embodied.runtime.tokenplan_resident_model import (
    TOKENPLAN_RESIDENT_MODEL,
    TokenPlanResidentModel,
)


R1PRO_SINGLE_LEAF_WITNESS_AUTHOR_VERSION = (
    "behavior2026.r1pro-single-leaf-witness-author/v1"
)
_CARRY_ACTIONS = frozenset(
    {
        "move_backward",
        "move_forward",
        "move_left",
        "move_right",
        "turn_left",
        "turn_right",
        "look_up",
        "look_center",
        "look_down",
        "place_inside_at",
    }
)
_BASE_ACTIONS = frozenset(
    {
        "move_backward",
        "move_forward",
        "move_left",
        "move_right",
        "turn_left",
        "turn_right",
    }
)


class R1ProSingleLeafWitnessAuthorError(RuntimeError):
    """Fail-closed authoring, execution, or private-admission error."""


@dataclass(frozen=True, slots=True)
class PublicHLAExecution:
    status: str
    simulator_steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("status must be non-empty")
        if (
            isinstance(self.simulator_steps, bool)
            or not isinstance(self.simulator_steps, int)
            or self.simulator_steps < 0
        ):
            raise ValueError("simulator_steps must be non-negative")


@dataclass(frozen=True, slots=True)
class R1ProSingleLeafAuthorRuntime:
    """Already-assembled normal HLA runtime; no author-specific executor."""

    registry: ActionRegistry
    dispatcher: ActionPluginDispatcher
    environment: "PublicHLAEnvironmentFacade"
    model: TokenPlanResidentModel
    episode_id: str
    max_model_calls: int = 24
    max_controller_ticks: int = 10_000
    max_simulator_steps: int = 2_000

    def __post_init__(self) -> None:
        if not isinstance(self.registry, ActionRegistry):
            raise TypeError("registry must be ActionRegistry")
        if not isinstance(self.dispatcher, ActionPluginDispatcher):
            raise TypeError("dispatcher must be ActionPluginDispatcher")
        if not isinstance(self.model, TokenPlanResidentModel):
            raise TypeError("model must be TokenPlanResidentModel")
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        for name in (
            "max_model_calls",
            "max_controller_ticks",
            "max_simulator_steps",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.environment, PublicHLAEnvironmentFacade):
            raise TypeError(
                "environment must be the public HLA facade"
            )


class PublicHLAEnvironmentFacade:
    """Expose normal HLA execution without raw evaluator/private state APIs."""

    def __init__(self, delegate: Any) -> None:
        required = {
            "observe",
            "controller_services",
            "record_public_action_result",
            "settle",
        }
        if any(
            not callable(getattr(delegate, name, None))
            for name in required
        ):
            raise TypeError("delegate lacks normal resident HLA services")
        self.__delegate = delegate

    def observe(self) -> PublicRobotObservation:
        result = self.__delegate.observe()
        if not isinstance(result, PublicRobotObservation):
            raise R1ProSingleLeafWitnessAuthorError(
                "public HLA facade received an invalid observation"
            )
        return result

    def controller_services(
        self,
        action_index: int,
        action: ActionEnvelope,
    ) -> Mapping[str, Any]:
        return self.__delegate.controller_services(action_index, action)

    def record_public_action_result(
        self,
        action: ActionEnvelope,
        result: ActionPluginResult,
    ) -> None:
        self.__delegate.record_public_action_result(action, result)

    def settle(
        self,
        action: ActionEnvelope,
        result: ActionPluginResult,
        *,
        max_steps: int,
    ) -> int:
        return self.__delegate.settle(
            action,
            result,
            max_steps=max_steps,
        )


class NormalHLADispatcherExecutor:
    """Execute one action through the same dispatcher/service lifecycle."""

    def __init__(self, runtime: R1ProSingleLeafAuthorRuntime) -> None:
        self.runtime = runtime
        self._simulator_steps = 0

    @property
    def simulator_steps(self) -> int:
        return self._simulator_steps

    def execute(
        self,
        *,
        action_index: int,
        action: ActionEnvelope,
    ) -> PublicHLAExecution:
        context = ActionExecutionContext(
            episode_id=self.runtime.episode_id,
            action_index=action_index,
            services=self.runtime.environment.controller_services(
                action_index,
                action,
            ),
        )
        active = self.runtime.dispatcher.begin(
            self.runtime.registry,
            action,
            context,
        )
        result: ActionPluginResult | None = None
        for _ in range(self.runtime.max_controller_ticks):
            result = self.runtime.dispatcher.tick(active)
            if result is not None:
                break
        if result is None:
            result = self.runtime.dispatcher.abort(
                active,
                "controller_tick_budget_exceeded",
            )
        if not isinstance(result, ActionPluginResult):
            raise R1ProSingleLeafWitnessAuthorError(
                "normal HLA dispatcher returned an invalid result"
            )
        self.runtime.environment.record_public_action_result(
            action,
            result,
        )
        remaining = (
            self.runtime.max_simulator_steps
            - self._simulator_steps
            - active.simulator_steps
        )
        if remaining < 0:
            raise R1ProSingleLeafWitnessAuthorError(
                "author simulator-step budget exhausted in controller"
            )
        settled = self.runtime.environment.settle(
            action,
            result,
            max_steps=remaining,
        )
        if (
            isinstance(settled, bool)
            or not isinstance(settled, int)
            or settled < 0
        ):
            raise R1ProSingleLeafWitnessAuthorError(
                "normal HLA settle accounting is invalid"
            )
        steps = active.simulator_steps + settled
        self._simulator_steps += steps
        if self._simulator_steps > self.runtime.max_simulator_steps:
            raise R1ProSingleLeafWitnessAuthorError(
                "author simulator-step budget exhausted"
            )
        return PublicHLAExecution(
            status=result.status,
            simulator_steps=steps,
        )


def make_r1pro_single_leaf_witness_author(
    *,
    runtime_factory: Callable[..., R1ProSingleLeafAuthorRuntime],
) -> Callable[..., PublicSingleLeafWitness]:
    """Return the concrete callback consumed by the live entrypoint."""

    if not callable(runtime_factory):
        raise TypeError("runtime_factory must be callable")

    def callback(**kwargs: Any) -> PublicSingleLeafWitness:
        expected = {
            "evaluator",
            "control",
            "approach",
            "formal_wrist",
            "safe_carry",
            "run_dir",
            "live_public_snapshot",
            "step_once",
            "witness_author_identity",
            "cached_public_witness_prefix",
            "starter",
        }
        if set(kwargs) != expected:
            raise R1ProSingleLeafWitnessAuthorError(
                "witness author callback fields changed"
            )
        del kwargs["step_once"]  # normal HLA services own all physical steps
        evaluator = kwargs["evaluator"]
        formal = kwargs["formal_wrist"]
        safe_carry = kwargs["safe_carry"]
        author = kwargs["witness_author_identity"]
        prefix = tuple(kwargs["cached_public_witness_prefix"])
        run_dir = Path(kwargs["run_dir"])
        if (
            not isinstance(formal, PublicWristCalibrationRecord)
            or not isinstance(safe_carry, SafeCarryCalibrationRecord)
            or not safe_carry.passed
            or formal.hand != safe_carry.candidate.hand
            or formal.safe_carry_calibration_sha256 != safe_carry.sha256
        ):
            raise R1ProSingleLeafWitnessAuthorError(
                "author requires matching passing wrist/carry calibration"
            )
        if (
            not prefix
            or any(
                not isinstance(item, PublicWitnessDecision)
                for item in prefix
            )
            or tuple(item.sequence for item in prefix)
            != tuple(range(len(prefix)))
            or sum(
                item.action.get("action") == "grasp_at"
                for item in prefix
            )
            != 1
            or prefix[-1].action.get("action") != "grasp_at"
            or prefix[-1].public_result_status != "grasp_likely"
        ):
            raise R1ProSingleLeafWitnessAuthorError(
                "cached public prefix must end in one likely physical grasp"
            )
        runtime = runtime_factory(
            evaluator=evaluator,
            control=kwargs["control"],
            formal_wrist=formal,
            safe_carry=safe_carry,
            run_dir=run_dir,
            starter=kwargs["starter"],
            live_public_snapshot=kwargs["live_public_snapshot"],
            cached_public_witness_prefix=prefix,
        )
        if not isinstance(runtime, R1ProSingleLeafAuthorRuntime):
            raise R1ProSingleLeafWitnessAuthorError(
                "author runtime factory returned the wrong type"
            )
        registry = build_picking_up_toys_registry(
            calibrated_hands=(safe_carry.candidate.hand,)
        )
        if runtime.registry.manifest() != registry.manifest():
            raise R1ProSingleLeafWitnessAuthorError(
                "author runtime registry differs from frozen B1 registry"
            )
        verifier = OfficialPredicateVerifier(require_single_option=True)
        episode = EpisodeSpec(
            episode_id=runtime.episode_id,
            task_id=PICKING_UP_TOYS_PROFILE.task_name,
            split="train",
            instance_id="0",
            seed=0,
            instruction=PICKING_UP_TOYS_PROFILE.task_language,
            action_profile="r1pro_esi_single_calibrated_hand_v1",
            model_profile="TokenPlan:qwen3.8-max-preview",
            action_schema_version=registry.schema_version,
        )
        initial = verifier.evaluate(episode, evaluator)
        if (
            initial.official_success
            or any(initial.leaf_truth.values())
            or len(initial.leaf_truth) != 6
        ):
            raise R1ProSingleLeafWitnessAuthorError(
                "pre-author official leaves must be six false leaves"
            )
        # The inner author receives no raw evaluator and has no
        # evaluator_state method.  It can only observe and execute through
        # the public HLA facade.
        decisions = _run_public_author(
            runtime=runtime,
            registry=registry,
            prefix=prefix,
            calibrated_hand=safe_carry.candidate.hand,
            witness_author=author,
            run_dir=run_dir,
        )
        final = verifier.evaluate(episode, evaluator)
        if final.official_success:
            raise R1ProSingleLeafWitnessAuthorError(
                "single-leaf author unexpectedly completed the full task"
            )
        private_admission = PrivateSingleLeafAdmissionEvidence(
            before_leaf_truth=initial.leaf_truth,
            after_leaf_truth=final.leaf_truth,
            verifier_version=(
                "behavior2026.official-predicate-verifier/"
                "stable-predicate-identity-v2"
            ),
        )
        _write_json_atomic(
            run_dir / "private" / "witness_author_verifier.json",
            {
                "visibility": "private",
                "pre": initial.to_private_record(),
                "post": final.to_private_record(),
                "admission": private_admission.to_private_record(),
            },
        )
        return PublicSingleLeafWitness.freeze_after_private_canary(
            decisions=decisions,
            calibrated_hand=safe_carry.candidate.hand,
            witness_author=author,
            registry=registry,
            private_admission_evidence=private_admission,
        )

    return callback


def _run_public_author(
    *,
    runtime: R1ProSingleLeafAuthorRuntime,
    registry: ActionRegistry,
    prefix: Sequence[PublicWitnessDecision],
    calibrated_hand: str,
    witness_author: str,
    run_dir: Path,
) -> tuple[PublicWitnessDecision, ...]:
    """Run TokenPlan using only public observations and normal HLA services."""

    decisions = list(prefix)
    executor = NormalHLADispatcherExecutor(runtime)
    carry_motion_seen = False
    for model_index in range(runtime.max_model_calls):
        observation = runtime.environment.observe()
        prompt = _author_prompt(
            registry=registry,
            hand=calibrated_hand,
            action_index=model_index,
            remaining_calls=runtime.max_model_calls - model_index,
        )
        raw = runtime.model.complete(prompt, observation)
        try:
            raw_payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise R1ProSingleLeafWitnessAuthorError(
                "TokenPlan author response is not strict JSON"
            ) from exc
        original_reasoning = (
            raw_payload.get("reasoning")
            if isinstance(raw_payload, Mapping)
            else None
        )
        if (
            not isinstance(original_reasoning, str)
            or not original_reasoning.strip()
        ):
            raise R1ProSingleLeafWitnessAuthorError(
                "TokenPlan author requires an original reasoning field"
            )
        action = registry.parse_json(raw)
        if action.action not in _CARRY_ACTIONS:
            raise R1ProSingleLeafWitnessAuthorError(
                "author selected grasp/release/stop or an unknown action"
            )
        if action.action == "place_inside_at" and not carry_motion_seen:
            raise R1ProSingleLeafWitnessAuthorError(
                "author must execute physical carrying base motion first"
            )
        execution = executor.execute(
            action_index=model_index,
            action=action,
        )
        decision = PublicWitnessDecision.from_public_execution(
            sequence=len(decisions),
            registry=registry,
            action=action.to_dict(),
            observation=observation,
            public_result_status=execution.status,
            simulator_steps=execution.simulator_steps,
        )
        decisions.append(decision)
        _append_jsonl_atomic(
            run_dir / "public" / "witness_author_actions.jsonl",
            {
                **decision.to_manifest(),
                "witness_author": witness_author,
                "raw_response": raw,
                "original_reasoning": original_reasoning,
                "parsed_action": action.to_dict(),
            },
        )
        if (
            action.action in _BASE_ACTIONS
            and execution.status in {"completed", "motion_partial"}
        ):
            carry_motion_seen = True
        if action.action == "place_inside_at":
            return tuple(decisions)
    raise R1ProSingleLeafWitnessAuthorError(
        "TokenPlan author model-call budget exhausted"
    )


def _author_prompt(
    *,
    registry: ActionRegistry,
    hand: str,
    action_index: int,
    remaining_calls: int,
) -> str:
    schemas = {
        name: _thaw_json(registry.spec(name).manifest())
        for name in sorted(_CARRY_ACTIONS)
    }
    return (
        "You are controlling a physical R1Pro using only the current onboard "
        "RGB-D image and proprioception. The tennis ball is already held in "
        f"the {hand} gripper. Carry it to the visible toy box, actively "
        "adjust base and head view as needed, then select one normalized "
        "public RGB-D point clearly inside the box opening with "
        "place_inside_at. Collisions block motion. Never assume a map, hidden "
        "pose, object ID, goal truth, or verifier result. Return exactly one "
        "strict JSON action envelope and no Markdown. "
        f"Action index={action_index}; remaining calls={remaining_calls}. "
        f"Instruction={PICKING_UP_TOYS_PROFILE.task_language!r}. "
        f"Allowed action schemas={json.dumps(schemas, sort_keys=True)}"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _thaw_json(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_thaw_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"action schema is not strict JSON: {type(value).__name__}"
    )


def _append_jsonl_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    records: list[Any] = []
    if path.is_file():
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    records.append(dict(payload))
    _write_text_atomic(
        path,
        "".join(
            json.dumps(
                record,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
            for record in records
        ),
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_atomic(
        path,
        json.dumps(
            dict(payload),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
    )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
