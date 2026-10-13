from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from PIL import Image
import pytest

from lab import model_client


def _action(action: str, call_id: str) -> dict:
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "robot_action",
                                "arguments": json.dumps({"action": action}),
                            },
                        }
                    ],
                },
            }
        ]
    }


def test_scene_client_preserves_typed_public_recovery_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = {
        "protocol_version": model_client.SCENE_PROTOCOL_VERSION,
        "status": "ERROR",
        "error_type": "RECOVERABLE_SCENE_ERROR",
        "message": "RECOVERABLE_SCENE_ERROR",
        "error_id": "a" * 32,
        "recovery": {
            "scene_resident": True,
            "session_valid": True,
            "episode_active": False,
            "episode_id": None,
            "next_sequence": 0,
            "rolling_checkpoint_action_counts": [3],
        },
    }

    class Connection:
        def settimeout(self, seconds: int) -> None:
            assert seconds == 600

        def connect(self, path: str) -> None:
            assert path == "/tmp/test-scene.sock"

        def sendall(self, payload: bytes) -> None:
            assert b'"command":"step"' in payload

        def makefile(self, mode: str) -> io.BytesIO:
            assert mode == "rb"
            return io.BytesIO(model_client._canonical_json_bytes(response) + b"\n")

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        model_client.socket,
        "socket",
        lambda *args, **kwargs: Connection(),
    )
    client = model_client.SceneClient.__new__(model_client.SceneClient)
    client.control_token = "t" * 64
    client.socket_path = Path("/tmp/test-scene.sock")

    with pytest.raises(model_client.SceneRequestError) as captured:
        client.request("step", sequence=0, action="wait")

    assert captured.value.error_type == "RECOVERABLE_SCENE_ERROR"
    assert captured.value.error_id == "a" * 32
    assert captured.value.error_origin == "resident"
    assert captured.value.recovery["rolling_checkpoint_action_counts"] == [3]


def test_recovery_signature_binds_action_parameters_and_checkpoint() -> None:
    error = model_client.SceneRequestError(
        error_type="RECOVERABLE_SCENE_ERROR",
        message="RECOVERABLE_SCENE_ERROR",
        error_id="e" * 32,
        recovery={"error_fingerprint": "f" * 64},
    )
    base = model_client._automatic_scene_recovery_signature(
        error=error,
        step_fields={"sequence": 4, "action": "move_forward"},
        checkpoint_action_count=3,
    )

    assert base is not None
    assert base != model_client._automatic_scene_recovery_signature(
        error=error,
        step_fields={"sequence": 4, "action": "move_backward"},
        checkpoint_action_count=3,
    )
    assert base != model_client._automatic_scene_recovery_signature(
        error=error,
        step_fields={"sequence": 4, "action": "move_forward"},
        checkpoint_action_count=2,
    )


class _Provider:
    model = "scene-recovery-test-model"
    maximum_output_tokens = 2048
    thinking_tool_choice_auto = False
    action_tag_compatibility = False

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.seeds: list[int] = []
        self.preflight_calls = 0

    def capability_preflight(
        self,
        *,
        prompt_revision: object,
        artifact: Path,
        **_: object,
    ) -> dict:
        self.preflight_calls += 1
        report = {
            "status": "PASS",
            "model": self.model,
            "prompt_sha256": hashlib.sha256(
                prompt_revision.text.encode("utf-8")
            ).hexdigest(),
            "fixed_tool_schema_sha256": model_client._sha256_file(
                model_client.ACTION20_TOOL_SCHEMA_PATH
            ),
        }
        model_client._write_json(artifact, report)
        return report

    def complete(self, request: dict, *, call_dir: Path, seed: int) -> dict:
        call_dir.mkdir(parents=True, exist_ok=False)
        self.requests.append(request)
        self.seeds.append(seed)
        response = self.responses.pop(0)
        model_client._write_json(call_dir / "response.json", response)
        return response


class _RecoveringScene:
    def __init__(self, root: Path, *, checkpoint_available: bool) -> None:
        self.scene_run_dir = root / "scene"
        image_dir = self.scene_run_dir / "images"
        image_dir.mkdir(parents=True)
        self.images = []
        for camera, color, edge in (
            ("head", (0, 0, 255), 8),
            ("left_wrist", (255, 0, 0), 4),
            ("right_wrist", (0, 255, 0), 4),
        ):
            path = image_dir / f"{camera}.png"
            Image.new("RGB", (edge, edge), color).save(path)
            self.images.append(
                {
                    "camera": camera,
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        self.checkpoint_available = checkpoint_available
        self.active = False
        self.episode_id: str | None = None
        self.next_sequence = 0
        self.begin_ids: list[str] = []
        self.step_calls: list[dict] = []
        self.restore_calls: list[int] = []
        self.failed_once = False

    def _observation(self, observation_id: str) -> dict:
        return {
            "observation_id": observation_id,
            "kind": "physical",
            "images": self.images,
            "marker_targets": [],
        }

    def _recoverable_error(self, retained: list[int]) -> model_client.SceneRequestError:
        return model_client.SceneRequestError(
            error_type="RECOVERABLE_SCENE_ERROR",
            message="RECOVERABLE_SCENE_ERROR",
            error_id="e" * 32,
            recovery={
                "scene_resident": True,
                "session_valid": True,
                "episode_active": False,
                "episode_id": None,
                "next_sequence": 0,
                "rolling_checkpoint_action_counts": retained,
            },
        )

    def request(self, command: str, **fields: object) -> dict:
        if command == "begin_episode":
            assert not self.active
            self.active = True
            self.episode_id = str(fields["episode_id"])
            self.begin_ids.append(self.episode_id)
            self.next_sequence = 0
            return {
                "observation": self._observation(
                    f"{self.episode_id}_initial"
                )
            }
        if command == "step":
            assert self.active
            assert fields["sequence"] == self.next_sequence
            self.step_calls.append(dict(fields))
            if self.checkpoint_available and self.next_sequence == 0:
                self.next_sequence = 1
                return {
                    "action_result": {"status": "completed"},
                    "observation": self._observation("after_wait"),
                    "rolling_checkpoint": {
                        "milestone_action_count": 1,
                        "captured_action_count": 1,
                        "retained_action_counts": [1],
                    },
                }
            if not self.failed_once:
                self.failed_once = True
                self.active = False
                self.episode_id = None
                self.next_sequence = 0
                raise self._recoverable_error(
                    [1] if self.checkpoint_available else []
                )
            raise AssertionError("unexpected step after recovery")
        if command == "restore_checkpoint":
            assert not self.active
            captured = int(fields["captured_action_count"])
            assert captured == 1
            self.restore_calls.append(captured)
            self.active = True
            self.episode_id = self.begin_ids[0]
            self.next_sequence = captured
            return {
                "status": "PASS",
                "episode_id": self.episode_id,
                "next_sequence": captured,
                "restored_action_count": captured,
                "observation": self._observation("restored_action_1"),
            }
        if command == "end_episode":
            assert self.active
            self.active = False
            self.episode_id = None
            self.next_sequence = 0
            return {"status": "PASS"}
        if command == "status":
            return {
                "scene_resident": True,
                "session_valid": True,
                "episode_active": self.active,
                "episode_id": self.episode_id,
                "next_sequence": self.next_sequence,
            }
        raise AssertionError(f"unexpected command: {command}")


@pytest.mark.parametrize(
    ("scene_resident", "session_valid", "episode_active", "expected_status"),
    [
        (True, False, False, "FAIL_SCENE_RELOAD_REQUIRED"),
        (True, True, True, "FAIL_SCENE_RECOVERY_EXHAUSTED"),
    ],
)
def test_fresh_start_rejects_unavailable_scene_before_provider(
    tmp_path: Path,
    scene_resident: bool,
    session_valid: bool,
    episode_active: bool,
    expected_status: str,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    scene = _RecoveringScene(tmp_path, checkpoint_available=False)
    scene.active = episode_active
    scene.episode_id = "foreign_episode" if episode_active else None
    provider = _Provider([])
    commands: list[str] = []

    def request(command: str, **_fields: object) -> dict:
        commands.append(command)
        assert command == "status"
        return {
            "scene_resident": scene_resident,
            "session_valid": session_valid,
            "episode_active": episode_active,
            "episode_id": scene.episode_id,
            "next_sequence": 0,
        }

    scene.request = request  # type: ignore[method-assign]
    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="unavailable_scene",
        instruction="Do not call the provider.",
        maximum_model_calls=10,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == expected_status
    assert provider.preflight_calls == 0
    assert commands == ["status"]
    assert not (output / "provider_preflight.json").exists()
    assert scene.active is episode_active
    recovery = json.loads(
        (output / "scene_recovery_events" / "recovery_0001.json").read_text()
    )
    assert recovery["error_origin"] == "resident"


def test_fresh_start_checks_scene_before_provider_and_begin(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    scene = _RecoveringScene(tmp_path, checkpoint_available=False)
    provider = _Provider([_action("done", "done")])
    events: list[str] = []
    original_request = scene.request
    original_preflight = provider.capability_preflight

    def request(command: str, **fields: object) -> dict:
        events.append(command)
        return original_request(command, **fields)

    def preflight(**kwargs: object) -> dict:
        events.append("provider_preflight")
        return original_preflight(**kwargs)

    scene.request = request  # type: ignore[method-assign]
    provider.capability_preflight = preflight  # type: ignore[method-assign]
    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="ordered_start",
        instruction="Finish.",
        maximum_model_calls=10,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == "DONE"
    assert events[:3] == ["status", "provider_preflight", "begin_episode"]


@pytest.mark.parametrize("failure_point", ["initial_observation", "checkpoint"])
def test_post_begin_initialization_failure_always_writes_summary_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    scene = _RecoveringScene(tmp_path, checkpoint_available=False)
    provider = _Provider([])

    if failure_point == "initial_observation":
        monkeypatch.setattr(
            model_client,
            "_materialize_observation",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("synthetic initial observation failure")
            ),
        )
    else:
        monkeypatch.setattr(
            model_client,
            "_write_resume_checkpoint",
            lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("synthetic initial checkpoint failure")
            ),
        )

    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id=f"pre_loop_{failure_point}",
        instruction="Test pre-loop failure finalization.",
        maximum_model_calls=10,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == "FAIL_CLIENT_INITIALIZATION"
    assert summary["scene_reload_required"] is False
    assert summary["model_calls"] == 0
    assert summary["actions_sent_to_scene"] == 0
    assert scene.active is False
    assert json.loads((output / "summary.json").read_text()) == summary
    unavailable = json.loads(
        (output / "conversation_unavailable.json").read_text()
    )
    assert unavailable["reason"] == "post_begin_client_initialization_failed"


def test_runtime_client_exception_writes_terminal_summary_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    scene = _RecoveringScene(tmp_path, checkpoint_available=True)
    provider = _Provider([_action("wait", "wait")])

    monkeypatch.setattr(
        model_client,
        "_public_tool_result_from_scene_step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("synthetic public result failure")
        ),
    )
    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="runtime_client_failure",
        instruction="Test runtime failure finalization.",
        maximum_model_calls=10,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == "FAIL_CLIENT_RUNTIME"
    assert summary["terminal_client_error_type"] == "ValueError"
    assert summary["scene_reload_required"] is False
    assert summary["actions_sent_to_scene"] == 1
    assert summary["scene_step_attempts"] == 1
    assert summary["resume_available"] is False
    assert scene.active is False
    assert json.loads((output / "summary.json").read_text()) == summary


@pytest.mark.parametrize(
    "error_type",
    ["RECOVERABLE_SCENE_ERROR", "INTEGRATION_ERROR"],
)
def test_step_error_restores_matching_scene_and_conversation_checkpoint(
    tmp_path: Path,
    error_type: str,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    scene = _RecoveringScene(tmp_path, checkpoint_available=True)
    original_error = scene._recoverable_error

    def classified_error(retained: list[int]) -> model_client.SceneRequestError:
        error = original_error(retained)
        error.error_type = error_type
        error.args = (f"{error_type}: {error_type}",)
        return error

    scene._recoverable_error = classified_error  # type: ignore[method-assign]
    provider = _Provider(
        [
            _action("wait", "wait"),
            _action("move_forward", "failed_move"),
            _action("done", "done_after_restore"),
        ]
    )

    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="checkpoint_recovery",
        instruction="Test robust recovery.",
        maximum_model_calls=10,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == "DONE"
    assert summary["automatic_scene_recoveries"] == 1
    assert summary["checkpoint_restores"] == 1
    assert summary["decision_only_recoveries"] == 0
    assert summary["same_scene_restarts"] == 0
    assert summary["scene_reload_required"] is False
    assert summary["actions_sent_to_scene"] == 1
    assert summary["scene_step_attempts"] == 2
    assert scene.restore_calls == [1]
    last_request = json.dumps(provider.requests[-1], sort_keys=True)
    assert "failed_move" not in last_request
    assert "restored a verified checkpoint" in last_request
    recovery = json.loads(
        (output / "scene_recovery_events" / "recovery_0001.json").read_text()
    )
    assert recovery["outcome"] == "rolling_checkpoint_restored"
    assert recovery["details"]["conversation_rewound_to_same_action_count"] is True


def test_error_before_first_checkpoint_resets_same_scene_and_restarts_reasoning(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    scene = _RecoveringScene(tmp_path, checkpoint_available=False)
    provider = _Provider(
        [_action("move_forward", "failed_move"), _action("done", "done")]
    )

    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="reset_recovery",
        instruction="Test same-scene reset.",
        maximum_model_calls=10,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == "DONE"
    assert summary["automatic_scene_recoveries"] == 1
    assert summary["checkpoint_restores"] == 0
    assert summary["decision_only_recoveries"] == 0
    assert summary["same_scene_restarts"] == 1
    assert summary["scene_reload_required"] is False
    assert scene.begin_ids == ["reset_recovery", "reset_recovery_recovery_01"]
    assert "safely reset in the same loaded scene" in json.dumps(
        provider.requests[-1], sort_keys=True
    )


def test_invalid_session_requests_reload_without_blind_retry(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    scene = _RecoveringScene(tmp_path, checkpoint_available=False)
    provider = _Provider([_action("move_forward", "failed_move")])

    original = scene._recoverable_error

    def invalid(_: list[int]) -> model_client.SceneRequestError:
        error = original([])
        error.recovery["session_valid"] = False
        return error

    scene._recoverable_error = invalid  # type: ignore[method-assign]
    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="invalid_session",
        instruction="Test reload boundary.",
        maximum_model_calls=10,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == "FAIL_SCENE_RELOAD_REQUIRED"
    assert summary["scene_reload_required"] is True
    assert summary["automatic_scene_recoveries"] == 0
    assert scene.begin_ids == ["invalid_session"]


def test_protocol_error_rewinds_only_decision_when_scene_was_not_mutated(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    scene = _RecoveringScene(tmp_path, checkpoint_available=False)
    original_request = scene.request

    def request(command: str, **fields: object) -> dict:
        if command == "step" and not scene.failed_once:
            scene.failed_once = True
            scene.step_calls.append(dict(fields))
            raise model_client.SceneRequestError(
                error_type="SCENE_PROTOCOL_ERROR",
                message="SCENE_PROTOCOL_ERROR",
                error_id="p" * 32,
                recovery={
                    "scene_resident": True,
                    "session_valid": True,
                    "episode_active": True,
                    "episode_id": scene.episode_id,
                    "next_sequence": 0,
                    "rolling_checkpoint_action_counts": [],
                },
            )
        return original_request(command, **fields)

    scene.request = request  # type: ignore[method-assign]
    provider = _Provider(
        [_action("move_forward", "bad_decision"), _action("done", "done")]
    )
    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="protocol_recovery",
        instruction="Test protocol recovery.",
        maximum_model_calls=10,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == "DONE"
    assert summary["automatic_scene_recoveries"] == 1
    assert summary["decision_only_recoveries"] == 1
    assert summary["same_scene_restarts"] == 0
    assert scene.begin_ids == ["protocol_recovery"]
    assert "bad_decision" not in json.dumps(provider.requests[-1])
    record = json.loads(
        (output / "scene_recovery_events" / "recovery_0001.json").read_text()
    )
    assert record["outcome"] == "decision_rewound_without_scene_mutation"
    assert record["error_origin"] == "resident"
    assert provider.seeds == [19506, 19507]


def test_protocol_decision_repair_pauses_after_three_without_scene_reset(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    scene = _RecoveringScene(tmp_path, checkpoint_available=False)
    original_request = scene.request

    def request(command: str, **fields: object) -> dict:
        if command == "step":
            scene.step_calls.append(dict(fields))
            raise model_client.SceneRequestError(
                error_type="SCENE_PROTOCOL_ERROR",
                message="SCENE_PROTOCOL_ERROR",
                error_id="q" * 32,
                recovery={
                    "scene_resident": True,
                    "session_valid": True,
                    "episode_active": True,
                    "episode_id": scene.episode_id,
                    "next_sequence": 0,
                    "rolling_checkpoint_action_counts": [],
                },
            )
        return original_request(command, **fields)

    scene.request = request  # type: ignore[method-assign]
    provider = _Provider(
        [_action("move_forward", f"bad_{index}") for index in range(4)]
    )
    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="protocol_exhaustion",
        instruction="Test bounded protocol repair.",
        maximum_model_calls=10,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == "PAUSED_MODEL_PROTOCOL"
    assert summary["terminal_protocol_failure"] == (
        "scene_protocol_decision_repair_exhausted"
    )
    assert summary["automatic_scene_recoveries"] == 3
    assert summary["decision_only_recoveries"] == 3
    assert summary["same_scene_restarts"] == 0
    assert summary["resident_episode_preserved"] is True
    assert summary["resume_available"] is True
    assert scene.active is True
    assert scene.begin_ids == ["protocol_exhaustion"]
    assert len(scene.step_calls) == 4
    assert provider.seeds == [19506, 19507, 19508, 19509]
    checkpoint = json.loads((output / "resume_checkpoint.json").read_text())
    assert checkpoint["state"] == "PAUSED"
    assert checkpoint["next_sequence"] == 0
    assert checkpoint["counters"]["scene_recovery_count"] == 3
    assert checkpoint["counters"]["decision_only_recoveries"] == 3
    assert checkpoint["counters"]["same_scene_restarts"] == 0
    exhausted = json.loads(
        (output / "scene_recovery_events" / "recovery_0004.json").read_text()
    )
    assert exhausted["outcome"] == "decision_repair_budget_exhausted"
    assert exhausted["details"]["physical_scene_mutated"] is False

    resumed = model_client.run_model_episode(
        scene=scene,
        provider=_Provider([_action("done", "done_after_manual_resume")]),
        experiment_id="protocol_exhaustion",
        instruction="Test bounded protocol repair.",
        maximum_model_calls=10,
        physical_history_max=5,
        output_dir=output,
        resume_checkpoint=model_client._load_resume_checkpoint(
            output / "resume_checkpoint.json"
        ),
    )
    assert resumed["status"] == "DONE"
    assert resumed["automatic_scene_recoveries"] == 3
    assert resumed["decision_only_recoveries"] == 3
    assert (output / "scene_recovery_events" / "recovery_0001.json").is_file()
    assert (output / "scene_recovery_events" / "recovery_0004.json").is_file()


def test_same_scene_reset_budget_is_five_and_does_not_reload(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    scene = _RecoveringScene(tmp_path, checkpoint_available=False)
    original_request = scene.request

    def request(command: str, **fields: object) -> dict:
        if command == "step":
            assert scene.active
            scene.step_calls.append(dict(fields))
            scene.active = False
            scene.episode_id = None
            scene.next_sequence = 0
            raise scene._recoverable_error([])
        return original_request(command, **fields)

    scene.request = request  # type: ignore[method-assign]
    provider = _Provider(
        [_action("move_forward", f"failed_{index}") for index in range(6)]
    )
    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="reset_budget",
        instruction="Test bounded same-scene resets.",
        maximum_model_calls=10,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == "FAIL_SCENE_RECOVERY_EXHAUSTED"
    assert summary["scene_reload_required"] is False
    assert summary["automatic_scene_recoveries"] == 5
    assert summary["same_scene_restarts"] == 5
    assert summary["maximum_automatic_same_scene_resets"] == 5
    assert summary["decision_only_recoveries"] == 0
    assert summary["checkpoint_restores"] == 0
    assert scene.begin_ids == [
        "reset_budget",
        "reset_budget_recovery_01",
        "reset_budget_recovery_02",
        "reset_budget_recovery_03",
        "reset_budget_recovery_04",
        "reset_budget_recovery_05",
    ]
    assert len(scene.step_calls) == 6
    active_checkpoint = json.loads(
        (output / "resume_checkpoint.json").read_text()
    )
    assert active_checkpoint["counters"]["same_scene_restarts"] == 5
    assert active_checkpoint["counters"]["scene_recovery_count"] == 5
    exhausted = json.loads(
        (output / "scene_recovery_events" / "recovery_0006.json").read_text()
    )
    assert exhausted["outcome"] == "same_scene_reset_budget_exhausted"


def test_duplicate_scene_error_stops_after_one_same_scene_reset(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    scene = _RecoveringScene(tmp_path, checkpoint_available=False)
    original_request = scene.request

    def request(command: str, **fields: object) -> dict:
        if command == "step":
            assert scene.active
            scene.step_calls.append(dict(fields))
            scene.active = False
            scene.episode_id = None
            scene.next_sequence = 0
            error = scene._recoverable_error([])
            error.recovery["error_fingerprint"] = "f" * 64
            raise error
        return original_request(command, **fields)

    scene.request = request  # type: ignore[method-assign]
    provider = _Provider(
        [_action("move_forward", f"failed_{index}") for index in range(2)]
    )
    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="duplicate_scene_error",
        instruction="Test duplicate recovery suppression.",
        maximum_model_calls=10,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == "FAIL_SCENE_RELOAD_REQUIRED"
    assert summary["scene_reload_required"] is True
    assert summary["automatic_scene_recoveries"] == 1
    assert summary["same_scene_restarts"] == 1
    assert len(scene.step_calls) == 2
    assert scene.begin_ids == [
        "duplicate_scene_error",
        "duplicate_scene_error_recovery_01",
    ]
    suppressed = json.loads(
        (output / "scene_recovery_events" / "recovery_0002.json").read_text()
    )
    assert suppressed["outcome"] == "duplicate_scene_error_recovery_suppressed"
    assert suppressed["details"]["consecutive_occurrences"] == 2


def test_begin_episode_retries_with_new_id_without_reloading_scene(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    scene = _RecoveringScene(tmp_path, checkpoint_available=False)
    original_request = scene.request
    failed = False

    def request(command: str, **fields: object) -> dict:
        nonlocal failed
        if command == "begin_episode" and not failed:
            failed = True
            raise scene._recoverable_error([])
        return original_request(command, **fields)

    scene.request = request  # type: ignore[method-assign]
    summary = model_client.run_model_episode(
        scene=scene,
        provider=_Provider([_action("done", "done")]),
        experiment_id="begin_recovery",
        instruction="Test begin recovery.",
        maximum_model_calls=10,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == "DONE"
    assert summary["automatic_scene_recoveries"] == 1
    assert summary["same_scene_restarts"] == 1
    assert scene.begin_ids == ["begin_recovery_recovery_01"]


def test_lost_step_response_probes_status_before_deciding_to_reload(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    scene = _RecoveringScene(tmp_path, checkpoint_available=False)
    original_request = scene.request
    lost_once = False

    def request(command: str, **fields: object) -> dict:
        nonlocal lost_once
        if command == "step" and not lost_once:
            lost_once = True
            scene.step_calls.append(dict(fields))
            # The socket response was lost, but the server status proves the
            # action sequence did not advance, so replaying the action itself
            # would be unsafe and unnecessary.
            raise model_client.ModelClientError("scene control transport failed")
        return original_request(command, **fields)

    scene.request = request  # type: ignore[method-assign]
    provider = _Provider(
        [_action("move_forward", "lost_response"), _action("done", "done")]
    )
    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="transport_probe",
        instruction="Test transport recovery.",
        maximum_model_calls=10,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == "DONE"
    assert summary["automatic_scene_recoveries"] == 1
    assert summary["scene_reload_required"] is False
    assert scene.begin_ids == ["transport_probe"]
    record = json.loads(
        (output / "scene_recovery_events" / "recovery_0001.json").read_text()
    )
    assert record["outcome"] == "decision_rewound_without_scene_mutation"
    assert record["error_origin"] == "transport"
