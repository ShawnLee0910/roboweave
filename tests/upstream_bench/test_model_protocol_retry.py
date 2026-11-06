from __future__ import annotations

import errno
import hashlib
import io
import http.client
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

from PIL import Image
import pytest


BENCH_TRY_ROOT = (Path(__file__).resolve().parents[2] / "src/roboweave/_vendor/bench")
if str(BENCH_TRY_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_TRY_ROOT))
if str(BENCH_TRY_ROOT / "lab") not in sys.path:
    sys.path.insert(0, str(BENCH_TRY_ROOT / "lab"))

import model_client  # noqa: E402


def test_scene_and_action_after_images_are_hardlinked(tmp_path: Path) -> None:
    scene_run = tmp_path / "scene"
    scene_run.mkdir()
    source = scene_run / "head.png"
    Image.new("RGB", (2, 2), (1, 2, 3)).save(source)
    observation = model_client._materialize_observation(
        {
            "observation_id": "observation_0",
            "kind": "physical",
            "images": [
                {
                    "camera": "head",
                    "path": str(source),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            ],
            "marker_targets": [],
        },
        output_dir=tmp_path / "run",
        scene_run_dir=scene_run,
    )
    assert os.path.samefile(source, observation.images[0].path)

    composite = observation.images[0].path.parent / "three_view_composite.png"
    action_after = model_client._publish_action_after_view(
        observation,
        output_dir=tmp_path / "run",
        step_index=0,
        action_name="initial",
    )
    assert os.path.samefile(composite, action_after)


def test_image_hardlink_falls_back_to_copy_on_cross_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    destination = tmp_path / "destination.png"
    source.write_bytes(b"png")

    def cross_device(*_: object, **__: object) -> None:
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(model_client.os, "link", cross_device)
    model_client._link_or_copy(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert not os.path.samefile(source, destination)


def test_old_resident_marker_completion_is_normalized_to_selection_required() -> None:
    result = model_client._public_tool_result_from_scene_step(
        {"status": "completed"},
        {"kind": "marker"},
        expected_selection_stage=1,
    )
    assert result.to_public_dict() == {
        "status": "selection_required",
        "stage": 1,
    }


def test_new_resident_marker_stage_is_preserved_and_checked() -> None:
    result = model_client._public_tool_result_from_scene_step(
        {"status": "selection_required", "stage": 2},
        {"kind": "marker"},
        expected_selection_stage=2,
    )
    assert result.to_public_dict() == {
        "status": "selection_required",
        "stage": 2,
    }
    with pytest.raises(model_client.ModelClientError, match="stage changed"):
        model_client._public_tool_result_from_scene_step(
            {"status": "selection_required", "stage": 1},
            {"kind": "marker"},
            expected_selection_stage=2,
        )


def test_pick_blocked_category_is_preserved_at_the_client_boundary() -> None:
    result = model_client._public_tool_result_from_scene_step(
        {
            "status": "blocked",
            "blocked_category": "out_of_reach",
        },
        {"kind": "physical"},
        expected_selection_stage=None,
        allow_blocked_category=True,
    )
    assert result.to_public_dict() == {
        "status": "blocked",
        "blocked_category": "out_of_reach",
    }


def test_articulated_part_blocked_is_preserved_at_the_client_boundary() -> None:
    result = model_client._public_tool_result_from_scene_step(
        {
            "status": "blocked",
            "blocked_category": "articulated_part_blocked",
        },
        {"kind": "physical"},
        expected_selection_stage=None,
        allow_blocked_category=True,
    )
    assert result.to_public_dict()["blocked_category"] == (
        "articulated_part_blocked"
    )


def test_old_frozen_schema_does_not_enable_blocked_categories() -> None:
    result = model_client._public_tool_result_from_scene_step(
        {
            "status": "blocked",
            "blocked_category": "target_outside_local_workspace",
        },
        {"kind": "physical"},
        expected_selection_stage=None,
    )
    assert result.to_public_dict() == {"status": "blocked"}


def test_provider_hard_disables_environment_proxy(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _build_opener(*handlers: object) -> object:
        captured["handlers"] = handlers
        return object()

    monkeypatch.setattr(urllib.request, "build_opener", _build_opener)
    provider = model_client.OpenAICompatibleProvider(
        model="qwen3.8-max",
        base_url="https://example.invalid/v1",
        timeout_seconds=30,
        api_key_file=None,
    )

    assert provider.opener is not None
    handlers = captured["handlers"]
    assert isinstance(handlers, tuple)
    assert len(handlers) == 2
    assert isinstance(handlers[0], urllib.request.ProxyHandler)
    assert handlers[0].proxies == {}
    assert isinstance(handlers[1], model_client._NoProviderRedirect)


def test_provider_rewrites_only_the_wire_model() -> None:
    provider = model_client.OpenAICompatibleProvider(
        model="gpt-5.6-sol-max",
        wire_model="gpt-5.6-sol",
        base_url="http://127.0.0.1:1/v1",
        timeout_seconds=30,
        api_key_file=None,
    )
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, _maximum):
            return b'{}'

    class _Opener:
        def open(self, request, timeout):
            captured.update(json.loads(request.data))
            return _Response()

    provider.opener = _Opener()
    provider._post({"model": "gpt-5.6-sol-max"})
    assert captured["model"] == "gpt-5.6-sol"
    assert provider.model == "gpt-5.6-sol-max"


def test_ssh_askpass_reads_only_explicit_environment(monkeypatch, capsys) -> None:
    import ssh_password_askpass

    monkeypatch.delenv(model_client.SSH_PASSWORD_ENV, raising=False)
    assert ssh_password_askpass.main() == 1
    assert capsys.readouterr().out == ""
    monkeypatch.setenv(model_client.SSH_PASSWORD_ENV, "test-only-password")
    assert ssh_password_askpass.main() == 0
    assert capsys.readouterr().out == "test-only-password\n"


def _reasoning_only(reasoning: str = "I need to inspect the scene.") -> dict:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": reasoning,
                    "tool_calls": [],
                },
            }
        ]
    }


def test_content_think_without_action_is_classified_as_reasoning_only() -> None:
    response = _reasoning_only()
    message = response["choices"][0]["message"]
    message.pop("reasoning_content")
    message["content"] = "<think>I need to inspect the scene.</think>"

    assert model_client._native_response_metadata(response) == {
        "finish_reason": "stop",
        "native_reasoning_present": True,
        "tool_call_count": 0,
    }
    assert model_client._native_protocol_failure_type(response) == (
        "missing_tool_call_after_reasoning"
    )


def _single_action_tag(action: str, *, surrounding: str = "") -> dict:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": f"{surrounding}<{action}>{surrounding}",
                    "reasoning": "Use one public action.",
                    "tool_calls": [],
                    "refusal": None,
                    "annotations": None,
                    "audio": None,
                    "function_call": None,
                },
            }
        ]
    }


def _targeted_action_block(
    action: str = "pick",
    *,
    target: int = 1,
    target_description: str = "chicken on a plate",
) -> dict:
    response = _single_action_tag(action)
    response["choices"][0]["message"]["content"] = (
        "\n\n<robot_action>\n"
        "<parameter=action>\n"
        f"{action}\n"
        "</parameter>\n"
        "<parameter=target>\n"
        f"{target}\n"
        "</parameter>\n"
        "<parameter=target_description>\n"
        f"{target_description}\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    return response


def test_exact_single_action_tag_normalizes_without_target() -> None:
    response = _single_action_tag("move_forward", surrounding="\n\n")
    normalized, record = model_client._normalize_single_action_tag_response(
        response
    )
    choice = normalized["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "robot_action"
    assert json.loads(call["function"]["arguments"]) == {
        "action": "move_forward"
    }
    assert record["normalized_action"] == "move_forward"
    assert record["target_synthesized"] is False
    assert response["choices"][0]["message"]["content"] == (
        "\n\n<move_forward>\n\n"
    )


def test_action_tag_compatibility_isolates_content_think() -> None:
    response = _single_action_tag("move_forward")
    message = response["choices"][0]["message"]
    message.pop("reasoning")
    message["content"] = (
        "<think>Move closer to the visible object.</think>\n<move_forward>"
    )

    normalized, record = model_client._normalize_single_action_tag_response(
        response
    )

    normalized_message = normalized["choices"][0]["message"]
    assert normalized_message["content"] == (
        "<think>Move closer to the visible object.</think>"
    )
    assert "reasoning_content" not in normalized_message
    assert record["schema_version"] == (
        "behavior-hl-provider-action-tag-compatibility/v2-reasoning-source"
    )
    assert record["reasoning_source"] == "content_think"
    assert response["choices"][0]["message"]["content"].startswith("<think>")


def test_exact_rynnbrain_targeted_block_normalizes_all_arguments() -> None:
    response = _targeted_action_block()
    normalized, record = model_client._normalize_single_action_tag_response(
        response
    )
    call = normalized["choices"][0]["message"]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {
        "action": "pick",
        "target": 1,
        "target_description": "chicken on a plate",
    }
    assert record["source_syntax"] == "rynnbrain_targeted_action_block_v1"
    assert record["normalized_target"] == 1
    assert record["target_synthesized"] is False
    assert response["choices"][0]["message"]["tool_calls"] == []


def test_targeted_block_rejects_target_for_non_marker_action() -> None:
    with pytest.raises(ValueError):
        model_client._normalize_single_action_tag_response(
            _targeted_action_block("move_forward")
        )


@pytest.mark.parametrize(
    "response",
    [
        _single_action_tag("not_an_action"),
        _single_action_tag("move_forward", surrounding="prose"),
        {
            **_single_action_tag("move_forward"),
            "choices": [
                {
                    **_single_action_tag("move_forward")["choices"][0],
                    "message": {
                        **_single_action_tag("move_forward")["choices"][0][
                            "message"
                        ],
                        "content": "<move_forward><turn_left>",
                    },
                }
            ],
        },
    ],
)
def test_action_tag_compatibility_rejects_non_exact_content(response: dict) -> None:
    with pytest.raises(ValueError):
        model_client._normalize_single_action_tag_response(response)


def _done(call_id: str = "call_done") -> dict:
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "The episode should end.",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "robot_action",
                                "arguments": json.dumps({"action": "done"}),
                            },
                        }
                    ],
                },
            }
        ]
    }


def _action(
    action: str,
    *,
    call_id: str,
    target: int | None = None,
    target_description: str | None = None,
    repeat: int | None = None,
    move_distance_m: float | None = None,
) -> dict:
    arguments: dict[str, object] = {"action": action}
    if target is not None:
        arguments["target"] = target
    if target_description is not None:
        arguments["target_description"] = target_description
    if repeat is not None:
        arguments["repeat"] = repeat
    if move_distance_m is not None:
        arguments["move_distance_m"] = move_distance_m
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
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
            }
        ]
    }


class _Provider:
    model = "rynnbrain1.1-122b-a10b"
    maximum_output_tokens = 2048

    def __init__(self, responses: list[dict | BaseException]) -> None:
        self.responses = list(responses)
        self.seeds: list[int] = []
        self.requests: list[bytes] = []

    def capability_preflight(self, *, artifact: Path, **_: object) -> dict:
        report = {"status": "PASS", "simulator_touched": False}
        model_client._write_json(artifact, report)
        return report

    def complete(self, request: dict, *, call_dir: Path, seed: int) -> dict:
        call_dir.mkdir(parents=True, exist_ok=False)
        self.seeds.append(seed)
        self.requests.append(model_client._canonical_json_bytes(request))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        model_client._write_json(call_dir / "response.json", response)
        return response


class _Scene:
    def __init__(self, root: Path) -> None:
        self.scene_run_dir = root / "scene"
        image_dir = self.scene_run_dir / "initial"
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
        self.end_reasons: list[str] = []

    def request(self, command: str, **fields: object) -> dict:
        if command == "status":
            return {
                "scene_resident": True,
                "session_valid": True,
                "episode_active": False,
                "episode_id": None,
                "next_sequence": 0,
            }
        if command == "begin_episode":
            return {
                "observation": {
                    "observation_id": "initial",
                    "kind": "physical",
                    "images": self.images,
                    "marker_targets": [],
                }
            }
        if command == "end_episode":
            self.end_reasons.append(str(fields["reason"]))
            return {"status": "PASS"}
        raise AssertionError(f"unexpected scene command: {command}")


class _OverheadModeScene(_Scene):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.begin_fields: list[dict[str, object]] = []

    def request(self, command: str, **fields: object) -> dict:
        if command == "begin_episode":
            self.begin_fields.append(dict(fields))
        return super().request(command, **fields)


class _MarkerScene(_Scene):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.step_fields: list[dict[str, object]] = []

    def request(self, command: str, **fields: object) -> dict:
        if command != "step":
            return super().request(command, **fields)
        self.step_fields.append(dict(fields))
        sequence = fields["sequence"]
        if sequence == 0:
            assert fields == {"sequence": 0, "action": "pick"}
            kind = "marker"
            targets = [1, 2]
            observation_id = "pick_markers"
        elif sequence == 1:
            assert fields == {"sequence": 1, "action": "pick", "target": 1}
            kind = "physical"
            targets = []
            observation_id = "after_pick"
        else:
            raise AssertionError(f"unexpected step sequence: {sequence}")
        return {
            "action_result": {"status": "completed"},
            "observation": {
                "observation_id": observation_id,
                "kind": kind,
                "images": self.images,
                "marker_targets": targets,
            },
        }


class _TwoStageMarkerScene(_Scene):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.step_fields: list[dict[str, object]] = []

    def request(self, command: str, **fields: object) -> dict:
        if command != "step":
            return super().request(command, **fields)
        sequence = fields["sequence"]
        self.step_fields.append(dict(fields))
        if sequence == 0:
            assert fields == {"sequence": 0, "action": "place_inside"}
            status = {"status": "selection_required", "stage": 1}
            kind = "marker"
            targets = [1, 2]
            observation_id = "inside_destination_markers"
        elif sequence == 1:
            assert fields == {
                "sequence": 1,
                "action": "place_inside",
                "target": 1,
            }
            status = {"status": "selection_required", "stage": 2}
            kind = "marker"
            targets = [3, 4]
            observation_id = "inside_region_markers"
        elif sequence == 2:
            assert fields == {
                "sequence": 2,
                "action": "place_inside",
                "target": 3,
            }
            status = {"status": "completed"}
            kind = "physical"
            targets = []
            observation_id = "after_place_inside"
        else:
            raise AssertionError(f"unexpected step sequence: {sequence}")
        return {
            "action_result": status,
            "observation": {
                "observation_id": observation_id,
                "kind": kind,
                "images": self.images,
                "marker_targets": targets,
            },
        }


class _OfficialTerminalScene(_Scene):
    def request(self, command: str, **fields: object) -> dict:
        if command != "step":
            return super().request(command, **fields)
        assert fields == {"sequence": 0, "action": "wait"}
        return {
            "action_result": {"status": "completed"},
            "observation": {
                "observation_id": "official_success_after",
                "kind": "physical",
                "images": self.images,
                "marker_targets": [],
            },
            "rolling_checkpoint": None,
            "episode_terminal": True,
            "session_valid": True,
        }


class _BlockedOfficialTerminalScene(_OfficialTerminalScene):
    def request(self, command: str, **fields: object) -> dict:
        result = super().request(command, **fields)
        if command == "step":
            result["action_result"] = {"status": "blocked"}
        return result


class _ScriptedStatusScene(_Scene):
    def __init__(self, root: Path, statuses: list[str]) -> None:
        super().__init__(root)
        self.statuses = list(statuses)
        self.step_fields: list[dict[str, object]] = []

    def request(self, command: str, **fields: object) -> dict:
        if command != "step":
            return super().request(command, **fields)
        sequence = fields["sequence"]
        assert isinstance(sequence, int)
        assert fields == {"sequence": sequence, "action": "wait"}
        self.step_fields.append(dict(fields))
        return {
            "action_result": {"status": self.statuses[sequence]},
            "observation": {
                "observation_id": f"after_wait_{sequence:03d}",
                "kind": "physical",
                "images": self.images,
                "marker_targets": [],
            },
        }


class _RepeatScene(_Scene):
    def __init__(
        self,
        root: Path,
        statuses: list[str],
        *,
        action: str = "turn_right",
        terminal_step: int | None = None,
    ) -> None:
        super().__init__(root)
        self.statuses = statuses
        self.action = action
        self.terminal_step = terminal_step
        self.step_fields: list[dict[str, object]] = []

    def request(self, command: str, **fields: object) -> dict:
        if command != "step":
            return super().request(command, **fields)
        sequence = fields["sequence"]
        assert isinstance(sequence, int)
        assert fields in (
            {"sequence": sequence, "action": self.action},
            {
                "sequence": sequence,
                "action": self.action,
                "navigation_substeps": 2,
            },
        )
        self.step_fields.append(dict(fields))
        terminal = self.terminal_step == sequence + 1
        response = {
            "action_result": {
                "status": self.statuses[sequence],
                "body_odometry": {
                    "forward_m": 0.0,
                    "left_m": 0.0,
                    "yaw_deg": -10.0,
                },
            },
            "observation": {
                "observation_id": f"repeat_{sequence + 1}",
                "kind": "physical",
                "images": self.images,
                "marker_targets": [],
            },
        }
        if self.statuses[sequence] == "blocked":
            response["action_result"]["blocked_category"] = (
                "navigation_blocked"
            )
        if terminal:
            response.update(
                {
                    "episode_terminal": True,
                    "session_valid": True,
                    "rolling_checkpoint": None,
                }
            )
        return response


class _MarkerThenBlockedScene(_Scene):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.step_fields: list[dict[str, object]] = []

    def request(self, command: str, **fields: object) -> dict:
        if command != "step":
            return super().request(command, **fields)
        sequence = fields["sequence"]
        self.step_fields.append(dict(fields))
        if sequence == 0:
            assert fields == {"sequence": 0, "action": "pick"}
            status = "completed"  # Legacy resident Marker spelling.
            kind = "marker"
            targets = [1]
            observation_id = "pick_marker"
        elif sequence == 1:
            assert fields == {"sequence": 1, "action": "pick", "target": 1}
            status = "blocked"
            kind = "physical"
            targets = []
            observation_id = "after_blocked_pick"
        elif sequence == 2:
            assert fields == {"sequence": 2, "action": "wait"}
            status = "blocked"
            kind = "physical"
            targets = []
            observation_id = "after_blocked_wait"
        else:
            raise AssertionError(f"unexpected step sequence: {sequence}")
        return {
            "action_result": {"status": status},
            "observation": {
                "observation_id": observation_id,
                "kind": kind,
                "images": self.images,
                "marker_targets": targets,
            },
        }


def _run(
    tmp_path: Path,
    responses: list[dict | BaseException],
    *,
    maximum_provider_attempts: int = 10,
) -> tuple[dict, _Provider, _Scene, Path]:
    output = tmp_path / "output"
    output.mkdir()
    provider = _Provider(responses)
    scene = _Scene(tmp_path)
    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="retry_test",
        instruction="Test task.",
        maximum_model_calls=30,
        maximum_provider_attempts_per_protocol_attempt=(
            maximum_provider_attempts
        ),
        physical_history_max=5,
        output_dir=output,
    )
    return summary, provider, scene, output


@pytest.mark.parametrize("automatic_approach", [False, True])
def test_model_client_sends_private_episode_modes(
    tmp_path: Path,
    automatic_approach: bool,
) -> None:
    output = tmp_path / f"overhead_mode_{automatic_approach}"
    output.mkdir()
    scene = _OverheadModeScene(tmp_path)

    summary = model_client.run_model_episode(
        scene=scene,
        provider=_Provider([_done()]),
        experiment_id="overhead_mode",
        instruction="Test task.",
        maximum_model_calls=1,
        physical_history_max=5,
        output_dir=output,
        overhead_view_mode="on_blocked",
        automatic_interaction_approach_enabled=automatic_approach,
    )

    assert summary["status"] == "DONE"
    begin_fields = {
        "episode_id": "overhead_mode",
        "overhead_view_mode": "on_blocked",
    }
    if automatic_approach:
        begin_fields["automatic_interaction_approach_enabled"] = True
    assert scene.begin_fields == [begin_fields]
    frozen_schema = json.loads(
        (output / "action20_tool_schema.json").read_text(encoding="utf-8")
    )
    description = frozen_schema["function"]["description"]
    provenance = json.loads((output / "client_manifest.json").read_text())["prompt_provenance"]
    assert provenance["rendered_prompt_sha256"] == hashlib.sha256((output / "system_prompt.txt").read_bytes()).hexdigest()
    assert provenance["tool_schema_sha256"] == hashlib.sha256(json.dumps(
        frozen_schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    assert provenance["source_sha256"] == hashlib.sha256(model_client.ACTION20_CORE_INSTRUCTIONS_PATH.read_bytes()).hexdigest()
    assert provenance["overhead_view_mode"] == "on_blocked"
    assert provenance["automatic_interaction_approach_enabled"] is automatic_approach
    assert ("Automatic interaction approach is enabled" in description) is (
        automatic_approach
    )
    assert "After a blocked action, an overhead top-down view may appear" in (
        description
    )


def test_chatml_is_available_and_grows_during_the_episode(
    tmp_path: Path,
) -> None:
    output = tmp_path / "live_chatml"
    output.mkdir()

    class InspectingProvider(_Provider):
        def __init__(self) -> None:
            super().__init__(
                [
                    _action("wait", call_id="first_action"),
                    _done(),
                ]
            )
            self.snapshots: list[dict] = []

        def complete(self, request: dict, *, call_dir: Path, seed: int) -> dict:
            self.snapshots.append(
                json.loads(
                    (output / "conversation_chatml.json").read_text()
                )
            )
            return super().complete(request, call_dir=call_dir, seed=seed)

    provider = InspectingProvider()
    summary = model_client.run_model_episode(
        scene=_ScriptedStatusScene(tmp_path, ["completed"]),
        provider=provider,
        experiment_id="live_chatml",
        instruction="Test task.",
        maximum_model_calls=2,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == "DONE"
    assert [message["role"] for message in provider.snapshots[0]["messages"]] == [
        "system",
        "user",
    ]
    assert [message["role"] for message in provider.snapshots[1]["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


def test_official_terminal_step_finishes_without_second_scene_request(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    provider = _Provider([_action("wait", call_id="terminal_action")])
    scene = _OfficialTerminalScene(tmp_path)

    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="official_terminal_episode",
        instruction="Turn on the requested device.",
        maximum_model_calls=30,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == "DONE"
    assert summary["terminated_by_official_success"] is True
    assert summary["actions_sent_to_scene"] == 1
    assert summary["macro_actions_resolved"] == 1
    assert summary["resident_episode_preserved"] is False
    assert scene.end_reasons == []
    checkpoint = json.loads(
        (output / "resume_checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["state"] == "TERMINAL"
    assert checkpoint["next_sequence"] == 1
    conversation = json.loads(
        (output / "conversation_private.json").read_text(encoding="utf-8")
    )
    assert conversation["conversation_state"] == "terminal"
    assert conversation["events"][-1] == {
        "event_index": len(conversation["events"]) - 1,
        "kind": "episode_end",
        "termination_source": "official_task_success",
    }


@pytest.mark.parametrize(
    ("statuses", "repeat", "terminal_step", "expected_status"),
    [
        (["completed"] * 4, 4, None, "FAIL_BUDGET_EXHAUSTED"),
        (["completed", "completed", "blocked"], 5, None, "FAIL_BUDGET_EXHAUSTED"),
        (["completed", "completed"], 5, 2, "DONE"),
    ],
)
def test_navigation_repeat_preserves_each_physical_step(
    tmp_path: Path,
    statuses: list[str],
    repeat: int,
    terminal_step: int | None,
    expected_status: str,
) -> None:
    output = tmp_path / "repeat_output"
    output.mkdir()
    provider = _Provider(
        [_action("turn_right", call_id="repeat_turn", repeat=repeat)]
    )
    scene = _RepeatScene(tmp_path, statuses, terminal_step=terminal_step)

    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="repeat_episode",
        instruction="Test navigation repeat.",
        maximum_action_decisions=1,
        maximum_model_calls=1,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == expected_status
    assert summary["model_calls"] == 1
    assert summary["macro_actions_resolved"] == 1
    assert summary["actions_sent_to_scene"] == len(statuses)
    assert scene.step_fields == [
        {"sequence": index, "action": "turn_right"}
        for index in range(len(statuses))
    ]
    transcript = json.loads(
        (output / "conversation_private.json").read_text(encoding="utf-8")
    )
    tool_result = next(
        event for event in transcript["events"] if event["kind"] == "tool_result"
    )
    public_result = json.loads(tool_result["message"]["content"])
    assert public_result["status"] == statuses[-1]
    assert public_result["action_decisions_remaining"] == 0
    assert public_result["repeat_requested"] == repeat
    assert public_result["repeat_executed"] == len(statuses)
    assert [step["status"] for step in public_result["steps"]] == statuses
    if statuses[-1] == "blocked":
        assert public_result["blocked_category"] == (
            "navigation_blocked"
        )
        assert public_result["steps"][-1]["blocked_category"] == (
            "navigation_blocked"
        )
    else:
        assert "blocked_category" not in public_result
    repeated_observations = [
        event
        for event in transcript["events"]
        if event.get("repeat_step_index") is not None
    ]
    assert [event["observation"]["observation_id"] for event in repeated_observations] == [
        f"repeat_{index}" for index in range(1, len(statuses) + 1)
    ]
    assert [event["repeat_step_index"] for event in repeated_observations] == list(
        range(1, len(statuses) + 1)
    )


def test_navigation_distance_preset_batches_safe_substeps_per_public_step(
    tmp_path: Path,
) -> None:
    output = tmp_path / "distance_preset"
    output.mkdir()
    scene = _RepeatScene(
        tmp_path, ["completed"] * 2, action="move_forward"
    )

    summary = model_client.run_model_episode(
        scene=scene,
        provider=_Provider(
            [
                _action(
                    "move_forward",
                    call_id="distance_preset",
                    repeat=2,
                    move_distance_m=0.3,
                )
            ]
        ),
        experiment_id="distance_preset",
        instruction="Test navigation distance preset.",
        maximum_action_decisions=1,
        maximum_model_calls=1,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["actions_sent_to_scene"] == 2
    assert summary["macro_actions_resolved"] == 1
    assert scene.step_fields == [
        {
            "sequence": index,
            "action": "move_forward",
            "navigation_substeps": 2,
        }
        for index in range(2)
    ]


def test_consecutive_blocked_limit_is_terminal_and_defaults_to_thirty(
    tmp_path: Path,
) -> None:
    output = tmp_path / "blocked_limit"
    output.mkdir()
    provider = _Provider(
        [_action("wait", call_id=f"blocked_{index}") for index in range(4)]
    )
    scene = _ScriptedStatusScene(tmp_path, ["blocked", "blocked", "blocked"])

    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="blocked_limit",
        instruction="Test task.",
        maximum_model_calls=10,
        maximum_consecutive_blocked=3,
        physical_history_max=5,
        output_dir=output,
    )

    assert model_client.DEFAULT_MAXIMUM_CONSECUTIVE_BLOCKED == 30
    assert summary["status"] == "FAIL_CONSECUTIVE_BLOCKED"
    assert summary["maximum_consecutive_blocked"] == 3
    assert summary["consecutive_blocked_at_end"] == 3
    assert summary["maximum_consecutive_blocked_observed"] == 3
    assert summary["terminated_by_consecutive_blocked"] is True
    assert summary["terminal_stall_reason"] == "consecutive_blocked_limit"
    assert summary["model_calls"] == 3
    assert summary["actions_sent_to_scene"] == 3
    assert summary["macro_actions_resolved"] == 3
    assert len(provider.responses) == 1
    assert scene.end_reasons == ["budget_exhausted"]
    checkpoint = json.loads(
        (output / "resume_checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["state"] == "TERMINAL"
    transcript = json.loads(
        (output / "conversation_private.json").read_text(encoding="utf-8")
    )
    assert transcript["conversation_state"] == "terminal"
    assert transcript["events"][-1]["termination_reason"] == (
        "consecutive_blocked_limit"
    )


def test_official_success_precedes_the_blocked_streak_limit(tmp_path: Path) -> None:
    output = tmp_path / "blocked_official_success"
    output.mkdir()
    provider = _Provider([_action("wait", call_id="terminal_blocked")])
    scene = _BlockedOfficialTerminalScene(tmp_path)

    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="blocked_official_success",
        instruction="Test task.",
        maximum_model_calls=1,
        maximum_consecutive_blocked=1,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == "DONE"
    assert summary["terminated_by_official_success"] is True
    assert summary["terminated_by_consecutive_blocked"] is False
    assert summary["consecutive_blocked_at_end"] == 1
    assert scene.end_reasons == []


def test_completed_macro_action_resets_the_blocked_streak(tmp_path: Path) -> None:
    output = tmp_path / "blocked_reset"
    output.mkdir()
    provider = _Provider(
        [_action("wait", call_id=f"status_{index}") for index in range(5)]
    )
    scene = _ScriptedStatusScene(
        tmp_path,
        ["blocked", "completed", "blocked", "blocked"],
    )

    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="blocked_reset",
        instruction="Test task.",
        maximum_model_calls=10,
        maximum_consecutive_blocked=2,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == "FAIL_CONSECUTIVE_BLOCKED"
    assert summary["actions_sent_to_scene"] == 4
    assert summary["consecutive_blocked_at_end"] == 2
    assert summary["maximum_consecutive_blocked_observed"] == 2


def test_marker_selection_micro_step_does_not_count_as_a_macro_action(
    tmp_path: Path,
) -> None:
    output = tmp_path / "marker_blocked_limit"
    output.mkdir()
    provider = _Provider(
        [
            _action("pick", call_id="request_pick"),
            _action("pick", call_id="select_pick", target=1),
            _action("wait", call_id="blocked_wait"),
            _done(),
        ]
    )
    scene = _MarkerThenBlockedScene(tmp_path)

    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="marker_blocked_limit",
        instruction="Test task.",
        maximum_model_calls=10,
        maximum_consecutive_blocked=2,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == "FAIL_CONSECUTIVE_BLOCKED"
    assert summary["actions_sent_to_scene"] == 3
    assert summary["macro_actions_resolved"] == 2
    assert summary["consecutive_blocked_at_end"] == 2
    assert len(provider.responses) == 1
    third_request = json.loads(provider.requests[2])
    semantic_outcome = next(
        item["text"]
        for message in third_request["messages"]
        for item in message.get("content", ())
        if isinstance(item, dict)
        and item.get("type") == "text"
        and item.get("text", "").startswith("Previous manipulation outcome: ")
    )
    folded_result = json.loads(
        semantic_outcome.removeprefix("Previous manipulation outcome: ")
    )
    assert folded_result["action_decisions_remaining"] == 149


def test_two_call_marker_transaction_consumes_one_public_decision(
    tmp_path: Path,
) -> None:
    output = tmp_path / "two_call_budget"
    output.mkdir()
    provider = _Provider(
        [
            _action("pick", call_id="request_pick"),
            _action("pick", call_id="select_pick", target=1),
            _action("wait", call_id="must_not_run"),
        ]
    )
    scene = _MarkerScene(tmp_path)

    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="two_call_budget",
        instruction="Test task.",
        maximum_action_decisions=1,
        maximum_model_calls=10,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == "FAIL_BUDGET_EXHAUSTED"
    assert summary["macro_actions_resolved"] == 1
    assert summary["maximum_action_decisions"] == 1
    assert summary["model_calls"] == 2
    assert summary["actions_sent_to_scene"] == 2
    assert summary["public_budget_exhausted"] is True
    assert len(provider.responses) == 1
    transcript = json.loads(
        (output / "conversation_private.json").read_text(encoding="utf-8")
    )
    public_results = [
        json.loads(event["message"]["content"])
        for event in transcript["events"]
        if event.get("kind") == "tool_result"
    ]
    assert [
        result["action_decisions_remaining"] for result in public_results
    ] == [0, 0]


def test_three_call_marker_transaction_consumes_one_public_decision(
    tmp_path: Path,
) -> None:
    output = tmp_path / "three_call_budget"
    output.mkdir()
    provider = _Provider(
        [
            _action("place_inside", call_id="request_inside"),
            _action("place_inside", call_id="select_destination", target=1),
            _action("place_inside", call_id="select_region", target=3),
            _action("wait", call_id="must_not_run"),
        ]
    )
    scene = _TwoStageMarkerScene(tmp_path)

    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="three_call_budget",
        instruction="Test task.",
        maximum_action_decisions=1,
        maximum_model_calls=10,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == "FAIL_BUDGET_EXHAUSTED"
    assert summary["macro_actions_resolved"] == 1
    assert summary["model_calls"] == 3
    assert summary["actions_sent_to_scene"] == 3
    assert summary["public_budget_exhausted"] is True
    assert len(provider.responses) == 1
    transcript = json.loads(
        (output / "conversation_private.json").read_text(encoding="utf-8")
    )
    public_results = [
        json.loads(event["message"]["content"])
        for event in transcript["events"]
        if event.get("kind") == "tool_result"
    ]
    assert [
        result["action_decisions_remaining"] for result in public_results
    ] == [0, 0, 0]


def test_protocol_retry_does_not_consume_public_action_budget(
    tmp_path: Path,
) -> None:
    output = tmp_path / "retry_public_budget"
    output.mkdir()
    provider = _Provider(
        [
            _reasoning_only("retry me"),
            _action("wait", call_id="resolved_wait"),
            _action("wait", call_id="must_not_run"),
        ]
    )
    scene = _ScriptedStatusScene(tmp_path, ["completed"])

    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="retry_public_budget",
        instruction="Test task.",
        maximum_action_decisions=1,
        maximum_model_calls=10,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == "FAIL_BUDGET_EXHAUSTED"
    assert summary["macro_actions_resolved"] == 1
    assert summary["model_calls"] == 2
    assert summary["invalid_model_responses"] == 1
    assert summary["protocol_retries"] == 1
    assert len(provider.responses) == 1
    transcript = json.loads(
        (output / "conversation_private.json").read_text(encoding="utf-8")
    )
    tool_result = next(
        event for event in transcript["events"] if event.get("kind") == "tool_result"
    )
    assert json.loads(tool_result["message"]["content"])[
        "action_decisions_remaining"
    ] == 0


def test_reasoning_length_gets_one_repair_without_scene_action(tmp_path: Path) -> None:
    response = _reasoning_only("truncated reasoning")
    response["choices"][0]["finish_reason"] = "length"
    summary, provider, scene, output = _run(
        tmp_path,
        [response, _done()],
    )

    assert summary["status"] == "DONE"
    assert summary["model_calls"] == 2
    assert summary["protocol_retries"] == 1
    assert summary["terminal_protocol_failure"] is None
    assert scene.end_reasons == ["done"]
    first, repaired = map(json.loads, provider.requests)
    assert repaired['messages'][:-1] == first['messages']
    assert repaired['tools'] == first['tools']
    notice = repaired['messages'][-1]['content'][0]['text']
    assert 'No robot action was dispatched' in notice
    assert 'C-label' not in notice
    assert summary['actions_sent_to_scene'] == 0
    attempt = json.loads(
        (output / "provider_calls" / "call_0001" / "protocol_attempt.json").read_text()
    )
    assert attempt["finish_reason"] == "length"
    assert attempt["failure_type"] == "reasoning_budget_exhausted"
    assert attempt["retry_allowed"] is True


def test_repeated_length_pauses_after_one_repair(tmp_path: Path) -> None:
    response = _reasoning_only('truncated')
    response['choices'][0]['finish_reason'] = 'length'
    summary, provider, scene, output = _run(tmp_path, [response, response, _done()])
    assert summary['model_calls'] == 2
    assert summary['protocol_retries'] == 1
    assert summary['actions_sent_to_scene'] == 0
    assert summary['terminal_protocol_failure'] == 'reasoning_budget_exhausted'
    assert len(provider.responses) == 1
    attempt = json.loads((output / 'provider_calls/call_0002/protocol_attempt.json').read_text())
    assert attempt['retry_allowed'] is False
    assert attempt['length_repair_index'] == 1


def test_marker_length_repair_preserves_current_selection(tmp_path: Path) -> None:
    response = _reasoning_only('truncated marker reasoning')
    response['choices'][0]['finish_reason'] = 'length'
    provider = _Provider([_action('pick', call_id='pick0'), response,
                          _action('pick', target=1, call_id='pick1'), _done()])
    scene = _MarkerScene(tmp_path)
    output = tmp_path / 'output'; output.mkdir()
    summary = model_client.run_model_episode(scene=scene, provider=provider,
        experiment_id='marker_repair', instruction='Pick a target.',
        maximum_model_calls=10, physical_history_max=5, output_dir=output)
    assert summary['status'] == 'DONE'
    assert summary['protocol_retries'] == 1
    assert scene.step_fields == [{'sequence': 0, 'action': 'pick'},
                                 {'sequence': 1, 'action': 'pick', 'target': 1}]
    first, repaired = map(json.loads, provider.requests[1:3])
    assert first['messages'] == repaired['messages'][:-1]
    assert 'Selection is pending for pick' in repaired['messages'][-1]['content'][0]['text']


def test_length_with_tool_is_not_replayed(tmp_path: Path) -> None:
    response = _done()
    response['choices'][0]['finish_reason'] = 'length'
    summary, provider, _, _ = _run(tmp_path, [response])
    assert summary['status'] == 'FAIL_MODEL_PROTOCOL'
    assert summary['model_calls'] == 1
    assert summary['protocol_retries'] == 0
    assert summary['actions_sent_to_scene'] == 0


def test_bare_marker_schema_preserves_cli_contract():
    original = model_client.load_action20_tool_schema()
    bare = model_client._bare_marker_schema(original)
    assert 'pointer endpoint' in bare['function']['description']
    assert 'pointer endpoint' not in original['function']['description']
    assert bare['function']['parameters'] == original['function']['parameters']
    assert model_client.load_action20_tool_schema() == original


def test_length_without_visible_reasoning_still_gets_repair(tmp_path: Path) -> None:
    response = {'choices': [{'finish_reason': 'length',
                             'message': {'role': 'assistant', 'content': None}}]}
    summary, _, _, _ = _run(tmp_path, [response, _done()])
    assert summary['status'] == 'DONE'
    assert summary['protocol_retries'] == 1


def test_length_repair_failure_does_not_open_more_retries(tmp_path: Path) -> None:
    response = _reasoning_only('truncated')
    response['choices'][0]['finish_reason'] = 'length'
    summary, provider, _, _ = _run(tmp_path, [response, _reasoning_only('still no call'), _done()])
    assert summary['model_calls'] == 2
    assert summary['protocol_retries'] == 1
    assert len(provider.responses) == 1


def test_length_repair_respects_model_call_budget(tmp_path: Path) -> None:
    response = _reasoning_only('truncated')
    response['choices'][0]['finish_reason'] = 'length'
    provider = _Provider([response, _done()])
    scene = _Scene(tmp_path)
    output = tmp_path / 'output'; output.mkdir()
    summary = model_client.run_model_episode(scene=scene, provider=provider,
        experiment_id='no_remaining_calls', instruction='Test.',
        maximum_model_calls=1, physical_history_max=5, output_dir=output)
    assert summary['model_calls'] == 1
    assert summary['protocol_retries'] == 0
    assert len(provider.responses) == 1


def test_internal_model_call_cap_is_not_reported_as_public_budget(
    tmp_path: Path,
) -> None:
    output = tmp_path / "internal_call_cap"
    output.mkdir()
    provider = _Provider([_action("wait", call_id="only_call")])
    scene = _ScriptedStatusScene(tmp_path, ["completed"])

    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="internal_call_cap",
        instruction="Test task.",
        maximum_action_decisions=3,
        maximum_model_calls=1,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == "FAIL_INTERNAL_MODEL_CALL_CAP"
    assert summary["macro_actions_resolved"] == 1
    assert summary["public_budget_exhausted"] is False
    assert summary["internal_model_call_cap_exhausted"] is True
    assert summary["terminal_protocol_failure"] == (
        "internal_model_call_cap_exhausted"
    )


def test_reasoning_only_retries_twice_without_changing_request(tmp_path: Path) -> None:
    summary, provider, scene, output = _run(
        tmp_path,
        [_reasoning_only("first"), _reasoning_only("second"), _done()],
    )

    assert summary["status"] == "DONE"
    assert summary["model_calls"] == 3
    assert summary["invalid_model_responses"] == 2
    assert summary["protocol_retries"] == 2
    assert summary["decisions_recovered_after_retry"] == 1
    assert summary["terminal_protocol_failure"] is None
    assert summary["maximum_consecutive_blocked"] == 30
    assert provider.seeds == [19506, 19507, 19508]
    assert len(set(provider.requests)) == 1
    assert scene.end_reasons == ["done"]
    manifest = json.loads((output / "client_manifest.json").read_text())
    assert manifest["maximum_consecutive_blocked"] == 30
    transcript = json.loads((output / "conversation_private.json").read_text())
    assistant_events = [
        event for event in transcript["events"]
        if event["kind"] == "assistant_tool_call"
    ]
    assert len(assistant_events) == 1
    assert assistant_events[0]["message"]["tool_calls"][0]["function"][
        "arguments"
    ] == '{"action":"done"}'
    chatml_path = output / "conversation_chatml.json"
    assert chatml_path.is_file()
    chatml = json.loads(chatml_path.read_text())
    assert chatml["type"] == "chatml"
    assert [message["role"] for message in chatml["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert chatml["messages"][1]["content"][1]["image"] == (
        "observations/initial/three_view_composite.png"
    )
    assistant_text = "".join(
        item["text"] for item in chatml["messages"][2]["content"]
    )
    assert "<think>" in assistant_text
    assert "<tool_call>" in assistant_text
    assert "<parameter=action>\ndone\n</parameter>" in assistant_text
    assert "<tool_response>" in chatml["messages"][3]["content"][0]["text"]
    serialized = json.dumps(chatml)
    assert "raw_response" not in serialized
    assert "private_episode_contract" not in serialized
    selected_evidence = json.loads(
        (
            output
            / "private"
            / "selected_target_visual_evidence.json"
        ).read_text()
    )
    assert selected_evidence["record_count"] == 0
    assert selected_evidence["ground_truth_semantics_included"] is False


def test_episode_can_opt_in_to_exact_action_tag_compatibility(
    tmp_path: Path,
) -> None:
    output = tmp_path / "action_tag_output"
    output.mkdir()
    response = _single_action_tag("done")
    response_message = response["choices"][0]["message"]
    response_message.pop("reasoning")
    raw_content = "<think>The episode should end.</think>\n<done>"
    response_message["content"] = raw_content
    provider = _Provider([response])
    provider.thinking_tool_choice_auto = True
    provider.action_tag_compatibility = True
    scene = _Scene(tmp_path)
    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="action_tag_compatibility",
        instruction="Test task.",
        maximum_model_calls=1,
        physical_history_max=5,
        output_dir=output,
    )
    assert summary["status"] == "DONE"
    assert summary["invalid_model_responses"] == 0
    assert summary["action_tag_compatibility"] is True
    assert summary["action_tag_compatibility_applied"] == 1
    attempt = json.loads(
        (output / "provider_calls/call_0001/protocol_attempt.json").read_text()
    )
    assert attempt["action_tag_compatibility_applied"] is True
    normalization = json.loads(
        (
            output
            / "provider_calls/call_0001/compatibility_normalization.json"
        ).read_text()
    )
    assert normalization["schema_version"] == (
        "behavior-hl-provider-action-tag-compatibility/v2-reasoning-source"
    )
    assert normalization["reasoning_source"] == "content_think"
    assert normalization["normalized_action"] == "done"
    raw = json.loads(
        (output / "provider_calls/call_0001/response.json").read_text()
    )
    assert raw["choices"][0]["message"]["content"] == raw_content
    transcript = json.loads((output / "conversation_private.json").read_text())
    assistant_event = next(
        event
        for event in transcript["events"]
        if event["kind"] == "assistant_tool_call"
    )
    assert assistant_event["native_reasoning_source"] == "content_think"
    assert assistant_event["raw_response"] == raw


def test_targeted_action_block_still_requires_current_marker_identity(
    tmp_path: Path,
) -> None:
    output = tmp_path / "targeted_action_tag_output"
    output.mkdir()
    provider = _Provider(
        [
            _single_action_tag("pick"),
            _targeted_action_block(target=1),
            _single_action_tag("done"),
        ]
    )
    provider.thinking_tool_choice_auto = True
    provider.action_tag_compatibility = True
    scene = _MarkerScene(tmp_path)
    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="targeted_action_tag_compatibility",
        instruction="Test task.",
        maximum_model_calls=3,
        physical_history_max=5,
        output_dir=output,
    )
    assert summary["status"] == "DONE"
    assert summary["actions_sent_to_scene"] == 2
    assert summary["macro_actions_resolved"] == 1
    assert summary["action_tag_compatibility_applied"] == 3
    assert scene.step_fields == [
        {"sequence": 0, "action": "pick"},
        {"sequence": 1, "action": "pick", "target": 1},
    ]


def test_targeted_action_block_rejects_stale_marker_without_scene_mutation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "stale_targeted_action_tag_output"
    output.mkdir()
    provider = _Provider(
        [_single_action_tag("pick")]
        + [_targeted_action_block(target=3) for _ in range(10)]
    )
    provider.thinking_tool_choice_auto = True
    provider.action_tag_compatibility = True
    scene = _MarkerScene(tmp_path)
    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="stale_targeted_action_tag_compatibility",
        instruction="Test task.",
        maximum_model_calls=11,
        physical_history_max=5,
        output_dir=output,
    )
    assert summary["status"] == "PAUSED_MODEL_PROTOCOL"
    assert summary["actions_sent_to_scene"] == 1
    assert summary["macro_actions_resolved"] == 0
    assert summary["invalid_model_responses"] == 4
    assert summary["protocol_retries"] == 3
    assert summary["marker_selection_retries"] == 3
    assert summary["resident_episode_preserved"] is True
    assert summary["terminal_protocol_failure"] == (
        "marker_selection_repair_exhausted"
    )
    assert scene.step_fields == [{"sequence": 0, "action": "pick"}]
    assert scene.end_reasons == []
    assert provider.seeds == [19506, 19506, 19507, 19508, 19509]
    original_selection_request = json.loads(provider.requests[1])
    repaired_request = json.loads(provider.requests[2])
    assert repaired_request["messages"][:-1] == (
        original_selection_request["messages"]
    )
    assert "current Marker observation are unchanged" in json.dumps(
        repaired_request,
        ensure_ascii=False,
    )
    checkpoint = model_client._load_resume_checkpoint(
        output / "resume_checkpoint.json"
    )
    assert checkpoint["state"] == "PAUSED"
    assert checkpoint["next_sequence"] == 1


def test_generic_task_identity_is_frozen_in_model_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "generic_task_output"
    output.mkdir()
    provider = _Provider([_done()])
    scene = _Scene(tmp_path)
    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="task0_identity",
        instruction="Turn on the radio sitting on the table.",
        task_id=0,
        task_name="turning_on_radio",
        task_split="train",
        task_instance_id="0",
        task_seed=19506,
        maximum_model_calls=1,
        physical_history_max=5,
        output_dir=output,
    )
    assert summary["status"] == "DONE"
    manifest = json.loads((output / "client_manifest.json").read_text())
    assert manifest["task_id"] == 0
    assert manifest["task_name"] == "turning_on_radio"
    assert manifest["task_split"] == "train"
    contract = json.loads((output / "conversation_private.json").read_text())[
        "private_episode_contract"
    ]
    assert contract["task_id"] == "0"
    assert contract["split"] == "train"
    assert contract["instance_id"] == "0"
    assert contract["instruction"] == "Turn on the radio sitting on the table."


def _write_instruction_review(
    path: Path,
    *,
    task_id: int = 25,
    task_name: str = "clearing_food_from_table_into_fridge",
    instruction: str = "Reviewed Task25 instruction.",
) -> None:
    model_client._write_json(
        path,
        {
            "schema_version": (
                "behavior2026.certified53-public-instruction-rewrites/v1"
            ),
            "status": "DRAFT_FOR_HUMAN_REVIEW",
            "deterministic_public_audit": {"status": "PASS"},
            "rewrites": [
                {
                    "task_id": task_id,
                    "task_name": task_name,
                    "instruction": instruction,
                }
            ],
        },
    )


def test_reviewed_instruction_is_selected_and_frozen(tmp_path: Path) -> None:
    review = tmp_path / "instructions.json"
    _write_instruction_review(review)
    instruction, provenance = model_client._load_review_instruction(
        review,
        task_id=25,
        task_name="clearing_food_from_table_into_fridge",
    )
    assert instruction == "Reviewed Task25 instruction."
    assert provenance == {
        "kind": "explicit_instruction_review",
        "source_path": str(review.resolve()),
        "source_sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
        "review_status": "DRAFT_FOR_HUMAN_REVIEW",
        "reviewer": None,
        "reviewed_at": None,
        "scope_task_ids": None,
        "rewrite_count": 1,
        "deterministic_public_audit": "PASS",
        "selection_authorization": "explicit_cli_instruction_review",
    }

    output = tmp_path / "reviewed_output"
    output.mkdir()
    summary = model_client.run_model_episode(
        scene=_Scene(tmp_path),
        provider=_Provider([_done()]),
        experiment_id="reviewed_instruction",
        instruction=instruction,
        instruction_provenance=provenance,
        maximum_model_calls=1,
        physical_history_max=5,
        output_dir=output,
    )
    assert summary["status"] == "DONE"
    manifest = json.loads((output / "client_manifest.json").read_text())
    assert manifest["instruction_sha256"] == hashlib.sha256(
        instruction.encode("utf-8")
    ).hexdigest()
    assert manifest["instruction_provenance"] == provenance


def test_instruction_review_rejects_wrong_or_ambiguous_task(tmp_path: Path) -> None:
    review = tmp_path / "instructions.json"
    _write_instruction_review(review, task_id=0, task_name="turning_on_radio")
    with pytest.raises(model_client.ModelClientError):
        model_client._load_review_instruction(
            review,
            task_id=25,
            task_name="clearing_food_from_table_into_fridge",
        )

    payload = json.loads(review.read_text())
    payload["rewrites"].append(dict(payload["rewrites"][0]))
    model_client._write_json(review, payload)
    with pytest.raises(model_client.ModelClientError):
        model_client._load_review_instruction(
            review,
            task_id=0,
            task_name="turning_on_radio",
        )


def test_ten_invalid_responses_fail_without_scene_step(tmp_path: Path) -> None:
    summary, provider, scene, output = _run(
        tmp_path,
        [_reasoning_only(f"invalid-{index}") for index in range(10)],
    )

    assert summary["status"] == "FAIL_MODEL_PROTOCOL"
    assert summary["model_calls"] == 10
    assert summary["actions_sent_to_scene"] == 0
    assert summary["invalid_model_responses"] == 10
    assert summary["protocol_retries"] == 9
    assert summary["decisions_recovered_after_retry"] == 0
    assert summary["terminal_protocol_failure"] == "missing_tool_call_after_reasoning"
    assert summary["maximum_protocol_attempts_per_decision"] == 10
    assert summary["maximum_provider_attempts_per_protocol_attempt"] == 10
    assert provider.seeds == list(range(19506, 19516))
    assert scene.end_reasons == ["client_error"]
    tenth = json.loads(
        (output / "provider_calls" / "call_0010" / "protocol_attempt.json").read_text()
    )
    assert tenth["retry_allowed"] is False
    assert tenth["attempt_index"] == 10
    assert tenth["maximum_attempts"] == 10
    assert tenth["failure_type"] == "missing_tool_call_after_reasoning"
    chatml = json.loads((output / "conversation_chatml.json").read_text())
    assert [message["role"] for message in chatml["messages"]] == [
        "system",
        "user",
    ]


def test_model_client_keeps_description_out_of_scene_and_writes_private_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    provider = _Provider(
        [
            _action("pick", call_id="call_request"),
            _action(
                "pick",
                call_id="call_select",
                target=1,
                target_description="roasted chicken on white plate",
            ),
            _done(),
        ]
    )
    scene = _MarkerScene(tmp_path)

    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="marker_evidence_test",
        instruction="Test task.",
        maximum_model_calls=3,
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == "DONE"
    assert scene.step_fields == [
        {"sequence": 0, "action": "pick"},
        {"sequence": 1, "action": "pick", "target": 1},
    ]
    tool_schemas = [
        json.loads(request)["tools"] for request in provider.requests
    ]
    assert all(schema == tool_schemas[0] for schema in tool_schemas)
    evidence = json.loads(
        (
            output
            / "private"
            / "selected_target_visual_evidence.json"
        ).read_text()
    )
    assert evidence["record_count"] == 1
    record = evidence["records"][0]
    assert record["selected_marker"] == "C1"
    assert record["target_description"] == (
        "roasted chicken on white plate"
    )
    assert record["status"] == "completed"
    assert record["stage"] == 1
    assert record["marker_frame"]["observation"]["observation_id"] == (
        "pick_markers"
    )


def test_retryable_provider_failures_retry_same_decision_and_recover(
    tmp_path: Path,
) -> None:
    summary, provider, scene, output = _run(
        tmp_path,
        [
            model_client.ProviderRequestError(
                "provider HTTP request failed with status 500",
                failure_type="provider_http_500",
                retryable=True,
                status_code=500,
                response_body=b"internal sk-not-a-real-secret-12345678",
            ),
            model_client.ProviderRequestError(
                "provider request timed out",
                failure_type="provider_timeout",
                retryable=True,
            ),
            _done(),
        ],
    )

    assert summary["status"] == "DONE"
    assert summary["model_calls"] == 3
    assert summary["provider_failures"] == 2
    assert summary["provider_retries"] == 2
    assert summary["decisions_recovered_after_provider_retry"] == 1
    assert summary["terminal_provider_failure"] is None
    assert provider.seeds == [19506, 19506, 19506]
    assert len(set(provider.requests)) == 1
    assert scene.end_reasons == ["done"]
    first = json.loads(
        (output / "provider_calls" / "call_0001" / "provider_error.json").read_text()
    )
    assert first["http_status"] == 500
    assert first["retry_allowed"] is True
    assert first["provider_attempt_index"] == 1
    assert first["maximum_provider_attempts_per_protocol_attempt"] == 10
    assert first["protocol_attempt_index"] == 1
    assert first["response_body_preview"] == "internal sk-<redacted>"


def test_enabled_provider_backoff_waits_without_changing_the_request(
    monkeypatch,
    tmp_path: Path,
) -> None:
    waits: list[float] = []
    monkeypatch.setattr(model_client.time, "sleep", waits.append)
    output = tmp_path / "output"
    output.mkdir()
    provider = _Provider(
        [
            model_client.ProviderRequestError(
                "temporary disconnect",
                failure_type="provider_transport_error",
                retryable=True,
            ),
            model_client.ProviderRequestError(
                "temporary overload",
                failure_type="provider_http_503",
                retryable=True,
                status_code=503,
            ),
            _done(),
        ]
    )
    scene = _Scene(tmp_path)
    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="provider_backoff_test",
        instruction="Test task.",
        maximum_model_calls=30,
        maximum_provider_attempts_per_protocol_attempt=10,
        provider_retry_policy={
            "initial_backoff_seconds": 2.0,
            "maximum_backoff_seconds": 30.0,
            "maximum_elapsed_seconds": 600.0,
            "jitter_ratio": 0.2,
        },
        physical_history_max=5,
        output_dir=output,
    )

    assert summary["status"] == "DONE"
    assert len(waits) == 2
    assert 1.6 <= waits[0] <= 2.4
    assert 3.2 <= waits[1] <= 4.8
    assert len(set(provider.requests)) == 1
    first = json.loads(
        (output / "provider_calls" / "call_0001" / "provider_error.json")
        .read_text()
    )
    assert first["retry_delay_seconds"] == waits[0]
    assert first["retry_time_budget_exhausted"] is False


def test_model_specific_provider_retry_limit_stops_after_three_attempts(
    tmp_path: Path,
) -> None:
    errors = [
        model_client.ProviderRequestError(
            "provider request timed out",
            failure_type="provider_timeout",
            retryable=True,
        )
        for _ in range(3)
    ]
    summary, provider, scene, output = _run(
        tmp_path,
        errors,
        maximum_provider_attempts=3,
    )

    assert summary["status"] == "FAIL_PROVIDER"
    assert summary["model_calls"] == 3
    assert summary["provider_failures"] == 3
    assert summary["provider_retries"] == 2
    assert summary["maximum_provider_attempts_per_protocol_attempt"] == 3
    assert scene.end_reasons == ["client_error"]
    assert provider.seeds == [19506, 19506, 19506]
    third = json.loads(
        (output / "provider_calls" / "call_0003" / "provider_error.json").read_text()
    )
    assert third["maximum_attempts"] == 3
    assert third["maximum_provider_attempts_per_protocol_attempt"] == 3
    assert third["retry_allowed"] is False


def test_ten_retryable_provider_failures_stop_without_scene_step(
    tmp_path: Path,
) -> None:
    errors = [
        model_client.ProviderRequestError(
            "provider transport request failed",
            failure_type="provider_transport_error",
            retryable=True,
        )
        for _ in range(10)
    ]
    summary, provider, scene, output = _run(tmp_path, errors)

    assert summary["status"] == "FAIL_PROVIDER"
    assert summary["model_calls"] == 10
    assert summary["actions_sent_to_scene"] == 0
    assert summary["provider_failures"] == 10
    assert summary["provider_retries"] == 9
    assert summary["terminal_provider_failure"] == "provider_transport_error"
    assert provider.seeds == [19506] * 10
    assert scene.end_reasons == ["client_error"]
    tenth = json.loads(
        (output / "provider_calls" / "call_0010" / "provider_error.json").read_text()
    )
    assert tenth["retry_allowed"] is False
    assert tenth["provider_attempt_index"] == 10
    assert tenth["maximum_attempts"] == 10
    assert tenth["protocol_attempt_index"] == 1


def test_paused_client_opens_five_provider_retry_windows(
    monkeypatch,
    tmp_path: Path,
) -> None:
    waits: list[float] = []
    monkeypatch.setattr(model_client.time, "sleep", waits.append)
    output = tmp_path / "provider_windows"
    output.mkdir()
    provider = _Provider(
        [
            model_client.ProviderRequestError(
                "provider request timed out",
                failure_type="provider_timeout",
                retryable=True,
            )
            for _ in range(6)
        ]
    )
    scene = _Scene(tmp_path)

    summary = model_client.run_model_episode(
        scene=scene,
        provider=provider,
        experiment_id="provider_windows",
        instruction="Test task.",
        maximum_model_calls=30,
        maximum_provider_attempts_per_protocol_attempt=1,
        physical_history_max=5,
        output_dir=output,
        pause_on_client_failure=True,
    )

    assert summary["status"] == "PAUSED_PROVIDER"
    assert summary["model_calls"] == 6
    assert summary["provider_retries"] == 5
    assert summary["automatic_provider_retry_windows"] == 5
    assert summary["consecutive_provider_retry_windows_at_end"] == 5
    assert waits == [60.0] * 5
    assert scene.end_reasons == []
    assert model_client._load_provider_retry_state(output) == (5, 5)
    last = json.loads(
        (output / "provider_calls/call_0006/provider_error.json").read_text()
    )
    assert last["retry_allowed"] is False
    assert last["provider_retry_window_index"] == 5
    assert last["automatic_retry_window_allowed"] is False


def test_nonretryable_provider_failure_stops_after_one_attempt(
    tmp_path: Path,
) -> None:
    summary, provider, scene, output = _run(
        tmp_path,
        [
            model_client.ProviderRequestError(
                "provider HTTP request failed with status 400",
                failure_type="provider_http_400",
                retryable=False,
                status_code=400,
                response_body=b"bad request",
            )
        ],
    )

    assert summary["status"] == "FAIL_PROVIDER"
    assert summary["model_calls"] == 1
    assert summary["provider_failures"] == 1
    assert summary["provider_retries"] == 0
    assert summary["terminal_provider_failure"] == "provider_http_400"
    assert provider.seeds == [19506]
    assert scene.end_reasons == ["client_error"]
    error = json.loads(
        (output / "provider_calls" / "call_0001" / "provider_error.json").read_text()
    )
    assert error["http_status"] == 400
    assert error["retry_allowed"] is False


def test_http_500_keeps_only_redacted_request_manifest_and_body_hash(
    tmp_path: Path,
) -> None:
    provider = model_client.OpenAICompatibleProvider(
        model="rynnbrain1.1-122b-a10b",
        base_url="http://127.0.0.1:18133/v1",
        timeout_seconds=30,
        api_key_file=None,
    )

    sent: dict[str, bytes] = {}

    class _FailingOpener:
        def open(self, request: object, **__: object) -> object:
            sent["body"] = request.data
            raise urllib.error.HTTPError(
                "http://127.0.0.1:18133/v1/chat/completions",
                500,
                "Internal Server Error",
                {},
                io.BytesIO(b"assertion failed"),
            )

    provider.opener = _FailingOpener()
    with pytest.raises(model_client.ProviderRequestError) as captured:
        provider.complete(
            {
                "model": provider.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AAAA"},
                            }
                        ],
                    }
                ],
                "tools": [],
            },
            call_dir=tmp_path / "call",
            seed=19506,
        )
    assert captured.value.failure_type == "provider_http_500"
    assert captured.value.retryable is True
    assert captured.value.status_code == 500
    assert captured.value.response_body == b"assertion failed"
    assert not (tmp_path / "call" / "request.json").exists()
    assert captured.value.request_body_path is None
    assert captured.value.request_body_sha256 == hashlib.sha256(
        sent["body"]
    ).hexdigest()
    manifest = (tmp_path / "call" / "request_manifest.json").read_bytes()
    assert b"data:image/png;base64,AAAA" not in manifest
    assert b"data:image/png;base64,<redacted>" in manifest


def test_remote_disconnect_is_retryable_without_preserving_request(
    tmp_path: Path,
) -> None:
    provider = model_client.OpenAICompatibleProvider(
        model="qwen3.5-9b",
        base_url="http://10.28.1.20:57133/v1",
        timeout_seconds=30,
        api_key_file=None,
    )

    class _DisconnectingOpener:
        def open(self, *_: object, **__: object) -> object:
            raise http.client.RemoteDisconnected("peer closed connection")

    provider.opener = _DisconnectingOpener()
    with pytest.raises(model_client.ProviderRequestError) as captured:
        provider.complete(
            {"model": provider.model, "messages": [], "tools": []},
            call_dir=tmp_path / "disconnect",
            seed=19506,
        )
    assert captured.value.failure_type == "provider_transport_error"
    assert captured.value.retryable is True
    assert captured.value.request_body_sha256 is not None
    assert captured.value.request_body_path is None
    assert not (tmp_path / "disconnect" / "request.json").exists()


def test_reasoning_length_failure_is_classified() -> None:
    response = _reasoning_only()
    response["choices"][0]["finish_reason"] = "length"
    assert (
        model_client._native_protocol_failure_type(response)
        == "reasoning_budget_exhausted"
    )


def test_provider_sends_only_chat_template_thinking_and_frozen_seed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    provider = model_client.OpenAICompatibleProvider(
        model="rynnbrain1.1-122b-a10b",
        base_url="http://127.0.0.1:18133/v1",
        timeout_seconds=30,
        api_key_file=None,
        maximum_output_tokens=1024,
        temperature=0,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.0,
    )
    observed: dict = {}

    def _post(request: dict) -> dict:
        observed.update(request)
        return _done()

    monkeypatch.setattr(provider, "_post", _post)
    provider.complete(
        {"model": provider.model, "messages": [], "tools": []},
        call_dir=tmp_path / "call",
        seed=19507,
    )

    assert "enable_thinking" not in observed
    assert observed["chat_template_kwargs"] == {"enable_thinking": True}
    assert observed["temperature"] == 0
    assert observed["max_tokens"] == 1024
    assert "top_p" not in observed
    assert "top_k" not in observed
    assert "repetition_penalty" not in observed
    assert observed["seed"] == 19507


def test_provider_sends_official_glm53_flash_parameters(
    monkeypatch,
    tmp_path: Path,
) -> None:
    provider = model_client.OpenAICompatibleProvider(
        model="glm-flash",
        base_url="https://example.invalid/v1",
        timeout_seconds=30,
        api_key_file=None,
        maximum_output_tokens=2048,
        temperature=1.0,
        top_p=0.95,
        preserve_thinking=True,
        reasoning_effort="max",
    )
    observed: dict = {}

    def _post(request: dict) -> dict:
        observed.update(request)
        return _done()

    monkeypatch.setattr(provider, "_post", _post)
    provider.complete(
        {"model": provider.model, "messages": [], "tools": []},
        call_dir=tmp_path / "glm53",
        seed=19506,
    )

    assert observed["temperature"] == 1.0
    assert observed["top_p"] == 0.95
    assert observed["max_tokens"] == 2048
    assert observed["reasoning_effort"] == "max"
    assert observed["thinking"] == {
        "type": "enabled",
        "clear_thinking": False,
    }
    assert observed["chat_template_kwargs"] == {
        "reasoning_effort": "max",
        "clear_thinking": False,
    }
    assert "preserve_thinking" not in observed
    assert model_client._provider_generation_record(provider)["thinking"] == {
        "type": "enabled",
        "clear_thinking": False,
    }


def test_provider_records_wall_time_and_usage_metrics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    provider = model_client.OpenAICompatibleProvider(
        model="qwen3.5-9b",
        base_url="http://127.0.0.1:18133/v1",
        timeout_seconds=30,
        api_key_file=None,
    )
    response = _done()
    response["usage"] = {
        "prompt_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 40},
        "completion_tokens": 20,
        "completion_tokens_details": {"reasoning_tokens": 7},
        "total_tokens": 120,
    }
    timestamps = iter((10.0, 11.25))
    monkeypatch.setattr(model_client.time, "monotonic", lambda: next(timestamps))
    monkeypatch.setattr(provider, "_post", lambda _: response)

    provider.complete(
        {"model": provider.model, "messages": [], "tools": []},
        call_dir=tmp_path / "call_metrics",
        seed=19506,
    )

    metrics = json.loads(
        (tmp_path / "call_metrics/provider_metrics.json").read_text()
    )
    assert metrics == {
        "schema_version": "behavior-hl-provider-metrics/v1",
        "response_received": True,
        "provider_wall_seconds": 1.25,
        "server_inference_seconds": None,
        "network_excluded": False,
        "usage": {
            "input_tokens": 100,
            "cached_input_tokens": 40,
            "uncached_input_tokens": 60,
            "output_tokens": 20,
            "reasoning_output_tokens": 7,
        },
    }


def test_provider_metrics_summary_is_bounded_and_fail_closed(tmp_path: Path) -> None:
    calls = tmp_path / "provider_calls"
    for index, wall, input_tokens, cached_tokens, output_tokens in (
        (1, 2.0, 100, 40, 20),
        (2, 4.0, 200, 160, 30),
    ):
        call = calls / f"call_{index:04d}"
        call.mkdir(parents=True)
        (call / "provider_metrics.json").write_text(
            json.dumps(
                {
                    "response_received": True,
                    "provider_wall_seconds": wall,
                    "usage": {
                        "input_tokens": input_tokens,
                        "cached_input_tokens": cached_tokens,
                        "uncached_input_tokens": input_tokens - cached_tokens,
                        "output_tokens": output_tokens,
                        "reasoning_output_tokens": 5,
                    },
                }
            )
        )

    summary = model_client._aggregate_provider_metrics(tmp_path, 3)
    assert summary["recorded_call_count"] == 2
    assert summary["provider_wall_seconds_total"] == 6.0
    assert summary["provider_wall_seconds_p50"] == 2.0
    assert summary["provider_wall_seconds_p95"] == 4.0
    assert summary["usage"]["input_tokens"] == 300
    assert summary["cache_hit_rate"] == 0.666667
    assert summary["network_excluded"] is False
    assert summary["monetary_cost"]["status"] == "UNPRICED"


def test_provider_can_use_native_thinking_with_auto_tool_choice(
    monkeypatch,
    tmp_path: Path,
) -> None:
    provider = model_client.OpenAICompatibleProvider(
        model="qwen3.7-plus",
        base_url="https://example.invalid/v1",
        timeout_seconds=30,
        api_key_file=None,
        thinking_tool_choice_auto=True,
    )
    observed: dict = {}

    def _post(request: dict) -> dict:
        observed.update(request)
        return _done()

    monkeypatch.setattr(provider, "_post", _post)
    provider.complete(
        {
            "model": provider.model,
            "messages": [],
            "tools": [],
            "tool_choice": {
                "type": "function",
                "function": {"name": "robot_action"},
            },
        },
        call_dir=tmp_path / "call_auto",
        seed=19506,
    )

    assert observed["tool_choice"] == "auto"
    assert observed["enable_thinking"] is True
    assert "chat_template_kwargs" not in observed


def test_qwen38_max_visual_thinking_generation_parameters(
    monkeypatch,
    tmp_path: Path,
) -> None:
    provider = model_client.OpenAICompatibleProvider(
        model="qwen3.8-max",
        base_url="https://example.invalid/v1",
        timeout_seconds=30,
        api_key_file=None,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.0,
        preserve_thinking=True,
    )
    observed: dict = {}

    def _post(request: dict) -> dict:
        observed.update(request)
        return _done()

    monkeypatch.setattr(provider, "_post", _post)
    provider.complete(
        {"model": provider.model, "messages": [], "tools": []},
        call_dir=tmp_path / "max_visual",
        seed=19506,
    )

    assert observed["temperature"] == 0.6
    assert observed["top_p"] == 0.95
    assert observed["top_k"] == 20
    assert observed["repetition_penalty"] == 1.0
    assert observed["preserve_thinking"] is True
    assert "reasoning_effort" not in observed
    assert "thinking_budget" not in observed
