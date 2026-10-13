from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys

from PIL import Image


BENCH_TRY_ROOT = (Path(__file__).resolve().parents[2] / "src/esi_benchmark/_vendor/bench")
if str(BENCH_TRY_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_TRY_ROOT))

from Agent import (  # noqa: E402
    EpisodeConversation,
    PrivateEpisodeContract,
    PublicImage,
    PublicObservation,
    PublicToolResult,
    load_prompt_revision,
)
from agent_observation import materialize_observation  # noqa: E402


def _png(path: Path, color: tuple[int, int, int], edge: int) -> PublicImage:
    buffer = io.BytesIO()
    Image.new("RGB", (edge, edge), color).save(buffer, format="PNG")
    raw = buffer.getvalue()
    path.write_bytes(raw)
    return PublicImage(
        camera=path.stem,
        path=path,
    )


def _observation(root: Path, name: str) -> PublicObservation:
    directory = root / name
    directory.mkdir()
    return PublicObservation(
        observation_id=name,
        images=(
            _png(directory / "head.png", (0, 0, 255), 3),
            _png(directory / "left_wrist.png", (255, 0, 0), 2),
            _png(directory / "right_wrist.png", (0, 255, 0), 2),
        ),
    )


def _production_sized_observation(root: Path, name: str) -> PublicObservation:
    directory = root / name
    directory.mkdir()
    return PublicObservation(
        observation_id=name,
        images=(
            _png(directory / "head.png", (0, 0, 255), 960),
            _png(directory / "left_wrist.png", (255, 0, 0), 480),
            _png(directory / "right_wrist.png", (0, 255, 0), 480),
        ),
    )


def _four_view_observation(root: Path, name: str) -> PublicObservation:
    directory = root / name
    directory.mkdir()
    return PublicObservation(
        observation_id=name,
        images=(
            _png(directory / "head.png", (0, 0, 255), 6),
            _png(directory / "left_wrist.png", (255, 0, 0), 3),
            _png(directory / "right_wrist.png", (0, 255, 0), 3),
            _png(directory / "overhead.png", (255, 255, 0), 5),
        ),
    )


def _conversation(
    tmp_path: Path,
    *,
    physical_history_max: int = 5,
) -> EpisodeConversation:
    return EpisodeConversation(
        PrivateEpisodeContract(
            episode_id="test",
            task_id="task",
            split="train",
            instance_id="0",
            seed=19506,
            model="model",
            project_root=tmp_path,
            instruction="Test task.",
            physical_history_max=physical_history_max,
        ),
        _observation(tmp_path, "observation_0"),
    )


def _completion(index: int, action: str = "wait") -> dict:
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
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


def _image_urls(request: dict) -> list[str]:
    return [
        item["image_url"]["url"]
        for message in request["messages"]
        if isinstance(message.get("content"), list)
        for item in message["content"]
        if item.get("type") == "image_url"
    ]


def test_three_view_is_one_head_dominant_composite(tmp_path: Path) -> None:
    observation = _observation(tmp_path, "observation")

    with Image.open(io.BytesIO(observation.read_model_composite_png())) as image:
        image.load()
        assert image.size == (6, 4)
        assert image.getpixel((0, 0)) == (255, 0, 0)
        assert image.getpixel((0, 3)) == (0, 255, 0)
        assert image.getpixel((5, 2)) == (0, 0, 255)

    assert hashlib.sha256(observation.read_model_composite_png()).hexdigest() == (
        "369c541d7a3a8c71cc68e4b2d2722e8b34af5eb53c6c94d809d425437b30ac2f"
    )


def test_four_view_is_one_1280_by_960_equivalent_composite(
    tmp_path: Path,
) -> None:
    observation = _four_view_observation(tmp_path, "four_view")

    with Image.open(io.BytesIO(observation.read_model_composite_png())) as image:
        image.load()
        assert image.size == (8, 6)
        assert image.getpixel((0, 0)) == (255, 0, 0)
        assert image.getpixel((0, 2)) == (0, 255, 0)
        assert image.getpixel((0, 4)) == (255, 255, 0)
        assert image.getpixel((7, 3)) == (0, 0, 255)


def test_four_view_observation_emits_one_image_url(tmp_path: Path) -> None:
    observation = _four_view_observation(tmp_path, "four_request")
    conversation = EpisodeConversation(
        PrivateEpisodeContract(
            episode_id="four",
            task_id="task",
            split="train",
            instance_id="0",
            seed=19506,
            model="model",
            project_root=tmp_path,
            instruction="Test task.",
        ),
        observation,
    )

    urls = _image_urls(conversation.build_native_request())
    assert len(urls) == 1
    with Image.open(
        io.BytesIO(base64.b64decode(urls[0].split(",", 1)[1]))
    ) as image:
        assert image.size == (8, 6)


def test_scene_four_view_payload_materializes_into_one_model_image(
    tmp_path: Path,
) -> None:
    scene_run = tmp_path / "scene"
    scene_run.mkdir()
    source = _four_view_observation(scene_run, "scene_observation")
    payload = {
        "observation_id": "scene_observation",
        "kind": "physical",
        "images": [
            {
                "camera": image.camera,
                "path": str(image.path),
                "sha256": hashlib.sha256(image.path.read_bytes()).hexdigest(),
            }
            for image in source.images
        ],
        "marker_targets": [],
    }
    observation = materialize_observation(
        payload,
        output_dir=tmp_path / "run",
        scene_run_dir=scene_run,
        link_or_copy=shutil.copy2,
        hash_file=lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
        error_type=ValueError,
    )
    conversation = EpisodeConversation(
        PrivateEpisodeContract(
            episode_id="materialized",
            task_id="task",
            split="train",
            instance_id="0",
            seed=19506,
            model="model",
            project_root=tmp_path,
            instruction="Test task.",
        ),
        observation,
    )

    assert len(_image_urls(conversation.build_native_request())) == 1


def test_one_observation_emits_exactly_one_image_url(tmp_path: Path) -> None:
    conversation = _conversation(tmp_path)
    request = conversation.build_native_request()

    urls = _image_urls(request)
    assert len(urls) == 1
    raw = base64.b64decode(urls[0].split(",", 1)[1])
    with Image.open(io.BytesIO(raw)) as image:
        assert image.size == (6, 4)


def test_history_is_one_composite_per_step_and_bounded(tmp_path: Path) -> None:
    conversation = _conversation(tmp_path)
    for index in range(1, 8):
        conversation.append_assistant_completion(_completion(index))
        conversation.append_tool_result(PublicToolResult("completed"))
        conversation.append_observation(
            _observation(tmp_path, f"observation_{index}"),
            kind="physical",
        )
        expected = min(index + 1, 6)
        assert len(_image_urls(conversation.build_native_request())) == expected


def test_twenty_history_is_one_composite_per_step_and_bounded(
    tmp_path: Path,
) -> None:
    conversation = _conversation(tmp_path, physical_history_max=20)
    for index in range(1, 24):
        conversation.append_assistant_completion(_completion(index))
        conversation.append_tool_result(PublicToolResult("completed"))
        conversation.append_observation(
            _observation(tmp_path, f"observation_{index}"),
            kind="physical",
        )
        expected = min(index + 1, 21)
        assert len(_image_urls(conversation.build_native_request())) == expected
    assert conversation.artifact_transcript()["visual_history"] == {
        "completed_physical_history_max": 20,
        "model_observation_max": 21,
    }


def test_history_window_does_not_mutate_fixed_behavior_prompt() -> None:
    five = load_prompt_revision(physical_history_max=5)
    twenty = load_prompt_revision(physical_history_max=20)
    assert twenty.text == five.text
    assert "historical composites" not in five.text
    assert "history implementation" not in five.text


def test_episode_budget_is_rendered_into_prompt() -> None:
    prompt = load_prompt_revision(maximum_action_decisions=3)

    assert "at most 3 action decisions for the entire task" in prompt.text
    assert "at most 150 action decisions for the entire task" not in prompt.text


def test_all_bounded_history_and_current_views_stay_full_resolution(tmp_path: Path) -> None:
    conversation = EpisodeConversation(
        PrivateEpisodeContract(
            episode_id="resolution_test",
            task_id="task",
            split="train",
            instance_id="0",
            seed=19506,
            model="model",
            project_root=tmp_path,
            instruction="Test task.",
        ),
        _production_sized_observation(tmp_path, "large_0"),
    )
    for index in range(1, 6):
        conversation.append_assistant_completion(_completion(index))
        conversation.append_tool_result(PublicToolResult("completed"))
        conversation.append_observation(
            _production_sized_observation(tmp_path, f"large_{index}"),
            kind="physical",
        )

    sizes = []
    for url in _image_urls(conversation.build_native_request()):
        raw = base64.b64decode(url.split(",", 1)[1])
        with Image.open(io.BytesIO(raw)) as image:
            sizes.append(image.size)
    assert sizes == [
        (1440, 960),
        (1440, 960),
        (1440, 960),
        (1440, 960),
        (1440, 960),
        (1440, 960),
    ]


def test_marker_keeps_unannotated_current_as_separate_composite(
    tmp_path: Path,
) -> None:
    conversation = _conversation(tmp_path)
    conversation.append_assistant_completion(_completion(1, action="pick"))
    conversation.append_tool_result(
        PublicToolResult("selection_required", stage=1)
    )
    conversation.append_observation(
        _observation(tmp_path, "marker_1"),
        kind="marker",
        marker_targets=(1, 2),
    )

    assert conversation.active_observation_ids == (
        "observation_0",
        "marker_1",
    )
    assert len(_image_urls(conversation.build_native_request())) == 2
