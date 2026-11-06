from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


BENCH_TRY_ROOT = (Path(__file__).resolve().parents[2] / "src/roboweave/_vendor/bench")
LAB_ROOT = Path(__file__).resolve().parent
if str(BENCH_TRY_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_TRY_ROOT))
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from Agent import SystemPromptRevision, parse_native_tool_message  # noqa: E402
from native_tool_preflight import (  # noqa: E402
    NativeToolPreflightError,
    PROBE_STEPS,
    run_native_tool_preflight,
)


def _prompt_revision() -> SystemPromptRevision:
    text = "Use the native robot_action function for every decision."
    return SystemPromptRevision(
        text=text,
        source_path=Path(__file__),
    )


def _completion(
    action: str,
    target: int | None,
    index: int,
    target_description: str | None = None,
) -> dict:
    arguments = {"action": action}
    if target is not None:
        arguments["target"] = target
        arguments["target_description"] = target_description
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
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
            }
        ]
    }


class NativeToolPreflightTests(unittest.TestCase):
    def test_each_structured_reasoning_field_is_isolated(self) -> None:
        for field in ("reasoning_content", "reasoning"):
            with self.subTest(field=field):
                message = _completion("wait", None, 1)["choices"][0]["message"]
                message[field] = "Inspect the current observation."

                decision = parse_native_tool_message(message)

                self.assertEqual(
                    decision.reasoning_content,
                    "Inspect the current observation.",
                )
                self.assertEqual(decision.reasoning_source, field)

    def test_content_think_reasoning_is_isolated(self) -> None:
        message = _completion("wait", None, 1)["choices"][0]["message"]
        message["content"] = "\n<think>\nInspect the scene.\n</think>\n"

        decision = parse_native_tool_message(message)

        self.assertEqual(decision.reasoning_content, "Inspect the scene.")
        self.assertEqual(decision.reasoning_source, "content_think")
        self.assertEqual(decision.action, "wait")

    def test_duplicate_structured_and_content_reasoning_is_deduplicated(self) -> None:
        message = _completion("wait", None, 1)["choices"][0]["message"]
        message["reasoning"] = "Inspect the scene."
        message["content"] = "<think>\nInspect the scene.\n</think>"

        decision = parse_native_tool_message(message)

        self.assertEqual(decision.reasoning_content, "Inspect the scene.")
        self.assertEqual(decision.reasoning_source, "reasoning")

    def test_identical_reasoning_aliases_are_normalized(self) -> None:
        message = _completion("wait", None, 1)["choices"][0]["message"]
        message["reasoning"] = "same native reasoning"
        message["reasoning_content"] = "same native reasoning"

        decision = parse_native_tool_message(message)

        self.assertEqual(decision.reasoning_content, "same native reasoning")
        self.assertEqual(decision.reasoning_source, "reasoning_content")

    def test_different_reasoning_aliases_fail_closed(self) -> None:
        message = _completion("wait", None, 1)["choices"][0]["message"]
        message["reasoning"] = "first"
        message["reasoning_content"] = "second"

        with self.assertRaisesRegex(ValueError, "appeared differently"):
            parse_native_tool_message(message)

    def test_structured_reasoning_cannot_escape_think_boundary(self) -> None:
        for field in ("reasoning_content", "reasoning"):
            for marker in ("<think>", "</think>"):
                with self.subTest(field=field, marker=marker):
                    message = _completion("wait", None, 1)["choices"][0][
                        "message"
                    ]
                    message[field] = f"Inspect the scene. {marker}"

                    with self.assertRaisesRegex(ValueError, "reserved"):
                        parse_native_tool_message(message)

    def test_conflicting_or_malformed_content_think_fails_closed(self) -> None:
        cases = (
            "<think>different</think>",
            "<think>same native reasoning",
            "prose <think>same native reasoning</think>",
            "<think><think>nested</think></think>",
            "<think>same native reasoning</think>ordinary prose",
        )
        for content in cases:
            with self.subTest(content=content):
                message = _completion("wait", None, 1)["choices"][0]["message"]
                message["reasoning_content"] = "same native reasoning"
                message["content"] = content

                with self.assertRaises(ValueError):
                    parse_native_tool_message(message)

    def test_non_marker_target_fields_are_semantically_ignored(self) -> None:
        message = _completion("turn_left", 7, 1, "irrelevant wall")[
            "choices"
        ][0]["message"]

        decision = parse_native_tool_message(message)

        self.assertEqual(decision.action, "turn_left")
        self.assertIsNone(decision.target)
        self.assertIsNone(decision.target_description)
        self.assertEqual(decision.function_arguments(), {"action": "turn_left"})

    def test_complete_round_trip(self) -> None:
        requests: list[dict] = []

        def post(request: dict) -> dict:
            requests.append(request)
            step = PROBE_STEPS[len(requests) - 1]
            return _completion(
                step.expected_action,
                step.expected_target,
                len(requests),
                step.expected_target_description,
            )

        report = run_native_tool_preflight(
            post=post,
            model="test-model",
            prompt_revision=_prompt_revision(),
        )

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["step_count"], 5)
        self.assertEqual(len(requests), 5)
        expected_fields = {
            "action",
            "move_distance_m",
            "repeat",
            "target",
            "target_description",
            "turn_angle_deg",
        }
        self.assertEqual(
            set(requests[0]["tools"][0]["function"]["parameters"]["properties"]),
            expected_fields,
        )
        self.assertEqual(
            set(requests[-1]["tools"][0]["function"]["parameters"]["properties"]),
            expected_fields,
        )
        self.assertTrue(
            all(request["tools"] == requests[0]["tools"] for request in requests)
        )
        self.assertEqual(
            json.loads(requests[1]["messages"][-2]["content"]),
            {"status": "completed"},
        )
        self.assertEqual(
            json.loads(requests[2]["messages"][-2]["content"])["status"],
            "blocked",
        )
        self.assertEqual(
            json.loads(
                requests[-1]["messages"][-3]["tool_calls"][0]["function"][
                    "arguments"
                ]
            ),
            {"action": "pick"},
        )

    def test_wrong_followup_action_fails_closed(self) -> None:
        call_count = 0

        def post(request: dict) -> dict:
            nonlocal call_count
            del request
            call_count += 1
            step = PROBE_STEPS[0 if call_count == 1 else 1]
            action = "look_up" if call_count >= 2 else step.expected_action
            return _completion(
                action,
                step.expected_target,
                call_count,
                step.expected_target_description,
            )

        with self.assertRaisesRegex(
            NativeToolPreflightError,
            "completed_followup expected",
        ):
            run_native_tool_preflight(
                post=post,
                model="test-model",
                prompt_revision=_prompt_revision(),
            )

        self.assertEqual(call_count, 4)

    def test_invalid_first_attempt_recovers_with_same_step(self) -> None:
        calls = 0

        def post(request: dict) -> dict:
            nonlocal calls
            calls += 1
            completed_steps = (calls - 2) if calls >= 2 else 0
            step = PROBE_STEPS[completed_steps]
            if calls == 1:
                return _completion("look_up", None, calls)
            return _completion(
                step.expected_action,
                step.expected_target,
                calls,
                step.expected_target_description,
            )

        report = run_native_tool_preflight(
            post=post,
            model="test-model",
            prompt_revision=_prompt_revision(),
        )

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["model_call_count"], 6)
        self.assertEqual(report["steps"][0]["attempt_count"], 2)
        self.assertTrue(report["steps"][0]["recovered_after_retry"])

    def test_thinking_auto_preflight_uses_exact_provider_contract(self) -> None:
        requests: list[dict] = []

        def post(request: dict) -> dict:
            requests.append(request)
            step = PROBE_STEPS[len(requests) - 1]
            response = _completion(
                step.expected_action,
                step.expected_target,
                len(requests),
                step.expected_target_description,
            )
            response["choices"][0]["message"]["reasoning_content"] = "native"
            return response

        report = run_native_tool_preflight(
            post=post,
            model="qwen3.7-plus",
            prompt_revision=_prompt_revision(),
            thinking_tool_choice_auto=True,
        )

        self.assertTrue(report["thinking_tool_choice_auto"])
        self.assertTrue(report["native_reasoning_present"])
        self.assertTrue(all(request["tool_choice"] == "auto" for request in requests))
        self.assertTrue(all(request["enable_thinking"] is True for request in requests))
        self.assertTrue(all("chat_template_kwargs" not in request for request in requests))


if __name__ == "__main__":
    unittest.main()
