from pathlib import Path
import hashlib
import json
import sys

import pytest


ROOT = (Path(__file__).resolve().parents[2] / "src/roboweave/_vendor/bench")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Agent import (  # noqa: E402
    ACTION20_TOOL_SCHEMA,
    load_action20_core_instructions,
    with_automatic_interaction_approach_tool_schema,
    with_overhead_view_tool_schema,
)


def test_prompt_validation_is_structural_and_keeps_privacy_boundary(
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text(
        "Use robot_action for one safe embodied action per turn.",
        encoding="utf-8",
    )
    assert load_action20_core_instructions(prompt).startswith("Use robot_action")

    prompt.write_text("Use robot_action with api_key=private", encoding="utf-8")
    with pytest.raises(ValueError, match="private data"):
        load_action20_core_instructions(prompt)


def test_automatic_interaction_approach_replaces_direct_reach_rule() -> None:
    configured = with_automatic_interaction_approach_tool_schema(
        ACTION20_TOOL_SCHEMA
    )
    original = ACTION20_TOOL_SCHEMA["function"]["description"]
    description = configured["function"]["description"]

    assert "within about 1.7 m horizontally" in original
    assert "within about 1.7 m horizontally" not in description
    assert "out_of_reach" in original
    assert "out_of_reach" in description
    assert "target_outside_local_workspace" not in original
    assert "target_outside_local_workspace" not in description
    assert "Automatic interaction approach is enabled" in description


def test_overhead_view_schema_describes_the_active_composite() -> None:
    always = with_overhead_view_tool_schema(ACTION20_TOOL_SCHEMA, "always")
    on_blocked = with_overhead_view_tool_schema(
        ACTION20_TOOL_SCHEMA,
        "on_blocked",
    )

    assert "one four-view RGB composite" in always["function"]["description"]
    assert "After a blocked action, an overhead top-down view may appear" in (
        on_blocked["function"]["description"]
    )


@pytest.mark.parametrize("budget_sentences", (0, 2))
def test_strict_loader_requires_one_budget_sentence(tmp_path, budget_sentences):
    from agent_prompt import load_common_system_prompt_revision

    source = tmp_path / "core.txt"
    source.write_text("Use robot_action.\n" +
        "You have a budget of at most 150 action decisions for the entire task.\n" * budget_sentences)
    with pytest.raises(ValueError, match="budget contract"):
        load_common_system_prompt_revision(source)


def test_provenance_distinguishes_source_rendered_text_and_effective_tools(tmp_path):
    from agent_prompt import load_common_system_prompt_revision, prompt_provenance_record

    source = tmp_path / "core.txt"
    raw = b"Use robot_action. You have a budget of at most 150 action decisions for the entire task."
    source.write_bytes(raw)
    revision = load_common_system_prompt_revision(source, maximum_action_decisions=37)
    source.write_text("changed after load")
    tools = with_overhead_view_tool_schema(ACTION20_TOOL_SCHEMA, "always")
    tools = with_automatic_interaction_approach_tool_schema(tools)
    record = prompt_provenance_record(
        revision, tools, instruction="Move the cup.", maximum_action_decisions=37,
        harness_adapter="common_tool_calling_v1", overhead_view_mode="always",
        automatic_interaction_approach_enabled=True, physical_history_max=5,
    )
    assert record["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert record["rendered_prompt_sha256"] == hashlib.sha256(revision.text.encode()).hexdigest()
    assert record["source_sha256"] != record["rendered_prompt_sha256"]
    assert record["tool_schema_sha256"] == hashlib.sha256(json.dumps(
        tools, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode()).hexdigest()
    assert record["maximum_action_decisions"] == 37
    assert record["overhead_view_mode"] == "always"
    assert record["automatic_interaction_approach_enabled"] is True
    assert "at most 37 action decisions" in revision.text
    assert revision.text.count("Common Tool-Calling Harness:") == 1
