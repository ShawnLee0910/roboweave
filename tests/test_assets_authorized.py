"""Optional CPU audit over an owner's separately imported private L0 inputs.

Run with ESI_BENCHMARK_ASSET_ROOT set to an imported asset directory.
No simulator, resident connection, provider request or original-source write.
"""

import json
import os
from pathlib import Path
import shutil

import pytest

from roboweave import assets
from roboweave._native.bootstrap import activate
from roboweave.tasks import list_tasks


def _read(path):
    text = path.read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(text)


def _undo_relocations(value, changes):
    for change in reversed(changes):
        if change["kind"] == "single_task_selection":
            continue
        parts = [part.replace("~1", "/").replace("~0", "~") for part in change["pointer"].split("/")[1:]]
        current = value
        for part in parts[:-1]:
            current = current[int(part)] if isinstance(current, list) else current[part]
        current[int(parts[-1]) if isinstance(current, list) else parts[-1]] = change["old"]
    return value


def _expected_selection(before, roles, task_id):
    if "release_tasks" in roles:
        return [item for item in before if item["runtime_instruction_id"] == task_id]
    if "task_specs" in roles:
        return [item for item in before if item["task_spec_id"] == task_id]
    if "registration_catalog" in roles:
        before["tasks"] = [item for item in before["tasks"] if item.get("task_spec_id", item.get("candidate_id")) == task_id]
        before["task_count"] = 1
    if "package" in roles:
        before.update(task_spec_ids=[task_id], task_count=1)
    if "handoff" in roles:
        before["samples"] = [item for item in before["samples"] if item["task_spec_id"] == task_id]
        before["packages"] = [item for item in before["packages"] if item["name"] == before["samples"][0]["package_name"]]
        before["count"] = 1
    if "prepared_index" in roles:
        before["instructions"] = {task_id: before["instructions"][task_id]}
    return before


@pytest.mark.skipif(not os.environ.get("ESI_BENCHMARK_ASSET_ROOT"), reason="Requires separately authorized private assets; never bundled")
def test_all_100_private_inputs_compile_without_semantic_changes(tmp_path):
    root = Path(os.environ["ESI_BENCHMARK_ASSET_ROOT"]).absolute()
    config = {"profile": "l0-english-v2", "assets": {"root": str(root)}}
    assert assets.check_assets(config)["ok"]
    activate()
    from instruction_handoff import compile_handoff, native_instruction_arguments
    from lab.prepared_state_inputs import load_prepared
    from behavior_hl.embodied.evaluator_bridge.prepared_model_state import validate_bundle
    from behavior_hl.embodied.evaluator_bridge.registered_l0_task_runtime import load_registered_l0_tasks

    for task in list_tasks("L0"):
        destination = tmp_path / task["task_id"]
        try:
            result = assets.materialize_task_inputs(config, task, destination)
            receipt = json.loads(Path(result["materialization_receipt"]).read_text())
            _, rows = assets._selected(config, task)
            by_key = {row["key"]: row for row in rows}
            for record in receipt["files"]:
                row = by_key[record["key"]]
                original, derived = assets._path(root, row), destination / record["derived_path"]
                if original.suffix in (".json", ".jsonl"):
                    expected = _expected_selection(_read(original), row["roles"], task["runtime_instruction_id"])
                    assert _undo_relocations(_read(derived), record["changes"]) == expected, (task["task_id"], row["key"])
                else:
                    assert original.read_bytes() == derived.read_bytes()
            matrix = compile_handoff(Path(result["handoff_json"]), {"model": 1})
            assert set(matrix["instructions"]) == {task["runtime_instruction_id"]}
            native = native_instruction_arguments(Path(result["handoff_json"]), result["instruction_id"])
            bundle = load_prepared(Path(result["prepared_index_json"]), matrix["handoff_sha256"], result["instruction_id"])
            validate_bundle(bundle, instruction_id=result["instruction_id"], seed=result["model_task"]["task_seed"],
                            observation_profile="r1pro_three_rgb_head_eval")
            assert result["model_task"]["task_name"] == result["instruction_id"]
            assert isinstance(result["model_task"]["task_instance_id"], str)
            assert result["model_task"]["budget"] == task["budget"]
            if "--runtime-registration" in native:
                overlay = Path(native[native.index("--reset-pose-overlay") + 1]) if "--reset-pose-overlay" in native else None
                registered = load_registered_l0_tasks(Path(native[native.index("--runtime-registration") + 1]), reset_pose_overlay=overlay)
                assert set(registered) == {result["instruction_id"]}
            else:
                specs = _read(Path(native[native.index("--l0-taskspecs") + 1]))
                assert len(specs) == 1 and specs[0]["task_spec_id"] == result["instruction_id"]
        finally:
            if destination.exists():
                shutil.rmtree(destination)
    assert assets.check_assets(config)["ok"]
