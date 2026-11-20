import hashlib
import json
from collections import Counter
from importlib.resources import files

import pytest

from roboweave.tasks import get_task, list_tasks, profile_for, validate_model_task


def test_catalog_identity_instruction_budget_and_counts():
    tasks = list_tasks()
    assert Counter(task["family"] for task in tasks) == {"L0": 100, "L1-A": 30, "L1-M": 30, "L1-V": 30, "L1-P": 30, "L2": 20}
    assert len({(task["family"], task["task_id"]) for task in tasks}) == 240
    for task in tasks:
        assert hashlib.sha256(task["instruction_en"].encode()).hexdigest() == task["instruction_sha256"]
        assert len(task["source"]["sha256"]) == 64
        assert all(type(value) is int and value > 0 for value in task["budget"].values())
    first = get_task("am200v1_a001_b2_control", "L0")
    assert first["instruction_en"] == "The magazine is on the right side of the desktop. Put it on the coffee table."
    assert first["budget"] == {"maximum_action_decisions": 33, "maximum_model_calls": 330,
                                "repeats_per_model": 1, "wall_clock_seconds": 3600}
    assert first["source"]["sha256"] == "5daca755447954c2b1bd2c9a0cafb9cf5e92c78f8cf470666ffee1531dfbd969"


def test_public_catalog_is_a_strict_whitelist_and_contains_no_private_paths():
    allowed = {"task_id", "instruction_id", "runtime_instruction_id", "family", "instruction_en",
               "instruction_sha256", "scene_model", "budget", "action_mode", "profile", "source"}
    tasks = list_tasks()
    assert all(set(task) == allowed | ({"split"} if task["family"] == "L2" else set()) for task in tasks)
    serialized = json.dumps(tasks)
    for private in ("/liyuanshuai/", "prepared_index", "goal_contract", "source_task_identity",
                    "human_success_episode", "private_runtime", "hidden_scope"):
        assert private not in serialized
    assert all(set(task["budget"]) <= {"maximum_action_decisions", "maximum_model_calls",
                                       "repeats_per_model", "wall_clock_seconds"} for task in tasks)


def test_profiles_are_separate_and_l2_proposals_are_not_executable_tasks():
    assert len(list_tasks("L2")) == 20
    assert {task["profile"] for task in list_tasks("L2")} == {"l2-collection37"}
    assert Counter(task["action_mode"] for task in list_tasks("L2")) == {"Symbolic": 9, "Physical": 11}
    assert profile_for("l2-proposals")["proposal_count"] == 36
    assert all(profile_for(task)["execution_supported"] for task in list_tasks())
    assert all(profile_for(task)["live_validation"] == "NOT_RUN" for task in list_tasks())
    assert all(not profile_for(task)["official_benchmark_eligible"] for task in list_tasks())
    assert profile_for("l2-proposals")["execution_supported"] is False
    assert len({profile_for(task)["runtime_revision"] for task in list_tasks("L1-P")}) == 3
    with pytest.raises(ValueError, match="Unknown"):
        get_task("unknown")
    with pytest.raises(ValueError, match="Unknown"):
        list_tasks("invented")


def test_records_are_independent_between_callers():
    task = list_tasks("L0")[0]
    task["budget"]["maximum_action_decisions"] = 1
    assert list_tasks("L0")[0]["budget"]["maximum_action_decisions"] == 33


def _model_binding(task):
    return {"task_id": task["source"]["row_index"],
            "task_name": task["task_id"] if task["family"] == "L1-V" else task["runtime_instruction_id"],
            "instruction": task["instruction_en"], "budget": dict(task["budget"]), **task["budget"],
            "instruction_provenance": {"profile": task["profile"], "instruction_sha256": task["instruction_sha256"]}}


def test_all_frozen_model_bindings_preserve_public_and_native_handles():
    for task in list_tasks():
        assert validate_model_task(task["profile"], _model_binding(task)) == task


@pytest.mark.parametrize("field,value", [("task_id", True), ("task_name", "wrong-task"),
    ("instruction", "wrong instruction"), ("budget", {}), ("maximum_action_decisions", 999),
    ("maximum_model_calls", 330.0), ("wall_clock_seconds", 1), ("repeats_per_model", True),
    ("profile", "l1-a30"), ("instruction_provenance", {})])
def test_model_binding_rejects_post_materialization_changes(field, value):
    task = list_tasks("L0")[0]
    with pytest.raises(ValueError, match="[Ff]rozen"):
        validate_model_task(task["profile"], {**_model_binding(task), field: value})


def test_asset_manifest_is_metadata_only_and_accounts_for_aliases():
    manifest = json.loads(files("roboweave").joinpath("data/assets/l0-english-v2.json").read_text())
    assert manifest["upstream_runtime_reference_count"] == 715
    assert manifest["unique_runtime_file_count"] == 713
    assert len(manifest["files"]) == 769  # Top-level task input and 55 implicit-loader dependencies.
    assert manifest["supplemental_closure"]["file_count"] == 55
    assert manifest["unique_runtime_bytes"] == 97073009
    for item in manifest["files"]:
        assert set(item) - {"handoff_relative_path"} == {"key", "source_root", "source_path", "sha256", "bytes", "roles", "task_ids", "source_reference_count"}
        assert not item["source_path"].startswith("/")
        assert ".." not in item["source_path"].split("/")
