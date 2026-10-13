import hashlib
import json
from pathlib import Path

import pytest

from esi_benchmark import assets


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    manifest = {"profile": "l0-english-v2", "source_roots": ["bundle"], "files": []}
    task = {"task_id": "one", "runtime_instruction_id": "one", "profile": "l0-english-v2",
            "instruction_en": "Place the cup on the table.", "instruction_sha256": "public-instruction-hash",
            "budget": {"maximum_action_decisions": 3, "maximum_model_calls": 5, "wall_clock_seconds": 60},
            "source": {"row_index": 0}}

    def add(name, content, role):
        path = source / name
        text = "".join(json.dumps(item) + "\n" for item in content) if name.endswith(".jsonl") else json.dumps(content)
        path.write_text(text)
        manifest["files"].append({"key": name, "source_root": "bundle", "source_path": name,
                                  "sha256": sha(path), "bytes": path.stat().st_size,
                                  "roles": [role], "task_ids": ["one"]})
        return {"path": "/frozen/bundle/" + name, "sha256": sha(path)}

    goal = {"task_spec_id": "one", "goal": {"predicate": "ontop", "object": "cup"},
            "process": {"must_hold": ["closed"]}, "initial": {"pose": [0.0, 1.0, 0.2]}}
    specs = add("specs.jsonl", [goal, {**goal, "task_spec_id": "two"}], "task_specs")
    registry = add("catalog.json", {"task_count": 2, "tasks": [goal, {**goal, "task_spec_id": "two"}]}, "registration_catalog")
    package = add("package.json", {"task_spec_ids": ["one", "two"], "task_count": 2,
                                   "task_specs": specs, "runtime": {"registration_catalog": registry}}, "package")
    prepared = add("prepared.json", {"pose": [1.0, 2.0, 0.0], "goal": False}, "prepared_state")
    handoff = add("handoff.json", {"count": 2, "model_conditions": {"action_mode": "Symbolic"},
                                  "samples": [{"task_spec_id": "one", "package_name": "p", "seed": 17},
                                              {"task_spec_id": "two", "package_name": "p", "seed": 17}],
                                  "packages": [{"name": "p", **package}]}, "handoff")
    index = add("index.json", {"handoff_sha256": handoff["sha256"], "instructions": {"one": prepared, "two": prepared}}, "prepared_index")
    source_ref = add("source.json", {"package": package, "task_specs_ref": specs, "seed": 17,
                                    "bench_instruction_binding": {"source_task_identity": {"task_name": "one", "split": "train", "instance_id": 0}}}, "source")
    raw = {"runtime_instruction_id": "one", "instruction_en": task["instruction_en"], "budget": task["budget"],
           "source": source_ref, "handoff": handoff, "prepared_index": index}
    add("tasks.json", [raw, {**raw, "runtime_instruction_id": "two"}], "release_tasks")
    monkeypatch.setattr(assets, "asset_manifest", lambda profile="l0-english-v2": manifest)
    config = {"profile": "l0-english-v2", "assets": {"root": str(tmp_path / "local"), "source_roots": {"bundle": str(source)}}}
    return config, task, manifest, source, tmp_path, goal


def test_import_is_checked_before_writing_and_existing_bytes_are_immutable(inputs):
    config, task, manifest, source, tmp_path, _ = inputs
    (source / "prepared.json").write_text("changed")
    with pytest.raises(ValueError, match="mismatch"):
        assets.prepare_assets(config)
    assert not Path(config["assets"]["root"]).exists()


def test_complete_import_resolve_and_materialize_preserve_semantics(inputs):
    config, task, manifest, source, tmp_path, goal = inputs
    before = {item.name: sha(item) for item in source.iterdir()}
    assert assets.prepare_assets(config)["ok"]
    assert assets.prepare_assets(config)["copied"] == 0
    assert assets.resolve_task_inputs(config, task)["model_task"]["maximum_action_decisions"] == 3
    assert assets.resolve_task_inputs(config, task)["model_task"]["task_instance_id"] == "0"
    result = assets.materialize_task_inputs(config, task, tmp_path / "run-inputs")
    selected = [json.loads(line) for line in Path(result["task_specs_jsonl"]).read_text().splitlines()]
    assert selected == [goal]
    package = json.loads(Path(result["benchmark_package_json"]).read_text())
    assert package["task_spec_ids"] == ["one"] and package["task_count"] == 1
    assert package["task_specs"]["sha256"] == sha(Path(result["task_specs_jsonl"]))
    catalog = json.loads(Path(package["runtime"]["registration_catalog"]["path"]).read_text())
    assert catalog["tasks"] == [goal]
    handoff = json.loads(Path(result["handoff_json"]).read_text())
    assert handoff["samples"] == [{"task_spec_id": "one", "package_name": "p", "seed": 17}]
    index = json.loads(Path(result["prepared_index_json"]).read_text())
    assert set(index["instructions"]) == {"one"}
    assert index["handoff_sha256"] == sha(Path(result["handoff_json"]))
    assert handoff["packages"][0]["sha256"] == sha(Path(result["benchmark_package_json"]))
    assert {item.name: sha(item) for item in source.iterdir()} == before
    assert assets.check_assets(config)["ok"]
    receipt = json.loads(Path(result["materialization_receipt"]).read_text())
    assert len(receipt["files"]) == len(manifest["files"])
    assert all(item["original_sha256"] == before[item["key"]] for item in receipt["files"])
    assert any(change["kind"] == "single_task_selection" for item in receipt["files"] for change in item["changes"])
    with pytest.raises(ValueError, match="new output"):
        assets.materialize_task_inputs(config, task, tmp_path / "run-inputs")


def test_missing_hash_mismatch_and_no_overwrite(inputs):
    config, task, manifest, source, tmp_path, _ = inputs
    assert len(assets.check_assets(config)["missing"]) == len(manifest["files"])
    assets.prepare_assets(config)
    destination = Path(config["assets"]["root"]) / "files/bundle/prepared.json"
    destination.write_text("wrong")
    assert assets.check_assets(config)["invalid"] == ["prepared.json"]
    with pytest.raises(ValueError, match="overwrite"):
        assets.prepare_assets(config)
    assert destination.read_text() == "wrong"


def test_import_rejects_traversal_symlink_and_source_destination(inputs):
    config, task, manifest, source, tmp_path, _ = inputs
    manifest["files"][0]["source_path"] = "../outside.json"
    with pytest.raises(ValueError, match="Unsafe"):
        assets.prepare_assets(config)
    manifest["files"][0]["source_path"] = "specs.jsonl"
    path = source / "specs.jsonl"
    target = tmp_path / "outside.json"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="escapes|Symlink"):
        assets.prepare_assets(config)
    path.unlink()
    target.rename(path)
    config["assets"]["root"] = str(source / "new-assets")
    with pytest.raises(ValueError, match="overlaps"):
        assets.prepare_assets(config)
    assert not (source / "new-assets").exists()


def test_bundle_and_explicit_mapping_import(inputs):
    config, task, manifest, source, tmp_path, _ = inputs
    mapping = {row["key"]: str(source / row["source_path"]) for row in manifest["files"]}
    assert assets.prepare_assets(config, mapping=mapping)["ok"]
    second = {"profile": "l0-english-v2", "assets": {"root": str(tmp_path / "second")}}
    assert assets.prepare_assets(second, bundle=config["assets"]["root"])["ok"]
    with pytest.raises(ValueError, match="Unknown"):
        assets.prepare_assets(second, mapping={"unknown": "ignored"})


def test_public_instruction_or_budget_change_is_rejected(inputs):
    config, task, manifest, source, tmp_path, _ = inputs
    assets.prepare_assets(config)
    task["budget"] = {**task["budget"], "maximum_action_decisions": 99}
    with pytest.raises(ValueError, match="instruction or budget"):
        assets.resolve_task_inputs(config, task)


def test_source_without_redundant_taskspec_reference_uses_frozen_package(inputs):
    config, task, manifest, source, tmp_path, _ = inputs
    path = source / "source.json"
    payload = json.loads(path.read_text())
    del payload["task_specs_ref"]
    path.write_text(json.dumps(payload))
    release = source / "tasks.json"
    payload = json.loads(release.read_text())
    for row in payload:
        row["source"]["sha256"] = sha(path)
    release.write_text(json.dumps(payload))
    for row in manifest["files"]:
        if row["key"] in ("source.json", "tasks.json"):
            row["sha256"] = sha(source / row["key"])
            row["bytes"] = (source / row["key"]).stat().st_size
    assets.prepare_assets(config)
    resolved = assets.resolve_task_inputs(config, task)
    assert Path(resolved["task_specs_jsonl"]).name == "specs.jsonl"


def test_implicit_package_lookup_and_nested_source_root_are_relocated(inputs):
    config, task, manifest, source, tmp_path, _ = inputs
    manifest["source_roots"].extend(["worktrees", "action20-capability-benchmark"])
    path = source / "frozen.json"
    original_package = json.loads((source / "package.json").read_text())
    frozen = {**original_package,
              "source_package": {"path": "/frozen/bundle/package.json", "sha256": sha(source / "package.json")},
              "expansion_root": "/frozen/worktrees/action20-capability-benchmark/l0-expansion-v1"}
    path.write_text(json.dumps(frozen))
    manifest["files"].append({"key": "frozen.json", "source_root": "bundle", "source_path": "frozen.json",
                              "sha256": sha(path), "bytes": path.stat().st_size, "roles": ["package"],
                              "task_ids": ["one"], "handoff_relative_path": "implicit/frozen/package.json"})
    assets.prepare_assets(config)
    result = assets.materialize_task_inputs(config, task, tmp_path / "run")
    derived = Path(result["handoff_json"]).parent / "implicit/frozen/package.json"
    assert derived.is_file()
    payload = json.loads(derived.read_text())
    assert payload["source_package"]["sha256"] == sha(Path(result["benchmark_package_json"]))
    assert payload["expansion_root"] == str(tmp_path / "run/files/worktrees/action20-capability-benchmark/l0-expansion-v1")
    assert payload["task_spec_ids"] == ["one"]


def test_legacy_candidate_id_registration_preserves_the_selected_record():
    record = {"candidate_id": "one", "profile": {"goal": [["ontop", "cup", "table"]]}}
    value = {"tasks": [record, {"candidate_id": "two"}], "task_count": 2}
    projected = assets._project(value, {"roles": ["registration_catalog"]}, "one")
    assert projected == {"tasks": [record], "task_count": 1}
    assert value["task_count"] == 2
