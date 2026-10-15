import hashlib
import json
import os
from pathlib import Path
import stat

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


def test_compact_materialization_preserves_native_two_mib_input_limit(inputs):
    config, task, manifest, source, tmp_path, _ = inputs
    value = {"evidence": [[0] * 32] * 7000}
    assert len(json.dumps(value, indent=2).encode()) > 2 * 1024 * 1024
    path = source / "donor.json"
    path.write_text(json.dumps(value, separators=(",", ":")))
    assert path.stat().st_size < 2 * 1024 * 1024
    manifest["files"].append({"key": "donor", "source_root": "bundle", "source_path": path.name,
                              "sha256": sha(path), "bytes": path.stat().st_size,
                              "roles": ["prepared_oracle_dependency"], "task_ids": ["one"], "selection": None})
    assets.prepare_assets(config)
    result = assets.materialize_task_inputs(config, task, tmp_path / "run-inputs")
    derived = Path(result["mapping"]["bundle/donor.json"])
    assert derived.stat().st_size < 2 * 1024 * 1024
    assert json.loads(derived.read_text()) == value


def test_explicit_relative_paths_and_file_and_canonical_hash_bindings(inputs):
    config, task, manifest, source, tmp_path, _ = inputs
    config["runtime"] = {"data_root": str(tmp_path / "licensed-data")}
    value = {"relative": "./prepared.json", "file_hash": "before", "canonical_hash": "before",
             "scene_asset": "/old/BEHAVIOR-1K-2026-data/scenes/object.usd", "semantic_goal_hash": "unchanged",
             "donor_ids": ["one", "two"], "runtime": {"root": "/frozen/bundle/runtime", "commit": "unchanged"}}
    path = source / "binding.json"
    path.write_text(json.dumps(value))
    manifest["files"].append({"key": "binding", "source_root": "bundle", "source_path": path.name,
                              "sha256": sha(path), "bytes": path.stat().st_size,
                              "roles": ["v_collection_binding"], "task_ids": ["one"], "selection": None,
                              "path_bindings": {"./prepared.json": "prepared.json"},
                              "hash_bindings": [{"pointer": "/file_hash", "key": "prepared.json"},
                                                {"pointer": "/canonical_hash", "key": "prepared.json", "encoding": "canonical_json"}]})
    assets.prepare_assets(config)
    result = assets.materialize_task_inputs(config, task, tmp_path / "run-inputs")
    derived = json.loads(Path(result["mapping"]["bundle/binding.json"]).read_text())
    prepared = Path(result["mapping"]["bundle/prepared.json"])
    assert derived["relative"] == str(prepared)
    assert derived["file_hash"] == sha(prepared)
    canonical = json.dumps(json.loads(prepared.read_text()), ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False).encode()
    assert derived["canonical_hash"] == hashlib.sha256(canonical).hexdigest()
    assert derived["scene_asset"] == str(tmp_path / "licensed-data/scenes/object.usd")
    assert derived["donor_ids"] == ["one", "two"] and derived["semantic_goal_hash"] == "unchanged"
    from esi_benchmark.profiles import source_roots
    assert derived["runtime"] == {"root": str(source_roots(task["profile"])["runtime_root"]), "commit": "unchanged"}


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


@pytest.mark.parametrize("mask", [0o000, 0o022, 0o077, 0o777])
def test_private_import_and_materialization_permissions_ignore_umask(inputs, mask):
    config, task, _, source, tmp_path, _ = inputs
    before = {path: stat.S_IMODE(path.stat().st_mode) for path in [tmp_path, source, *source.iterdir()]}
    config["assets"]["root"] = str(tmp_path / "nested" / "local")
    previous = os.umask(mask)
    try:
        assets.prepare_assets(config)
        assets.materialize_task_inputs(config, task, tmp_path / "run" / "private" / "inputs")
    finally:
        os.umask(previous)
    for root in (tmp_path / "nested", tmp_path / "run"):
        for path in [root, *root.rglob("*")]:
            assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)
    assert {path: stat.S_IMODE(path.stat().st_mode) for path in before} == before


def test_owned_import_permissions_are_tightened_without_changing_source(inputs):
    config, _, _, source, _, _ = inputs
    assets.prepare_assets(config)
    root = Path(config["assets"]["root"])
    root.chmod(0o755)
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o644)
    before = {path: (sha(path), path.stat().st_mode) for path in source.iterdir()}
    assert assets.prepare_assets(config)["copied"] == 0
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in root.rglob("*") if path.is_file())
    assert {path: (sha(path), path.stat().st_mode) for path in source.iterdir()} == before


def test_unmarked_nonempty_import_root_is_unchanged(inputs):
    config, _, _, _, _, _ = inputs
    root = Path(config["assets"]["root"])
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    existing = root / "someone-elses-file"
    existing.write_text("keep")
    with pytest.raises(ValueError, match="ownership marker"):
        assets.prepare_assets(config)
    assert stat.S_IMODE(root.stat().st_mode) == 0o755
    assert existing.read_text() == "keep"


def test_existing_root_not_owned_by_current_user_is_unchanged(inputs, monkeypatch):
    config, _, _, _, _, _ = inputs
    root = Path(config["assets"]["root"])
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    monkeypatch.setattr(assets.os, "geteuid", lambda: root.stat().st_uid + 1)
    with pytest.raises(ValueError, match="owned by the current user"):
        assets.prepare_assets(config)
    assert stat.S_IMODE(root.stat().st_mode) == 0o755
    assert not list(root.iterdir())


def test_source_nested_in_owned_destination_is_rejected_before_permissions_change(inputs):
    config, _, _, source, tmp_path, _ = inputs
    config["assets"]["root"] = str(tmp_path)
    marker = tmp_path / ".esi-benchmark-assets.json"
    marker.write_text(json.dumps({"profile": "l0-english-v2"}))
    before = {path: path.stat().st_mode for path in [tmp_path, marker, source, *source.iterdir()]}
    with pytest.raises(ValueError, match="overlaps"):
        assets.prepare_assets(config)
    assert {path: path.stat().st_mode for path in before} == before


def test_import_refuses_hardlinked_destination_without_chmodding_source(inputs):
    config, _, manifest, source, _, _ = inputs
    assets.prepare_assets(config)
    row = manifest["files"][0]
    original = source / row["source_path"]
    destination = assets._path(config["assets"]["root"], row)
    destination.unlink()
    os.link(original, destination)
    before = original.stat().st_mode
    with pytest.raises(ValueError, match="singly linked"):
        assets.prepare_assets(config)
    assert original.stat().st_mode == before
    assert sha(original) == row["sha256"]


def test_concurrent_exclusive_create_conflict_keeps_competing_file(inputs, monkeypatch):
    config, _, manifest, _, _, _ = inputs
    destination = assets._path(config["assets"]["root"], manifest["files"][0])
    original_open = assets._private_open

    def competing_open(path, *, exclusive=True):
        if path == destination and exclusive:
            destination.write_bytes(b"created by another importer")
        return original_open(path, exclusive=exclusive)

    monkeypatch.setattr(assets, "_private_open", competing_open)
    with pytest.raises(FileExistsError):
        assets.prepare_assets(config)
    assert destination.read_bytes() == b"created by another importer"


def test_source_open_failure_does_not_remove_competing_destination(inputs, monkeypatch):
    config, _, manifest, source, _, _ = inputs
    row = manifest["files"][0]
    source_path = source / row["source_path"]
    destination = assets._path(config["assets"]["root"], row)
    original_open = Path.open
    reads = 0

    def failing_source_open(path, *args, **kwargs):
        nonlocal reads
        if path == source_path and args == ("rb",):
            reads += 1
            if reads == 2:
                destination.write_bytes(b"created during source failure")
                raise OSError("source became unreadable")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_source_open)
    with pytest.raises(OSError, match="source became unreadable"):
        assets.prepare_assets(config)
    assert destination.read_bytes() == b"created during source failure"


@pytest.mark.parametrize("failure", ["copy_error", "hash_mismatch"])
def test_failed_copy_cleans_up_only_its_created_file(inputs, monkeypatch, failure):
    config, _, manifest, _, _, _ = inputs
    destination = assets._path(config["assets"]["root"], manifest["files"][0])

    def broken_copy(incoming, outgoing):
        assert stat.S_IMODE(os.fstat(outgoing.fileno()).st_mode) == 0o600
        outgoing.write(b"incomplete or changed source")
        if failure == "copy_error":
            raise OSError("copy failed")

    monkeypatch.setattr(assets.shutil, "copyfileobj", broken_copy)
    with pytest.raises((OSError, ValueError), match="copy failed|Source changed"):
        assets.prepare_assets(config)
    assert not destination.exists()


def test_profile_claimed_after_preflight_is_not_overwritten(inputs, monkeypatch):
    config, _, _, _, _, _ = inputs
    root = Path(config["assets"]["root"])
    marker = root / ".esi-benchmark-assets.json"
    original_directory = assets._private_directory

    def other_profile_claim(path, *, exist_ok=True):
        original_directory(path, exist_ok=exist_ok)
        if path == root:
            marker.write_text(json.dumps({"schema_version": 1, "profile": "l1-a30"}))

    monkeypatch.setattr(assets, "_private_directory", other_profile_claim)
    with pytest.raises(ValueError, match="ownership/profile mismatch"):
        assets.prepare_assets(config)
    assert json.loads(marker.read_text())["profile"] == "l1-a30"
    assert set(root.iterdir()) == {marker}


def test_concurrent_task_imports_preserve_one_complete_receipt(inputs, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    config, task, manifest, _, _, _ = inputs
    root = Path(config["assets"]["root"])
    marker = root / ".esi-benchmark-assets.json"
    for index, row in enumerate(manifest["files"]):
        row["task_ids"] = ["one" if index % 2 else "two"]
    ready = Barrier(2)
    original_open = assets.os.open

    def simultaneous_open(path, flags, mode=0o777, *, dir_fd=None):
        fd = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == root:
            ready.wait(timeout=5)
        return fd

    monkeypatch.setattr(assets.os, "open", simultaneous_open)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(assets.prepare_assets, config, task={**task, "task_id": task_id})
                   for task_id in ("one", "two")]
        assert all(result.result(timeout=10)["ok"] for result in pending)
    receipt = json.loads((root / "import_receipt.json").read_text())
    assert receipt["profile"] == config["profile"]
    assert set(receipt["files"]) == {row["key"] for row in manifest["files"]}
    assert assets.check_assets(config)["ok"]


def test_wrong_profile_rejection_preserves_existing_permissions(inputs):
    config, _, _, _, _, _ = inputs
    root = Path(config["assets"]["root"])
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    marker = root / ".esi-benchmark-assets.json"
    marker.write_text(json.dumps({"schema_version": 1, "profile": "l1-a30"}))
    marker.chmod(0o644)
    before = {path: path.stat().st_mode for path in (root, marker)}
    with pytest.raises(ValueError, match="ownership/profile mismatch"):
        assets.prepare_assets(config)
    assert {path: path.stat().st_mode for path in before} == before
    assert json.loads(marker.read_text())["profile"] == "l1-a30"


def test_other_profile_winning_root_creation_keeps_its_permissions(inputs, monkeypatch):
    config, _, _, _, _, _ = inputs
    root = Path(config["assets"]["root"])
    marker = root / ".esi-benchmark-assets.json"
    original_directory = assets._private_directory
    before = {}

    def competing_creation(path, *, exist_ok=True):
        if path == root and not root.exists():
            root.mkdir(mode=0o755)
            root.chmod(0o755)
            marker.write_text(json.dumps({"schema_version": 1, "profile": "l1-a30"}))
            marker.chmod(0o644)
            before.update({item: item.stat().st_mode for item in (root, marker)})
        original_directory(path, exist_ok=exist_ok)

    monkeypatch.setattr(assets, "_private_directory", competing_creation)
    with pytest.raises(ValueError, match="ownership/profile mismatch"):
        assets.prepare_assets(config)
    assert {path: path.stat().st_mode for path in before} == before
    assert json.loads(marker.read_text())["profile"] == "l1-a30"


def test_materialization_rechecks_the_bytes_it_actually_reads(inputs, monkeypatch):
    config, task, manifest, _, tmp_path, _ = inputs
    assets.prepare_assets(config)
    prepared = next(row for row in manifest["files"] if row["key"] == "prepared.json")
    original_resolve = assets.resolve_task_inputs

    def modified_after_resolution(config, task):
        result = original_resolve(config, task)
        assets._path(config["assets"]["root"], prepared).write_text('{"pose":[99,99,99],"goal":true}')
        return result

    monkeypatch.setattr(assets, "resolve_task_inputs", modified_after_resolution)
    with pytest.raises(ValueError, match="changed during materialization"):
        assets.materialize_task_inputs(config, task, tmp_path / "run-inputs")
    assert not (tmp_path / "run-inputs/materialization_receipt.json").exists()
