import copy
import json
from pathlib import Path

import pytest

from esi_benchmark.config import load_config, validate_config


def valid():
    return {"schema_version": "esi-benchmark/config-v1", "profile": "l0-english-v2", "mode": "replay",
            "run": {"task_id": "example", "family": "L0", "output_root": "outputs", "history_frames": 5}}


def test_resolves_paths_against_config_not_cwd(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(valid()))
    monkeypatch.chdir(tmp_path.parent)
    assert load_config(path)["run"]["output_root"] == str(tmp_path / "outputs")


@pytest.mark.parametrize("field,value", [("mode", "mock_live"), ("mode", []), ("mode", {}), ("profile", "latest"), ("unknown", 1)])
def test_rejects_unknown_semantics(field, value):
    config = valid()
    config[field] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_rejects_duplicate_fields(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"mode":"replay","mode":"live"}')
    with pytest.raises(ValueError, match="Duplicate"):
        load_config(path)


def test_rejects_unfrozen_history_and_implicit_live():
    config = valid()
    config["run"]["history_frames"] = 3
    with pytest.raises(ValueError):
        validate_config(config)
    config = valid()
    config["mode"] = "live"
    with pytest.raises(ValueError, match="assets.root"):
        validate_config(config)


def test_interpreter_symlink_keeps_virtual_environment_entry(tmp_path):
    original = tmp_path / "base-python"
    original.touch()
    entry = tmp_path / "venv/bin/python"
    entry.parent.mkdir(parents=True)
    entry.symlink_to(original)
    config = valid()
    config["runtime"] = {"python": "venv/bin/python"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert load_config(path)["runtime"]["python"] == str(entry)
