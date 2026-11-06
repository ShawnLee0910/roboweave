"""Exercise old inputs through the renamed public API and native worker."""

import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from roboweave.config import validate_config
from roboweave.scoring import score_record
from roboweave.trajectory import _evidence


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_imports_share_modules_and_native_loader_state():
    script = '''
import importlib, sys
before = list(sys.path)
import roboweave, esi_benchmark
assert roboweave is esi_benchmark
for name in ("tasks", "profiles", "models", "scoring", "runtime", "cli", "_native.bootstrap"):
    assert importlib.import_module("roboweave." + name) is importlib.import_module("esi_benchmark." + name)
assert sys.path == before
assert "Agent" not in sys.modules
assert "behavior_hl" not in sys.modules
from importlib.resources import files
assert files("roboweave") == files("esi_benchmark")
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("prefix", ["roboweave", "esi-benchmark"])
def test_config_accepts_only_the_two_declared_v1_names(prefix):
    config = json.loads((ROOT / "configs/replay.json").read_text())
    config["schema_version"] = prefix + "/config-v1"
    before = copy.deepcopy(config)
    assert validate_config(config) == before
    config["schema_version"] = prefix + "/config-v99"
    with pytest.raises(ValueError, match="schema_version"):
        validate_config(config)


@pytest.mark.parametrize("example", ["score-record.json", "history-score-record.json"])
def test_old_and_new_score_records_have_identical_results(example, tmp_path):
    record = json.loads((ROOT / "examples" / example).read_text())
    old = copy.deepcopy(record)
    old["schema_version"] = old["schema_version"].replace("roboweave/", "esi-benchmark/")
    before = copy.deepcopy(old)
    legacy_dir, current_dir = tmp_path / "legacy", tmp_path / "current"
    legacy = score_record(old, process_directory=legacy_dir)
    current = score_record(record, process_directory=current_dir)
    assert json.dumps(legacy, sort_keys=True).replace(str(legacy_dir), "$PROCESS") == (
        json.dumps(current, sort_keys=True).replace(str(current_dir), "$PROCESS"))
    assert old == before


@pytest.mark.parametrize("schema", ["roboweave/result-evidence-v2", "esi-benchmark/result-evidence-v2"])
def test_evidence_aliases_do_not_bypass_strict_task_binding(tmp_path, schema):
    with pytest.raises(ValueError, match="task binding"):
        _evidence(tmp_path, {"evidence_schema_version": schema, "evidence": {}})


@pytest.mark.parametrize("schema", [None, [], "esi-benchmark/score-record-v99", "roboweave/score-record-v99"])
def test_score_aliases_reject_unsupported_versions(schema):
    with pytest.raises(ValueError, match="score record"):
        score_record({"schema_version": schema})
