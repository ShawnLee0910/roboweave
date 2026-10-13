from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys

import pytest

from esi_benchmark._native.bootstrap import activate


def test_installed_bundle_locations():
    roots = activate()
    assert (roots["bench_root"] / "action20_tool_schema.json").is_file()
    assert (roots["runtime_root"] / "src/behavior_hl/__init__.py").is_file()
    assert not roots["bench_root"].is_symlink()


def test_rejects_other_profile_and_mixed_source(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        activate("l1-physical")
    monkeypatch.setitem(sys.modules, "Agent", SimpleNamespace(__file__=str(tmp_path / "Agent.py")))
    with pytest.raises(RuntimeError, match="mix"):
        activate()


@pytest.mark.parametrize("name", [
    "agent_protocol", "agent_observation", "model_client", "lab.model_client",
    "behavior_hl.navigation", "behavior_hl.not_in_bundle", "compile_l0_fixed_reset_poses",
])
def test_checks_all_bundled_import_names_each_time(monkeypatch, tmp_path, name):
    activate()
    monkeypatch.setitem(sys.modules, name, SimpleNamespace(__file__=str(tmp_path / "foreign.py")))
    with pytest.raises(RuntimeError, match=name):
        activate()


@pytest.mark.parametrize("origin", [None, "wrong-bundled-file"])
def test_requires_the_exact_module_file(monkeypatch, origin):
    roots = activate()
    if origin is not None:
        origin = str(roots["bench_root"] / "agent_prompt.py")
    monkeypatch.setitem(sys.modules, "agent_protocol", SimpleNamespace(__file__=origin))
    with pytest.raises(RuntimeError, match="agent_protocol"):
        activate()


@pytest.mark.parametrize("locations", [None, [], ["foreign"], ["candidate", "foreign"]])
def test_rejects_missing_or_mixed_namespace_paths(monkeypatch, tmp_path, locations):
    roots = activate()
    namespace = SimpleNamespace(__file__=None)
    if locations is not None:
        namespace.__path__ = [str(roots["bench_root"] / "lab") if value == "candidate"
                              else str(tmp_path / "lab") for value in locations]
    monkeypatch.setitem(sys.modules, "lab", namespace)
    with pytest.raises(RuntimeError, match="lab"):
        activate()


def test_rejects_foreign_search_path_on_valid_runtime_package(monkeypatch, tmp_path):
    runtime = activate()["runtime_root"] / "src/behavior_hl"
    monkeypatch.setitem(sys.modules, "behavior_hl", SimpleNamespace(
        __file__=str(runtime / "__init__.py"), __path__=[str(runtime), str(tmp_path)]))
    with pytest.raises(RuntimeError, match="behavior_hl"):
        activate()


def _isolated_python(script, tmp_path, *arguments):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    result = subprocess.run([sys.executable, "-c", script, *map(str, arguments)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("entrypoint", ["activate"])
def test_rejects_preloaded_protocol_with_different_tolerance(tmp_path, entrypoint):
    bench = activate()["bench_root"]
    source = (bench / "agent_protocol.py").read_text()
    assert "_NAVIGATION_DISTANCE_ABS_TOL_M = 1e-3 + 1e-12" in source
    (tmp_path / "agent_protocol.py").write_text(source.replace(
        "_NAVIGATION_DISTANCE_ABS_TOL_M = 1e-3 + 1e-12", "_NAVIGATION_DISTANCE_ABS_TOL_M = 0.02"))
    _isolated_python("""
import sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import agent_protocol
assert agent_protocol._NAVIGATION_DISTANCE_ABS_TOL_M == 0.02
from esi_benchmark._native.bootstrap import activate
from esi_benchmark.cli import doctor
try:
    {"activate": activate, "doctor": doctor}[sys.argv[3]]()
except RuntimeError as error:
    assert "agent_protocol" in str(error)
else:
    raise AssertionError("foreign protocol was accepted")
assert sys.modules["agent_protocol"] is agent_protocol
""", tmp_path, tmp_path, bench, entrypoint)


@pytest.mark.parametrize("entrypoint", ["activate"])
def test_rejects_foreign_runtime_submodule_in_isolated_process(tmp_path, entrypoint):
    foreign = tmp_path / "l0_verifier_adapter.py"
    foreign.write_text("EXTERNAL_SOURCE = True\n")
    _isolated_python("""
import importlib.util
import sys
from esi_benchmark._native.bootstrap import activate
from esi_benchmark.cli import doctor
activate()
import behavior_hl.embodied.evaluator_bridge
name = "behavior_hl.embodied.evaluator_bridge.l0_verifier_adapter"
spec = importlib.util.spec_from_file_location(name, sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
assert module.EXTERNAL_SOURCE
try:
    {"activate": activate, "doctor": doctor}[sys.argv[2]]()
except RuntimeError as error:
    assert name in str(error)
else:
    raise AssertionError("foreign verifier was accepted")
assert sys.modules[name] is module
""", tmp_path, foreign, entrypoint)


def test_normal_namespace_and_repeated_doctor_in_isolated_process(tmp_path):
    _isolated_python("""
import sys
from esi_benchmark._native.bootstrap import activate
from esi_benchmark.cli import doctor
roots = activate()
import lab
import lab.scene_protocol
import model_client
assert lab.__file__ is None
assert doctor()["ok"]
sys.path.insert(0, "/nonexistent-other-checkout")
activate()
assert sys.path[0] == str(roots["bench_root"])
assert doctor()["ok"]
""", tmp_path)
