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
    result = subprocess.run([sys.executable, "-c", script, *map(str, arguments)],
                            cwd=tmp_path, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                                "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
                            capture_output=True, text=True, timeout=30)
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


@pytest.mark.parametrize("regular_package", [False, True])
def test_frozen_namespace_cannot_be_extended_by_foreign_search_paths(tmp_path, regular_package):
    foreign = tmp_path / "foreign"
    (foreign / "lab").mkdir(parents=True)
    (foreign / "lab/foreign_probe.py").write_text("EXTERNAL = True\n")
    if regular_package:
        (foreign / "lab/__init__.py").write_text("EXTERNAL = True\n")
    _isolated_python("""
import importlib
import sys
from esi_benchmark._native.bootstrap import activate
sys.path.insert(0, sys.argv[1])
roots = activate()
import lab
import lab.scene_protocol
assert lab.__file__ is None
assert list(lab.__path__) == [str(roots["bench_root"] / "lab")]
try:
    importlib.import_module("lab.foreign_probe")
except ModuleNotFoundError:
    pass
else:
    raise AssertionError("foreign namespace extension was accepted")
activate()
""", tmp_path, foreign)


def test_frozen_import_reads_verified_source_and_leaves_other_loaders_alone(tmp_path):
    _isolated_python("""
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import py_compile
import sys
from esi_benchmark import profiles
from esi_benchmark._native import bootstrap
root = Path.cwd()
bench, runtime, third_party = (root / name for name in ("bench", "runtime", "third_party"))
for directory in (bench / "lab", runtime / "src", runtime / "scripts", third_party):
    directory.mkdir(parents=True)
for directory, name in ((bench, "frozen_probe"), (third_party, "ordinary_probe")):
    source = directory / (name + ".py")
    source.write_text("value = 2\\n")
    timestamp = source.stat().st_mtime
    py_compile.compile(str(source), doraise=True)
    source.write_text("value = 1\\n")
    os.utime(source, (timestamp, timestamp))
manifest = bench / "SOURCE_IDENTITY.json"
manifest.write_text(json.dumps({"upstream_revision": "a" * 40, "files": {
    "frozen_probe.py": hashlib.sha256((bench / "frozen_probe.py").read_bytes()).hexdigest()}}))
assert profiles.verify_bundle(bench, "a" * 40, hashlib.sha256(manifest.read_bytes()).hexdigest())["status"] == "PASS"
bootstrap.source_roots = lambda _: {"bench_root": bench, "runtime_root": runtime}
sys.path.insert(0, str(third_party))
bootstrap.activate()
import frozen_probe
import ordinary_probe
assert frozen_probe.value == 1, "verified source was replaced by unverified bytecode"
assert ordinary_probe.value == 2
assert type(ordinary_probe.__loader__) is importlib.machinery.SourceFileLoader
assert list((bench / "__pycache__").glob("*.pyc")), "normal install caches must be preserved"
""", tmp_path)


def test_projected_verifier_dynamic_import_reads_hashed_source(tmp_path, monkeypatch):
    import hashlib
    import py_compile
    from behavior_hl.embodied.evaluator_bridge import l0_taskspec_live_adapter as adapter

    verifier = tmp_path / "l0_verifier_adapter.py"
    verifier.write_text("class ProjectedGoalVerifier:\n    value = 2\n")
    timestamp = verifier.stat().st_mtime
    py_compile.compile(str(verifier), doraise=True)
    verifier.write_text("class ProjectedGoalVerifier:\n    value = 1\n")
    os.utime(verifier, (timestamp, timestamp))
    monkeypatch.setattr(adapter, "__file__", str(tmp_path / "l0_taskspec_live_adapter.py"))
    factory = adapter._import_projected_verifier_class
    factory.cache_clear()
    try:
        cls = factory(str(verifier), hashlib.sha256(verifier.read_bytes()).hexdigest())
        assert cls.value == 1, "dynamic verifier ignored its source digest and loaded bytecode"
    finally:
        factory.cache_clear()


def test_activation_rejects_preloaded_source_with_unverified_loader(tmp_path):
    _isolated_python("""
import importlib.util
import sys
from esi_benchmark.profiles import source_roots
from esi_benchmark._native.bootstrap import activate
path = source_roots()["bench_root"] / "action20_protocol.py"
spec = importlib.util.spec_from_file_location("action20_protocol", path)
module = importlib.util.module_from_spec(spec)
sys.modules["action20_protocol"] = module
spec.loader.exec_module(module)
try:
    activate()
except RuntimeError as error:
    assert "action20_protocol" in str(error)
else:
    raise AssertionError("preloaded source may have executed unverified bytecode")
""", tmp_path)
