from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from esi_benchmark.bootstrap import activate


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
