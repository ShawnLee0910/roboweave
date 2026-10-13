"""The expected outputs were produced by two frozen upstream Git revisions."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_installed_code_matches_frozen_source_golden(tmp_path):
    report_path = tmp_path / "equivalence.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/check_equivalence.py"), "--report", str(report_path)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(report_path.read_text())
    assert report["case_count"] == 75
    assert report["differences"] == []
    assert report["reference_output_sha256"] == report["candidate_output_sha256"]
    recorded = json.loads((ROOT / "provenance/equivalence.json").read_text())
    assert recorded["reference_mode"] == "frozen_git_subprocess"
    assert report["inputs_sha256"] == recorded["inputs_sha256"]
    assert report["reference_source_sha256"] == recorded["reference_source_sha256"]
    assert report["reference_output_sha256"] == recorded["reference_output_sha256"]


def test_normalization_preserves_semantic_fields():
    spec = importlib.util.spec_from_file_location("check_equivalence", ROOT / "tools/check_equivalence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = {"path": "/fixture/run/image.png", "budget": 150, "status": "blocked", "q": 0.5}
    normalized = module.normalize(value, [(Path("/fixture/run"), "$WORK")])
    assert normalized == {**value, "path": "$WORK/image.png"}
    assert value["path"] == "/fixture/run/image.png"
    for field, replacement in (("budget", 149), ("status", "completed"), ("q", 1.0)):
        altered = module.normalize({**value, field: replacement}, [(Path("/fixture/run"), "$WORK")])
        assert hashlib.sha256(module.canonical(altered)).digest() != hashlib.sha256(module.canonical(normalized)).digest()
