import importlib.util
from pathlib import Path


def test_scan_reports_only_locations_and_never_secret_values(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "tools/audit_candidate.py"
    spec = importlib.util.spec_from_file_location("candidate_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "src").mkdir()
    synthetic = "https:" + "//user:" + "password@example.invalid"
    (tmp_path / "src/fixture.py").write_text(f'value = "{synthetic}"')
    monkeypatch.setattr(module, "ROOT", tmp_path)
    result = module.audit()
    assert result["secret_or_symlink_findings"] == [{"path": "src/fixture.py", "kind": "credential_url"}]
    assert "password" not in repr(result)
