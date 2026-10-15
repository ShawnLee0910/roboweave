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


def test_internal_absolute_paths_are_listed_for_every_distributable_file_type(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "tools/audit_candidate.py"
    spec = importlib.util.spec_from_file_location("candidate_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "src/_vendor").mkdir(parents=True)
    internal = "/" + "liyuanshuai" + "/historical-checkout"
    (tmp_path / "src/_vendor/SOURCE_IDENTITY.json").write_text(f'{{"root": "{internal}"}}')
    (tmp_path / "src/_vendor/profiles.yaml").write_text(f"tro_state: {internal}/state.json\n")
    (tmp_path / "src/clean.json").write_text('{"root": "relative"}')
    monkeypatch.setattr(module, "ROOT", tmp_path)
    assert module.audit()["source_files_with_historical_absolute_paths"] == [
        "src/_vendor/SOURCE_IDENTITY.json", "src/_vendor/profiles.yaml"]


def test_scan_includes_top_level_distribution_files_and_root_symlinks(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "tools/audit_candidate.py"
    spec = importlib.util.spec_from_file_location("candidate_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    synthetic = "https:" + "//user:" + "password@example.invalid"
    (tmp_path / "README.md").write_text(synthetic)
    (tmp_path / "src").symlink_to(tmp_path / "absent", target_is_directory=True)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    findings = module.audit()["secret_or_symlink_findings"]
    assert {tuple(finding.values()) for finding in findings} == {
        ("README.md", "credential_url"), ("src", "symlink")}
    assert "password" not in repr(findings)
