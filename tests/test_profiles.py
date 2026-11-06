"""CPU checks for frozen-source identity and import-only safety adaptations."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from roboweave import profiles


def frozen_bundle(root):
    root.mkdir()
    raw = b"value = 1\n"
    (root / "module.py").write_bytes(raw)
    identity = {"upstream_revision": "a" * 40,
                "files": {"module.py": hashlib.sha256(raw).hexdigest()}}
    (root / "SOURCE_IDENTITY.json").write_text(json.dumps(identity))
    return root


def test_verify_bundle_accepts_exact_files(tmp_path):
    root = frozen_bundle(tmp_path / "bundle")
    digest = hashlib.sha256((root / "SOURCE_IDENTITY.json").read_bytes()).hexdigest()
    result = profiles.verify_bundle(root, "a" * 40, digest)
    assert result["status"] == "PASS"
    assert result["file_count"] == 1
    json.dumps(result)


@pytest.mark.parametrize("mutation", ["changed", "missing", "extra", "symlink", "manifest", "revision"])
def test_verify_bundle_rejects_changed_identity(tmp_path, mutation):
    root = frozen_bundle(tmp_path / "bundle")
    manifest_digest = hashlib.sha256((root / "SOURCE_IDENTITY.json").read_bytes()).hexdigest()
    revision = "a" * 40
    if mutation == "changed":
        (root / "module.py").write_text("value = 2\n")
    elif mutation == "missing":
        (root / "module.py").unlink()
    elif mutation == "extra":
        (root / "other.py").write_text("pass\n")
    elif mutation == "symlink":
        (root / "module.py").rename(tmp_path / "outside.py")
        (root / "module.py").symlink_to(tmp_path / "outside.py")
    elif mutation == "manifest":
        with (root / "SOURCE_IDENTITY.json").open("a") as stream:
            stream.write(" ")
    else:
        revision = "b" * 40
    with pytest.raises(ValueError):
        profiles.verify_bundle(root, revision, manifest_digest)


def test_source_roots_do_not_follow_registry_escape(monkeypatch, tmp_path):
    monkeypatch.setattr(profiles, "_PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(profiles, "_entries", lambda _: {"bench": ("a" * 40, {"path": "../outside"})})
    with pytest.raises(ValueError, match="escapes"):
        profiles.source_roots("test")


@pytest.mark.parametrize("redirect", ["root", "parent"])
@pytest.mark.parametrize("operation", ["source_roots", "registered_revision", "verify_bundle"])
def test_registered_bundle_rejects_symlink_entries(monkeypatch, tmp_path, redirect, operation):
    root = frozen_bundle(tmp_path / "bundle")
    digest = hashlib.sha256((root / "SOURCE_IDENTITY.json").read_bytes()).hexdigest()
    if redirect == "root":
        link = tmp_path / "redirect"
        link.symlink_to(root, target_is_directory=True)
    else:
        (tmp_path / "redirect").symlink_to(tmp_path, target_is_directory=True)
        link = tmp_path / "redirect/bundle"
    entry = {"path": str(link.relative_to(tmp_path)), "manifest_sha256": digest}
    monkeypatch.setattr(profiles, "_PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(profiles, "_registry", lambda: {"bundles": {"bench": {"a" * 40: entry}}})
    monkeypatch.setattr(profiles, "_entries", lambda _: {"bench": ("a" * 40, entry)})
    with pytest.raises(ValueError, match="symlink"):
        if operation == "source_roots":
            profiles.source_roots("test")
        elif operation == "registered_revision":
            profiles.registered_revision(root)
        else:
            profiles.verify_bundle(root, "a" * 40)


def test_registered_bundles_are_complete_and_syntax_valid():
    package = Path(profiles.__file__).parent
    registry = profiles._registry()
    for kind, revisions in registry["bundles"].items():
        for revision, entry in revisions.items():
            root = package / entry["path"]
            profiles.verify_bundle(root, revision, entry["manifest_sha256"])
            for path in root.rglob("*.py"):
                ast.parse(path.read_bytes(), filename=str(path))
            if kind == "bench":
                assert (root / "lab/package_scene_server.py").is_file()
                source = (root / "lab/model_client.py").read_text()
                assert "class _NoProviderRedirect(" in source
                assert "BEHAVIOR_HL_PROJECT_ROOT" in source
                assert "api_key_env" in source
            else:
                assert (root / "src/behavior_hl/embodied/evaluator_bridge/registered_l0_task_runtime.py").is_file()
                assert not (root / "configs/materials/vray_maps.mdl").exists()
                assert not (root / "configs/materials/vray_materials.mdl").exists()


@pytest.mark.parametrize("profile", ["l0-english-v2", "l1-a30", "l1-m30-public-v2", "l1-v30",
                                    "l1-p-inside5", "l1-p-ontop5", "l1-p-additions20", "l2-collection37"])
def test_each_profile_resolves_exact_revisions(profile):
    report = profiles.verify_sources(profile)
    assert report["status"] == "PASS"
    expected = profiles.source_revisions(profile)
    assert {kind: value["revision"] for kind, value in report["sources"].items()} == expected


def test_l2_proposals_never_alias_an_executable_profile():
    with pytest.raises(ValueError):
        profiles.source_roots("l2-proposals")


def _importer():
    path = Path(__file__).resolve().parents[1] / "tools/import_profile_sources.py"
    spec = importlib.util.spec_from_file_location("profile_source_importer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_importer_follows_root_lab_and_relative_dependencies(monkeypatch):
    module = _importer()
    monkeypatch.setattr(module, "read_records", lambda _: [{"source_path": "Agent.py"}])
    files = {"Agent.py": b"from lab.first import run\n",
             "lab/first.py": b"from .second import run\n",
             "lab/second.py": b"from third import run\n",
             "lab/third.py": b"def run(): pass\n",
             "lab/unrelated.py": b"raise RuntimeError('must not import')\n"}
    assert set(module.bench_closure(files)) == set(files) - {"lab/unrelated.py"}


def test_importer_adapts_only_matching_context_and_preserves_new_logic():
    module = _importer()
    old = "a = 1\nb = 2\nc = 3\nd = 4\ne = 5\n"
    adapted = old.replace("b = 2", "b = 6")
    newer = old.replace("d = 4", "d = 7")
    result = module.apply_adaptation(old, adapted, newer, "fixture.py")
    assert result == newer.replace("b = 2", "b = 6")
    with pytest.raises(ValueError, match="requires review"):
        module.apply_adaptation(old, adapted, newer.replace("b = 2", "b = 9"), "fixture.py")


def test_importer_limits_source_only_adaptation_to_projected_verifier_loader():
    module = _importer()
    original = "def load():\n    spec = importlib.util.spec_from_file_location(module_name, path)\n    return spec\n"
    result = module.adapt_projected_verifier_loader(original)
    assert "loader=_SourceOnlyLoader(module_name, str(path))" in result
    assert result.endswith("    return spec\n")
    ast.parse(result)
    with pytest.raises(ValueError, match="requires review"):
        module.adapt_projected_verifier_loader(result)


def _scene_identity_module():
    path = profiles.source_roots()["bench_root"] / "lab/scene_source_identity.py"
    spec = importlib.util.spec_from_file_location("bundled_scene_identity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scene_identity_uses_frozen_revisions_without_candidate_git(monkeypatch):
    module = _scene_identity_module()
    monkeypatch.setattr(module, "subprocess", object())
    revisions = profiles.source_revisions()
    for kind, revision in revisions.items():
        root = profiles.source_roots()[kind + "_root"]
        assert module.git_source_head(root) == revision
        identity = module.git_source_identity(root)
        assert identity == {"root": str(root), "git_commit": revision, "dirty": False}
        assert module.validate_source_identity(identity) == identity


def test_scene_identity_rejects_missing_registered_manifest(monkeypatch, tmp_path):
    module = _scene_identity_module()
    root = tmp_path / "source"
    root.mkdir()
    monkeypatch.setattr(profiles, "_PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(profiles, "_registry", lambda: {"bundles": {
        "bench": {"a" * 40: {"path": "source", "manifest_sha256": "b" * 64}}, "runtime": {}}})
    monkeypatch.setattr(module, "subprocess", object())
    with pytest.raises(ValueError, match="manifest is missing"):
        module.git_source_identity(root)


def test_external_checkout_requires_exact_git_root_and_retains_dirty_flag(tmp_path):
    module = _scene_identity_module()
    root = tmp_path / "external"
    root.mkdir()
    (root / "subdirectory").mkdir()
    source = root / "subdirectory/source.py"
    source.write_text("value = 1\n")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE).decode().strip()

    git("init", "-q")
    git("add", "subdirectory/source.py")
    git("-c", "user.name=CPU Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-qm", "source identity fixture")
    revision = git("rev-parse", "HEAD")
    identity = module.git_source_identity(root)
    assert identity == {"root": str(root), "git_commit": revision, "dirty": False}
    assert module.validate_source_identity(identity) == identity
    with pytest.raises(ValueError, match="independent Git checkout"):
        module.git_source_head(root / "subdirectory")
    source.write_text("value = 2\n")
    module._cached_git_source_identity.cache_clear()
    identity = module.git_source_identity(root)
    assert identity["dirty"] is True
    with pytest.raises(ValueError, match="source changed"):
        module.validate_source_identity(identity)
