"""Resolve and verify immutable source bundles without importing native code."""

import hashlib
import json
from pathlib import Path

from .tasks import profile_for

DEFAULT_PROFILE = "l0-english-v2"
_PACKAGE_ROOT = Path(__file__).resolve().parent


def _registry():
    return json.loads((_PACKAGE_ROOT / "data/source_bundles.json").read_text())


def source_revisions(profile=DEFAULT_PROFILE):
    frozen = profile_for(profile)
    try:
        return {kind: frozen[kind + "_revision"] for kind in ("bench", "runtime")}
    except KeyError:
        raise ValueError(f"Profile has no executable source identity: {profile}") from None


def _entries(profile):
    registry = _registry()["bundles"]
    entries = {}
    for kind, revision in source_revisions(profile).items():
        try:
            entries[kind] = (revision, registry[kind][revision])
        except KeyError:
            raise ValueError(f"No frozen {kind} source bundle for profile {profile!r}") from None
    return entries


def source_roots(profile=DEFAULT_PROFILE):
    roots = {}
    for kind, (_, entry) in _entries(profile).items():
        relative = Path(entry["path"])
        root = (_PACKAGE_ROOT / relative).resolve()
        if relative.is_absolute() or not root.is_relative_to(_PACKAGE_ROOT):
            raise ValueError(f"Source bundle path escapes the installed package: {kind}")
        if not root.is_dir():
            raise ValueError(f"Installed source bundle is missing: {kind}")
        roots[kind + "_root"] = root
    return roots


def registered_revision(root):
    """Recognize installed roots even when their identity manifest is missing."""
    root = Path(root).resolve()
    for entries in _registry()["bundles"].values():
        for revision, entry in entries.items():
            if (_PACKAGE_ROOT / entry["path"]).resolve() == root:
                return revision
    return None


def verify_bundle(root, revision, manifest_sha256=None):
    """Fail closed on missing, modified, unexpected or redirected source files."""
    root = Path(root).resolve()
    if manifest_sha256 is None:
        entries = [entries[revision] for entries in _registry()["bundles"].values()
                   if revision in entries]
        entries = [entry for entry in entries if (_PACKAGE_ROOT / entry["path"]).resolve() == root]
        if len(entries) != 1:
            raise ValueError("Source root is not the registered frozen bundle")
        manifest_sha256 = entries[0]["manifest_sha256"]
    manifest_path = root / "SOURCE_IDENTITY.json"
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ValueError("Source identity manifest is missing or invalid") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if digest != manifest_sha256:
        raise ValueError("Source identity manifest changed")
    if manifest.get("upstream_revision") != revision:
        raise ValueError("Source bundle revision differs from the frozen profile")
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Source identity has no file hashes")
    actual = set()
    for path in root.rglob("*"):
        if "__pycache__" in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            raise ValueError("Source bundle contains a symlink")
        if path.is_file() and path != manifest_path:
            actual.add(path.relative_to(root).as_posix())
    if actual != set(expected):
        raise ValueError("Source bundle file inventory changed")
    for name, expected_digest in expected.items():
        path = root / name
        if Path(name).is_absolute() or not path.resolve().is_relative_to(root):
            raise ValueError("Source identity path escapes its bundle")
        try:
            actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError(f"Source bundle file is unavailable: {name}") from exc
        if actual_digest != expected_digest:
            raise ValueError(f"Source bundle file changed: {name}")
    return {"revision": revision, "manifest_sha256": digest,
            "file_count": len(expected), "status": "PASS"}


def verify_sources(profile=DEFAULT_PROFILE):
    roots = source_roots(profile)
    reports = {}
    for kind, (revision, entry) in _entries(profile).items():
        reports[kind] = verify_bundle(roots[kind + "_root"], revision,
                                      entry["manifest_sha256"])
    return {"profile": profile, "status": "PASS", "sources": reports}
