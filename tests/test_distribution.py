"""A source archive must contain and run its advertised verification tools."""

import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def test_source_archive_preserves_verification_material(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("src", "tests", "tools", "docs", "configs", "examples", "provenance", "reports"):
        shutil.copytree(ROOT / name, source / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
    for name in ("pyproject.toml", "MANIFEST.in", "README.md", "LICENSE_STATUS.md", "THIRD_PARTY_NOTICES.md"):
        shutil.copyfile(ROOT / name, source / name)
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp_path)}
    build = subprocess.run(
        [sys.executable, "-c", "from setuptools.build_meta import build_sdist; build_sdist('dist')"],
        cwd=source, env=environment, capture_output=True, text=True, timeout=90,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    archives = list((source / "dist").glob("*.tar.gz"))
    assert len(archives) == 1
    unpacked = tmp_path / "unpacked"
    with tarfile.open(archives[0], "r:gz") as archive:
        members = archive.getmembers()
        assert all(not PurePosixPath(m.name).is_absolute() and ".." not in PurePosixPath(m.name).parts
                   and (m.isfile() or m.isdir()) for m in members)
        names = {"/".join(PurePosixPath(m.name).parts[1:]) for m in members if m.isfile()}
        for directory in ("tests", "tools", "reports", "provenance"):
            # Compare the build's snapshot; the live checkout may gain test reports.
            for path in (source / directory).rglob("*"):
                if path.is_file() and path.suffix in {".py", ".json", ".txt", ".log", ".xml", ".md"}:
                    assert str(path.relative_to(source)) in names
        # Members were built locally and checked above; no links or escaping paths are allowed.
        archive.extractall(unpacked)
    release = next(unpacked.iterdir())
    environment["PYTHONPATH"] = str(release / "src")
    checks = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_equivalence.py", "tests/test_release_audit.py",
         "-q", "-p", "no:cacheprovider", "--basetemp", str(tmp_path / "checks")],
        cwd=release, env=environment, capture_output=True, text=True, timeout=90,
    )
    assert checks.returncode == 0, checks.stdout + checks.stderr
    cache = source / "src/roboweave/_vendor/bench/__pycache__"
    cache.mkdir(exist_ok=True)
    (cache / "stale.cpython-312.pyc").write_bytes(b"synthetic cache; never import")
    wheel = subprocess.run(
        [sys.executable, "-c", "from setuptools.build_meta import build_wheel; build_wheel('wheels')"],
        cwd=source, env=environment, capture_output=True, text=True, timeout=90,
    )
    assert wheel.returncode == 0, wheel.stdout + wheel.stderr
    with zipfile.ZipFile(next((source / "wheels").glob("*.whl"))) as archive:
        assert not any(name.endswith((".pyc", ".pyo")) for name in archive.namelist())
