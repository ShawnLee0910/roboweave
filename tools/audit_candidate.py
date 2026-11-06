#!/usr/bin/env python3
"""Audit the candidate's distributable files; report locations, never secret values."""

import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SCANNED_ROOTS = ("src", "configs", "examples", "docs", "provenance", "reports", "tests", "tools")
SCANNED_FILES = ("pyproject.toml", "MANIFEST.in", "README.md", "LICENSE_STATUS.md", "THIRD_PARTY_NOTICES.md")
SECRET_PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    "provider_key": re.compile(rb"\b(?:sk-[A-Za-z0-9_-]{24,}|AKIA[A-Z0-9]{16})\b"),
    "credential_url": re.compile(rb"https?://[^/\s:@]+:[^/\s@]+@"),
}
# Exact synthetic literals exercise URL rejection and provider-error redaction.
AUDITED_TEST_LITERALS = {
    ("tests/test_models.py", "credential_url", "ec25b8fa54a9d7a878a86e6f38233af24d5d27d184b396136d6d7f0251d33c19"),
    ("reports/pytest.xml", "credential_url", "ec25b8fa54a9d7a878a86e6f38233af24d5d27d184b396136d6d7f0251d33c19"),
    ("reports/rc2/pytest.xml", "credential_url", "ec25b8fa54a9d7a878a86e6f38233af24d5d27d184b396136d6d7f0251d33c19"),
    ("reports/rc2/sdist-pytest.xml", "credential_url", "ec25b8fa54a9d7a878a86e6f38233af24d5d27d184b396136d6d7f0251d33c19"),
    ("reports/rc3/pytest.xml", "credential_url", "ec25b8fa54a9d7a878a86e6f38233af24d5d27d184b396136d6d7f0251d33c19"),
    ("reports/rc3/final-related.xml", "credential_url", "ec25b8fa54a9d7a878a86e6f38233af24d5d27d184b396136d6d7f0251d33c19"),
    ("reports/rc4/pytest.xml", "credential_url", "ec25b8fa54a9d7a878a86e6f38233af24d5d27d184b396136d6d7f0251d33c19"),
    ("reports/rc4/final-related.xml", "credential_url", "ec25b8fa54a9d7a878a86e6f38233af24d5d27d184b396136d6d7f0251d33c19"),
    ("reports/rc5/remaining.xml", "credential_url", "ec25b8fa54a9d7a878a86e6f38233af24d5d27d184b396136d6d7f0251d33c19"),
    ("tests/upstream_bench/test_model_protocol_retry.py", "provider_key", "d7741caceb829b07fe116f5ba10ea4415e361d02efbedddd352674b25b678ced"),
}


def audit():
    count = 0
    byte_count = 0
    findings = []
    syntax_errors = []
    internal_paths = []
    synthetic_findings = []
    for relative in (*SCANNED_ROOTS, *SCANNED_FILES):
        base = ROOT / relative
        if base.is_symlink():
            findings.append({"path": relative, "kind": "symlink"})
            continue
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")) if base.is_dir() else (base,):
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_symlink():
                findings.append({"path": str(path.relative_to(ROOT)), "kind": "symlink"})
                continue
            if not path.is_file():
                continue
            if path.stat().st_size > 16 * 1024 * 1024:
                findings.append({"path": str(path.relative_to(ROOT)), "kind": "oversized_file"})
                continue
            content = path.read_bytes()
            count += 1
            byte_count += len(content)
            for kind, pattern in SECRET_PATTERNS.items():
                for match in pattern.finditer(content):
                    relative = str(path.relative_to(ROOT))
                    key = (relative, kind, hashlib.sha256(match.group()).hexdigest())
                    if key in AUDITED_TEST_LITERALS:
                        synthetic_findings.append({"path": relative, "kind": kind, "literal_sha256": key[2]})
                    else:
                        findings.append({"path": relative, "kind": kind})
            if path.suffix == ".py":
                try:
                    ast.parse(content, filename=str(path.relative_to(ROOT)))
                except SyntaxError as exc:
                    syntax_errors.append({"path": str(path.relative_to(ROOT)), "line": exc.lineno})
            if re.search(rb"/(?:liyuanshuai|root)/", content):
                internal_paths.append(str(path.relative_to(ROOT)))
    return {"schema_version": "roboweave/source-audit-v1", "files": count, "bytes": byte_count,
            "secret_or_symlink_findings": findings, "syntax_errors": syntax_errors,
            "reviewed_synthetic_test_literals": synthetic_findings,
            "source_files_with_historical_absolute_paths": internal_paths,
            "coverage": [*SCANNED_ROOTS, *SCANNED_FILES],
            "limits": "Pattern scan, not proof of absence of every possible credential. Historical path references require reachability review."}


if __name__ == "__main__":
    report = audit()
    destination = ROOT / "provenance" / "source_audit.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "source_files_with_historical_absolute_paths"}, indent=2))
    raise SystemExit(bool(report["secret_or_symlink_findings"] or report["syntax_errors"]))
