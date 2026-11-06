from pathlib import Path
import sys
import tempfile
import os

ROOT = Path(__file__).resolve().parents[1]
(ROOT / ".tmp").mkdir(exist_ok=True)
tempfile.tempdir = str(ROOT / ".tmp")
os.environ["TMPDIR"] = tempfile.tempdir
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(ROOT / "src"))

from roboweave._native.bootstrap import activate

activate()
# Upstream tests share small CPU-only neighboring helpers.
for relative in ("tests/upstream_bench", "tests/upstream_runtime", "tests/upstream_runtime/embodied"):
    sys.path.insert(0, str(ROOT / relative))
