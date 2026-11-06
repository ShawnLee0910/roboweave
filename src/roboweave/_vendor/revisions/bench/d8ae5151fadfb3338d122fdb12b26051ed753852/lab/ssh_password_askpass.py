#!/usr/bin/env python3
"""Minimal non-interactive SSH askpass helper.

The password is supplied only through the parent process environment.  This
helper never stores or logs it and refuses an empty value.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    password = os.environ.get("BENCH_SSH_PASSWORD", "")
    if not password:
        return 1
    sys.stdout.write(password)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
