"""Private Oracle websocket transport policy.

This module is deliberately dependency-light so the resident policy server can
import its keepalive contract without importing OmniGibson or the evaluator.
The official client/server defaults remain untouched; this policy applies only
to the local Oracle control boundary.
"""

from __future__ import annotations


ORACLE_WEBSOCKET_PING_INTERVAL_S = None
ORACLE_WEBSOCKET_PING_TIMEOUT_S = None


def oracle_websocket_server_options() -> dict[str, object]:
    """Return kwargs for the private Oracle ``websockets.serve`` call."""

    return {
        "ping_interval": ORACLE_WEBSOCKET_PING_INTERVAL_S,
        "ping_timeout": ORACLE_WEBSOCKET_PING_TIMEOUT_S,
    }


def oracle_websocket_keepalive_manifest() -> dict[str, object]:
    """Return auditable transport settings for the private Oracle boundary."""

    return {
        "scope": "private_oracle",
        "client": {
            "ping_interval_s": ORACLE_WEBSOCKET_PING_INTERVAL_S,
            "ping_timeout_s": ORACLE_WEBSOCKET_PING_TIMEOUT_S,
        },
        "server": {
            "ping_interval_s": ORACLE_WEBSOCKET_PING_INTERVAL_S,
            "ping_timeout_s": ORACLE_WEBSOCKET_PING_TIMEOUT_S,
        },
        "reason": "long_cpu_precompute",
        "qa_default_unchanged": True,
    }
