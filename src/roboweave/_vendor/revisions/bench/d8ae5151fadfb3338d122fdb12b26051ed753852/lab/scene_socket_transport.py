"""Pure socket-response boundary for the resident scene service."""

from __future__ import annotations

import socket


CLIENT_DISCONNECT_ERRORS = (
    BrokenPipeError,
    ConnectionAbortedError,
    ConnectionResetError,
)


def send_response_or_client_gone(
    connection: socket.socket,
    payload: bytes,
) -> bool:
    """Send one response without allowing a dead client to stop the scene."""

    if not isinstance(payload, bytes) or not payload:
        raise TypeError("scene response payload must be nonempty bytes")
    try:
        connection.sendall(payload)
    except CLIENT_DISCONNECT_ERRORS:
        return False
    return True


__all__ = ["CLIENT_DISCONNECT_ERRORS", "send_response_or_client_gone"]
