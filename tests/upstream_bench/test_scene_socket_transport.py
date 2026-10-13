from __future__ import annotations

import pytest

from lab.scene_socket_transport import send_response_or_client_gone


class _Connection:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.payloads: list[bytes] = []

    def sendall(self, payload: bytes) -> None:
        if self.error is not None:
            raise self.error
        self.payloads.append(payload)


@pytest.mark.parametrize(
    "error",
    [BrokenPipeError(), ConnectionResetError(), ConnectionAbortedError()],
)
def test_disconnected_client_is_connection_local(error: BaseException) -> None:
    assert send_response_or_client_gone(_Connection(error), b"{}\n") is False


def test_live_client_receives_exact_payload() -> None:
    connection = _Connection()
    assert send_response_or_client_gone(connection, b'{"status":"PASS"}\n') is True
    assert connection.payloads == [b'{"status":"PASS"}\n']


@pytest.mark.parametrize("payload", [b"", bytearray(b"x"), "x", None])
def test_invalid_payload_fails_closed(payload) -> None:
    with pytest.raises(TypeError):
        send_response_or_client_gone(_Connection(), payload)
