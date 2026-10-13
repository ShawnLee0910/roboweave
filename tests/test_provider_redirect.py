"""Exercise urllib's real redirect processing without opening a socket."""

from email.message import Message
import io
import json
import socket
import urllib.request
import urllib.response

import pytest

from lab import model_client


class _MemoryTransport(urllib.request.BaseHandler):
    handler_order = 100

    def __init__(self, status, location):
        self.status = status
        self.location = location
        self.requests = []

    def https_open(self, request):
        self.requests.append(request)
        headers = Message()
        code = self.status if len(self.requests) == 1 else 200
        if code != 200:
            headers["Location"] = self.location
        response = urllib.response.addinfourl(
            io.BytesIO(b"{}"), headers, request.full_url, code
        )
        response.msg = "synthetic response"
        return response

    http_open = https_open


@pytest.fixture
def provider_factory(monkeypatch):
    monkeypatch.setenv("ESI_TEST_REDIRECT_KEY", "synthetic-redirect-key")

    def reject_socket(*args, **kwargs):
        pytest.fail("redirect tests must never open a socket")

    monkeypatch.setattr(socket, "create_connection", reject_socket)

    def create(api):
        generation = {}
        if api in {"responses", "anthropic_messages"}:
            generation["reasoning_effort"] = "medium"
        if api == "responses":
            generation.update(
                reasoning_context="all_turns",
                reasoning_mode="standard",
                text_verbosity="low",
            )
        return model_client.OpenAICompatibleProvider(
            model="synthetic-model",
            base_url="https://provider.example/v1",
            timeout_seconds=1,
            api_key_file=None,
            api_key_env="ESI_TEST_REDIRECT_KEY",
            provider_api=api,
            **generation,
        )

    return create


@pytest.mark.parametrize("api", ["chat_completions", "responses", "anthropic_messages"])
@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize(
    "location",
    [
        "https://other.example/collect",
        "http://other.example/collect",
        "http://provider.example/collect",
        "/same-origin",
    ],
)
def test_provider_rejects_redirect_without_forwarding_credentials(
    provider_factory, api, status, location
):
    provider = provider_factory(api)
    transport = _MemoryTransport(status, location)
    provider.opener.add_handler(transport)

    with pytest.raises(model_client.ProviderRequestError) as captured:
        provider._post({"model": provider.model})

    error = captured.value
    assert error.status_code == status
    assert error.failure_type == f"provider_http_{status}"
    assert error.retryable is False
    assert len(transport.requests) == 1
    sent = transport.requests[0]
    assert sent.get_method() == "POST"
    if api == "anthropic_messages":
        assert sent.get_header("X-api-key") == "synthetic-redirect-key"
    else:
        assert sent.get_header("Authorization") == "Bearer synthetic-redirect-key"
    assert "synthetic-redirect-key" not in json.dumps(error.artifact_fields())
    assert location not in str(error)


@pytest.mark.parametrize("api", ["chat_completions", "responses", "anthropic_messages"])
def test_successful_authenticated_response_is_unchanged(provider_factory, api):
    provider = provider_factory(api)
    transport = _MemoryTransport(200, None)
    provider.opener.add_handler(transport)

    assert provider._post({"model": provider.model}) == {}
    assert len(transport.requests) == 1
