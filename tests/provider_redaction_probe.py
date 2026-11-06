"""Offline diagnostic capture against one frozen provider implementation."""
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import urllib.error


def main():
    profile, output, package = sys.argv[1:]
    sys.path.insert(0, package)
    from roboweave._native.bootstrap import activate
    activate(profile)
    import model_client as native
    from native_tool_preflight import PROBE_STEPS

    def no_network(*args, **kwargs):
        raise AssertionError("Diagnostic fixture must never open a socket")

    socket.socket = no_network
    dummy = "DUMMY_CREDENTIAL_NEVER_REAL_71842"
    os.environ["ESI_DIAGNOSTIC_TEST_KEY"] = dummy
    samples = [f"Authorization: Bearer {dummy}", f"authorization = bearer\t{dummy}",
               json.dumps({"Authorization": f"Bearer {dummy}"}),
               json.dumps({"x-api-key": dummy}), f"'api_key': '{dummy}'"]
    for sample in samples:
        error = native.ProviderRequestError(sample, failure_type="fixture", retryable=False,
                                            response_body=sample.encode())
        assert dummy not in str(error) and dummy not in json.dumps(error.artifact_fields())
    boundary = "x" * (native.MAX_PROVIDER_ERROR_PREVIEW_CHARS - 4) + dummy
    error = native.ProviderRequestError("echo:" + dummy, failure_type="fixture", retryable=False,
                                        response_body=boundary.encode(), redaction_values=(dummy,))
    fields = error.artifact_fields()
    assert dummy not in str(error) and "DUMMY" not in fields["response_body_preview"]
    assert fields["response_body_bytes"] == len(boundary.encode())
    assert fields["response_body_sha256"] == hashlib.sha256(boundary.encode()).hexdigest()
    assert native._redact_provider_error_text("quota exhausted") == "quota exhausted"

    for credential in ("DUMMY_é_CREDENTIAL_28471", 'DUMMY_"_\\_CREDENTIAL_28471'):
        lower = json.dumps({"detail": "prefix " + credential + " suffix"})
        escaped = "".join("\\u%04x" % ord(character) for character in credential)
        for body in (lower, lower.replace("\\u00e9", "\\u00E9"),
                     '{"detail":"prefix ' + escaped + ' suffix"}'):
            error = native.ProviderRequestError(body, failure_type="fixture", retryable=False,
                response_body=body.encode(), redaction_values=(credential,))
            fields = error.artifact_fields()
            assert credential not in json.loads(str(error))["detail"]
            assert credential not in json.loads(fields["response_body_preview"])["detail"]
            assert fields["response_body_sha256"] == hashlib.sha256(body.encode()).hexdigest()
            assert fields["response_body_bytes"] == len(body.encode())
    for body in (b'{"detail":"' + dummy.encode(), b'{"detail":"' + dummy.encode() + b'\xff"}',
                 b'[' * 20000 + b'"' + dummy.encode() + b'"' + b']' * 20000):
        error = native.ProviderRequestError("fixture", failure_type="fixture", retryable=False,
            response_body=body, redaction_values=(dummy,))
        assert error.artifact_fields()["response_body_preview"].startswith("<redacted")
        assert error.artifact_fields()["response_body_sha256"] == hashlib.sha256(body).hexdigest()
    body = b'{"detail":"\\ud800"}'
    error = native.ProviderRequestError("fixture", failure_type="fixture", retryable=False,
        response_body=body, redaction_values=(dummy,))
    assert native._canonical_json_bytes(error.artifact_fields())
    assert error.artifact_fields()["response_body_sha256"] == hashlib.sha256(body).hexdigest()

    def provider(api="chat_completions"):
        generation = ({"reasoning_effort": "medium", "reasoning_context": "all_turns",
                       "reasoning_mode": "standard", "text_verbosity": "low"} if api == "responses"
                      else {"reasoning_effort": "high", "maximum_output_tokens": 1024}
                      if api == "anthropic_messages" else {})
        return native.OpenAICompatibleProvider(model="fixture", base_url="https://unused.invalid/v1",
            provider_api=api, timeout_seconds=1, api_key_file=None,
            api_key_env="ESI_DIAGNOSTIC_TEST_KEY", require_api_key=True, **generation)

    raw_bodies = []

    def http_error(request, code):
        headers = {key.lower(): value for key, value in request.header_items()}
        credential = headers.get("x-api-key") or headers["authorization"].removeprefix("Bearer ")
        assert credential == dummy
        body = json.dumps({"detail": "prefix" + credential + "suffix",
                           "Authorization": "Bearer " + credential}).encode()
        raw_bodies.append(body)
        return urllib.error.HTTPError(request.full_url, code, "fixture failure", {}, io.BytesIO(body))

    class FailureOpener:
        def open(self, request, **kwargs):
            raise http_error(request, 401)

    apis = ["chat_completions", "responses"]
    if profile != "l2-collection37":
        apis.append("anthropic_messages")
    for api in apis:
        transport = provider(api)
        transport.opener = FailureOpener()
        try:
            transport._post({"model": "fixture"})
        except native.ProviderRequestError as failure:
            artifact = failure.artifact_fields()
            assert dummy not in str(failure) and dummy not in json.dumps(artifact)
            assert failure.response_body == raw_bodies[-1]
            assert artifact["response_body_bytes"] == len(raw_bodies[-1])
            assert artifact["response_body_sha256"] == hashlib.sha256(raw_bodies[-1]).hexdigest()
            assert artifact["http_status"] == 401 and artifact["retryable"] is False
        else:
            raise AssertionError("HTTP failure was not propagated")

    for credential in (dummy, 'DUMMY_ESCAPED_"_\\_é_CREDENTIAL_28471'):
        os.environ["ESI_DIAGNOSTIC_TEST_KEY"] = credential
        reason = json.dumps({"detail": "prefix " + credential + " suffix"}).replace("\\u00e9", "\\u00E9")
        for failure_type in (urllib.error.URLError, ConnectionError, TimeoutError):
            class TransportFailureOpener:
                def open(self, request, **kwargs):
                    raise failure_type(reason)

            transport = provider()
            transport.opener = TransportFailureOpener()
            try:
                transport._post({"model": "fixture"})
            except native.ProviderRequestError as failure:
                for text in (str(failure), failure.artifact_fields()["exception_message"]):
                    assert credential not in json.loads(text.split(": ", 1)[1])["detail"]
                assert failure.failure_type == ("provider_timeout" if failure_type is TimeoutError else "provider_transport_error")
                assert failure.retryable is True
            else:
                raise AssertionError("Transport failure was not propagated")
    os.environ["ESI_DIAGNOSTIC_TEST_KEY"] = dummy

    for unsafe in (dummy + "\nnext", dummy + "\rnext", dummy + "\u2603"):
        os.environ["ESI_DIAGNOSTIC_TEST_KEY"] = unsafe
        transport = provider()
        transport.opener = type("NeverCalled", (), {"open": no_network})()
        try:
            transport._post({"model": "fixture"})
        except native.ModelClientError as failure:
            assert str(failure) == "provider API key contains invalid HTTP header characters"
            assert dummy not in str(failure)
        else:
            raise AssertionError("Unsafe HTTP header credential was accepted")
    os.environ["ESI_DIAGNOSTIC_TEST_KEY"] = dummy

    class RetryOpener:
        count = 0

        def open(self, request, **kwargs):
            self.count += 1
            if self.count == 1:
                raise http_error(request, 500)
            step = PROBE_STEPS[self.count - 2]
            arguments = {"action": step.expected_action}
            if step.expected_target is not None:
                arguments.update(target=step.expected_target, target_description=step.expected_target_description)
            response = {"model": "fixture", "choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{"id": "fixture-" + str(self.count),
                    "type": "function", "function": {"name": "robot_action", "arguments": json.dumps(arguments)}}]}}]}
            return io.BytesIO(json.dumps(response).encode())

    transport = provider()
    transport.opener = RetryOpener()
    prompt = native.load_common_system_prompt_revision(native.ACTION20_CORE_INSTRUCTIONS_PATH,
        physical_history_max=5, maximum_action_decisions=33)
    path = Path(output) / "provider_preflight.json"
    report = transport.capability_preflight(prompt_revision=prompt, physical_history_max=5,
        maximum_action_decisions=33, artifact=path, maximum_provider_attempts=2,
        provider_retry_policy=native.ProviderRetryPolicy())
    assert report["status"] == "PASS" and report["provider_failures"] == report["provider_retries"] == 1
    assert dummy not in path.read_text()
    failure = report["provider_failure_attempts"][0]
    assert failure["http_status"] == 500 and failure["retryable"] is True
    assert failure["response_body_sha256"] == hashlib.sha256(raw_bodies[-1]).hexdigest()
    assert failure["response_body_bytes"] == len(raw_bodies[-1])
    print(json.dumps({"profile": profile, "transports": apis, "status": "PASS", "network_used": False}))


if __name__ == "__main__":
    main()
