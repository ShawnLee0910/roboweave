from __future__ import annotations

from datetime import datetime, timezone

import pytest

from provider_retry import ProviderRetryPolicy, parse_retry_after_seconds


def test_exponential_backoff_is_bounded_deterministic_and_jittered() -> None:
    policy = ProviderRetryPolicy(
        initial_backoff_seconds=2.0,
        maximum_backoff_seconds=30.0,
        maximum_elapsed_seconds=600.0,
        jitter_ratio=0.2,
    )
    delays = [
        policy.delay_seconds(retry_index=index, jitter_key="episode:sequence")
        for index in range(1, 10)
    ]
    assert delays == [
        policy.delay_seconds(retry_index=index, jitter_key="episode:sequence")
        for index in range(1, 10)
    ]
    assert 1.6 <= delays[0] <= 2.4
    assert 3.2 <= delays[1] <= 4.8
    assert all(24.0 <= delay <= 36.0 for delay in delays[4:])


def test_retry_after_is_a_minimum() -> None:
    policy = ProviderRetryPolicy(
        initial_backoff_seconds=2.0,
        maximum_backoff_seconds=30.0,
        maximum_elapsed_seconds=600.0,
        jitter_ratio=0.2,
    )
    assert policy.delay_seconds(
        retry_index=1,
        jitter_key="request",
        retry_after_seconds=45.0,
    ) == 45.0


def test_retry_after_parses_seconds_and_http_date() -> None:
    assert parse_retry_after_seconds("12") == 12.0
    now = datetime(2026, 8, 18, tzinfo=timezone.utc).timestamp()
    assert parse_retry_after_seconds(
        "Tue, 18 Aug 2026 00:00:30 GMT",
        now_epoch_seconds=now,
    ) == 30.0
    assert parse_retry_after_seconds("not-a-date") is None


@pytest.mark.parametrize(
    "value",
    [
        {"initial_backoff_seconds": 2.0},
        {
            "initial_backoff_seconds": 2.0,
            "maximum_backoff_seconds": 1.0,
            "maximum_elapsed_seconds": 600.0,
            "jitter_ratio": 0.2,
        },
        {
            "initial_backoff_seconds": 2.0,
            "maximum_backoff_seconds": 30.0,
            "maximum_elapsed_seconds": 0.0,
            "jitter_ratio": 0.2,
        },
    ],
)
def test_invalid_policy_fails_closed(value: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ProviderRetryPolicy.from_mapping(value)
