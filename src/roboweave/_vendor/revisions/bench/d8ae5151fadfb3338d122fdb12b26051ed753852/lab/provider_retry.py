"""Deterministic, bounded retry timing for model-provider requests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class ProviderRetryPolicy:
    """Application-level retry timing layered above one HTTP request."""

    initial_backoff_seconds: float = 0.0
    maximum_backoff_seconds: float = 0.0
    maximum_elapsed_seconds: float = 0.0
    jitter_ratio: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.initial_backoff_seconds,
            self.maximum_backoff_seconds,
            self.maximum_elapsed_seconds,
            self.jitter_ratio,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError("provider retry policy values must be finite numbers")
        if not 0.0 <= float(self.initial_backoff_seconds) <= 60.0:
            raise ValueError("provider retry initial backoff must be in [0, 60]")
        if not (
            float(self.initial_backoff_seconds)
            <= float(self.maximum_backoff_seconds)
            <= 300.0
        ):
            raise ValueError(
                "provider retry maximum backoff must be between initial and 300"
            )
        if not 0.0 <= float(self.maximum_elapsed_seconds) <= 3600.0:
            raise ValueError("provider retry elapsed budget must be in [0, 3600]")
        if not 0.0 <= float(self.jitter_ratio) <= 0.5:
            raise ValueError("provider retry jitter ratio must be in [0, 0.5]")
        if self.enabled and float(self.maximum_elapsed_seconds) <= 0.0:
            raise ValueError("enabled provider retry backoff needs an elapsed budget")

    @property
    def enabled(self) -> bool:
        return float(self.initial_backoff_seconds) > 0.0

    def to_record(self) -> dict[str, float]:
        return {
            "initial_backoff_seconds": float(self.initial_backoff_seconds),
            "maximum_backoff_seconds": float(self.maximum_backoff_seconds),
            "maximum_elapsed_seconds": float(self.maximum_elapsed_seconds),
            "jitter_ratio": float(self.jitter_ratio),
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
    ) -> "ProviderRetryPolicy":
        if value is None:
            return cls()
        expected = {
            "initial_backoff_seconds",
            "maximum_backoff_seconds",
            "maximum_elapsed_seconds",
            "jitter_ratio",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("provider retry policy fields changed")
        return cls(**{field: value[field] for field in expected})

    def delay_seconds(
        self,
        *,
        retry_index: int,
        jitter_key: str,
        retry_after_seconds: float | None = None,
    ) -> float:
        """Return one deterministic jittered delay for retry_index >= 1."""

        if isinstance(retry_index, bool) or not isinstance(retry_index, int):
            raise TypeError("provider retry index must be an integer")
        if retry_index < 1:
            raise ValueError("provider retry index must be positive")
        if not isinstance(jitter_key, str) or not jitter_key:
            raise ValueError("provider retry jitter key must be nonempty")
        base = min(
            float(self.maximum_backoff_seconds),
            float(self.initial_backoff_seconds) * (2.0 ** (retry_index - 1)),
        )
        digest = hashlib.sha256(
            f"{jitter_key}:{retry_index}".encode("utf-8")
        ).digest()
        unit_interval = int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)
        signed = unit_interval * 2.0 - 1.0
        delay = base * (1.0 + float(self.jitter_ratio) * signed)
        if retry_after_seconds is not None:
            if (
                isinstance(retry_after_seconds, bool)
                or not isinstance(retry_after_seconds, (int, float))
                or not math.isfinite(float(retry_after_seconds))
                or float(retry_after_seconds) < 0.0
            ):
                raise ValueError("Retry-After seconds must be finite and nonnegative")
            delay = max(delay, float(retry_after_seconds))
        return round(max(0.0, delay), 6)


def parse_retry_after_seconds(
    value: str | None,
    *,
    now_epoch_seconds: float | None = None,
) -> float | None:
    """Parse either delta-seconds or an HTTP-date Retry-After value."""

    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    try:
        seconds = float(normalized)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(normalized)
        except (TypeError, ValueError, OverflowError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        now = (
            datetime.now(timezone.utc).timestamp()
            if now_epoch_seconds is None
            else float(now_epoch_seconds)
        )
        seconds = parsed.timestamp() - now
    if not math.isfinite(seconds) or seconds < 0.0:
        return None
    return round(seconds, 6)
