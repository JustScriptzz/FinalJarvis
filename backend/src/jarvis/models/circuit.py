"""Per-provider circuit breaker.

Prevents a dead local server or a rate-limited cloud API from adding latency to
every request. State is process-local; a multi-process deployment would move
this to Redis alongside the event bus.
"""

from __future__ import annotations

import time

from ..core.config import CircuitBreakerConfig


class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig) -> None:
        self._config = config
        self._state = "closed"
        self._failures = 0
        self._opened_at = 0.0
        self._half_open_calls = 0

    @property
    def state(self) -> str:
        return self._state

    def allow(self) -> bool:
        if self._state == "closed":
            return True
        if self._state == "open":
            if time.monotonic() - self._opened_at >= self._config.reset_timeout_s:
                self._state = "half_open"
                self._half_open_calls = 0
                return True
            return False
        if self._half_open_calls < self._config.half_open_max_calls:
            self._half_open_calls += 1
            return True
        return False

    def record_success(self) -> None:
        self._state = "closed"
        self._failures = 0
        self._half_open_calls = 0

    def record_failure(self) -> None:
        if self._state == "half_open":
            self._state = "open"
            self._opened_at = time.monotonic()
            self._half_open_calls = 0
            return
        self._failures += 1
        if self._failures >= self._config.failure_threshold:
            self._state = "open"
            self._opened_at = time.monotonic()

    def snapshot(self) -> dict[str, object]:
        retry_after = None
        if self._state == "open":
            retry_after = max(
                0.0,
                self._config.reset_timeout_s - (time.monotonic() - self._opened_at),
            )
        return {
            "state": self._state,
            "failures": self._failures,
            "retry_after_s": round(retry_after, 2) if retry_after is not None else None,
        }


__all__ = ["CircuitBreaker"]
