"""Retry wrapper for Gemini API calls that hit transient server errors.

Gemini occasionally returns 503 UNAVAILABLE, 429 rate-limit, or 5xx transient
errors when capacity spikes on Google's side. The Google SDK already retries
internally a few times; this wrapper adds an outer layer of longer waits so
most transient failures become eventual successes.

Usage:
    from feeling_engine._gemini_retry import gemini_with_retry

    response = gemini_with_retry(
        client.models.generate_content,
        model=..., contents=..., config=...,
    )

Not imported as a decorator to keep call sites readable — wrap the single
call at the site instead.
"""
from __future__ import annotations

import time
from typing import Any, Callable, TypeVar


T = TypeVar("T")

RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
RETRYABLE_MESSAGE_MARKERS = (
    "503", "502", "504", "500",
    "unavailable", "overloaded",
    "rate limit", "too many requests",
    "deadline exceeded",
)

DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_WAIT_SEC = 30.0
DEFAULT_MAX_WAIT_SEC = 120.0
DEFAULT_BACKOFF_MULTIPLIER = 2.0


def _is_retryable(exc: BaseException) -> bool:
    """Return True if exception looks like a transient Gemini server error."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int) and status in RETRYABLE_STATUSES:
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in RETRYABLE_MESSAGE_MARKERS)


def gemini_with_retry(
    fn: Callable[..., T],
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_wait_sec: float = DEFAULT_INITIAL_WAIT_SEC,
    max_wait_sec: float = DEFAULT_MAX_WAIT_SEC,
    backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
    verbose: bool = True,
    **kwargs: Any,
) -> T:
    """Call fn(*args, **kwargs) with retry on transient Gemini server errors.

    Args:
        fn: the callable to invoke (e.g., client.models.generate_content).
        *args / **kwargs: passed to fn.
        max_retries: number of retry attempts AFTER the first try. Default 3.
        initial_wait_sec: wait before first retry. Default 30s.
        max_wait_sec: cap on exponential backoff. Default 120s.
        backoff_multiplier: wait *= this each retry. Default 2.0.
        verbose: print retry progress. Default True.

    Returns:
        fn's return value if any attempt succeeds.

    Raises:
        the last exception if all attempts fail, or immediately if the
        exception is not retryable (auth errors, bad-request, etc.).
    """
    wait = initial_wait_sec
    last_exc: BaseException | None = None

    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except BaseException as e:
            if not _is_retryable(e):
                raise
            last_exc = e
            if attempt == max_retries:
                break
            if verbose:
                etype = type(e).__name__
                print(f"    ! transient Gemini error ({etype}): "
                      f"retry {attempt+1}/{max_retries} in {wait:.0f}s...",
                      flush=True)
            time.sleep(wait)
            wait = min(wait * backoff_multiplier, max_wait_sec)

    assert last_exc is not None  # defensive; should always be set if we reach here
    raise last_exc
