"""Unit tests for the Gemini retry wrapper."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from feeling_engine._gemini_retry import gemini_with_retry, _is_retryable


# ─── _is_retryable ─────────────────────────────────────────────

def test_is_retryable_by_status_code_attribute():
    e = Exception("weird")
    e.status_code = 503
    assert _is_retryable(e) is True


def test_is_retryable_by_code_attribute():
    e = Exception("weird")
    e.code = 429
    assert _is_retryable(e) is True


def test_is_retryable_by_message_marker():
    assert _is_retryable(Exception("503 UNAVAILABLE — try later")) is True
    assert _is_retryable(Exception("Rate limit exceeded")) is True
    assert _is_retryable(Exception("Model is overloaded")) is True


def test_is_retryable_rejects_client_errors():
    e = Exception("bad request")
    e.status_code = 400
    assert _is_retryable(e) is False

    e2 = Exception("not authorized")
    e2.status_code = 401
    assert _is_retryable(e2) is False

    assert _is_retryable(ValueError("invalid input")) is False


# ─── gemini_with_retry ─────────────────────────────────────────

def test_returns_result_on_success():
    fn = MagicMock(return_value="ok")
    result = gemini_with_retry(fn, "a", x=1, verbose=False)
    assert result == "ok"
    fn.assert_called_once_with("a", x=1)


def test_retries_on_retryable_error_then_succeeds():
    # First call raises 503, second call returns "ok"
    err = Exception("503 UNAVAILABLE")
    fn = MagicMock(side_effect=[err, "ok"])
    # Use tiny wait to keep test fast
    result = gemini_with_retry(fn, verbose=False, initial_wait_sec=0.01,
                                max_wait_sec=0.01, backoff_multiplier=1.0)
    assert result == "ok"
    assert fn.call_count == 2


def test_reraises_non_retryable_immediately():
    fn = MagicMock(side_effect=ValueError("malformed input"))
    with pytest.raises(ValueError, match="malformed"):
        gemini_with_retry(fn, verbose=False, initial_wait_sec=0.01)
    fn.assert_called_once()


def test_gives_up_after_max_retries():
    err = Exception("503 UNAVAILABLE")
    fn = MagicMock(side_effect=err)
    with pytest.raises(Exception, match="503"):
        gemini_with_retry(fn, verbose=False, max_retries=2,
                          initial_wait_sec=0.01, max_wait_sec=0.01,
                          backoff_multiplier=1.0)
    # 1 initial + 2 retries = 3 total attempts
    assert fn.call_count == 3


def test_exponential_backoff_respects_max_wait(monkeypatch):
    """Verify wait time grows exponentially but is capped at max_wait_sec."""
    waits = []
    monkeypatch.setattr("time.sleep", lambda s: waits.append(s))
    err = Exception("503")
    fn = MagicMock(side_effect=[err, err, err, "ok"])
    gemini_with_retry(fn, verbose=False, max_retries=3,
                      initial_wait_sec=1.0, max_wait_sec=3.0,
                      backoff_multiplier=2.0)
    # waits: 1 (initial), 2 (doubled), 3 (would be 4 but capped)
    assert waits == [1.0, 2.0, 3.0]
