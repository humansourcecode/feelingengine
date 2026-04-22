"""Pytest configuration for Feeling Engine.

Three opt-in flags control tests that require external services:

    --run-e2e     → live ElevenLabs + Modal + Anthropic pipeline (~$0.05/run)
    --run-modal   → Modal-only smoke test on a ~3-sec synthetic WAV (~$0.03/run)
    --run-gemini  → Gemini video API smoke test on a 5-sec synthetic clip (~$0.01/run)

All off by default; marked tests are skipped unless the corresponding flag is set.
"""
from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run E2E tests that hit live APIs (costs money).",
    )
    parser.addoption(
        "--run-modal",
        action="store_true",
        default=False,
        help="Run Modal-only TRIBE smoke tests (~$0.03 per run).",
    )
    parser.addoption(
        "--run-gemini",
        action="store_true",
        default=False,
        help="Run Gemini video API smoke tests (~$0.01 per run).",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: live E2E test (skipped without --run-e2e)")
    config.addinivalue_line("markers", "modal: Modal smoke test (skipped without --run-modal)")
    config.addinivalue_line("markers", "gemini: Gemini smoke test (skipped without --run-gemini)")


def pytest_collection_modifyitems(config, items):
    run_e2e = config.getoption("--run-e2e")
    run_modal = config.getoption("--run-modal")
    run_gemini = config.getoption("--run-gemini")

    skip_e2e = pytest.mark.skip(
        reason="E2E test — run with --run-e2e to enable (costs money).",
    )
    skip_modal = pytest.mark.skip(
        reason="Modal smoke test — run with --run-modal to enable (costs ~$0.03).",
    )
    skip_gemini = pytest.mark.skip(
        reason="Gemini smoke test — run with --run-gemini to enable (costs ~$0.01).",
    )

    for item in items:
        has_gemini = item.get_closest_marker("gemini") is not None
        has_modal = item.get_closest_marker("modal") is not None
        has_e2e = item.get_closest_marker("e2e") is not None

        # Explicit per-marker gating — check actual markers, not path keywords,
        # so tests under tests/e2e/ aren't auto-treated as e2e-marked.
        if has_gemini and not run_gemini:
            item.add_marker(skip_gemini)
        elif has_modal and not run_modal:
            item.add_marker(skip_modal)
        elif has_e2e and not run_e2e:
            item.add_marker(skip_e2e)
