"""Pytest configuration for Feeling Engine.

Two opt-in flags control tests that require external services:

    --run-e2e     → live ElevenLabs + Modal + Anthropic pipeline (~$0.05/run)
    --run-modal   → Modal-only smoke test on a ~3-sec synthetic WAV (~$0.03/run)

Both are off by default; tests marked `e2e` or `modal` are skipped unless
the corresponding flag is set.
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


def pytest_collection_modifyitems(config, items):
    run_e2e = config.getoption("--run-e2e")
    run_modal = config.getoption("--run-modal")

    skip_e2e = pytest.mark.skip(
        reason="E2E test — run with --run-e2e to enable (costs money).",
    )
    skip_modal = pytest.mark.skip(
        reason="Modal smoke test — run with --run-modal to enable (costs ~$0.03).",
    )

    for item in items:
        if "e2e" in item.keywords and not run_e2e:
            item.add_marker(skip_e2e)
        if "modal" in item.keywords and not run_modal:
            item.add_marker(skip_modal)
