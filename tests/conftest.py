"""Pytest configuration for Feeling Engine.

Adds a --run-e2e flag. E2E tests hit live APIs (ElevenLabs, Modal,
Anthropic) and cost real money. Skipped by default; opt in with:
    pytest --run-e2e
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


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-e2e"):
        return
    skip_e2e = pytest.mark.skip(
        reason="E2E test — run with --run-e2e to enable (costs money).",
    )
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)
