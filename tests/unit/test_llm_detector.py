"""Unit tests for llm_detector — parsing + validation logic, no API calls."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from feeling_engine.mechanisms.llm_detector import (
    _build_vocabulary_section,
    _strip_code_fences,
    detect_from_video,
)
from feeling_engine.mechanisms.vocabulary import MECHANISM_LABELS


def test_vocabulary_section_includes_all_28_labels():
    section = _build_vocabulary_section()
    for spec in MECHANISM_LABELS:
        assert f"**{spec.name}**" in section
        assert spec.phenomenology.split(".")[0][:30] in section


def test_strip_code_fences_removes_generic_fences():
    assert _strip_code_fences("```\nhello\n```") == "hello"


def test_strip_code_fences_removes_language_tagged_fences():
    assert _strip_code_fences("```json\n{\"a\": 1}\n```") == '{"a": 1}'


def test_strip_code_fences_passes_plain_text_through():
    assert _strip_code_fences('{"a": 1}') == '{"a": 1}'


def _mock_response_with_applications(apps_json: list) -> MagicMock:
    """Build a MagicMock shaped like a Gemini generate_content response."""
    payload = json.dumps({"applications": apps_json})
    part = MagicMock()
    part.text = payload
    candidate = MagicMock()
    candidate.content.parts = [part]
    response = MagicMock()
    response.candidates = [candidate]
    usage = MagicMock()
    usage.prompt_token_count = 100
    usage.candidates_token_count = 50
    response.usage_metadata = usage
    return response


def _mock_client_that_returns(apps_json: list) -> MagicMock:
    client = MagicMock()
    # Upload returns ACTIVE-state file on first call
    uploaded = MagicMock()
    uploaded.state.name = "ACTIVE"
    uploaded.name = "files/abc123"
    client.files.upload.return_value = uploaded
    client.files.get.return_value = uploaded
    client.models.generate_content.return_value = _mock_response_with_applications(apps_json)
    return client


@pytest.fixture
def fixture_video(tmp_path):
    """Any file exists — _probe_duration will be patched."""
    p = tmp_path / "sample.mp4"
    p.write_bytes(b"fake-mp4-bytes")
    return p


def test_detect_parses_valid_applications(fixture_video, monkeypatch):
    monkeypatch.setenv("GOOGLE_AI_API_KEY", "test-key")
    client = _mock_client_that_returns([
        {"label": "body-turn", "start_sec": 1.0, "end_sec": 2.5,
         "intensity": 0.8, "confidence": 0.9, "evidence": "sudden pause"},
        {"label": "recognition", "start_sec": 3.0, "end_sec": 4.0,
         "intensity": 0.6, "confidence": 0.7, "evidence": "familiar face"},
    ])

    with patch("feeling_engine.mechanisms.llm_detector._probe_duration", return_value=5.0), \
         patch("google.genai.Client", return_value=client):
        apps = detect_from_video(fixture_video, verbose=False)

    assert len(apps) == 2
    assert apps[0].label == "body-turn"
    assert apps[0].tier == 1
    assert apps[0].start_sec == 1.0
    assert apps[0].intensity == 0.8
    assert apps[0].signals["detector"] == "llm-gemini"
    assert apps[0].signals["evidence"] == "sudden pause"
    assert apps[1].label == "recognition"


def test_detect_filters_unknown_labels(fixture_video, monkeypatch):
    monkeypatch.setenv("GOOGLE_AI_API_KEY", "test-key")
    client = _mock_client_that_returns([
        {"label": "body-turn", "start_sec": 1.0, "end_sec": 2.0,
         "intensity": 0.5, "confidence": 0.5, "evidence": "x"},
        {"label": "invented-label", "start_sec": 2.0, "end_sec": 3.0,
         "intensity": 0.5, "confidence": 0.5, "evidence": "y"},
    ])

    with patch("feeling_engine.mechanisms.llm_detector._probe_duration", return_value=5.0), \
         patch("google.genai.Client", return_value=client):
        apps = detect_from_video(fixture_video, verbose=False)

    # Only the valid label should pass through
    assert len(apps) == 1
    assert apps[0].label == "body-turn"


def test_detect_clamps_intensity_and_confidence(fixture_video, monkeypatch):
    monkeypatch.setenv("GOOGLE_AI_API_KEY", "test-key")
    client = _mock_client_that_returns([
        {"label": "body-turn", "start_sec": 0.0, "end_sec": 1.0,
         "intensity": 1.5, "confidence": -0.2, "evidence": "x"},
    ])

    with patch("feeling_engine.mechanisms.llm_detector._probe_duration", return_value=5.0), \
         patch("google.genai.Client", return_value=client):
        apps = detect_from_video(fixture_video, verbose=False)

    assert len(apps) == 1
    assert apps[0].intensity == 1.0
    assert apps[0].confidence == 0.0


def test_detect_sorts_by_start_sec(fixture_video, monkeypatch):
    monkeypatch.setenv("GOOGLE_AI_API_KEY", "test-key")
    client = _mock_client_that_returns([
        {"label": "body-turn", "start_sec": 4.0, "end_sec": 5.0,
         "intensity": 0.5, "confidence": 0.5, "evidence": ""},
        {"label": "recognition", "start_sec": 1.0, "end_sec": 2.0,
         "intensity": 0.5, "confidence": 0.5, "evidence": ""},
    ])

    with patch("feeling_engine.mechanisms.llm_detector._probe_duration", return_value=5.0), \
         patch("google.genai.Client", return_value=client):
        apps = detect_from_video(fixture_video, verbose=False)

    assert [a.start_sec for a in apps] == [1.0, 4.0]


def test_detect_raises_on_missing_api_key(fixture_video, monkeypatch):
    monkeypatch.delenv("GOOGLE_AI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_AI_API_KEY"):
        detect_from_video(fixture_video, verbose=False)


def test_detect_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        detect_from_video(tmp_path / "does-not-exist.mp4")
