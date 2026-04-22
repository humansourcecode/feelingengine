"""Unit tests for preprocess.multimodal — parsing + transcript slicing, no API calls."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from feeling_engine.preprocess.multimodal import (
    _slice_transcript,
    _strip_code_fences,
    preprocess_video,
    enriched_transcript,
    EnrichedSegment,
)


def test_slice_transcript_returns_empty_for_none():
    assert _slice_transcript(None, 0.0, 5.0) == ""


def test_slice_transcript_returns_empty_for_no_words_key():
    assert _slice_transcript({}, 0.0, 5.0) == ""


def test_slice_transcript_selects_words_in_window():
    transcript = {"words": [
        {"word": "one",   "start": 0.0, "end": 0.5},
        {"word": "two",   "start": 1.0, "end": 1.5},
        {"word": "three", "start": 3.0, "end": 3.5},
        {"word": "four",  "start": 6.0, "end": 6.5},
    ]}
    assert _slice_transcript(transcript, 0.0, 5.0) == "one two three"
    assert _slice_transcript(transcript, 3.0, 7.0) == "three four"


def test_slice_transcript_includes_boundary_words():
    transcript = {"words": [
        {"word": "edge", "start": 4.9, "end": 5.1},
    ]}
    assert _slice_transcript(transcript, 0.0, 5.0) == "edge"


def test_strip_code_fences_handles_plain_text():
    assert _strip_code_fences("just text") == "just text"


def test_enriched_transcript_concatenates_segments():
    segs = [
        EnrichedSegment(0.0, 1.0, "", "v1", "a1", "p1", "m1", "SEG1"),
        EnrichedSegment(1.0, 2.0, "", "v2", "a2", "p2", "m2", "SEG2"),
    ]
    out = enriched_transcript(segs)
    assert "SEG1" in out and "SEG2" in out
    assert out == "SEG1\nSEG2"


def _mock_response_with_chunks(chunks_json: list) -> MagicMock:
    payload = json.dumps(chunks_json)
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


def _mock_client(chunks_json: list) -> MagicMock:
    client = MagicMock()
    uploaded = MagicMock()
    uploaded.state.name = "ACTIVE"
    uploaded.name = "files/xyz"
    client.files.upload.return_value = uploaded
    client.files.get.return_value = uploaded
    client.models.generate_content.return_value = _mock_response_with_chunks(chunks_json)
    return client


@pytest.fixture
def fixture_video(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"fake-mp4")
    return p


def test_preprocess_returns_segments(fixture_video, monkeypatch):
    monkeypatch.setenv("GOOGLE_AI_API_KEY", "test-key")
    client = _mock_client([
        {"start_sec": 0.0, "end_sec": 5.0,
         "visual_cues": "A face fills the frame.",
         "audio_cues": "Quiet room tone.",
         "prosodic_cues": "Pause.",
         "implied_emotional_moment": "Held breath."},
    ])

    with patch("feeling_engine.preprocess.multimodal._probe_duration", return_value=5.0), \
         patch("google.genai.Client", return_value=client):
        segs = preprocess_video(fixture_video, chunk_sec=5.0, verbose=False)

    assert len(segs) == 1
    assert segs[0].start_sec == 0.0
    assert segs[0].end_sec == 5.0
    assert "A face fills" in segs[0].visual_cues
    assert "VISUAL:" in segs[0].synthesized_description
    assert "AUDIO:" in segs[0].synthesized_description


def test_preprocess_merges_transcript_speech_into_synthesized(fixture_video, monkeypatch):
    monkeypatch.setenv("GOOGLE_AI_API_KEY", "test-key")
    client = _mock_client([
        {"start_sec": 0.0, "end_sec": 5.0,
         "visual_cues": "x", "audio_cues": "x",
         "prosodic_cues": "x", "implied_emotional_moment": "x"},
    ])

    transcript = {"words": [
        {"word": "hello", "start": 1.0, "end": 1.5},
        {"word": "world", "start": 2.0, "end": 2.5},
    ]}

    with patch("feeling_engine.preprocess.multimodal._probe_duration", return_value=5.0), \
         patch("google.genai.Client", return_value=client):
        segs = preprocess_video(fixture_video, chunk_sec=5.0,
                                transcript=transcript, verbose=False)

    assert segs[0].transcript_snippet == "hello world"
    assert "SPEECH:" in segs[0].synthesized_description
    assert "hello world" in segs[0].synthesized_description


def test_preprocess_raises_on_missing_api_key(fixture_video, monkeypatch):
    monkeypatch.delenv("GOOGLE_AI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_AI_API_KEY"):
        preprocess_video(fixture_video, verbose=False)


def test_preprocess_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        preprocess_video(tmp_path / "no.mp4")
