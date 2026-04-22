"""Unit tests for llm_detector — parsing + validation logic, no API calls."""
from __future__ import annotations

from pathlib import Path

import pytest

from feeling_engine.mechanisms.llm_detector import (
    _build_vocabulary_section,
    _merge_frame_labels_to_applications,
    _strip_code_fences,
    detect_from_video,
)
from feeling_engine.mechanisms.vocabulary import MECHANISM_LABELS


def _frames(n: int, interval: float = 1.0) -> list:
    """Build n fake (ts, path) frame tuples at `interval`-second spacing."""
    return [(round(i * interval, 3), Path(f"/tmp/frame_{i:04d}.jpg")) for i in range(n)]


def _classification(frame_index: int, label: str, intensity: float = 0.8,
                    confidence: float = 0.9, evidence: str = "") -> dict:
    return {
        "frame_index": frame_index,
        "labels": [{
            "label": label,
            "intensity": intensity,
            "confidence": confidence,
            "visual_cue": evidence,
            "audio_cue": "",
            "prosodic_cue": "",
            "contrast": "",
        }],
    }


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


# ─── Tests for _merge_frame_labels_to_applications ─────────────────
# These cover the core parse/filter/clamp/sort logic without mocking
# the whole detect_from_video pipeline. They supply per-frame
# classifications in the shape that `_classify_frames_chunk` produces.


def test_merge_parses_valid_classifications():
    frames = _frames(6, interval=0.5)  # ts = 0.0 .. 2.5
    classifications = [
        _classification(2, "body-turn", 0.8, 0.9, "sudden pause"),
        _classification(3, "body-turn", 0.8, 0.9, "sudden pause"),
        _classification(4, "body-turn", 0.8, 0.9, "sudden pause"),
        _classification(5, "body-turn", 0.8, 0.9, "sudden pause"),
    ]
    apps = _merge_frame_labels_to_applications(classifications, frames, 0.5, "gemini-test")
    assert len(apps) == 1
    assert apps[0].label == "body-turn"
    assert apps[0].tier == 1
    assert apps[0].start_sec == 1.0
    assert apps[0].intensity == 0.8
    assert apps[0].signals["detector"] == "llm-gemini-v2"
    assert apps[0].signals["visual_cue"] == "sudden pause"


def test_merge_filters_unknown_labels():
    frames = _frames(4, interval=1.0)
    classifications = [
        _classification(1, "body-turn"),
        _classification(2, "invented-label"),
    ]
    apps = _merge_frame_labels_to_applications(classifications, frames, 1.0, "gemini-test")
    assert len(apps) == 1
    assert apps[0].label == "body-turn"


def test_merge_clamps_intensity_and_confidence():
    frames = _frames(2, interval=1.0)
    classifications = [
        _classification(0, "body-turn", intensity=1.5, confidence=-0.2),
    ]
    apps = _merge_frame_labels_to_applications(classifications, frames, 1.0, "gemini-test")
    assert len(apps) == 1
    assert apps[0].intensity == 1.0
    assert apps[0].confidence == 0.0


def test_merge_sorts_by_start_sec():
    frames = _frames(6, interval=1.0)
    classifications = [
        _classification(4, "body-turn"),
        _classification(1, "recognition"),
    ]
    apps = _merge_frame_labels_to_applications(classifications, frames, 1.0, "gemini-test")
    assert [a.start_sec for a in apps] == [1.0, 4.0]


# ─── Tests for detect_from_video pre-flight errors ─────────────────


def test_detect_raises_on_missing_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_AI_API_KEY", raising=False)
    p = tmp_path / "sample.mp4"
    p.write_bytes(b"fake")
    with pytest.raises(RuntimeError, match="GOOGLE_AI_API_KEY"):
        detect_from_video(p, verbose=False)


def test_detect_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        detect_from_video(tmp_path / "does-not-exist.mp4")
