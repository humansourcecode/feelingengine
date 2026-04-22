"""Gemini video API smoke tests — opt-in, low-cost.

Runs a 5-second synthetic video (darkblue solid + 440Hz tone, ~27KB) through
both the multimodal pre-processor and the LLM detector, verifies the response
shape. The video content is intentionally bare so results will be minimal —
this validates API wiring + parsing, not detection quality.

Activation: pytest tests/e2e/test_gemini_video_smoke.py --run-gemini

Cost per run: ~$0.01-0.02 (synthetic clip, two API calls).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_clip.mp4"


@pytest.mark.gemini
def test_multimodal_preprocess_smoke():
    if not FIXTURE.exists():
        pytest.skip(f"fixture missing: {FIXTURE}")
    if not os.environ.get("GOOGLE_AI_API_KEY"):
        pytest.skip("GOOGLE_AI_API_KEY not set")

    from feeling_engine.preprocess.multimodal import preprocess_video, enriched_transcript

    segs = preprocess_video(FIXTURE, chunk_sec=5.0, verbose=False)

    assert isinstance(segs, list)
    assert len(segs) >= 1
    s = segs[0]
    assert s.start_sec == 0.0
    assert 0.0 < s.end_sec <= 5.5
    assert isinstance(s.visual_cues, str) and len(s.visual_cues) > 0
    assert isinstance(s.synthesized_description, str)
    assert "VISUAL:" in s.synthesized_description

    text = enriched_transcript(segs)
    assert len(text) > 0


@pytest.mark.gemini
def test_llm_detector_smoke():
    if not FIXTURE.exists():
        pytest.skip(f"fixture missing: {FIXTURE}")
    if not os.environ.get("GOOGLE_AI_API_KEY"):
        pytest.skip("GOOGLE_AI_API_KEY not set")

    from feeling_engine.mechanisms.llm_detector import detect_from_video
    from feeling_engine.mechanisms.vocabulary import LabelApplication, MECHANISM_LABELS

    valid_labels = {s.name for s in MECHANISM_LABELS}
    apps = detect_from_video(FIXTURE, verbose=False)

    assert isinstance(apps, list)
    # Synthetic video may fire zero or many labels — either is acceptable here.
    for a in apps:
        assert isinstance(a, LabelApplication)
        assert a.label in valid_labels
        assert 0.0 <= a.intensity <= 1.0
        assert 0.0 <= a.confidence <= 1.0
        assert a.signals.get("detector") == "llm-gemini"
