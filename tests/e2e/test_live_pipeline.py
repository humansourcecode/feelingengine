"""Full live-API end-to-end tests.

These hit real external services and cost real money:
  - ElevenLabs (TTS)
  - Modal (TRIBE GPU inference) — ~$0.08 per run
  - Anthropic Claude (Layer 4) — ~$0.01 per call

Skipped by default. Run with:
    pytest --run-e2e

Requires a .env file (or exported environment) with:
    ELEVENLABS_API_KEY
    MODAL_TOKEN_ID / MODAL_TOKEN_SECRET
    HUGGINGFACE_ACCESS_TOKEN
    ANTHROPIC_API_KEY

The TRIBE function must be deployed on Modal (`modal deploy tribe_modal.py`).
"""
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

FIXTURES = Path(__file__).parent / "fixtures"
OUTPUT = Path(__file__).parent / "output"

# Brug — a reassuring, deep, middle-aged American male voice, closer match to
# Steve Jobs than the default Rachel narrator.
JOBS_ANALOG_VOICE_ID = "rMYrXwoflZJOTJaBiWpa"


def _require_env(*keys):
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        pytest.skip(f"Missing env var(s): {', '.join(missing)}")


def test_elevenlabs_synthesize_short(tmp_path):
    """ElevenLabs → MP3 file with non-zero duration."""
    _require_env("ELEVENLABS_API_KEY")
    from feeling_engine.adapters.tts.elevenlabs import ElevenLabsAdapter

    adapter = ElevenLabsAdapter(voice_id=JOBS_ANALOG_VOICE_ID)
    result = adapter.synthesize(
        text="No one wants to die.",
        output_path=tmp_path / "short.mp3",
    )
    assert result.audio_path.exists()
    assert result.audio_path.stat().st_size > 0
    assert result.duration_seconds > 0
    assert result.provider == "elevenlabs"


def test_list_voices_returns_library():
    _require_env("ELEVENLABS_API_KEY")
    from feeling_engine.adapters.tts.elevenlabs import ElevenLabsAdapter

    adapter = ElevenLabsAdapter()
    voices = adapter.list_voices()
    assert voices, "Expected at least one voice in account"
    first = voices[0]
    assert first.voice_id
    assert first.name


def test_full_text_to_arc_pipeline(tmp_path):
    """End-to-end: text → TTS → Modal TRIBE → Translator → Fire."""
    _require_env(
        "ELEVENLABS_API_KEY",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "HUGGINGFACE_ACCESS_TOKEN",
    )
    from feeling_engine.adapters.tts.elevenlabs import ElevenLabsAdapter
    from feeling_engine.adapters.compute.modal_tribe import ModalTRIBEAdapter
    from feeling_engine.adapters.brain_model.tribev2 import TRIBEv2Adapter
    from feeling_engine.pipeline import FeelingPipeline
    from feeling_engine.fire.matcher import FireMatcher

    text = (FIXTURES / "jobs_short.txt").read_text()
    mp3_path = tmp_path / "jobs_short.mp3"

    tts = ElevenLabsAdapter(voice_id=JOBS_ANALOG_VOICE_ID)
    tts_result = tts.synthesize(text, mp3_path)
    assert tts_result.audio_path.exists()

    compute = ModalTRIBEAdapter()
    prediction = compute.predict(mp3_path.read_bytes(), mp3_path.name)
    assert prediction.predictions.size > 0

    pipeline = FeelingPipeline(brain_adapter=TRIBEv2Adapter())
    arc = pipeline.analyze_predictions(prediction.predictions, change_points_only=True)
    assert arc.n_timesteps > 0

    matcher = FireMatcher()
    matches = matcher.match_arc(arc, top_k=3)
    # Short 4s clip may not produce strong matches, but matcher must run cleanly
    assert isinstance(matches, list)
