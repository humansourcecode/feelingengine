"""Feeling Engine — brain-to-emotion translation for content analysis.

The first open-source pipeline that translates brain-response predictions
into human-readable emotional arcs with cross-cultural precedent matching.

Architecture:
    Content (text/audio) → TTS (if text) → TRIBE v2 (brain prediction)
    → Change Detection → Neuroscience Mapping → LLM Synthesis
    → Emotional Arc → Fire (cross-domain matching) → Precedents

All infrastructure is pluggable via adapters. You provide your own API keys.
"""

__version__ = "0.1.0"

# Public surface
from feeling_engine.voice_picker import pick_voice, VoiceChoice  # noqa: E402

__all__ = ["pick_voice", "VoiceChoice"]
