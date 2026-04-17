"""Base interface for Text-to-Speech adapters.

Any TTS provider that converts text to natural-sounding audio can be
plugged in here. The adapter's job is: text in, audio file path out.

TRIBE v2 was trained on REAL human speech. Higher-quality TTS (natural
prosody, emotion, pauses) produces more accurate brain predictions than
robotic TTS. ElevenLabs is recommended; gTTS is NOT recommended.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TTSResult:
    """Result of a text-to-speech synthesis."""
    audio_path: Path
    duration_seconds: float
    voice_id: str | None = None
    provider: str = "unknown"


@dataclass
class VoiceInfo:
    """Provider-agnostic voice metadata for user selection."""
    voice_id: str
    name: str
    category: str = "unknown"   # e.g., premade, cloned, professional
    description: str = ""       # free-text descriptor (gender, age, style)
    labels: dict | None = None  # provider-specific tags (gender, accent, etc.)


class TTSAdapter(ABC):
    """Interface for TTS providers. Implement for each provider."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice_id: str | None = None,
    ) -> TTSResult:
        """Convert text to speech audio.

        Args:
            text: the content to synthesize
            output_path: where to save the audio file
            voice_id: optional provider-specific voice identifier

        Returns:
            TTSResult with audio file path and metadata
        """

    @abstractmethod
    def list_voices(self) -> list[VoiceInfo]:
        """Return voices available in the user's account for this provider.

        Lets users pick a voice that matches their content (gender, age,
        register, style). Mismatched voices produce lower-quality TRIBE
        predictions — choose the closest match to the intended speaker.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name (e.g., 'ElevenLabs')."""
