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

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name (e.g., 'ElevenLabs')."""
