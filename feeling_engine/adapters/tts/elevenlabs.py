"""ElevenLabs TTS adapter for Feeling Engine.

Converts text to high-quality speech audio for TRIBE brain prediction.
Only used when analyzing text content that has no existing audio recording.
If real audio exists (speeches, songs, podcasts), use that directly instead.

Uses ElevenLabs' most expressive model (multilingual_v2) by default
because TRIBE was trained on real human speech — more natural TTS
produces more accurate brain predictions.

Requires: ELEVENLABS_API_KEY in environment.
Cost: included in ElevenLabs monthly subscription.

Usage:
    adapter = ElevenLabsAdapter()
    result = adapter.synthesize(
        text="No one wants to die...",
        output_path=Path("output.mp3"),
    )
"""
from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from feeling_engine.adapters.tts.base import TTSAdapter, TTSResult


# Voice settings optimized for TRIBE accuracy.
# Low stability + high style = most human-like delivery = best for TRIBE.
# High stability + low style = robotic = poor predictions (same problem as gTTS).
DEFAULT_SETTINGS = {
    "stability": 0.5,           # natural variation, not robotic
    "similarity_boost": 0.75,   # maintain voice identity
    "style": 0.7,               # emotionally expressive
    "use_speaker_boost": True,
}

# ElevenLabs model selection.
# multilingual_v2 is the most expressive — best for TRIBE accuracy.
# turbo_v2 is faster but flatter — worse for TRIBE.
DEFAULT_MODEL = "eleven_multilingual_v2"


@dataclass
class ElevenLabsTimingWord:
    """Word-level timing from ElevenLabs alignment."""
    word: str
    start_s: float
    end_s: float


class ElevenLabsAdapter(TTSAdapter):
    """ElevenLabs TTS adapter.

    Synthesizes text to speech and optionally returns word-level timing
    metadata for aligning brain predictions to specific words.

    Args:
        voice_id: ElevenLabs voice ID. If None, uses a default narrator.
        model: ElevenLabs model ID. Defaults to multilingual_v2 (most expressive).
        settings: voice settings dict. Defaults to TRIBE-optimized settings.
    """

    def __init__(
        self,
        voice_id: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        settings: Optional[dict] = None,
    ):
        self._api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "ELEVENLABS_API_KEY not set in environment. "
                "Add it to your .env file."
            )

        self._voice_id = voice_id or self._get_default_voice_id()
        self._model = model
        self._settings = settings or DEFAULT_SETTINGS.copy()

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice_id: Optional[str] = None,
    ) -> TTSResult:
        """Convert text to speech audio.

        Args:
            text: content to synthesize
            output_path: where to save the audio file (MP3)
            voice_id: override voice for this call (optional)

        Returns:
            TTSResult with audio path, duration, and metadata
        """
        try:
            from elevenlabs import ElevenLabs
        except ImportError:
            raise ImportError(
                "elevenlabs package required. Install: pip install elevenlabs"
            )

        client = ElevenLabs(api_key=self._api_key)
        vid = voice_id or self._voice_id

        # Generate audio
        audio_generator = client.text_to_speech.convert(
            voice_id=vid,
            text=text,
            model_id=self._model,
            voice_settings={
                "stability": self._settings.get("stability", 0.5),
                "similarity_boost": self._settings.get("similarity_boost", 0.75),
                "style": self._settings.get("style", 0.7),
                "use_speaker_boost": self._settings.get("use_speaker_boost", True),
            },
        )

        # Collect audio bytes from generator
        audio_bytes = b""
        for chunk in audio_generator:
            audio_bytes += chunk

        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(audio_bytes)

        # Get duration via probe
        duration = self._probe_duration(output_path)

        return TTSResult(
            audio_path=output_path,
            duration_seconds=duration,
            voice_id=vid,
            provider="elevenlabs",
        )

    def synthesize_with_timing(
        self,
        text: str,
        output_path: Path,
        voice_id: Optional[str] = None,
    ) -> tuple[TTSResult, list[ElevenLabsTimingWord]]:
        """Synthesize and also return word-level timing.

        Word-level timing is essential for the Translator — it maps
        brain prediction timesteps to specific words in the content.
        Without timing, we know WHAT the brain does but not WHERE
        in the text it happens.

        Returns:
            (TTSResult, list of word timings)
        """
        try:
            from elevenlabs import ElevenLabs
        except ImportError:
            raise ImportError(
                "elevenlabs package required. Install: pip install elevenlabs"
            )

        client = ElevenLabs(api_key=self._api_key)
        vid = voice_id or self._voice_id

        # Use the with-timestamps endpoint
        response = client.text_to_speech.convert_with_timestamps(
            voice_id=vid,
            text=text,
            model_id=self._model,
            voice_settings={
                "stability": self._settings.get("stability", 0.5),
                "similarity_boost": self._settings.get("similarity_boost", 0.75),
                "style": self._settings.get("style", 0.7),
                "use_speaker_boost": self._settings.get("use_speaker_boost", True),
            },
        )

        # Collect audio bytes and timing data
        audio_bytes = b""
        timing_words = []

        for chunk in response:
            if chunk.get("audio_base64"):
                import base64
                audio_bytes += base64.b64decode(chunk["audio_base64"])

            if chunk.get("alignment"):
                alignment = chunk["alignment"]
                chars = alignment.get("characters", [])
                starts = alignment.get("character_start_times_seconds", [])
                ends = alignment.get("character_end_times_seconds", [])

                # Reconstruct words from character-level timing
                current_word = ""
                word_start = None
                for char, start, end in zip(chars, starts, ends):
                    if char == " " and current_word:
                        timing_words.append(ElevenLabsTimingWord(
                            word=current_word,
                            start_s=word_start,
                            end_s=end,
                        ))
                        current_word = ""
                        word_start = None
                    else:
                        if word_start is None:
                            word_start = start
                        current_word += char

                if current_word and word_start is not None:
                    timing_words.append(ElevenLabsTimingWord(
                        word=current_word,
                        start_s=word_start,
                        end_s=ends[-1] if ends else word_start,
                    ))

        # Save audio
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(audio_bytes)

        duration = self._probe_duration(output_path)

        result = TTSResult(
            audio_path=output_path,
            duration_seconds=duration,
            voice_id=vid,
            provider="elevenlabs",
        )

        return result, timing_words

    def list_voices(self) -> list[dict]:
        """List available voices in the user's ElevenLabs account."""
        try:
            from elevenlabs import ElevenLabs
        except ImportError:
            raise ImportError("elevenlabs package required")

        client = ElevenLabs(api_key=self._api_key)
        voices = client.voices.get_all()

        return [
            {
                "voice_id": v.voice_id,
                "name": v.name,
                "category": getattr(v, "category", "unknown"),
                "description": getattr(v, "description", ""),
            }
            for v in voices.voices
        ]

    @property
    def provider_name(self) -> str:
        return "ElevenLabs"

    def _get_default_voice_id(self) -> str:
        """Get a default voice ID.

        Uses ELEVENLABS_VOICE_ID from env if set, otherwise falls back
        to ElevenLabs' 'Rachel' voice (neutral, expressive narrator).
        """
        env_voice = os.getenv("ELEVENLABS_VOICE_ID")
        if env_voice:
            return env_voice

        # Rachel — neutral female narrator, good default for analysis
        return "21m00Tcm4TlvDq8ikWAM"

    @staticmethod
    def _probe_duration(path: Path) -> float:
        """Get audio duration in seconds via ffprobe."""
        import subprocess
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format=duration", "-of",
                 "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True, timeout=10,
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0
