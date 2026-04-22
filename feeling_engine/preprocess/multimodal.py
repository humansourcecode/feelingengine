"""Multimodal pre-processor — enrich video/audio content for Tier 2/3 detectors.

Uses Gemini 2.5 Flash with native video upload (single upload → single query
for the whole video → timestamped structured description). No frame chunking.

Gemini's native video handling captures motion, cuts, rhythm, and audio cues
that static-frame analysis would miss — exactly the signals Tier 2/3 detectors
need to fire on visual/music-dominant content.

Commercial-safe: uses Gemini API (commercial), our vocabulary, our detectors.
No TRIBE dependency.

Usage:
    from feeling_engine.preprocess.multimodal import preprocess_video

    segments = preprocess_video("video.mp4", chunk_sec=5.0)
    for seg in segments:
        print(seg.start_sec, seg.synthesized_description)

    # Feed into Tier 2/3 detectors
    from feeling_engine.mechanisms.api import detect_mechanisms
    enriched_text = "\n".join(s.synthesized_description for s in segments)
    labels = detect_mechanisms(text=enriched_text, mode="text-only")
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_CHUNK_SEC = 5.0


@dataclass
class EnrichedSegment:
    """One chunk of content enriched with multimodal description."""
    start_sec: float
    end_sec: float
    transcript_snippet: str                # speech text (if any) in this chunk
    visual_cues: str
    audio_cues: str
    prosodic_cues: str
    implied_emotional_moment: str
    synthesized_description: str            # concat for Tier 2/3 consumption

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Utilities ────────────────────────────────────────────────────

def _probe_duration(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True, timeout=30,
    )
    return float(result.stdout.strip())


def _slice_transcript(transcript: Optional[dict], start_sec: float,
                      end_sec: float) -> str:
    """Return words from transcript that fall within the given window."""
    if not transcript:
        return ""
    words = transcript.get("words")
    if not words:
        return ""
    selected = [
        w["word"] for w in words
        if w.get("start") is not None and w.get("end") is not None
        and w["end"] >= start_sec and w["start"] <= end_sec
    ]
    return " ".join(selected).strip()


# ─── Gemini call ───────────────────────────────────────────────────

_PROMPT = """You are analyzing a {duration}-second video for emotional-mechanism detection.

Divide the video into {n_chunks} sequential {chunk_sec}-second chunks.
For EACH chunk, describe the moment in four dimensions. Be specific, concrete, concise.
Stay grounded in what's actually visible or audible — no speculation.

Return valid JSON: an array of exactly {n_chunks} objects (one per chunk, in time order).
Each object MUST have exactly these string keys:

- "start_sec": numeric start of this chunk
- "end_sec": numeric end of this chunk
- "visual_cues": what is visible — faces, composition, cuts, lighting, gestures, body language. 1-2 sentences.
- "audio_cues": what is audible — silence, music presence, sound design, background audio. 1 sentence.
- "prosodic_cues": how speech is delivered — pause, emphasis, lowered register, breathiness. 1 sentence. Use "no speech" if none.
- "implied_emotional_moment": the emotional shape of this moment in plain language (no jargon). 1-2 sentences.

Do not include markdown formatting, code fences, or commentary outside the JSON array.
"""


def _upload_and_wait(client, video_path: Path, verbose: bool = True) -> object:
    """Upload video file to Gemini; poll until state is ACTIVE."""
    from feeling_engine._gemini_retry import gemini_with_retry
    if verbose:
        print(f"  Uploading {video_path.name} to Gemini...", end="", flush=True)
    t0 = time.time()
    video_file = gemini_with_retry(client.files.upload, file=str(video_path),
                                   verbose=verbose)
    while video_file.state.name == "PROCESSING":
        time.sleep(3)
        video_file = gemini_with_retry(client.files.get, name=video_file.name,
                                       verbose=verbose)
    if video_file.state.name != "ACTIVE":
        raise RuntimeError(f"Gemini video processing failed: {video_file.state.name}")
    if verbose:
        print(f" ACTIVE in {time.time()-t0:.0f}s", flush=True)
    return video_file


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # Drop opening fence (and any language tag)
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", text)
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _call_gemini_for_chunks(client, video_file, duration_sec: float,
                            chunk_sec: float, verbose: bool = True) -> list[dict]:
    """Query Gemini with the uploaded video; return parsed list of chunk dicts."""
    from google.genai import types

    n_chunks = max(1, int((duration_sec + chunk_sec - 0.001) // chunk_sec))
    prompt = _PROMPT.format(
        duration=f"{duration_sec:.1f}",
        n_chunks=n_chunks,
        chunk_sec=chunk_sec,
    )

    if verbose:
        print(f"  Querying Gemini for {n_chunks} chunks...", end="", flush=True)
    t0 = time.time()

    # thinking_budget: Flash supports 0 to skip; Pro requires thinking mode.
    # Use a modest budget that satisfies both.
    thinking_budget = 0 if "flash" in GEMINI_MODEL.lower() else 2048
    from feeling_engine._gemini_retry import gemini_with_retry
    response = gemini_with_retry(
        client.models.generate_content,
        model=GEMINI_MODEL,
        contents=[video_file, prompt],
        config=types.GenerateContentConfig(
            max_output_tokens=min(65000, 400 * n_chunks + 1000),
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        ),
        verbose=verbose,
    )

    text = ""
    for part in response.candidates[0].content.parts:
        if hasattr(part, "text") and part.text:
            text += part.text
    text = _strip_code_fences(text)

    if verbose:
        usage = response.usage_metadata
        in_tok = usage.prompt_token_count or 0
        out_tok = usage.candidates_token_count or 0
        cost = (in_tok * 0.15 + out_tok * 0.60) / 1_000_000
        print(f" done in {time.time()-t0:.0f}s, "
              f"{in_tok:,} in / {out_tok:,} out, ${cost:.4f}", flush=True)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini returned non-JSON: {text[:500]}") from e

    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")
    return parsed


# ─── Public API ───────────────────────────────────────────────────

def preprocess_video(
    video_path,
    chunk_sec: float = DEFAULT_CHUNK_SEC,
    transcript: Optional[dict] = None,
    verbose: bool = True,
) -> list[EnrichedSegment]:
    """Enrich a video with per-chunk multimodal descriptions via Gemini.

    Args:
        video_path: path to video/audio file (mp4, mov, mp3, wav, etc.)
        chunk_sec: desired chunk window in seconds (default 5.0)
        transcript: optional word-level transcript dict with "words" list;
            speech snippets get merged into the per-chunk output.
        verbose: print progress.

    Returns:
        time-ordered list of EnrichedSegment.
    """
    from google import genai

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    api_key = os.environ.get("GOOGLE_AI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_AI_API_KEY not set in environment")

    duration = _probe_duration(video_path)
    if verbose:
        print(f"  duration={duration:.1f}s, chunk_sec={chunk_sec}", flush=True)

    client = genai.Client(api_key=api_key)
    video_file = _upload_and_wait(client, video_path, verbose=verbose)

    try:
        chunks = _call_gemini_for_chunks(
            client, video_file, duration, chunk_sec, verbose=verbose,
        )
    finally:
        # Always try to delete the uploaded file, even on error
        try:
            client.files.delete(name=video_file.name)
        except Exception:
            pass

    segments: list[EnrichedSegment] = []
    for c in chunks:
        start_sec = float(c.get("start_sec", 0.0))
        end_sec = float(c.get("end_sec", start_sec + chunk_sec))
        snippet = _slice_transcript(transcript, start_sec, end_sec)

        visual = str(c.get("visual_cues", "")).strip()
        audio = str(c.get("audio_cues", "")).strip()
        prosody = str(c.get("prosodic_cues", "")).strip()
        moment = str(c.get("implied_emotional_moment", "")).strip()

        synthesized = (
            f"[{start_sec:.1f}-{end_sec:.1f}s] "
            f"VISUAL: {visual} "
            f"AUDIO: {audio} "
            f"PROSODY: {prosody} "
            f"MOMENT: {moment}"
            + (f" SPEECH: \"{snippet}\"" if snippet else "")
        )

        segments.append(EnrichedSegment(
            start_sec=start_sec,
            end_sec=end_sec,
            transcript_snippet=snippet,
            visual_cues=visual,
            audio_cues=audio,
            prosodic_cues=prosody,
            implied_emotional_moment=moment,
            synthesized_description=synthesized,
        ))

    segments.sort(key=lambda s: s.start_sec)
    if verbose:
        print(f"  {len(segments)} segments enriched", flush=True)
    return segments


def enriched_transcript(segments: list[EnrichedSegment]) -> str:
    """Concatenate segment descriptions for feeding to Tier 2/3 detectors."""
    return "\n".join(s.synthesized_description for s in segments)
