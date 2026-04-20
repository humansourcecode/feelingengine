"""Intelligent voice selection for TTS.

Given content text and available voices (from any TTSAdapter that implements
list_voices()), pick the voice whose characteristics best match the content's
tone, speaker, and intended affect.

Uses an LLM (Claude by default) when available; falls back to keyword
heuristics otherwise. Designed to be optional — callers who prefer a specific
voice can skip this entirely.

Usage:

    from feeling_engine.voice_picker import pick_voice
    from feeling_engine.adapters.tts.elevenlabs import ElevenLabsAdapter

    tts = ElevenLabsAdapter()
    choice = pick_voice(text="the content...", tts_adapter=tts)

    tts.synthesize(text, output_path, voice_id=choice.voice_id)
    # choice.rationale explains why this voice was picked
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional, List

from feeling_engine.adapters.tts.base import TTSAdapter, VoiceInfo


@dataclass
class VoiceChoice:
    """Result of voice selection."""
    voice_id: str
    name: str
    rationale: str
    method: str              # "llm" | "heuristic" | "fallback"
    runner_up_voice_ids: List[str] = None


# ─── Heuristic fallback ────────────────────────────────────────

_CONTENT_CUES = {
    # (keyword patterns) -> preferred voice tags
    "romantic_letter":      ([r"\bmy (dear|love|darling)\b", r"\bbeloved\b", r"\bbetrothed\b", r"\bmy wife\b"], {"gender": "male", "style": "warm"}),
    "intimate_first_person":([r"\bI (remember|confess|admit)\b", r"\bmy (heart|soul|dear)\b"], {"style": "warm", "style_alt": "soft"}),
    "poetic_reflective":    ([r"\bthou\b", r"\bo,?\s", r"\bhark\b", r"\bere\b", r"\bnay\b"], {"style": "measured", "age": "middle"}),
    "oratory_defiant":      ([r"\bgive me\b", r"\bliberty\b", r"\bfreedom\b", r"\btyranny\b", r"\bshall not\b"], {"style": "strong", "age": "middle"}),
    "narrative_neutral":    ([r"\bonce upon\b", r"\bmeanwhile\b", r"\bthere (once|was)\b"], {"style": "neutral"}),
    "instructional":        ([r"\bstep \d+\b", r"\bfirst,\b", r"\bnext,\b", r"\btutorial\b"], {"style": "clear", "style_alt": "calm"}),
    "dramatic":             ([r"\bdeath\b", r"\bdoom\b", r"\bfate\b", r"\banguish\b"], {"style": "dramatic"}),
    "contemplative":        ([r"\bperhaps\b", r"\bI wonder\b", r"\bconsider\b", r"\bone could\b"], {"style": "thoughtful", "style_alt": "measured"}),
}


def _content_cue_tags(text: str) -> dict:
    """Return union of voice-preference tags triggered by content cues."""
    text_lower = text.lower()
    tags = {}
    for cue_name, (patterns, voice_tags) in _CONTENT_CUES.items():
        if any(re.search(p, text_lower) for p in patterns):
            for k, v in voice_tags.items():
                tags.setdefault(k, set()).add(v)
    return tags


def _score_voice_heuristic(voice: VoiceInfo, preferred_tags: dict) -> int:
    """Crude matching: count how many preferred tag values appear in voice labels/description."""
    score = 0
    haystack = " ".join([
        voice.name or "",
        voice.description or "",
        json.dumps(voice.labels or {}),
    ]).lower()
    for tag_key, values in preferred_tags.items():
        for val in values:
            if val.lower() in haystack:
                score += 1
    return score


def _pick_heuristic(text: str, voices: List[VoiceInfo]) -> Optional[VoiceChoice]:
    """Keyword-matching fallback when no LLM is available."""
    if not voices:
        return None
    tags = _content_cue_tags(text)
    if not tags:
        return None
    scored = [(_score_voice_heuristic(v, tags), v) for v in voices]
    scored.sort(key=lambda x: -x[0])
    best_score, best_voice = scored[0]
    if best_score == 0:
        return None
    return VoiceChoice(
        voice_id=best_voice.voice_id,
        name=best_voice.name,
        rationale=f"Heuristic match on content cues: {dict((k, sorted(v)) for k, v in tags.items())}. Score {best_score}.",
        method="heuristic",
        runner_up_voice_ids=[v.voice_id for _, v in scored[1:4]],
    )


# ─── LLM-based picker ─────────────────────────────────────────

def _pick_llm(text: str, voices: List[VoiceInfo], llm_client=None) -> Optional[VoiceChoice]:
    """Use Claude to analyze content tone + match to available voices."""
    if not voices:
        return None
    if llm_client is None:
        try:
            import anthropic
            llm_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        except Exception:
            return None
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None

    # Sample first ~2000 chars of content (enough to characterize tone)
    sample = text[:2000]

    voices_json = [
        {
            "voice_id": v.voice_id,
            "name": v.name,
            "category": v.category,
            "description": v.description,
            "labels": v.labels or {},
        }
        for v in voices[:30]  # cap to avoid huge prompts
    ]

    prompt = f"""You are picking the best voice for a text-to-speech rendering.

The voice should match the IMPLIED speaker, not just sound "professional."
Mismatches reduce downstream brain-response prediction (TRIBE v2) quality.

## Priorities (in order)

1. **Speaker identity.** If the content is by a known historical figure
   (e.g., Patrick Henry = American, Revolutionary era; John Keats = British,
   early 19th c.; Sullivan Ballou = American, Civil War), match the voice's
   accent + apparent era to that identity. A British narrator reading
   Patrick Henry is a mismatch.
2. **Gender.** If the speaker's gender is implied (first-person letter
   to a wife = male; diary of a woman = female), match it.
3. **Age / register.** Match apparent age and formality.
4. **Emotional tone.** Warm for intimate, firm for oratorical, measured
   for contemplative, etc.

A "warm narrator" voice is NOT a universal default — pick the voice whose
*accent, era, and gender* fit the implied speaker first, then optimize
tone within that constraint.

Available voices (JSON):
{json.dumps(voices_json, indent=2)}

Content sample (first portion of full text):
\"\"\"
{sample}
\"\"\"

Respond in JSON only, no prose:
{{
  "implied_speaker": "<brief description of who is speaking: gender, nationality, era>",
  "voice_id": "<chosen voice_id>",
  "name": "<voice name>",
  "rationale": "<one sentence: why this voice's accent/era/gender/tone matches the implied speaker>",
  "runner_up_voice_ids": ["<second choice>", "<third choice>"]
}}"""

    try:
        msg = llm_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = msg.content[0].text
        # extract JSON block
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            return None
        parsed = json.loads(match.group(0))
        return VoiceChoice(
            voice_id=parsed["voice_id"],
            name=parsed.get("name", ""),
            rationale=parsed.get("rationale", ""),
            method="llm",
            runner_up_voice_ids=parsed.get("runner_up_voice_ids", []),
        )
    except Exception as e:
        return None


# ─── Public API ───────────────────────────────────────────────

def pick_voice(
    text: str,
    tts_adapter: Optional[TTSAdapter] = None,
    voices: Optional[List[VoiceInfo]] = None,
    llm_client=None,
    enabled: bool = True,
) -> Optional[VoiceChoice]:
    """Intelligently select a voice for the given text.

    Args:
        text: the content to be synthesized
        tts_adapter: any TTSAdapter; its list_voices() will be called
        voices: list of VoiceInfo (alternative to tts_adapter)
        llm_client: optional Anthropic client; auto-created from env if None
        enabled: if False, returns None (caller uses default voice)

    Returns:
        VoiceChoice with voice_id + rationale, or None if disabled / no match.

    Strategy:
        1. If LLM available → use Claude to analyze content + match voices
        2. Fallback → keyword heuristics on content cues
        3. If neither produces a match → return None (caller's default applies)
    """
    if not enabled:
        return None

    if voices is None:
        if tts_adapter is None:
            return None
        try:
            voices = tts_adapter.list_voices()
        except Exception:
            return None

    if not voices:
        return None

    # Try LLM first
    choice = _pick_llm(text, voices, llm_client=llm_client)
    if choice:
        return choice

    # Fall back to heuristics
    choice = _pick_heuristic(text, voices)
    if choice:
        return choice

    return None
