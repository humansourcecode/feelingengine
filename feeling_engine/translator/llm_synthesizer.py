"""Layer 4 — LLM Synthesis (content-aware + context-aware label refinement).

Takes Layer 3's scored emotion labels + brain data + the actual content
being analyzed + optional viewer context, and asks Claude/Gemini to
refine the labels with content-specific + context-specific precision.

Without Layer 4: "high interoception + language suppression → awe"
With Layer 4:    "high interoception at the moment Jobs says 'no one
                  wants to die' → mortality_awareness (not generic awe)"

Layer 4 is the difference between correct-family and correct-specific.
It's also where viewer context modifies the interpretation:
same brain data + different context → different emotional labels.

Requires: ANTHROPIC_API_KEY or GOOGLE_AI_API_KEY in environment.
Cost: ~$0.01-0.03 per timestep refined.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from feeling_engine.translator.brain_to_emotion import EmotionLabel, TimestepEmotion
from feeling_engine.translator.change_detector import ChangePoint

VOCABULARY_PATH = Path(__file__).parent / "vocabulary.yaml"


@dataclass
class ContextProfile:
    """Viewer context that modifies emotional interpretation.

    Same brain scan + different context → different emotional arc.
    The scan is the measurement; context shapes the interpretation.
    """
    name: str  # e.g., "posthumous_knowledge"
    description: str  # e.g., "Viewer knows Jobs died of cancer in 2011"
    life_event: Optional[str] = None
    knowledge: Optional[str] = None
    cultural: Optional[str] = None
    identity: Optional[str] = None
    bias: Optional[str] = None
    physical_state: Optional[str] = None
    social_setting: Optional[str] = None

    def to_prompt_text(self) -> str:
        """Render as text for inclusion in LLM prompt."""
        parts = [f"Context: {self.name}"]
        parts.append(f"Description: {self.description}")
        for field_name in ["life_event", "knowledge", "cultural", "identity",
                           "bias", "physical_state", "social_setting"]:
            value = getattr(self, field_name)
            if value:
                parts.append(f"  {field_name}: {value}")
        return "\n".join(parts)


@dataclass
class SynthesizedLabel:
    """An emotion label refined by Layer 4 with content + context grounding."""
    term: str
    confidence: str  # HIGH / MODERATE / LOW / SPECULATIVE
    reasoning: str  # why this term fits given content + context + brain data
    brain_grounding: str  # which brain regions support this
    content_grounding: str  # what the content says at this moment
    context_influence: Optional[str] = None  # how context modified the label
    original_layer3_term: Optional[str] = None  # what Layer 3 suggested before refinement


@dataclass
class SynthesisResult:
    """Layer 4 output for a single timestep."""
    timestep: int
    refined_labels: list[SynthesizedLabel]
    raw_llm_response: Optional[str] = None
    context_used: Optional[str] = None
    cost_estimate_usd: float = 0.0


class LLMSynthesizer:
    """Content-aware + context-aware emotion label refinement via LLM.

    Uses Claude (preferred) or Gemini to refine Layer 3's vocabulary
    scores with knowledge of WHAT the content says at each moment
    and WHO the viewer is.

    Usage:
        synth = LLMSynthesizer()
        result = synth.refine_timestep(
            timestep_emotion=te,
            content_at_timestep="No one wants to die...",
            context=ContextProfile(name="posthumous", description="..."),
        )
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: Optional[str] = None,
        vocabulary_path: Path | None = None,
    ):
        self.provider = provider
        self.model = model or self._default_model(provider)
        self._vocabulary_terms = self._load_vocabulary_terms(
            vocabulary_path or VOCABULARY_PATH
        )

    def refine_timestep(
        self,
        timestep_emotion: TimestepEmotion,
        content_at_timestep: str,
        context: Optional[ContextProfile] = None,
    ) -> SynthesisResult:
        """Refine emotion labels for a single timestep using LLM.

        Args:
            timestep_emotion: Layer 3 output (scored labels + brain state)
            content_at_timestep: what the speaker says / text at this moment
            context: optional viewer context profile

        Returns:
            SynthesisResult with refined labels
        """
        prompt = self._build_prompt(timestep_emotion, content_at_timestep, context)
        response = self._call_llm(prompt)
        labels = self._parse_response(response, timestep_emotion)

        return SynthesisResult(
            timestep=timestep_emotion.timestep,
            refined_labels=labels,
            raw_llm_response=response,
            context_used=context.name if context else None,
            cost_estimate_usd=0.02,  # rough estimate
        )

    def refine_arc(
        self,
        timestep_emotions: list[TimestepEmotion],
        content_segments: list[str],
        context: Optional[ContextProfile] = None,
        change_points_only: bool = True,
    ) -> list[SynthesisResult]:
        """Refine labels across an entire arc.

        Args:
            timestep_emotions: list of Layer 3 outputs
            content_segments: parallel list of content text per timestep
            context: optional viewer context
            change_points_only: if True, only refine change-point timesteps
                (saves cost; non-change-points use Layer 3 labels as-is)

        Returns:
            list of SynthesisResult, one per refined timestep
        """
        results = []
        for te, content in zip(timestep_emotions, content_segments):
            if change_points_only and not te.is_change_point:
                continue

            result = self.refine_timestep(te, content, context)
            results.append(result)

        return results

    def _build_prompt(
        self,
        te: TimestepEmotion,
        content: str,
        context: Optional[ContextProfile],
    ) -> str:
        """Build the LLM prompt for refinement."""

        # Format brain state
        brain_lines = []
        for cat, val in sorted(te.brain_state.items(), key=lambda x: -abs(x[1])):
            sign = "+" if val >= 0 else ""
            brain_lines.append(f"  {cat}: {sign}{val:.4f}")
        brain_text = "\n".join(brain_lines)

        # Format Layer 3 suggestions
        l3_lines = []
        for label in te.primary[:5]:
            met_regions = [r for r, g in label.brain_grounding.items() if g.get("met")]
            l3_lines.append(
                f"  - {label.term} (score={label.score:.2f}, "
                f"brain match: {', '.join(met_regions) or 'none'})"
            )
        l3_text = "\n".join(l3_lines)

        # Format change info
        change_text = ""
        if te.change_info:
            ci = te.change_info
            change_text = (
                f"\nBRAIN STATE TRANSITION at this moment:\n"
                f"  {ci.category} is {ci.direction} "
                f"(type: {ci.transition_type.value}, delta: {ci.delta:+.4f})\n"
                f"  This is a significant change point in the brain response."
            )

        # Format context
        context_text = ""
        if context:
            context_text = (
                f"\nVIEWER CONTEXT (this changes the emotional interpretation):\n"
                f"{context.to_prompt_text()}\n"
                f"\nConsider how this specific viewer's context would modify "
                f"their emotional experience of this content."
            )

        # Format vocabulary list
        vocab_text = ", ".join(sorted(self._vocabulary_terms))

        prompt = f"""You are analyzing a brain-response prediction to identify the precise emotional experience at a specific moment in content.

BRAIN ACTIVATION DATA (from TRIBE v2 fMRI prediction):
{brain_text}

DIMENSIONS (computed from brain data):
  Valence: {te.dimensions.get('valence', 0):.3f} (positive=pleasant, negative=unpleasant)
  Arousal: {te.dimensions.get('arousal', 0):.3f} (positive=activated, negative=calm)
  Body-focus: {te.dimensions.get('body_focus', 0):.3f} (positive=interoceptive, negative=cognitive)
{change_text}

CONTENT AT THIS MOMENT (what the listener is hearing):
  "{content}"

LAYER 3 SUGGESTIONS (vocabulary-matched, pre-refinement):
{l3_text}
{context_text}

YOUR TASK:
Given the brain data AND the content at this moment{' AND the viewer context' if context else ''}, select the 3-5 most precise emotional terms from the vocabulary below. You may keep Layer 3's suggestions, replace them, or reorder them.

For each term you select:
1. State the term
2. Rate confidence: HIGH (brain data + content strongly support), MODERATE (plausible), LOW (uncertain), SPECULATIVE (possible but weak evidence)
3. Explain in one sentence WHY this term fits, grounding your reasoning in:
   - Which brain regions support it (cite specific activation values)
   - What the content says that makes this term precise (not just generically fitting)
   {'- How the viewer context modifies the emotional experience' if context else ''}
4. Note which Layer 3 suggestion (if any) this refines or replaces

AVAILABLE VOCABULARY:
{vocab_text}

Respond in this exact JSON format:
[
  {{
    "term": "mortality_awareness",
    "confidence": "HIGH",
    "reasoning": "Interoception at +0.47 with language at -0.10 indicates body-level processing overriding verbal comprehension. The content explicitly discusses death ('no one wants to die'), making mortality_awareness more precise than the generic 'awe' suggested by Layer 3.",
    "brain_grounding": "interoception +0.47 (very high body-sensing), language -0.10 (verbal suppression)",
    "content_grounding": "Speaker says 'No one wants to die' — explicit mortality content",
    "context_influence": null,
    "replaces_layer3": "awe"
  }}
]"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM provider."""
        if self.provider == "anthropic":
            return self._call_anthropic(prompt)
        elif self.provider == "google":
            return self._call_google(prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _call_anthropic(self, prompt: str) -> str:
        """Call Claude via Anthropic API."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install: pip install anthropic"
            )

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in environment")

        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _call_google(self, prompt: str) -> str:
        """Call Gemini via Google AI API."""
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai package required. Install: pip install google-genai"
            )

        api_key = os.getenv("GOOGLE_AI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_AI_API_KEY not set in environment")

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=[prompt],
        )
        return response.text

    def _parse_response(
        self,
        response: str,
        te: TimestepEmotion,
    ) -> list[SynthesizedLabel]:
        """Parse LLM response into SynthesizedLabel objects."""
        # Extract JSON from response (handle markdown code blocks)
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: try to find JSON array in the response
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                try:
                    items = json.loads(text[start:end])
                except json.JSONDecodeError:
                    return self._fallback_labels(te)
            else:
                return self._fallback_labels(te)

        labels = []
        for item in items:
            labels.append(SynthesizedLabel(
                term=item.get("term", "unknown"),
                confidence=item.get("confidence", "SPECULATIVE"),
                reasoning=item.get("reasoning", ""),
                brain_grounding=item.get("brain_grounding", ""),
                content_grounding=item.get("content_grounding", ""),
                context_influence=item.get("context_influence"),
                original_layer3_term=item.get("replaces_layer3"),
            ))

        return labels

    def _fallback_labels(self, te: TimestepEmotion) -> list[SynthesizedLabel]:
        """Fallback: use Layer 3 labels if LLM parsing fails."""
        return [
            SynthesizedLabel(
                term=label.term,
                confidence="LOW",
                reasoning=f"LLM parsing failed; using Layer 3 label. {label.reasoning}",
                brain_grounding=str(label.brain_grounding),
                content_grounding="(LLM refinement unavailable)",
                original_layer3_term=label.term,
            )
            for label in te.primary[:5]
        ]

    @staticmethod
    def _default_model(provider: str) -> str:
        defaults = {
            "anthropic": "claude-sonnet-4-6",
            "google": "gemini-2.0-flash",
        }
        return defaults.get(provider, "claude-sonnet-4-6")

    @staticmethod
    def _load_vocabulary_terms(path: Path) -> list[str]:
        with open(path) as f:
            data = yaml.safe_load(f)
        return list(data.get("terms", {}).keys())
