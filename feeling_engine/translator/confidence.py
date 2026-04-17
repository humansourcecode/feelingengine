"""Layer 5 — Confidence scoring for emotion labels.

Each EmotionLabel gets a confidence rating based on:
- How many brain-region expectations were met
- How strong the activation magnitudes are
- Whether the transition type matches the term's affinity
- How close the dimensional coordinates are

Confidence levels: HIGH, MODERATE, LOW, SPECULATIVE
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from feeling_engine.translator.brain_to_emotion import EmotionLabel, TimestepEmotion


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    SPECULATIVE = "speculative"


@dataclass
class ScoredLabel:
    """An EmotionLabel with an added confidence assessment."""
    label: EmotionLabel
    confidence: ConfidenceLevel
    confidence_factors: dict  # breakdown of what contributed


def score_confidence(label: EmotionLabel) -> ScoredLabel:
    """Assess confidence in a single emotion label."""
    factors = {}

    # Factor 1: overall match score
    factors["match_score"] = label.score
    score_ok = label.score >= 0.5

    # Factor 2: brain-region expectations met ratio
    if label.brain_grounding:
        met = sum(1 for g in label.brain_grounding.values() if g.get("met"))
        total = len(label.brain_grounding)
        factors["expectations_met"] = f"{met}/{total}"
        expectations_ok = met / total >= 0.6 if total > 0 else False
    else:
        factors["expectations_met"] = "none defined"
        expectations_ok = False

    # Factor 3: dimensional proximity
    factors["dimensional_distance"] = label.dimensional_distance
    dimension_ok = label.dimensional_distance < 0.8

    # Factor 4: transition match
    factors["transition_match"] = label.transition_match

    # Compute confidence level
    true_count = sum([score_ok, expectations_ok, dimension_ok, label.transition_match])

    if true_count >= 4:
        confidence = ConfidenceLevel.HIGH
    elif true_count >= 3:
        confidence = ConfidenceLevel.MODERATE
    elif true_count >= 2:
        confidence = ConfidenceLevel.LOW
    else:
        confidence = ConfidenceLevel.SPECULATIVE

    return ScoredLabel(
        label=label,
        confidence=confidence,
        confidence_factors=factors,
    )


def score_timestep(timestep_emotion: TimestepEmotion) -> list[ScoredLabel]:
    """Score confidence for all labels at a timestep."""
    return [score_confidence(label) for label in timestep_emotion.primary]
