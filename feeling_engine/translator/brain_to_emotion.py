"""Layer 3 — Neuroscience-grounded brain-to-emotion mapping.

Takes brain region activations (from BrainModelAdapter) and change points
(from change_detector) and maps them to emotional vocabulary terms using
dimensional coordinates and brain-region expectations defined in vocabulary.yaml.

Each brain state is projected into a dimensional space (valence, arousal,
body_focus) and matched against vocabulary terms whose dimensional
coordinates and brain-region expectations best fit the observed data.

This is the core of the Feeling Engine — the mapping between
neural prediction and human-readable emotion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from feeling_engine.adapters.brain_model.base import BrainRegionProfile, BrainTimeSeries
from feeling_engine.translator.change_detector import (
    ChangeAnalysis,
    ChangePoint,
    TransitionType,
)

VOCABULARY_PATH = Path(__file__).parent / "vocabulary.yaml"


@dataclass
class EmotionLabel:
    """A single emotional label with its brain grounding."""
    term: str
    score: float  # how well this term matches the brain data (0-1)
    brain_grounding: dict  # which brain regions support this label
    transition_match: bool  # does the transition type match this term's affinity?
    dimensional_distance: float  # distance in valence/arousal/body_focus space
    reasoning: str  # one-sentence explanation of why this term was selected


@dataclass
class TimestepEmotion:
    """Emotional interpretation of a single timestep."""
    timestep: int
    primary: list[EmotionLabel]  # top 3-5 labels, ranked by score
    brain_state: dict  # raw brain category activations at this timestep
    dimensions: dict  # computed valence, arousal, body_focus
    is_change_point: bool
    change_info: Optional[ChangePoint] = None


@dataclass
class EmotionalArc:
    """Complete emotional arc — the final output of the Translator.

    A sequence of TimestepEmotion entries, with summary statistics
    and arc-level narrative.
    """
    timesteps: list[TimestepEmotion]
    n_timesteps: int
    change_points_used: int
    arc_summary: list[dict] = field(default_factory=list)  # condensed arc segments
    metadata: dict = field(default_factory=dict)


class BrainToEmotionMapper:
    """Maps brain activations to emotional vocabulary using dimensional
    coordinates and brain-region expectations.

    Usage:
        mapper = BrainToEmotionMapper()
        arc = mapper.map(brain_time_series, change_analysis)
    """

    # Dimensional computation weights
    # How each brain category contributes to the three dimensions
    DIMENSION_WEIGHTS = {
        "valence": {
            "reward": 0.4,
            "core_affect": 0.3,
            "social": 0.2,
            "regulation": 0.1,
            "interoception": -0.1,
        },
        "arousal": {
            "core_affect": 0.35,
            "interoception": 0.3,
            "reward": 0.15,
            "regulation": -0.1,
            "memory": -0.1,
            "social": 0.0,
            "language": 0.0,
        },
        "body_focus": {
            "interoception": 0.5,
            "core_affect": 0.15,
            "language": -0.35,
            "regulation": -0.15,
            "memory": -0.1,
        },
    }

    def __init__(self, vocabulary_path: Path | None = None):
        path = vocabulary_path or VOCABULARY_PATH
        with open(path) as f:
            data = yaml.safe_load(f)

        self.terms = data["terms"]
        self.thresholds = data["thresholds"]
        self._term_list = list(self.terms.keys())

    def map(
        self,
        brain_ts: BrainTimeSeries,
        change_analysis: ChangeAnalysis,
        top_k: int = 5,
        change_points_only: bool = False,
    ) -> EmotionalArc:
        """Map a brain time series to an emotional arc.

        Args:
            brain_ts: brain region activations per timestep
            change_analysis: detected change points from Layer 2
            top_k: number of emotion labels per timestep
            change_points_only: if True, only label change-point timesteps
                (faster, produces sparser arc)

        Returns:
            EmotionalArc with per-timestep emotion labels
        """
        # Index change points by timestep for fast lookup
        cp_by_timestep: dict[int, list[ChangePoint]] = {}
        for cp in change_analysis.change_points:
            cp_by_timestep.setdefault(cp.timestep, []).append(cp)

        timestep_emotions: list[TimestepEmotion] = []

        for profile in brain_ts.profiles:
            t = profile.timestep
            is_cp = t in cp_by_timestep

            if change_points_only and not is_cp:
                continue

            # Compute dimensional coordinates from brain activations
            dims = self._compute_dimensions(profile.categories)

            # Get the dominant change point info for this timestep
            change_info = None
            if is_cp:
                cps = cp_by_timestep[t]
                change_info = max(cps, key=lambda c: c.delta_magnitude)

            # Score each vocabulary term against this brain state
            scored_labels = self._score_all_terms(
                categories=profile.categories,
                dimensions=dims,
                change_info=change_info,
            )

            # Take top_k
            top_labels = sorted(scored_labels, key=lambda l: -l.score)[:top_k]

            timestep_emotions.append(TimestepEmotion(
                timestep=t,
                primary=top_labels,
                brain_state=dict(profile.categories),
                dimensions=dims,
                is_change_point=is_cp,
                change_info=change_info,
            ))

        # Build arc summary (condensed narrative segments)
        arc_summary = self._build_arc_summary(timestep_emotions)

        return EmotionalArc(
            timesteps=timestep_emotions,
            n_timesteps=len(timestep_emotions),
            change_points_used=len(cp_by_timestep),
            arc_summary=arc_summary,
        )

    # Brain activations are typically in [-0.3, +0.5] range.
    # Vocabulary dimensions are in [-1, +1] range.
    # Scale factor normalizes brain-derived dimensions to vocabulary scale.
    DIMENSION_SCALE = 3.5

    def _compute_dimensions(self, categories: dict[str, float]) -> dict:
        """Project brain activations into dimensional emotion space.

        Brain activations are small values (typically ±0.3). Vocabulary
        terms use a [-1, +1] scale. We scale brain-derived dimensions
        to match, so dimensional distance scoring works correctly.
        """
        dims = {}
        for dim_name, weights in self.DIMENSION_WEIGHTS.items():
            raw = sum(
                categories.get(cat, 0.0) * weight
                for cat, weight in weights.items()
            )
            scaled = max(-1.0, min(1.0, raw * self.DIMENSION_SCALE))
            dims[dim_name] = round(scaled, 4)
        return dims

    def _score_all_terms(
        self,
        categories: dict[str, float],
        dimensions: dict,
        change_info: ChangePoint | None,
    ) -> list[EmotionLabel]:
        """Score every vocabulary term against the current brain state."""
        labels = []
        for term_name, term_def in self.terms.items():
            score, distance, grounding, transition_match, reasoning = (
                self._score_term(term_name, term_def, categories, dimensions, change_info)
            )
            labels.append(EmotionLabel(
                term=term_name,
                score=score,
                brain_grounding=grounding,
                transition_match=transition_match,
                dimensional_distance=distance,
                reasoning=reasoning,
            ))
        return labels

    def _score_term(
        self,
        term_name: str,
        term_def: dict,
        categories: dict[str, float],
        dimensions: dict,
        change_info: ChangePoint | None,
    ) -> tuple[float, float, dict, bool, str]:
        """Score a single vocabulary term against the brain state.

        Returns: (score, dimensional_distance, grounding, transition_match, reasoning)
        """
        # 1. Dimensional distance (lower = better match)
        d_valence = (dimensions.get("valence", 0) - term_def.get("valence", 0)) ** 2
        d_arousal = (dimensions.get("arousal", 0) - term_def.get("arousal", 0)) ** 2
        d_body = (dimensions.get("body_focus", 0) - term_def.get("body_focus", 0)) ** 2
        distance = (d_valence + d_arousal + d_body) ** 0.5
        dimensional_score = max(0, 1.0 - distance)  # closer = higher score

        # 2. Brain-region expectation match
        expectations = term_def.get("brain_expectations", {})
        region_score, grounding = self._check_expectations(categories, expectations)

        # 3. Transition type match
        transition_match = False
        transition_bonus = 0.0
        if change_info is not None:
            affinities = term_def.get("transition_affinity", [])
            if change_info.transition_type.value in affinities:
                transition_match = True
                transition_bonus = 0.15

        # Composite score (weighted combination)
        score = (
            dimensional_score * 0.35
            + region_score * 0.45
            + transition_bonus
            + (0.05 if transition_match else 0.0)
        )
        score = min(1.0, max(0.0, score))

        # Reasoning
        reasoning = self._build_reasoning(
            term_name, categories, dimensions, grounding, change_info, transition_match
        )

        return score, distance, grounding, transition_match, reasoning

    def _check_expectations(
        self,
        categories: dict[str, float],
        expectations: dict[str, str],
    ) -> tuple[float, dict]:
        """Check how well brain activations match a term's expectations.

        Returns: (match_score 0-1, grounding dict)
        """
        if not expectations:
            return 0.5, {}

        matches = 0
        total = len(expectations)
        grounding = {}

        for region, expected_level in expectations.items():
            actual = categories.get(region, 0.0)
            met = self._meets_expectation(actual, expected_level)
            matches += 1 if met else 0
            grounding[region] = {
                "expected": expected_level,
                "actual": round(actual, 4),
                "met": met,
            }

        return matches / total if total > 0 else 0.5, grounding

    def _meets_expectation(self, actual: float, expected: str) -> bool:
        """Check if an activation level meets a qualitative expectation."""
        t = self.thresholds
        checks = {
            "very_high": actual >= t["very_high"],
            "high": actual >= t["high"],
            "moderate": actual >= t["moderate"],
            "low": actual < t["moderate"],
            "negative": actual < t["negative"],
            "declining": actual < 0,  # simplified; proper version needs temporal context
            "declining_then_rising": False,  # needs temporal context from change detector
            "spike": actual >= t["very_high"],  # simplified
        }
        return checks.get(expected, actual >= t["moderate"])

    def _build_reasoning(
        self,
        term_name: str,
        categories: dict,
        dimensions: dict,
        grounding: dict,
        change_info: ChangePoint | None,
        transition_match: bool,
    ) -> str:
        """Build a one-sentence explanation for why this term was selected."""
        met_regions = [r for r, g in grounding.items() if g.get("met")]
        unmet_regions = [r for r, g in grounding.items() if not g.get("met")]

        parts = []
        if met_regions:
            parts.append(f"{', '.join(met_regions)} match expected pattern")
        if unmet_regions:
            parts.append(f"{', '.join(unmet_regions)} don't match")
        if transition_match and change_info:
            parts.append(
                f"transition type ({change_info.transition_type.value}) "
                f"aligns with {term_name}'s affinity"
            )

        return f"{term_name}: {'; '.join(parts)}" if parts else f"{term_name}: baseline match"

    def _build_arc_summary(
        self, timesteps: list[TimestepEmotion]
    ) -> list[dict]:
        """Condense the full timestep list into narrative arc segments."""
        return build_arc_summary(timesteps)


def build_arc_summary(timesteps: list["TimestepEmotion"]) -> list[dict]:
    """Condense a timestep list into narrative arc segments.

    Groups consecutive timesteps sharing the same top-ranked emotion into
    a single segment. Uses whatever is currently in `te.primary[0].term`,
    so callers can run this again after Layer 4 refinement to get a
    Layer-4-aware summary.
    """
    if not timesteps:
        return []

    segments = []
    current_top_term = None
    segment_start = None

    for te in timesteps:
        if not te.primary:
            continue

        top_term = te.primary[0].term

        if top_term != current_top_term:
            if current_top_term is not None:
                segments.append({
                    "start_timestep": segment_start,
                    "end_timestep": te.timestep - 1,
                    "dominant_emotion": current_top_term,
                    "is_change_point": te.is_change_point,
                })
            current_top_term = top_term
            segment_start = te.timestep

    if current_top_term is not None:
        segments.append({
            "start_timestep": segment_start,
            "end_timestep": timesteps[-1].timestep,
            "dominant_emotion": current_top_term,
            "is_change_point": False,
        })

    return segments
