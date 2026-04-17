"""Feeling Engine Pipeline — orchestrates the full analysis flow.

Content (text/audio) → TTS (if text) → TRIBE (brain prediction)
→ Region Mapping → Change Detection → Brain-to-Emotion Mapping
→ Confidence Scoring → Emotional Arc

Usage:
    from feeling_engine.pipeline import FeelingPipeline
    from feeling_engine.adapters.brain_model.tribev2 import TRIBEv2Adapter

    pipeline = FeelingPipeline(brain_adapter=TRIBEv2Adapter())

    # From pre-computed TRIBE profiles (JSON)
    arc = pipeline.analyze_profiles("path/to/profiles.json")

    # From raw TRIBE predictions (numpy)
    arc = pipeline.analyze_predictions(predictions_array)
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from feeling_engine.adapters.brain_model.base import BrainModelAdapter, BrainTimeSeries
from feeling_engine.translator.change_detector import ChangeAnalysis, detect_changes
from feeling_engine.translator.brain_to_emotion import (
    BrainToEmotionMapper,
    EmotionalArc,
)
from feeling_engine.translator.confidence import (
    ConfidenceLevel,
    ScoredLabel,
    score_timestep,
)


class FeelingPipeline:
    """End-to-end pipeline from brain data to emotional arc.

    Currently supports two entry points:
    1. Pre-computed TRIBE profiles (JSON) — for existing data
    2. Raw numpy prediction arrays — for fresh TRIBE runs

    Future: will add text/audio entry points once TTS and compute
    adapters are wired in.
    """

    def __init__(
        self,
        brain_adapter: BrainModelAdapter,
        vocabulary_path: Path | None = None,
        change_threshold: float = 0.08,
        top_k: int = 5,
    ):
        self.brain_adapter = brain_adapter
        self.mapper = BrainToEmotionMapper(vocabulary_path=vocabulary_path)
        self.change_threshold = change_threshold
        self.top_k = top_k

    def analyze_profiles(
        self,
        profiles_path: str | Path,
        change_points_only: bool = False,
    ) -> EmotionalArc:
        """Analyze from pre-computed TRIBE profile JSON.

        Expected format (from tribe_modal.py output):
        [{"timestep": 0, "categories": {"core_affect": 0.33, ...}}, ...]
        """
        path = Path(profiles_path)
        data = json.loads(path.read_text())

        # Handle both list format and dict-with-list format
        if isinstance(data, dict):
            profiles = data.get("profiles", data.get("categories", [data]))
        elif isinstance(data, list):
            profiles = data
        else:
            raise ValueError(f"Unexpected profile format in {path}")

        # Use the adapter's profile-loading method
        from feeling_engine.adapters.brain_model.tribev2 import TRIBEv2Adapter
        if isinstance(self.brain_adapter, TRIBEv2Adapter):
            brain_ts = self.brain_adapter.map_from_profiles(profiles)
        else:
            raise NotImplementedError(
                "Profile loading only supported for TRIBEv2Adapter. "
                "Use analyze_predictions() for other adapters."
            )

        return self._run_pipeline(brain_ts, change_points_only)

    def analyze_predictions(
        self,
        predictions: np.ndarray,
        change_points_only: bool = False,
    ) -> EmotionalArc:
        """Analyze from raw TRIBE prediction array.

        Args:
            predictions: shape (n_timesteps, n_vertices)
        """
        brain_ts = self.brain_adapter.map_to_regions(predictions)
        return self._run_pipeline(brain_ts, change_points_only)

    def _run_pipeline(
        self,
        brain_ts: BrainTimeSeries,
        change_points_only: bool,
    ) -> EmotionalArc:
        """Core pipeline: change detection → mapping → confidence."""

        # Layer 2: Change detection
        change_analysis = detect_changes(
            brain_ts,
            threshold=self.change_threshold,
        )

        # Layer 3: Brain-to-emotion mapping
        arc = self.mapper.map(
            brain_ts,
            change_analysis,
            top_k=self.top_k,
            change_points_only=change_points_only,
        )

        # Layer 5: Confidence scoring (applied per-timestep)
        for te in arc.timesteps:
            scored = score_timestep(te)
            # Annotate labels with confidence
            for sl in scored:
                sl.label.reasoning = (
                    f"[{sl.confidence.value.upper()}] {sl.label.reasoning}"
                )

        return arc

    def format_arc_text(self, arc: EmotionalArc) -> str:
        """Render an emotional arc as human-readable text."""
        lines = []
        lines.append(f"Emotional Arc ({arc.n_timesteps} timesteps, "
                      f"{arc.change_points_used} change points)")
        lines.append("=" * 60)

        for te in arc.timesteps:
            marker = " ⚡" if te.is_change_point else ""
            lines.append(f"\nTimestep {te.timestep}{marker}")
            lines.append(f"  Brain: {_fmt_brain(te.brain_state)}")
            lines.append(f"  Dimensions: V={te.dimensions.get('valence', 0):.3f} "
                         f"A={te.dimensions.get('arousal', 0):.3f} "
                         f"B={te.dimensions.get('body_focus', 0):.3f}")

            if te.change_info:
                ci = te.change_info
                lines.append(f"  Change: {ci.category} {ci.direction} "
                             f"({ci.transition_type.value}, Δ={ci.delta:+.3f})")

            for i, label in enumerate(te.primary[:3]):
                lines.append(f"  {'→' if i == 0 else ' '} {label.term} "
                             f"(score={label.score:.2f}) — {label.reasoning}")

        if arc.arc_summary:
            lines.append("\n" + "=" * 60)
            lines.append("ARC SUMMARY")
            for seg in arc.arc_summary:
                lines.append(f"  t{seg['start_timestep']}-{seg['end_timestep']}: "
                             f"{seg['dominant_emotion']}")

        return "\n".join(lines)

    def arc_to_dict(self, arc: EmotionalArc) -> dict:
        """Serialize an emotional arc to a JSON-compatible dict."""
        return {
            "n_timesteps": arc.n_timesteps,
            "change_points_used": arc.change_points_used,
            "arc_summary": arc.arc_summary,
            "timesteps": [
                {
                    "timestep": te.timestep,
                    "is_change_point": te.is_change_point,
                    "brain_state": te.brain_state,
                    "dimensions": te.dimensions,
                    "emotions": [
                        {
                            "term": l.term,
                            "score": round(l.score, 4),
                            "confidence": next(
                                (sl.confidence.value
                                 for sl in score_timestep(te)
                                 if sl.label.term == l.term),
                                "unknown"
                            ),
                            "brain_grounding": l.brain_grounding,
                            "reasoning": l.reasoning,
                        }
                        for l in te.primary[:5]
                    ],
                    "change": {
                        "category": te.change_info.category,
                        "direction": te.change_info.direction,
                        "delta": round(te.change_info.delta, 4),
                        "transition_type": te.change_info.transition_type.value,
                    } if te.change_info else None,
                }
                for te in arc.timesteps
            ],
        }


def _fmt_brain(brain_state: dict) -> str:
    """Format brain state dict for display."""
    parts = []
    for cat, val in sorted(brain_state.items(), key=lambda x: -abs(x[1])):
        sign = "+" if val >= 0 else ""
        parts.append(f"{cat}={sign}{val:.3f}")
    return " | ".join(parts[:4])
