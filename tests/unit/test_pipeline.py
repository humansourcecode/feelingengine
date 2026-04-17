"""Pipeline (Mode 1) — offline profile-loading integration test.

Runs the full translator pipeline (Layers 2, 3, 5) against a small
pre-computed TRIBE profile fixture. No external APIs are hit; this
test is safe to run anywhere.
"""
from pathlib import Path

import pytest

from feeling_engine.adapters.brain_model.tribev2 import TRIBEv2Adapter
from feeling_engine.pipeline import FeelingPipeline
from feeling_engine.translator.brain_to_emotion import build_arc_summary


FIXTURE = Path(__file__).parent / "fixtures" / "tiny_profiles.json"


@pytest.fixture
def pipeline():
    return FeelingPipeline(brain_adapter=TRIBEv2Adapter())


def test_profile_fixture_exists():
    assert FIXTURE.exists(), f"Fixture missing: {FIXTURE}"


def test_pipeline_produces_arc(pipeline):
    arc = pipeline.analyze_profiles(FIXTURE)
    assert arc.n_timesteps > 0
    assert len(arc.timesteps) == arc.n_timesteps


def test_arc_timesteps_have_primary_labels(pipeline):
    arc = pipeline.analyze_profiles(FIXTURE)
    labeled = [te for te in arc.timesteps if te.primary]
    assert labeled, "At least one timestep should have primary labels"
    for te in labeled:
        assert te.primary[0].term
        assert te.primary[0].score >= 0


def test_arc_serializes_to_dict(pipeline):
    """arc_to_dict output must be JSON-compatible (round-trip via json.dumps)."""
    import json
    arc = pipeline.analyze_profiles(FIXTURE)
    d = pipeline.arc_to_dict(arc)
    # Should not raise
    json.dumps(d)
    assert d["n_timesteps"] == arc.n_timesteps


def test_arc_summary_reflects_primary_labels(pipeline):
    """Regression for #56: arc_summary must follow te.primary[0].term.

    If we swap out the primary labels (simulating Layer 4 refinement),
    build_arc_summary() should produce a summary based on the new labels.
    """
    arc = pipeline.analyze_profiles(FIXTURE)

    # Fake a Layer 4 refinement: force every labeled timestep to 'mortality_awareness'
    from feeling_engine.translator.brain_to_emotion import EmotionLabel
    for te in arc.timesteps:
        if te.primary:
            te.primary = [EmotionLabel(
                term="mortality_awareness",
                score=1.0,
                brain_grounding={},
                transition_match=True,
                dimensional_distance=0.0,
                reasoning="test-forced",
            )]

    new_summary = build_arc_summary(arc.timesteps)
    # All (or nearly all) segments should now be mortality_awareness
    terms = {seg["dominant_emotion"] for seg in new_summary}
    assert "mortality_awareness" in terms
    assert len(terms) == 1, f"Expected single term after refinement, got: {terms}"


def test_change_points_only_returns_subset(pipeline):
    full_arc = pipeline.analyze_profiles(FIXTURE)
    cp_arc = pipeline.analyze_profiles(FIXTURE, change_points_only=True)
    assert cp_arc.n_timesteps <= full_arc.n_timesteps
