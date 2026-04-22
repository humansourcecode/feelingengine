"""Unit tests for rendering.brain_renderer — signatures + vertex mapping + render plumbing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from feeling_engine.rendering.signatures import (
    CATEGORIES,
    MECHANISM_SIGNATURE,
    get_signature,
    signature_as_vector,
)
from feeling_engine.mechanisms.vocabulary import MECHANISM_LABELS


# ─── signatures ───────────────────────────────────────────────────

def test_every_vocabulary_label_has_a_signature():
    vocab_names = {spec.name for spec in MECHANISM_LABELS}
    sig_names = set(MECHANISM_SIGNATURE)
    assert vocab_names == sig_names, (
        f"missing signatures: {vocab_names - sig_names}; "
        f"extra signatures: {sig_names - vocab_names}"
    )


def test_signatures_use_only_valid_categories():
    for label, sig in MECHANISM_SIGNATURE.items():
        for cat in sig:
            assert cat in CATEGORIES, f"{label} uses unknown category {cat}"


def test_signature_values_are_in_range():
    for label, sig in MECHANISM_SIGNATURE.items():
        for cat, val in sig.items():
            assert -1.0 <= val <= 1.0, (
                f"{label}.{cat} = {val} out of [-1, 1]"
            )


def test_signature_as_vector_returns_all_categories():
    v = signature_as_vector("body-turn")
    assert set(v) == set(CATEGORIES)
    # Unlisted categories should be 0.0
    assert v["reward"] == 0.0
    # Listed categories should match
    assert v["interoception"] == 0.80


def test_get_signature_raises_for_unknown():
    with pytest.raises(KeyError, match="unknown"):
        get_signature("nonexistent-label")


# ─── vertex mapping ───────────────────────────────────────────────

def test_category_regions_all_nonempty():
    from feeling_engine.rendering.brain_renderer import CATEGORY_REGIONS
    for cat in CATEGORIES:
        assert cat in CATEGORY_REGIONS, f"category {cat} missing from CATEGORY_REGIONS"
        assert len(CATEGORY_REGIONS[cat]) > 0, f"category {cat} has no regions"


def test_vertex_activation_maps_return_correct_shape():
    """This hits the annotation file but no network; fast enough for unit test."""
    from feeling_engine.rendering.brain_renderer import (
        _vertex_activation_maps, ANNOT_LH,
    )
    if not ANNOT_LH.exists():
        pytest.skip("HCP-MMP annotation file not available")

    lh, rh = _vertex_activation_maps("body-turn", intensity=1.0)
    assert isinstance(lh, np.ndarray)
    assert isinstance(rh, np.ndarray)
    # fsaverage full mesh = 163842 vertices per hemisphere
    assert lh.shape == (163842,)
    assert rh.shape == (163842,)
    # body-turn has nonzero interoception signal → some vertices should be nonzero
    assert (lh != 0.0).sum() > 0
    assert (rh != 0.0).sum() > 0


def test_vertex_activations_scale_with_intensity():
    from feeling_engine.rendering.brain_renderer import _vertex_activation_maps, ANNOT_LH
    if not ANNOT_LH.exists():
        pytest.skip("HCP-MMP annotation file not available")

    lh_full, _ = _vertex_activation_maps("body-turn", intensity=1.0)
    lh_half, _ = _vertex_activation_maps("body-turn", intensity=0.5)
    # Same vertices should be lit; values scaled
    nonzero = lh_full != 0
    assert np.allclose(lh_half[nonzero], lh_full[nonzero] * 0.5)


def test_distinct_labels_produce_different_activation_maps():
    from feeling_engine.rendering.brain_renderer import _vertex_activation_maps, ANNOT_LH
    if not ANNOT_LH.exists():
        pytest.skip("HCP-MMP annotation file not available")

    lh_body, _ = _vertex_activation_maps("body-turn")
    lh_rec, _ = _vertex_activation_maps("recognition")
    # Different signatures → different activation maps
    assert not np.array_equal(lh_body, lh_rec)


# ─── render API validation (no actual rendering in unit tests — too slow) ─

def test_render_rejects_unknown_view():
    from feeling_engine.rendering.brain_renderer import render_mechanism_brain
    with pytest.raises(ValueError, match="view must be one of"):
        render_mechanism_brain("body-turn", "/tmp/unused.png", view="bogus-view")


def test_render_rejects_unknown_label():
    from feeling_engine.rendering.brain_renderer import render_mechanism_brain
    with pytest.raises(KeyError, match="unknown mechanism"):
        render_mechanism_brain("invented-label", "/tmp/unused.png")
