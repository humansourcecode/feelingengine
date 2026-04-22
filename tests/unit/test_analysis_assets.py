"""Unit tests for rendering.analysis_assets — key-moment selection + bundle shape."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from feeling_engine.mechanisms.vocabulary import LabelApplication, SequenceMatch
from feeling_engine.rendering.analysis_assets import (
    KeyMoment,
    AnalysisBundle,
    extract_key_moments,
    _score_firing,
    _enforce_temporal_spread,
    _labels_in_sequences,
    render_analysis_bundle,
)


def _app(label, start, end, intensity=0.7, confidence=0.8, evidence=""):
    return LabelApplication(
        label=label, tier=1,
        start_sec=start, end_sec=end,
        intensity=intensity, confidence=confidence,
        signals={"evidence": evidence, "detector": "test"},
    )


def _seq(name, start, end, matched_labels):
    return SequenceMatch(
        name=name, start_sec=start, end_sec=end,
        matched_labels=matched_labels,
        positions=[int(start), int(end)],
        partial=False,
    )


# ─── score_firing / labels_in_sequences ──────────────────────────

def test_score_firing_baseline_is_intensity_times_confidence():
    a = _app("body-turn", 0, 1, intensity=0.8, confidence=0.5)
    score = _score_firing(a, set())
    assert score == pytest.approx(0.4)


def test_score_firing_boosts_when_in_sequence():
    a = _app("body-turn", 0, 1, intensity=0.8, confidence=0.5)
    base = _score_firing(a, set())
    boosted = _score_firing(a, {"body-turn"})
    assert boosted > base
    assert boosted == pytest.approx(base * 1.2)


def test_labels_in_sequences_collects_all_matched_labels():
    seqs = [
        _seq("s1", 0, 10, ["body-turn", "recognition"]),
        _seq("s2", 15, 25, ["drift", "body-turn"]),
    ]
    names = _labels_in_sequences(seqs)
    assert names == {"body-turn", "recognition", "drift"}


# ─── extract_key_moments ─────────────────────────────────────────

def test_extract_key_moments_returns_empty_for_empty_arc():
    moments = extract_key_moments([], [], duration_sec=60.0, n=5)
    assert moments == []


def test_extract_key_moments_picks_top_n():
    arc = [
        _app("body-turn",         10, 12, intensity=0.9, confidence=0.9),
        _app("drift",             15, 17, intensity=0.3, confidence=0.4),
        _app("recognition",       25, 27, intensity=0.8, confidence=0.9),
        _app("vulnerability-transfer", 40, 42, intensity=0.7, confidence=0.8),
        _app("pattern-break",     55, 57, intensity=0.2, confidence=0.3),
    ]
    moments = extract_key_moments(arc, [], duration_sec=60.0, n=3)
    assert len(moments) == 3
    # Should exclude low-score firings (drift 0.12, pattern-break 0.06)
    picked_labels = {m.label for m in moments}
    assert "drift" not in picked_labels
    assert "pattern-break" not in picked_labels


def test_extract_key_moments_sorted_by_start_sec():
    arc = [
        _app("body-turn",   50, 52, intensity=0.9, confidence=0.9),
        _app("recognition", 10, 12, intensity=0.9, confidence=0.9),
        _app("evocation",   30, 32, intensity=0.9, confidence=0.9),
    ]
    moments = extract_key_moments(arc, [], duration_sec=60.0, n=5)
    assert [m.start_sec for m in moments] == [10, 30, 50]


def test_extract_key_moments_indexes_are_1_based_and_sequential():
    arc = [_app("body-turn", i * 10, i * 10 + 2,
                intensity=0.9, confidence=0.9) for i in range(5)]
    moments = extract_key_moments(arc, [], duration_sec=60.0, n=5)
    assert [m.index for m in moments] == [1, 2, 3, 4, 5]


def test_extract_key_moments_annotates_sequence_membership():
    arc = [
        _app("body-turn",   10, 12, intensity=0.9, confidence=0.9),
        _app("recognition", 30, 32, intensity=0.9, confidence=0.9),
    ]
    seqs = [_seq("intimacy-deepening", 5, 25, ["body-turn"])]
    moments = extract_key_moments(arc, seqs, duration_sec=60.0, n=5)
    body_turn_moment = next(m for m in moments if m.label == "body-turn")
    recognition_moment = next(m for m in moments if m.label == "recognition")
    assert body_turn_moment.in_sequence == "intimacy-deepening"
    assert recognition_moment.in_sequence is None


def test_extract_key_moments_captures_co_firing():
    arc = [
        _app("body-turn",   10, 12, intensity=0.9, confidence=0.9),
        _app("vulnerability-transfer", 10.5, 12.5,
             intensity=0.7, confidence=0.8),
        _app("recognition", 40, 42, intensity=0.8, confidence=0.9),
    ]
    moments = extract_key_moments(arc, [], duration_sec=60.0, n=5)
    body_turn_moment = next(m for m in moments if m.label == "body-turn")
    assert "vulnerability-transfer" in body_turn_moment.co_firing


# ─── temporal spread ──────────────────────────────────────────────

def test_temporal_spread_avoids_clustering():
    # All five high-score firings clustered in first 5 seconds
    arc = [_app("body-turn", i * 0.5, i * 0.5 + 0.3,
                intensity=0.9, confidence=0.9) for i in range(5)]
    moments = extract_key_moments(arc, [], duration_sec=60.0, n=3)
    # Should have picked some but not all from the cluster
    assert len(moments) == 3
    # The gap-relax fallback will still allow clustering if no spread possible,
    # but peaks should differ
    peaks = [m.peak_sec for m in moments]
    assert len(set(peaks)) == 3


# ─── render_analysis_bundle (mocked ffmpeg + brain renderer) ──────

def test_render_bundle_produces_expected_structure(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")

    arc = [
        _app("body-turn",   10, 12, intensity=0.9, confidence=0.9,
             evidence="pause + inward shift"),
        _app("recognition", 30, 32, intensity=0.8, confidence=0.9,
             evidence="familiar phrase"),
    ]
    seqs = [_seq("intimacy-deepening", 5, 25, ["body-turn"])]

    out_dir = tmp_path / "bundle"

    with patch("feeling_engine.rendering.analysis_assets._probe_duration",
               return_value=60.0), \
         patch("feeling_engine.rendering.analysis_assets.subprocess.run",
               return_value=MagicMock(stdout="", returncode=0)), \
         patch("feeling_engine.rendering.analysis_assets.render_mechanism_brain",
               return_value=tmp_path / "fake_brain.png"):
        bundle = render_analysis_bundle(
            video_path=video,
            arc=arc,
            out_dir=out_dir,
            sequences=seqs,
            n_key_moments=5,
            source_url="https://example.com/vid",
            verbose=False,
        )

    assert isinstance(bundle, AnalysisBundle)
    assert bundle.video_source == "https://example.com/vid"
    assert bundle.duration_sec == 60.0
    assert bundle.total_labels == 2
    assert bundle.total_sequences == 1
    assert len(bundle.key_moments) == 2  # only 2 apps, so ≤5 moments
    assert bundle.mechanism_counts == {"body-turn": 1, "recognition": 1}

    # analysis.json written
    assert (out_dir / "analysis.json").exists()

    # Relative paths populated on moments
    for km in bundle.key_moments:
        assert km["clip_path"].startswith("clips/moment_")
        assert km["still_path"].startswith("stills/moment_")
        assert km["brain_path"].startswith("brains/moment_")
