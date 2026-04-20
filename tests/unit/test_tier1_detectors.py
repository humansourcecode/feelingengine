"""Tier 1 detectors — absolute vs σ-mode parity tests.

σ-mode thresholds are a faithful port of absolute thresholds, calibrated
against the bundled Steve Jobs Stanford commencement clip so that σ-mode
detection on that content is preserved to within ~5% (allowing for
rounding of σ constants to 2 decimals). These tests pin that guarantee.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from feeling_engine.mechanisms.tier1_detectors import (
    detect_tier1,
    compute_axis_stats,
)

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).parent.parent.parent
JOBS_BUNDLE = REPO_ROOT / "examples" / "arcs" / "steve_jobs_death_pivot.json"


def _make_timeline(values_per_step: list[dict]) -> list:
    return [
        {"timestep": i, "categories": v}
        for i, v in enumerate(values_per_step)
    ]


def test_compute_axis_stats_shape():
    tl = _make_timeline([
        {"interoception": 0.0, "core_affect": 0.0, "regulation": 0.0,
         "reward": 0.0, "memory": 0.0, "social": 0.0, "language": 0.0},
        {"interoception": 0.1, "core_affect": 0.0, "regulation": 0.0,
         "reward": 0.0, "memory": 0.0, "social": 0.0, "language": 0.0},
    ])
    stats = compute_axis_stats(tl)
    assert set(stats) == {"val", "deriv"}
    assert set(stats["val"]) == {"interoception", "core_affect", "regulation",
                                  "reward", "memory", "social", "language"}
    assert stats["val"]["interoception"]["mean"] == 0.05
    # std of [0.0, 0.1] via statistics.stdev is ~0.0707
    assert abs(stats["val"]["interoception"]["std"] - 0.0707) < 0.001


def test_absolute_mode_unchanged_when_axis_stats_none():
    """Absolute mode must behave identically whether axis_stats is None or unset."""
    tl = _make_timeline([
        {"interoception": 0.05, "core_affect": 0.02, "regulation": 0.10,
         "reward": 0.02, "memory": 0.00, "social": 0.05, "language": 0.15},
        {"interoception": 0.12, "core_affect": 0.04, "regulation": 0.10,
         "reward": 0.04, "memory": 0.03, "social": 0.05, "language": 0.10},
        {"interoception": 0.15, "core_affect": 0.04, "regulation": 0.11,
         "reward": 0.05, "memory": 0.04, "social": 0.06, "language": 0.05},
    ])
    out_no_stats = detect_tier1(tl)
    out_none = detect_tier1(tl, axis_stats=None)
    assert Counter(a.label for a in out_no_stats) == Counter(a.label for a in out_none)


def test_sigma_mode_preserves_jobs_behavior():
    """Faithful-port guarantee: σ-mode on the Steve Jobs calibration clip
    matches absolute-mode within 5% total-application drift."""
    if not JOBS_BUNDLE.exists():
        import pytest
        pytest.skip(
            f"Jobs calibration bundle not found at {JOBS_BUNDLE}. "
            "This is shipped at examples/arcs/steve_jobs_death_pivot.json."
        )

    bundle = json.loads(JOBS_BUNDLE.read_text())
    profiles = bundle["tribe_profiles"]
    abs_out = detect_tier1(profiles)
    sig_out = detect_tier1(profiles, axis_stats=compute_axis_stats(profiles))

    drift = abs(len(sig_out) - len(abs_out)) / max(len(abs_out), 1)
    assert drift <= 0.05, (
        f"σ-mode drifted {drift:.1%} from absolute-mode on Jobs baseline — "
        f"exceeds 5% faithful-port tolerance. abs={len(abs_out)} σ={len(sig_out)}"
    )


def test_sigma_mode_unlocks_quieter_content():
    """On synthetic low-magnitude content, σ-mode should fire more than absolute.
    Absolute mode's thresholds miss events that are σ-significant within the video."""
    # Quiet signal with clear step at t=5 (10x std scale)
    steps = []
    for i in range(10):
        # low noise baseline
        base = {a: 0.001 * ((i + hash(a)) % 3) for a in
                ["interoception", "core_affect", "regulation",
                 "reward", "memory", "social", "language"]}
        if i == 5:
            # pronounced multi-axis change — small absolute, large σ
            base["interoception"] = 0.04
            base["core_affect"] = 0.03
            base["language"] = -0.03
        steps.append(base)
    tl = _make_timeline(steps)

    abs_out = detect_tier1(tl)
    sig_out = detect_tier1(tl, axis_stats=compute_axis_stats(tl))

    # At t=5 (the real activity moment), σ-mode should fire strictly more
    # activity mechanisms than absolute mode. Drift (quiescence) labels
    # don't count — we're testing detection of events, not silence.
    def activity_labels_at(out, t):
        return {a.label for a in out if a.start_sec == t and a.label != "drift"}

    abs_event = activity_labels_at(abs_out, 5)
    sig_event = activity_labels_at(sig_out, 5)
    assert len(sig_event) > len(abs_event), (
        f"σ-mode missed events absolute missed too: "
        f"abs t=5: {abs_event}  σ t=5: {sig_event}"
    )
    assert abs_event.issubset(sig_event), (
        "σ-mode should be a strict superset of absolute at the event moment"
    )
