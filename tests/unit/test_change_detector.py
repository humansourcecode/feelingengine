"""Layer 2 — Change detector tests."""
from feeling_engine.adapters.brain_model.base import BrainRegionProfile, BrainTimeSeries
from feeling_engine.translator.change_detector import detect_changes, TransitionType


def _flat_series(n: int = 10):
    """A flat baseline series — no change points should be found."""
    profiles = [
        BrainRegionProfile(
            categories={"interoception": 0.01, "language": 0.01},
            timestep=i,
        )
        for i in range(n)
    ]
    return BrainTimeSeries(
        profiles=profiles,
        n_timesteps=n,
        category_names=["interoception", "language"],
    )


def _spiking_series(spike_at: int = 5, n: int = 10):
    """A series with a single large spike in interoception at `spike_at`."""
    profiles = []
    for i in range(n):
        interoception = 0.5 if i == spike_at else 0.01
        profiles.append(BrainRegionProfile(
            categories={"interoception": interoception, "language": 0.01},
            timestep=i,
        ))
    return BrainTimeSeries(
        profiles=profiles,
        n_timesteps=n,
        category_names=["interoception", "language"],
    )


def test_flat_series_produces_no_change_points():
    analysis = detect_changes(_flat_series(), threshold=0.08)
    assert analysis.change_points == []


def test_spike_is_detected():
    analysis = detect_changes(_spiking_series(spike_at=5), threshold=0.08)
    # Both the spike (t=5) and the fall back (t=6) should be detected
    assert len(analysis.change_points) >= 2
    interoception_cps = [cp for cp in analysis.change_points if cp.category == "interoception"]
    assert any(cp.timestep == 5 and cp.direction == "rising" for cp in interoception_cps)
    assert any(cp.timestep == 6 and cp.direction == "falling" for cp in interoception_cps)


def test_threshold_filters_small_changes():
    """A change below threshold should not produce a change point."""
    profiles = [
        BrainRegionProfile(categories={"a": 0.01}, timestep=0),
        BrainRegionProfile(categories={"a": 0.04}, timestep=1),  # delta 0.03, below 0.08 threshold
    ]
    bts = BrainTimeSeries(profiles=profiles, n_timesteps=2, category_names=["a"])
    analysis = detect_changes(bts, threshold=0.08)
    assert analysis.change_points == []


def test_transition_types_are_enum_instances():
    """Transition types should be proper TransitionType values."""
    analysis = detect_changes(_spiking_series(), threshold=0.08)
    for cp in analysis.change_points:
        assert isinstance(cp.transition_type, TransitionType)
