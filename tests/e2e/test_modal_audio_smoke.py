"""Modal/TRIBE audio smoke test — opt-in, low-cost.

Runs a tiny synthetic audio file (~3 seconds, ~90KB) through the Modal
TRIBE adapter and verifies the shape of the returned prediction. This is
NOT a correctness test — the audio is three sine-wave tones, so the brain
predictions will be meaningless. What it validates:

  - Modal authentication + deployment are live
  - `ModalTRIBEAdapter.predict_from_file()` returns a prediction with
    a non-empty `metadata["profiles"]` list of the right shape
  - The pipeline doesn't crash on non-speech audio

Cost per run: ~$0.03-0.05 (short audio + container startup).
Activation: `pytest tests/e2e/test_modal_audio_smoke.py --run-modal`

Marker `modal` is separate from `e2e` so you can opt into Modal-hitting
tests independently of the other live-API tests.
"""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "hello_world.wav"


@pytest.mark.modal
def test_modal_audio_smoke():
    """End-to-end: send a tiny audio file to the deployed TRIBE function,
    verify we get back a prediction with the expected shape."""
    if not FIXTURE.exists():
        pytest.skip(f"fixture missing: {FIXTURE}")

    # Import here so this module can be collected even when modal isn't installed
    try:
        from feeling_engine.adapters.compute.modal_tribe import ModalTRIBEAdapter
    except ImportError as e:
        pytest.skip(f"Modal adapter not installable: {e}")

    adapter = ModalTRIBEAdapter()
    prediction = adapter.predict_from_file(FIXTURE)

    # Prediction object has the expected attributes
    assert prediction is not None, "adapter returned None"
    assert hasattr(prediction, "metadata"), "prediction missing .metadata"

    profiles = prediction.metadata.get("profiles")
    assert profiles is not None, "prediction.metadata['profiles'] missing"
    assert isinstance(profiles, list), "profiles should be a list"
    assert len(profiles) > 0, "profiles list is empty"

    # Each profile is {timestep, categories: {7 axes}}
    first = profiles[0]
    assert "timestep" in first, "profile missing 'timestep'"
    assert "categories" in first, "profile missing 'categories'"
    expected_axes = {
        "interoception", "core_affect", "regulation", "reward",
        "memory", "social", "language",
    }
    actual_axes = set(first["categories"].keys())
    assert expected_axes.issubset(actual_axes), (
        f"categories missing axes: {expected_axes - actual_axes}"
    )

    # The fixture is ~3 seconds, TRIBE produces 1 timestep/sec — give slack
    assert 2 <= len(profiles) <= 10, (
        f"unexpected profile count for 3s audio: {len(profiles)}"
    )
