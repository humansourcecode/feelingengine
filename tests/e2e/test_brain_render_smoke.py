"""Brain renderer smoke test — opt-in (slow; downloads fsaverage first run, ~30-50s per render).

Activation: pytest tests/e2e/test_brain_render_smoke.py --run-gemini
(Uses the same --run-gemini flag since this is a slow-but-local opt-in;
no external API cost. Keeping under one flag for simplicity.)
"""
from __future__ import annotations

import pytest
from pathlib import Path


@pytest.mark.gemini  # Reuse the opt-in flag for slow local tests
def test_render_produces_png():
    from feeling_engine.rendering.brain_renderer import render_mechanism_brain, ANNOT_LH
    if not ANNOT_LH.exists():
        pytest.skip("HCP-MMP annotation file not available")

    out_dir = Path(__file__).parent / "fixtures" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smoke_body_turn.png"

    result = render_mechanism_brain(
        label="body-turn",
        output_path=out_path,
        view="lateral_left",  # single panel = fastest
        title="body-turn (smoke test)",
    )

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 5_000  # Not a blank/corrupt file
