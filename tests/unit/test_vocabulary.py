"""Vocabulary file sanity checks."""
from pathlib import Path

import yaml


VOCAB_PATH = Path(__file__).parent.parent.parent / "feeling_engine" / "translator" / "vocabulary.yaml"


def test_vocabulary_file_exists():
    assert VOCAB_PATH.exists(), f"Vocabulary not found at {VOCAB_PATH}"


def test_vocabulary_has_schema_and_terms():
    data = yaml.safe_load(VOCAB_PATH.read_text())
    assert "schema_version" in data
    assert "terms" in data
    assert isinstance(data["terms"], dict)


def test_vocabulary_size_meets_minimum():
    data = yaml.safe_load(VOCAB_PATH.read_text())
    # Published spec: 60 brain-grounded terms
    assert len(data["terms"]) >= 50


def test_every_term_has_required_dimensions():
    data = yaml.safe_load(VOCAB_PATH.read_text())
    required = {"valence", "arousal", "body_focus", "brain_expectations"}
    for term, spec in data["terms"].items():
        missing = required - set(spec.keys())
        assert not missing, f"{term} missing: {missing}"


def test_dimension_ranges_are_sane():
    data = yaml.safe_load(VOCAB_PATH.read_text())
    for term, spec in data["terms"].items():
        for dim in ("valence", "arousal", "body_focus"):
            val = spec[dim]
            assert -1.0 <= val <= 1.0, f"{term}.{dim}={val} out of [-1,1]"


def test_mortality_awareness_is_body_dominant():
    """Specific term spec-check: mortality_awareness must be interoception-heavy."""
    data = yaml.safe_load(VOCAB_PATH.read_text())
    assert "mortality_awareness" in data["terms"]
    ma = data["terms"]["mortality_awareness"]
    assert ma["body_focus"] > 0.5
    assert ma["brain_expectations"].get("interoception") in ("high", "very_high")
