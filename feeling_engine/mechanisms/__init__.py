"""Feeling Engine — Mechanism Labels

Content-agnostic mechanism labels derived from TRIBE brain-state output
+ transcript analysis. Each label describes a structural move in content
(inward-pivot, vulnerability-transfer, universal-recognition, etc.) rather
than a feeling word.

Public API:

    from feeling_engine.mechanisms import detect_mechanisms, detect_sequences

    # Tier 1 only (requires TRIBE output)
    timeline = detect_mechanisms(
        tribe_categories=tribe_json,      # list[{timestep, categories:{...}}]
        transcript=transcript_json,        # optional — enables Tier 2/3
    )

    # Sequence arc detection (after mechanism labeling)
    arcs = detect_sequences(timeline)

See docs/mechanism_labels.md for the full vocabulary and methodology.
See docs/detector_validation.md for validation protocols.
"""

from feeling_engine.mechanisms.api import detect_mechanisms, detect_sequences
from feeling_engine.mechanisms.vocabulary import (
    MECHANISM_LABELS,
    SEQUENCES,
    LabelApplication,
    SequenceMatch,
)
from feeling_engine.mechanisms.prompts import (
    PROMPTS,
    get_prompts,
    get_interview_prompts,
    get_example_responses,
    all_mechanisms,
)

__all__ = [
    "detect_mechanisms",
    "detect_sequences",
    "MECHANISM_LABELS",
    "SEQUENCES",
    "LabelApplication",
    "SequenceMatch",
    "PROMPTS",
    "get_prompts",
    "get_interview_prompts",
    "get_example_responses",
    "all_mechanisms",
]
