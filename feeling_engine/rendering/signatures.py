"""Canonical activation signatures per mechanism label.

For each of the 28 mechanism labels, defines the expected activation pattern
across the 7 emotional brain-region categories:

    interoception, regulation, core_affect, social, reward, memory, language

Values are in [-1.0, +1.0]. Sign indicates direction (+ = elevated activation,
- = suppressed). Magnitude indicates how distinctive that axis is for this label.

These signatures are OUR IP. They're derived from:
  - The TRIBE-signature fields in vocabulary.py (phenomenological descriptions)
  - Tier 1 detector logic (which axes the detectors read for each label)
  - Emotion-neuroscience literature (Lindquist, Barrett, Satpute)

They are the synthetic baseline. When the private arc library grows enough
that empirical per-label signatures can be aggregated from real TRIBE mining,
those will override these. Until then, these carry the full load.

Only categories that meaningfully differentiate the label are listed.
Unlisted categories are implicit zero.
"""
from __future__ import annotations

from typing import Dict


CATEGORIES = (
    "interoception", "regulation", "core_affect", "social",
    "reward", "memory", "language",
)


MECHANISM_SIGNATURE: Dict[str, Dict[str, float]] = {
    # ── I. Interoception ─────────────────────────────────────
    "body-turn":       {"interoception": +0.80, "language": -0.40, "regulation": +0.20},
    "body-surge":      {"interoception": +0.90, "core_affect": +0.60},
    "body-anchor":     {"interoception": +0.70, "regulation": +0.30, "language": -0.30},
    "sensation-flood": {"interoception": +0.85, "core_affect": +0.75, "reward": +0.40},

    # ── II. Core affect ─────────────────────────────────────
    "affect-rise":  {"core_affect": +0.85, "interoception": +0.40},
    "affect-fade":  {"core_affect": -0.50, "regulation": +0.30},
    "restraint":    {"regulation": +0.80, "core_affect": -0.30, "language": +0.20},
    "release":      {"core_affect": +0.80, "interoception": +0.50, "reward": +0.40,
                     "regulation": -0.30},

    # ── III. Threshold / transition (Tier 2/3) ──────────────
    "threshold-approach": {"regulation": +0.60, "core_affect": +0.50,
                           "interoception": +0.40},
    "withdrawal":         {"regulation": +0.70, "social": -0.40,
                           "interoception": -0.30},

    # ── IV. Anticipation & satisfaction ─────────────────────
    "anticipation":      {"reward": +0.70, "regulation": +0.50, "core_affect": +0.40},
    "satisfaction-peak": {"reward": +0.90, "core_affect": +0.60, "interoception": +0.40},

    # ── V. Recognition / evocation ──────────────────────────
    "recognition":           {"memory": +0.75, "social": +0.45, "language": +0.30},
    "evocation":             {"memory": +0.80, "core_affect": +0.50},
    "universal-recognition": {"memory": +0.70, "social": +0.65, "language": +0.40},

    # ── VI. Social / interpersonal ──────────────────────────
    "intimacy-turn":          {"social": +0.85, "interoception": +0.50, "language": -0.20},
    "opposition":             {"social": +0.50, "regulation": +0.70, "core_affect": +0.40},
    "vulnerability-transfer": {"social": +0.80, "interoception": +0.60, "core_affect": +0.50},
    "boundary-establish":     {"regulation": +0.70, "social": +0.40, "language": +0.40},

    # ── VII. Language dynamics ──────────────────────────────
    "word-focus":    {"language": +0.80, "regulation": +0.30},
    "word-recede":   {"language": -0.60, "interoception": +0.40},
    "contemplation": {"regulation": +0.60, "language": +0.40, "memory": +0.40},

    # ── VIII. Structural / rhythmic ─────────────────────────
    "inward-pivot":       {"interoception": +0.60, "regulation": +0.50, "language": -0.20},
    "pattern-break":      {"core_affect": +0.70, "interoception": +0.60, "regulation": +0.40},
    "stakes-compression": {"regulation": +0.60, "core_affect": +0.60, "language": +0.30},
    "expansion":          {"social": +0.40, "memory": +0.50, "regulation": +0.40,
                           "core_affect": +0.40},

    # ── IX. Attentional / dissonance ────────────────────────
    "drift":      {"regulation": -0.40, "language": -0.40, "interoception": -0.20},
    "dissonance": {"regulation": +0.60, "core_affect": +0.40, "language": -0.30},
}


def get_signature(label: str) -> Dict[str, float]:
    """Return the canonical 7-category signature for a mechanism label.

    Raises KeyError if the label is unknown. Returned dict contains only
    the categories that differentiate the label — unlisted categories are
    implicit zero.
    """
    if label not in MECHANISM_SIGNATURE:
        raise KeyError(f"unknown mechanism label: {label!r}")
    return dict(MECHANISM_SIGNATURE[label])


def signature_as_vector(label: str) -> Dict[str, float]:
    """Return the full 7-category vector for a label (zeros for unlisted cats)."""
    sig = get_signature(label)
    return {cat: float(sig.get(cat, 0.0)) for cat in CATEGORIES}
