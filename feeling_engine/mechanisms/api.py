"""Public API for mechanism detection.

Two modes:

    # FULL mode — requires TRIBE output + transcript (or plain text).
    # Highest fidelity. Output is NON-COMMERCIAL only (TRIBE is CC BY-NC 4.0).
    timeline = detect_mechanisms(
        tribe_categories=tribe_json,
        transcript=transcript_json,
    )

    # TEXT-ONLY mode — no TRIBE required. Commercially usable output.
    # Lower fidelity (Tier 1 labels unavailable; only Tier 2/3 linguistic).
    timeline = detect_mechanisms(
        text=content_text,
        mode="text-only",
    )

    # Sequence detection works identically for both modes.
    arcs = detect_sequences(timeline)

Each LabelApplication is tagged with `tribe_grounded` (True/False) so
downstream code can filter by commercial usability.

See docs/mechanism_labels.md for the vocabulary; docs/detector_validation.md
for validation protocols. See the "Commercial Use" section below for
licensing guidance.

## Commercial Use

- **FULL mode output is CC BY-NC 4.0 derivative.** Use in research, free
  educational content, non-monetized channels only. Do NOT use in
  ad-supported videos, paid products, consulting deliverables, or SaaS.
- **TEXT-ONLY mode output is unrestricted.** Uses only your own IP
  (Feeling Engine vocabulary + detectors + your input text). Safe for
  any commercial use.
- When in doubt, prefer text-only mode for any output that might be
  monetized.
"""
from __future__ import annotations

from typing import List, Optional, Literal
from feeling_engine.mechanisms.vocabulary import LabelApplication, SequenceMatch
from feeling_engine.mechanisms.tier1_detectors import detect_tier1, DEFAULT_THRESHOLDS
from feeling_engine.mechanisms.tier2_detectors import detect_tier23, detect_tier23_text_only
from feeling_engine.mechanisms.sequences import detect_sequences as _detect_sequences


Mode = Literal["full", "text-only"]


def detect_mechanisms(
    tribe_categories: Optional[list] = None,
    transcript: Optional[dict] = None,
    text: Optional[str] = None,
    mode: Optional[Mode] = None,
    thresholds: Optional[dict] = None,
    include_tier23: bool = True,
    axis_stats: Optional[dict] = None,
) -> List[LabelApplication]:
    """Detect mechanism labels on content.

    Args:
        tribe_categories: TRIBE timeline (list of {timestep, categories}).
            Required for full mode. If None, switches to text-only mode.
        transcript: word-level transcript {"words":[{"word","start","end"}]}.
            Used for per-second Tier 2/3 disambiguation in full mode.
        text: plain text of the content. Used for text-only mode.
        mode: "full" (requires TRIBE) or "text-only". If not set, inferred
            from inputs: TRIBE provided → full; else text-only.
        thresholds: optional Tier 1 threshold overrides (full mode only).
        include_tier23: if False, only Tier 1 runs (full mode only).
        axis_stats: optional output of tier1_detectors.compute_axis_stats().
            If provided, Tier 1 runs in σ-mode (per-video z-normalized
            thresholds) instead of absolute mode. See Decisions #67, #69.

    Returns:
        List of LabelApplication sorted by (start_sec, -intensity).
        Each application is tagged with `tribe_grounded` (True/False) in
        its signals dict for downstream commercial-use filtering.

    Modes:
        FULL (TRIBE + text) — all 28 labels available. Output is
            non-commercial only (CC BY-NC 4.0, derivative of TRIBE).
        TEXT-ONLY (no TRIBE) — Tier 2/3 linguistic labels only
            (~8 labels). Output is commercially usable.
    """
    # Infer mode
    if mode is None:
        mode = "full" if tribe_categories is not None else "text-only"

    if mode == "full":
        if tribe_categories is None:
            raise ValueError("full mode requires tribe_categories")
        return _detect_full(tribe_categories, transcript, thresholds,
                            include_tier23, axis_stats)
    elif mode == "text-only":
        content_text = text
        if content_text is None and transcript and "words" in transcript:
            content_text = " ".join(w["word"] for w in transcript["words"])
        if not content_text:
            raise ValueError("text-only mode requires text or transcript with words")
        return _detect_text_only(content_text)
    else:
        raise ValueError(f"unknown mode: {mode}")


def _detect_full(
    tribe_categories: list,
    transcript: Optional[dict],
    thresholds: Optional[dict],
    include_tier23: bool,
    axis_stats: Optional[dict] = None,
) -> List[LabelApplication]:
    t1 = detect_tier1(tribe_categories, thresholds=thresholds, axis_stats=axis_stats)
    for app in t1:
        app.signals["tribe_grounded"] = True
    results = list(t1)

    if include_tier23 and (transcript or True):
        t23 = detect_tier23(tribe_categories, transcript=transcript)
        for app in t23:
            app.signals["tribe_grounded"] = True
        results.extend(t23)

    results.sort(key=lambda a: (a.start_sec, -a.intensity))
    return results


def _detect_text_only(text: str) -> List[LabelApplication]:
    """Text-only detection — no TRIBE input. Commercially usable output."""
    results = detect_tier23_text_only(text)
    for app in results:
        app.signals["tribe_grounded"] = False
    results.sort(key=lambda a: (a.start_sec, -a.intensity))
    return results


def detect_sequences(
    applications: List[LabelApplication],
    min_coverage: float = 1.0,
) -> List[SequenceMatch]:
    """Match named narrative sequences against a mechanism timeline.

    Args:
        applications: output from detect_mechanisms()
        min_coverage: fraction of pattern labels required (1.0 = all;
            0.75 allows partial matches)

    Returns:
        List of SequenceMatch.
    """
    return _detect_sequences(applications, min_coverage=min_coverage)
