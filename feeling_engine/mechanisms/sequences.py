"""Sequence matcher — detect narrative arc patterns in a mechanism stream.

Given a timeline of detected mechanism labels, find which named sequences
(joke-structure, tragic-arc, awakening-build, etc.) appear in the content.

Matching is order-preserving with tolerance for substitutions and skips.
"""
from __future__ import annotations

from typing import List
from feeling_engine.mechanisms.vocabulary import (
    LabelApplication, SequenceMatch, SEQUENCES
)


def _label_stream_by_second(applications: List[LabelApplication]) -> List[dict]:
    """Group label applications by second, with highest-intensity label dominant."""
    by_second: dict = {}
    for app in applications:
        t = int(app.start_sec)
        if t not in by_second:
            by_second[t] = []
        by_second[t].append(app)
    max_t = max(by_second.keys()) if by_second else 0
    stream = []
    for t in range(max_t + 1):
        apps = sorted(by_second.get(t, []), key=lambda a: -a.intensity)
        stream.append({
            "t": t,
            "labels": [a.label for a in apps],
            "top": apps[0].label if apps else None,
        })
    return stream


def detect_sequences(
    applications: List[LabelApplication],
    min_coverage: float = 1.0,
) -> List[SequenceMatch]:
    """Match each seed sequence against the detected label stream.

    Args:
        applications: list of LabelApplication (the detected mechanism timeline)
        min_coverage: fraction of pattern labels that must be found in order
                      (1.0 = all pattern labels; 0.75 = partial OK)

    Returns:
        List of SequenceMatch — one per detected sequence.
    """
    stream = _label_stream_by_second(applications)
    matches: List[SequenceMatch] = []

    for seq in SEQUENCES:
        positions: List[int] = []
        matched: List[str] = []
        search_from = 0
        for target in seq.pattern:
            found_at = None
            for idx in range(search_from, len(stream)):
                if target in stream[idx]["labels"]:
                    found_at = idx
                    break
            if found_at is not None:
                positions.append(found_at)
                matched.append(target)
                search_from = found_at + 1
            else:
                # allow skip if we're within flex tolerance
                if not seq.flex:
                    break
        coverage = len(positions) / len(seq.pattern)
        if coverage >= min_coverage:
            matches.append(SequenceMatch(
                name=seq.name,
                start_sec=float(positions[0]) if positions else 0.0,
                end_sec=float(positions[-1] + 1) if positions else 0.0,
                matched_labels=matched,
                positions=positions,
                partial=coverage < 1.0))

    return matches
