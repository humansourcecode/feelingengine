"""Tier 2/3 detectors — TRIBE + transcript disambiguation.

These detectors require a transcript (word-level or just plain text) in
addition to the TRIBE timeline. They use linguistic markers to disambiguate
labels that share TRIBE signatures (e.g., intimacy-turn vs opposition).

Current implementation uses regex pattern matching on common linguistic
markers. A future upgrade would use an LLM adapter for richer semantic
analysis — the interface is designed so the pattern-matching backend can
be swapped for an LLM without changing callers.
"""
from __future__ import annotations

import re
from typing import List, Optional
from feeling_engine.mechanisms.vocabulary import LabelApplication


# ─── Linguistic markers ─────────────────────────────────────────
UNIVERSAL_MARKERS = [
    r"\bno one\b", r"\beveryone\b", r"\beverybody\b", r"\bnobody\b",
    r"\bwe all\b", r"\ball of us\b", r"\bevery\b", r"\ball share\b",
    r"\bhumans?\b(?=.*(?:all|always|never|must))",
]
TIME_LIMIT_MARKERS = [
    r"\btime is limited\b", r"\bbefore it.?s too late\b",
    r"\bsomeday\b", r"\bnot too long\b", r"\bcleared away\b",
    r"\bwhile you\b", r"\blimited\b", r"\brunning out\b",
    r"\bwe (?:all )?must die\b", r"\bwill (?:die|end|fade)\b",
]
INTIMACY_MARKERS = [
    r"\btogether\b", r"\bwe\b", r"\bshare\b", r"\binner voice\b",
    r"\byour heart\b", r"\byour own\b", r"\bmy (?:love|dear)\b",
    r"\bbetween us\b", r"\bconfide\b", r"\btrust\b",
]
OPPOSITION_MARKERS = [
    # Require structural contradiction, not bare negation.
    # Sentence-initial contrast ("But that's..." at the start of a sentence)
    r"(?:^|[.!?]\s+)but\b",
    r"(?:^|[.!?]\s+)however,",
    # Explicit contest verbs
    r"\brefuse to\b", r"\breject\b", r"\bdefy\b", r"\bdeny\b",
    r"\bopposed? to\b", r"\bagainst (?:what|the|him|her|them|you)\b",
    # Position-contest frames
    r"\byou (?:say|claim|think) .{0,40} but\b",
    r"\bthey (?:say|claim|think) .{0,40} but\b",
    # Corrective framing
    r"\bthat.?s (?:not (?:true|right)|wrong)\b",
    r"\bI disagree\b",
]
CONTEMPLATION_MARKERS = [
    r"\bperhaps\b", r"\blikely\b", r"\bmaybe\b", r"\bmight\b",
    r"\bsort of\b", r"\bin a way\b", r"\bsome (?:say|argue|claim)\b",
    r"\bit seems\b", r"\bone could\b", r"\bquestion (?:of|whether)\b",
]
HEDGE_MARKERS = [
    r"\bis as it should\b", r"\bvery likely\b", r"\bquite\b",
    r"\barguably\b", r"\bin some sense\b",
]
SCALE_MARKERS = [  # for expansion
    r"\bvast\b", r"\benormous\b", r"\bbeyond\b", r"\binfinite\b",
    r"\buniverse\b", r"\bcosmic\b", r"\bbillions?\b", r"\btrillions?\b",
    r"\blight.?years?\b", r"\bendless\b", r"\bincomprehensible\b",
]
SELF_PROTECT_MARKERS = [  # for withdrawal
    r"\bI shouldn.?t have\b", r"\bI wish I hadn.?t\b",
    r"\bforget I said\b", r"\bnever mind\b", r"\bnothing\b",
    r"\bI don.?t want to\b", r"\bpulled back\b",
]
BOUNDARY_MARKERS = [  # for boundary-establish
    r"\bthat.?s not (?:who|what) I\b", r"\bwe don.?t\b",
    r"\bthis line\b", r"\bI won.?t\b", r"\bnever\b",
]


def _any_match(patterns: List[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _words_in_window(
    transcript: Optional[dict],
    start_sec: float,
    end_sec: float,
    pad: float = 1.5,
) -> str:
    """Return lowercase text covering [start-pad, end+pad]."""
    if not transcript or "words" not in transcript:
        return ""
    ws = transcript["words"]
    return " ".join(
        w["word"] for w in ws
        if w.get("start", 0) >= start_sec - pad and w.get("end", 0) < end_sec + pad
    ).lower()


def detect_tier23_text_only(text: str) -> List[LabelApplication]:
    """Text-only linguistic detection — no TRIBE input required.

    Returns commercially-usable label applications. Lower confidence than
    the TRIBE-grounded version because there's no brain-signature gating.
    Used when the pipeline is in 'text-only' mode (see api.detect_mechanisms).

    Fires labels purely on linguistic markers. Produces per-line detection
    (not per-second), with start/end indicating line number.
    """
    out: List[LabelApplication] = []
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for idx, line in enumerate(lines):
        low = line.lower()

        def add(name: str, tier: int, intensity: float, confidence: float, markers: str):
            out.append(LabelApplication(
                label=name, tier=tier,
                start_sec=float(idx), end_sec=float(idx + 1),
                intensity=max(0.0, min(1.0, intensity)),
                confidence=confidence,
                signals={"matched_markers": markers, "line": idx + 1,
                         "snippet_6w": " ".join(line.split()[:6])}))

        if _any_match(UNIVERSAL_MARKERS, low):
            add("universal-recognition", 2, 0.7, 0.70, "universal")
        if _any_match(TIME_LIMIT_MARKERS, low):
            add("stakes-compression", 2, 0.7, 0.70, "time_limit")
        if _any_match(INTIMACY_MARKERS, low) and not _any_match([r"\bbut\b"], low):
            add("intimacy-turn", 2, 0.6, 0.65, "intimacy")
        if _any_match(OPPOSITION_MARKERS, low):
            add("opposition", 2, 0.55, 0.60, "opposition")
        if _any_match(CONTEMPLATION_MARKERS, low) or _any_match(HEDGE_MARKERS, low):
            add("contemplation", 2, 0.65, 0.70, "contemplation")
        if _any_match(SCALE_MARKERS, low):
            add("expansion", 3, 0.6, 0.55, "scale")
        if _any_match(SELF_PROTECT_MARKERS, low):
            add("withdrawal", 3, 0.55, 0.55, "self_protect")
        if _any_match(BOUNDARY_MARKERS, low):
            add("boundary-establish", 3, 0.5, 0.50, "boundary")

    return out


def detect_tier23(
    timeline: list,
    transcript: Optional[dict] = None,
    text: Optional[str] = None,
) -> List[LabelApplication]:
    """Apply Tier 2/3 disambiguation using transcript/text.

    Args:
        timeline: TRIBE timeline (list of {timestep, categories})
        transcript: word-level transcript with {words:[{word,start,end}]}
                    enables per-second disambiguation
        text: plain text fallback if word-level not available
              (whole-text matching, no temporal resolution)

    Returns:
        List of LabelApplication for Tier 2/3 labels.
    """
    out: List[LabelApplication] = []

    for i, step in enumerate(timeline):
        cats = step["categories"]
        t = step.get("timestep", i)

        def deriv(attr: str) -> float:
            if i == 0:
                return 0.0
            return cats[attr] - timeline[i - 1]["categories"][attr]

        window_text = _words_in_window(transcript, t, t + 1, pad=2.0) if transcript else (text or "")

        def add(name: str, tier: int, intensity: float, confidence: float, signals: dict):
            out.append(LabelApplication(
                label=name, tier=tier,
                start_sec=float(t), end_sec=float(t + 1),
                intensity=max(0.0, min(1.0, intensity)),
                confidence=confidence,
                signals=signals))

        # universal-recognition (Tier 2)
        if (deriv("memory") > 0.02 or deriv("core_affect") > 0.02) \
           and _any_match(UNIVERSAL_MARKERS, window_text):
            add("universal-recognition", 2, 0.8, 0.85,
                {"matched_markers": "universal"})

        # stakes-compression (Tier 2)
        if _any_match(TIME_LIMIT_MARKERS, window_text) \
           and (cats["regulation"] > 0.08 or deriv("reward") > 0.02):
            add("stakes-compression", 2, 0.75, 0.80,
                {"matched_markers": "time_limit"})

        # intimacy-turn (Tier 2)
        if deriv("social") > 0.02 \
           and _any_match(INTIMACY_MARKERS, window_text) \
           and not _any_match([r"\bbut\b", r"\bhowever\b", r"\bnot\b"], window_text):
            add("intimacy-turn", 2, min(1.0, (cats["social"] + 0.1) * 3), 0.75,
                {"matched_markers": "intimacy"})

        # opposition (Tier 2)
        if deriv("social") > 0.02 \
           and cats["core_affect"] < -0.02 \
           and _any_match(OPPOSITION_MARKERS, window_text):
            add("opposition", 2, 0.6, 0.60,
                {"matched_markers": "opposition"})

        # contemplation (Tier 2)
        if cats["language"] > 0.15 \
           and (_any_match(CONTEMPLATION_MARKERS, window_text) or _any_match(HEDGE_MARKERS, window_text)):
            add("contemplation", 2, cats["language"] * 3, 0.75,
                {"matched_markers": "contemplation"})

        # expansion (Tier 3)
        if cats["core_affect"] > 0.05 and _any_match(SCALE_MARKERS, window_text):
            add("expansion", 3, 0.65, 0.55,
                {"matched_markers": "scale"})

        # withdrawal (Tier 3)
        if cats["regulation"] > 0.08 and cats["interoception"] > 0.05 \
           and cats["social"] < -0.05 \
           and _any_match(SELF_PROTECT_MARKERS, window_text):
            add("withdrawal", 3, 0.6, 0.55,
                {"matched_markers": "self_protect"})

        # boundary-establish (Tier 3)
        if _any_match(BOUNDARY_MARKERS, window_text) and cats["social"] > 0.02:
            add("boundary-establish", 3, 0.55, 0.50,
                {"matched_markers": "boundary"})

    return out
