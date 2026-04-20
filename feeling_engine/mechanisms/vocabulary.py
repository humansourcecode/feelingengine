"""Mechanism label vocabulary — data definitions.

28 content-agnostic mechanism labels grouped by TRIBE axis coverage.
Each label has a tier (1, 2, 3), TRIBE signature, phenomenological description,
and optional detector spec for Tier 2/3 disambiguation.

Also defines 10 seed narrative sequences composed of mechanism chains.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LabelApplication:
    """One detected mechanism at a specific moment in content."""
    label: str
    tier: int                  # 1, 2, or 3
    start_sec: float
    end_sec: float
    intensity: float           # 0.0 - 1.0
    confidence: float          # 0.0 - 1.0
    signals: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class SequenceMatch:
    """One detected narrative sequence arc."""
    name: str
    start_sec: float
    end_sec: float
    matched_labels: List[str]     # the mechanism chain that triggered this sequence
    positions: List[int]          # seconds where each pattern step was matched
    partial: bool                 # true if some pattern steps were substituted or skipped


@dataclass
class MechanismSpec:
    """Definition of a single mechanism label."""
    name: str
    tier: int
    tribe_signature: str          # human-readable
    phenomenology: str
    exemplars: List[str]
    detector_spec: Optional[dict] = None


# ─── The 28 mechanism labels ──────────────────────────────────────

MECHANISM_LABELS: List[MechanismSpec] = [
    # ─── I. Interoception ───
    MechanismSpec("body-turn", 1,
        "interoception rising, language dropping",
        "attention drops into the body — breath catches, words recede",
        ["Jobs t=59", "Keats 'My heart aches'", "first-touch intimate scene"]),
    MechanismSpec("body-surge", 1,
        "interoception sudden spike from low baseline",
        "something physical moves through — heart lifts or drops, shock lands",
        ["Ballou 'last breath on battlefield'", "horror jump scare", "'I love you' reveal"]),
    MechanismSpec("body-anchor", 1,
        "interoception sustained high + other axes quiet",
        "sustained body presence — breath slow, nothing else competes",
        ["guided meditation", "slow-mo sports", "silence after news breaks"]),
    MechanismSpec("sensation-flood", 1,
        "interoception + core_affect both sustained high",
        "the body is filled — sensory overwhelm, more than can be processed",
        ["concert climax", "first bite of extraordinary food", "erotic sensory scene"]),

    # ─── II. Core affect ───
    MechanismSpec("affect-rise", 1,
        "|core_affect| magnitude increasing",
        "feeling thickens — the moment gets more charged",
        ["Keats build to 'easeful Death'", "escalating argument", "crowd roaring"]),
    MechanismSpec("affect-fade", 1,
        "core_affect returning toward baseline after peak",
        "the storm passes — settling, quieter",
        ["post-climax stillness", "end of joke laughter", "Frost 'miles to go before I sleep'"]),

    # ─── III. Regulation ───
    MechanismSpec("restraint", 1,
        "regulation sustained high + core_affect elevated but not released",
        "something withheld — pressure that doesn't let out",
        ["holding back tears at toast", "composed under attack", "meditation effort"]),
    MechanismSpec("release", 1,
        "regulation sudden drop",
        "something let go — unguarded, an opening",
        ["laughter breaking tension", "tears finally coming", "'let me be honest' transition"]),
    MechanismSpec("threshold-approach", 2,
        "regulation rising + reward rising simultaneously",
        "something about to turn — tension winding, leaning in without yet arriving",
        ["speech building to reveal", "musical pre-drop", "pause before confession"],
        detector_spec={
            "shared_with": ["anticipation"],
            "source": "transcript",
            "signals": ["tension-building language without reward-promise",
                       "cadence markers (short sentences, rising pitch)",
                       "explicit delay markers"],
            "failure_modes": ["slow anticipation reads as threshold-approach"]}),
    MechanismSpec("withdrawal", 3,
        "regulation + interoception rising, social-axis suppressed",
        "shrinking inward — body contracts in self-protection",
        ["public embarrassment", "post-caught shame", "pulling back from vulnerability"],
        detector_spec={
            "shared_with": ["restraint"],
            "source": "transcript + social-axis direction",
            "signals": ["self-protective language", "subject withdrawing from disclosure",
                       "social-axis suppression (not sustain)"],
            "failure_modes": ["quiet reflection looks like withdrawal"]}),

    # ─── IV. Reward ───
    MechanismSpec("anticipation", 1,
        "reward rising + regulation steady",
        "leaning forward — wanting what's next, attention sharpens",
        ["cliffhanger", "'let me tell you a story'", "riddle with deferred answer"]),
    MechanismSpec("satisfaction-peak", 1,
        "reward peaks then slight fall, with prominence",
        "a click — the answer lands, a small rest",
        ["comedic punchline", "mystery solution", "resolution chord"]),

    # ─── V. Memory ───
    MechanismSpec("recognition", 1,
        "memory rising, core_affect modest",
        "wait — I know this — familiarity before full context",
        ["quoted famous line", "musical motif return", "'remember when...'"]),
    MechanismSpec("evocation", 1,
        "memory + core_affect rising together",
        "a scene returns — not just memory, feeling again",
        ["smell-triggered memory", "'I was seventeen and...'", "revisiting old neighborhood"]),
    MechanismSpec("universal-recognition", 2,
        "interoception + memory + social rising together",
        "oh — that's true — body confirms what the words claim, no argument needed",
        ["'No one wants to die'", "'we all want to be loved'", "'the light goes out'"],
        detector_spec={
            "shared_with": ["evocation"],
            "source": "transcript",
            "signals": ["universal claim markers (everyone, no one, we all, every body)",
                       "content unarguable at body level"],
            "failure_modes": ["personal claim in universal terms mis-fires"]}),

    # ─── VI. Social ───
    MechanismSpec("intimacy-turn", 2,
        "social rising + regulation dropping",
        "something soft — room narrows, distance closes",
        ["'never told anyone'", "lean-in before reveal", "held eye contact"],
        detector_spec={
            "shared_with": ["opposition"],
            "source": "transcript",
            "signals": ["positive affect valence", "affiliative language (we, together, share)",
                       "softening markers"],
            "failure_modes": ["dark intimacy (shared grief) can read as opposition"]}),
    MechanismSpec("opposition", 2,
        "social rising + regulation elevated + core_affect negative trend",
        "positions collide — I am against you, air tightens",
        ["political debate", "escalating argument", "'but you say...' rhetoric"],
        detector_spec={
            "shared_with": ["intimacy-turn"],
            "source": "transcript",
            "signals": ["contradiction markers (but, no, however)",
                       "position-marking (I think, my view)",
                       "contested-other framing"],
            "failure_modes": ["ironic opposition (friends roasting) reads as real opposition"]}),
    MechanismSpec("vulnerability-transfer", 1,
        "social + interoception rising together",
        "body opens because theirs is open — catching what speaker is risking",
        ["Jobs 'I was 17 years old'", "Ballou letter", "therapy breakthrough"]),
    MechanismSpec("boundary-establish", 3,
        "social axis activates with self-other sharpening (signature not yet fully operationalized)",
        "a wall goes up — line drawn, position stated",
        ["'that's not who I am'", "parental refusal", "organizational 'we don't do that here'"],
        detector_spec={
            "shared_with": ["opposition"],
            "source": "transcript + contextual framing",
            "signals": ["distinction language (I'm not that, we don't, this line)",
                       "refusal/negation patterns",
                       "position-declarative syntax"],
            "failure_modes": ["any refusal can be misread; distinguishing self-definition from contest is hard"]}),

    # ─── VII. Language ───
    MechanismSpec("word-focus", 1,
        "language high + interoception + core_affect low",
        "all attention on the words — following the thread, thinking not feeling",
        ["technical lecture", "news anchor", "syllogism"]),
    MechanismSpec("word-recede", 1,
        "language dropping while other dimensions rise",
        "the words stop mattering — what's said is less than what's felt",
        ["Jobs death pivot", "Beethoven silent movement", "moment before a kiss"]),
    MechanismSpec("contemplation", 2,
        "language sustained high + memory rising + regulation high",
        "sustained meaning-sift — thinking through something, integrating, weighing",
        ["philosophical voiceover", "character wrestling with decision", "meditation teacher explaining"],
        detector_spec={
            "shared_with": ["word-focus"],
            "source": "transcript + temporal",
            "signals": ["content complexity (abstract vs concrete)",
                       "hedging markers (perhaps, on the other hand)",
                       "longer pauses"],
            "failure_modes": ["slow technical explanation mis-fires"]}),

    # ─── VIII. Multi-axis ───
    MechanismSpec("inward-pivot", 1,
        "language drops + interoception rises",
        "external → internal — world falls away, it's about what's happening in me",
        ["Jobs death pivot", "Keats 'fade far away, dissolve'", "meditation transition"]),
    MechanismSpec("pattern-break", 1,
        "sudden shift across multiple axes simultaneously",
        "wait — what just happened — ground moved, attention redeploys",
        ["rhetorical turn mid-sentence", "dissonant note", "unexpected scene cut"]),
    MechanismSpec("stakes-compression", 2,
        "regulation + reward shift + core_affect rising",
        "time is shorter than I thought — NOW matters, no more later",
        ["'your time is limited'", "terminal diagnosis scene", "'before it's too late' framing"],
        detector_spec={
            "shared_with": ["pattern-break"],
            "source": "transcript",
            "signals": ["time-horizon-collapsing language (time is, before it's too late, while you still can)",
                       "explicit finitude markers",
                       "urgency framing"],
            "failure_modes": ["rhetorical urgency mis-fires without genuine stakes"]}),
    MechanismSpec("expansion", 3,
        "core_affect positive + mild interoception + memory activating (no unique signature)",
        "outward opening — mind widens, something larger than me",
        ["nature documentary vista", "scientific wonder scene", "religious awe moment"],
        detector_spec={
            "shared_with": ["affect-rise"],
            "source": "transcript + contextual",
            "signals": ["scale/grandness language (vast, enormous, beyond)",
                       "external subject of inspiration",
                       "absence of body-inward markers"],
            "failure_modes": ["any positive affect can mis-fire; weakest detector in vocabulary"]}),
    MechanismSpec("drift", 1,
        "very low activation across all axes + mild reward maintenance",
        "passive waiting — attention deployed minimally, time stretches",
        ["DMV waiting scene", "filler dialogue", "ambient scene"]),
    MechanismSpec("dissonance", 3,
        "multi-axis mismatch — language active while affect runs counter",
        "something's off — cognitive mismatch, ironic register",
        ["ironic humor", "satirical monologue", "unreliable narrator revealed"],
        detector_spec={
            "shared_with": [],  # no direct neighbor; comparison-based
            "source": "transcript semantics vs affective prediction",
            "signals": ["content semantically positive + affect negative (or vice versa)",
                       "ironic markers", "satirical framing"],
            "failure_modes": ["subtle irony may not produce measurable mismatch"]}),
]

MECHANISMS_BY_NAME = {m.name: m for m in MECHANISM_LABELS}


# ─── Seed sequence vocabulary (10 patterns) ──────────────────────

@dataclass
class SequenceSpec:
    name: str
    pattern: List[str]
    description: str
    flex: bool = True           # allow insertions/omissions within tolerance


SEQUENCES: List[SequenceSpec] = [
    SequenceSpec("joke-structure",
        ["anticipation", "pattern-break", "release"],
        "setup · twist · laughter"),
    SequenceSpec("tragic-arc",
        ["vulnerability-transfer", "affect-rise", "body-surge", "affect-fade"],
        "exposure · build · devastation · settling"),
    SequenceSpec("awakening-build",
        ["universal-recognition", "body-turn", "stakes-compression", "inward-pivot"],
        "truth-naming · body engages · time shortens · turn inward"),
    SequenceSpec("reveal-structure",
        ["threshold-approach", "pattern-break", "satisfaction-peak"],
        "tension · twist · lands"),
    SequenceSpec("intimacy-deepening",
        ["intimacy-turn", "vulnerability-transfer", "body-anchor"],
        "closing distance · risk shared · presence"),
    SequenceSpec("conflict-escalation",
        ["opposition", "affect-rise", "stakes-compression", "pattern-break"],
        "clash · heat · urgency · turn"),
    SequenceSpec("awe-expansion",
        ["body-anchor", "affect-rise", "expansion", "evocation"],
        "stillness · intensifying · opening · memory joins"),
    SequenceSpec("contemplation-spiral",
        ["word-focus", "contemplation", "recognition", "body-turn"],
        "parse · think · realize · feel"),
    SequenceSpec("shame-contraction",
        ["body-surge", "withdrawal", "restraint", "body-anchor"],
        "flush · retreat · hold · settle"),
    SequenceSpec("sensory-immersion",
        ["anticipation", "sensation-flood", "body-anchor", "affect-fade"],
        "leaning in · overwhelmed · present · settling"),
]

SEQUENCES_BY_NAME = {s.name: s for s in SEQUENCES}
