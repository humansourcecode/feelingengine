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
class Exemplar:
    """A reference example for a mechanism, with optional context."""
    tag: str
    context: Optional[str] = None

    def __str__(self) -> str:
        return self.tag


def _ex(tag: str, context: str | None = None) -> Exemplar:
    return Exemplar(tag=tag, context=context)


@dataclass
class MechanismSpec:
    """Definition of a single mechanism label."""
    name: str
    tier: int
    tribe_signature: str          # human-readable
    phenomenology: str
    exemplars: List[Exemplar]
    detector_spec: Optional[dict] = None


# ─── The 28 mechanism labels ──────────────────────────────────────

MECHANISM_LABELS: List[MechanismSpec] = [
    # ─── I. Interoception ───
    MechanismSpec("body-turn", 1,
        "interoception rising, language dropping",
        "attention drops into the body — breath catches, words recede",
        [_ex("Jobs t=59", "59 seconds into Steve Jobs' 2005 Stanford commencement speech, he pauses mid-thought about dropping out of college. The words trail off and you watch him feel the weight of the memory before he speaks again."),
         _ex("Keats 'My heart aches'", "The opening line of John Keats' poem 'Ode to a Nightingale' (1819). No explanation, no setup — the poem drops you straight into a physical ache. Your body registers it before your mind catches up."),
         _ex("first-touch intimate scene")]),
    MechanismSpec("body-surge", 1,
        "interoception sudden spike from low baseline",
        "something physical moves through — heart lifts or drops, shock lands",
        [_ex("Ballou 'last breath on battlefield'", "Sullivan Ballou's 1861 letter to his wife Sarah, written a week before he was killed at Bull Run. He describes imagining his last breath on the battlefield — the physical reality of dying cuts through the tenderness of the letter."),
         _ex("horror jump scare"),
         _ex("'I love you' reveal")]),
    MechanismSpec("body-anchor", 1,
        "interoception sustained high + other axes quiet",
        "sustained body presence — breath slow, nothing else competes",
        [_ex("guided meditation"),
         _ex("slow-mo sports", "A slow-motion replay of a decisive play. The crowd noise drops away, the movement stretches out, and all that's left is watching the body in motion. Pure physical attention."),
         _ex("silence after news breaks", "The moment right after someone receives devastating news. No reaction yet — just the body holding still, processing through sensation before thought arrives.")]),
    MechanismSpec("sensation-flood", 1,
        "interoception + core_affect both sustained high",
        "the body is filled — sensory overwhelm, more than can be processed",
        [_ex("concert climax", "The peak moment at a live concert when the bass shakes your chest, the lights are blinding, and the crowd is moving as one. More input than you can consciously process."),
         _ex("first bite of extraordinary food"),
         _ex("erotic sensory scene")]),

    # ─── II. Core affect ───
    MechanismSpec("affect-rise", 1,
        "|core_affect| magnitude increasing",
        "feeling thickens — the moment gets more charged",
        [_ex("Keats build to 'easeful Death'", "In 'Ode to a Nightingale,' Keats slowly builds from aching to longing across several stanzas until he arrives at the phrase 'easeful Death' — the emotional charge has been thickening line by line."),
         _ex("escalating argument"),
         _ex("crowd roaring")]),
    MechanismSpec("affect-fade", 1,
        "core_affect returning toward baseline after peak",
        "the storm passes — settling, quieter",
        [_ex("post-climax stillness"),
         _ex("end of joke laughter"),
         _ex("Frost 'miles to go before I sleep'", "The closing lines of Robert Frost's 'Stopping by Woods on a Snowy Evening' (1923). After the quiet beauty of the snowy woods, the repeated line settles the poem back to earth — duties remain, the moment passes.")]),

    # ─── III. Regulation ───
    MechanismSpec("restraint", 1,
        "regulation sustained high + core_affect elevated but not released",
        "something withheld — pressure that doesn't let out",
        [_ex("holding back tears at toast", "Giving a wedding toast about someone you love. The emotion is right there, but you hold it together because the moment isn't about you breaking down."),
         _ex("composed under attack"),
         _ex("meditation effort", "The sustained effort of staying with the breath when your mind wants to wander. Not calm — actively held calm.")]),
    MechanismSpec("release", 1,
        "regulation sudden drop",
        "something let go — unguarded, an opening",
        [_ex("laughter breaking tension"),
         _ex("tears finally coming"),
         _ex("'let me be honest' transition", "The moment in a conversation when someone drops the diplomatic phrasing and says what they actually think. The guard comes down and the real thing comes out.")]),
    MechanismSpec("threshold-approach", 2,
        "regulation rising + reward rising simultaneously",
        "something about to turn — tension winding, leaning in without yet arriving",
        [_ex("speech building to reveal"),
         _ex("musical pre-drop", "The build-up in electronic or orchestral music right before the beat drops. Everything is rising — volume, tempo, tension — and you know something is about to hit."),
         _ex("pause before confession")],
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
        [_ex("public embarrassment"),
         _ex("post-caught shame", "The moment after being caught in a lie or mistake. The body contracts — shoulders drop, eyes go down, the instinct is to become smaller."),
         _ex("pulling back from vulnerability")],
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
        [_ex("cliffhanger"),
         _ex("'let me tell you a story'", "The phrase itself activates anticipation — your brain shifts into receiving mode, leaning forward for what comes next."),
         _ex("riddle with deferred answer")]),
    MechanismSpec("satisfaction-peak", 1,
        "reward peaks then slight fall, with prominence",
        "a click — the answer lands, a small rest",
        [_ex("comedic punchline"),
         _ex("mystery solution"),
         _ex("resolution chord", "The final chord of a piece of music that resolves the harmonic tension. Everything that was unresolved clicks into place.")]),

    # ─── V. Memory ───
    MechanismSpec("recognition", 1,
        "memory rising, core_affect modest",
        "wait — I know this — familiarity before full context",
        [_ex("quoted famous line"),
         _ex("musical motif return", "When a film score brings back a melody from earlier. You recognize it before you can name where you heard it — the familiarity arrives before the thought."),
         _ex("'remember when...'")]),
    MechanismSpec("evocation", 1,
        "memory + core_affect rising together",
        "a scene returns — not just memory, feeling again",
        [_ex("smell-triggered memory", "A scent pulls you back to a specific place and time — not just the facts of it, but how it felt to be there. The memory isn't recalled, it's re-experienced."),
         _ex("'I was seventeen and...'"),
         _ex("revisiting old neighborhood")]),
    MechanismSpec("universal-recognition", 2,
        "interoception + memory + social rising together",
        "oh — that's true — body confirms what the words claim, no argument needed",
        [_ex("'No one wants to die'", "From Steve Jobs' Stanford speech. He says it plainly and your body agrees before your mind can even consider arguing. A statement so fundamentally true it bypasses thought."),
         _ex("'we all want to be loved'"),
         _ex("'the light goes out'", "A metaphor for death or loss that lands as physical truth. You don't interpret it — you feel the absence it describes.")],
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
        [_ex("'never told anyone'"),
         _ex("lean-in before reveal", "The physical lean-in during conversation when someone is about to share something personal. The space between you narrows before the words arrive."),
         _ex("held eye contact")],
        detector_spec={
            "shared_with": ["opposition"],
            "source": "transcript",
            "signals": ["positive affect valence", "affiliative language (we, together, share)",
                       "softening markers"],
            "failure_modes": ["dark intimacy (shared grief) can read as opposition"]}),
    MechanismSpec("opposition", 2,
        "social rising + regulation elevated + core_affect negative trend",
        "positions collide — I am against you, air tightens",
        [_ex("political debate"),
         _ex("escalating argument"),
         _ex("'but you say...' rhetoric", "The rhetorical move of restating someone's position before disagreeing with it. The air tightens because you know the counter is coming.")],
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
        [_ex("Jobs 'I was 17 years old'", "In his Stanford speech, Jobs shares being 17 and reading a quote about living each day as your last. He's not performing emotion — he's genuinely exposed, and your body opens in response to his risk."),
         _ex("Ballou letter", "Sullivan Ballou's 1861 Civil War letter to his wife. He writes about love and death with complete openness, knowing he may not survive. Reading it, you feel what he's risking by putting these words on paper."),
         _ex("therapy breakthrough")]),
    MechanismSpec("boundary-establish", 3,
        "social axis activates with self-other sharpening (signature not yet fully operationalized)",
        "a wall goes up — line drawn, position stated",
        [_ex("'that's not who I am'"),
         _ex("parental refusal", "A parent saying 'no' to something a child desperately wants — not in anger, but with a firmness that defines the boundary between them."),
         _ex("organizational 'we don't do that here'")],
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
        [_ex("technical lecture"),
         _ex("news anchor"),
         _ex("syllogism", "A logical argument where each step follows the last: 'All men are mortal; Socrates is a man; therefore Socrates is mortal.' Pure word-tracking — you're following the logic, not feeling anything.")]),
    MechanismSpec("word-recede", 1,
        "language dropping while other dimensions rise",
        "the words stop mattering — what's said is less than what's felt",
        [_ex("Jobs death pivot", "The moment in Jobs' Stanford speech when he shifts from talking about career setbacks to talking about death. The specific words become secondary — you're no longer tracking his argument, you're feeling the weight."),
         _ex("Beethoven silent movement", "The quiet passages in Beethoven's late string quartets where the music almost disappears. What remains isn't sound — it's the presence in the silence between the notes."),
         _ex("moment before a kiss")]),
    MechanismSpec("contemplation", 2,
        "language sustained high + memory rising + regulation high",
        "sustained meaning-sift — thinking through something, integrating, weighing",
        [_ex("philosophical voiceover"),
         _ex("character wrestling with decision"),
         _ex("meditation teacher explaining", "A meditation guide talking through a concept like impermanence — you're not just hearing words, you're actively trying to integrate what they mean against your own experience.")],
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
        [_ex("Jobs death pivot", "When Jobs says 'remembering that I'll be dead soon,' the speech stops being about Apple or college and becomes about mortality. The external world falls away."),
         _ex("Keats 'fade far away, dissolve'", "In 'Ode to a Nightingale,' Keats writes about wanting to 'fade far away, dissolve, and quite forget.' The poem turns from observing the bird to dissolving into pure internal sensation."),
         _ex("meditation transition", "The moment in guided meditation when the teacher stops talking about the technique and says something like 'now just be with whatever is here.' External instruction ends, internal experience begins.")]),
    MechanismSpec("pattern-break", 1,
        "sudden shift across multiple axes simultaneously",
        "wait — what just happened — ground moved, attention redeploys",
        [_ex("rhetorical turn mid-sentence"),
         _ex("dissonant note", "A single wrong note in an otherwise harmonious piece. Everything you were tracking resets — your attention snaps to the disruption."),
         _ex("unexpected scene cut")]),
    MechanismSpec("stakes-compression", 2,
        "regulation + reward shift + core_affect rising",
        "time is shorter than I thought — NOW matters, no more later",
        [_ex("'your time is limited'", "From Jobs' Stanford speech: 'Your time is limited, so don't waste it living someone else's life.' The horizon collapses — what felt abstract becomes immediate."),
         _ex("terminal diagnosis scene"),
         _ex("'before it's too late' framing")],
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
        [_ex("nature documentary vista", "A wide aerial shot in a nature documentary revealing the scale of a landscape. Your perspective physically widens — you feel small in a way that's awe, not threat."),
         _ex("scientific wonder scene"),
         _ex("religious awe moment")],
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
        [_ex("DMV waiting scene", "A scene in a movie where a character sits in a waiting room at the DMV. Nothing is happening, and you feel the time stretching."),
         _ex("filler dialogue"),
         _ex("ambient scene")]),
    MechanismSpec("dissonance", 3,
        "multi-axis mismatch — language active while affect runs counter",
        "something's off — cognitive mismatch, ironic register",
        [_ex("ironic humor"),
         _ex("satirical monologue", "A comedian or character saying something absurd with complete sincerity. The words say one thing, the tone says another — your brain catches the gap."),
         _ex("unreliable narrator revealed", "The moment you realize the character telling the story has been lying or distorting events. Everything you accepted gets reframed, and the mismatch between what was said and what's true hits at once.")],
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
