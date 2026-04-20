# Mechanism Labels — Seed Vocabulary (Generation 1)

**Status:** Draft for review · iteration expected
**Audience:** Feeling Engine pipeline designers and downstream consumers (arc miners, compose-workflow builders, content researchers)
**Companion docs:** `methodology.md` (full pipeline) · `detector_validation.md` (validation protocols)

---

## Purpose

The Feeling Engine currently outputs feeling words (Desire) and topic-bound synthesized labels (Layer 4). Both are useful but have known limitations:

- **Feeling words** (`curiosity`, `vulnerability`, `dread`) describe *results* — what someone feels — not *mechanisms* — what the content is doing to produce that feeling.
- **Topic-bound labels** (`mortality_awareness`) only generalize to content about the same topic.

A **mechanism label** names a content-agnostic structural move — a pattern in content that reliably produces a particular brain-state pattern, independent of topic. `universal-recognition`, `inward-pivot`, `vulnerability-transfer` should be deployable with mushroom farming, childbirth, or combat just as readily as with death.

This doc drafts the first-generation seed vocabulary, grouped by TRIBE axis to ensure breadth coverage.

---

## The multi-source architecture

The Feeling Engine is not a TRIBE-only pipeline. It's a **multi-source system** that combines brain-state measurement (TRIBE) with content analysis (LLM on transcript). Different mechanisms require different combinations of signals to detect reliably.

This is an important distinction: **TRIBE is the source of truth for brain state.** Its output authoritatively answers *what is the brain doing right now*. But a mechanism label is not a brain state — it's an *interpretation* of a brain state in content context. Some mechanisms are fully determined by TRIBE alone; others need additional signal to pin down.

### Detection tiers

Every label in this vocabulary is classified by its detection path:

| Tier | Detection path | What it means |
|------|----------------|----------------|
| **1** | TRIBE-only | The TRIBE signature is *unique* to this label. Detector is a signal-processing algorithm applied to the 7-axis output. |
| **2** | TRIBE + LLM-on-transcript | The TRIBE signature narrows the candidate set; transcript analysis picks the specific label. |
| **3** | TRIBE + LLM + discriminator | TRIBE doesn't even narrow to a clean set; additional signals (framing, audio features, contextual inference) are needed. These labels carry the most detection uncertainty. |

Analogy: a doctor measures blood pressure with a cuff (ground truth for BP) and asks about diet (context). The diagnosis *salt-induced hypertension* integrates both. The cuff isn't demoted — it's still authoritative for what it measures. The diagnosis is a *higher-level construct* that requires multiple inputs.

Same here. TRIBE remains authoritative for brain state. Mechanism labels are multi-source derived constructs. Rigor comes from **documenting per-label which sources are load-bearing and how disambiguation works**. See `detector_validation.md` for validation protocols.

---

## Design principles

1. **Content-agnostic.** If the label name implies a topic (`mortality`, `romance`, `success`), it's wrong. Labels describe the *move*, not the *material*.
2. **TRIBE-grounded.** Every label must have a TRIBE signature (the brain-state pattern it corresponds to), even if disambiguation requires additional signal.
3. **Phenomenologically describable.** Each label must be expressible in everyday body-language (tightening, opening, leaning, receding). No jargon.
4. **Detection-path explicit.** Tier classification is mandatory. Tier 2 and Tier 3 labels require a detector spec documenting the disambiguation source.
5. **Finite but discoverable.** Target 20-30 labels for Generation 1. Too few → low resolution. Too many → boundaries blur.

---

## Label schema

Each label has these fields:

```
NAME             kebab-case, content-agnostic
TIER             1 / 2 / 3
TRIBE signature  the 7-dimensional pattern at the brain level
Phenomenology    concrete body-felt description (everyday language)
Exemplars        2-3 instances from maximally different content domains
DETECTOR SPEC    (Tier 2/3 only) — disambiguation source, signals, failure modes
```

---

## Vocabulary (28 labels)

### I. Interoception mechanisms (body-focused)

#### `body-turn` — **Tier 1**
- **TRIBE signature:** interoception rising, language dropping
- **Phenomenology:** attention drops into the body — breath catches, the chest registers, words recede
- **Exemplars:**
  - Jobs t=59, "No one wants to die" (speech)
  - Keats, "My heart aches" opening (poetry)
  - First-touch moment in an intimate scene (film)

#### `body-surge` — **Tier 1**
- **TRIBE signature:** interoception sudden spike from low baseline
- **Phenomenology:** something physical moves through — heart lifts or drops, a shock lands
- **Exemplars:**
  - Ballou, "when my last breath escapes me on the battlefield" (letter)
  - Horror-film jump scare (film)
  - "I love you" reveal in a quiet scene (film/romance)

#### `body-anchor` — **Tier 1**
- **TRIBE signature:** interoception high + stable across multiple seconds, other axes quiet
- **Phenomenology:** sustained body presence — breath is slow, nothing else competes
- **Exemplars:**
  - Guided meditation instruction (self-help)
  - Slow-motion sports commentary on a decisive play (sports)
  - Held silence after news breaks (journalism)

#### `sensation-flood` — **Tier 1**
- **TRIBE signature:** interoception + core_affect both sustained high
- **Phenomenology:** the body is filled — sensory overwhelm, can't parse cleanly, more than can be processed
- **Exemplars:**
  - Climax of a concert (music)
  - Eating something extraordinary for the first time (food media)
  - Erotic scene with sensory focus (film)

### II. Core affect mechanisms (valence + intensity)

#### `affect-rise` — **Tier 1**
- **TRIBE signature:** |core_affect| magnitude increasing
- **Phenomenology:** feeling thickens — the moment gets more charged, the room heavier or brighter
- **Exemplars:**
  - Keats building to "half in love with easeful Death" (poetry)
  - Argument escalating in drama (film)
  - Crowd roaring before a goal (sports)

#### `affect-fade` — **Tier 1**
- **TRIBE signature:** core_affect returning toward baseline after peak
- **Phenomenology:** the storm passes — settling, what's left is quieter
- **Exemplars:**
  - Post-climax stillness in film (film)
  - End of a joke when laughter dies (comedy)
  - Frost, "miles to go before I sleep" (poetry)

### III. Regulation mechanisms (emotion modulation)

#### `restraint` — **Tier 1**
- **TRIBE signature:** regulation sustained high, core_affect elevated but not released
- **Phenomenology:** something withheld — a pressure that doesn't let out
- **Exemplars:**
  - Holding back tears at a wedding toast (everyday)
  - Political figure staying composed under attack (news)
  - Meditation effort during distraction (self-help)

#### `release` — **Tier 1**
- **TRIBE signature:** regulation sudden drop
- **Phenomenology:** something let go — unguarded, an opening
- **Exemplars:**
  - Laughter that breaks tension (comedy)
  - Tears finally coming in grief (film)
  - "OK, let me be honest..." transition (speech)

#### `threshold-approach` — **Tier 2**
- **TRIBE signature:** regulation rising + reward rising simultaneously
- **Phenomenology:** something about to turn — tension winding, the air changes, leaning in without yet arriving
- **Exemplars:**
  - Speech building toward a reveal (speech)
  - Musical pre-drop (music)
  - Pause before a confession (drama)
- **DETECTOR SPEC:**
  - *Shared TRIBE signature with:* `anticipation` (both involve rising reward)
  - *Disambiguation source:* LLM-on-transcript
  - *Disambiguation signals:* presence of tension-building language without explicit reward-promise; cadence shifts (shorter sentences, rising pitch markers); explicit delay-markers ("and then...", "wait for it")
  - *Known failure modes:* slow-building anticipation in calm content reads as threshold-approach when it's really just anticipation
  - *Validation:* see `detector_validation.md`

#### `withdrawal` — **Tier 3**
- **TRIBE signature:** regulation + interoception rising, social-axis suppressed
- **Phenomenology:** shrinking inward — wanting to disappear, the body contracts in self-protection
- **Exemplars:**
  - Public embarrassment moment (drama)
  - Shame after being caught (film)
  - Pulling back from a vulnerable admission (memoir)
- **DETECTOR SPEC:**
  - *Shared TRIBE signature with:* `restraint` (both elevate regulation)
  - *Disambiguation source:* LLM-on-transcript + social-axis direction
  - *Disambiguation signals:* self-protective language ("I shouldn't have," "I wish I hadn't"); subject withdrawing from prior disclosure; social-axis specifically suppresses rather than sustains
  - *Known failure modes:* quiet reflection can look like withdrawal when it's actually contemplation
  - *Validation:* requires human-labeled corpus of shame/hiding moments vs. quiet thought

### IV. Reward mechanisms (motivation + expectation)

#### `anticipation` — **Tier 1**
- **TRIBE signature:** reward rising + regulation steady
- **Phenomenology:** leaning forward — wanting what's next, time slightly speeds, attention sharpens
- **Exemplars:**
  - Cliffhanger in serialized TV (film)
  - Jobs, "Let me tell you a story" (speech)
  - Riddle with deferred answer (puzzle)

#### `satisfaction-peak` — **Tier 1**
- **TRIBE signature:** reward peaks then slight fall
- **Phenomenology:** a click — the answer lands, a small rest, the puzzle resolves
- **Exemplars:**
  - Comedic punchline (comedy)
  - Mystery solution (fiction)
  - Resolution chord in music (music)

### V. Memory mechanisms (recall + binding)

#### `recognition` — **Tier 1**
- **TRIBE signature:** memory rising, core_affect modest
- **Phenomenology:** *wait — I know this* — familiarity before full context
- **Exemplars:**
  - Quoted line from a famous speech (speech)
  - Motif return in music (music)
  - "Remember when..." opener (conversation)

#### `evocation` — **Tier 1**
- **TRIBE signature:** memory + core_affect rising together
- **Phenomenology:** a scene returns — not just memory, feeling again
- **Exemplars:**
  - Smell-triggered childhood memory (memoir)
  - "I was seventeen and..." (speech)
  - Revisiting an old neighborhood (film)

#### `universal-recognition` — **Tier 2**
- **TRIBE signature:** interoception + memory + social rising together
- **Phenomenology:** *oh — that's true* — the body confirms what the words claim; no argument needed
- **Exemplars:**
  - "No one wants to die" (speech)
  - "We all want to be loved" (any rhetoric)
  - "The light goes out eventually" (memoir)
- **DETECTOR SPEC:**
  - *Shared TRIBE signature with:* `evocation` (both fire memory + affect)
  - *Disambiguation source:* LLM-on-transcript
  - *Disambiguation signals:* speaker makes a universal claim vs. a personal anecdote; phrasing like "everyone," "no one," "we all"; content unarguable at body level (biology, time, death, hunger)
  - *Known failure modes:* personal claim stated in universal terms ("Nobody understands me") can mis-fire
  - *Validation:* corpus of universal claims vs. personal memories

### VI. Social mechanisms (self-other relational)

#### `intimacy-turn` — **Tier 2**
- **TRIBE signature:** social rising + regulation dropping
- **Phenomenology:** something soft — the room narrows, the distance closes, we're in this together
- **Exemplars:**
  - "I'll tell you something I've never told anyone" (memoir)
  - Lean-in gesture before a reveal (drama)
  - Eye contact held a beat too long (film)
- **DETECTOR SPEC:**
  - *Shared TRIBE signature with:* `opposition` (both activate social axis with regulation shift)
  - *Disambiguation source:* LLM-on-transcript
  - *Disambiguation signals:* positive affect valence; affiliative language ("we," "together," "share"); softening prosody markers; absence of contradiction/contest markers
  - *Known failure modes:* dark intimacy (shared grief, mutual fear) can be misread as opposition because of negative affect
  - *Validation:* corpus of intimate scenes vs. confrontational scenes

#### `opposition` — **Tier 2**
- **TRIBE signature:** social rising + regulation elevated + core_affect negative trend
- **Phenomenology:** positions collide — I am against you, the air tightens between us
- **Exemplars:**
  - Political debate confrontation (news)
  - Argument escalating (film)
  - Rhetorical "but you say..." moment (speech)
- **DETECTOR SPEC:**
  - *Shared TRIBE signature with:* `intimacy-turn` (both activate social axis)
  - *Disambiguation source:* LLM-on-transcript
  - *Disambiguation signals:* contradiction markers ("but," "no," "however"); position-marking ("I think," "my view"); speaker addressing a contested other; oppositional valence in phrasing
  - *Known failure modes:* ironic opposition (friends roasting each other) reads as opposition but is intimate. Flag when oppositional markers + positive valence markers co-occur.
  - *Validation:* corpus of debates vs. intimate scenes

#### `vulnerability-transfer` — **Tier 1**
- **TRIBE signature:** social + interoception rising together
- **Phenomenology:** the body opens because theirs is open — catching what the speaker is risking
- **Exemplars:**
  - Jobs, "I was 17 years old" opening (speech)
  - Ballou's entire letter (letter)
  - Therapy breakthrough scene (film)

#### `boundary-establish` — **Tier 3**
- **TRIBE signature:** social axis activates with specific pattern of self-other sharpening (not yet operationally defined)
- **Phenomenology:** a wall goes up — a line is drawn, position stated, separation named
- **Exemplars:**
  - "That's not who I am" (memoir)
  - Parental refusal to a child (family)
  - Organizational "we don't do that here" (corporate)
- **DETECTOR SPEC:**
  - *TRIBE signature issue:* social-axis "sharpening" has no clean operational definition in the 7-axis output; this label may be TRIBE-inferable only indirectly
  - *Disambiguation source:* LLM-on-transcript + contextual framing
  - *Disambiguation signals:* explicit distinction language ("I'm not that," "we don't," "this line"); refusal/negation patterns; position-declarative syntax
  - *Known failure modes:* any refusal can be misread as boundary-establish when it's simply contradiction (opposition). Distinguishing a *self-definition* refusal from a *contest* refusal is the core challenge.
  - *Validation:* requires manual review; detector is weakest in this tier

### VII. Language mechanisms (word + meaning processing)

#### `word-focus` — **Tier 1**
- **TRIBE signature:** language high, interoception + core_affect low
- **Phenomenology:** all attention on the words — following the thread, thinking rather than feeling
- **Exemplars:**
  - Technical explanation in a lecture (education)
  - News anchor reading a report (journalism)
  - Syllogism in an argument (philosophy)

#### `word-recede` — **Tier 1**
- **TRIBE signature:** language dropping while other dimensions rise
- **Phenomenology:** the words stop mattering — what's said is less than what's felt
- **Exemplars:**
  - Jobs death pivot (speech)
  - Beethoven's silent movement (music)
  - The moment before a kiss (film)

#### `contemplation` — **Tier 2**
- **TRIBE signature:** language sustained high + memory rising + regulation high
- **Phenomenology:** sustained meaning-sift — thinking *through* something, weighing, integrating; not just parsing
- **Exemplars:**
  - Philosophical essay voiceover (philosophy)
  - Character wrestling with a decision (drama)
  - Meditation teacher explaining a concept (self-help)
- **DETECTOR SPEC:**
  - *Shared TRIBE signature with:* `word-focus` (both elevate language axis)
  - *Disambiguation source:* LLM-on-transcript + temporal signature
  - *Disambiguation signals:* content complexity (abstract vs. concrete); presence of hedging, qualification ("perhaps," "on the other hand"); longer pauses; content requires integration rather than parsing
  - *Known failure modes:* slow technical explanation can look like contemplation; thoughtful but straightforward content is ambiguous
  - *Validation:* requires human-labeled corpus of contemplative vs. informational content

### VIII. Multi-axis mechanisms (complex signatures)

Note: we previously called these "compound mechanisms." We're dropping that grouping — every label is a mechanism, some just have more complex signatures.

#### `inward-pivot` — **Tier 1**
- **TRIBE signature:** language drops + interoception rises, anchored by social context
- **Phenomenology:** external → internal — the world falls away, it's about what's happening in me
- **Exemplars:**
  - Jobs death pivot (speech)
  - Keats, "Fade far away, dissolve" (poetry)
  - Meditation transition from external sound to breath (self-help)

#### `pattern-break` — **Tier 1**
- **TRIBE signature:** sudden shift across multiple axes simultaneously (detectable via derivative analysis)
- **Phenomenology:** *wait — what just happened* — the ground moved, attention has to redeploy
- **Exemplars:**
  - Rhetorical turn that changes subject mid-sentence (speech)
  - Dissonant note in expected progression (music)
  - Cut to unexpected scene (film)

#### `stakes-compression` — **Tier 2**
- **TRIBE signature:** regulation + reward shift simultaneously, core_affect rising
- **Phenomenology:** time is shorter than I thought — NOW matters, no more later
- **Exemplars:**
  - "Your time is limited" (speech)
  - Terminal diagnosis scene (film)
  - Political "before it's too late" framing (news)
- **DETECTOR SPEC:**
  - *Shared TRIBE signature with:* `pattern-break` (both multi-axis shifts)
  - *Disambiguation source:* LLM-on-transcript
  - *Disambiguation signals:* time-horizon-collapsing language ("your time is," "before it's too late," "while you still can"); explicit finitude markers; urgency framing
  - *Known failure modes:* any urgency-framed content can trigger; must distinguish genuine stakes-compression from rhetorical flourish
  - *Validation:* corpus of genuine stakes moments vs. performative urgency

#### `expansion` — **Tier 3**
- **TRIBE signature:** core_affect positive + mild interoception + memory activating — but no unique combination identified; shares signature elements with multiple labels
- **Phenomenology:** outward opening — mind widens, something larger than me, the moment enlarges
- **Exemplars:**
  - Nature documentary vista (film)
  - Scientific wonder scene (education)
  - Religious awe moment (speech/film)
- **DETECTOR SPEC:**
  - *TRIBE signature issue:* no unique combination in 7-axis output; reads as "positive affect-rise + nothing specifically body-inward"
  - *Disambiguation source:* LLM-on-transcript + contextual framing
  - *Disambiguation signals:* language of scale/grandness ("vast," "enormous," "beyond"); subject is external and inspiring; absence of body-inward markers
  - *Known failure modes:* any positive affect can be misread as expansion; the "mind-widens" quality is phenomenologically distinct but TRIBE-ambiguous
  - *Validation:* this label carries the most detection uncertainty in the vocabulary. Requires human review in most cases. May need to be refactored or merged after validation.

#### `drift` — **Tier 1**
- **TRIBE signature:** very low activation across all axes + mild reward maintenance (attention-sustain without target)
- **Phenomenology:** passive waiting — attention deployed minimally, time stretches
- **Exemplars:**
  - DMV waiting-room scene (film)
  - Filler dialogue between plot beats (fiction)
  - Ambient scene establishing place (film)

#### `dissonance` — **Tier 3**
- **TRIBE signature:** multi-axis mismatch — specifically, language processing active while affect runs counter to expected valence
- **Phenomenology:** *something's off* — cognitive mismatch, ironic register, the words and the feeling don't align
- **Exemplars:**
  - Ironic humor (comedy)
  - Satirical monologue (comedy)
  - Unreliable narrator revealed (fiction)
- **DETECTOR SPEC:**
  - *TRIBE signature issue:* detecting mismatch requires comparing content semantics to affective response — not just reading brain state
  - *Disambiguation source:* LLM-on-transcript comparing semantic content (what's said) to affective prediction (what the content *should* produce) vs. TRIBE's actual affective response
  - *Disambiguation signals:* content semantically positive while affect is negative (or vice versa); explicit irony markers; satirical framing; meta-commentary
  - *Known failure modes:* subtle irony may not produce measurable affect mismatch; deadpan delivery is borderline
  - *Validation:* requires corpus of ironic/satirical content with human-labeled irony confidence

---

## Organization for users (alternative grouping by narrative function)

The TRIBE-axis grouping above ensures the vocabulary covers each brain dimension. But for creators and viewers labeling content, a narrative-function grouping may be more intuitive:

| Function | Labels |
|----------|--------|
| **Opening** (setup, orientation) | `word-focus` · `drift` · `recognition` · `anticipation` |
| **Deepening** (going inward, intimate) | `body-turn` · `body-anchor` · `inward-pivot` · `intimacy-turn` · `vulnerability-transfer` · `contemplation` |
| **Building** (tension, stakes, intensity) | `threshold-approach` · `affect-rise` · `restraint` · `stakes-compression` · `sensation-flood` |
| **Turning** (pivots, breaks, reveals) | `pattern-break` · `release` · `satisfaction-peak` · `body-surge` · `word-recede` |
| **Confrontation** (conflict, distinction, mismatch) | `opposition` · `boundary-establish` · `withdrawal` · `dissonance` |
| **Amplification** (widening, universal, evoking) | `expansion` · `universal-recognition` · `evocation` |
| **Settling** (closure, fade, rest) | `affect-fade` |

Same 28 labels; different mental model. Axis grouping answers "does this vocabulary cover every brain dimension?" Narrative grouping answers "what's available at each point in my story?"

---

## Known collapse / split candidates

After empirical validation, some of these may need adjustment:

- **`body-surge` vs `satisfaction-peak`** — both transient spikes but on different axes; validate whether they reliably co-occur
- **`threshold-approach` vs `anticipation`** — both involve reward rising; validate whether regulation axis reliably distinguishes
- **`contemplation` vs sustained `word-focus`** — may need temporal threshold to split
- **`affect-rise`** — may need to split into positive / negative variants if valence changes the experiential quality too much
- **`expansion`** — weakest-grounded label; may need to merge into "positive affect-rise with grand context" or be refactored entirely

---

## Validation approach

Full protocols in `detector_validation.md`. Summary:

1. **Corpus coverage test.** Run TRIBE on every entry in the 38-item corpus. For each, attempt to label with at least one mechanism. Labels that never fire across the corpus → prune. Entries that can't be labeled → add missing mechanisms.

2. **Inter-rater agreement.** Label the same content twice, independently, using only this doc. Labels that drift between runs need tighter definitions.

3. **Detector accuracy.** For each Tier 2 / Tier 3 label, build a validation corpus with human ground-truth labels. Measure detector accuracy. Target >80% for Tier 2, >70% for Tier 3.

4. **Generative test.** Given a target mechanism, can Claude generate a novel scene in an unrelated domain that would deploy it? Human judgment of plausibility is the first-pass metric; TRIBE on the generated scene is the second.

---

---

## Label application schema

A mechanism label is applied to a specific moment in content. Each application produces a record with these fields:

```
{
  label:       string,   // one of 28
  tier:        1 | 2 | 3,
  start_sec:   float,    // when the mechanism begins
  end_sec:     float,    // when it ends
  intensity:   float,    // 0.0 – 1.0
  confidence:  float,    // 0.0 – 1.0 (how sure the detector is)
  signals:     object    // raw inputs that led to detection
}
```

### Intensity (0.0 – 1.0)

How strongly the mechanism is present. Computed per tier:

- **Tier 1:** magnitude of the TRIBE signature. For `body-surge`, intensity = normalized interoception spike amplitude.
- **Tier 2:** weighted combination of TRIBE magnitude + LLM disambiguation confidence. For `universal-recognition`, intensity = `0.6 × TRIBE_signature_magnitude + 0.4 × LLM_universal_claim_confidence`.
- **Tier 3:** same formula but with wider tolerance and mandatory human review below 0.6.

Intensity is what distinguishes a gentle body-turn from a devastating one. Same label, same phenomenology, different magnitude.

### Duration

`end_sec - start_sec`. Minimum detection window: 0.5 seconds (shorter = noise). Maximum: clip-length (a mechanism can sustain). Sub-second spikes merge into their parent label application.

### Confidence

How sure the detector is that this label is correct. Lower for Tier 2/3, always reported. When confidence is low, the label application should be reviewed or flagged in the pipeline output.

---

## The sequence layer

Mechanisms are moments. Stories are sequences. The vocabulary needs a *composition layer* above individual labels that names common narrative structures — how mechanisms chain into arcs.

### Representation

A sequence is a named pattern:

```
{
  name:        string,
  description: phenomenological gloss,
  pattern:     ordered list of mechanism labels,
  flex_points: which positions in the pattern allow substitution,
  min_duration: typical minimum length,
  max_duration: typical maximum length
}
```

A detected sequence is matched from a stream of detected mechanism labels (with some tolerance — flex points, minor reordering, optional intermediates).

### Seed sequence vocabulary (10 patterns)

These are the narrative arc patterns we'll start with. Each composes existing mechanism labels.

| Sequence | Pattern | Phenomenology |
|----------|---------|---------------|
| `joke-structure` | `anticipation` → `pattern-break` → `release` | setup · twist · laughter |
| `tragic-arc` | `vulnerability-transfer` → `affect-rise` → `body-surge` → `affect-fade` | exposure · build · devastation · settling |
| `awakening-build` | `universal-recognition` → `body-turn` → `stakes-compression` → `inward-pivot` | truth-naming · body-engages · time shortens · turn inward |
| `reveal-structure` | `threshold-approach` → `pattern-break` → `satisfaction-peak` | tension · twist · lands |
| `intimacy-deepening` | `intimacy-turn` → `vulnerability-transfer` → `body-anchor` | closing distance · risk shared · presence |
| `conflict-escalation` | `opposition` → `affect-rise` → `stakes-compression` → `pattern-break` | clash · heat · urgency · turn |
| `awe-expansion` | `body-anchor` → `affect-rise` → `expansion` → `evocation` | stillness · intensifying · opening · memory joins |
| `contemplation-spiral` | `word-focus` → `contemplation` → `recognition` → `body-turn` | parse · think · realize · feel |
| `shame-contraction` | `body-surge` → `withdrawal` → `restraint` → `body-anchor` | flush · retreat · hold · settle |
| `sensory-immersion` | `anticipation` → `sensation-flood` → `body-anchor` → `affect-fade` | leaning in · overwhelmed · present · settling |

Sequences are **templates, not prescriptions.** Real content often has variations — mechanisms inserted, omitted, reordered. The flex_points field specifies where variation is tolerated.

### Detection

Sequences are detected *after* mechanism labels. The pipeline:

1. Runs mechanism detectors on the full timeline → label stream with timestamps.
2. Runs sequence matcher over the label stream → detected sequences with start/end times.
3. Uses DP-style alignment with tolerance for substitutions, omissions, minor reorderings.

### Open sequence questions

- **How strict is the match?** Exact sequence vs. partial vs. "this content contains these labels in roughly this order."
- **Nesting.** A `sensory-immersion` inside a `tragic-arc`. Do we allow recursive sequences? (Probably yes.)
- **Generation.** Given a sequence, can Fire generate content that would produce it? This is where sequences become the target for cross-domain generation, not just analysis.

---

## Open questions

1. **Is 28 the right number?** Probably lands between 24-30 after validation. Some Tier 3 labels may merge or drop.

2. **Tier 3 labels are the weakest.** `boundary-establish`, `expansion`, `withdrawal`, `dissonance` all carry real detection uncertainty. Should we accept that some labels require heavy human review, or design around only high-confidence detection?

3. **How much should detector specs live in this doc vs. `detector_validation.md`?** Current draft has them inline; alternative is terse labels here and full specs in the validation doc.

4. **Sequence handling.** Mechanisms are moments. Sequences (build → release → settle) are a meta-structure. Should the vocabulary include sequence templates, or should sequences be a separate layer?

5. **Should `affect-rise` split into positive/negative variants?** Valence may change the experiential quality enough to warrant distinct labels.

6. **Weights for detector confidence.** When multiple sources agree (TRIBE + transcript), detection confidence is high. When they disagree, how do we resolve? This might need explicit rules per label.

---

## Next steps after review

- [ ] User reviews + reacts holistically to the 28 labels and tier structure
- [ ] Iterate on label names, detector specs, tier classifications
- [ ] Build validation corpus per `detector_validation.md` protocols
- [ ] Run Tier 1 detectors on the 38-entry corpus to confirm coverage
- [ ] Build Tier 2/3 detectors (LLM prompt templates) and measure accuracy
- [ ] Integrate into Feeling Engine pipeline as a new output layer
- [ ] Refactor Video 1 narrative to introduce mechanism labels (if vocabulary converges in time)
