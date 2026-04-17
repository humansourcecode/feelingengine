# Methodology

*How Feeling Engine translates brain-response predictions into emotional arcs, and why the design choices are what they are.*

---

## Design principle: brain measurement first, interpretation second

Most "emotion detection" pipelines operate on surface features — text sentiment, facial-action units, vocal pitch — and guess at what someone is feeling. That approach works for coarse affect (positive/negative) but breaks down on anything that requires distinguishing awe from mortality awareness, or longing from acceptance.

Feeling Engine inverts the stack:

1. **Measure the brain** (TRIBE v2 predicts per-region activation from content)
2. **Interpret the measurement** (Translator maps brain state to emotional vocabulary, optionally refined with content + context)

This separation matters because it gives every emotional label a falsifiable grounding. When the output says "mortality_awareness (HIGH)," it means: *interoception activated +0.47, language suppressed –0.10, transition was a rapid_spike, and these expectations are defined in advance in the vocabulary.* If any of those conditions aren't met, the label doesn't fire.

This also enables the **Context at Layer 4** principle: the brain scan is context-free (TRIBE produces the same prediction regardless of who's watching), but the *interpretation* of that scan changes with the viewer's context. Same brain data, different context, different emotional arc — but the measurement stays honest.

---

## TRIBE v2: the measurement substrate

[TRIBE v2](https://github.com/facebookresearch/tribev2) is a multimodal transformer from Meta FAIR that predicts fMRI brain responses to naturalistic content (speech, audio, video). It was trained on publicly available fMRI datasets where subjects watched movies and listened to speech with simultaneous brain recording.

For each timestep of input audio/video, TRIBE outputs activation values across ~20,484 cortical vertices on the fsaverage5 mesh. Those vertex-level predictions are the raw measurement we work with.

**Why TRIBE v2 specifically:**
- Publicly available model checkpoints (CC BY-NC 4.0 license)
- State-of-the-art fit to real fMRI data from naturalistic viewing
- Designed to handle audio, video, and multimodal input
- Predicts across the whole cortex, not just a handful of emotion-specific regions

**What TRIBE does *not* do:**
- It does not measure anyone's actual brain — it *predicts* what an average subject's fMRI response would look like
- It does not output emotions — that's Feeling Engine's job
- It is not context-aware — it predicts the same response regardless of who the viewer is

Feeling Engine treats TRIBE's output as a best-available proxy for real brain response, not ground truth.

---

## The 7-category region reduction

TRIBE produces activation at ~20k cortical vertices. The Human Connectome Project (HCP) atlas groups these into ~180 cortical regions per hemisphere. Both are far too granular for emotional interpretation.

We collapse this down to **7 functional categories** chosen because each has a well-documented role in emotional processing:

| Category | What it captures | Example regions (HCP) |
|---|---|---|
| **core_affect** | General emotional valence and intensity | a32pr, p32, OFC, 25 |
| **interoception** | Body-state awareness, visceral sensing | FOP1–5, PoI1–2, MI, Ig |
| **regulation** | Executive control, emotional management | 9m, 10r, SCEF, 46, a9-46v |
| **social** | Social cognition, theory of mind, mentalizing | STS (a/p/v/d), TE1a, 7m |
| **reward** | Motivation, pleasure, reward prediction | OFC, pOFC, a32pr, 25 |
| **memory** | Retrieval, recognition, familiarity | EC, PreS, H, PHA1–3, RSC |
| **language** | Verbal/semantic processing, speech comprehension | 44, 45, 47l, A4, A5, STSda |

Sources for the category groupings include Barrett's work on interoception and the brain basis of emotion, Ochsner & Gross on cognitive regulation, and the HCP atlas documentation. The exact region lists are defined in `feeling_engine/adapters/brain_model/tribev2.py`.

**Caveats:**
- These groupings are coarse. Some regions (e.g. OFC) legitimately belong to multiple categories.
- The MVP adapter uses a simplified mean-activation proxy across vertices, not proper HCP parcellation. Moving to true HCP-based averaging is on the roadmap and will make the measurement more accurate.
- Functional labels like "regulation" or "reward" describe modal roles, not exclusive function. Real brains are messier than category labels suggest.

---

## Dimensional space: valence, arousal, body_focus

Before matching to vocabulary, we project the 7-category brain state into a three-dimensional space:

- **Valence** ∈ [−1, +1] — pleasant / unpleasant
- **Arousal** ∈ [−1, +1] — activated / deactivated
- **Body_focus** ∈ [−1, +1] — interoceptive / cognitive

The first two dimensions come from **Russell's circumplex model of affect** (Russell, 1980), which remains the most validated low-dimensional representation of emotion.

We add **body_focus** as a third dimension because the interoception-vs-cognition axis is what distinguishes emotions that feel similar in valence + arousal but differ in phenomenology. Fear and anger share high arousal and negative valence, but fear is much more body-dominant. Awe and reverence share positive valence but differ in body involvement. This axis is motivated by Barrett's **theory of constructed emotion** (Barrett, 2017), which argues that interoceptive signals are core to emotional differentiation.

Each timestep's brain state is projected to `(valence, arousal, body_focus)` using weighted contributions from the 7 categories. The projection weights are defined in `brain_to_emotion.py` and are transparent — you can inspect them to understand why a given brain state maps where it does.

---

## The 60-term vocabulary

The vocabulary is the heart of Layer 3. Each of the ~60 terms is defined with:

1. **Dimensional coordinates** — where it sits in (valence, arousal, body_focus) space
2. **Brain region expectations** — which of the 7 categories should be high, low, moderate, or suppressed
3. **Transition affinity** — which change types (onset, rapid_spike, decline, reversal, etc.) this term typically co-occurs with

Example from `vocabulary.yaml`:

```yaml
mortality_awareness:
  valence: -0.6
  arousal: 0.7
  body_focus: 0.9
  brain_expectations:
    interoception: very_high   # > 0.35
    language: negative         # < -0.05
    core_affect: high          # > 0.25
  transition_affinity: [rapid_spike, peak]
```

For a term to fire at a given timestep, the brain state must be close to the term's dimensional coordinates AND satisfy most of the brain-region expectations AND (ideally) match the transition type. Matching is scored out of 1.0 with each constraint contributing.

**Why this matters:** a label like `mortality_awareness` cannot fire from text sentiment alone. It requires measurable (or TRIBE-predicted) interoception activation *and* language suppression *and* a rapid spike. If the content says "death" but the brain prediction shows language activation and calm interoception, the term won't fire. This is what makes the labels falsifiable.

**Why 60 terms (approximately):**
- Too few → coarse, indistinguishable from valence/arousal alone
- Too many → terms overlap too much to disambiguate
- 60 is a working compromise informed by clustering studies of emotion-concept networks (Cowen & Keltner, 2017, find ~27 distinct affect categories; we extend to 60 to capture finer-grained states specific to long-form content like speeches and film)

Vocabulary design is not done — terms will be added and refined as real-world usage reveals gaps.

---

## Layer-by-layer rationale

### Layer 1 — TRIBE v2 (measurement)

Produces per-timestep brain activation. We treat its output as the measurement; we do not modify it. Adapter code aggregates vertex-level predictions into the 7 categories.

### Layer 2 — Change detection

The emotional arc is defined by *transitions*, not static states. Two seconds of interoception = +0.47 is less interpretively rich than the *moment* interoception spiked from +0.05 to +0.47.

Layer 2 scans the time series per category, detects deltas exceeding a threshold (default 0.08), and classifies each into a transition type: `onset`, `rapid_spike`, `gradual_rise`, `peak`, `plateau`, `decline`, `rapid_drop`, `reversal`, or `baseline`. These transition types matter because the vocabulary knows which terms typically co-occur with which transitions — a `rapid_spike` makes `awe` and `mortality_awareness` more likely; a `gradual_rise` makes `persuasion` and `trust` more likely.

### Layer 3 — Dimensional mapping + vocabulary matching

For each timestep:
1. Project brain state to (valence, arousal, body_focus)
2. Compute dimensional distance to each vocabulary term
3. Check brain-region expectations: how many are met?
4. Check transition affinity: does the current transition type match any of this term's affinities?
5. Combine into a score; return top-K labels

This is the "brain-grounded" part: labels only rise to the top when their dimensional coordinates, brain expectations, AND transition affinity all align with the measurement.

### Layer 4 — LLM synthesis (optional, content + context aware)

Layer 3 produces correct-*family* labels (awe, composure, clarity). Layer 4 narrows those to correct-*specific* labels by passing the brain data + Layer 3 candidates + the actual content text + optional viewer context to a language model (Claude Sonnet 4.6 by default).

Example:
- Layer 3 output at Jobs death pivot: `awe` (score 0.77, HIGH)
- Layer 4 with content "no one wants to die" + context "viewer knows Jobs survived cancer": `mortality_awareness` (HIGH), `body_response` (HIGH), `defiance` (MODERATE)

Layer 4 is where **context modifies interpretation, not measurement**. The same brain data + different context produces a different arc. This is the right place for viewer-specific reasoning because TRIBE itself is context-free.

Layer 4 is optional. When run with `--no-layer4`, you get Layer 3 labels only.

### Layer 5 — Confidence scoring

Every label is tagged with one of four confidence levels:

| Level | Means |
|---|---|
| **HIGH** | Strong match score, most brain expectations met, transition affinity aligned |
| **MODERATE** | Reasonable match; 1–2 expectations unmet |
| **LOW** | Marginal match; multiple expectations unmet |
| **SPECULATIVE** | Score passes threshold but grounding is weak |

Confidence is computed post-hoc from the brain grounding, not from LLM guesswork. This lets downstream consumers (research, editorial, paid analysis) treat HIGH labels as actionable and SPECULATIVE labels as directions for further inquiry.

---

## Fire: cross-domain matching

Fire runs on top of the completed arc. Given a sequence of feelings, it finds corpus entries (literature, poetry, speeches, letters, etc.) whose emotional signatures most closely match.

Matching combines two signals:
- **Cosine similarity** over vocabulary-space vectors (do the same feelings appear?) — 70% weight
- **Levenshtein sequence distance** (do the feelings appear in the same order?) — 30% weight

The 70/30 weighting was chosen to prioritize vocabulary overlap while still rewarding structural similarity. Different weightings produce meaningfully different rankings; this is a design parameter, not a discovered truth.

Fire's utility is not "tell me what this piece of content is" — it's "what else has this emotional structure?" The answers are often cross-cultural and cross-period, which is the point: it surfaces precedents that the surface features of content don't reveal.

---

## What this does NOT do

- **Detect actual brain states in real people.** TRIBE predicts what an average viewer's brain response might be based on its training data. Individual variation is not modeled.
- **Predict emotional *response* with clinical reliability.** Outputs are research tools, not diagnostic instruments.
- **Generate content from feelings.** That's the "Fire generative" direction, on the Phase 6 roadmap — not yet implemented.
- **Handle video or image content.** Currently audio + text only. Video support is on the roadmap.
- **Produce ground-truth emotional arcs.** The methodology is grounded but not validated against human self-report at scale. That validation is part of the research agenda.

Use the outputs as *interpretive hypotheses with explicit grounding*, not as claims of fact.

---

## Caveats and known limitations

1. **HCP parcellation proxy.** The current TRIBE adapter uses a simplified mean-across-vertices approach rather than true HCP region-based averaging. Moving to proper parcellation is a priority upgrade and will change exact activation values.

2. **Vocabulary design is iterative.** The current 60 terms reflect working intuitions informed by dimensional emotion research, but they have not been validated against inter-rater agreement or clinical taxonomies. Some terms will be removed, merged, or refined as usage reveals what's working.

3. **Layer 4 depends on an LLM.** LLMs can hallucinate rationales that sound plausible but aren't grounded in the brain data. The prompt structure requires explicit brain-grounding citations, but this is a weakness to watch.

4. **TRIBE predictions are not real brain data.** Everything downstream inherits whatever biases or gaps exist in TRIBE's training distribution. Content that looks very different from TRIBE's training data (experimental audio, non-naturalistic compositions) may produce unreliable predictions.

5. **Cost scales with analysis length.** Modal TRIBE cost is roughly linear in audio duration. A 60-second clip is ~$0.08; a 15-minute speech is ~$1.20. Budget accordingly.

---

## References

- Barrett, L. F. (2017). *How Emotions Are Made: The Secret Life of the Brain.*
- Cowen, A. S., & Keltner, D. (2017). Self-report captures 27 distinct categories of emotion bridged by continuous gradients. *PNAS, 114*(38).
- Ochsner, K. N., & Gross, J. J. (2005). The cognitive control of emotion. *Trends in Cognitive Sciences, 9*(5).
- Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology, 39*(6).
- Meta FAIR. TRIBE v2 model and codebase. https://github.com/facebookresearch/tribev2
- Glasser et al. (2016). A multi-modal parcellation of human cerebral cortex. *Nature, 536*.
