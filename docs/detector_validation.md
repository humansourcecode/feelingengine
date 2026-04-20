# Detector Validation — Mechanism Label Pipeline

**Status:** Draft for review
**Companion doc:** `mechanism_labels.md` (the vocabulary itself)
**Audience:** Feeling Engine pipeline designers, anyone validating or extending the mechanism label layer

---

## Purpose

The mechanism label vocabulary uses multiple signal sources (TRIBE + LLM-on-transcript + occasional auxiliary signals). This is intentional — some labels can't be distinguished from brain state alone. But multi-source detection only counts as a *reputable* source if:

1. Each label specifies which sources are load-bearing
2. Disambiguation signals are operationally defined, not hand-waved
3. Detectors are measured against human-labeled ground truth
4. Known failure modes are documented
5. Accuracy is tracked over time as the pipeline evolves

This doc specifies the validation protocols.

---

## Validation philosophy

Three claims any label must support to be accepted into production:

### 1. The TRIBE signature claim

Every label asserts a TRIBE signature — a pattern in the 7-axis output that corresponds to the mechanism. For Tier 1 labels, this signature is *unique*. For Tier 2 and Tier 3, it narrows a candidate set.

**Validation requirement:** exemplars labeled by humans must produce *statistically similar* TRIBE signatures. If the three exemplars for `vulnerability-transfer` produce wildly different TRIBE outputs, either the signature claim is wrong or the exemplars are mis-matched.

### 2. The disambiguation claim (Tier 2 / Tier 3)

For labels with non-unique TRIBE signatures, the detector spec lists disambiguation signals (linguistic markers, prosody, framing). The claim: these signals reliably distinguish the target label from its neighbors.

**Validation requirement:** on a held-out corpus where both label A and its neighbor B appear, the detector using the specified disambiguation signals must achieve the accuracy target (see below) on distinguishing them.

### 3. The transfer claim

The label is content-agnostic — it applies across different topics, media, and domains. This is what makes the "same arc, different stories" premise of cross-medium arc matching work.

**Validation requirement:** exemplars must span at least 3 maximally different content domains. If all exemplars for a label come from poetry, the transfer claim is unsupported.

---

## Accuracy targets

| Tier | Target accuracy | Rationale |
|------|-----------------|-----------|
| **Tier 1** | >95% | TRIBE signature is unique; detector is deterministic signal-processing. High accuracy expected. |
| **Tier 2** | >80% | LLM-on-transcript disambiguation introduces classification uncertainty. 80% matches reasonable text-classification baselines. |
| **Tier 3** | >70% | Multi-source, ambiguous signatures, subtle phenomenology. Below 70% means the label is unreliable enough to warrant human review every time. |

Labels that fail their tier's accuracy target after validation are candidates for:
- Tier demotion (move from Tier 2 to Tier 3)
- Merger (combine with a neighbor label)
- Refactor (rework the definition)
- Removal (if no path to reliable detection)

---

## Corpus construction

Validation requires labeled corpora. Here's how we build them.

### Minimum sizes per label

| Tier | Positive examples | Negative examples (neighbors) | Total per label |
|------|------|------|------|
| Tier 1 | 20 | 30 (from other Tier 1 labels) | 50 |
| Tier 2 | 30 | 30 (from specific neighbor labels it shares signature with) | 60 |
| Tier 3 | 40 | 40 | 80 |

For 28 labels, that's roughly 50 × 18 (T1) + 60 × 6 (T2) + 80 × 4 (T3) = **~1,580 labeled instances** for full validation. That's large but not prohibitive — can be built incrementally.

### Corpus sourcing

Candidate corpora in priority order:

1. **Hand-curated domain-diverse corpus** — literature, poetry, speeches, advertising, scripts, viral content, letters, news, philosophy, lyrics. A small (~30-50 entry) coverage-check set labeled at piece level; every mechanism should be exemplified in at least 3 distinct domains.
2. **Beat-level excerpts** — 10-30 second clips extracted from longer content, labeled at moment level.
3. **Synthetic examples** — LLM-generated exemplars for rare labels (tests vocabulary but doesn't validate reliability).
4. **Public benchmark datasets** — emotion/affect corpora with existing annotations (e.g., SEMAINE, MELD) adapted where possible.

### Human labeling protocol

For each candidate instance:

1. **Two annotators label independently** using only `mechanism_labels.md` as reference.
2. **Inter-annotator agreement** computed (Cohen's kappa or percent agreement).
3. **Disagreements reviewed** — if the two annotators can't align after review, the instance is ambiguous. Either improve the label definition or exclude the instance from the validation set.
4. **Agreed labels become ground truth.**

Target inter-annotator kappa: >0.7 (substantial agreement). Below that, the label definition is too vague.

---

## Per-label validation requirements

For each label, the validation suite tracks:

### Tier 1 labels

- **Coverage test:** detector fires on ≥90% of held-out positive examples
- **False-positive rate:** detector fires on ≤5% of negative examples (other-label instances)
- **Signature stability:** TRIBE signatures across positive examples have cosine similarity ≥0.7 to the label's canonical signature
- **Domain transfer:** positive examples from ≥3 content domains; accuracy does not drop >10% between domains

### Tier 2 labels

- Everything in Tier 1, PLUS:
- **Disambiguation test:** when placed next to neighbor labels (ones sharing TRIBE signature), detector achieves >80% correct classification
- **Linguistic signal audit:** each listed disambiguation signal is verified to actually help (ablate it; if accuracy doesn't drop, it's not load-bearing)
- **Failure-mode coverage:** the detector's known failure modes (from spec) are verified on a held-out set — do they actually fail as expected, or has the behavior changed?

### Tier 3 labels

- Everything in Tier 2, PLUS:
- **Human review trigger:** detector confidence below threshold (TBD per label) triggers mandatory human review
- **Alternate-source supplementation:** if LLM-on-transcript alone isn't reliable enough, flag where audio-prosody or other signals could be added
- **Refactor review:** Tier 3 labels are reviewed every 6 months for refactor / merge / removal decisions. Persistent low accuracy is a signal the label is conceptually unclear.

---

## Detector spec template

Every Tier 2 / Tier 3 label gets a detector spec with this structure (same as in `mechanism_labels.md`):

```
Label: <name>
Tier: <2 | 3>
─────────────────────────────────────────────
Shared TRIBE signature with: <neighbor labels>
Disambiguation source: <TRIBE | transcript | audio | contextual | combinations>
Disambiguation signals:
  - <concrete signal 1>
  - <concrete signal 2>
  - ...
Detector confidence metric: <how 0-1 score is computed>
Validation corpus:
  - Positive: <source / size>
  - Negative: <source / size — must include neighbor labels>
Accuracy target: <tier-appropriate>
Known failure modes:
  - <mode 1: conditions + expected behavior>
  - <mode 2: ...>
Review trigger: <when to re-validate; default every 6 months>
```

---

## Known hard cases — what to expect

These are the labels most likely to fail validation or need refactoring. Name them upfront so we're not surprised.

1. **`expansion`** — no unique TRIBE signature. Relies entirely on LLM detection of grandness/scale language. Likely accuracy <70%. May merge into `affect-rise` with a `valence: positive, scope: expansive` attribute.
2. **`boundary-establish`** — vague TRIBE signature ("social sharpening"). Distinguishing self-definition refusal from contest refusal is conceptually thin. Candidate for merger with `opposition` under a valence/target parameter.
3. **`dissonance`** — requires semantic-content-vs-affective-response mismatch detection. Ironic content is hard; deadpan delivery is borderline. Accuracy target 70% may be generous.
4. **`withdrawal` vs `restraint`** — phenomenologically distinct (hiding vs. holding) but TRIBE signatures may overlap. Inter-annotator agreement is the first test; if humans can't agree, the labels need tightening.
5. **`threshold-approach` vs `anticipation`** — both involve rising reward. Distinguishing "tension without payoff" from "expectation of payoff" may collapse in practice.

For each of these, budget extra validation time and be open to refactor.

---

## Ongoing validation

After initial validation, the system needs continuous checking:

1. **Per-video validation.** Each video analyzed by the pipeline gets spot-checked — flag moments where the detector fires with low confidence or where a human would disagree.
2. **Drift detection.** If detector accuracy on a rolling sample drops >5%, something has changed (TRIBE update, prompt drift, corpus shift). Investigate.
3. **Label-level dashboards.** Track per-label usage frequency, confidence distributions, disagreement rates. Labels that never fire or always disagree need attention.
4. **Quarterly vocabulary review.** Every 3 months, review the whole vocabulary. Add labels where gaps are found; prune labels that don't earn their place.

---

## What validation is NOT

A few things this pipeline does *not* claim:

1. **That mechanism labels are neuroscientific categories.** They're content-level labels grounded in brain-state signatures. The philosophy of how words describe brain states is hard; we're making a pragmatic engineering choice about what's reliably detectable.
2. **That TRIBE is the only source of truth.** TRIBE is authoritative for brain state. Mechanism labels are multi-source derived constructs, explicitly.
3. **That 100% accuracy is achievable.** Every detection system has errors. The goal is accuracy high enough to be *useful* — to identify patterns creators can deploy, and to find cross-domain matches worth studying.
4. **That labels are permanent.** Generation 1 is a starting point. Labels will be added, merged, split, removed as we learn. Validation is ongoing.

---

## Open questions

1. **Who does the human labeling?** For validation corpus, two annotators is standard. Who are they? Can Claude serve as one annotator if it's disclosed and measured?
2. **What's the budget for validation?** ~1,580 labeled instances, two annotators each = ~3,160 human labels. Depending on annotator speed and cost, this is tens of hours minimum. Worth naming upfront.
3. **When does a failing Tier 3 label get cut?** Three strikes at quarterly review? One year of persistent low accuracy? Define the rule so it doesn't become a permanent gray-area.
4. **How do we publish validation results?** For channel credibility, "our labels have X% accuracy on Y validation corpus" is a meaningful transparency claim. Worth building toward making this public.
