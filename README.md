# Feeling Engine

**A brain-response-grounded emotional analysis pipeline.**

Feeling Engine takes content (audio, text, or pre-computed brain predictions) and produces a timestep-by-timestep emotional arc — not guessed from surface sentiment, but derived from predicted brain activation patterns via Meta FAIR's [TRIBE v2](https://github.com/facebookresearch/tribev2) model.

It answers the question: **"What does this content feel like to a listener, moment by moment — and why?"**

> Alpha. Interfaces are stable enough to use. Expect sharp edges.

---

## What It Does

```
Content (audio / text) → TRIBE v2 (brain prediction)
                       → Change detection (where the brain state shifts)
                       → Dimensional mapping (valence / arousal / body-focus)
                       → LLM synthesis (content- and context-aware labels)
                       → Confidence scoring
                       → Emotional arc + cross-domain matches
```

Output example (Steve Jobs, Stanford 2005 — real run on the death-pivot moment):

```
Timestep 59 ⚡
  Brain: interoception=+0.472 | regulation=+0.339 | core_affect=+0.329 | reward=+0.186
  Dimensions: V=+0.521  A=+0.848  B=+0.907
  Change: interoception rising (rapid_spike, Δ=+0.199)
  → awe (score=0.77) [HIGH] — interoception, core_affect, language match
    expected pattern; transition type (rapid_spike) aligns with awe's affinity.
```

With Layer 4 enabled (Claude reads the actual words + optional viewer context), `awe` is refined to a content-specific label like `mortality_awareness` when the content is "no one wants to die" — Layer 3 identifies the brain-response family; Layer 4 names the specific emotion.

Plus Fire, a cross-domain matcher that surfaces precedents in other media with the same emotional structure (e.g. "this speech shares an arc with Keats' Ode to a Nightingale and a Civil War last letter").

---

## Installation

```bash
git clone https://github.com/humansourcecode/feelingengine
cd feelingengine
pip install numpy pyyaml anthropic modal elevenlabs  # core
# torch — only needed if you run TRIBE locally instead of via Modal
```

Proper packaging (`pyproject.toml`) is on the roadmap. Until then, install dependencies directly.

Copy `.env.example` to `.env` and fill in the keys you'll use. You do **not** need all of them — only the ones corresponding to the entry points you want:

| Entry point | Keys needed |
|---|---|
| Pre-computed TRIBE profiles (JSON) | `ANTHROPIC_API_KEY` (for Layer 4, optional) |
| Audio file → TRIBE | `MODAL_TOKEN_ID/SECRET`, `HUGGINGFACE_ACCESS_TOKEN` |
| Text → TTS → TRIBE | above + `ELEVENLABS_API_KEY` |

---

## Quickstart

### Mode 1 — Pre-computed TRIBE profiles (free, instant)

```bash
python examples/analyze_speech.py --profiles path/to/profiles.json
```

### Mode 2 — Audio file (requires Modal deployment)

```bash
python examples/analyze_speech.py --audio speech.mp3
```

### Mode 3 — Text (requires ElevenLabs + Modal)

```bash
# 1. List your ElevenLabs voices to find one that matches your content's speaker
python examples/analyze_speech.py --list-voices

# 2. Analyze with a voice that fits (gender, age, register)
python examples/analyze_speech.py --text speech.txt --voice-id <voice_id>
```

**Why the voice choice matters:** TRIBE predicts brain activation from *audio features* (timbre, prosody, pace), not just words. A dramatic male political speech synthesized in a neutral female narrator voice produces a brain prediction for that narrator reading the text — not for the original delivery. Match the voice to the intended speaker for meaningful results.

**Better yet: if real audio of the content exists, skip TTS entirely — use `--audio` on the original recording.**

### Add context for Layer 4 refinement

```bash
python examples/analyze_speech.py --profiles profiles.json \
  --content "No one wants to die..." \
  --context "Viewer knows Jobs died in 2011"
```

### Use as a library

```python
from feeling_engine.pipeline import FeelingPipeline
from feeling_engine.adapters.brain_model.tribev2 import TRIBEv2Adapter
from feeling_engine.fire.matcher import FireMatcher

pipeline = FeelingPipeline(brain_adapter=TRIBEv2Adapter())
arc = pipeline.analyze_profiles("profiles.json")
print(pipeline.format_arc_text(arc))

matcher = FireMatcher()
matches = matcher.match_arc(arc, top_k=5)
print(matcher.format_matches(matches))
```

---

## Architecture

Five layers, loosely coupled:

| Layer | Role | File |
|---|---|---|
| **1** — Brain prediction | TRIBE v2 predicts per-vertex brain activation from content | `adapters/brain_model/tribev2.py` |
| **2** — Change detection | Finds timestep-level transitions (onset, rapid_spike, reversal, …) | `translator/change_detector.py` |
| **3** — Dimensional mapping | Maps brain state → valence/arousal/body-focus → 60-term emotional vocabulary | `translator/brain_to_emotion.py` + `vocabulary.yaml` |
| **4** — LLM synthesis | Claude/Gemini refines labels using the actual content + optional viewer context | `translator/llm_synthesizer.py` |
| **5** — Confidence | HIGH / MODERATE / LOW / SPECULATIVE scoring based on brain-grounding strength | `translator/confidence.py` |

**Fire matcher** (`fire/matcher.py`) runs on top of the output arc: cosine + Levenshtein similarity against a corpus of pre-annotated media to find cross-domain precedents.

### Design principles

- **Brain measurement first, interpretation second.** TRIBE produces a context-free prediction. Layer 4 is where content and viewer context modify the *interpretation* — the scan itself doesn't change.
- **Adapters are swappable.** TRIBE v2 today, other brain models tomorrow; Modal today, local GPU or another provider tomorrow; ElevenLabs today, any TTS tomorrow.
- **Every emotion label is brain-grounded.** The 60-term vocabulary has explicit region expectations (interoception high, language suppressed, etc.) — labels only fire when the brain pattern supports them.

---

## Costs (verified 2026-04-16)

| Service | Rate | Typical analysis |
|---|---|---|
| Modal A100-40GB | $0.000583/sec | ~$0.08 per 60-sec TRIBE run |
| Anthropic Claude Sonnet 4.6 | $3 / $15 per MTok | ~$0.01 per Layer 4 refinement |
| ElevenLabs | Monthly subscription | Covered by plan |

**Modal gives $30/month free credits on the Starter plan** — enough for ~375 runs.

Full pipeline analysis of a 60-second clip: **~$0.11.**

---

## Licensing — Read This

**Feeling Engine code: MIT.** Use it however you want.

**TRIBE v2 (the brain model): CC BY-NC 4.0.** Non-commercial only.

What this means:
- Research, education, and personal use — fine
- Non-commercial YouTube content and blog posts — fine
- **Paid commercial services that use TRIBE predictions — not without a license from Meta**

You can still build commercial products on the Translator + Fire layers (Layers 2–5) if you supply brain predictions from another source. For commercial use of TRIBE itself, contact [Meta FAIR](https://github.com/facebookresearch/tribev2).

---

## What's Tested

- ✅ **Mode 1 end-to-end** — profile load → change detection → Layers 2/3/5 → Fire matcher, verified against Jobs Stanford data (2026-04-17)
- ✅ Layer 4 synthesis with Claude Sonnet 4.6 (separate run)
- ✅ Fire matcher + sample corpus (8 entries, 6 domains)
- ⚠️ **Modes 2–3 (live Modal + ElevenLabs)** — adapter code written, not yet run end-to-end against production APIs
- ⚠️ No pytest suite yet — coming with the packaging work

See `examples/analyze_speech.py` for the canonical usage.

---

## Project Status

Built by [Landon Fears](https://feelingengineer.com) as the research tool behind [Human Source Code](https://youtube.com/@HumanSourceCode), a channel investigating how content produces feeling.

Follow: [@humancodebase](https://x.com/humancodebase) · [@feelingengineer](https://instagram.com/feelingengineer)

---

## Credits

- **TRIBE v2** — [Meta FAIR](https://github.com/facebookresearch/tribev2) · brain prediction model, CC BY-NC 4.0
- **Emotional vocabulary** — grounded in Russell's circumplex model + Barrett's theory of constructed emotion
- **Fire corpus (sample)** — 8 public-domain / widely-cited excerpts for demonstration

---

## Contributing

Open an issue first for anything non-trivial. Style: direct, minimal, brain-grounded. Labels must be justifiable from the brain data, not from surface sentiment.
