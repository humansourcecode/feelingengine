# Feeling Engine

[![Tests](https://github.com/humansourcecode/feelingengine/actions/workflows/test.yml/badge.svg)](https://github.com/humansourcecode/feelingengine/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](pyproject.toml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/humansourcecode/feelingengine/blob/main/examples/colab/feeling_engine_demo.ipynb)

**A brain-response-grounded pipeline for analyzing content by its mechanisms, not its topic.**

Feeling Engine takes content (audio, text, video, or pre-computed brain predictions) and produces:

1. **A timestep-by-timestep emotional arc** — derived from predicted brain activation patterns via Meta FAIR's [TRIBE v2](https://github.com/facebookresearch/tribev2), not guessed from surface sentiment.
2. **Mechanism detection** — 28 content-agnostic labels (body-turn, pattern-break, vulnerability-transfer, drift, …) that fire when brain-response patterns match their signatures. Works across genres, mediums, and durations. Runs TRIBE-free via an LLM-detector variant when you want a cheaper path.
3. **Arc mining** — build a library of mechanism-arcs from outlier content you want to learn from. Each arc lives in vocabulary form (shareable, topic-independent) and can be matched across disjoint mediums.
4. **Rendering + export** — turn any arc into brain-hemisphere images (canonical atlas, no TRIBE at runtime) and platform-specific asset bundles (Substack / X / YouTube) so analysis flows directly into publication.

It answers two questions that most content analysis tools can't:

- **"What does this content feel like to a listener, moment by moment — and why?"**
- **"What other content, regardless of topic or medium, produces the same structural journey?"**

> Alpha. Interfaces are stable enough to use. Expect sharp edges.

**Want to see it first?** Three zero-install entry points:
- [**Walkthrough**](examples/walkthrough.md) — narrated reading of one 60-second arc (Steve Jobs death pivot), showing what the engine measures and names.
- [**Colab demo**](https://colab.research.google.com/github/humansourcecode/feelingengine/blob/main/examples/colab/feeling_engine_demo.ipynb) — runs the interpretation pipeline end-to-end on four pre-mined arcs. Zero setup.
- [**Static inspector**](examples/inspector/) — brain timeline + mechanism firings + hemisphere renderer, playable locally via any static HTTP server.

---

## What It Does

```
Content (YouTube URL / audio / text / video file)
  → [optional] Clip trimmer       — Gemini strips intros/ads/outros
  → Mechanism detector            — TRIBE v2 (grounded)  OR  LLM Detector v2 (TRIBE-free)
  → Mechanism arc + sequences     — 28 labels, 10 named sequences, timestamps, intensity
  → Arc library                   — SQLite, queryable by facet/mechanism/sequence
  → Rendering + platform export   — brain images, clips, stills, per-platform drafts
  → [optional] Translator         — emotion-label refinement (awe, mortality_awareness, ...)
```

Output example (Steve Jobs, Stanford 2005 — bundled arc, 60-second excerpt):

```
Clip: 60s · 75 mechanism firings · 2 named sequences

Sequence: awakening-build  (t=2-59s)
  t=2s   universal-recognition   "No one wants to die..."
  t=22s  body-turn               "...death is the destination we all share"
  t=25s  stakes-compression      "...life's change agent"
  t=58s  inward-pivot            "...the new is you"

Sequence: contemplation-spiral  (t=11-48s)
  t=11s  word-focus      →  t=14s contemplation  →  t=24s recognition  →  t=47s body-turn

Top mechanisms: evocation (10×), recognition (9×), word-focus (8×),
                intimacy-turn (7×), body-turn (5×), contemplation (5×)
```

The arc is **content-agnostic by construction** — the same 28-label vocabulary fires across speeches, POV narratives, wordless documentaries, and adversarial explainers (four bundled arcs demonstrate this). Arcs compose into a library queryable by mechanism or sequence, enabling cross-domain discovery ("what other content fires `awakening-build` at high intensity?").

Emotion-label refinement is available as an optional **Translator** pass (5 layers, Claude/Gemini-driven) when you want named emotions on top of the mechanism structure — but mechanisms are the primary output.

---

## Installation

### 1. System dependencies

```bash
# macOS
brew install ffmpeg python@3.11

# Ubuntu / Debian
apt install ffmpeg python3.11-venv
```

Feeling Engine uses `ffprobe` (shipped with ffmpeg) for audio duration. Python 3.10+ required.

### 2. Install the package

```bash
git clone https://github.com/humansourcecode/feelingengine
cd feelingengine

python -m venv .venv && source .venv/bin/activate

# Core install (Mode 1 only — offline profile analysis)
pip install -e .

# Recommended — installs all optional provider SDKs
pip install -e ".[all,dev]"
```

Available extras: `[tts]` (ElevenLabs), `[llm]` (Anthropic), `[gemini]` (Google), `[compute]` (Modal), `[all]` (everything), `[dev]` (pytest + coverage).

### 3. Smoke test — zero API keys needed

The simplest sanity check — works with just the core install:

```bash
python examples/analyze_speech.py \
  --profiles tests/unit/fixtures/tiny_profiles.json \
  --no-layer4
```

If that prints an emotional arc and cross-domain matches, your install is correct. No API keys or external services used.

**Prefer to try it without installing?** Open the [**Colab notebook**](https://colab.research.google.com/github/humansourcecode/feelingengine/blob/main/examples/colab/feeling_engine_demo.ipynb) — runs the interpretation pipeline end-to-end on four pre-mined arcs (speech, POV narrative, visual/music, adversarial explainer). Zero setup; no HF tokens, no Modal account.

**To also run the test suite** (requires `[dev]` extras):

```bash
pip install -e ".[dev]"
python -m pytest tests/unit                 # 22 unit tests
```

Use `python -m pytest` (not bare `pytest`) to guarantee it runs against your venv's Python, not a globally-installed pytest.

### 4. Configure API keys (only for paid features)

```bash
cp .env.example .env
# Edit .env — fill in ONLY the keys for features you want
```

You don't need all keys. Each entry point has its own minimum:

| Entry point | Keys required |
|---|---|
| Pre-computed TRIBE profiles (`--profiles`) | None |
| Pre-computed profiles + Layer 4 refinement | `ANTHROPIC_API_KEY` |
| Audio file → TRIBE (`--audio`) | `MODAL_TOKEN_ID/SECRET`, `HUGGINGFACE_ACCESS_TOKEN` |
| Text → TTS → TRIBE (`--text`) | All of the above + `ELEVENLABS_API_KEY` |

Running a mode without the required keys produces a clear error pointing you at the missing key. Nothing will crash cryptically.

### 5. Deploy TRIBE to Modal (required for `--audio` and `--text` modes)

TRIBE v2 model weights are gated on HuggingFace — [request access](https://huggingface.co/facebook/tribev2), then:

```bash
modal setup                              # one-time Modal auth
export HUGGINGFACE_ACCESS_TOKEN=hf_...   # your token
modal deploy deploy/tribe_modal.py       # first deploy takes ~10 min
```

See [`deploy/README.md`](deploy/README.md) for details, costs, and teardown.

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
from feeling_engine.mechanisms.api import detect_mechanisms, detect_sequences

pipeline = FeelingPipeline(brain_adapter=TRIBEv2Adapter())
arc = pipeline.analyze_profiles("profiles.json")
print(pipeline.format_arc_text(arc))

# Layer on mechanism detection
from feeling_engine.mechanisms.tier1_detectors import compute_axis_stats
axis_stats = compute_axis_stats(pipeline.profiles)
mech_arc = detect_mechanisms(
    tribe_categories=pipeline.profiles,
    transcript=pipeline.transcript,
    axis_stats=axis_stats,   # enables σ-mode
)
sequences = detect_sequences(mech_arc)
```

---

## Mechanism Detection

Beyond the 60-term emotional vocabulary, Feeling Engine detects **mechanisms** — content-agnostic structural moves (body-turn, pattern-break, vulnerability-transfer, drift, recognition, …). A mechanism fires when brain-response patterns match its signature. 28 mechanisms ship in v1, with 10 named narrative sequences (joke-structure, intimacy-deepening, contemplation-spiral, …) composed from them.

```python
from feeling_engine.mechanisms.api import detect_mechanisms, detect_sequences
from feeling_engine.mechanisms.tier1_detectors import compute_axis_stats

# Load or generate TRIBE profiles first, then:
axis_stats = compute_axis_stats(profiles)  # per-video normalization stats
arc = detect_mechanisms(
    tribe_categories=profiles,
    transcript=transcript,
    axis_stats=axis_stats,   # enables σ-mode (see below)
)
sequences = detect_sequences(arc)
```

### Two threshold modes

- **Absolute (default):** thresholds are raw magnitudes, tuned against a speech-heavy baseline. Predictable, but under-fires on quieter content types (music/visual).
- **σ-mode (pass `axis_stats=...`):** thresholds are interpreted as standard-deviation multipliers relative to *this video's own* signal distribution. Content-agnostic by construction; on a silent-video corpus it unlocks 10× more firings vs. absolute. Faithful port of absolute defaults via Jobs-baseline calibration — existing content's output is preserved within ~2%. See `docs/mechanism_labels.md` and `docs/detector_validation.md`.

Full mechanism vocabulary, TRIBE signatures, and cross-domain exemplars: [`docs/mechanism_labels.md`](docs/mechanism_labels.md).

---

## Arc Mining

Feeding Feeling Engine a URL builds an **arc library** — a collection of mechanism-form arcs extracted from outlier content. Arcs are:

- **Topic-independent** — stored in the shared 28-label vocabulary, not tied to the source's subject matter
- **Medium-agnostic** — the same arc shape can describe a 5-second video, a 2-hour film, a song, or a podcast
- **Commercial-safe** — derived from (CC BY-NC) TRIBE output but the arcs themselves are vocabulary-form (your IP)

```bash
# Mine one YouTube URL (channel median auto-computed via yt-dlp)
python -m feeling_engine.mining.arc_miner mine <YouTube URL> \
  --db arc_library.db --niche "architecture-history"

# Mine a batch from a urls file
python -m feeling_engine.mining.arc_miner mine-batch urls.tsv --db arc_library.db

# List the library
python -m feeling_engine.mining.arc_miner list --db arc_library.db
```

Both absolute and σ-mode arcs are stored per entry, so downstream consumers can match by either. See [`examples/arcs/`](examples/arcs/) for four pre-mined bundles across disparate content types (Steve Jobs commencement, visual/music documentary, POV narrative, adversarial explainer) — useful as fixtures for testing, prototyping, or running the demo notebook without a Modal account. The Jobs bundle also ships with **pre-rendered brain hemispheres for all 60 timesteps** so the inspector works without any runtime rendering.

## Preprocess — Clip Trimmer

Real-world content is rarely pre-trimmed. A YouTube review might have a 30-second intro, mid-roll ad, and outro credits that muddy the mechanism signal. **Clip Trimmer** uses Gemini to densely classify every frame (scene / branded / title / commentary / ad / transition) and emits an ffmpeg cut list that preserves only the content itself.

```python
from feeling_engine.preprocess.clip_trimmer import trim_by_dense_classification

result = trim_by_dense_classification(
    video_path="raw_video.mp4",
    output_path="trimmed.mp4",
    content_type="commencement speech",
    sample_interval=2.0,
)
print(f"Trimmed {result.kept_sec:.1f}s of {result.total_sec:.1f}s content")
```

## LLM Detector v2 — TRIBE-free mechanism detection

Full TRIBE requires Modal + HuggingFace access and costs ~$2 per 10-minute video. When you want mechanism detection *without* running a brain model — for exploration, or for content where TRIBE isn't needed — **LLM Detector v2** classifies mechanisms directly from multimodal inputs (frames + audio + Whisper transcript).

```python
from feeling_engine.mechanisms.llm_detector import detect_from_video

arc = detect_from_video(
    video_path="clip.mp4",
    content_type="movie scene",     # biases the prompt's semantic frame
    chunk_frames=40,                # balances throughput vs PROHIBITED_CONTENT locality
)
```

Output schema matches the TRIBE-grounded detector, so the same arc library / inspector / platform-export tooling works on both. TRIBE-grounded mechanisms get `tribe_grounded=1`; LLM-detected get `tribe_grounded=0`.

## Rendering — brain maps + platform export

Once you have an arc, **rendering** turns it into publication-ready deliverables:

```python
from feeling_engine.rendering.brain_renderer import render_mechanism_brain
from feeling_engine.rendering.analysis_assets import render_analysis_bundle
from feeling_engine.rendering.platform_export import export_all

# 1. Canonical brain images for any mechanism (no TRIBE at runtime)
render_mechanism_brain(
    label="vulnerability-transfer",
    output_path="vt_brain.png",
    view="lateral_both",          # or lateral_left / lateral_right / medial_both / all
    intensity=0.8,
)

# 2. Extract key moments + clips + stills + brain images from an arc
bundle = render_analysis_bundle(
    video_path="clip.mp4",
    arc=arc,
    sequences=sequences,
    out_dir="analysis/",
    n_key_moments=5,
)

# 3. Export per-platform drafts (Substack essay / X thread / YouTube script)
export_all(bundle, platforms=["substack", "x", "youtube"])
```

Brain renderer uses the public HCP-MMP cortical atlas + 28 per-mechanism activation signatures — so any mined arc, mined years ago on whatever hardware, can be re-rendered without re-running TRIBE. Platform export emits ready-to-paste drafts with timestamped key moments, embedded brain/scene images, and opening hooks tailored per platform.

---

## Architecture

Five translator layers plus downstream modules for mining, rendering, and export:

| Layer | Role | File |
|---|---|---|
| **1** — Brain prediction | TRIBE v2 predicts per-vertex brain activation from content | `adapters/brain_model/tribev2.py` |
| **2** — Change detection | Finds timestep-level transitions (onset, rapid_spike, reversal, …) | `translator/change_detector.py` |
| **3** — Dimensional mapping | Maps brain state → valence/arousal/body-focus → 60-term emotional vocabulary | `translator/brain_to_emotion.py` + `vocabulary.yaml` |
| **4** — LLM synthesis | Claude/Gemini refines labels using the actual content + optional viewer context | `translator/llm_synthesizer.py` |
| **5** — Confidence | HIGH / MODERATE / LOW / SPECULATIVE scoring based on brain-grounding strength | `translator/confidence.py` |
| **Preprocess** | Dense-frame Gemini classifier + ffmpeg-based clip trimming (scene vs ad / intro / commentary) | `preprocess/clip_trimmer.py` |
| **Mechanisms** | 28-label + 10-sequence mechanism detection. TRIBE-grounded (σ-mode supported) OR TRIBE-free via LLM detector v2 (multimodal frames + audio + transcript). | `mechanisms/` |
| **Mining** | URL → TRIBE → mechanism arc → sqlite arc library. Subcommands: `mine` / `mine-batch` / `list`. Trimodal routing (video/audio/text) handled inside `mine` based on source type. | `mining/arc_miner.py` |
| **Rendering** | Canonical brain renderer (no TRIBE runtime), platform asset export (Substack/X/YouTube), key-moment selection | `rendering/` |
| **Fire matcher** *(archived library fn)* | Corpus-based precedent matcher. Present for compatibility; new work uses mechanism/sequence queries on the arc library instead. | `fire/matcher.py` |

### Design principles

- **Brain measurement first, interpretation second.** TRIBE produces a context-free prediction. Layer 4 is where content and viewer context modify the *interpretation* — the scan itself doesn't change.
- **Adapters are swappable.** TRIBE v2 today, other brain models tomorrow; Modal today, local GPU or another provider tomorrow; ElevenLabs today, any TTS tomorrow.
- **Every emotion label is brain-grounded.** The 60-term vocabulary has explicit region expectations (interoception high, language suppressed, etc.) — labels only fire when the brain pattern supports them.

For the full methodology — why the 7 brain-region categories, how dimensional mapping grounds the vocabulary, what this pipeline does NOT do — see [`docs/methodology.md`](docs/methodology.md).

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

You can still build commercial products on the Translator layers (2–5), the LLM detector, clip trimmer, brain renderer, and platform export if you supply brain predictions from another source (or don't use TRIBE at all — LLM Detector v2 is TRIBE-free). For commercial use of TRIBE itself, contact [Meta FAIR](https://github.com/facebookresearch/tribev2).

---

## What's Tested

- ✅ **Full pipeline end-to-end** — text → ElevenLabs → Modal TRIBE → Translator (Layers 2/3/4/5), verified 2026-04-17
- ✅ **~111 test functions** across 18 files — covering change detection, Layer 3 mapping, confidence scoring, vocabulary, Mode 1 pipeline, tier1 mechanism detectors (abs + σ mode parity + faithful-port), LLM detector v2, clip trimmer, brain renderer, analysis bundle, platform export, Gemini retry, and arc miner integration
- ✅ Layer 4 synthesis with Claude Sonnet 4.6
- ✅ Arc miner end-to-end: YouTube / audio / text modes all trimodally mined on Modal, both threshold modes stored
- ✅ Canonical brain renderer + platform export smoke-tested against bundled arc fixtures

```bash
pytest tests/unit                   # 89 unit tests, no external calls
pytest tests/integration            # 15 integration tests
pytest tests/e2e --run-e2e          # 7 E2E tests (some incur API cost)
```

See `examples/analyze_speech.py` for the canonical usage and `tests/README.md` for test conventions.

---

## Project Status

Built by [Landon Fears](https://feelingengineer.com) as the research tool behind [Human Source Code](https://youtube.com/@HumanSourceCode), a channel investigating how content produces feeling.

Follow: [@humancodebase](https://x.com/humancodebase) · [@feelingengineer](https://instagram.com/feelingengineer)

---

## Credits

- **TRIBE v2** — [Meta FAIR](https://github.com/facebookresearch/tribev2) · brain prediction model, CC BY-NC 4.0
- **HCP-MMP** — Glasser et al. multi-modal cortical parcellation, used for the canonical brain renderer
- **Emotional vocabulary** — grounded in Russell's circumplex model + Barrett's theory of constructed emotion
- **Gemini** — Google's multimodal model used by Clip Trimmer and LLM Detector v2
- **Fire corpus (archived sample)** — 8 public-domain / widely-cited excerpts from v0.1; present for backwards-compat

---

## Development

Running the test suite locally:

```bash
pip install -e ".[dev]"
python -m pytest tests/unit              # 22 unit tests, ~0.5s, no API calls
python -m pytest tests/e2e --run-e2e     # 3 live-API tests, ~$0.05 per run
```

**Isolated reproducibility check** — verifies the repo installs and passes tests from a clean environment, not your local one:

```bash
docker build -t feeling-engine-test .
docker run --rm feeling-engine-test python -m pytest tests/unit
```

The Dockerfile installs only what the README tells users to install, so a successful build is evidence that the install instructions actually work on a machine that isn't yours.

**CI** (GitHub Actions) runs unit tests + Mode 1 smoke test on every push and PR across a matrix of Python 3.10/3.11/3.12 × Ubuntu/macOS, plus a package-build check. See [`.github/workflows/test.yml`](.github/workflows/test.yml).

## Contributing

Open an issue first for anything non-trivial. Style: direct, minimal, brain-grounded. Labels must be justifiable from the brain data, not from surface sentiment.
