# Walkthrough — What Feeling Engine reads

A narrated reading of one bundled arc: **Steve Jobs, Stanford 2005 commencement — the death pivot (60 seconds, starting at t=714s of the full speech).** The arc is at [`arcs/steve_jobs_death_pivot.json`](arcs/steve_jobs_death_pivot.json); pre-rendered brain hemisphere images for all 60 timesteps are at [`arcs/steve_jobs_death_pivot/brain/`](arcs/steve_jobs_death_pivot/).

Open this while reading: [the static inspector](inspector/) renders the same data with a scrubbable brain timeline.

---

## The content

> "No one wants to die. Even people who want to go to heaven don't want to die to get there. And yet, death is the destination we all share. No one has ever escaped it. And that is, as it should be, because **death is very likely the single best invention of life**. It's life's change agent. It clears out the old to make way for the new. Right now, the new is you. But someday, not too long from now…"

Sixty seconds. One paragraph. Not the cancer-diagnosis passage — the earlier, philosophical framing where Jobs is saying something universal before anything personal lands.

## What the pipeline actually produces

A **mechanism arc** — a list of timestamped label firings — and a list of **named sequences** that fire when a known chain of labels appears in order. This is the primary output; emotion labels (if you enable Layer 4) are a secondary interpretation on top.

For this 60-second clip:

- **75 mechanism firings** (absolute-threshold mode)
- **74 firings** in σ-mode (per-video normalized thresholds; same arc, different intensities)
- **2 named sequences** detected: `awakening-build` and `contemplation-spiral`

Top 6 most-frequent mechanisms across the clip:

| Mechanism | Firings | What it means |
|---|---|---|
| `evocation` | 10× | A scene returns with feeling — memory + core affect rising together |
| `recognition` | 9× | *I know this* — memory activates before full context arrives |
| `word-focus` | 8× | Attention on the words, cognitive rather than felt |
| `intimacy-turn` | 7× | Distance closes — room narrows, social axis opens |
| `body-turn` | 5× | Attention drops into the body — interoception rising, language falling |
| `contemplation` | 5× | Meaning-sift — sustained thinking-through, integrating |

Where those land matters as much as how many fire.

## The two sequences that give the clip its shape

**`awakening-build`** — truth-naming → body engages → time shortens → turn inward (t=2–59s)

```
t= 2s  universal-recognition   "No one wants to die..."  — universal claim, body confirms
t=22s  body-turn               "...death is the destination we all share"
t=25s  stakes-compression      "...life's change agent"
t=58s  inward-pivot            "...the new is you"
```

Four beats across the entire clip. This is why the passage works as *openers* for the speech — it walks the listener through a full awakening arc before any personal content arrives.

**`contemplation-spiral`** — parse → think → realize → feel (t=11–48s)

```
t=11s  word-focus              listener tracking the words
t=14s  contemplation           meaning-sift starts
t=24s  recognition             pattern locks in
t=47s  body-turn               the body catches up
```

These two sequences fire concurrently — which is the texture of the piece. Jobs is not building a crescendo; he's giving the listener space to think *and* feel at the same time, then closing the loop.

## Why this matters across domains

The 28-label vocabulary is **content-agnostic by construction**. Open the inspector dropdown and switch between the four bundled arcs:

| Arc | Genre | Sequences that fire |
|---|---|---|
| **Steve Jobs — Death Pivot** | speech | `awakening-build`, `contemplation-spiral` |
| **PastMorph — Evolution of Homes** | visual/music documentary, no dialogue | — (mechanisms fire; sequences require σ-mode) |
| **Iceberger — Your Life as Every Bloods Rank** | POV narrative | `joke-structure` (setup / twist / release) |
| **Corporate Playbook — 90 Day Trap** | adversarial consumer explainer | `intimacy-deepening` |

Four maximally-different genres. Same vocabulary. The mining of these three (plus PastMorph) surfaced **four mechanisms that fired in ALL of them**: `body-turn`, `drift`, `recognition`, `vulnerability-transfer`. That's the empirical hypothesis underlying the engine — there's a small set of universal mechanisms that fire regardless of topic, and a larger set of signatures that distinguish genres.

The Jobs clip cross-references each of these. You can see the match structurally without running TRIBE a second time.

## Two paths into producing an arc like this

**Path A — Full TRIBE** (highest fidelity; requires Modal + HuggingFace access):

```bash
python -m feeling_engine.mining.arc_miner mine https://youtube.com/watch?v=UF8uR6Z6KLc \
  --db arc_library.db --niche "commencement-speech"
```

Runs Modal A100, trimodal TRIBE v2, mechanism detection, sequence detection. ~$0.08 for a 60-second clip. Output: exactly the arc shown above.

**Path B — LLM Detector v2** (no TRIBE; fastest iteration):

```python
from feeling_engine.mechanisms.llm_detector import detect_from_video

arc = detect_from_video(
    video_path="jobs_60s.mp4",
    content_type="commencement speech",
)
```

Gemini classifies mechanisms directly from frames + audio + Whisper transcript per chunk. No brain model runtime required. Output schema identical to Path A (`tribe_grounded=0` flag distinguishes). For most content types the two agree on structural moves; they diverge on subtle interoceptive nuance that TRIBE's brain model catches.

**Preprocess** — if your source is a raw YouTube download with intros, mid-rolls, or commentary:

```python
from feeling_engine.preprocess.clip_trimmer import trim_by_dense_classification
trim_by_dense_classification(
    video_path="raw_commencement_speech.mp4",
    output_path="jobs_60s.mp4",
    content_type="commencement speech",
)
```

Gemini densely classifies every frame (`scene / branded / title / commentary / ad / transition`) and emits an ffmpeg cut list that strips everything except the content itself.

## What to do from here

- **Open [the inspector](inspector/)** and scrub this arc. Watch the 7 brain-axis bars move; note when language drops and interoception rises.
- **Render any mechanism as brain images** without running TRIBE: `render_mechanism_brain(label="body-turn", output_path=..., view="lateral_both")`. Uses the public HCP-MMP atlas + 28 per-mechanism activation signatures, no runtime TRIBE.
- **Bundle analysis for publication**: `render_analysis_bundle(...)` extracts key moments + clips + stills + brain images, and `export_all(bundle, platforms=["substack", "x", "youtube"])` bundles them into per-platform drafts.
- **Mine your own corpus** (`arc_miner mine <URL>`) and see which mechanisms fire across your content type. Universals and signatures emerge empirically within ~10 arcs.

The engine doesn't tell you what to feel. It measures what a brain would do in response to the content, and names the pattern in a vocabulary that travels across mediums. What the pattern *means* is still yours to argue.
