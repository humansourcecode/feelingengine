# Feeling Engine Inspector

A zero-dependency static-HTML tool for replaying any mined arc from [`examples/arcs/`](../arcs/). Shows:

- **Brain activity timeline** — 7 HCP categories over the content's duration
- **Mechanism firings** — every detected label, when it fires, at what intensity
- **Per-moment detail** — brain state, active mechanisms, transcript word at the scrubbed timestep
- **σ-mode toggle** — switch between absolute thresholds and per-video z-normalized thresholds

## Run it

Browsers block `fetch()` of local files, so you need any static server:

```bash
cd feeling-engine/examples
python3 -m http.server 8080
```

Then open `http://localhost:8080/inspector/`.

## What's in v1

- Works with the three arc bundles in `examples/arcs/` (dropdown selector)
- SVG timelines, pure vanilla JS, no build step
- Scrubber-driven per-moment detail panel

## What's deliberately not in v1

- Live TRIBE inference (use `examples/analyze_speech.py --audio ...` for that)
- Audio/video playback alongside the timeline — arcs are published vocabulary-only; if you want synchronized media, host the clip yourself and wire it in
- Multiple story-beat interpretations per moment — planned as a v2 upgrade to Layer 4 output
- Sequence-match highlighting — sequences are in the bundle data but not yet rendered

## Adding your own arc

Run `python -m feeling_engine.mining.arc_miner mine <URL>` against any YouTube video, then export via the same script that produced the bundles in `examples/arcs/`. Drop the JSON into `examples/arcs/` and add an `<option>` to `index.html`'s dropdown.
