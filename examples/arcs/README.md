# Example Arcs

Three pre-mined arcs from publicly-available YouTube content. Each file bundles everything needed to demo Feeling Engine's interpretation pipeline without running TRIBE locally:

- **`tribe_profiles`** — per-timestep brain activation across 7 HCP categories (from Modal TRIBE A100-80GB, trimodal on mp4)
- **`arc_absolute`** — mechanism applications under absolute thresholds (Steve-Jobs calibrated)
- **`arc_sigma`** — mechanism applications under per-video σ-normalized thresholds (content-agnostic; thresholds interpreted relative to this video's own signal distribution)
- **`sequences_absolute` / `sequences_sigma`** — named narrative-sequence matches in each mode
- **`axis_stats`** — per-axis mean+std for value and derivative signals
- **`transcript`** — word-level transcript (auto-captions)
- **`counts`** — quick summary of labels/sequences per mode

| File | Channel | Duration | Type | σ-mechanisms |
|---|---|---|---|---|
| `steve_jobs_death_pivot.json` + `steve_jobs_death_pivot/brain/` | Stanford | 60s excerpt | Speech (death-pivot clip), **ships pre-rendered hemispheres** | 74 |
| `pastmorph_evolution_of_homes.json` | @PastMorph | 4m45 | Visual/music, no dialogue | 168 |
| `iceberger_bloods_rank.json` | @Iceberger99 | 7m29 | POV narrative | 886 |
| `corporate_playbook_90_day_trap.json` | @the_corporateplaybook | 9m24 | Adversarial explainer | 1300 |

## Why these four

They span content-type extremes: no-dialogue visual, first-person narrative, speech-heavy explainer, plus the Stanford Death Pivot (the original calibration clip). Together they stress-test the detector across axis-variance profiles — the case that motivates σ-normalization (see `docs/detector_validation.md`). The Jobs bundle also ships with pre-rendered hemisphere PNGs (nilearn `plot_surf` on the raw TRIBE predictions); the other three do not yet — see the inspector README for how it handles that gracefully.

## Usage

```python
import json
bundle = json.load(open("examples/arcs/pastmorph_evolution_of_homes.json"))

# Re-run mechanism detection from the bundled profiles (no TRIBE call needed):
from feeling_engine.mechanisms.api import detect_mechanisms, detect_sequences
from feeling_engine.mechanisms.tier1_detectors import compute_axis_stats

axis_stats = compute_axis_stats(bundle["tribe_profiles"])
arc = detect_mechanisms(
    tribe_categories=bundle["tribe_profiles"],
    transcript=bundle["transcript"],
    axis_stats=axis_stats,  # σ-mode
)
sequences = detect_sequences(arc)
print(f"{len(arc)} mechanism applications, {len(sequences)} sequence matches")
```

Or skip re-running and use the pre-computed `arc_absolute` / `arc_sigma` fields directly.

## Licensing

- **Vocabulary-form arcs** (the label applications, sequence matches, axis stats) — MIT via Feeling Engine.
- **TRIBE profiles** — derivative of CC BY-NC 4.0 (Meta FAIR). Research/non-commercial use only.
- **Source content** — not redistributed. URLs point to the original publicly-available YouTube videos; re-fetch via yt-dlp if you want the raw audio/video.

## Regenerating

See `feeling_engine/mining/arc_miner.py`:

```bash
python -m feeling_engine.mining.arc_miner mine <YouTube URL> \
  --db arc_library.db --niche <niche> --channel-median <views>
```
