# Feeling Engine — Capabilities (v0.3)

**Purpose:** single source of truth for what Feeling Engine does. Each row maps one user-visible capability to its code entry point, documentation, and verifying test. If a row exists but any cell is stale, the repo is in drift. If a capability ships without a row, it isn't shipped.

**Sync ritual:** walked before every Ship publication and at each minor version bump.

**Last synced:** 2026-04-22 (v0.3 release)

---

## Mechanism detection

| Capability | Code | Doc | Test |
|---|---|---|---|
| Mechanism detection from TRIBE profiles | `feeling_engine.mechanisms.api.detect_mechanisms(tribe_categories, transcript, axis_stats=...)` | README §Mechanism Detection · examples/walkthrough.md | `tests/unit/test_tier1_detectors.py`, `tests/integration/test_mechanism_detection.py` |
| Mechanism detection without TRIBE (multimodal LLM) | `feeling_engine.mechanisms.llm_detector.detect_from_video(video_path, content_type, chunk_frames)` | README §LLM Detector v2 · examples/walkthrough.md | `tests/unit/test_llm_detector.py`, `tests/e2e/test_gemini_video_smoke.py` |
| Named sequence detection | `feeling_engine.mechanisms.api.detect_sequences(arc)` | README §Mechanism Detection · docs/mechanism_labels.md | `tests/unit/test_sequences.py` |
| σ-mode threshold normalization (per-video z-normalized) | `feeling_engine.mechanisms.tier1_detectors.compute_axis_stats(profiles)` + pass `axis_stats=` to `detect_mechanisms` | README §Two threshold modes · docs/detector_validation.md | `tests/unit/test_tier1_detectors.py` (σ-mode parity + faithful-port assertions) |
| 28-label mechanism vocabulary + 10 named sequences | `feeling_engine.mechanisms.vocabulary.MECHANISMS_BY_NAME`, `SEQUENCES_BY_NAME` | docs/mechanism_labels.md | `tests/unit/test_vocabulary.py` |

## Translator (emotion-label synthesis)

| Capability | Code | Doc | Test |
|---|---|---|---|
| Pipeline entry — arc from profiles | `feeling_engine.pipeline.FeelingPipeline.analyze_profiles(path)` | README §Mode 1 · §Use as a library | `tests/unit/test_pipeline.py` |
| Layer 2 — change detection | `feeling_engine.translator.change_detector` | README §Architecture · docs/methodology.md | `tests/unit/test_change_detector.py` |
| Layer 3 — dimensional mapping → 60-term vocabulary | `feeling_engine.translator.brain_to_emotion` + `vocabulary.yaml` | docs/methodology.md | `tests/unit/test_brain_to_emotion.py` |
| Layer 4 — LLM synthesis (content + viewer context) | `feeling_engine.translator.llm_synthesizer` | README §Add context for Layer 4 | `tests/integration/test_layer4_synthesis.py` |
| Layer 5 — confidence scoring | `feeling_engine.translator.confidence` | README §Architecture | `tests/unit/test_confidence.py` |

## Preprocess

| Capability | Code | Doc | Test |
|---|---|---|---|
| Dense-frame clip trimming (scene/ad/intro removal) | `feeling_engine.preprocess.clip_trimmer.trim_by_dense_classification(video_path, output_path, content_type)` | README §Preprocess | `tests/unit/test_multimodal_preprocess.py` |
| Multimodal frame/audio/transcript helpers | `feeling_engine.preprocess.multimodal` | — (internal shared helper) | `tests/unit/test_multimodal_preprocess.py` |
| Gemini API retry wrapper | `feeling_engine.preprocess._gemini_retry` (actually `feeling_engine._gemini_retry`) | — (internal) | `tests/unit/test_gemini_retry.py` |

## Mining

| Capability | Code | Doc | Test |
|---|---|---|---|
| Mine single YouTube URL → arc library | `python -m feeling_engine.mining.arc_miner mine <URL> --db <path> --niche <tag>` | README §Arc Mining · examples/walkthrough.md | `tests/integration/test_arc_miner.py` |
| Mine batch from urls file | `python -m feeling_engine.mining.arc_miner mine-batch <file> --db <path>` | README §Arc Mining | `tests/integration/test_arc_miner.py` |
| List library contents | `python -m feeling_engine.mining.arc_miner list --db <path>` | README §Arc Mining | — |
| Auto-computed channel median (live yt-dlp lookup) | Implicit in `mine`; warns on `outlier_ratio < 1.5` and catalogs < 10 | README §Arc Mining (narrative only) | `tests/integration/test_arc_miner.py` |
| Trimodal routing (video → full TRIBE, audio → audio path, text → TTS+TRIBE) | Implicit in `mine`; routes by file extension | — (planned doc) | `tests/integration/test_arc_miner.py` |

## Rendering

| Capability | Code | Doc | Test |
|---|---|---|---|
| Canonical brain renderer (mechanism label → hemisphere PNG, no TRIBE runtime) | `feeling_engine.rendering.brain_renderer.render_mechanism_brain(label, output_path, view, intensity)` | README §Rendering · examples/walkthrough.md | `tests/unit/test_brain_renderer.py`, `tests/e2e/test_brain_render_smoke.py` |
| 28 per-mechanism activation signatures | `feeling_engine.rendering.signatures` | — (data, referenced by renderer) | `tests/unit/test_brain_renderer.py` |
| Analysis bundle (key moments + clips + stills + brains) | `feeling_engine.rendering.analysis_assets.render_analysis_bundle(video_path, arc, sequences, out_dir)` | README §Rendering | `tests/unit/test_analysis_assets.py` |
| Key-moment selection (intensity × confidence + temporal spread + sequence bonus) | `feeling_engine.rendering.analysis_assets.extract_key_moments(arc, sequences, duration_sec, n)` | — (internal scoring) | `tests/unit/test_analysis_assets.py` |
| Substack export | `feeling_engine.rendering.platform_export.export_substack(bundle, out_path)` | README §Rendering | `tests/unit/test_platform_export.py` |
| X thread export | `feeling_engine.rendering.platform_export.export_x_thread(bundle, out_path)` | README §Rendering | `tests/unit/test_platform_export.py` |
| YouTube script export | `feeling_engine.rendering.platform_export.export_youtube_script(bundle, out_path)` | README §Rendering | `tests/unit/test_platform_export.py` |
| Multi-platform bundle | `feeling_engine.rendering.platform_export.export_all(bundle, platforms=[...])` | README §Rendering | `tests/unit/test_platform_export.py` |

## Inspector + bundled examples

| Capability | Code | Doc | Test |
|---|---|---|---|
| Static inspector UI (brain timeline + mechanism firings + scrubber + σ-toggle) | `examples/inspector/index.html`, `app.js`, `style.css` | examples/inspector/README.md · README §Want to see it first | — (manual UI test; no automated coverage yet) |
| Four bundled arcs | `examples/arcs/*.json` | examples/arcs/README.md · examples/walkthrough.md | — (fixtures used by other tests) |
| Pre-rendered Jobs hemisphere PNGs (60 timesteps × 2 hemispheres) | `examples/arcs/steve_jobs_death_pivot/brain/*.png` | examples/walkthrough.md · README §Arc Mining | — (static assets) |
| Colab demo notebook | `examples/colab/feeling_engine_demo.ipynb` | README header badge · §Smoke test | — (manual; notebook) |
| Narrated walkthrough | `examples/walkthrough.md` | Linked from README header | — (prose doc) |

## Adapters

| Capability | Code | Doc | Test |
|---|---|---|---|
| TRIBE v2 brain adapter (Modal A100-80GB) | `feeling_engine.adapters.brain_model.tribev2.TRIBEv2Adapter` | README §Deploy TRIBE to Modal · deploy/README.md | `tests/integration/test_tribe_adapter.py` |
| ElevenLabs TTS adapter | `feeling_engine.adapters.tts.elevenlabs.ElevenLabsAdapter` | README §Mode 3 | `tests/unit/test_elevenlabs_adapter.py` |
| Modal compute adapter | `deploy/tribe_modal.py` | deploy/README.md | `tests/e2e/test_modal_smoke.py` (opt-in, requires Modal auth) |

## Archived / legacy

| Capability | Code | Status |
|---|---|---|
| Fire matcher (cross-domain precedent lookup against sample corpus) | `feeling_engine.fire.matcher.FireMatcher` | Present for backwards-compat; new work should use mechanism/sequence queries on the arc library. Sample corpus in `corpus/` is a v0.1 artifact. |

---

## What's NOT yet in feeling-engine (flagged cross-repo dependencies)

The following capabilities exist in HSC's private tooling but are **not yet ported** to feeling-engine. External users cannot reproduce these:

- **Library ingestion CLI** (`library_ingest.py` in HSC) — multi-faceted tags + authored content alongside mined arcs. Planned as feeling-engine **v0.4**.
- **Library query CLI** (`library_query.py` in HSC) — facet/mechanism/sequence/sibling queries with deep-link generation. Planned as feeling-engine **v0.4**.
- **Schema v1** — `tags_json`, `source_type`, authored fields. v0.3 arcs use the v0 schema (single `niche` column). Migration path will ship with v0.4.
- **Fire corpus migration script** — HSC-specific (Fire corpus archived in HSC); will not port.
- **Multi-arc inspector** (cross-arc navigation via shared mechanisms) — spec exists (HSC DEFINE #3, deferred); requires library query to be in feeling-engine first.
- **7-pass arc composition methodology** — HSC creator workflow, not library infrastructure; out of scope for feeling-engine.

---

## Sync protocol

**When a capability changes in code:**
1. Update the CAPABILITIES.md row (code entry, doc ref, test ref)
2. Update the referenced doc section
3. Update / add the referenced test
4. Verify all three cells agree before landing the change

**When a capability is added:**
1. Add a row first (code entry = planned path)
2. Write the test against the planned entry
3. Implement the capability
4. Update doc references

**When a capability is removed or deprecated:**
1. Move the row to "Archived / legacy" with a status note
2. Preserve the code for backwards-compat OR schedule removal with a major version bump
3. Doc sections referencing the capability must be updated to reflect new status

**Cross-repo sync trigger:**

Any change that affects the external contract (arc_library schema, CLI surface, Python API signatures, output JSON shape, bundled fixture format) requires a parallel update in HSC's consumption layer, OR an explicit "unsupported in v0.x, port in v0.x+1" note here.
