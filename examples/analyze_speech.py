#!/usr/bin/env python3
"""Example: Analyze a speech using Feeling Engine.

This demonstrates the full pipeline:
    TRIBE brain data → Translator → Fire cross-domain matching

Three input modes:
    1. From pre-computed TRIBE profiles (JSON) — no GPU needed
    2. From audio file via Modal GPU — requires Modal deployment
    3. From text via ElevenLabs TTS → Modal GPU — requires both services

Usage:
    # Mode 1: Pre-computed profiles (free, instant)
    python examples/analyze_speech.py --profiles path/to/profiles.json

    # Mode 2: Audio file → TRIBE → Translator → Fire
    python examples/analyze_speech.py --audio speech.mp3

    # Mode 3: Text → ElevenLabs → TRIBE → Translator → Fire
    #   (first run --list-voices to pick a voice that fits your content)
    python examples/analyze_speech.py --list-voices
    python examples/analyze_speech.py --text speech.txt --voice-id <id>

    # Add content + viewer context for Layer 4 refinement
    python examples/analyze_speech.py --profiles profiles.json \\
        --content "No one wants to die..." \\
        --context "Viewer knows Jobs died in 2011"

    # Write the synthesized MP3 somewhere specific (Mode 3)
    python examples/analyze_speech.py --text speech.txt \\
        --voice-id <id> --tts-output out/speech.mp3

    # Output as JSON
    python examples/analyze_speech.py --profiles profiles.json --json

    # Skip Fire matching or Layer 4 to reduce cost/time
    python examples/analyze_speech.py --profiles profiles.json --no-fire --no-layer4
"""
import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from feeling_engine.adapters.brain_model.tribev2 import TRIBEv2Adapter
from feeling_engine.pipeline import FeelingPipeline
from feeling_engine.fire.matcher import FireMatcher


def main():
    parser = argparse.ArgumentParser(
        description="Analyze content using Feeling Engine",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--profiles", "-p",
        help="Path to pre-computed TRIBE profiles JSON",
    )
    input_group.add_argument(
        "--audio", "-a",
        help="Path to audio file (requires Modal deployment)",
    )
    input_group.add_argument(
        "--text", "-t",
        help="Path to text file (requires ElevenLabs + Modal)",
    )
    input_group.add_argument(
        "--list-voices", action="store_true",
        help="List ElevenLabs voices in your account and exit (helps pick --voice-id)",
    )

    parser.add_argument("--voice-id",
                        help="ElevenLabs voice ID to use for --text mode. "
                             "Run --list-voices to see options. "
                             "Gender, age, and register should match your content's speaker — "
                             "mismatched voices produce worse TRIBE predictions.")
    parser.add_argument("--tts-output",
                        default=None,
                        help="Where to save the synthesized MP3 (--text mode). "
                             "Defaults to alongside the input text file.")
    parser.add_argument("--content", "-c", help="Content text at the key moment (for Layer 4)")
    parser.add_argument("--context", help="Viewer context description (for Layer 4)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of Fire matches")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--change-points-only", action="store_true",
                        help="Only analyze change-point timesteps (faster)")
    parser.add_argument("--no-fire", action="store_true", help="Skip Fire matching")
    parser.add_argument("--no-layer4", action="store_true", help="Skip LLM refinement")

    args = parser.parse_args()

    # ── Voice listing mode ──
    if args.list_voices:
        _print_voice_list()
        return

    # ── Pipeline setup ──
    pipeline = FeelingPipeline(brain_adapter=TRIBEv2Adapter())
    matcher = FireMatcher() if not args.no_fire else None

    # ── Get emotional arc ──
    if args.profiles:
        print(f"Loading TRIBE profiles from {args.profiles}...")
        arc = pipeline.analyze_profiles(
            args.profiles,
            change_points_only=args.change_points_only,
        )
    elif args.audio:
        print(f"Running TRIBE on {args.audio} via Modal...")
        from feeling_engine.adapters.compute.modal_tribe import ModalTRIBEAdapter
        compute = ModalTRIBEAdapter()
        audio_bytes = Path(args.audio).read_bytes()
        prediction = compute.predict(audio_bytes, Path(args.audio).name)
        arc = pipeline.analyze_predictions(
            prediction.predictions,
            change_points_only=args.change_points_only,
        )
    elif args.text:
        print(f"Converting {args.text} to speech via ElevenLabs...")
        from feeling_engine.adapters.tts.elevenlabs import ElevenLabsAdapter
        from feeling_engine.adapters.compute.modal_tribe import ModalTRIBEAdapter

        tts = ElevenLabsAdapter(voice_id=args.voice_id)
        if args.voice_id:
            print(f"  Using voice_id: {args.voice_id}")
        else:
            print("  No --voice-id provided; using default narrator (Rachel). "
                  "Run --list-voices to pick a voice that matches your content.")
        text = Path(args.text).read_text()
        if args.tts_output:
            tts_out_path = Path(args.tts_output)
        else:
            # Default: write next to the input text file
            tts_out_path = Path(args.text).with_suffix(".mp3")
        tts_out_path.parent.mkdir(parents=True, exist_ok=True)
        tts_result = tts.synthesize(text, tts_out_path)
        print(f"  TTS saved: {tts_out_path} ({tts_result.duration_seconds:.1f}s)")

        print(f"Running TRIBE on synthesized audio ({tts_result.duration_seconds:.1f}s)...")
        compute = ModalTRIBEAdapter()
        audio_bytes = tts_result.audio_path.read_bytes()
        prediction = compute.predict(audio_bytes, "tts_output.mp3")
        arc = pipeline.analyze_predictions(
            prediction.predictions,
            change_points_only=args.change_points_only,
        )

    # ── Layer 4: LLM refinement (optional) ──
    if not args.no_layer4 and args.content:
        from feeling_engine.translator.llm_synthesizer import (
            LLMSynthesizer, ContextProfile,
        )
        from feeling_engine.translator.brain_to_emotion import build_arc_summary
        synth = LLMSynthesizer()

        context = None
        if args.context:
            context = ContextProfile(
                name="user_provided",
                description=args.context,
            )

        print("Refining with Layer 4 (LLM synthesis)...")
        for te in arc.timesteps:
            if te.is_change_point or not args.change_points_only:
                result = synth.refine_timestep(te, args.content, context)
                # Replace Layer 3 labels with Layer 4 refined labels
                te.primary = [
                    type(te.primary[0])(
                        term=sl.term,
                        score=1.0,
                        brain_grounding={},
                        transition_match=True,
                        dimensional_distance=0.0,
                        reasoning=f"[{sl.confidence}] {sl.reasoning}",
                    )
                    for sl in result.refined_labels[:5]
                ] if te.primary else []

        # Recompute arc summary so it reflects Layer 4 refined labels
        arc.arc_summary = build_arc_summary(arc.timesteps)

    # ── Output ──
    if args.json:
        output = pipeline.arc_to_dict(arc)
        if matcher and not args.no_fire:
            matches = matcher.match_arc(arc, top_k=args.top_k)
            output["fire_matches"] = [
                {
                    "title": m.entry.title,
                    "domain": m.entry.domain,
                    "score": round(m.score, 4),
                    "shared_feelings": m.shared_feelings,
                    "excerpt": m.entry.excerpt,
                }
                for m in matches
            ]
        print(json.dumps(output, indent=2))
    else:
        print()
        print(pipeline.format_arc_text(arc))
        if matcher and not args.no_fire:
            print()
            matches = matcher.match_arc(arc, top_k=args.top_k)
            print(matcher.format_matches(matches))


def _print_voice_list():
    """Print the user's ElevenLabs voice library for --voice-id selection."""
    from feeling_engine.adapters.tts.elevenlabs import ElevenLabsAdapter

    try:
        adapter = ElevenLabsAdapter()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    voices = adapter.list_voices()

    if not voices:
        print("No voices found in your ElevenLabs account.")
        return

    # Group by category for readability
    by_category: dict[str, list] = {}
    for v in voices:
        by_category.setdefault(v.category, []).append(v)

    print(f"\n{len(voices)} voices in your ElevenLabs account")
    print("=" * 72)

    for category in sorted(by_category):
        print(f"\n── {category} ──")
        for v in sorted(by_category[category], key=lambda x: x.name.lower()):
            print(f"  {v.name}")
            print(f"    voice_id: {v.voice_id}")
            if v.description:
                desc = v.description if len(v.description) <= 80 else v.description[:77] + "..."
                print(f"    {desc}")

    print("\n" + "=" * 72)
    print("Use with:  --text speech.txt --voice-id <voice_id>")
    print("Tip: match gender, age, and register to your content's speaker for")
    print("     better TRIBE predictions.")


if __name__ == "__main__":
    try:
        main()
    except (ImportError, RuntimeError, FileNotFoundError) as e:
        # Expected CLI-level failure modes — missing deps, missing env keys,
        # missing files. Print cleanly without the full traceback.
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
