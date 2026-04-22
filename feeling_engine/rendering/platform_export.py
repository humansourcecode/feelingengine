"""Cross-platform asset export — one analysis bundle → platform-specific renders.

Consumes the AnalysisBundle produced by render_analysis_bundle() and writes:

    export_substack()       → article.md  (durable long-form; primary)
    export_x_thread()       → thread.json (amplification; sequenced tweets)
    export_youtube_script() → video_script.md (OBS walkthrough plan)

Each exporter reads only bundle + on-disk referenced assets. No re-analysis.
Callers can invoke individual exporters or use export_all().

Intentionally NOT built in this pass (deferred, needs video composition):
- TikTok/Reels/Shorts vertical-video cuts with overlay text + brain images
- LinkedIn variant (similar to Substack, slightly different framing)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


# ─── Helpers ────────────────────────────────────────────────────────

def _bundle_dir(bundle) -> Path:
    """Return the directory the bundle was rendered into."""
    return Path(bundle.out_dir if hasattr(bundle, "out_dir") else bundle["out_dir"])


def _as_dict(bundle) -> dict:
    return bundle.to_dict() if hasattr(bundle, "to_dict") else bundle


def _mmss(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


def _moment_summary_line(m: dict) -> str:
    """One-line human summary of a key moment."""
    pieces = [
        f"**{m['label']}** @ {_mmss(m['peak_sec'])}",
        f"intensity {m['intensity']:.2f}",
    ]
    if m.get("in_sequence"):
        pieces.append(f"within sequence *{m['in_sequence']}*")
    if m.get("co_firing"):
        pieces.append(f"co-firing: {', '.join(m['co_firing'][:3])}")
    return " · ".join(pieces)


# ─── Substack article (markdown) ───────────────────────────────────

_SUBSTACK_OPENING_TEMPLATE = """# Mechanism analysis — {video_name}

{opening_hook}

---

"""

_SUBSTACK_CLOSING_TEMPLATE = """

---

## What fired across the whole piece

| Mechanism | Firings |
|---|---:|
{mechanism_rows}

{sequence_block}

*Generated from Feeling Engine analysis of [{video_name}]({video_source}).*
"""


def _build_opening_hook(bundle: dict) -> str:
    n = len(bundle["key_moments"])
    total = bundle["total_labels"]
    dur_m = bundle["duration_sec"] / 60
    if n == 0:
        return (f"A {dur_m:.1f}-minute piece. No salient mechanism firings were "
                f"strong enough to feature.")
    top_labels = list(bundle["mechanism_counts"])[:3]
    return (
        f"A {dur_m:.1f}-minute piece. {total} mechanism firings total across the "
        f"Feeling Engine's 28-label vocabulary, with {n} key moments selected for "
        f"detailed breakdown. Dominant mechanisms: {', '.join(top_labels)}."
    )


def export_substack(bundle, out_path=None) -> Path:
    """Render the bundle as a Substack-ready markdown article.

    Each key moment becomes a section with: heading, embedded clip reference,
    brain image, moment metadata, evidence. Closes with mechanism-count table
    + detected sequences.
    """
    b = _as_dict(bundle)
    bundle_dir = _bundle_dir(bundle)
    if out_path is None:
        out_path = bundle_dir / "article.md"
    out_path = Path(out_path)

    video_source = b.get("video_source") or ""
    video_name = Path(video_source).name if not video_source.startswith("http") \
                 else video_source.rsplit("/", 1)[-1] or video_source

    out = [_SUBSTACK_OPENING_TEMPLATE.format(
        video_name=video_name,
        opening_hook=_build_opening_hook(b),
    )]

    for m in b["key_moments"]:
        out.append(f"## Moment {m['index']} — {m['label']} @ {_mmss(m['peak_sec'])}\n")

        # Clip embed — Substack supports direct video embeds via file reference
        if m.get("clip_path"):
            out.append(f"![clip]({m['clip_path']})\n")
        # Still as fallback
        if m.get("still_path"):
            out.append(f"![still at {m['peak_sec']:.1f}s]({m['still_path']})\n")
        # Brain image inline
        if m.get("brain_path"):
            out.append(f"![brain — {m['label']}]({m['brain_path']})\n")

        # Moment summary
        out.append(_moment_summary_line(m) + "\n")

        # Evidence from the detector
        if m.get("evidence"):
            out.append(f"*{m['evidence']}*\n")

        out.append("")

    # Mechanism counts table
    mechanism_rows = "\n".join(
        f"| {lbl} | {n} |" for lbl, n in b["mechanism_counts"].items()
    )

    # Named sequences, if any
    if b["sequences"]:
        seq_lines = ["## Named sequences detected", ""]
        for s in b["sequences"]:
            seq_lines.append(
                f"- **{s['name']}** ({_mmss(s['start_sec'])}-{_mmss(s['end_sec'])}) "
                f"— {', '.join(s['matched_labels'])}"
            )
        sequence_block = "\n".join(seq_lines)
    else:
        sequence_block = ""

    out.append(_SUBSTACK_CLOSING_TEMPLATE.format(
        mechanism_rows=mechanism_rows or "| (none) | 0 |",
        sequence_block=sequence_block,
        video_name=video_name,
        video_source=video_source,
    ))

    out_path.write_text("\n".join(out))
    return out_path


# ─── X / Threads thread (JSON) ─────────────────────────────────────

def export_x_thread(bundle, out_path=None,
                    max_posts: int = 10) -> Path:
    """Render the bundle as a sequenced JSON thread for X / Threads.

    Structure:
        {
          "platform": "x",
          "posts": [
            {"index": 1, "text": "...", "media": [...]},
            ...
          ]
        }

    Opening post hooks the thread; one post per key moment (with clip or still
    reference); closing post summarizes + CTA.
    """
    b = _as_dict(bundle)
    bundle_dir = _bundle_dir(bundle)
    if out_path is None:
        out_path = bundle_dir / "thread.json"
    out_path = Path(out_path)

    video_source = b.get("video_source") or ""
    video_name = (video_source.rsplit("/", 1)[-1]
                  if video_source.startswith("http")
                  else Path(video_source).name)

    posts = []
    # Opening post
    top_labels = list(b["mechanism_counts"])[:3]
    opening = (
        f"Measured {video_name} with the Feeling Engine. "
        f"{b['total_labels']} mechanism firings across the 28-label vocabulary. "
        f"Dominant: {', '.join(top_labels)}.\n\n"
        f"Key moments below 🧵"
    )
    posts.append({
        "index": 1,
        "text": opening,
        "media": [],
    })

    # Per-moment posts (cap at max_posts - 2 to leave room for open+close)
    for i, m in enumerate(b["key_moments"][:max_posts - 2], start=2):
        parts = [
            f"{i-1}/ {m['label']} @ {_mmss(m['peak_sec'])}",
            f"intensity {m['intensity']:.2f} · confidence {m['confidence']:.2f}",
        ]
        if m.get("in_sequence"):
            parts.append(f"within *{m['in_sequence']}*")
        if m.get("evidence"):
            parts.append(m["evidence"][:180])
        text = "\n\n".join(parts)
        media = []
        if m.get("still_path"):
            media.append({"type": "image", "path": m["still_path"]})
        if m.get("brain_path"):
            media.append({"type": "image", "path": m["brain_path"]})
        if m.get("clip_path"):
            media.append({"type": "video", "path": m["clip_path"]})
        posts.append({
            "index": i,
            "text": text,
            "media": media,
        })

    # Closing post
    closing_parts = [
        f"Full analysis + brain images + clips:\n{video_source}",
    ]
    if b["sequences"]:
        seq_names = ", ".join(s["name"] for s in b["sequences"])
        closing_parts.append(f"Named sequences detected: {seq_names}.")
    closing_parts.append(
        "If you make content and want this lens on your own work — reply."
    )
    posts.append({
        "index": len(posts) + 1,
        "text": "\n\n".join(closing_parts),
        "media": [],
    })

    thread = {"platform": "x", "posts": posts}
    out_path.write_text(json.dumps(thread, indent=2))
    return out_path


# ─── YouTube OBS script (markdown walkthrough) ──────────────────────

def export_youtube_script(bundle, out_path=None) -> Path:
    """Render an OBS-session walkthrough script for YouTube.

    Structured as a scripted walkthrough plan: intro bookend, per-moment
    on-screen actions (what clip to play, what brain to show, what to narrate),
    closing bookend. This is a script plan, not a verbatim transcript — the
    OBS session captures the creator narrating over these cues.
    """
    b = _as_dict(bundle)
    bundle_dir = _bundle_dir(bundle)
    if out_path is None:
        out_path = bundle_dir / "video_script.md"
    out_path = Path(out_path)

    video_source = b.get("video_source") or ""
    video_name = (video_source.rsplit("/", 1)[-1]
                  if video_source.startswith("http")
                  else Path(video_source).name)

    lines = [f"# OBS session plan — {video_name}", ""]
    lines.append("## Intro bookend (scripted, face-cam)")
    lines.append("")
    lines.append(
        f"Introduce the piece and the 28-mechanism vocabulary. Tell viewers "
        f"this is a {b['duration_sec']/60:.1f}-minute measurement with "
        f"{len(b['key_moments'])} key moments. Preview the dominant mechanisms: "
        f"{', '.join(list(b['mechanism_counts'])[:3])}."
    )
    lines.append("")
    lines.append("## Key moments (on-screen analysis, face PIP)")
    lines.append("")
    for m in b["key_moments"]:
        lines.append(f"### Moment {m['index']} — {m['label']} @ {_mmss(m['peak_sec'])}")
        lines.append("")
        lines.append("**On-screen actions:**")
        if m.get("clip_path"):
            lines.append(f"1. Play clip: `{m['clip_path']}`")
        if m.get("brain_path"):
            lines.append(f"2. Show brain image: `{m['brain_path']}` (side panel)")
        if m.get("still_path"):
            lines.append(f"3. Pause on still: `{m['still_path']}` (if needed)")
        lines.append("")
        lines.append("**Narration cues:**")
        lines.append(f"- Name the mechanism: *{m['label']}* (intensity {m['intensity']:.2f}).")
        if m.get("evidence"):
            lines.append(f"- Describe what fires: {m['evidence']}")
        if m.get("in_sequence"):
            lines.append(f"- Place in sequence: *{m['in_sequence']}*")
        if m.get("co_firing"):
            lines.append(f"- Note co-firing mechanisms: "
                         f"{', '.join(m['co_firing'][:3])}")
        lines.append("")
    lines.append("## Outro bookend (scripted, face-cam)")
    lines.append("")
    if b["sequences"]:
        seq_names = ", ".join(s["name"] for s in b["sequences"])
        lines.append(f"Summarize the sequences that assembled: {seq_names}.")
    lines.append("")
    lines.append(
        "Close with the takeaway: what the measurement reveals about this "
        "piece specifically, and the invitation for viewers to submit their "
        "own content for analysis."
    )
    lines.append("")

    out_path.write_text("\n".join(lines))
    return out_path


# ─── Orchestrator ────────────────────────────────────────────────────

_EXPORTERS = {
    "substack": export_substack,
    "x": export_x_thread,
    "youtube": export_youtube_script,
}


def export_all(bundle, platforms: Optional[list] = None) -> dict:
    """Run all (or selected) exporters. Returns dict of platform → output path.

    Default platforms: all three built-in (substack, x, youtube).
    """
    platforms = platforms or list(_EXPORTERS)
    results = {}
    for p in platforms:
        fn = _EXPORTERS.get(p)
        if fn is None:
            continue
        results[p] = fn(bundle)
    return results
